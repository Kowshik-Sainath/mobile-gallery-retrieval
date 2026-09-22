"""
STNet + TASK-former Modules for T+SBIR.

Phase 3 Architectural Upgrade:
  - TASKformerCrossAttention: replaces the old SketchGuidedAttentionPooling.
    The old version used the sketch vector (B,D) as a single-token Q over photo patches,
    which is mathematically equivalent to a weighted sum with a fixed query — it cannot
    capture multi-scale spatial grounding.

    The TASK-former formulation (from janesjanes/tsbir) inverts this:
      Q  = photo patch tokens  (B, N, D)   — spatial positions that LOOK for guidance
      K,V = [sketch ‖ text]   (B, 2, D)   — the semantic anchor that provides guidance
    This lets every spatial patch position independently attend to the sketch and text
    signals, producing a query-conditioned photo embedding that retains spatial structure.

  - SketchObjectDetectionHead: auxiliary L_OD loss head.
    For each of the N photo patch tokens, predicts a binary sketch-object presence score.
    Trained with binary focal loss against a soft target derived from cosine similarity
    between each patch and the sketch embedding. Forces spatial grounding — the model
    must learn WHICH photo regions correspond to the sketch strokes.

  - SketchReconstructionDecoder: unchanged — auxiliary L_SR regularizer.

Reference:
  TASKformer: janesjanes/tsbir (CVPR 2023)
  STNet multi-loss: vl2g/CSTBIR (AAAI 2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TASKformerCrossAttention(nn.Module):
    """
    TASK-former Cross-Attention Bottleneck.

    Architecture:
        Q  = photo patch tokens projected  (B, N, D)
        K,V = fused [sketch_embed ‖ text_embed] projected  (B, 2, D)

        Multi-head cross-attention (n_heads=8, head_dim=D/8).
        Residual connection + LayerNorm on patch tokens.
        Global average pool over N patch positions → (B, D).
        Final linear projection → (B, D).

    Why this works for scene-level FS-COCO:
        GAP-then-fuse (late fusion) discards the 7×7 spatial map before the
        sketch/text signal can influence it. By running cross-attention BEFORE
        pooling, every spatial patch can independently decide how much it should
        contribute based on relevance to the sketch/text query.
    """

    def __init__(self, embed_dim: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, f"embed_dim {embed_dim} must be divisible by n_heads {n_heads}"
        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        self.scale     = self.head_dim ** -0.5

        # Photo patches → Q
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # Fused [sketch ‖ text] → K, V
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Output
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm_patches = nn.LayerNorm(embed_dim)   # pre-norm on patch Q
        self.norm_ctx     = nn.LayerNorm(embed_dim)   # pre-norm on context K/V
        self.norm_out     = nn.LayerNorm(embed_dim)   # post-norm after residual
        self.dropout      = nn.Dropout(dropout)
        self.final_proj   = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,     # (B, N, D) — photo patch spatial tokens
        sketch_embed: torch.Tensor,     # (B, D)
        text_embed: torch.Tensor,       # (B, D)
    ):
        """
        Returns:
            attended_photo : (B, D)         — spatially grounded photo embedding
            attn_probs     : (B, N_heads, N, 2) — attention weights for analysis
        """
        B, N, D = patch_tokens.shape

        # --- Build context: (B, 2, D) = [sketch, text] ---
        # sketch_embed and text_embed are both (B, D)
        ctx = torch.stack([sketch_embed, text_embed], dim=1)   # (B, 2, D)

        # --- Pre-norm ---
        q_in  = self.norm_patches(patch_tokens)   # (B, N, D)
        kv_in = self.norm_ctx(ctx)                # (B, 2, D)

        # --- Project Q, K, V ---
        Q = self.q_proj(q_in)     # (B, N, D)
        K = self.k_proj(kv_in)    # (B, 2, D)
        V = self.v_proj(kv_in)    # (B, 2, D)

        # --- Reshape to multi-head: (B, H, seq, head_dim) ---
        def split_heads(x, seq_len):
            return x.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            # → (B, H, seq, head_dim)

        Q = split_heads(Q, N)     # (B, H, N, head_dim)
        K = split_heads(K, 2)     # (B, H, 2, head_dim)
        V = split_heads(V, 2)     # (B, H, 2, head_dim)

        # --- Scaled dot-product attention ---
        # attn_weights: (B, H, N, 2)  — each patch attends to [sketch, text]
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, 2)
        attn_probs   = F.softmax(attn_weights, dim=-1)
        attn_probs   = self.dropout(attn_probs)

        # Attended output: (B, H, N, head_dim)
        attended = torch.matmul(attn_probs, V)

        # --- Merge heads: (B, N, D) ---
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        attended = self.out_proj(attended)       # (B, N, D)

        # --- Residual + LayerNorm ---
        attended = self.norm_out(patch_tokens + self.dropout(attended))  # (B, N, D)

        # --- Pool over spatial patches → (B, D) ---
        pooled = attended.mean(dim=1)           # (B, D)
        output = self.final_proj(pooled)        # (B, D)

        return output, attn_probs


class SketchObjectDetectionHead(nn.Module):
    """
    Auxiliary Sketch-Guided Object Detection Head (L_OD).

    Purpose:
        Forces spatial grounding. For each of the N photo patch tokens, predicts
        whether that patch region is relevant to the sketch (binary heatmap).

        The target heatmap is derived automatically from the cosine similarity between
        each photo patch and the sketch embedding — no external bounding-box labels needed.
        Patches with cos_sim > threshold are treated as 'sketch-relevant' (target=1).

    Loss:
        Binary focal loss (alpha=0.25, gamma=2.0). Focal loss is critical because
        most patches (background) are not sketch-relevant → severe class imbalance
        that vanilla BCE cannot handle.

    Reference: vl2g/CSTBIR L_OD formulation.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        relevance_threshold: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.relevance_threshold = relevance_threshold
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Per-patch binary classifier
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,   # (B, N, D) — photo patch tokens (normalised)
        sketch_embed: torch.Tensor,   # (B, D)    — sketch embedding (normalised)
    ):
        """
        Returns:
            loss_od: scalar — binary focal loss for sketch-patch relevance
            heatmap: (B, N)  — predicted relevance logits (for visualisation)
        """
        B, N, D = patch_tokens.shape

        # --- Compute soft targets from cosine sim (no grad) ---
        with torch.no_grad():
            # patch_tokens: (B, N, D),  sketch_embed: (B, D) → (B, 1, D)
            sketch_q = F.normalize(sketch_embed, dim=-1).unsqueeze(1)   # (B, 1, D)
            patches_n = F.normalize(patch_tokens, dim=-1)               # (B, N, D)
            sim = (patches_n * sketch_q).sum(dim=-1)                    # (B, N)
            targets = (sim > self.relevance_threshold).float()          # (B, N) binary

        # --- Per-patch logits ---
        logits = self.head(patch_tokens).squeeze(-1)   # (B, N)

        # --- Binary focal loss ---
        loss_od = self._focal_loss(logits, targets)
        return loss_od, logits

    def _focal_loss(
        self,
        logits: torch.Tensor,   # (B, N)
        targets: torch.Tensor,  # (B, N) binary float
    ) -> torch.Tensor:
        """
        Binary focal loss:
            FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        p   = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (B, N)

        p_t     = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.focal_alpha + (1 - targets) * (1 - self.focal_alpha)
        weight  = alpha_t * (1 - p_t) ** self.focal_gamma

        focal = (weight * bce).mean()
        return focal


class SketchReconstructionDecoder(nn.Module):
    """
    Auxiliary Sketch Reconstruction Decoder (L_SR).
    Reconstructs 224×224 raster sketch from the sketch embedding to enforce
    sketch-specific structural representations. Training-only; discarded on export.

    Reference: vl2g/CSTBIR L_SR regularizer.
    """
    def __init__(self, embed_dim: int = 512, out_channels: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(embed_dim, 256 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14×14
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),  # 28×28
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1),  # 56×56
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  16,  kernel_size=4, stride=2, padding=1),  # 112×112
            nn.BatchNorm2d(16),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1),  # 224×224
            nn.Sigmoid()
        )

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        B = feature_vector.shape[0]
        x = self.fc(feature_vector).view(B, 256, 7, 7)
        return self.decoder(x)
