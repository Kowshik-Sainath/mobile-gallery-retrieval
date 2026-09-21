"""
Composite Text + Sketch Based Image Retrieval Model.
Integrates MobileCLIP backbone, PEFT LoRA adapter, STNet Attention Pooling,
text adapter, and Auxiliary Sketch Reconstruction.

Fixes applied:
  BUG 2 — encode_photo uses context-manager-based adapter disable (guaranteed cleanup).
  BUG 3 — encode_photo_patches uses PatchTokenExtractor (real FastViT conv_exp hook).
  FIX D  — Text adapter (2-layer bottleneck) added for FS-COCO long caption adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import (
    load_mobileclip_backbone,
    apply_sketch_lora,
    set_active_sketch_adapter,
    disable_vision_adapters,
)
from .stnet_modules import SketchGuidedAttentionPooling, SketchReconstructionDecoder
from .patch_hook import PatchTokenExtractor


class TSBIRCompositeModel(nn.Module):
    """
    Composite Model for Text + Sketch Based Image Retrieval (T+SBIR).

    Architecture:
      - MobileCLIP-S1 (FastViT, <50M params) backbone — frozen.
      - Named LoRA adapter 'sketch' on vision encoder for sketch-domain adaptation.
      - Photo encoding runs with adapters DISABLED via PEFT context manager.
      - PatchTokenExtractor hooks conv_exp output for real (B, 49, D) patch tokens.
      - SketchGuidedAttentionPooling: sketch embedding attends over 49 patch tokens.
      - Text adapter (2-layer bottleneck 256d) for FS-COCO long captions (FIX D).
      - Auxiliary SketchReconstructionDecoder for structural regularization (train-only).
    """

    def __init__(
        self,
        backbone_name: str = "mobileclip_s1",
        checkpoint_path: str = "checkpoints/mobileclip_s1.pt",
        device: str = "cuda",
        temperature: float = 0.07,
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature

        # --- Load backbone (BUG 1: param-count checked inside, no silent fallback) ---
        self.backbone, self.tokenizer, self.preprocess = load_mobileclip_backbone(
            model_name=backbone_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        # --- Attach named sketch LoRA adapter (BUG 2 / FIX A: verified attachment) ---
        self.backbone = apply_sketch_lora(self.backbone, adapter_name="sketch")

        # --- PatchTokenExtractor: hooks conv_exp for (B, N, D) spatial tokens ---
        # BUG 3: replaces the degenerate .repeat(1, 49, 1) fallback entirely.
        self.patch_extractor = PatchTokenExtractor(
            self.backbone.image_encoder, embed_dim=512
        )

        embed_dim = 512
        self.embed_dim = embed_dim

        # --- Sketch + Text fusion MLP ---
        self.composite_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- FIX D: Text adapter for FS-COCO long captions ---
        # Frozen text tower + small trained bottleneck adapter (256d).
        self.text_adapter = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

        # --- STNet Attention Pooling + Reconstruction Decoder ---
        self.attention_pooling = SketchGuidedAttentionPooling(embed_dim=embed_dim)
        self.sketch_decoder = SketchReconstructionDecoder(embed_dim=embed_dim)

    # -----------------------------------------------------------------------
    # Internal helper — safe adapter disable
    # -----------------------------------------------------------------------

    def _disable_adapter_ctx(self):
        """
        Returns a PEFT disable_adapter() context manager if the image encoder
        is a PEFT model, otherwise returns a no-op contextmanager.
        This makes encode_photo / encode_photo_patches safe even if PEFT
        wrapping was skipped (e.g. during first-time init before PEFT attaches).
        """
        img_enc = self.backbone.image_encoder
        if hasattr(img_enc, 'disable_adapter'):
            return img_enc.disable_adapter()
        # Fallback: no-op context manager
        from contextlib import nullcontext
        return nullcontext()

    # -----------------------------------------------------------------------
    # Encoding methods
    # -----------------------------------------------------------------------

    def encode_sketch(self, sketch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes raster sketch using LoRA-adapted vision tower ('sketch' adapter).
        BUG 2: set_active_sketch_adapter now raises on failure — no silent no-op.
        """
        set_active_sketch_adapter(self.backbone, "sketch")
        sketch_features = self.backbone.encode_image(sketch_tensor)
        return F.normalize(sketch_features, dim=-1)

    def encode_text(self, text_list) -> torch.Tensor:
        """
        Encodes text query through frozen text tower + trainable text adapter (FIX D).
        Note: text tower has no LoRA — no adapter switching needed here.
        """
        tokens = self.tokenizer(text_list).to(self.device)
        text_features = self.backbone.encode_text(tokens)
        # FIX D: apply small trained text adapter (residual)
        adapted = text_features + self.text_adapter(text_features)
        return F.normalize(adapted, dim=-1)

    def encode_photo(self, photo_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes photo using frozen vision tower with LoRA adapters DISABLED.
        BUG 2: uses safe _disable_adapter_ctx() — works with or without PEFT wrapping.
        """
        with self._disable_adapter_ctx():
            photo_features = self.backbone.encode_image(photo_tensor)
        return F.normalize(photo_features, dim=-1)

    def encode_photo_patches(self, photo_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts pre-pooling spatial patch tokens (B, N, D) from FastViT conv_exp
        output via PatchTokenExtractor hook.

        BUG 3 FIX: hooks the real (B, 512, 7, 7) feature map → (B, 49, 512).
        No longer uses the degenerate .repeat(1, 49, 1) fallback.

        Assertions:
          - N > 1  (enforced by PatchTokenExtractor.extract())
          - token std > 1e-5  (enforced by PatchTokenExtractor.extract())
        """
        with self._disable_adapter_ctx():
            # Forward pass fires the conv_exp hook internally
            _ = self.backbone.encode_image(photo_tensor)

        # Retrieve tokens captured by the hook
        patch_tokens = self.patch_extractor.extract()   # (B, N, embed_dim)
        return F.normalize(patch_tokens, dim=-1)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        sketch_tensor: torch.Tensor,
        text_list,
        photo_tensor: torch.Tensor,
        target_sketch_tensor: torch.Tensor | None = None,
    ):
        """
        Full forward pass: sketch + text → composite query vs attended photo.

        Returns dict with:
          loss, loss_infonce, loss_rec,
          composite_query, attended_photo, recon_sketch
        """
        # 1. Compute disentangled embeddings
        e_sketch = self.encode_sketch(sketch_tensor)         # (B, D)
        e_text = self.encode_text(text_list)                 # (B, D)
        photo_patches = self.encode_photo_patches(photo_tensor)  # (B, 49, D) — REAL tokens

        # 2. Fuse sketch + text into composite query
        raw_composite = torch.cat([e_sketch, e_text], dim=-1)
        composite_query = F.normalize(self.composite_fusion(raw_composite), dim=-1)

        # 3. Sketch-guided attention pooling over real patch token sequence
        attended_photo, attn_probs = self.attention_pooling(e_sketch, photo_patches)
        attended_photo = F.normalize(attended_photo, dim=-1)

        # 4. InfoNCE contrastive loss
        sim_matrix = torch.matmul(composite_query, attended_photo.T) / self.temperature
        batch_size = sim_matrix.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        loss_infonce = F.cross_entropy(sim_matrix, labels)

        # 5. Auxiliary sketch reconstruction (training only)
        loss_rec = torch.tensor(0.0, device=self.device)
        recon_sketch = None
        if target_sketch_tensor is not None and self.training:
            recon_sketch = self.sketch_decoder(e_sketch)
            loss_rec = F.mse_loss(recon_sketch, target_sketch_tensor)

        total_loss = loss_infonce + 0.5 * loss_rec

        return {
            'loss': total_loss,
            'loss_infonce': loss_infonce,
            'loss_rec': loss_rec,
            'composite_query': composite_query,
            'attended_photo': attended_photo,
            'recon_sketch': recon_sketch,
        }
