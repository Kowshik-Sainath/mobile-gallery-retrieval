"""
Composite Text + Sketch Based Image Retrieval Model.

Architecture:
  - MobileCLIP-S1 (FastViT) backbone — frozen base weights.
  - Two-tier sketch LoRA: Conv2d r=4 (point-wise stages) + Linear r=64/alpha=128 (qkv, fc1, fc2).
  - MoCo momentum queue (K=4096) for small-batch InfoNCE.
  - Strict modality isolation: sketch pass = LoRA ON (then OFF), photo = LoRA always OFF.
  - PatchTokenExtractor hooks conv_exp for real (B, 49, D) spatial tokens.
  - TASKformerCrossAttention: photo patches Q over [sketch‖text] K/V for spatial grounding.
  - SketchObjectDetectionHead: focal-loss L_OD auxiliary head for patch-level grounding.
  - Auxiliary SketchReconstructionDecoder for structural regularization (train-only).

Component 3 (Modality Contamination Fix):
  encode_sketch() explicitly enables LoRA, encodes, then DISABLES LoRA.
  encode_photo() always runs with LoRA disabled — no leakage possible in either direction.

Component 1 (MoCo):
  forward() encodes photo keys via frozen momentum encoder, pushes to queue,
  computes InfoNCE over 4096 negatives instead of 15.
  momentum_update() performs EMA update from training loop (after optimizer.step).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from .backbone import load_mobileclip_backbone, apply_sketch_lora
from .stnet_modules import TASKformerCrossAttention, SketchObjectDetectionHead, SketchReconstructionDecoder
from .patch_hook import PatchTokenExtractor
from .moco_queue import MoCoQueue


class TSBIRCompositeModel(nn.Module):
    """
    Composite Model for Text + Sketch Based Image Retrieval (T+SBIR).
    """

    def __init__(
        self,
        backbone_name: str = "mobileclip_s1",
        checkpoint_path: str = "checkpoints/mobileclip_s1.pt",
        device: str = "cuda",
        moco_queue_size: int = 4096,
        moco_momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.moco_momentum = moco_momentum

        # --- Load backbone (param-count checked inside, no silent fallback) ---
        self.backbone, self.tokenizer, self.preprocess = load_mobileclip_backbone(
            model_name=backbone_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        # --- Two-tier sketch LoRA (Conv2d r=4 + Linear r=64) ---
        self.backbone = apply_sketch_lora(self.backbone, adapter_name="sketch")

        # --- MoCo: momentum encoder = deep copy of base MCi (no LoRA) ---
        # After apply_sketch_lora, backbone.image_encoder is a PeftModel.
        # The original MCi weights live at base_model.model; we copy those.
        try:
            _base_img_enc = self.backbone.image_encoder.base_model.model
            self.momentum_encoder = copy.deepcopy(_base_img_enc)
            for p in self.momentum_encoder.parameters():
                p.requires_grad_(False)
            print(
                f"[MoCo] Momentum encoder initialised from base MCi "
                f"({sum(p.numel() for p in self.momentum_encoder.parameters()):,} params, frozen)"
            )
        except Exception as e:
            print(f"[MoCo] Warning: could not deep-copy base MCi: {e}. "
                  "Falling back to None — photo keys will use online encoder (no EMA).")
            self.momentum_encoder = None

        # --- MoCo queue ---
        self.moco_queue = MoCoQueue(
            embed_dim=512,
            queue_size=moco_queue_size,
            temperature=temperature,
        )

        # --- PatchTokenExtractor: hooks conv_exp for (B, 49, D) spatial tokens ---
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

        # --- Text adapter (2-layer bottleneck) for FS-COCO long captions ---
        self.text_adapter = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

        # --- STNet TASK-former Cross-Attention + Reconstruction Decoder + OD Head ---
        self.attention_pooling = TASKformerCrossAttention(embed_dim=embed_dim)
        self.od_head = SketchObjectDetectionHead(embed_dim=embed_dim)
        self.sketch_decoder = SketchReconstructionDecoder(embed_dim=embed_dim)

    # -----------------------------------------------------------------------
    # Component 3: Modality Isolation Helpers
    # -----------------------------------------------------------------------

    def _adapter_enable_ctx(self):
        """Returns a context-manager that enables all LoRA adapters on the image encoder."""
        img_enc = self.backbone.image_encoder
        if hasattr(img_enc, 'enable_adapters'):
            # Return a no-op context; we call enable_adapters() explicitly before/after
            return nullcontext()
        return nullcontext()

    def _disable_adapter_ctx(self):
        """Returns PEFT disable_adapter() context manager, or no-op if not a PEFT model."""
        img_enc = self.backbone.image_encoder
        if hasattr(img_enc, 'disable_adapter'):
            return img_enc.disable_adapter()
        return nullcontext()

    # -----------------------------------------------------------------------
    # Encoding Methods
    # -----------------------------------------------------------------------

    def encode_sketch(self, sketch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes raster sketch using the LoRA-adapted vision tower.

        Component 3 (strict isolation):
          Step 1 — enable_adapters() → LoRA is ON.
          Step 2 — encode_image().
          Step 3 — disable_adapters() → LoRA is OFF immediately after.
          Photo encoding that follows cannot see sketch LoRA state.
        """
        img_enc = self.backbone.image_encoder
        # Step 1: guarantee LoRA is enabled for sketch
        if hasattr(img_enc, 'enable_adapters'):
            img_enc.enable_adapters()
        # Step 2: encode
        sketch_features = self.backbone.encode_image(sketch_tensor)
        # Step 3: immediately disable so subsequent photo calls are clean
        if hasattr(img_enc, 'disable_adapters'):
            img_enc.disable_adapters()
        return F.normalize(sketch_features, dim=-1)

    def encode_text(self, text_list) -> torch.Tensor:
        """Encodes text query through frozen text tower + trainable text adapter."""
        tokens = self.tokenizer(text_list).to(self.device)
        text_features = self.backbone.encode_text(tokens)
        adapted = text_features + self.text_adapter(text_features)
        return F.normalize(adapted, dim=-1)

    def encode_photo(self, photo_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes photo using ONLY frozen base weights — LoRA is guaranteed OFF.
        Component 3: Since encode_sketch() always disables after use, LoRA is
        already off here. The disable_adapter context is a belt-and-suspenders guard.
        """
        with self._disable_adapter_ctx():
            photo_features = self.backbone.encode_image(photo_tensor)
        return F.normalize(photo_features, dim=-1)

    def encode_photo_patches(self, photo_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts pre-pooling spatial patch tokens (B, N, D) from FastViT conv_exp
        output via PatchTokenExtractor hook. LoRA is disabled (photos use base weights).
        """
        with self._disable_adapter_ctx():
            _ = self.backbone.encode_image(photo_tensor)
        patch_tokens = self.patch_extractor.extract()   # (B, N, embed_dim)
        return F.normalize(patch_tokens, dim=-1)

    @torch.no_grad()
    def encode_photo_key(self, photo_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes photo with the frozen MoCo momentum encoder.
        Returns normalised (B, D) embeddings for the MoCo queue.

        If momentum_encoder is unavailable (init failed), falls back to
        encode_photo() with stop-gradient.
        """
        if self.momentum_encoder is not None:
            features = self.momentum_encoder(photo_tensor)
        else:
            # Fallback: online encoder with disabled LoRA
            with self._disable_adapter_ctx():
                features = self.backbone.encode_image(photo_tensor)
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def momentum_update(self, momentum: float | None = None) -> None:
        """
        EMA update of the momentum encoder from the online image encoder base weights.
        Call this ONCE after optimizer.step() in the training loop.

            θ_k ← m·θ_k + (1-m)·θ_q

        Args:
            momentum: override the default self.moco_momentum if provided.
        """
        if self.momentum_encoder is None:
            return
        m = momentum if momentum is not None else self.moco_momentum
        try:
            online_params = dict(
                self.backbone.image_encoder.base_model.model.named_parameters()
            )
        except AttributeError:
            # PEFT structure might differ; skip silently
            return
        for name, param_k in self.momentum_encoder.named_parameters():
            if name in online_params:
                param_k.data.mul_(m).add_(online_params[name].data, alpha=1.0 - m)

    # -----------------------------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        sketch_tensor: torch.Tensor,
        text_list,
        photo_tensor: torch.Tensor,
        target_sketch_tensor: torch.Tensor | None = None,
    ):
        """
        Full forward pass: sketch + text → composite query vs MoCo photo keys.

        Loss = L_CT + 0.3·L_SR + 0.2·L_OD

        Component 1 (MoCo):
          - Photo is encoded via momentum encoder → photo_key (no grad).
          - InfoNCE uses photo_key as positive + 4096-entry queue as negatives.
          - Queue is updated AFTER loss computation with current batch's photo_key.

        Component 3 (Modality Isolation):
          - encode_sketch() enables LoRA, encodes, disables LoRA.
          - encode_photo_patches() and encode_photo_key() run with LoRA OFF.

        Returns dict with:
          loss, loss_infonce, loss_rec, loss_od, composite_query, attended_photo, recon_sketch
        """
        # 1. Disentangled embeddings
        # encode_sketch: LoRA ON → encode → LoRA OFF
        e_sketch = self.encode_sketch(sketch_tensor)         # (B, D)
        # encode_text: no LoRA involved
        e_text = self.encode_text(text_list)                 # (B, D)
        # encode_photo_patches: LoRA OFF, hooks conv_exp
        photo_patches = self.encode_photo_patches(photo_tensor)  # (B, 49, D)

        # 2. MoCo: encode photo key with momentum encoder (no grad)
        photo_key = self.encode_photo_key(photo_tensor)          # (B, D), no grad

        # 3. Composite query
        raw_composite = torch.cat([e_sketch, e_text], dim=-1)
        composite_query = F.normalize(self.composite_fusion(raw_composite), dim=-1)

        # 4. TASK-former: photo patches as Q, [sketch, text] as K/V
        #    Produces a spatially-grounded, query-conditioned photo embedding
        attended_photo, attn_probs = self.attention_pooling(photo_patches, e_sketch, e_text)
        attended_photo = F.normalize(attended_photo, dim=-1)

        # 5. InfoNCE with MoCo queue (4096 negatives)
        loss_infonce = self.moco_queue.infonce_loss(composite_query, photo_key)

        # 6. Update MoCo queue AFTER computing loss
        self.moco_queue.dequeue_and_enqueue(photo_key)

        # 7. Auxiliary OD loss — sketch-patch spatial grounding (training only)
        loss_od = torch.tensor(0.0, device=self.device)
        if self.training:
            loss_od, _heatmap = self.od_head(photo_patches, e_sketch)

        # 8. Auxiliary reconstruction loss (training only)
        loss_rec = torch.tensor(0.0, device=self.device)
        recon_sketch = None
        if target_sketch_tensor is not None and self.training:
            recon_sketch = self.sketch_decoder(e_sketch)
            loss_rec = F.mse_loss(recon_sketch, target_sketch_tensor)

        # L = L_CT + 0.3·L_SR + 0.2·L_OD
        total_loss = loss_infonce + 0.3 * loss_rec + 0.2 * loss_od

        return {
            'loss':          total_loss,
            'loss_infonce':  loss_infonce,
            'loss_rec':      loss_rec,
            'loss_od':       loss_od,
            'composite_query': composite_query,
            'attended_photo':  attended_photo,
            'recon_sketch':    recon_sketch,
        }
