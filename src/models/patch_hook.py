"""
PatchTokenExtractor for MobileCLIP-S1 (FastViT backbone).

MobileCLIP-S1's image_encoder is a FastViT model with this tail:
  model.network.7  — 4x AttentionBlock (last transformer stage, spatial (B, C, H, W))
  model.conv_exp   — MobileOneBlock expansion conv  → (B, 512, 7, 7) at 224x224 input
  model.head.pool  — GlobalPool2D                   → (B, 512)  [SQUEEZE POINT]

We hook after model.conv_exp (before GlobalPool) to capture (B, C, H, W) feature maps,
then reshape to (B, N, D) where N = H*W spatial patch positions.

This gives REAL diverse spatial tokens suitable for SketchGuidedAttentionPooling —
not the degenerate .repeat(1,49,1) fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenExtractor(nn.Module):
    """
    Registers a forward hook on FastViT's conv_exp layer (the last spatial
    feature map before global pooling) to capture patch tokens.

    For MobileCLIP-S1 at 224x224 input:
        conv_exp output → (B, 512, 7, 7)  → reshape → (B, 49, 512)

    A learned linear projection is added if the channel dim != embed_dim.

    Usage:
        extractor = PatchTokenExtractor(image_encoder, embed_dim=512)
        patch_tokens = extractor(image_tensor)   # (B, N, D)
        del extractor   # call .remove() to clean up hooks
    """

    def __init__(self, image_encoder: nn.Module, embed_dim: int = 512):
        super().__init__()
        self._captured: torch.Tensor | None = None
        self._hook_handle = None

        conv_exp = self._find_conv_exp(image_encoder)
        if conv_exp is None:
            # Debug: show what attributes the encoder has
            attrs = [a for a in dir(image_encoder) if not a.startswith('_')]
            raise RuntimeError(
                f"PatchTokenExtractor: could not find 'conv_exp' layer.\n"
                f"  image_encoder type: {type(image_encoder).__name__}\n"
                f"  Top-level attributes: {attrs[:20]}\n"
                f"  Searched: image_encoder, .model, .base_model, .base_model.model, "
                f".base_model.model.model"
            )

        self._hook_handle = conv_exp.register_forward_hook(self._capture_hook)

        # Detect device from model parameters for the dry-run dummy tensor
        try:
            _model_device = next(image_encoder.parameters()).device
        except StopIteration:
            _model_device = torch.device('cpu')

        # Determine channel dimension by doing a dry-run forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=_model_device)
            _ = image_encoder(dummy)            # triggers hook, sets _captured
            if self._captured is None:
                raise RuntimeError("Hook did not fire during dry-run forward pass.")
            captured_channels = self._captured.shape[1]  # (B, C, H, W)

        # Projection: map C → embed_dim if needed
        if captured_channels != embed_dim:
            self.proj = nn.Linear(captured_channels, embed_dim)
        else:
            self.proj = nn.Identity()

        self._embed_dim = embed_dim
        self._n_patches: int | None = None  # set after first real forward pass

    @staticmethod
    def _find_conv_exp(image_encoder: nn.Module):
        """
        Searches for the 'conv_exp' attribute at any depth in the module tree.

        Robust to any PEFT wrapping depth:
          - Plain MCi:    image_encoder.model.conv_exp
          - PEFT-wrapped: image_encoder.base_model.model.model.conv_exp

        Uses named_modules() so it works regardless of PEFT version or nesting.
        """
        # Check the root itself first (unlikely but safe)
        if hasattr(image_encoder, 'conv_exp'):
            return image_encoder.conv_exp

        # Walk ALL submodules at any depth — stop at first hit
        for _name, submodule in image_encoder.named_modules():
            if hasattr(submodule, 'conv_exp'):
                return submodule.conv_exp

        return None

    def _capture_hook(self, module, input, output):
        """Called by PyTorch after conv_exp's forward(). Captures (B, C, H, W)."""
        self._captured = output.detach() if not self.training else output


    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensor: (B, 3, 224, 224)
        Returns:
            patch_tokens: (B, N, embed_dim)  where N = H*W (e.g. 49 for 7x7)
        """
        self._captured = None
        # The image_encoder is called externally; this module only post-processes.
        # Call via the parent composite model, not here directly.
        raise RuntimeError(
            "Do not call PatchTokenExtractor.forward() directly. "
            "Use extract_after_encoder(image_encoder, image_tensor) instead."
        )

    def extract(self) -> torch.Tensor:
        """
        Returns the patch tokens captured by the hook from the last forward pass.
        Must be called immediately after image_encoder(image_tensor).

        Returns:
            patch_tokens: (B, N, embed_dim)
        """
        if self._captured is None:
            raise RuntimeError(
                "No feature map captured. Call image_encoder(image_tensor) first."
            )
        feat = self._captured           # (B, C, H, W)
        B, C, H, W = feat.shape
        N = H * W

        # Reshape (B, C, H, W) → (B, N, C)
        tokens = feat.permute(0, 2, 3, 1).reshape(B, N, C)  # (B, N, C)

        # Project to embed_dim if needed
        tokens = self.proj(tokens)      # (B, N, embed_dim)

        assert N > 1, (
            f"PatchTokenExtractor: got N={N} patch tokens. "
            "Expected N>1 (e.g. 49 for 7x7 feature maps at 224x224 input)."
        )

        # Verify tokens are not degenerate (all identical)
        token_std = tokens.std().item()
        assert token_std > 1e-5, (
            f"PatchTokenExtractor: patch tokens have near-zero std ({token_std:.2e}). "
            "Feature maps may be collapsed — check that conv_exp hook is firing correctly."
        )

        self._n_patches = N
        return tokens

    def remove(self):
        """Removes the forward hook. Call when done to avoid memory leaks."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        try:
            self.remove()
        except Exception:
            pass
