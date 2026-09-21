"""
Unit Tests for T+SBIR Correctness Fixes A1–A5.

Updated to match fixed APIs:
  - backbone.py: load_mobileclip_backbone (checkpoint_path, not pretrained= kwarg)
  - composite_model.py: TSBIRCompositeModel (backbone_name, checkpoint_path)
  - patch_hook.py: PatchTokenExtractor (real N > 1 tokens, non-degenerate std)
  - combiner.py: FeedbackCombinerReranker (feature_dim kwarg)

Run:
  pytest tests/test_correctness_fixes.py -v
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest
import torch
import torch.nn.functional as F

from src.models.backbone import load_mobileclip_backbone, apply_sketch_lora
from src.models.composite_model import TSBIRCompositeModel
from src.models.stnet_modules import SketchGuidedAttentionPooling
from src.reranker.combiner import FeedbackCombinerReranker


CHECKPOINT_PATH = "checkpoints/mobileclip_s1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Test A1 — Small backbone loading and hard failure on invalid name
# ---------------------------------------------------------------------------

def test_a1_small_backbone_loads_under_50m_params():
    """BUG 1: Backbone loads, param count < 50M, FP32 size printed, no silent fallback."""
    model, tokenizer, preprocess = load_mobileclip_backbone(
        model_name="mobileclip_s1",
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )
    assert model is not None
    assert tokenizer is not None
    assert preprocess is not None

    # BUG 1: Vision encoder params must be < 30M
    # MobileCLIP-S1 vision encoder: ~21.4M | ViT-B/32 vision: ~86M
    img_enc = getattr(model, 'image_encoder', getattr(model, 'visual', None))
    assert img_enc is not None, "No image_encoder attribute found on backbone."
    vision_params = sum(p.numel() for p in img_enc.parameters())
    assert vision_params < 30_000_000, (
        f"Expected vision encoder < 30M params (small backbone), got {vision_params:,}. "
        "ViT-B/32 fallback may have occurred (its vision encoder is ~86M)."
    )
    print(f"[A1] Vision encoder params: {vision_params:,} OK")


def test_a1_bad_checkpoint_raises():
    """BUG 1: Explicitly raises FileNotFoundError when checkpoint is missing."""
    caught = False
    try:
        load_mobileclip_backbone(
            model_name="mobileclip_s1",
            checkpoint_path="nonexistent_path/model.pt",
            device=DEVICE,
        )
    except FileNotFoundError:
        caught = True
    assert caught, "Expected FileNotFoundError for missing checkpoint path."
    print("[A1] FileNotFoundError raised correctly OK")


# ---------------------------------------------------------------------------
# Test A2 — LoRA disentanglement: sketch vs photo produce different embeddings
# ---------------------------------------------------------------------------

def test_a2_disentangled_sketch_photo_adapters():
    """BUG 2: encode_sketch and encode_photo produce measurably different embeddings."""
    model = TSBIRCompositeModel(
        backbone_name="mobileclip_s1",
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )
    model.to(DEVICE)
    model.eval()

    dummy_img = torch.randn(2, 3, 224, 224, device=DEVICE)

    with torch.no_grad():
        sketch_emb = model.encode_sketch(dummy_img)
        photo_emb  = model.encode_photo(dummy_img)

    # Both must be unit-normalized
    assert torch.allclose(
        torch.norm(sketch_emb, dim=-1),
        torch.ones(2, device=DEVICE), atol=1e-3
    ), "sketch_emb should be unit-normalized."

    assert torch.allclose(
        torch.norm(photo_emb, dim=-1),
        torch.ones(2, device=DEVICE), atol=1e-3
    ), "photo_emb should be unit-normalized."

    # Embeddings must differ (LoRA is active for sketch, disabled for photo)
    cosine_sim = F.cosine_similarity(sketch_emb, photo_emb, dim=-1).mean().item()
    diff = 1.0 - cosine_sim
    assert diff > 1e-4, (
        f"Sketch and Photo embeddings are nearly identical (cosine diff={diff:.6f}). "
        "LoRA adapter switching may be a no-op — check BUG 2 fix."
    )
    print(f"[A2] Sketch vs Photo embedding cosine diff: {diff:.4f} OK")


# ---------------------------------------------------------------------------
# Test A3 — Patch token extraction: real (B, N, D) with N > 1, non-degenerate
# ---------------------------------------------------------------------------

def test_a3_patch_token_extraction_real_tokens():
    """BUG 3: encode_photo_patches returns (B, N, D) with N > 1 and non-zero std."""
    model = TSBIRCompositeModel(
        backbone_name="mobileclip_s1",
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )
    model.to(DEVICE)
    model.eval()

    dummy_img = torch.randn(2, 3, 224, 224, device=DEVICE)

    with torch.no_grad():
        patches = model.encode_photo_patches(dummy_img)

    B, N, D = patches.shape
    assert B == 2,  f"Expected batch size 2, got {B}."
    assert N > 1,   f"Expected N > 1 patch tokens, got N={N}. Degenerate repeat fallback?"
    assert D == 512, f"Expected embed_dim=512, got D={D}."

    # Non-degenerate: tokens must not all be identical
    token_std = patches.std().item()
    assert token_std > 1e-4, (
        f"Patch tokens have near-zero std ({token_std:.2e}). "
        "Tokens are likely from the degenerate .repeat(1, 49, 1) fallback — BUG 3 not fixed."
    )
    print(f"[A3] Patch tokens: B={B}, N={N}, D={D}, std={token_std:.4f} OK")


def test_a3_attention_pooling_on_real_patches():
    """BUG 3: SketchGuidedAttentionPooling over real patch tokens produces diverse attention."""
    model = TSBIRCompositeModel(
        backbone_name="mobileclip_s1",
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )
    model.to(DEVICE)
    model.eval()

    dummy_img = torch.randn(2, 3, 224, 224, device=DEVICE)

    with torch.no_grad():
        e_sketch = model.encode_sketch(dummy_img)
        patches  = model.encode_photo_patches(dummy_img)
        attended, attn_map = model.attention_pooling(e_sketch, patches)

    B, N, D = patches.shape
    assert attended.shape == (2, 512), f"Expected attended shape (2, 512), got {attended.shape}."
    assert attn_map.shape == (2, 1, N), f"Expected attn_map shape (2, 1, {N}), got {attn_map.shape}."

    # Attention weights should NOT be uniform (which happens with degenerate tokens)
    attn_std = attn_map.std().item()
    assert attn_std > 1e-6, (
        f"Attention weights are near-uniform (std={attn_std:.2e}). "
        "This indicates degenerate patch tokens — BUG 3 may still be present."
    )
    print(f"[A3] Attention pooling: attended={attended.shape}, attn_std={attn_std:.4f} OK")


# ---------------------------------------------------------------------------
# Test A5 — Combiner reranker forward pass shape
# ---------------------------------------------------------------------------

def test_a5_combiner_forward_shape():
    """A5: FeedbackCombinerReranker produces (B, 1) logits."""
    combiner = FeedbackCombinerReranker(feature_dim=512).to(DEVICE)
    dummy_query     = torch.randn(4, 512, device=DEVICE)
    dummy_candidate = torch.randn(4, 512, device=DEVICE)

    logits = combiner(dummy_query, dummy_candidate)
    assert logits.shape == (4, 1), f"Expected (4, 1) logits, got {logits.shape}."
    print(f"[A5] Combiner logits: {logits.shape} OK")


def test_a5_combiner_pretrained_loads():
    """A5: Pretrained combiner weights load successfully if combiner_pretrained.pt exists."""
    ckpt_path = "checkpoints/combiner_pretrained.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"combiner_pretrained.pt not found — run pretrain_reranker.py first.")

    combiner = FeedbackCombinerReranker(feature_dim=512).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    combiner.load_state_dict(ckpt['state_dict'])
    print(f"[A5] Pretrained combiner loaded (epoch {ckpt.get('epoch','?')}, loss {ckpt.get('loss',0):.4f}) OK")


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("Running T+SBIR Correctness Fix Unit Tests (A1-A5)...")
    print("=" * 65)

    test_a1_small_backbone_loads_under_50m_params()
    print("[PASSED] A1: Small backbone loads, params < 50M OK")

    test_a1_bad_checkpoint_raises()
    print("[PASSED] A1: FileNotFoundError on bad checkpoint OK")

    test_a2_disentangled_sketch_photo_adapters()
    print("[PASSED] A2: Sketch vs Photo embeddings differ (LoRA disentangled) OK")

    test_a3_patch_token_extraction_real_tokens()
    print("[PASSED] A3: Patch tokens are real (B, N>1, D), non-degenerate OK")

    test_a3_attention_pooling_on_real_patches()
    print("[PASSED] A3: Attention pooling produces non-uniform weights OK")

    test_a5_combiner_forward_shape()
    print("[PASSED] A5: Combiner produces (B,1) logits OK")

    print("=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
