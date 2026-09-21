"""
MobileCLIP Backbone & PEFT LoRA Adapter Setup.
Integrates Apple's official ml-mobileclip (FastViT-based) with PEFT LoRA adaptation.

Fixes applied:
  BUG 1 — Param-count assertion (<50M); no silent fallback to large backbones.
  BUG 2 — Adapter switching functions raise explicitly instead of silent pass.
  FIX A  — LoRA target module matching verified at runtime; assert > 0 matched.
"""

import os
import torch
import torch.nn as nn

try:
    import mobileclip
    MOBILECLIP_AVAILABLE = True
except ImportError:
    MOBILECLIP_AVAILABLE = False

# Python 3.13 + Windows fix: importlib.metadata.packages_distributions() raises
# OSError [Errno 22] on corrupted dist-info entries. PEFT/transformers calls this
# at import time. Monkey-patch it to return {} on failure so imports still work.
try:
    import importlib.metadata as _imeta
    _orig_pkgs_dist = _imeta.packages_distributions

    def _safe_packages_distributions():
        try:
            return _orig_pkgs_dist()
        except OSError:
            return {}

    _imeta.packages_distributions = _safe_packages_distributions
except Exception:
    pass

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except (ImportError, OSError) as _peft_err:
    PEFT_AVAILABLE = False
    import warnings
    warnings.warn(
        f"PEFT import failed ({_peft_err}). LoRA will not be available.\n"
        "Fix: .venv\\Scripts\\python.exe -m pip install "
        "'transformers>=4.44,<4.52' 'peft>=0.12,<0.14'"
    )



# ---------------------------------------------------------------------------
# BUG 1 FIX — Load backbone with hard param-count check, no silent fallback
# ---------------------------------------------------------------------------

def load_mobileclip_backbone(
    model_name: str = "mobileclip_s1",
    checkpoint_path: str = "checkpoints/mobileclip_s1.pt",
    device: str = "cuda"
):
    """
    Loads Apple's native MobileCLIP backbone and freezes all weights.

    Raises:
        ImportError   — if ml-mobileclip package is not installed.
        FileNotFoundError — if checkpoint file does not exist.
        RuntimeError  — if loading fails OR if param count exceeds 50M
                        (guard against accidentally loading ViT-B/32 or similar).
    """
    if not MOBILECLIP_AVAILABLE:
        raise ImportError(
            "mobileclip package is required. "
            "Install via: pip install -e ./ml-mobileclip"
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"MobileCLIP checkpoint not found at '{checkpoint_path}'. "
            f"Download mobileclip_s1.pt and place it at '{checkpoint_path}'."
        )

    try:
        model, _, preprocess = mobileclip.create_model_and_transforms(
            model_name, pretrained=checkpoint_path
        )
        tokenizer = mobileclip.get_tokenizer(model_name)
        model = model.to(device)
    except Exception as err:
        raise RuntimeError(
            f"Failed to load backbone '{model_name}' from '{checkpoint_path}': {err}. "
            "Silent fallback to larger models is DISABLED."
        ) from err

    # BUG 1 — Hard assertion: must be a small backbone < 50M params
    total_params = sum(p.numel() for p in model.parameters())
    fp32_size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    # BUG 1 — Hard assertion: vision encoder must be a small backbone
    # Check ONLY the vision encoder (image_encoder), not the full model.
    # MobileCLIP-S1 vision: ~21M params | Text: ~63M params | Total: ~85M
    # ViT-B/32 vision encoder: ~86M params — would correctly fail this check.
    img_enc = getattr(model, 'image_encoder', getattr(model, 'visual', None))
    if img_enc is not None:
        vision_params = sum(p.numel() for p in img_enc.parameters())
        vision_fp32_mb = sum(
            p.numel() * p.element_size() for p in img_enc.parameters()
        ) / (1024 * 1024)
    else:
        vision_params = total_params
        vision_fp32_mb = fp32_size_mb

    MAX_VISION_PARAMS = 30_000_000  # 30M: catches ViT-B/32 (86M), passes MobileCLIP-S1 (21M)
    if vision_params > MAX_VISION_PARAMS:
        raise RuntimeError(
            f"Vision encoder of '{model_name}' has {vision_params:,} parameters ({vision_fp32_mb:.1f} MB). "
            f"Exceeds the {MAX_VISION_PARAMS//1_000_000}M budget for mobile deployment. "
            "Check that the correct checkpoint was loaded — ViT-B/32 fallback is not acceptable."
        )

    print(
        f"[Backbone] '{model_name}' loaded | "
        f"Vision params: {vision_params:,} ({vision_fp32_mb:.1f} MB FP32) | "
        f"Total model params: {total_params:,}"
    )

    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer, preprocess


# ---------------------------------------------------------------------------
# BUG 2 + FIX A — LoRA attachment with verified target modules
# ---------------------------------------------------------------------------

def apply_sketch_lora(
    model,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    adapter_name: str = "sketch"
):
    """
    Attaches a named PEFT LoRA adapter ('sketch') to the vision encoder.

    FIX A: Detects actual Linear modules in the vision encoder by iterating
    named_modules() and checking isinstance(module, nn.Linear) — no substring
    guessing. Asserts at least one target is matched before attaching.

    For MobileCLIP-S1 (FastViT): the MHSA AttentionBlocks in model.network.7
    contain Linear layers (qkv, proj) — these are the correct LoRA targets.
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is required for LoRA adapter support. "
            "Install with: pip install peft"
        )

    # Get the image encoder sub-module
    img_enc = getattr(model, 'image_encoder', getattr(model, 'visual', None))
    if img_enc is None:
        raise RuntimeError(
            "Cannot find image encoder on backbone model. "
            "Expected attribute 'image_encoder' or 'visual'."
        )

    # FIX A — Collect a mapping of leaf_name → set of all module types with that name.
    # PEFT target_modules matches by leaf name substring across the WHOLE model tree.
    # If a leaf name like 'proj' is shared by both nn.Linear AND nn.Sequential,
    # PEFT will attempt to wrap the Sequential too and crash.
    # Solution: only use leaf names that are *exclusively* nn.Linear everywhere.
    from collections import defaultdict
    leaf_to_types: dict[str, set] = defaultdict(set)
    for full_name, module in img_enc.named_modules():
        if not full_name:
            continue
        leaf = full_name.split('.')[-1]
        leaf_to_types[leaf].add(type(module))

    # Leaf names where every module with that name is nn.Linear — no collisions
    safe_linear_leaves = {
        leaf for leaf, types in leaf_to_types.items()
        if types == {nn.Linear}
    }

    # Prefer canonical attention projection names from the safe set
    preferred = {'qkv', 'proj', 'out_proj', 'in_proj', 'fc1', 'fc2', 'q_proj', 'k_proj', 'v_proj'}
    found_targets = list(safe_linear_leaves & preferred)

    if not found_targets:
        # Second pass: try all safe linear leaves
        found_targets = list(safe_linear_leaves)

    if not found_targets:
        # Last resort: list all unique Linear leaf names (may have collisions, but is better
        # than failing entirely). PEFT will raise a clear error if this happens.
        all_linear_leaves = set()
        for full_name, module in img_enc.named_modules():
            if isinstance(module, nn.Linear):
                all_linear_leaves.add(full_name.split('.')[-1])
        found_targets = list(all_linear_leaves)
        print(
            f"[LoRA] Warning: no collision-free leaf names found. "
            f"Using all Linear leaves (may hit PEFT errors): {sorted(found_targets)}"
        )

    assert len(found_targets) > 0, (
        f"apply_sketch_lora: No Linear modules found in {type(img_enc).__name__}. "
        "Cannot attach LoRA adapter."
    )

    # Log which leaf names are safe and which were excluded due to type collisions
    excluded = {
        leaf for leaf, types in leaf_to_types.items()
        if isinstance(list(types)[0], type) and nn.Linear in types and len(types) > 1
    }
    if excluded:
        print(f"[LoRA] Excluded ambiguous leaf names (shared with non-Linear modules): {sorted(excluded)}")
    print(f"[LoRA] Attaching '{adapter_name}' adapter to modules: {sorted(found_targets)}")

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=found_targets,
        lora_dropout=lora_dropout,
        bias="none"
    )

    try:
        model.image_encoder = get_peft_model(img_enc, config, adapter_name=adapter_name)
    except Exception as e:
        # Re-raise: silent fallback is not allowed here
        raise RuntimeError(
            f"PEFT get_peft_model failed for adapter '{adapter_name}': {e}"
        ) from e

    # Verify attachment: count lora_ prefixed parameters
    lora_params = [
        n for n, _ in model.image_encoder.named_parameters()
        if 'lora_' in n
    ]
    assert len(lora_params) > 0, (
        "apply_sketch_lora: get_peft_model ran without error, "
        "but no 'lora_' parameters found. LoRA was not attached."
    )
    trainable = sum(
        p.numel() for n, p in model.image_encoder.named_parameters()
        if 'lora_' in n
    )
    print(
        f"[LoRA] '{adapter_name}' attached successfully | "
        f"LoRA parameter tensors: {len(lora_params)} | "
        f"Trainable LoRA params: {trainable:,}"
    )

    return model


# ---------------------------------------------------------------------------
# BUG 2 FIX — Adapter switching: no bare except pass
# ---------------------------------------------------------------------------

def set_active_sketch_adapter(model, adapter_name: str = "sketch"):
    """
    Activates the named sketch LoRA adapter on the vision encoder.
    Raises RuntimeError if the encoder is not a PEFT model.
    """
    img_enc = getattr(model, 'image_encoder', getattr(model, 'visual', None))

    if img_enc is None:
        raise RuntimeError(
            "set_active_sketch_adapter: no 'image_encoder' or 'visual' attribute found."
        )

    if not hasattr(img_enc, 'set_adapter'):
        raise RuntimeError(
            "set_active_sketch_adapter: image_encoder is not a PEFT model "
            "(no 'set_adapter' method). Was apply_sketch_lora() called?"
        )

    img_enc.set_adapter(adapter_name)


def disable_vision_adapters(model):
    """
    Disables all LoRA adapters on the vision encoder for clean photo encoding.
    Uses PEFT's context-manager-safe disable_adapters() method.
    Raises RuntimeError if the encoder is not a PEFT model.
    """
    img_enc = getattr(model, 'image_encoder', getattr(model, 'visual', None))

    if img_enc is None:
        raise RuntimeError(
            "disable_vision_adapters: no 'image_encoder' or 'visual' attribute found."
        )

    if not hasattr(img_enc, 'disable_adapters'):
        raise RuntimeError(
            "disable_vision_adapters: image_encoder is not a PEFT model "
            "(no 'disable_adapters' method). Was apply_sketch_lora() called?"
        )

    img_enc.disable_adapters()
