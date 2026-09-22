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
    r_linear: int = 16,
    r_conv: int = 4,
    lora_alpha_linear: int = 32,
    lora_alpha_conv: int = 8,
    lora_dropout: float = 0.05,
    adapter_name: str = "sketch"
):
    """
    Two-Tier LoRA for MobileCLIP-S1 (FastViT backbone):

    Tier 1 — Attention layers (network.7):
        Target: qkv Linear modules.
        Rank r=16, alpha=32. These layers handle semantic cross-modal attention.

    Tier 2 — RepMixer convolutional stages (network.1–6):
        Target: reparam_conv Conv2d modules (1×1 MobileOneBlock convs).
        Rank r=4, alpha=8. These layers handle spatial edge-detection filters
        critical for freehand sketch domain adaptation.

    Design:
        Uses a SINGLE LoraConfig with rank_pattern / lora_alpha_pattern to
        assign different ranks per-module type, avoiding the need for two
        separate PEFT adapter attachment calls (which would require two
        get_peft_model calls and complex adapter merging).

    Safety:
        - Only leaf names exclusively mapping to nn.Linear are used for Tier 1.
        - Only leaf names exclusively mapping to nn.Conv2d are used for Tier 2.
        - Name collisions (leaf appears as both Linear and Sequential/Conv2d)
          are excluded — same logic as the original fix for 'proj' bug.
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

    # --- Build leaf_name → {set of module types} map ---
    from collections import defaultdict
    from dataclasses import fields as dc_fields
    leaf_to_types: dict[str, set] = defaultdict(set)
    for full_name, module in img_enc.named_modules():
        if not full_name:
            continue
        leaf = full_name.split('.')[-1]
        leaf_to_types[leaf].add(type(module))

    # --- Tier 1: collision-free Linear-only leaf names ---
    safe_linear_leaves = {
        leaf for leaf, types in leaf_to_types.items()
        if types == {nn.Linear}
    }
    preferred_linear = {'qkv', 'proj', 'out_proj', 'in_proj', 'fc1', 'fc2',
                        'q_proj', 'k_proj', 'v_proj'}
    linear_targets = sorted(safe_linear_leaves & preferred_linear)
    if not linear_targets:
        linear_targets = sorted(safe_linear_leaves)

    # --- Tier 2: Dynamic 1x1 Conv2d across the whole vision encoder ---
    conv_targets_set = set()
    for full_name, module in img_enc.named_modules():
        if isinstance(module, nn.Conv2d) and getattr(module, 'groups', 1) == 1:
            leaf = full_name.split('.')[-1]
            conv_targets_set.add(leaf)

    # Prevent PEFT crashes by excluding leaf names used by non-Conv2d modules
    # AND explicitly ban known depthwise convolutional leaf names.
    unsafe_depthwise_leaves = {'reparam_conv', 'conv'}
    
    safe_conv2d_leaves = {
        leaf for leaf, types in leaf_to_types.items()
        if types == {nn.Conv2d} and leaf not in unsafe_depthwise_leaves
    }
    conv_targets = sorted(conv_targets_set & safe_conv2d_leaves)

    all_targets = linear_targets + conv_targets
    assert len(all_targets) > 0, (
        f"apply_sketch_lora: No safe LoRA targets found in {type(img_enc).__name__}."
    )

    # --- Log exclusions ---
    ambiguous_linear = {
        leaf for leaf, types in leaf_to_types.items()
        if nn.Linear in types and len(types) > 1
    }
    if ambiguous_linear:
        print(f"[LoRA] Excluded ambiguous Linear leaves: {sorted(ambiguous_linear)}")

    print(
        f"[LoRA] Two-tier targeting:\n"
        f"  Tier 1 Linear  (r={r_linear}): {linear_targets}\n"
        f"  Tier 2 Conv2d  (r={r_conv}):   {conv_targets}"
    )

    # --- Build rank_pattern for per-module rank overrides ---
    # rank_pattern keys are regex patterns matched against the full module path.
    import re
    rank_pattern = {}
    alpha_pattern = {}
    for target in conv_targets:
        # Match the leaf name precisely at the end of the module path
        pattern = f'.*\\.{re.escape(target)}$'
        rank_pattern[pattern] = r_conv
        alpha_pattern[pattern] = lora_alpha_conv

    # Check whether this PEFT version supports rank_pattern
    peft_config_fields = {f.name for f in dc_fields(LoraConfig)}
    use_rank_pattern = 'rank_pattern' in peft_config_fields and bool(conv_targets)

    if use_rank_pattern:
        config = LoraConfig(
            r=r_linear,                    # default rank for Linear tiers
            lora_alpha=lora_alpha_linear,
            target_modules=all_targets,
            rank_pattern=rank_pattern,          # override Conv2d layers to r_conv
            alpha_pattern=alpha_pattern,        # PEFT API: alpha_pattern (not lora_alpha_pattern)
            lora_dropout=lora_dropout,
            bias="none",
        )
        print(f"[LoRA] Using rank_pattern for per-tier rank assignment.")

    else:
        # Fallback: single rank for all targets (older PEFT without rank_pattern)
        print(
            f"[LoRA] rank_pattern not supported by this PEFT version. "
            f"Falling back to single rank r={r_linear} for all targets."
        )
        config = LoraConfig(
            r=r_linear,
            lora_alpha=lora_alpha_linear,
            target_modules=all_targets,
            lora_dropout=lora_dropout,
            bias="none",
        )

    try:
        model.image_encoder = get_peft_model(img_enc, config, adapter_name=adapter_name)
    except Exception as e:
        if conv_targets:
            print(
                f"[LoRA] Two-tier attach failed ({e}). "
                f"Retrying with Linear-only targets (Tier 1 only)."
            )
            config_fallback = LoraConfig(
                r=r_linear,
                lora_alpha=lora_alpha_linear,
                target_modules=linear_targets,
                lora_dropout=lora_dropout,
                bias="none",
            )
            model.image_encoder = get_peft_model(img_enc, config_fallback, adapter_name=adapter_name)
        else:
            raise RuntimeError(
                f"PEFT get_peft_model failed for adapter '{adapter_name}': {e}"
            ) from e

    # Verify: count LoRA params per tier
    lora_linear_params = sum(
        p.numel() for n, p in model.image_encoder.named_parameters()
        if 'lora_' in n and any(t in n for t in linear_targets)
    )
    lora_conv_params = sum(
        p.numel() for n, p in model.image_encoder.named_parameters()
        if 'lora_' in n and any(t in n for t in conv_targets)
    )
    total_lora = lora_linear_params + lora_conv_params
    assert total_lora > 0, (
        "apply_sketch_lora: no 'lora_' parameters found after attachment."
    )
    print(
        f"[LoRA] '{adapter_name}' attached | "
        f"Linear LoRA params: {lora_linear_params:,} | "
        f"Conv2d LoRA params: {lora_conv_params:,} | "
        f"Total trainable: {total_lora:,}"
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
