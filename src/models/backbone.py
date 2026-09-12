"""
MobileCLIP Backbone & PEFT LoRA Adapter Setup.
Integrates apple/ml-mobileclip checkpoints with huggingface/peft lightweight adaptation.
"""

import torch
import torch.nn as nn

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def load_mobileclip_backbone(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cuda"):
    """
    Loads distilled MobileCLIP or OpenCLIP backbone model and freezes all weights.
    """
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip_torch is required. Please install via pip install open_clip_torch.")

    # Try loading MobileCLIP variant if model_name starts with mobileclip, otherwise fallback to OpenCLIP ViT-B/32
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained, 
            device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception:
        # Fallback to standard lightweight ViT-B-32
        print(f"Notice: '{model_name}' not available directly in open_clip registry. Falling back to ViT-B-32.")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='laion2b_s34b_b79k', 
            device=device
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Freeze all parameters in the backbone
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer, preprocess


def apply_sketch_lora(model, r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Attaches PEFT LoRA adapter modules to vision tower for lightweight sketch adaptation.
    """
    if not PEFT_AVAILABLE:
        print("Warning: PEFT library not installed. Returning backbone without LoRA.")
        return model

    target_modules = ["out_proj", "in_proj", "c_fc", "c_proj", "query", "value"]
    
    # Filter target modules present in vision tower
    found_targets = []
    for name, module in model.visual.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, (nn.Linear, nn.Conv2d)):
                found_targets.append(target)
    
    found_targets = list(set(found_targets))
    if not found_targets:
        found_targets = ["out_proj", "c_fc", "c_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=found_targets,
        lora_dropout=lora_dropout,
        bias="none"
    )

    try:
        model.visual = get_peft_model(model.visual, config)
        print(f"Successfully attached LoRA adapter to vision tower target modules: {found_targets}")
    except Exception as e:
        print(f"Notice: LoRA attachment via PEFT failed ({e}). Proceeding with task-specific projection heads.")

    return model
