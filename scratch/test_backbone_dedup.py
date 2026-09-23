import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from src.models.composite_model import TSBIRCompositeModel

device = "cpu"
model = TSBIRCompositeModel(device=device)
ckpt_path = "checkpoints/checkpoint_best.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
model.eval()

# Inspect LoRA parameters in model.backbone.image_encoder
lora_params = {}
for name, p in model.backbone.image_encoder.named_parameters():
    if "lora_" in name:
        lora_params[name] = p.detach().cpu().numpy()

print(f"Total LoRA tensors: {len(lora_params)}")
total_params = sum(p.size for p in lora_params.values())
total_bytes = sum(p.nbytes for p in lora_params.values())
print(f"Total LoRA parameters: {total_params:,}")
print(f"Total LoRA FP32 size: {total_bytes / (1024*1024):.2f} MB")
print(f"Total LoRA FP16 size: {total_bytes / (2 * 1024*1024):.2f} MB")

# Save as npz to test disk footprint
npz_path = "scratch/sketch_lora_delta.npz"
np.savez_compressed(npz_path, **lora_params)
print(f"Compressed NPZ footprint: {os.path.getsize(npz_path) / (1024*1024):.2f} MB")

# Inspect the modules where LoRA is attached
for k in list(lora_params.keys())[:10]:
    print(f"  {k}: shape {lora_params[k].shape}")
