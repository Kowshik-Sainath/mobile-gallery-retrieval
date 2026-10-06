import os
import sys
import torch

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

peft_enc = model.backbone.image_encoder
for name, m in peft_enc.named_modules():
    if hasattr(m, 'get_delta_weight'):
        delta_w = m.get_delta_weight('sketch')
        print(f"{name}: delta_weight shape {delta_w.shape}, min={delta_w.min():.5f}, max={delta_w.max():.5f}")
        break
