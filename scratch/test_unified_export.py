import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DynamicLoRAVisionEncoder(nn.Module):
    def __init__(self, peft_image_encoder):
        super().__init__()
        self.encoder = peft_image_encoder

    def forward(self, x: torch.Tensor, is_sketch: torch.Tensor) -> torch.Tensor:
        # If is_sketch == 1.0, scale = 1.0; if is_sketch == 0.0, scale = 0.0
        # In PEFT, lora_scale = alpha / r. If we multiply adapter output by is_sketch:
        feats = self.encoder(x)
        return F.normalize(feats, dim=-1)

dummy_img = torch.randn(1, 3, 224, 224)
dummy_flag = torch.tensor([1.0], dtype=torch.float32)

wrapper = DynamicLoRAVisionEncoder(model.backbone.image_encoder)
out = wrapper(dummy_img, dummy_flag)
print(f"Wrapper output shape: {out.shape}")
