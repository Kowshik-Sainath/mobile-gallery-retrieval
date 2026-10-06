import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
from src.models.composite_model import TSBIRCompositeModel
from src.models.backbone import set_active_sketch_adapter
import onnx
import onnxruntime as ort

device = "cpu"
model = TSBIRCompositeModel(device=device)
ckpt_path = "checkpoints/checkpoint_best.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
model.eval()

# Let's test PyTorch outputs for a test tensor
test_img = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    pt_photo = model.encode_photo(test_img)
    pt_sketch = model.encode_sketch(test_img)

diff_pt = (pt_photo - pt_sketch).abs().max().item()
print(f"PyTorch Photo vs Sketch max difference: {diff_pt:.4f} (they are distinct: {diff_pt > 0.01})")

# Let's inspect how to dynamically scale PEFT LoRA in image_encoder
# In PEFT, every lora module has .scaling['sketch']
# What if we monkeypatch or wrap the lora forward with a dynamic multiplier?
class DynamicLoRAScaling(nn.Module):
    def __init__(self, peft_encoder):
        super().__init__()
        self.encoder = peft_encoder
        self.multiplier = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor, is_sketch: torch.Tensor) -> torch.Tensor:
        # We can dynamically pass is_sketch to modules or use a hook
        pass
