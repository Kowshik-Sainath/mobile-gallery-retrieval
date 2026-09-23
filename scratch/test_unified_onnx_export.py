import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
from src.models.composite_model import TSBIRCompositeModel
import onnx
import onnxruntime as ort

device = "cpu"
model = TSBIRCompositeModel(device=device)
ckpt = torch.load("checkpoints/checkpoint_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt)), strict=False)
model.eval()

# Let's inspect PEFT modules in model.backbone.image_encoder
peft_enc = copy.deepcopy(model.backbone.image_encoder)

# In PEFT, every LoraLayer has forward(x).
# Can we modify each LoraLayer's forward to multiply its lora delta by a global/module scalar?
class GatedLoraLayer(nn.Module):
    def __init__(self, original_lora_module):
        super().__init__()
        self.base_layer = original_lora_module.base_layer
        self.lora_A = original_lora_module.lora_A
        self.lora_B = original_lora_module.lora_B
        self.scaling = original_lora_module.scaling
        self.active_adapter = "sketch"

    def forward(self, x: torch.Tensor, is_sketch: torch.Tensor = None) -> torch.Tensor:
        result = self.base_layer(x)
        if is_sketch is not None:
            # lora_B(lora_A(x)) * scaling * is_sketch
            a_mod = self.lora_A[self.active_adapter]
            b_mod = self.lora_B[self.active_adapter]
            scale = self.scaling[self.active_adapter]
            delta = b_mod(a_mod(x)) * scale
            # is_sketch is (1, 1, 1, 1) or scalar
            result = result + delta * is_sketch
        return result

print("Testing GatedLoraLayer concept...")
