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
peft_deltas = {}
for name, m in peft_enc.named_modules():
    if hasattr(m, 'get_delta_weight'):
        delta_w = m.get_delta_weight('sketch')
        onnx_name = name.replace("base_model.", "") + ".weight"
        peft_deltas[onnx_name] = delta_w

print(f"Total PEFT modules with get_delta_weight: {len(peft_deltas)}")

import onnx
from onnx import numpy_helper
m_photo = onnx.load("exported_models/photo_encoder_fp32.onnx")
m_sketch = onnx.load("exported_models/sketch_encoder_fp32.onnx")

inits_p = {init.name: numpy_helper.to_array(init) for init in m_photo.graph.initializer}
inits_s = {init.name: numpy_helper.to_array(init) for init in m_sketch.graph.initializer}

diff_inits = {}
for k in inits_p:
    if k in inits_s:
        d = inits_s[k] - inits_p[k]
        if abs(d).max() > 0:
            diff_inits[k] = d

print(f"Total differing ONNX initializers: {len(diff_inits)}")

missing_in_peft = [k for k in diff_inits if k not in peft_deltas]
missing_in_onnx = [k for k in peft_deltas if k not in diff_inits]
print(f"Missing in PEFT: {missing_in_peft}")
print(f"Missing in ONNX: {missing_in_onnx}")
