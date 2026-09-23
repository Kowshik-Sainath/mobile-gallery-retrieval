import os
import sys
import copy
import torch
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

sys.path.insert(0, os.path.abspath('.'))
from src.models.composite_model import TSBIRCompositeModel

device = "cpu"
model = TSBIRCompositeModel(device=device)
ckpt = torch.load("checkpoints/checkpoint_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt)), strict=False)
model.eval()

# Extract from PEFT get_delta_weight
peft_enc = model.backbone.image_encoder
peft_deltas = {}
for name, m in peft_enc.named_modules():
    if hasattr(m, 'get_delta_weight'):
        delta_w = m.get_delta_weight('sketch').detach().cpu().numpy()
        onnx_name = name.replace("base_model.", "") + ".weight"
        peft_deltas[onnx_name] = delta_w

# Match the 4 qkv weights to onnx::MatMul names
qkv_names = [
    'model.model.network.7.0.token_mixer.qkv.weight',
    'model.model.network.7.1.token_mixer.qkv.weight',
    'model.model.network.7.2.token_mixer.qkv.weight',
    'model.model.network.7.3.token_mixer.qkv.weight'
]
onnx_matmul_names = [
    'onnx::MatMul_1753',
    'onnx::MatMul_1758',
    'onnx::MatMul_1763',
    'onnx::MatMul_1768'
]
for qkv_n, mm_n in zip(qkv_names, onnx_matmul_names):
    # PyTorch Linear weight is (out_features, in_features)
    # ONNX MatMul B matrix is (in_features, out_features)
    peft_deltas[mm_n] = peft_deltas[qkv_n].T

p_photo = "exported_models/photo_encoder_fp32.onnx"
p_sketch = "exported_models/sketch_encoder_fp32.onnx"
m_photo = onnx.load(p_photo)

m_patched = copy.deepcopy(m_photo)
patched = 0
for init in m_patched.graph.initializer:
    if init.name in peft_deltas:
        arr = numpy_helper.to_array(init)
        new_arr = arr + peft_deltas[init.name]
        init.CopyFrom(numpy_helper.from_array(new_arr.astype(arr.dtype), name=init.name))
        patched += 1

print(f"Patched {patched}/90 initializers from PEFT get_delta_weight.")

sess_sketch = ort.InferenceSession(p_sketch, providers=['CPUExecutionProvider'])
sess_patched = ort.InferenceSession(m_patched.SerializeToString(), providers=['CPUExecutionProvider'])

test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
out_sketch = sess_sketch.run(None, {"sketch_input": test_input})[0].squeeze(0)
out_patched = sess_patched.run(None, {"image_input": test_input})[0].squeeze(0)

diff = np.max(np.abs(out_sketch - out_patched))
cos = float(np.dot(out_sketch, out_patched) / (np.linalg.norm(out_sketch) * np.linalg.norm(out_patched) + 1e-8))
print(f"PEFT Delta Patch vs Sketch Encoder: max_diff={diff:.2e}, cosine similarity={cos:.8f}")
