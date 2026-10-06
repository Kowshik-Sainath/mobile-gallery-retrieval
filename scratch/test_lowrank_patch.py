import os
import sys
import time
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
ckpt_path = "checkpoints/checkpoint_best.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
model.eval()

# Extract low-rank LoRA parameters: A, B, and scaling
peft_enc = model.backbone.image_encoder
lora_deltas_dict = {}

t0 = time.time()
# Map each adapted module name to its (lora_A, lora_B, scaling)
adapted_modules = {}
for name, module in peft_enc.named_modules():
    if hasattr(module, 'lora_A') and 'sketch' in module.lora_A:
        a_weight = module.lora_A['sketch'].weight.detach().cpu().numpy()
        b_weight = module.lora_B['sketch'].weight.detach().cpu().numpy()
        scale = module.scaling['sketch']
        adapted_modules[name] = (a_weight, b_weight, scale)

print(f"Extracted {len(adapted_modules)} adapted modules in {(time.time() - t0)*1000:.1f} ms")

# Save as compact NPZ
export_dict = {}
for mod_name, (a_w, b_w, scale) in adapted_modules.items():
    export_dict[f"{mod_name}.A"] = a_w
    export_dict[f"{mod_name}.B"] = b_w
    export_dict[f"{mod_name}.scale"] = np.array([scale], dtype=np.float32)

npz_path = "exported_models/sketch_lora_delta.npz"
np.savez_compressed(npz_path, **export_dict)
sz_mb = os.path.getsize(npz_path) / (1024 * 1024)
print(f"Saved {npz_path}: {sz_mb:.2f} MB")

# Now let's test reconstructing Delta W and applying to photo_encoder_fp32.onnx
p_photo = "exported_models/photo_encoder_fp32.onnx"
p_sketch = "exported_models/sketch_encoder_fp32.onnx"
m_photo = onnx.load(p_photo)

t0 = time.time()
data = np.load(npz_path)

# Compute dense deltas from low-rank A and B
# Conv2d: A is (r, in_ch, 1, 1), B is (out_ch, r, 1, 1) -> W is (out_ch, in_ch, 1, 1)
# Linear: A is (r, in_feat), B is (out_feat, r) -> W is (out_feat, in_feat)
# In ONNX initializer names: e.g. "model.model.network.0.0.convffn.fc1.weight"
# Module name in PEFT: "base_model.model.model.network.0.0.convffn.fc1"
reconstructed_deltas = {}
for key in data.files:
    if key.endswith(".A"):
        mod_name = key[:-2]
        a_w = data[f"{mod_name}.A"]
        b_w = data[f"{mod_name}.B"]
        scale = data[f"{mod_name}.scale"][0]
        # Match ONNX initializer name
        # Strip "base_model." prefix
        onnx_name = mod_name.replace("base_model.", "") + ".weight"
        if a_w.ndim == 4: # Conv2d 1x1
            # b_w: (out_ch, r, 1, 1), a_w: (r, in_ch, 1, 1)
            b_mat = b_w.squeeze(-1).squeeze(-1) # (out_ch, r)
            a_mat = a_w.squeeze(-1).squeeze(-1) # (r, in_ch)
            delta_dense = (b_mat @ a_mat * scale)[:, :, None, None]
        else: # Linear
            delta_dense = (b_w @ a_w) * scale
        reconstructed_deltas[onnx_name] = delta_dense

m_sketch_from_delta = copy.deepcopy(m_photo)
patched = 0
for init in m_sketch_from_delta.graph.initializer:
    if init.name in reconstructed_deltas:
        arr = numpy_helper.to_array(init)
        new_arr = arr + reconstructed_deltas[init.name]
        init.CopyFrom(numpy_helper.from_array(new_arr.astype(arr.dtype), name=init.name))
        patched += 1

recon_time_ms = (time.time() - t0) * 1000
print(f"Reconstructed and patched {patched} weights in {recon_time_ms:.1f} ms")

# Test parity with sketch_encoder_fp32
sess_ref = ort.InferenceSession(p_sketch, providers=['CPUExecutionProvider'])
sess_recon = ort.InferenceSession(m_sketch_from_delta.SerializeToString(), providers=['CPUExecutionProvider'])

test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
out_ref = sess_ref.run(None, {"sketch_input": test_input})[0].squeeze(0)
out_recon = sess_recon.run(None, {"image_input": test_input})[0].squeeze(0)

diff = np.max(np.abs(out_ref - out_recon))
cos = float(np.dot(out_ref, out_recon) / (np.linalg.norm(out_ref) * np.linalg.norm(out_recon) + 1e-8))
print(f"FP32 Low-rank Delta Patch vs True Sketch Encoder: max_diff={diff:.2e}, cos={cos:.8f}")
