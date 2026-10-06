import os
import sys
import copy
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

p_photo_int8 = "exported_models/photo_encoder_int8.onnx"
p_sketch_int8 = "exported_models/sketch_encoder_int8.onnx"
npz_delta = "exported_models/sketch_lora_delta.npz"

m_photo_int8 = onnx.load(p_photo_int8)
m_sketch_int8 = onnx.load(p_sketch_int8)

# Load low-rank delta
delta_data = np.load(npz_delta)

# Map names
qkv_map = {
    'model.model.network.7.0.token_mixer.qkv.weight': 'onnx::MatMul_1753',
    'model.model.network.7.1.token_mixer.qkv.weight': 'onnx::MatMul_1758',
    'model.model.network.7.2.token_mixer.qkv.weight': 'onnx::MatMul_1763',
    'model.model.network.7.3.token_mixer.qkv.weight': 'onnx::MatMul_1768'
}

reconstructed_deltas = {}
for key in delta_data.files:
    if key.endswith(".A"):
        mod_name = key[:-2]
        a_w = delta_data[f"{mod_name}.A"]
        b_w = delta_data[f"{mod_name}.B"]
        scale = delta_data[f"{mod_name}.scale"][0]
        onnx_name = mod_name.replace("base_model.", "") + ".weight"
        if a_w.ndim == 4:
            b_mat = b_w.squeeze(-1).squeeze(-1)
            a_mat = a_w.squeeze(-1).squeeze(-1)
            delta = (b_mat @ a_mat * scale)[:, :, None, None]
        else:
            delta = (b_w @ a_w) * scale
        reconstructed_deltas[onnx_name] = delta

for qkv_n, mm_n in qkv_map.items():
    if qkv_n in reconstructed_deltas:
        reconstructed_deltas[mm_n] = reconstructed_deltas[qkv_n].T

# Check which initializers in photo_int8 match reconstructed_deltas
m_patched = copy.deepcopy(m_photo_int8)
patched = 0
for init in m_patched.graph.initializer:
    if init.name in reconstructed_deltas:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32:
            new_arr = arr + reconstructed_deltas[init.name]
            init.CopyFrom(numpy_helper.from_array(new_arr.astype(arr.dtype), name=init.name))
            patched += 1

print(f"Patched {patched} float32 initializers in photo_int8.")

sess_sketch_int8 = ort.InferenceSession(p_sketch_int8, providers=['CPUExecutionProvider'])
sess_patched = ort.InferenceSession(m_patched.SerializeToString(), providers=['CPUExecutionProvider'])

test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
out_sketch_int8 = sess_sketch_int8.run(None, {"sketch_input": test_input})[0].squeeze(0)
out_patched = sess_patched.run(None, {"image_input": test_input})[0].squeeze(0)

cos = float(np.dot(out_sketch_int8, out_patched) / (np.linalg.norm(out_sketch_int8) * np.linalg.norm(out_patched) + 1e-8))
print(f"Patched INT8 Base vs True Sketch INT8: cosine similarity={cos:.5f}")
