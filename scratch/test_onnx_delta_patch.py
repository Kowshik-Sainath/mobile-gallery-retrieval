import os
import copy
import onnx
from onnx import numpy_helper
import numpy as np
import onnxruntime as ort

p_photo = "exported_models/photo_encoder_fp32.onnx"
p_sketch = "exported_models/sketch_encoder_fp32.onnx"

m_photo = onnx.load(p_photo)
m_sketch = onnx.load(p_sketch)

inits_photo = {init.name: numpy_helper.to_array(init) for init in m_photo.graph.initializer}
inits_sketch = {init.name: numpy_helper.to_array(init) for init in m_sketch.graph.initializer}

# Compute the exact delta tensors: Delta_W = W_sketch - W_photo
delta_tensors = {}
for name, arr_s in inits_sketch.items():
    if name in inits_photo:
        arr_p = inits_photo[name]
        diff = arr_s.astype(np.float32) - arr_p.astype(np.float32)
        if np.max(np.abs(diff)) > 0:
            delta_tensors[name] = diff

print(f"Total delta weight tensors: {len(delta_tensors)}")
total_delta_params = sum(d.size for d in delta_tensors.values())
print(f"Total delta parameters: {total_delta_params:,}")

# Let's save the deltas to a compressed file
delta_path = "scratch/sketch_lora_delta_dense.npz"
np.savez_compressed(delta_path, **delta_tensors)
print(f"Dense delta footprint: {os.path.getsize(delta_path) / (1024*1024):.2f} MB")

# Now let's test: Start from photo base model, apply delta_tensors in memory, and run inference!
m_patched = copy.deepcopy(m_photo)
patched_count = 0
for init in m_patched.graph.initializer:
    if init.name in delta_tensors:
        orig_arr = numpy_helper.to_array(init)
        new_arr = orig_arr + delta_tensors[init.name]
        new_init = numpy_helper.from_array(new_arr.astype(orig_arr.dtype), name=init.name)
        init.CopyFrom(new_init)
        patched_count += 1

print(f"Patched {patched_count} initializers in photo base model.")

# Now test inference vs sketch_encoder_fp32.onnx
sess_sketch = ort.InferenceSession(p_sketch, providers=['CPUExecutionProvider'])
sess_patched = ort.InferenceSession(m_patched.SerializeToString(), providers=['CPUExecutionProvider'])

test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
out_sketch = sess_sketch.run(None, {"sketch_input": test_input})[0].squeeze(0)
# m_patched has input_name 'image_input'
out_patched = sess_patched.run(None, {"image_input": test_input})[0].squeeze(0)

diff = np.max(np.abs(out_sketch - out_patched))
cos = float(np.dot(out_sketch, out_patched) / (np.linalg.norm(out_sketch) * np.linalg.norm(out_patched) + 1e-8))
print(f"Patched Base vs Sketch Encoder: max_diff={diff:.2e}, cosine similarity={cos:.8f}")
