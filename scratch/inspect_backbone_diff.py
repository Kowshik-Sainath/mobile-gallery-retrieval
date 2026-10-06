import os
import onnx
import numpy as np

p_photo = "exported_models/photo_encoder_int8.onnx"
p_sketch = "exported_models/sketch_encoder_int8.onnx"

print(f"Photo INT8 size:  {os.path.getsize(p_photo):,} bytes")
print(f"Sketch INT8 size: {os.path.getsize(p_sketch):,} bytes")
with open(p_photo, "rb") as f1, open(p_sketch, "rb") as f2:
    b1 = f1.read()
    b2 = f2.read()
print(f"Byte identical: {b1 == b2}")

# Count differing bytes
diff_bytes = sum(c1 != c2 for c1, c2 in zip(b1, b2)) + abs(len(b1) - len(b2))
print(f"Differing bytes: {diff_bytes:,} / {len(b1):,} ({diff_bytes / len(b1) * 100:.2f}%)")

m_photo = onnx.load(p_photo)
m_sketch = onnx.load(p_sketch)

inits_photo = {init.name: onnx.numpy_helper.to_array(init) for init in m_photo.graph.initializer}
inits_sketch = {init.name: onnx.numpy_helper.to_array(init) for init in m_sketch.graph.initializer}

print(f"Initializers in Photo: {len(inits_photo)}, Sketch: {len(inits_sketch)}")
diff_weights = []
for name, arr_p in inits_photo.items():
    if name in inits_sketch:
        arr_s = inits_sketch[name]
        if arr_p.shape == arr_s.shape:
            diff = np.max(np.abs(arr_p.astype(np.float32) - arr_s.astype(np.float32)))
            if diff > 0:
                diff_weights.append((name, arr_p.shape, arr_p.dtype, diff))

print(f"\nNumber of differing initializer weights: {len(diff_weights)}")
for name, shape, dtype, diff in diff_weights[:15]:
    print(f"  {name}: shape={shape}, dtype={dtype}, max_diff={diff:.6f}")
