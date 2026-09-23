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
from src.data.fscoco_dataset import FSCOCODataset

device = "cpu"
model = TSBIRCompositeModel(device=device)
ckpt = torch.load("checkpoints/checkpoint_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt)), strict=False)
model.eval()

# 1. Base photo graph
p_photo = "exported_models/photo_encoder_fp32.onnx"
m_photo = onnx.load(p_photo)

# 2. Extract delta from model
peft_enc = model.backbone.image_encoder
peft_deltas = {}
for name, m in peft_enc.named_modules():
    if hasattr(m, 'get_delta_weight'):
        delta_w = m.get_delta_weight('sketch').detach().cpu().numpy()
        onnx_name = name.replace("base_model.", "") + ".weight"
        peft_deltas[onnx_name] = delta_w

# 4 MatMul layers
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
    peft_deltas[mm_n] = peft_deltas[qkv_n].T

# Apply delta to base photo model in memory
m_sketch_from_delta = copy.deepcopy(m_photo)
for init in m_sketch_from_delta.graph.initializer:
    if init.name in peft_deltas:
        arr = numpy_helper.to_array(init)
        new_arr = arr + peft_deltas[init.name]
        init.CopyFrom(numpy_helper.from_array(new_arr.astype(arr.dtype), name=init.name))

sess_delta = ort.InferenceSession(m_sketch_from_delta.SerializeToString(), providers=['CPUExecutionProvider'])

# Test on the 5 sanity test samples
dataset = FSCOCODataset(root_dir="fscoco", split="test")
indices = [0, 50, 100, 150, 200]

print("=" * 65)
print("SKETCH PARITY CHECK: PyTorch encode_sketch() vs. Base-Plus-Delta")
print("=" * 65)

sims = []
for idx in indices:
    sample = dataset[idx]
    s_tensor = sample["sketch"].unsqueeze(0)
    with torch.no_grad():
        pt_out = model.encode_sketch(s_tensor).squeeze(0).numpy()
    onnx_out = sess_delta.run(None, {"image_input": s_tensor.numpy()})[0].squeeze(0)
    cos = float(np.dot(pt_out, onnx_out) / (np.linalg.norm(pt_out) * np.linalg.norm(onnx_out) + 1e-8))
    sims.append(cos)
    print(f"Sample {idx:3d}: Cosine Similarity = {cos:.5f} (Gate > 0.95: {cos > 0.95})")

min_sim = min(sims)
mean_sim = np.mean(sims)
print("-" * 65)
print(f"Min: {min_sim:.5f} | Mean: {mean_sim:.5f} | Gate Status: {'PASS' if min_sim > 0.95 else 'FAIL'}")
print("=" * 65)
