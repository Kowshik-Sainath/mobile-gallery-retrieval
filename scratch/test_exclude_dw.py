import sys, os
sys.path.insert(0, os.path.abspath('.'))
import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType
from export_mobile import VisionCalibrationDataReader

p_fp32 = "exported_models/photo_encoder_fp32.onnx"
m = onnx.load(p_fp32)

# Find all depthwise conv nodes and token mixer convs to exclude
nodes_to_exclude = []
for n in m.graph.node:
    if n.op_type == 'Conv':
        groups = 1
        for a in n.attribute:
            if a.name == 'group':
                groups = a.i
        # Exclude depthwise convolutions and patch embed
        if groups > 1 or 'patch_embed' in n.name or 'token_mixer' in n.name:
            nodes_to_exclude.append(n.name)

print(f"Total Conv nodes: {len([n for n in m.graph.node if n.op_type=='Conv'])}")
print(f"Excluding {len(nodes_to_exclude)} sensitive Conv nodes (depthwise, patch_embed, token_mixer)")

sess_fp32 = ort.InferenceSession(p_fp32, providers=['CPUExecutionProvider'])
dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)
out_fp32 = sess_fp32.run(None, {"image_input": dummy})[0].squeeze(0)

calib = VisionCalibrationDataReader(data_dir="fscoco", input_name="image_input", key="photo", n_samples=50)

p_out = "scratch/photo_no_dw.onnx"
quantize_static(
    model_input=p_fp32, model_output=p_out,
    calibration_data_reader=calib,
    nodes_to_exclude=nodes_to_exclude,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=False
)

sz = os.path.getsize(p_out) / (1024 * 1024)
sess = ort.InferenceSession(p_out, providers=['CPUExecutionProvider'])
out = sess.run(None, {"image_input": dummy})[0].squeeze(0)
cos = float(np.dot(out_fp32, out) / (np.linalg.norm(out_fp32) * np.linalg.norm(out) + 1e-8))
print(f"Result with depthwise excluded: size={sz:.1f} MB (Budget < 45 MB) | Cosine sim={cos:.5f} (Gate > 0.95)")
