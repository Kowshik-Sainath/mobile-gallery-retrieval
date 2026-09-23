import sys, os
sys.path.insert(0, os.path.abspath('.'))
import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType
from export_mobile import VisionCalibrationDataReader

p_fp32 = "exported_models/photo_encoder_fp32.onnx"
m = onnx.load(p_fp32)

sess_fp32 = ort.InferenceSession(p_fp32, providers=['CPUExecutionProvider'])
calib = VisionCalibrationDataReader(data_dir="fscoco", input_name="image_input", key="photo", n_samples=30)
real_sample = calib.data[0]["image_input"]
out_fp32 = sess_fp32.run(None, {"image_input": real_sample})[0].squeeze(0)

# Target nodes:
# All MatMul/Gemm
target_nodes = [n.name for n in m.graph.node if n.op_type in ('MatMul', 'Gemm')]

# Plus 1x1 convs in network.4 (fc1, fc2)
for n in m.graph.node:
    if 'network.4' in n.name and ('fc1' in n.name or 'fc2' in n.name):
        target_nodes.append(n.name)
    # Plus 1x1 convs in network.2 (fc1, fc2)
    if 'network.2' in n.name and ('fc1' in n.name or 'fc2' in n.name):
        target_nodes.append(n.name)
    # Plus head proj
    if 'head' in n.name:
        target_nodes.append(n.name)

nodes_to_exclude = [n.name for n in m.graph.node if n.name not in target_nodes]

p_out = "scratch/photo_sub45.onnx"
calib.rewind()
quantize_static(
    model_input=p_fp32,
    model_output=p_out,
    calibration_data_reader=calib,
    nodes_to_exclude=nodes_to_exclude,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=False
)
sz = os.path.getsize(p_out) / (1024 * 1024)
sess = ort.InferenceSession(p_out, providers=['CPUExecutionProvider'])
out = sess.run(None, {"image_input": real_sample})[0].squeeze(0)
cos = float(np.dot(out_fp32, out) / (np.linalg.norm(out_fp32) * np.linalg.norm(out) + 1e-8))
print(f"Target sub-45MB: size={sz:.2f} MB (Budget < 45 MB: {sz < 45.0}) | Cosine sim={cos:.5f} (Gate > 0.95: {cos > 0.95})")
