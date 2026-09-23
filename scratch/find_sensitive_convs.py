import sys, os
sys.path.insert(0, os.path.abspath('.'))
import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType
from export_mobile import VisionCalibrationDataReader

p_fp32 = "exported_models/photo_encoder_fp32.onnx"
m = onnx.load(p_fp32)
all_conv_nodes = [n.name for n in m.graph.node if n.op_type == 'Conv']

sess_fp32 = ort.InferenceSession(p_fp32, providers=['CPUExecutionProvider'])
calib = VisionCalibrationDataReader(data_dir="fscoco", input_name="image_input", key="photo", n_samples=30)
real_sample = calib.data[0]["image_input"]
out_fp32 = sess_fp32.run(None, {"image_input": real_sample})[0].squeeze(0)

# Test quantizing only specific types of convs:
# 1. fc1 and fc2 (the 1x1 MLP convs)
fc_convs = [n for n in all_conv_nodes if 'fc1' in n or 'fc2' in n]
print(f"Total fc1/fc2 convs: {len(fc_convs)}")

# What if we ONLY quantize MatMul + fc1 + fc2?
p_fc = "scratch/photo_fc_only.onnx"
calib.rewind()
nodes_to_quantize = fc_convs + [n.name for n in m.graph.node if n.op_type == 'MatMul']
nodes_to_exclude = [n.name for n in m.graph.node if n.name not in nodes_to_quantize]

quantize_static(
    model_input=p_fp32,
    model_output=p_fc,
    calibration_data_reader=calib,
    nodes_to_exclude=nodes_to_exclude,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=False
)

sz = os.path.getsize(p_fc) / (1024 * 1024)
sess = ort.InferenceSession(p_fc, providers=['CPUExecutionProvider'])
out = sess.run(None, {"image_input": real_sample})[0].squeeze(0)
cos = float(np.dot(out_fp32, out) / (np.linalg.norm(out_fp32) * np.linalg.norm(out) + 1e-8))
print(f"MatMul + fc1/fc2: size={sz:.1f} MB (Budget < 45 MB: {sz < 45}) | Cosine sim={cos:.5f} (Gate > 0.95: {cos > 0.95})")
