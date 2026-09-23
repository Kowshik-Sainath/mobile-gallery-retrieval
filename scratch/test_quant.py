import sys, os
sys.path.insert(0, os.path.abspath('.'))
import os
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType, CalibrationDataReader
from export_mobile import VisionCalibrationDataReader

p_fp32 = "exported_models/photo_encoder_fp32.onnx"
sess_fp32 = ort.InferenceSession(p_fp32, providers=['CPUExecutionProvider'])

dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)
out_fp32 = sess_fp32.run(None, {"image_input": dummy})[0].squeeze(0)

# Test Option A: quantize_dynamic
p_dyn = "scratch/photo_dyn.onnx"
quantize_dynamic(p_fp32, p_dyn, weight_type=QuantType.QInt8)
sess_dyn = ort.InferenceSession(p_dyn, providers=['CPUExecutionProvider'])
out_dyn = sess_dyn.run(None, {"image_input": dummy})[0].squeeze(0)
cos_dyn = float(np.dot(out_fp32, out_dyn) / (np.linalg.norm(out_fp32) * np.linalg.norm(out_dyn) + 1e-8))
print(f"Dynamic INT8 size: {os.path.getsize(p_dyn)/(1024*1024):.1f} MB | Cosine sim: {cos_dyn:.5f}")

# Test Option B: quantize_static with QUInt8 activations and per_channel=False
p_stat_u = "scratch/photo_stat_u.onnx"
calib = VisionCalibrationDataReader(data_dir="fscoco", input_name="image_input", key="photo", n_samples=50)
quantize_static(
    model_input=p_fp32, model_output=p_stat_u,
    calibration_data_reader=calib,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=False
)
sess_stat_u = ort.InferenceSession(p_stat_u, providers=['CPUExecutionProvider'])
out_stat_u = sess_stat_u.run(None, {"image_input": dummy})[0].squeeze(0)
cos_stat_u = float(np.dot(out_fp32, out_stat_u) / (np.linalg.norm(out_fp32) * np.linalg.norm(out_stat_u) + 1e-8))
print(f"Static QUInt8 per_channel=False size: {os.path.getsize(p_stat_u)/(1024*1024):.1f} MB | Cosine sim: {cos_stat_u:.5f}")

# Test Option C: quantize_static with QInt8 and per_channel=False
p_stat_s = "scratch/photo_stat_s.onnx"
calib.rewind()
quantize_static(
    model_input=p_fp32, model_output=p_stat_s,
    calibration_data_reader=calib,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=False
)
sess_stat_s = ort.InferenceSession(p_stat_s, providers=['CPUExecutionProvider'])
out_stat_s = sess_stat_s.run(None, {"image_input": dummy})[0].squeeze(0)
cos_stat_s = float(np.dot(out_fp32, out_stat_s) / (np.linalg.norm(out_fp32) * np.linalg.norm(out_stat_s) + 1e-8))
print(f"Static QInt8 per_channel=False size: {os.path.getsize(p_stat_s)/(1024*1024):.1f} MB | Cosine sim: {cos_stat_s:.5f}")
