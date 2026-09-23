import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod
from export_mobile import VisionCalibrationDataReader

p_fp32 = "exported_models/photo_encoder_fp32.onnx"
sess_fp32 = ort.InferenceSession(p_fp32, providers=['CPUExecutionProvider'])

calib = VisionCalibrationDataReader(data_dir="fscoco", input_name="image_input", key="photo", n_samples=50)
real_sample = calib.data[0]["image_input"]
out_fp32 = sess_fp32.run(None, {"image_input": real_sample})[0].squeeze(0)

# Test 1: QuantFormat.QOperator
print("--- Testing QuantFormat.QOperator ---")
p_qop = "scratch/photo_qop.onnx"
calib.rewind()
try:
    quantize_static(
        model_input=p_fp32, model_output=p_qop,
        calibration_data_reader=calib,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    sess = ort.InferenceSession(p_qop, providers=['CPUExecutionProvider'])
    out = sess.run(None, {"image_input": real_sample})[0].squeeze(0)
    cos = float(np.dot(out_fp32, out) / (np.linalg.norm(out_fp32) * np.linalg.norm(out) + 1e-8))
    sz = os.path.getsize(p_qop) / (1024 * 1024)
    print(f"QOperator: size={sz:.1f} MB | Cosine sim on real photo: {cos:.5f}")
except Exception as e:
    print(f"QOperator failed: {e}")

# Test 2: QDQ without reduce_range
print("--- Testing QDQ without reduce_range, QUInt8 ---")
p_noreduce = "scratch/photo_noreduce.onnx"
calib.rewind()
try:
    quantize_static(
        model_input=p_fp32, model_output=p_noreduce,
        calibration_data_reader=calib,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    sess = ort.InferenceSession(p_noreduce, providers=['CPUExecutionProvider'])
    out = sess.run(None, {"image_input": real_sample})[0].squeeze(0)
    cos = float(np.dot(out_fp32, out) / (np.linalg.norm(out_fp32) * np.linalg.norm(out) + 1e-8))
    sz = os.path.getsize(p_noreduce) / (1024 * 1024)
    print(f"QDQ QUInt8: size={sz:.1f} MB | Cosine sim on real photo: {cos:.5f}")
except Exception as e:
    print(f"QDQ QUInt8 failed: {e}")
