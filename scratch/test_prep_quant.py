import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from export_mobile import VisionCalibrationDataReader

p_fp32 = "exported_models/photo_encoder_fp32.onnx"
p_prep = "scratch/photo_prep.onnx"
p_int8 = "scratch/photo_prep_int8.onnx"

print("1. Running quant_pre_process...")
quant_pre_process(p_fp32, p_prep)
print(f"Preprocessed model saved ({os.path.getsize(p_prep)/(1024*1024):.1f} MB)")

print("2. Calibrating and quantizing...")
calib = VisionCalibrationDataReader(data_dir="fscoco", input_name="image_input", key="photo", n_samples=50)
real_sample = calib.data[0]["image_input"]

sess_fp32 = ort.InferenceSession(p_fp32, providers=['CPUExecutionProvider'])
out_fp32 = sess_fp32.run(None, {"image_input": real_sample})[0].squeeze(0)

quantize_static(
    model_input=p_prep,
    model_output=p_int8,
    calibration_data_reader=calib,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=True,
)

sz = os.path.getsize(p_int8) / (1024 * 1024)
sess = ort.InferenceSession(p_int8, providers=['CPUExecutionProvider'])
out = sess.run(None, {"image_input": real_sample})[0].squeeze(0)
cos = float(np.dot(out_fp32, out) / (np.linalg.norm(out_fp32) * np.linalg.norm(out) + 1e-8))
print(f"Preprocessed INT8: size={sz:.1f} MB | Cosine sim={cos:.5f}")
