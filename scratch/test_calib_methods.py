import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod
from export_mobile import VisionCalibrationDataReader

p_fp32 = "exported_models/photo_encoder_fp32.onnx"
sess_fp32 = ort.InferenceSession(p_fp32, providers=['CPUExecutionProvider'])
dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)
out_fp32 = sess_fp32.run(None, {"image_input": dummy})[0].squeeze(0)

calib = VisionCalibrationDataReader(data_dir="fscoco", input_name="image_input", key="photo", n_samples=50)

for method_name, method in [
    ("MinMax", CalibrationMethod.MinMax),
    ("Entropy", CalibrationMethod.Entropy),
    ("Percentile", CalibrationMethod.Percentile),
]:
    p_out = f"scratch/photo_calib_{method_name}.onnx"
    calib.rewind()
    try:
        quantize_static(
            model_input=p_fp32, model_output=p_out,
            calibration_data_reader=calib,
            calibrate_method=method,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )
        sess = ort.InferenceSession(p_out, providers=['CPUExecutionProvider'])
        out = sess.run(None, {"image_input": dummy})[0].squeeze(0)
        cos = float(np.dot(out_fp32, out) / (np.linalg.norm(out_fp32) * np.linalg.norm(out) + 1e-8))
        sz = os.path.getsize(p_out) / (1024 * 1024)
        print(f"Method {method_name}: size={sz:.1f} MB | Cosine sim={cos:.5f}")
    except Exception as e:
        print(f"Method {method_name} failed: {e}")
