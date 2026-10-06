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

# Test excluding early stages:
# Option 1: Exclude network.0, network.1, network.2 from conv quantization
for exclude_early in [
    ['network.0', 'network.1'],
    ['network.0', 'network.1', 'network.2'],
    ['network.0', 'network.1', 'network.2', 'network.3'],
]:
    nodes_to_exclude = []
    for n in m.graph.node:
        # Exclude depthwise
        groups = 1
        for a in n.attribute:
            if a.name == 'group':
                groups = a.i
        if groups > 1 or 'patch_embed' in n.name or 'token_mixer' in n.name:
            nodes_to_exclude.append(n.name)
            continue
        # Exclude early stages
        if any(stage in n.name for stage in exclude_early):
            nodes_to_exclude.append(n.name)

    p_out = f"scratch/test_stage_{len(exclude_early)}.onnx"
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
    print(f"Exclude {exclude_early}: size={sz:.1f} MB (Budget < 45 MB: {sz < 45}) | Cosine sim={cos:.5f} (Gate > 0.95: {cos > 0.95})")
