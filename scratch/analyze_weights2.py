import sys, os
sys.path.insert(0, os.path.abspath('.'))
import onnx

m = onnx.load("exported_models/photo_encoder_fp32.onnx")
stage_sizes = {}
for init in m.graph.initializer:
    nbytes = len(init.raw_data)
    name = init.name
    stage = name.split('/')[1] if '/' in name else name
    if 'network' in name:
        # e.g. model.model.network.3...
        for part in name.split('.'):
            if part.isdigit():
                stage = f"network.{part}"
                break
    stage_sizes[stage] = stage_sizes.get(stage, 0) + nbytes

for k, v in sorted(stage_sizes.items(), key=lambda x: -x[1]):
    print(f"  {k:<35}: {v / (1024*1024):6.2f} MB")
