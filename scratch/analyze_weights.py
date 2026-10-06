import sys, os
sys.path.insert(0, os.path.abspath('.'))
import onnx
import numpy as np

m = onnx.load("exported_models/photo_encoder_fp32.onnx")
init_sizes = {}
for init in m.graph.initializer:
    # Size in bytes
    nbytes = len(init.raw_data)
    name = init.name
    # Categorize
    prefix = name.split('/')[1] if '/' in name else 'other'
    parts = name.split('.')
    cat = parts[2] if len(parts) > 2 else prefix
    init_sizes[cat] = init_sizes.get(cat, 0) + nbytes

print("Initializer size breakdown:")
for cat, sz in sorted(init_sizes.items(), key=lambda x: -x[1]):
    print(f"  {cat:<25}: {sz / (1024*1024):6.2f} MB")
print(f"Total initializer size: {sum(init_sizes.values()) / (1024*1024):.2f} MB")
