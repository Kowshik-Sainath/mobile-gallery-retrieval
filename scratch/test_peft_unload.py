import sys, os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn.functional as F
from src.models.composite_model import TSBIRCompositeModel

m = TSBIRCompositeModel(device='cpu')
ckpt = torch.load('checkpoints/checkpoint_best.pt', map_location='cpu', weights_only=False)
m.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
m.eval()

x = torch.randn(1, 3, 224, 224)
p_photo = m.encode_photo(x)
p_sketch = m.encode_sketch(x)

print('Photo vs Sketch max diff:', (p_photo - p_sketch).abs().max().item())

# Test unload() for photo encoder
unloaded_photo_model = m.backbone.image_encoder.unload()
print('Unloaded type:', type(unloaded_photo_model))
with torch.no_grad():
    p_unloaded = F.normalize(unloaded_photo_model(x), dim=-1)
print('Photo vs Unloaded max diff:', (p_photo - p_unloaded).abs().max().item())

# Now test on a fresh model for sketch merge_and_unload()
m2 = TSBIRCompositeModel(device='cpu')
m2.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
m2.eval()

from src.models.backbone import set_active_sketch_adapter
set_active_sketch_adapter(m2.backbone, 'sketch')
merged_sketch_model = m2.backbone.image_encoder.merge_and_unload()
print('Merged type:', type(merged_sketch_model))
with torch.no_grad():
    p_merged = F.normalize(merged_sketch_model(x), dim=-1)
print('Sketch vs Merged max diff:', (p_sketch - p_merged).abs().max().item())
