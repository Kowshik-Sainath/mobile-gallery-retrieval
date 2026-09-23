import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn.functional as F
from src.models.composite_model import TSBIRCompositeModel

m = TSBIRCompositeModel(device='cpu')
m.eval()
x = torch.randn(1, 3, 224, 224)
p1 = m.encode_photo(x)

# Check 1: base_model.model(x)
p2 = F.normalize(m.backbone.image_encoder.base_model.model(x), dim=-1)
print('diff p1 vs base_model.model:', (p1 - p2).abs().max().item())

# Check 2: disable_adapter context
with m.backbone.image_encoder.disable_adapter():
    p3 = F.normalize(m.backbone.image_encoder(x), dim=-1)
print('diff p1 vs disable_adapter():', (p1 - p3).abs().max().item())

# Check 3: what is inside m.backbone.image_encoder.base_model.model?
print('type of base_model.model:', type(m.backbone.image_encoder.base_model.model))
# Check what layers inside base_model.model are LoraLayer
for name, mod in m.backbone.image_encoder.base_model.model.named_modules():
    if 'lora' in type(mod).__name__.lower():
        print('Found lora layer:', name, type(mod).__name__)
        print('disable_adapters flag:', getattr(mod, 'disable_adapters', None))
        break
