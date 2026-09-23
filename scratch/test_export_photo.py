import sys, os
sys.path.insert(0, os.path.abspath('.'))

import copy
import torch
import torch.nn.functional as F
from src.models.composite_model import TSBIRCompositeModel
from export_mobile import PhotoEncoderExportable, try_onnx_export

m = TSBIRCompositeModel(device='cpu')
ckpt = torch.load('checkpoints/checkpoint_best.pt', map_location='cpu', weights_only=False)
m.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
m.eval()

photo_enc = copy.deepcopy(m.backbone.image_encoder).unload()
photo_model = PhotoEncoderExportable(photo_enc)
photo_model.eval()

dummy_img = torch.zeros(1, 3, 224, 224)
test_vec = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    pt_photo_test = m.encode_photo(test_vec)
    wrp_photo_test = photo_model(test_vec)
    pre_photo_diff = (pt_photo_test - wrp_photo_test).abs().max().item()
print('Pre-export photo diff:', pre_photo_diff)
assert pre_photo_diff < 1e-4

ok = try_onnx_export(
    photo_model, (dummy_img,), "scratch/test_photo.onnx",
    input_names=["image_input"], output_names=["image_embedding"],
    dynamic_axes={"image_input": {0: "batch"}, "image_embedding": {0: "batch"}},
)
print('Export success:', ok)
