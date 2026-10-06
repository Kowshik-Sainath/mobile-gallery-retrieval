import torch
import numpy as np
from src.models.composite_model import TSBIRCompositeModel

m = TSBIRCompositeModel(device="cpu")
ckpt = torch.load("checkpoints/checkpoint_best.pt", map_location="cpu", weights_only=False)
m.load_state_dict(ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt)), strict=False)
m.eval()

# Check with 20 random and real images
x = torch.randn(20, 3, 224, 224)
with torch.no_grad():
    p = m.encode_photo(x).numpy()
    s = m.encode_sketch(x).numpy()

cos_sims = [np.dot(p[i], s[i]) for i in range(20)]
print(f"Photo vs Sketch cosine similarity for identical input images: {np.mean(cos_sims):.4f} (range {np.min(cos_sims):.4f} - {np.max(cos_sims):.4f})")
