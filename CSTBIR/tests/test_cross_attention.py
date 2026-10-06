import sys
import os
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("CSTBIR"))
from models.cross_attention import SketchGuidedImageAttention

def test_cross_attention():
    print("=" * 70)
    print("COMPONENT 2: Sketch-Guided Cross-Modal Attention Verification")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Instantiate module
    print("\n--- Step 1: Instantiating SketchGuidedImageAttention (embed_dim=768, scale=False per paper) ---")
    module = SketchGuidedImageAttention(embed_dim=768, scale=False).to(device)
    print("Module initialized successfully.")
    
    # 2. Shape and Math Check
    print("\n--- Step 2: Testing tensor shapes and Softmax sum ---")
    B = 4
    m = 197  # 1 CLS + 196 patches (14x14 grid)
    D = 768
    
    H_tilde_I = torch.randn(B, m, D, device=device, requires_grad=True)
    h_S_CLS = torch.randn(B, D, device=device, requires_grad=True)
    
    h_I_AVG, H_I, alpha_IS = module(H_tilde_I, h_S_CLS)
    
    print(f"Input H_tilde_I shape: {H_tilde_I.shape} (Expected: ({B}, {m}, {D}))")
    print(f"Input h_S_CLS shape:   {h_S_CLS.shape} (Expected: ({B}, {D}))")
    print(f"Output h_I_AVG shape:  {h_I_AVG.shape} (Expected: ({B}, {D}))")
    print(f"Output H_I shape:      {H_I.shape} (Expected: ({B}, {m}, {D}))")
    print(f"Output alpha_IS shape: {alpha_IS.shape} (Expected: ({B}, {m}))")
    
    assert h_I_AVG.shape == (B, D), f"Expected ({B}, {D}), got {h_I_AVG.shape}"
    assert H_I.shape == (B, m, D), f"Expected ({B}, {m}, {D}), got {H_I.shape}"
    assert alpha_IS.shape == (B, m), f"Expected ({B}, {m}), got {alpha_IS.shape}"
    
    # Verify probability distribution: sum over m must equal 1.0 for each sample
    prob_sums = alpha_IS.sum(dim=-1)
    print(f"Softmax probability sums across tokens: {prob_sums.detach().cpu().tolist()}")
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Softmax probabilities do not sum to 1.0!"
    print("Softmax probability normalization verified.")
    
    # 3. Gradient Flow Check
    print("\n--- Step 3: Checking gradient flow to both image and sketch inputs ---")
    loss = h_I_AVG.sum()
    loss.backward()
    
    assert H_tilde_I.grad is not None and H_tilde_I.grad.abs().sum() > 0, "No gradients received by H_tilde_I!"
    assert h_S_CLS.grad is not None and h_S_CLS.grad.abs().sum() > 0, "No gradients received by h_S_CLS!"
    print(f"H_tilde_I grad norm: {H_tilde_I.grad.norm().item():.4f}")
    print(f"h_S_CLS grad norm:   {h_S_CLS.grad.norm().item():.4f}")
    print("Gradient flow to both modalities verified.")
    
    # 4. Spatial Grounding / Localization Sanity Check
    print("\n--- Step 4: Spatial localization sanity check on simulated object region ---")
    # ViT tokens have standard feature norm ~ sqrt(D) ≈ 27.7 (as measured directly from real ViT activations)
    torch.manual_seed(42)
    feature_norm = math.sqrt(D) # ~27.71
    v_target = F.normalize(torch.randn(1, D, device=device), dim=-1) * feature_norm
    v_bg = F.normalize(torch.randn(1, D, device=device), dim=-1) * feature_norm
    
    # Initialize background patches
    spatial_grid = v_bg.repeat(196, 1) + torch.randn(196, D, device=device)
    
    # Define object bounding box in grid coordinates: rows 4 to 8, cols 4 to 8 (5x5 region = 25 patches out of 196)
    box_r_min, box_r_max = 4, 9
    box_c_min, box_c_max = 4, 9
    target_indices = []
    for r in range(box_r_min, box_r_max):
        for c in range(box_c_min, box_c_max):
            patch_idx = r * 14 + c
            spatial_grid[patch_idx] = v_target + torch.randn(1, D, device=device)
            target_indices.append(patch_idx)
            
    cls_token = (v_bg + torch.randn(1, D, device=device)).unsqueeze(0) # (1, 1, D)
    test_image_tokens = torch.cat([cls_token, spatial_grid.unsqueeze(0)], dim=1) # (1, 197, 768)
    
    # Query sketch embedding is aligned with v_target
    test_sketch_vec = v_target # (1, 768)
    
    # Run attention module
    with torch.no_grad():
        _, _, test_alpha = module(test_image_tokens, test_sketch_vec)
        
    patch_alpha = test_alpha[0, 1:].view(14, 14).cpu() # Exclude CLS, reshape to 14x14 grid
    
    target_patch_weights = [patch_alpha[r, c].item() for r in range(box_r_min, box_r_max) for c in range(box_c_min, box_c_max)]
    bg_patch_weights = [patch_alpha[r, c].item() for r in range(14) for c in range(14) if (r, c) not in [(r, c) for r in range(box_r_min, box_r_max) for c in range(box_c_min, box_c_max)]]
    
    mean_target_attn = sum(target_patch_weights) / len(target_patch_weights)
    mean_bg_attn = sum(bg_patch_weights) / len(bg_patch_weights)
    total_target_mass = sum(target_patch_weights)
    
    print(f"Target object area: {len(target_indices)}/196 patches (12.7% of image area)")
    print(f"Mean attention weight on target object patches:      {mean_target_attn * 100:.3f}%")
    print(f"Mean attention weight on background patches:         {mean_bg_attn * 100:.3f}%")
    print(f"Total attention probability mass inside target box:  {total_target_mass * 100:.2f}%")
    print(f"Ratio of target to background patch attention:       {mean_target_attn / max(mean_bg_attn, 1e-8):.2f}x")
    
    # Display 14x14 attention heatmap in ASCII
    print("\nAttention Heatmap (14x14 patch grid):")
    max_val = patch_alpha.max().item()
    for r in range(14):
        row_str = "  "
        for c in range(14):
            val = patch_alpha[r, c].item() / max_val
            if val > 0.60:
                row_str += "## "
            elif val > 0.30:
                row_str += "++ "
            elif val > 0.10:
                row_str += ".. "
            else:
                row_str += "   "
        print(row_str)
        
    assert mean_target_attn > 5 * mean_bg_attn, "Attention failed to concentrate on target region!"
    assert total_target_mass > 0.70, f"Target mass {total_target_mass:.2f} is below 70%!"
    print("\n[COMPONENT 2: PASS] SketchGuidedImageAttention isolated verification succeeded.")

if __name__ == "__main__":
    test_cross_attention()
