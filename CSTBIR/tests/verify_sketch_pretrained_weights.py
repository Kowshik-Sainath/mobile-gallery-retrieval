import os
import sys
import torch

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.sketch_encoder import SketchEncoder

def verify_sketch_pretrained_weights():
    print("=" * 80)
    print("STEP 1: VERIFYING SKETCH ENCODER PRETRAINED VS RANDOM INITIALIZATION")
    print("=" * 80)
    
    # Check default torch / timm cache directories
    torch_cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"Torch Hub cache dir: {torch_cache_dir}")
    print(f"HF Hub cache dir:    {hf_cache_dir}")
    
    # 1. Instantiate pretrained=False (random init)
    print("\n--- Instantiating SketchEncoder(pretrained=False) ---")
    encoder_random = SketchEncoder(num_classes=258, pretrained=False)
    sum_random = sum(p.abs().sum().item() for p in encoder_random.vit.parameters())
    num_params_random = sum(p.numel() for p in encoder_random.vit.parameters())
    print(f"Random ViT parameter count: {num_params_random:,}")
    print(f"Random ViT weight abs-sum:  {sum_random:.4f}")
    
    # 2. Instantiate pretrained=True (real download / cached weights)
    print("\n--- Instantiating SketchEncoder(pretrained=True) ---")
    encoder_pretrained = SketchEncoder(num_classes=258, pretrained=True)
    sum_pretrained = sum(p.abs().sum().item() for p in encoder_pretrained.vit.parameters())
    num_params_pretrained = sum(p.numel() for p in encoder_pretrained.vit.parameters())
    print(f"Pretrained ViT parameter count: {num_params_pretrained:,}")
    print(f"Pretrained ViT weight abs-sum:  {sum_pretrained:.4f}")
    
    # 3. Compute L2 distance between patch embedding projection weights
    w_rand = encoder_random.vit.patch_embed.proj.weight
    w_pre = encoder_pretrained.vit.patch_embed.proj.weight
    l2_dist = torch.dist(w_rand, w_pre, p=2).item()
    print(f"\nPatch embedding weight shape: {w_pre.shape}")
    print(f"L2 distance (random vs pretrained patch_embed.proj.weight): {l2_dist:.6f}")
    
    # 4. Check timm / torch cache files for the downloaded weights
    print("\n--- Checking Cache File Locations ---")
    found_cache = []
    for root_dir in [torch_cache_dir, hf_cache_dir]:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if any(k in f.lower() for k in ["miil", "vit_base_patch16", "in21k", "224"]):
                        fpath = os.path.join(root, f)
                        fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
                        found_cache.append((fpath, fsize_mb))
                        print(f"  Found cached weight: {fpath} ({fsize_mb:.2f} MB)")
                        
    if not found_cache:
        # Check all files in torch cache
        if os.path.exists(torch_cache_dir):
            for f in os.listdir(torch_cache_dir):
                fpath = os.path.join(torch_cache_dir, f)
                fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  Torch cache file: {f} ({fsize_mb:.2f} MB)")
                
    # Gate 1 assertions
    assert l2_dist > 0.0, f"Gate 1 Failed: L2 distance is {l2_dist}, weights are identical!"
    assert abs(sum_random - sum_pretrained) > 100.0, f"Gate 1 Failed: weight sums too close ({sum_random} vs {sum_pretrained})!"
    print("\n[GATE 1 VERIFICATION RESULT]")
    print(f"  Random Init Weight Sum:     {sum_random:.4f}")
    print(f"  Pretrained Weight Sum:      {sum_pretrained:.4f}")
    print(f"  L2 Distance:                {l2_dist:.6f}")
    print(f"  Gate 1 Nonzero L2 Verified: True")
    print("=" * 80)

if __name__ == "__main__":
    verify_sketch_pretrained_weights()
