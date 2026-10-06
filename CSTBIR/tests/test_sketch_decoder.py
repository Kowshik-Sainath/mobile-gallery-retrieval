import sys
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.abspath("CSTBIR"))
from models.sketch_decoder import SketchReconstructionDecoder, SketchReconstructionLoss

def draw_synthetic_sketch(pattern_type: str, size: int = 224) -> torch.Tensor:
    """Creates a 1x224x224 binary sketch image with simple strokes."""
    sketch = torch.zeros(1, 1, size, size)
    if pattern_type == "circle":
        # Draw circle strokes
        y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
        dist = torch.sqrt(x**2 + y**2)
        sketch[0, 0, (dist > 0.45) & (dist < 0.55)] = 1.0
    elif pattern_type == "cross":
        # Draw cross strokes (thickness = 10 px)
        mid = size // 2
        sketch[0, 0, mid-5:mid+5, 40:size-40] = 1.0
        sketch[0, 0, 40:size-40, mid-5:mid+5] = 1.0
    return sketch

def test_sketch_decoder():
    print("=" * 70)
    print("COMPONENT 4: Sketch Reconstruction Decoder (L_SR) Verification")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Instantiate decoder
    print("\n--- Step 1: Instantiating SketchReconstructionDecoder (8 Blocks) ---")
    decoder = SketchReconstructionDecoder(in_dim=768).to(device)
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder initialized successfully. Total parameters: {total_params / 1e6:.2f}M")
    
    # 2. Shape verification
    print("\n--- Step 2: Testing Forward Pass and Output Shape ---")
    B = 2
    dummy_H_I = torch.randn(B, 197, 768, device=device, requires_grad=True)
    out_sketches = decoder(dummy_H_I)
    
    print(f"Input H_I shape:             {dummy_H_I.shape}")
    print(f"Output sketch tensor shape:  {out_sketches.shape} (Expected: ({B}, 1, 224, 224))")
    print(f"Output range: min={out_sketches.min().item():.4f}, max={out_sketches.max().item():.4f} (Sigmoid bounded in [0, 1])")
    
    assert out_sketches.shape == (B, 1, 224, 224), f"Expected ({B}, 1, 224, 224), got {out_sketches.shape}"
    assert 0.0 <= out_sketches.min().item() and out_sketches.max().item() <= 1.0, "Values not in [0, 1]"
    print("Output shapes and bounds verified.")
    
    # 3. Loss computation and gradient flow
    print("\n--- Step 3: Verifying Loss Formulation (BCE + DICE) & Gradient Flow ---")
    target_sketches = torch.cat([
        draw_synthetic_sketch("circle", 224),
        draw_synthetic_sketch("cross", 224)
    ], dim=0).to(device) # (2, 1, 224, 224)
    
    criterion = SketchReconstructionLoss(alpha=1.0, beta=1.0)
    initial_loss = criterion(out_sketches, target_sketches)
    print(f"Initial L_SR loss: {initial_loss.item():.4f}")
    assert not torch.isnan(initial_loss), "Loss computed NaN!"
    
    initial_loss.backward()
    assert dummy_H_I.grad is not None and dummy_H_I.grad.abs().sum() > 0, "No gradient propagated to H_I!"
    block1_grad = list(decoder.block1.parameters())[0].grad
    assert block1_grad is not None and block1_grad.abs().sum() > 0, "Block 1 received no gradients!"
    print(f"Gradient propagated to H_I (norm: {dummy_H_I.grad.norm().item():.4f})")
    print("Gradient flow through all 8 decoder blocks verified.")
    
    # 4. Isolated Overfitting / Reconstruction Sanity Check
    print("\n--- Step 4: Testing Reconstruction Overfitting Sanity Check ---")
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    decoder.train()
    
    losses = []
    for step in range(60):
        optimizer.zero_grad()
        preds = decoder(dummy_H_I)
        loss = criterion(preds, target_sketches)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 15 == 0:
            print(f"  Step {step + 1:2d} | L_SR: {loss.item():.4f}")
            
    final_loss = losses[-1]
    print(f"Final Step 60 | L_SR: {final_loss:.4f} (Drop: {losses[0]:.4f} -> {final_loss:.4f})")
    assert final_loss < 0.15, f"Loss failed to drop below 0.15, final loss: {final_loss}"
    
    # Evaluate reconstruction overlap
    with torch.no_grad():
        final_preds = decoder(dummy_H_I)
        # Compute Dice score on reconstructed samples
        for b, name in [(0, "circle"), (1, "cross")]:
            pred_bin = (final_preds[b, 0] > 0.5).float()
            gt_bin = target_sketches[b, 0]
            intersection = (pred_bin * gt_bin).sum().item()
            dice = (2.0 * intersection) / (pred_bin.sum().item() + gt_bin.sum().item() + 1e-6)
            print(f"Sample {b} ({name}) Reconstruction Dice Score: {dice * 100:.2f}%")
            assert dice > 0.80, f"Sample {b} ({name}) Dice score {dice:.2f} is below 80%!"
            
            # Print 14x14 downsampled ASCII view
            down_gt = F.interpolate(gt_bin.unsqueeze(0).unsqueeze(0), size=(14, 14), mode='area')[0, 0]
            down_pred = F.interpolate(final_preds[b:b+1], size=(14, 14), mode='area')[0, 0]
            print(f"\nTarget vs Reconstructed ({name}) [14x14 ASCII preview]:")
            print("  Target Sketch:                     Reconstructed Sketch:")
            for r in range(14):
                tgt_row = "".join(["##" if down_gt[r, c] > 0.15 else "  " for c in range(14)])
                pred_row = "".join(["##" if down_pred[r, c] > 0.15 else "  " for c in range(14)])
                print(f"  {tgt_row}        {pred_row}")
                
    print("\n[COMPONENT 4: PASS] SketchReconstructionDecoder isolated verification succeeded.")

if __name__ == "__main__":
    test_sketch_decoder()
