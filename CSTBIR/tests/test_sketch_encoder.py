import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Add CSTBIR to path
sys.path.insert(0, os.path.abspath("CSTBIR"))
from models.sketch_encoder import SketchEncoder

def test_sketch_encoder():
    print("=" * 70)
    print("COMPONENT 1: Sketch Encoder Verification")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Instantiate model
    print("\n--- Step 1: Instantiating SketchEncoder (258 classes, embed_dim=768) ---")
    model = SketchEncoder(num_classes=258, pretrained=False).to(device)
    total_params = sum(p.num_grad_exec() if hasattr(p, 'num_grad_exec') else p.numel() for p in model.parameters())
    print(f"SketchEncoder initialized successfully.")
    print(f"Total parameters: {total_params / 1e6:.2f}M | embed_dim: {model.embed_dim}")
    
    # 2. Test forward pass shapes
    print("\n--- Step 2: Testing forward pass and output shapes ---")
    B = 8
    dummy_sketches = torch.randn(B, 3, 224, 224, device=device)
    logits, h_cls = model(dummy_sketches, return_features=True)
    
    print(f"Input sketch shape:   {dummy_sketches.shape}")
    print(f"Output h_cls shape:   {h_cls.shape} (Expected: ({B}, 768))")
    print(f"Output logits shape:  {logits.shape} (Expected: ({B}, 258))")
    
    assert h_cls.shape == (B, 768), f"Expected ({B}, 768), got {h_cls.shape}"
    assert logits.shape == (B, 258), f"Expected ({B}, 258), got {logits.shape}"
    print("Output shapes verified successfully.")
    
    # 3. Test loss magnitude at initialization
    print("\n--- Step 3: Verifying loss magnitude at initialization ---")
    criterion = nn.CrossEntropyLoss()
    target_labels = torch.randint(0, 258, (B,), device=device)
    initial_loss = criterion(logits, target_labels).item()
    expected_loss = torch.log(torch.tensor(258.0)).item() # ~5.55
    print(f"Initial CrossEntropyLoss: {initial_loss:.4f} (Theoretical log(258) ~ {expected_loss:.4f})")
    assert 4.5 < initial_loss < 7.0, f"Initial loss {initial_loss} outside sane range [4.5, 7.0]"
    
    # 4. Test isolated classification adaptation (overfitting sanity check)
    print("\n--- Step 4: Testing adaptation on QuickDraw 258-class task (isolated optimization) ---")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    initial_acc = (logits.argmax(dim=-1) == target_labels).float().mean().item()
    print(f"Initial accuracy: {initial_acc * 100:.1f}%")
    
    losses = []
    for step in range(25):
        optimizer.zero_grad()
        l_out, _ = model(dummy_sketches, return_features=True)
        loss = criterion(l_out, target_labels)
        loss.backward()
        
        # Verify gradient flow
        if step == 0:
            classifier_grad = model.classifier.weight.grad
            backbone_grad = list(model.vit.parameters())[-1].grad
            assert classifier_grad is not None and classifier_grad.abs().sum() > 0, "Classifier head received no gradients!"
            assert backbone_grad is not None and backbone_grad.abs().sum() > 0, "ViT backbone received no gradients!"
            print("Gradient flow verified to both classifier head and ViT backbone.")
            
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 5 == 0:
            acc = (l_out.argmax(dim=-1) == target_labels).float().mean().item()
            print(f"  Step {step + 1:2d} | Loss: {loss.item():.4f} | Accuracy: {acc * 100:.1f}%")
            
    final_loss = losses[-1]
    final_acc = (l_out.argmax(dim=-1) == target_labels).float().mean().item()
    print(f"Final Step 25 | Loss: {final_loss:.4f} | Accuracy: {final_acc * 100:.1f}%")
    
    assert final_loss < 0.1, f"Loss failed to drop below 0.1, final loss: {final_loss}"
    assert final_acc == 1.0, f"Accuracy failed to reach 100% on sample batch, final acc: {final_acc}"
    print("\n[COMPONENT 1: PASS] SketchEncoder isolated verification succeeded.")

if __name__ == "__main__":
    test_sketch_encoder()
