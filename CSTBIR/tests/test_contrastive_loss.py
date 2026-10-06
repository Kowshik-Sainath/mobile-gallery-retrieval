import sys
import os
import math
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath("CSTBIR"))
from models.contrastive_loss import ContrastiveRetrievalLoss

def test_contrastive_loss():
    print("=" * 70)
    print("COMPONENT 6: Cross-Modal Contrastive Loss (L_CT) Verification")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Instantiate module
    print("\n--- Step 1: Instantiating ContrastiveRetrievalLoss (text_dim=512, img_dim=768, proj_dim=512) ---")
    criterion = ContrastiveRetrievalLoss(text_dim=512, img_dim=768, proj_dim=512).to(device)
    print("Module initialized successfully.")
    
    # 2. Shape and Loss Magnitude Check Across Multiple Batch Sizes
    print("\n--- Step 2: Testing Forward Pass and Random Init Loss Magnitudes ---")
    batch_sizes = [4, 8, 16, 32]
    for B in batch_sizes:
        h_T = torch.randn(B, 512, device=device)
        h_I = torch.randn(B, 768, device=device)
        
        loss, logits_t2i, logits_i2t = criterion(h_T, h_I)
        expected_loss = math.log(B)
        
        print(f"Batch B={B:2d} | Output logits: {logits_t2i.shape} | L_CT: {loss.item():.4f} (Theoretical log({B}) ~ {expected_loss:.4f})")
        
        assert logits_t2i.shape == (B, B), f"Expected ({B}, {B}), got {logits_t2i.shape}"
        assert logits_i2t.shape == (B, B), f"Expected ({B}, {B}), got {logits_i2t.shape}"
        assert not torch.isnan(loss), f"Loss computed NaN for B={B}!"
        assert loss.item() > 0.0, f"Loss is negative for B={B}!"
        # Sanity check: at random init, loss should be within reasonable margin of log(B)
        assert abs(loss.item() - expected_loss) < 1.0, f"Loss {loss.item()} diverged from log({B}) = {expected_loss}"
        
    print("Loss magnitudes verified across all batch sizes.")
    
    # 3. Gradient Flow Check
    print("\n--- Step 3: Checking Gradient Flow to Both Modalities & Projections ---")
    B = 8
    h_T = torch.randn(B, 512, device=device, requires_grad=True)
    h_I = torch.randn(B, 768, device=device, requires_grad=True)
    
    loss, _, _ = criterion(h_T, h_I)
    loss.backward()
    
    assert h_T.grad is not None and h_T.grad.abs().sum() > 0, "No gradients received by h_T_CLS!"
    assert h_I.grad is not None and h_I.grad.abs().sum() > 0, "No gradients received by h_I_AVG!"
    assert criterion.img_proj.weight.grad is not None, "No gradients received by img_proj!"
    assert criterion.logit_scale.grad is not None, "No gradients received by logit_scale!"
    
    print(f"h_T_CLS grad norm:      {h_T.grad.norm().item():.4f}")
    print(f"h_I_AVG grad norm:      {h_I.grad.norm().item():.4f}")
    print(f"img_proj grad norm:     {criterion.img_proj.weight.grad.norm().item():.4f}")
    print(f"logit_scale grad:       {criterion.logit_scale.grad.item():.4f}")
    print("Gradient flow verified to both encoders, projection layer, and logit scale.")
    
    # 4. Isolated Overfitting / Retrieval Alignment Sanity Check
    print("\n--- Step 4: Testing Contrastive Retrieval Optimization on Sample Batch ---")
    optimizer = optim.AdamW(criterion.parameters(), lr=1e-2)
    criterion.train()
    
    for step in range(25):
        optimizer.zero_grad()
        loss, logits_t2i, _ = criterion(h_T, h_I)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            gt = torch.arange(B, device=device)
            acc = (logits_t2i.argmax(dim=-1) == gt).float().mean().item()
            print(f"  Step {step + 1:2d} | L_CT: {loss.item():.4f} | Diagonal Retrieval Acc: {acc * 100:.1f}%")
            
    final_loss, final_logits, _ = criterion(h_T, h_I)
    gt = torch.arange(B, device=device)
    final_acc = (final_logits.argmax(dim=-1) == gt).float().mean().item()
    
    print(f"Final Step 25 | L_CT: {final_loss.item():.4f} | Diagonal Retrieval Acc: {final_acc * 100:.1f}%")
    assert final_acc == 1.0, "Retrieval accuracy failed to reach 100% on sample batch!"
    assert final_loss.item() < 0.05, f"L_CT failed to converge, final loss: {final_loss.item()}"
    
    print("\n[COMPONENT 6: PASS] ContrastiveRetrievalLoss isolated verification succeeded.")

if __name__ == "__main__":
    test_contrastive_loss()
