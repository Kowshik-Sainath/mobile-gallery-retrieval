import sys
import os
import math
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath("CSTBIR"))
from models.classification_heads import ObjectClassificationHeads, ObjectClassificationLoss

def test_classification_heads():
    print("=" * 70)
    print("COMPONENT 5: Object Classification Heads (L_CLS^T & L_CLS^I) Verification")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Instantiate module
    print("\n--- Step 1: Instantiating ObjectClassificationHeads (text_dim=512, img_dim=768, num_classes=258) ---")
    heads = ObjectClassificationHeads(text_dim=512, img_dim=768, num_classes=258).to(device)
    total_params = sum(p.numel() for p in heads.parameters())
    print(f"Heads initialized. Total parameters: {total_params / 1e3:.2f}K")
    
    # 2. Shape verification
    print("\n--- Step 2: Testing Forward Pass and Output Shapes ---")
    B = 8
    dummy_h_T = torch.randn(B, 512, device=device, requires_grad=True)
    dummy_h_I = torch.randn(B, 768, device=device, requires_grad=True)
    
    logits_txt, logits_img = heads(dummy_h_T, dummy_h_I)
    
    print(f"Input text embed shape:  {dummy_h_T.shape} (Expected: ({B}, 512))")
    print(f"Input image embed shape: {dummy_h_I.shape} (Expected: ({B}, 768))")
    print(f"Output logits_txt shape: {logits_txt.shape} (Expected: ({B}, 258))")
    print(f"Output logits_img shape: {logits_img.shape} (Expected: ({B}, 258))")
    
    assert logits_txt.shape == (B, 258), f"Expected ({B}, 258), got {logits_txt.shape}"
    assert logits_img.shape == (B, 258), f"Expected ({B}, 258), got {logits_img.shape}"
    print("Output shapes verified successfully.")
    
    # 3. Loss formulation and gradient flow
    print("\n--- Step 3: Verifying Loss Formulation (L_CLS = L_CLS^T + L_CLS^I) & Gradient Flow ---")
    target_labels = torch.randint(0, 258, (B,), device=device)
    loss_fn = ObjectClassificationLoss()
    
    total_loss, loss_txt, loss_img = loss_fn(logits_txt, logits_img, target_labels)
    expected_random_ce = math.log(258.0) # ~5.553
    
    print(f"L_CLS^T (text):  {loss_txt.item():.4f} (Expected random ~{expected_random_ce:.4f})")
    print(f"L_CLS^I (image): {loss_img.item():.4f} (Expected random ~{expected_random_ce:.4f})")
    print(f"Total L_CLS:     {total_loss.item():.4f} (Expected random ~{2*expected_random_ce:.4f})")
    
    assert 4.5 < loss_txt.item() < 7.0, f"loss_txt {loss_txt.item()} outside reasonable random initialization range"
    assert 4.5 < loss_img.item() < 7.0, f"loss_img {loss_img.item()} outside reasonable random initialization range"
    assert not torch.isnan(total_loss), "Loss computed NaN!"
    
    total_loss.backward()
    assert dummy_h_T.grad is not None and dummy_h_T.grad.abs().sum() > 0, "No gradients received by h_T_CLS!"
    assert dummy_h_I.grad is not None and dummy_h_I.grad.abs().sum() > 0, "No gradients received by h_I_AVG!"
    print(f"Gradient propagated to h_T_CLS (norm: {dummy_h_T.grad.norm().item():.4f})")
    print(f"Gradient propagated to h_I_AVG (norm: {dummy_h_I.grad.norm().item():.4f})")
    print("Gradient flow to both text and image encoders verified.")
    
    # 4. Isolated Overfitting Sanity Check
    print("\n--- Step 4: Testing Isolated Optimization on Sample Batch ---")
    optimizer = optim.AdamW(heads.parameters(), lr=1e-2)
    heads.train()
    
    for step in range(25):
        optimizer.zero_grad()
        l_txt, l_img = heads(dummy_h_T, dummy_h_I)
        loss, _, _ = loss_fn(l_txt, l_img, target_labels)
        loss.backward()
        optimizer.step()
        if (step + 1) % 5 == 0:
            acc_t = (l_txt.argmax(dim=-1) == target_labels).float().mean().item()
            acc_i = (l_img.argmax(dim=-1) == target_labels).float().mean().item()
            print(f"  Step {step + 1:2d} | L_CLS: {loss.item():.4f} | Text Acc: {acc_t * 100:.1f}% | Image Acc: {acc_i * 100:.1f}%")
            
    final_loss, final_loss_txt, final_loss_img = loss_fn(l_txt, l_img, target_labels)
    acc_t = (l_txt.argmax(dim=-1) == target_labels).float().mean().item()
    acc_i = (l_img.argmax(dim=-1) == target_labels).float().mean().item()
    
    print(f"Final Step 25 | Total L_CLS: {final_loss.item():.4f} | Text Acc: {acc_t * 100:.1f}% | Image Acc: {acc_i * 100:.1f}%")
    assert final_loss.item() < 0.10, f"L_CLS failed to drop below 0.10, got {final_loss.item()}"
    assert acc_t == 1.0 and acc_i == 1.0, "Both heads must achieve 100% accuracy on sample batch"
    
    print("\n[COMPONENT 5: PASS] ObjectClassificationHeads isolated verification succeeded.")

if __name__ == "__main__":
    test_classification_heads()
