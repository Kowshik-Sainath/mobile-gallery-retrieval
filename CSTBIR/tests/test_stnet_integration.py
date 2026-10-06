import sys
import os
import torch

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet

def test_stnet_integration():
    print("=" * 70)
    print("STEP 3: Full STNet Model Integration & 5-Loss Verification")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Instantiate STNet
    print("\n--- Step 1: Instantiating Integrated STNet ---")
    model = STNet(clip_model_name="ViT-B/16", num_classes=258, device=device, pretrained_sketch=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Integrated STNet initialized. Total parameters: {total_params / 1e6:.2f}M")
    
    # 2. Forward pass with complete multimodal batch
    print("\n--- Step 2: Testing Full Forward Pass (All 5 Losses) ---")
    B = 4
    dummy_text_tokens = torch.randint(1, 1000, (B, 77), device=device)
    dummy_images = torch.randn(B, 3, 224, 224, device=device)
    dummy_sketches = torch.randn(B, 3, 224, 224, device=device)
    dummy_gt_boxes = torch.tensor([
        [0.1, 0.1, 0.4, 0.4],
        [0.2, 0.3, 0.7, 0.8],
        [0.5, 0.5, 0.9, 0.9],
        [0.3, 0.1, 0.6, 0.5]
    ], device=device)
    dummy_gt_labels = torch.tensor([10, 45, 120, 210], device=device)
    dummy_target_sketches = torch.zeros(B, 1, 224, 224, device=device)
    dummy_target_sketches[:, :, 50:150, 50:150] = 1.0
    
    outputs = model(
        text_tokens=dummy_text_tokens,
        images=dummy_images,
        sketches=dummy_sketches,
        gt_boxes=dummy_gt_boxes,
        gt_labels=dummy_gt_labels,
        target_sketch_imgs=dummy_target_sketches
    )
    
    loss_total = outputs['loss_total']
    loss_ct = outputs['loss_ct']
    loss_cls_t = outputs['loss_cls_t']
    loss_cls_i = outputs['loss_cls_i']
    loss_od = outputs['loss_od']
    loss_sr = outputs['loss_sr']
    
    print("\nIndividual Loss Values:")
    print(f"  1. L_CT (Contrastive Retrieval):    {loss_ct.item():.4f}")
    print(f"  2. L_CLS^T (Text Classification):   {loss_cls_t.item():.4f}")
    print(f"  3. L_CLS^I (Image Classification):  {loss_cls_i.item():.4f}")
    print(f"  4. L_OD (Object Detection YOLO):    {loss_od.item():.4f}")
    print(f"  5. L_SR (Sketch Reconstruction):    {loss_sr.item():.4f}")
    print(f"  --> L_total (Unweighted Sum):       {loss_total.item():.4f}")
    
    # Assertions on loss behavior
    assert not torch.isnan(loss_total), "Total loss is NaN!"
    assert loss_ct.item() > 0.0, "L_CT must be positive"
    assert loss_cls_t.item() > 0.0, "L_CLS^T must be positive"
    assert loss_cls_i.item() > 0.0, "L_CLS^I must be positive"
    assert loss_od.item() > 0.0, "L_OD must be positive"
    assert loss_sr.item() > 0.0, "L_SR must be positive"
    
    expected_sum = (loss_ct + loss_cls_t + loss_cls_i + loss_od + loss_sr).item()
    assert abs(loss_total.item() - expected_sum) < 1e-4, "L_total is not an exact unweighted sum of all 5 losses!"
    print("Exact unweighted 5-loss summation verified.")
    
    # 3. Backward Pass Verification
    print("\n--- Step 3: Verifying Backward Pass & Gradients ---")
    loss_total.backward()
    
    # Check that each component received gradients
    has_clip_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.clip_model.parameters())
    has_sketch_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.sketch_encoder.parameters())
    has_od_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.od_head.parameters())
    has_decoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.sketch_decoder.parameters())
    has_cls_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cls_heads.parameters())
    
    print(f"CLIP Image/Text encoders received gradients: {has_clip_grad}")
    print(f"Sketch Encoder received gradients:           {has_sketch_grad}")
    print(f"OD Head received gradients:                   {has_od_grad}")
    print(f"Sketch Decoder received gradients:            {has_decoder_grad}")
    print(f"Classification Heads received gradients:      {has_cls_grad}")
    
    assert has_clip_grad and has_sketch_grad and has_od_grad and has_decoder_grad and has_cls_grad, "Missing gradients in one or more components!"
    print("Gradients verified across all components.")
    
    # 4. Gallery Retrieval Evaluation Sanity Check
    print("\n--- Step 4: Testing Gallery Ranking Similarity Matrix ---")
    model.eval()
    Q = 3
    K = 10
    query_texts = torch.randint(1, 1000, (Q, 77), device=device)
    query_sketches = torch.randn(Q, 768, device=device)
    gallery_images = torch.randn(K, 3, 224, 224, device=device)
    
    sims = model.compute_query_image_similarity(query_texts, query_sketches, gallery_images)
    print(f"Similarity matrix shape: {sims.shape} (Expected: ({Q}, {K}))")
    assert sims.shape == (Q, K), f"Expected ({Q}, {K}), got {sims.shape}"
    assert not torch.isnan(sims).any(), "NaN in similarity matrix!"
    
    # Check ranking
    ranks = torch.argsort(sims, dim=-1, descending=True)
    print(f"Rankings per query (top 5 indices for query 0): {ranks[0, :5].tolist()}")
    print("Gallery ranking similarity calculation verified.")
    
    print("\n[STEP 3: PASS] Integrated STNet model and 5-loss pipeline verified.")

if __name__ == "__main__":
    test_stnet_integration()
