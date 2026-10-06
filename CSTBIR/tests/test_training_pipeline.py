import sys
import os
import torch

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet
from CSTBIR.data.dataloader import CSTBIRDataset

def test_training_pipeline():
    print("=" * 70)
    print("STEP 3 & 4: End-to-End Training Pipeline Sanity Check (3 Iterations)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = STNet(num_classes=258, device=device, pretrained_sketch=False)
    ds = CSTBIRDataset(json_path="CSTBIR/data/CSTBIR_dataset.json", split="val") # small split for quick test
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    
    print("\nRunning 3 training steps...")
    for step in range(3):
        batch = ds.get_conflict_free_batch(batch_size=4)
        
        outputs = model(
            text_tokens=batch['text'].to(device),
            images=batch['image'].to(device),
            sketches=batch['sketch_img'].to(device) if batch['sketch_img'] is not None else None,
            sketch_embeds=batch['sketch_embed'].to(device) if batch['sketch_embed'] is not None else None,
            gt_boxes=batch['bbox'].to(device),
            gt_labels=batch['label'].to(device),
            target_sketch_imgs=batch['sketch_img'].to(device) if batch['sketch_img'] is not None else None
        )
        
        optimizer.zero_grad()
        loss = outputs['loss_total']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"Step {step+1}: L_total={outputs['loss_total'].item():.3f} | L_CT={outputs['loss_ct'].item():.3f} | L_CLS_T={outputs['loss_cls_t'].item():.3f} | L_CLS_I={outputs['loss_cls_i'].item():.3f} | L_OD={outputs['loss_od'].item():.3f} | L_SR={outputs['loss_sr'].item():.3f}")
        
    print("\n[END-TO-END PIPELINE: PASS] Training loop executed 3 steps with all 5 losses cleanly logged.")

if __name__ == "__main__":
    test_training_pipeline()
