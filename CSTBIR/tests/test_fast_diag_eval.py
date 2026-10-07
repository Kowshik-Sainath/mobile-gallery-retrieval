import os
import json
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet
from CSTBIR.clip import clip
from CSTBIR.evaluate_retrieval import compute_retrieval_metrics

def test_diag():
    device = 'cuda'
    model = STNet(num_classes=258, device=device, pretrained_sketch=False).to(device)
    ckpt = torch.load('CSTBIR/checkpoints/lambda_1.0/stnet_best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_data = json.load(open('CSTBIR/data/CSTBIR_dataset.json', 'r', encoding='utf-8'))
    val_queries = [x for x in all_data if x.get('split') == 'val']
    diag_queries = val_queries[:200]
    val_gallery_names = sorted(list(set(x['image'] for x in val_queries)))
    img_to_idx = {name: i for i, name in enumerate(val_gallery_names)}
    gt_indices = [img_to_idx[q['image']] for q in diag_queries]

    clip_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    sketch_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images_dir = 'CSTBIR/data/vg_images'
    sketches_dir = 'CSTBIR/data/quickdraw_sketches'

    t0 = time.time()
    # 1. Gallery tokens
    gallery_tokens = []
    for i in range(0, len(val_gallery_names), 64):
        batch_imgs = [clip_preprocess(Image.open(os.path.join(images_dir, n)).convert('RGB')) for n in val_gallery_names[i:i+64]]
        img_tensors = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            gallery_tokens.append(model.extract_image_tokens(img_tensors))
    H_gallery = torch.cat(gallery_tokens, dim=0) # (1000, 197, 768)
    t_gal = time.time() - t0

    # 2. Query tokens
    t0 = time.time()
    query_text_tokens = clip.tokenize([q['text'] for q in diag_queries], truncate=True).to(device)
    with torch.no_grad():
        h_T = model.encode_text(query_text_tokens)
        z_t = F.normalize(model.loss_ct_fn.txt_proj(h_T), p=2, dim=-1)

    sketch_imgs = [sketch_preprocess(Image.open(os.path.join(sketches_dir, q['sketch'])).convert('RGB')) for q in diag_queries]
    sketch_tensors = torch.stack(sketch_imgs).to(device)
    with torch.no_grad():
        h_S = model.encode_sketch(sketch_tensors)
    t_q = time.time() - t0

    # 3. Cross-attention & rank
    t0 = time.time()
    ranks = []
    K = len(val_gallery_names)
    for i in range(len(diag_queries)):
        sq = h_S[i:i+1].repeat(K, 1) # (1000, 768)
        with torch.no_grad():
            h_I_avg, _, _ = model.cross_attention(H_gallery, sq)
            z_i = F.normalize(model.loss_ct_fn.img_proj(h_I_avg), p=2, dim=-1)
            sims = (z_t[i:i+1] @ z_i.T).squeeze(0).cpu().numpy()
        sorted_idx = np.argsort(-sims)
        rank = int(np.where(sorted_idx == gt_indices[i])[0][0]) + 1
        ranks.append(rank)
    t_rank = time.time() - t0

    metrics = compute_retrieval_metrics(ranks)
    print(f"Timing: Gallery={t_gal:.1f}s | Query={t_q:.1f}s | Ranking={t_rank:.1f}s | Total={t_gal+t_q+t_rank:.1f}s")
    print(f"Diagnostic 200-sample Results: R@10={metrics['R@10']:.2f}%, R@20={metrics['R@20']:.2f}%, MdR={metrics['MdR']:.1f}")

if __name__ == '__main__':
    test_diag()
