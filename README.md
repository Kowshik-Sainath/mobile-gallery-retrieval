# Text + Sketch Based Image Retrieval (T+SBIR) for Edge AI (Mobile Photo Gallery)

A working research prototype of an **on-device Text + Sketch Based Image Retrieval System** designed for mobile photo galleries (target phones with 4–8GB RAM). 

This repository integrates and adapts official implementations from key published papers into a single unified edge pipeline.

---

## 1. Official Repository Lineage & Citation Table

To maintain strict academic citation integrity, the table below maps each system component to its originating public GitHub repository versus original integration glue code:

| Component | Source Repository | Citation / Paper | Usage & Adaptations |
| :--- | :--- | :--- | :--- |
| **STNet Attention Pooling & Decoder** | [`vl2g/CSTBIR`](https://github.com/vl2g/CSTBIR) | AAAI 2024 ("Composite Sketch+Text Queries...") | Ported `SketchGuidedAttentionPooling` (sketch query over image patch tokens) and `SketchReconstructionDecoder` (auxiliary training loss). |
| **FS-COCO Dataset Loader** | [`pinakinathc/fscoco`](https://github.com/pinakinathc/fscoco) | ECCV 2022 ("FS-COCO...") | Adapted triplet loader (`sketch`, `caption`, `photo`) for standard 9,525 train / 475 test split parsing. |
| **Distilled Vision/Text Backbone** | [`apple/ml-mobileclip`](https://github.com/apple/ml-mobileclip) | MobileCLIP (Apple Research) | Shared frozen backbone for fast edge inference. Fallback to OpenCLIP ViT-B/32. |
| **Sketch Domain Adapter** | [`huggingface/peft`](https://github.com/huggingface/peft) | Hugging Face PEFT | Attached LoRA (`LoraConfig`, `get_peft_model`) to vision tower target projections. |
| **Feedback Reranker MLP** | [`ABaldrati/CLIP4Cir`](https://github.com/ABaldrati/CLIP4Cir) | CVPR 2022 ("CLIP4Cir...") | Ported Gated MLP `Combiner` module (<1M params) for on-device incremental feedback fine-tuning. |
| **Near-Duplicate Suppression** | [`idealo/imagededup`](https://github.com/idealo/imagededup) | imagededup | Perceptual hashing (pHash) & cosine embedding clustering for duplicate collapse. |
| **Mobile Export & Quantization** | [`google-ai-edge/ai-edge-torch`](https://github.com/google-ai-edge/ai-edge-torch) / ONNX | ONNX Runtime Mobile | Strips training-only decoder heads and exports INT8 quantized `.onnx` models. |
| **Integration Glue & Orchestration** | *Original Code* | This Repository | Data transforms, multi-loss training loop (`train_fscoco.py`), metrics (`evaluate.py`), checkpointing (`checkpoint.py`), benchmarks. |

---

## 2. Environment Setup

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## 3. How to Run the Pipeline

### Phase 1 & 2 — Training on FS-COCO
To train the LoRA adapters + STNet attention pooling & reconstruction decoder:

```bash
python train_fscoco.py --data_dir fscoco --batch_size 16 --epochs 5
```

### Phase 4 — Recall@K Evaluation
To evaluate Recall@1, Recall@5, Recall@10, and NDCG@10 across Text-Only, Sketch-Only, and Composite (Sketch+Text) queries:

```bash
python evaluate.py --data_dir fscoco
```

### Phase 5 — Near-Duplicate Detection Validation
To validate on-device near-duplicate suppression precision/recall:

```bash
python evaluate_dedup.py
```

### Phase 6 — Feedback Reranker Latency Benchmark
To benchmark CPU fine-tuning latency per update step for the Gated MLP Combiner:

```bash
python benchmark_reranker.py
```

### Phase 7 — Mobile Export (ONNX / TFLite)
To quantize and export inference models for edge deployment:

```bash
python export_mobile.py
```
