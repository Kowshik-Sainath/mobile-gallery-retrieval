# Android On-Device Deployment Gap Analysis & Resolution

This document tracks technical gaps between the research/training Python prototype and a production-grade, offline-only Android deployment, along with their resolution status.

---

## 1. Tokenizer Portability (Status: RESOLVED)

### Prior Gap:
- Research implementation depended on Apple's `ml-mobileclip` and `open_clip` Python packages, which rely on Python C-extensions (`regex`, `ftfy`, `gzip`).
- Embedding a Python runtime or chaquopy on Android introduces ~25 MB bloat, battery drain, and native crash risk.

### Resolution:
- MobileCLIP-S1 text encoder utilizes the standard **OpenAI CLIP BPE** specification:
  - **Vocabulary Size**: 49,408 tokens.
  - **Context Length**: 77 tokens (`int64[1, 77]`).
  - **Start of Text (SOT)**: `49406` (`<start_of_text>`).
  - **End of Text (EOT)**: `49407` (`<end_of_text>`).
  - **Padding**: `0`.
- An Android-native, pure-Kotlin BPE tokenizer is implemented at [`docs/android/ClipTokenizer.kt`](file:///c:/Users/reddy/OneDrive%20-%20Amrita%20university/CV/Project/docs/android/ClipTokenizer.kt).
- **Parity Verification**: Verified in [`tests/test_tokenizer_parity.py`](file:///c:/Users/reddy/OneDrive%20-%20Amrita%20university/CV/Project/tests/test_tokenizer_parity.py):
  - **50 / 50 real FS-COCO captions** achieved **100% token-for-token match** (3,850 / 3,850 tokens identical).
  - Truncation, punctuation, alphanumeric edge cases verified identical.
  - Zero Python, C++, or NDK runtime dependencies on mobile.

---

## 2. Backbone Deduplication & Storage Footprint (Status: RESOLVED)

### Prior Gap:
- Naively exporting two complete INT8 FastViT vision models (`photo_encoder_int8.onnx` at 28.35 MB and `sketch_encoder_int8.onnx` at 28.36 MB) wasted ~28.35 MB of flash storage on 96.2% redundant weights.

### Resolution:
- Exported a single shared base vision graph: `photo_backbone_int8.onnx` (28.35 MB).
- Extracted low-rank LoRA adapter delta tensors into compact `sketch_lora_delta.npz` (3.01 MB across 90 modules / 827,392 float32 parameters).
- Implemented in-memory base-plus-delta loading in [`mobile_gallery_demo.py`](file:///c:/Users/reddy/OneDrive%20-%20Amrita%20university/CV/Project/mobile_gallery_demo.py):
  - In-memory delta application achieves **1.00000 cosine similarity** vs PyTorch `encode_sketch()`.
  - Saved 25.34 MB of redundant vision weights.

### Plain Storage Accounting:
| Exported Component | Format | Disk Size | Target Constraint | Status |
| :--- | :--- | :--- | :--- | :--- |
| `photo_backbone_int8.onnx` | INT8 ONNX | 28.35 MB | $<45$ MB per vision model | PASS |
| `sketch_lora_delta.npz` | Compressed NPZ | 3.01 MB | Low-rank adapter delta | PASS |
| `text_encoder_int8.onnx` | INT8 ONNX | 61.39 MB | Transformer text encoder | NOTE (see below) |
| `composite_fusion_mobile.onnx` | FP32 ONNX | 3.00 MB | Gated query fuser | PASS |
| `combiner_mobile.onnx` | FP32 ONNX | 3.01 MB | Feedback reranker | PASS |
| **Total Model Suite** | — | **98.76 MB** | $<45$ MB total budget | **EXCEEDS BUDGET** |

> [!NOTE]
> The deduplicated vision pipeline saved 28.35 MB, reducing vision storage from 56.7 MB to 31.36 MB. However, the total model bundle is **98.76 MB**, which exceeds the strict $<45$ MB model target by 53.76 MB. This is dominated by the INT8 text transformer (`text_encoder_int8.onnx` at 61.39 MB). For future work, knowledge distillation to a 4-layer MiniLM-CLIP text encoder (~15 MB) will bring the full suite under 45 MB.

---

## 3. Gallery Indexing & Reconciliation Lifecycle (Status: RESOLVED)

### Prior Gap:
- Inefficient single-image inference loops or re-encoding the entire gallery on every app startup caused extreme battery drain and poor UX.

### Resolution:
- Mirrored iOS `CLIP-Finder2` lifecycle pattern on Android target stack:
  1. **First-launch batching**: amortizes ONNX session warmup by batching 16–32 photos.
  2. **Launch-time reconciliation**: queries Android `MediaStore` and performs an SQL set-diff against `photo_embeddings.last_modified`. Unchanged photos are short-circuited with **0 embedding calls**.
  3. **Deletion pruning**: photos removed by the user in the system Gallery app are immediately detected and pruned from the SQLite database.
- Verified in [`tests/test_indexing_lifecycle.py`](file:///c:/Users/reddy/OneDrive%20-%20Amrita%20university/CV/Project/tests/test_indexing_lifecycle.py):
  - Relaunch on 10 unchanged photos: **0 embedding calls** (PASS).
  - Incremental modification/addition: embeds only modified photos (PASS).
  - Deletion of photos: database pruned immediately (PASS).

---

## 4. Preprocessing & Hardware Acceleration (Status: BENCHMARKED)

### Gap:
- Image resizing and normalization can bottleneck mobile NPU pipelines if performed inefficiently on CPU.

### Resolution:
- Benchmarking (documented in `benchmarks/benchmark_preprocessing.py`) quantifies CPU resize vs ONNX inference latency across batch sizes 1, 8, 16, 32.
- Android architectural specification adopts `android.graphics.HardwareRenderer` / `RenderEffect` hardware pipeline to eliminate CPU bitmap conversions.
