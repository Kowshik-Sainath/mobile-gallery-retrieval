# Text + Sketch Based Image Retrieval (T+SBIR) for Edge AI (Mobile Photo Gallery)

A production-grade, privacy-preserving, on-device **Text + Sketch Based Image Retrieval (T+SBIR)** system designed for consumer Android smartphones (tested on 4GB RAM devices running Android 14–16).

This project allows users to search their local mobile photo gallery using **natural language text**, **freehand QuickDraw sketches**, or **multimodal composite queries (text + sketch)**, with **CIRR-style compositional feedback refinement**—all executed 100% on-device without sending personal photos to the cloud.

---

## 1. System Architecture Overview

```
                      +-----------------------------+
                      |   User Search Input (UI)    |
                      +-----------------------------+
                        /                         \
                       / (Text Query)              \ (QuickDraw Sketch)
                      v                             v
           +--------------------+         +--------------------+
           |  Kotlin CLIP BPE   |         | Bézier Path Canvas |
           |  Tokenizer (77 ID) |         |  (224x224 RGB)     |
           +--------------------+         +--------------------+
                      |                             |
                      v                             v
           +--------------------+         +--------------------+
           | text_encoder_int8  |         | sketch_encoder_int8|
           | (MobileCLIP ViT-B) |         | (FastViT + LoRA)   |
           +--------------------+         +--------------------+
                      \                             /
              (512-D)  \                           / (512-D)
                        v                         v
                   +----------------------------------+
                   |   composite_fusion_mobile.onnx   |  < 1 ms latency
                   |   (Late Fusion Gated MLP)        |
                   +----------------------------------+
                                    |
                          Fused Query (512-D)
                                    v
       +--------------------------------------------------------------+
       |   In-Memory Contiguous Gallery Index (N x 512 FloatArray)    |
       |   Batched Inner Product Scan + Min-Heap Top-50 (< 40 ms)     |
       +--------------------------------------------------------------+
                                    |
                            Top-50 Candidates
                                    v
       +--------------------------------------------------------------+
       |   Feedback Refinement (CIRR-Style Compositional Steering)    |
       |   combiner_mobile.onnx(shown_candidate, new_query) -> 2 ms   |
       +--------------------------------------------------------------+
```

### Core Architecture Highlights:
1. **Shared Vision Backbone**: Apple MobileCLIP-S1 (`fastvit_s12`) with static INT8 symmetric quantization (`photo_backbone_int8.onnx`, 28.35 MB).
2. **Sketch Domain Adaptation**: Low-Rank Adaptation (LoRA rank=8, alpha=16) merged into the vision weights (`sketch_encoder_int8.onnx`, 28.35 MB) for instant QuickDraw stroke alignment.
3. **Text Encoder**: INT8 quantized MobileCLIP text encoder (`text_encoder_int8.onnx`, 61.39 MB) with a zero-dependency pure Kotlin BPE Tokenizer (`ClipTokenizer.kt`).
4. **Multimodal Late Fusion**: Lightweight feed-forward MLP (`composite_fusion_mobile.onnx`, 3.01 MB, <1M params) trained on FS-COCO, executing in **< 1 ms**.
5. **Interactive Compositional Combiner**: Gated MLP combiner (`combiner_mobile.onnx`, 3.01 MB) implementing CIRR-style residual steering for iterative gallery feedback in **2 ms**.

---

## 2. On-Device Empirical Benchmarks (Realme RMX3870, 4GB RAM)

All benchmarks below were recorded directly on a physical test device (`Realme RMX3870`, 4GB Physical RAM, Android 16 / SDK 36, 64-bit ARMv8-A SoC) indexing **2,735 local gallery photos**:

### A. Execution Provider Comparison (Text Mode)

| Provider Configuration | Cold Start (ms) | Warm / Steady (ms) | Engineering Analysis & Findings |
| :--- | :--- | :--- | :--- |
| **AUTO_NNAPI (2 threads)** | **4,718 ms** | **140 ms** | Heavy Android NNAPI driver JIT compilation on first run; HAL boundary marshaling adds overhead at steady-state. |
| **CPU_ONLY (2 threads)** | **599 ms** | **45 ms** | Instantaneous cold start, stable deterministic latency on ARM big cores. |
| **CPU_ONLY (4 threads)** | **663 ms** | **90 ms** | **2× slower at steady-state**: 4 threads cause thread context switching contention and thermal throttling on 4GB SoCs. |
| **XNNPACK (2 threads)** | **633 ms** | **44 ms** | **Fastest steady-state provider**: Optimized INT8 assembly kernels deliver 44 ms text embedding. |

> **Key Takeaway**: CPU/XNNPACK with **2 threads** is both faster on cold start (600ms vs 4,718ms) and faster at steady-state (44ms vs 140ms) than NNAPI for 77-token text sequences. The 4.7s cold start penalty of NNAPI is completely eliminated by our background warmup routine (`warmupSearchSessions()`), which pre-compiles all sessions in the background 4.6 seconds after launch.

### B. Stage-by-Stage Modality Latency Breakdown

```
================== STAGE-BY-STAGE MODALITY LATENCY PROFILING ==================
Search Modality       | Stage Breakdown                                                | Total (ms)
-------------------------------------------------------------------------------------------------
Text-Only             | Encode: 226ms  | Vector Scan: 40ms                             | 266 ms
Sketch-Only           | Encode: 210ms  | Vector Scan: 33ms                             | 243 ms
Composite (Text+Sk)   | Sk: 200ms | Tx: 147ms | Fusion: <1ms (0ms) | Scan: 8ms         | 358 ms
Refine (Combiner)     | Combiner: <1ms (0ms) | Scoped Top-K Rerank: 2ms                | 2 ms
=================================================================================================
Composite vs Text Ratio: 1.35x (Historical unoptimized ratio was 23x)
Isolated Fusion Module Latency: < 1 ms
=================================================================================================
```

### C. System Resource & Memory Reductions

| Metric | Before Optimization | After Optimization | Delta / Status |
| :--- | :--- | :--- | :--- |
| **Cold Idle PSS (RAM)** | 362.4 MB | **178.7 MB** | **-50.7% (-183.7 MB)** |
| **Cold Idle RSS (RAM)** | 491.5 MB | **312.4 MB** | **-36.5% (-179.1 MB)** |
| **Java Heap Allocated** | 81.3 MB | **19.1 MB** | **-76.5% (-62.2 MB)** |
| **Native Heap** | 209.0 MB | **95.7 MB** | **-54.2% (-113.3 MB)** |
| **Total Shipped APK Size** | ~236 MB | **137.64 MB** | **-41.7% (-98.4 MB)** |
| **Native Shared Libs in APK** | > 45 MB | **6.01 MB** | **-86.6% (single `arm64-v8a` slice)** |

---

## 3. Production Architecture vs. Rejected Research Experiments (Post-Mortem)

During empirical research and mobile profiling, several architectural avenues were rigorously tested and subsequently rejected or reverted based on on-device data:

### 1. STNet Attention Pooling (`vl2g/CSTBIR` AAAI 2024) — *REJECTED*
- **Mechanism**: Replaced Global Average Pooling (GAP) with cross-attention where the sketch embedding serves as query tokens $Q$ over unpooled photo patch tokens $K, V \in \mathbb{R}^{7 \times 7 \times 512}$.
- **Why It Failed for Mobile Galleries**: In gallery retrieval, the photo database must be **pre-computable offline**. Attention pooling creates a query-dependent photo embedding: calculating similarity against $N$ candidate photos requires passing all $N$ spatial patch tensors through the attention block at search time ($O(N \cdot \text{TransformerForward})$ instead of $O(N \cdot D)$ dot products). On a gallery of 4,000 photos, search latency exploded from 40ms to over 2 minutes. Furthermore, baseline GAP + Late Fusion achieved **19.10% R@1** on FS-COCO, matching or exceeding attention pooling.

### 2. TASK-former Cross-Attention Bottleneck (`janesjanes/tsbir` CVPR 2023) — *REJECTED*
- **Mechanism**: Inverted the cross-attention formulation so photo patch tokens $Q$ attend to fused $[sketch \parallel text]$ context tokens $K, V$.
- **Why It Failed for Edge Deployment**: Suffers from the identical query-time pre-computation bottleneck. Discarding GAP in favor of dynamic cross-attention prevents static vector indexing, rendering it fundamentally incompatible with mobile instant search.

### 3. On-Device Gradient Backpropagation Fine-Tuning — *REJECTED*
- **Mechanism**: Running SGD / Adam weight updates on device to adapt the model to user clicks.
- **Why It Failed**: Backward passes through ViT backbones on mobile CPUs cause massive thermal throttling, battery drain, and risk OS Out-Of-Memory (OOM) kills on 4GB devices. Instead, we adopted **zero-shot CIRR-style residual combiner inference**: `combiner_mobile.onnx` computes a modified query embedding in **< 1 ms** without modifying model weights.

### 4. 4-Thread Intra-Op Execution Budget — *REJECTED*
- **Mechanism**: Setting intra-op thread pools to 4 threads for ONNX inference.
- **Why It Degraded Performance**: Mobile SoCs feature heterogeneous core topologies (big.LITTLE). Spanning 4 threads across high-efficiency LITTLE cores and big cores causes severe thread preemption, lock convoying, and thermal throttling, slowing text inference from **45 ms (2 threads)** to **90 ms (4 threads)**. The production policy strictly enforces:
  - `interactiveThreads = 2` (foreground user queries)
  - `backgroundThreads = 1` (background photo indexer)

---

## 4. Vector Search: Exact Flat Scan vs. ANN on Mobile

A common question in mobile retrieval is whether to use Approximate Nearest Neighbor (ANN) vector indices such as HNSW, ScaNN, or IVF-PQ.

### Why Exact Flat Search + Min-Heap Top-50 is Optimal for Mobile Galleries:
1. **Gallery Scale**: Consumer mobile galleries typically range between 1,000 and 50,000 photos.
2. **Memory Footprint**:
   - Storing 5,000 512-D float32 vectors takes **10 MB** of contiguous RAM (`FloatArray(5000 * 512)`).
   - Even at 50,000 photos, the contiguous embedding matrix requires only **102 MB** of RAM.
3. **Scan Latency**:
   - Modern ARMv8-A NEON cores can execute 5,000 512-D dot products in **8 – 40 ms**.
   - With a bounded `PriorityQueue` Min-Heap of size $K=50$, Top-50 selection runs in $O(N \log 50)$, adding less than 2 ms.
4. **Zero Index Overhead**:
   - Building and re-indexing an HNSW graph on-device during gallery updates requires significant CPU cycles, graph memory overhead (often 2–3× the raw vector size), and complex serialization.
   - Exact flat scan provides **100% recall precision** with zero graph construction latency.

---

## 5. Official Repository Lineage & Academic Citations

To maintain strict academic integrity, the table below maps each component to its originating research paper and open-source implementation:

| Component | Source Repository | Citation / Paper | Usage & Adaptations |
| :--- | :--- | :--- | :--- |
| **Distilled Vision/Text Backbone** | [`apple/ml-mobileclip`](https://github.com/apple/ml-mobileclip) | MobileCLIP (Apple Research, CVPR 2024) | Pretrained `mobileclip_s1` (`fastvit_s12`) checkpoint. Quantized to static INT8 for edge deployment. |
| **FS-COCO Triplet Dataset** | [`pinakinathc/fscoco`](https://github.com/pinakinathc/fscoco) | FS-COCO (ECCV 2022) | Dataset parsing triplet tuples (`sketch`, `caption`, `photo`) for training and validation splits. |
| **Sketch Domain Adapter** | [`huggingface/peft`](https://github.com/huggingface/peft) | LoRA (Hu et al., ICLR 2022) | Applied rank-8 LoRA adapter to vision backbone; merged into weights via `merge_and_unload()`. |
| **Feedback Combiner MLP** | [`ABaldrati/CLIP4Cir`](https://github.com/ABaldrati/CLIP4Cir) | CLIP4Cir (CVPR 2022) | Gated MLP Combiner architecture with residual composition offset for candidate steering. |
| **Near-Duplicate Suppression** | [`idealo/imagededup`](https://github.com/idealo/imagededup) | imagededup | pHash perceptual hashing & cosine clustering for redundant photo collapse. |
| **Runtime & Acceleration** | [`microsoft/onnxruntime`](https://github.com/microsoft/onnxruntime) | ONNX Runtime Mobile | On-device execution engine with NNAPI hardware acceleration and CPU/XNNPACK fallback. |
| **Mobile Application** | *Original Code* | This Repository | Pure Kotlin Android app, custom Bézier QuickDraw canvas, Room DB, WorkManager lifecycle. |

---

## 6. Repository Structure

```
├── android_app/                    # Shipped Android Application (Production)
│   ├── app/
│   │   ├── src/main/assets/       # Quantized ONNX models (125.57 MB total)
│   │   │   ├── photo_backbone_int8.onnx      (28.35 MB)
│   │   │   ├── sketch_encoder_int8.onnx      (28.35 MB)
│   │   │   ├── text_encoder_int8.onnx        (61.39 MB)
│   │   │   ├── composite_fusion_mobile.onnx  (3.01 MB)
│   │   │   ├── combiner_mobile.onnx          (3.01 MB)
│   │   │   └── bpe_simple_vocab_16e6.txt.gz  (1.46 MB)
│   │   ├── src/main/java/com/tsbir/gallery/
│   │   │   ├── data/             # Room Database & TypeConverters
│   │   │   ├── ml/               # ModelManager (lazy loading, NNAPI/XNNPACK fallback)
│   │   │   │   └── ClipTokenizer.kt  # Zero-dependency Kotlin BPE Tokenizer
│   │   │   ├── ui/               # SketchCanvasView, ResultsAdapter, RefineDialog
│   │   │   ├── worker/           # GalleryIndexWorker & SearchCoordinator (yielding)
│   │   │   └── MainActivity.kt   # In-memory vector matrix, Top-50 min-heap, profiling
│   │   └── build.gradle.kts      # arm64-v8a target filter, noCompress config
│   └── README.md                 # Detailed Android developer build guide
│
├── src/                            # Research & PyTorch Training Pipeline
│   ├── data/                     # FS-COCO PyTorch DataLoader
│   ├── models/                   # CompositeModel, FastViT MobileCLIP, LoRA
│   ├── reranker/                 # FeedbackComposedRetriever, triplet mining
│   └── utils/                    # Checkpointing, logging, metrics
│
├── exported_models/                # Canonical FP32/INT8 ONNX model exports
├── tests/                          # Integration & regression test suite
│   ├── test_demo_e2e_fullscale.py# 300-sample chained ONNX pipeline validation
│   ├── test_tokenizer_parity.py  # Kotlin vs Python token-for-token parity test
│   ├── test_indexing_lifecycle.py# MediaStore diffing & deletion pruning tests
│   └── test_compositional_feedback_qualitative.py # CIRR steering semantic tests
│
├── train_fscoco.py                 # Multi-loss PyTorch training loop on FS-COCO
├── evaluate.py                     # 5-mode Recall@K & NDCG@10 evaluation
├── export_mobile.py                # PyTorch -> ONNX export & static INT8 quantization
└── requirements.txt                # Python environment dependencies
```

---

## 7. How to Build & Run

### A. Android Application (Recommended)

#### Prerequisites:
- Android Studio Iguana / Jellyfish / Koala or newer
- Android SDK 34+
- Physical ARM64 Android device (Android 7.0+ / API 24+, target SDK 34)

#### Command-Line Build & Installation:
```bash
cd android_app

# Assemble Debug APK with arm64-v8a slice (137.64 MB)
./gradlew assembleDebug

# Install directly to connected device
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Launch application
adb shell am start -n com.tsbir.gallery/.MainActivity
```

#### Running On-Device Benchmarks:
To trigger the automated execution provider shootout and modality profiling via ADB:
```bash
adb shell am start -n com.tsbir.gallery/.MainActivity --ez BENCHMARK true
adb logcat -s TSBIR_BENCHMARK TSBIR_PROFILE ModelManager
```

---

### B. Python Research & Training Pipeline

#### 1. Setup Environment:
```bash
pip install -r requirements.txt
```

#### 2. Train on FS-COCO:
```bash
python train_fscoco.py --data_dir fscoco --batch_size 16 --epochs 25 --lr 1e-4
```

#### 3. Evaluate Retrieval Metrics:
```bash
python evaluate.py --data_dir fscoco
```

#### 4. Export & Quantize ONNX Pipeline:
```bash
python export_mobile.py
```

#### 5. Run Integration Test Suite:
```bash
pytest tests/ -v
```

---

## 8. Known Limitations & Future Work
- **Ultra-Large Galleries (> 100,000 photos)**: While galleries up to 50,000 photos execute flat scan in < 40ms, galleries exceeding 100,000 items may benefit from scalar-quantized inner product search (INT8 dot products via NEON `SDOT` instructions) to bound RAM under 50 MB.
- **Backbone Deduplication**: Currently, `photo_backbone_int8.onnx` and `sketch_encoder_int8.onnx` are shipped as separate 28.35 MB models (56.7 MB total). Shipping a single shared backbone + applying the 3.01 MB `sketch_lora_delta.npz` dynamically at runtime via a custom ONNX Runtime operator would save an additional 25.34 MB in APK size.
