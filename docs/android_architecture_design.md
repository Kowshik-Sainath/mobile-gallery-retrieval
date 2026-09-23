# Offline-Only Low/Mid-Range Android Mobile Deployment Architecture

This document specifies the technical design, SQLite schema, background scheduling, memory budget management, and offline runtime architecture for deploying the Text + Sketch Based Image Retrieval (T+SBIR) system on Android mobile devices (4–8GB RAM).

---

## 1. Zero Network Dependency Architecture (Requirement B1)

- **100% Offline Execution**: All model weights (`vision_encoder_int8.onnx`, `text_encoder_int8.onnx`, `combiner_mobile.onnx`), tokenizer vocabulary files, and image pre-processing configurations are pre-bundled inside the Android App Bundle (`app/src/main/assets/`).
- **No HTTP/Cloud Calls**: The app initiates **zero network requests** at runtime. There are no remote API calls, no runtime model downloads, and no telemetry/analytics services requiring internet connectivity.
- **Airplane Mode Compliance**: Cold start, photo gallery scanning, embedding computation, sketch search, and feedback reranking operate seamlessly with Wi-Fi and Cellular data turned off.

---

## 2. On-Disk Flash Footprint & Quantization (Requirement B2, B5, B8)

- **Target Deployment Size**: **< 45 MB total model disk footprint**.
- **Quantization Strategy**: Static Post-Training INT8 Quantization (`QuantType.QUInt8`) calibrated on representative real photos and sketches from the FS-COCO dataset.
- **Excluded Components**: Training-only modules (`SketchReconstructionDecoder`, loss heads, optimizer states) are stripped from exported artifacts during ONNX export.

### Exported Asset Footprint (Deduplicated Architecture):
| Asset File | Format | Actual Disk Size | Role |
| :--- | :--- | :--- | :--- |
| `photo_backbone_int8.onnx` | INT8 ONNX | 28.35 MB | Shared base vision graph for photo gallery encoding |
| `sketch_lora_delta.npz` | Compressed NPZ | 3.01 MB | Low-rank LoRA delta tensors (in-memory merge for sketch encoding) |
| `text_encoder_int8.onnx` | INT8 ONNX | 61.39 MB | Encodes text query descriptions (77 tokens int64) |
| `composite_fusion_mobile.onnx` | FP32 ONNX | 3.00 MB | Gated query fuser (fuses sketch + text embeddings) |
| `combiner_mobile.onnx` | FP32 ONNX | 3.01 MB | Gated MLP feedback reranker (<1M params) |
| **Total Model Suite** | — | **98.76 MB** | Deduplicated vision saves 28.35 MB; exceeds 45 MB model target by 53.76 MB due to text transformer |

- **Runtime Engine**: **ONNX Runtime Mobile SDK** (`com.microsoft.onnxruntime:onnxruntime-android`) configured with **NNAPI** and **XNNPACK** execution providers for hardware acceleration on ARM CPUs/NPUs.

---

## 3. On-Device Embedding Database Schema & Incremental Indexing (Requirement B6)

The app utilizes a local **Room / SQLite Database** (`gallery_search.db`) to store 512-dimensional photo embeddings compactly.

### SQLite Schema:
```sql
CREATE TABLE IF NOT EXISTS photo_embeddings (
    photo_uri TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,          -- 512-element INT8 / FP16 compact binary vector (512 bytes)
    last_modified INTEGER NOT NULL,   -- Timestamp in millis for change detection
    cluster_id INTEGER DEFAULT -1,    -- Near-duplicate cluster identifier from pHash/cosine dedup
    indexed_at INTEGER NOT NULL       -- Timestamp when photo was indexed
);

CREATE INDEX IF NOT EXISTS idx_photo_modified ON photo_embeddings(last_modified);
CREATE INDEX IF NOT EXISTS idx_photo_cluster ON photo_embeddings(cluster_id);
```

### CLIP-Finder2-Style Gallery Reconciliation Lifecycle:
1. **First-Launch Batch Indexing**:
   - Background worker processes photos in configurable batches (16–32 photos) using `photo_backbone_int8.onnx`.
   - Batching amortizes NNAPI/ORT context switches and warmup overhead.
2. **Launch-Time Diff Reconciliation**:
   - Queries `MediaStore.Images.Media.EXTERNAL_CONTENT_URI` for current photo URIs and `date_modified`.
   - Compares timestamps against `photo_embeddings.last_modified` in a single SQL index lookup.
   - **Zero-Inference Short-Circuit**: If all photos are unchanged, exactly **0 embedding calls** are made.
3. **Incremental Modification/Addition**:
   - Only new photos or photos whose `last_modified` > DB timestamp are queued for embedding.
4. **Deletion Pruning**:
   - Photos present in SQLite but missing from `MediaStore` (deleted by user in Google Photos or File Manager) are immediately purged via `DELETE FROM photo_embeddings WHERE photo_uri IN (...)`, keeping the vector index consistent.

---

## 4. Zero-Idle CPU & Background Job Control (Requirement B4)

To prevent battery drain and maintain **0% idle CPU usage**:

1. **Push-Based Media Triggers**: The app registers an Android `ContentObserver` on `MediaStore.Images.Media.EXTERNAL_CONTENT_URI`. It **never** uses polling loops.
2. **WorkManager Scheduling**: When the `ContentObserver` detects new photos, it schedules a background job via Android `WorkManager` with the following battery-conscious constraints:
   - `setRequiresBatteryNotLow(true)`
   - `setRequiresStorageNotLow(true)`
   - `setRequiredNetworkType(NetworkType.NOT_REQUIRED)`
3. **Batch Amortization**: Background indexing processes photos in batches of 16–32 photos.
4. **Session Lifecycle & Memory Release**:
   - `OrtSession` handles are released or placed into a low-memory resident state after 30 seconds of inactivity.
   - When the user closes the app or navigates away, ONNX memory buffers are garbage-collected to return RAM to the OS.

---

## 5. RAM Footprint & Scalable Search Strategy (Requirement B3, B7)

### Peak RAM Budget:
- **Target Peak RAM**: **< 150 MB** above baseline OS app overhead.
- **Memory Control**: Photos are loaded and resized to $224 \times 224$ in batches of 16. Bitmaps are recycled immediately.

### Scalable Search Algorithm:
- **Gallery Size $\le$ 5,000 Photos (Brute-Force Vector Scan)**:
  - Embeddings are streamed from SQLite in chunks of 1,000 vectors ($1000 \times 512 \text{ bytes} \approx 512 \text{ KB}$).
  - Dot-product similarity search is performed using fast SIMD instructions (`arm_neon`). Search completes in **< 40 ms**.
- **Gallery Size $>$ 5,000 Photos (Quantized Flat Index / ANN)**:
  - Embeddings use quantized INT8 vector representations with a small flat index or lightweight mobile ANN library. Search completes in **< 120 ms**.

---

## 6. App Size & Distribution Budget (Requirement B9)

- **Format**: Android App Bundle (`.aab`).
- **Native ABI Splitting**: Configured with per-ABI splitting (`arm64-v8a`, `armeabi-v7a`). Users only download native `.so` binaries matching their specific CPU architecture.
- **Installed Size Breakdown**:
  - Native Libraries (ONNX Runtime Mobile + NNAPI): ~22 MB
  - INT8 Deduplicated Models: ~98.8 MB
  - APK Code & Android Assets (Tokenizer vocab): ~12 MB
  - **Total Installed Footprint**: **~132.8 MB** (exceeds <90 MB budget by 42.8 MB; distillation of text encoder to 15 MB required to meet <90 MB).

---

## 7. Preprocessing Benchmarking & Hardware Acceleration Findings

Empirical profiling on `photo_backbone_int8.onnx` across batch sizes produced the following latency breakdown:

| Batch Size | Preprocessing / Photo (Bicubic + [0, 1]) | ONNX INT8 Inference / Photo | Total / Photo | Preprocessing Fraction |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 1.81 ms | 21.21 ms | 23.01 ms | 7.8% |
| **8** | 2.22 ms | 19.89 ms | 22.10 ms | 10.0% |
| **16** | 2.07 ms | 25.02 ms | 27.09 ms | 7.6% |
| **32** | 2.15 ms | 27.18 ms | 29.33 ms | 7.3% |

### Key Architectural Takeaways:
- **Inference Dominance**: ONNX INT8 inference accounts for **92.4%** of end-to-end indexing time at batch size 16. Preprocessing accounts for only **7.6%** (2.07 ms / photo).
- **Hardware Canvas Offloading**: While CPU Bicubic resizing is relatively fast (2.07 ms), on low-end thermal-constrained SoCs (e.g. MediaTek Helio G85, Snapdragon 680), continuous CPU decoding and software resizing generates thermal throttling. Offloading image decoding and downscaling to Android's `HardwareRenderer` / `RenderEffect` GPU pipeline frees CPU cycles for background SQLite vector writes.

---

## 8. On-Device Model Profiler View Specification

An internal developer diagnostics overlay (`ModelProfilerDialogFragment`) provides real-time telemetry on the mobile device:

### Profiler Telemetry Metrics:
1. **Pipeline Latencies (ms)**:
   - `t_tokenize`: Tokenizer execution time (Kotlin BPE).
   - `t_sketch_infer`: Base ONNX + LoRA delta inference time.
   - `t_text_infer`: Text Transformer ONNX inference time.
   - `t_fusion`: Composite query fusion inference time.
   - `t_vector_scan`: SQLite dot-product similarity search time across $N$ gallery photos.
   - `t_rerank`: FeedbackCombiner inference time (when user submits candidate feedback).
2. **Memory & Thermal State**:
   - `ram_rss_mb`: Current process Resident Set Size.
   - `ram_peak_mb`: Peak memory usage during gallery batch indexing.
   - `thermal_status`: Android `ThermalStatusListener` level (`NONE`, `LIGHT`, `MODERATE`, `SEVERE`).
3. **Index Health**:
   - `total_photos`: Total indexed gallery photos.
   - `stale_entries`: Count of photos queued for reconciliation.
   - `short_circuit_hits`: Count of launches where zero inference was triggered.

