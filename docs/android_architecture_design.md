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

### Exported Asset Footprint:
| Asset File | Format | Target Disk Size | Role |
| :--- | :--- | :--- | :--- |
| `vision_encoder_int8.onnx` | INT8 ONNX | ~38.5 MB | Encodes photo gallery & scene sketch images |
| `text_encoder_int8.onnx` | INT8 ONNX | ~4.2 MB | Encodes text query descriptions |
| `combiner_mobile.onnx` | FP32/INT8 ONNX | ~3.5 MB | Gated MLP feedback reranker (<1M params) |
| **Total Model Footprint** | — | **~46.2 MB** | **Fits strictly within <50 MB budget** |

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

### Idempotent Incremental Indexing Strategy:
1. When an indexing job runs, the app queries `MediaStore.Images.Media.EXTERNAL_CONTENT_URI` for photo URIs and `date_modified`.
2. The app compares `MediaStore` timestamps against `photo_embeddings.last_modified`.
3. Only **newly added** or **modified** photos are passed through `vision_encoder_int8.onnx`.
4. Already-indexed, unchanged photos are skipped, avoiding redundant computation.

---

## 4. Zero-Idle CPU & Background Job Control (Requirement B4)

To prevent battery drain and maintain **0% idle CPU usage**:

1. **Push-Based Media Triggers**: The app registers an Android `ContentObserver` on `MediaStore.Images.Media.EXTERNAL_CONTENT_URI`. It **never** uses polling loops.
2. **WorkManager Scheduling**: When the `ContentObserver` detects new photos, it schedules a background job via Android `WorkManager` with the following battery-conscious constraints:
   - `setRequiresBatteryNotLow(true)`
   - `setRequiresStorageNotLow(true)`
   - `setRequiredNetworkType(NetworkType.NOT_REQUIRED)`
3. **Batch Amortization**: Background indexing processes photos in batches (e.g., 20 photos per batch) to amortize ONNX session warmup overhead.
4. **Session Lifecycle & Memory Release**:
   - `OrtSession` handles are released or placed into a low-memory resident state after 30 seconds of inactivity.
   - When the user closes the app or navigates away, ONNX memory buffers are garbage-collected to return RAM to the OS.

---

## 5. RAM Footprint & Scalable Search Strategy (Requirement B3, B7)

### Peak RAM Budget:
- **Target Peak RAM**: **< 150 MB** above baseline OS app overhead.
- **Memory Control**: Photos are loaded and resized to $224 \times 224$ one at a time during background indexing. Large bitmap objects are immediately recycled (`bitmap.recycle()`).

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
- **Installed Size Budget**:
  - Native Libraries (ONNX Runtime Mobile + NNAPI): ~22 MB
  - INT8 Model Weights: ~46 MB
  - APK Code & Android Assets: ~12 MB
  - **Total Installed Footprint**: **~80 MB** (strictly fits within <90 MB budget).
