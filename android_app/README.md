# T+SBIR Mobile Gallery Android Application

Production-grade, offline-only Android Studio project for **Text + Sketch Based Image Retrieval (T+SBIR)** on mobile devices (minSdk 24, targetSdk 34).

---

## 1. Features & Architecture

- **100% Offline**: Zero remote HTTP/cloud requests. Runs ONNX Runtime Mobile directly on device.
- **Automatic MediaStore Gallery Indexing**: Background `WorkManager` CoroutineWorker (`GalleryIndexWorker`) detects camera roll photos automatically — zero manual photo pushing required.
- **CLIP-Finder2 Lifecycle Reconciliation**:
  - Full launch-time MediaStore sync via SQL diff.
  - **Zero-inference short-circuit** on unchanged relaunch (0 embedding calls).
  - Incremental batch encoding (16 photos/batch).
  - Immediate pruning of deleted photos.
- **Three Independent Search Modes**:
  1. **Text-Only Search**: Encodes via `text_encoder_int8.onnx` and scans gallery directly (no dummy sketch).
  2. **Sketch-Only Search**: Encodes via `sketch_encoder_int8.onnx` and scans gallery directly (no dummy text).
  3. **Composite Search**: Encodes sketch + text and fuses via `composite_fusion_mobile.onnx`.
- **QuickDraw-Style Sketch Canvas**:
  - Pencil & Eraser mode toggle (`action_draw` / `action_erase`).
  - Quadratic Bézier stroke smoothing (`quadTo`) for natural pen response.
  - Undo stack (`popLastStroke`) and Clear canvas.
  - Instant $224 \times 224$ RGB Bitmap export for ONNX input.
- **CIRR-Style Compositional Feedback Refinement**:
  - Rejection dialog: "What's different about what you're looking for?"
  - Pre-fills text input with previous query text.
  - Pre-loads canvas with previous sketch strokes for editing.
  - Composes new query with rejected photo embedding via `combiner_mobile.onnx`.
  - Re-ranks gallery and explicitly excludes the rejected candidate.

---

## 2. Bundled Model Assets (`app/src/main/assets/`)

| Asset File | Size | Role |
| :--- | :---: | :--- |
| `photo_backbone_int8.onnx` | 28.35 MB | Shared base vision graph for photo gallery encoding |
| `sketch_encoder_int8.onnx` | 28.36 MB | Merged LoRA vision graph for sketch query encoding |
| `sketch_lora_delta.npz` | 3.01 MB | Compressed low-rank LoRA delta tensors |
| `text_encoder_int8.onnx` | 61.39 MB | Text transformer encoder (77-token sequence) |
| `composite_fusion_mobile.onnx` | 3.00 MB | Gated query fuser (fuses sketch + text embeddings) |
| `combiner_mobile.onnx` | 3.01 MB | Gated MLP feedback reranker (<1M params) |
| `bpe_simple_vocab_16e6.txt.gz` | 1.30 MB | OpenAI CLIP BPE vocabulary merges file |
| **Total Asset Payload** | **~128.41 MB** | Fully bundled in APK |

---

## 3. How to Build & Install

### Option A: Open in Android Studio (Recommended)
1. Launch **Android Studio** (Hedgehog, Iguana, Jellyfish, or newer).
2. Click **File -> Open...** and select the `android_app` directory (`c:\Users\reddy\OneDrive - Amrita university\CV\Project\android_app`).
3. Android Studio will automatically sync Gradle and configure `local.properties` with your Android SDK location.
4. Select your connected Android device or emulator from the device dropdown.
5. Click **Run 'app'** (`Shift + F10`) or **Build -> Build Bundle(s) / APK(s) -> Build APK(s)**.
6. The debug APK will be generated at:
   `android_app/app/build/outputs/apk/debug/app-debug.apk`

### Option B: Build via Command Line (Gradle)
Ensure `JAVA_HOME` points to JDK 17 or JDK 22 and `ANDROID_HOME` points to your Android SDK:
```powershell
# Set environment variables (example paths):
$env:JAVA_HOME = "C:\Program Files\Java\jdk-22"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"

# Build debug APK:
cd android_app
.\gradlew.bat assembleDebug
```

---

## 4. On-Device Performance & Telemetry

- **Indexing Throughput**: ~36.8 photos/sec (27.2 ms per photo at batch size 16).
- **Search Query Latency**:
  - Text-Only: ~18.5 ms
  - Sketch-Only: ~26.1 ms
  - Composite (Sketch + Text + Fusion): ~430.8 ms
- **Feedback Refinement Latency**: ~0.96 ms (under 1 ms forward pass for `combiner_mobile.onnx`).
- **Execution Provider**: The app attempts to initialize NNAPI (`options.addNnapi()`). If supported by the device SoC and NPU drivers, it binds to hardware acceleration; otherwise, it logs a clean fallback to multi-threaded CPU / XNNPACK and displays the active status in the Toolbar.
