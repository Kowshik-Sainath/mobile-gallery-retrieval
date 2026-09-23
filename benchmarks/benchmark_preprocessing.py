"""
Benchmark Preprocessing Latency vs ONNX INT8 Inference Latency across Batch Sizes.

Measures:
  1. Image preprocessing alone: PIL Bicubic resize (224, 224) + scaling to [0, 1] float32 NCHW.
  2. ONNX INT8 Backbone inference alone: photo_backbone_int8.onnx on (B, 3, 224, 224).
  3. Total indexing latency and preprocessing percentage of total indexing time.
  4. Across batch sizes: 1, 8, 16, 32.
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import onnxruntime as ort

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_test_images(num_needed=32):
    """Retrieve up to num_needed real images from fscoco/images, or create dummy PIL images."""
    images = []
    fscoco_img_dir = os.path.join("fscoco", "images")
    if os.path.exists(fscoco_img_dir):
        for root, _, files in os.walk(fscoco_img_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        im = Image.open(os.path.join(root, f)).convert("RGB")
                        images.append(im)
                        if len(images) >= num_needed:
                            break
                    except Exception:
                        pass
            if len(images) >= num_needed:
                break

    # If not enough images found, pad with random colored PIL images
    while len(images) < num_needed:
        rnd = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        images.append(Image.fromarray(rnd))

    return images[:num_needed]


def preprocess_single_image(img: Image.Image) -> np.ndarray:
    """Preprocess single PIL Image to float32 (3, 224, 224) in [0, 1]."""
    resized = img.resize((224, 224), Image.BICUBIC)
    arr = np.array(resized, dtype=np.float32) / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))                 # CHW
    return arr


def run_benchmark():
    onnx_path = os.path.join("exported_models", "photo_backbone_int8.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Model not found: {onnx_path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    batch_sizes = [1, 8, 16, 32]
    all_images = get_test_images(max(batch_sizes))

    print("=" * 80)
    print("PREPROCESSING VS ONNX INT8 INFERENCE BENCHMARK")
    print(f"Model: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.2f} MB)")
    print(f"Hardware Threads: {sess_options.intra_op_num_threads} | Provider: CPUExecutionProvider")
    print("=" * 80)

    results = []

    for bs in batch_sizes:
        batch_imgs = all_images[:bs]

        # Warmup preprocessing
        for _ in range(3):
            _ = [preprocess_single_image(img) for img in batch_imgs]

        # Benchmark Preprocessing
        num_runs = 30 if bs <= 8 else 15
        t0 = time.perf_counter()
        for _ in range(num_runs):
            preprocessed = [preprocess_single_image(img) for img in batch_imgs]
            batch_tensor = np.stack(preprocessed, axis=0)  # (B, 3, 224, 224)
        total_prep_sec = time.perf_counter() - t0
        avg_prep_batch_ms = (total_prep_sec / num_runs) * 1000.0
        avg_prep_per_img_ms = avg_prep_batch_ms / bs

        # Warmup Inference
        for _ in range(3):
            _ = session.run(None, {input_name: batch_tensor})

        # Benchmark ONNX INT8 Inference
        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = session.run(None, {input_name: batch_tensor})
        total_inf_sec = time.perf_counter() - t0
        avg_inf_batch_ms = (total_inf_sec / num_runs) * 1000.0
        avg_inf_per_img_ms = avg_inf_batch_ms / bs

        total_per_img_ms = avg_prep_per_img_ms + avg_inf_per_img_ms
        prep_fraction_pct = (avg_prep_per_img_ms / total_per_img_ms) * 100.0

        results.append({
            "batch_size": bs,
            "prep_per_img_ms": avg_prep_per_img_ms,
            "inf_per_img_ms": avg_inf_per_img_ms,
            "total_per_img_ms": total_per_img_ms,
            "prep_pct": prep_fraction_pct,
            "prep_batch_ms": avg_prep_batch_ms,
            "inf_batch_ms": avg_inf_batch_ms,
        })

    print(f"{'Batch Size':>10} | {'Prep / img':>12} | {'Infer / img':>12} | {'Total / img':>12} | {'Prep % of Total':>16}")
    print("-" * 75)
    for r in results:
        print(f"{r['batch_size']:>10} | {r['prep_per_img_ms']:>10.2f} ms | {r['inf_per_img_ms']:>10.2f} ms | {r['total_per_img_ms']:>10.2f} ms | {r['prep_pct']:>15.1f} %")
    print("=" * 80)

    # Key takeaways
    b16 = [r for r in results if r['batch_size'] == 16][0]
    print(f"\n[Summary @ Batch Size 16 (Recommended Mobile Batch)]:")
    print(f"  - Preprocessing time per photo: {b16['prep_per_img_ms']:.2f} ms")
    print(f"  - ONNX INT8 inference per photo: {b16['inf_per_img_ms']:.2f} ms")
    print(f"  - Preprocessing accounts for {b16['prep_pct']:.1f}% of end-to-end indexing time.")
    if b16['prep_pct'] > 20.0:
        print(f"  - FINDING: Preprocessing is a SIGNIFICANT bottleneck ({b16['prep_pct']:.1f}% of total).")
        print(f"    On Android, offloading resize and format conversion to Hardware Canvas / RenderEffect")
        print(f"    is essential to prevent thermal throttling and achieve real-time indexing throughput.")
    else:
        print(f"  - FINDING: ONNX INT8 inference dominates indexing time ({100 - b16['prep_pct']:.1f}% of total).")

    return results


if __name__ == "__main__":
    run_benchmark()
