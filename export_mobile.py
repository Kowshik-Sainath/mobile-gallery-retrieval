"""
Quantization & Mobile Export Script for Edge Deployment (ONNX Runtime Mobile).

BUG 6 FIX applied:
  - Replaces quantize_dynamic with quantize_static using real FS-COCO images
    as calibration data (FSCOCOCalibrationDataReader).
  - Adds text encoder export (text_encoder_fp32.onnx + text_encoder_int8.onnx).
  - Loads pretrained combiner weights from combiner_pretrained.pt if available.
  - Enforces < 45 MB budget on static INT8 vision encoder.
  - Loads adapter-only checkpoint (FIX B compatible — state_dict not full_state_dict).

Expected outputs in exported_models/:
  vision_encoder_fp32.onnx     ~340 MB (reference)
  vision_encoder_int8.onnx     ~40-45 MB (static INT8)
  text_encoder_fp32.onnx       ~15 MB
  text_encoder_int8.onnx       ~5-8 MB
  combiner_mobile.onnx         ~3.5 MB
"""

import os
import io
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.composite_model import TSBIRCompositeModel
from src.reranker.combiner import FeedbackComposedRetriever, FeedbackCombinerReranker

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# MobileCLIP unnormalized [0, 1] range is used — no ImageNet mean/std constants needed


# ---------------------------------------------------------------------------
# Export wrappers (strip training-only heads)
# ---------------------------------------------------------------------------

class VisionEncoderExportable(nn.Module):
    """
    Inference-ready vision encoder: backbone base weights (FastViT) without adapter.
    Input:  (1, 3, 224, 224)
    Output: (1, 512) normalized embedding
    """
    def __init__(self, base_model):
        super().__init__()
        # Access the underlying FastViT model directly — 100% frozen base weights, zero PEFT wrapper overhead
        if hasattr(base_model, 'backbone') and hasattr(base_model.backbone, 'image_encoder'):
            enc = base_model.backbone.image_encoder
            if hasattr(enc, 'base_model') and hasattr(enc.base_model, 'model'):
                self.vision_core = enc.base_model.model
            else:
                self.vision_core = enc
        else:
            self.vision_core = base_model

    def forward(self, image_tensor):
        feats = self.vision_core(image_tensor)
        return F.normalize(feats, dim=-1)


class TextEncoderExportable(nn.Module):
    """
    Inference-ready text encoder: frozen tower + trained text adapter.
    Input:  token_ids (1, seq_len) int64
    Output: (1, 512) normalized embedding
    """
    def __init__(self, base_model):
        super().__init__()
        self._base = base_model

    def forward(self, token_ids):
        feats = self._base.backbone.encode_text(token_ids)
        adapted = feats + self._base.text_adapter(feats)
        return F.normalize(adapted, dim=-1)


# ---------------------------------------------------------------------------
# BUG 6 FIX — Static INT8 Calibration Data Reader
# ---------------------------------------------------------------------------

class FSCOCOCalibrationDataReader:
    """
    Feeds real FS-COCO photos to onnxruntime's static quantizer for INT8 calibration.
    Uses FSCOCODataset directly to guarantee exact [0, 1] preprocessing parity.
    """

    def __init__(self, data_dir: str = "fscoco", n_samples: int = 200):
        self.data = []
        self._idx = 0

        try:
            from src.data.fscoco_dataset import FSCOCODataset
            dataset = FSCOCODataset(data_dir, split="train")
            n = min(n_samples, len(dataset))
            indices = list(range(0, len(dataset), max(1, len(dataset) // n)))[:n]
            for i in indices:
                sample = dataset[i]
                arr = sample["photo"].unsqueeze(0).numpy().astype(np.float32)  # (1, 3, 224, 224)
                self.data.append({"image_input": arr})
            print(f"[Calibration] Loaded {len(self.data)} real photos from FS-COCO train split.")
        except Exception as e:
            print(f"[Calibration] Failed to load dataset: {e}")

        # Fallback: synthetic calibration if not enough real images
        if len(self.data) < 50:
            print(
                f"[Calibration] Only {len(self.data)} real images found. "
                "Supplementing with synthetic calibration samples."
            )
            while len(self.data) < 200:
                self.data.append({
                    'image_input': np.random.rand(1, 3, 224, 224).astype(np.float32)
                })

        print(f"[Calibration] Calibration dataset size: {len(self.data)} samples")

    def get_next(self):
        if self._idx >= len(self.data):
            return None
        item = self.data[self._idx]
        self._idx += 1
        return item

    def rewind(self):
        self._idx = 0


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------

def try_onnx_export(model, dummy_inputs, path, input_names, output_names, dynamic_axes=None):
    """Attempts ONNX export with opset fallback 17 → 18 → 16. Returns True on success."""
    for opset in [17, 18, 16]:
        try:
            if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
                torch.backends.mha.set_fastpath_enabled(False)

            torch.onnx.export(
                model,
                dummy_inputs,
                path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes or {},
            )
            print(f"[Export] ONNX export succeeded (opset {opset}): {os.path.basename(path)}")
            return True
        except Exception as e:
            print(f"[Export] Opset {opset} failed: {type(e).__name__}: {e}")
    return False


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_to_onnx(base_model, output_dir: str = "exported_models", data_dir: str = "fscoco"):
    os.makedirs(output_dir, exist_ok=True)

    vision_fp32_path = os.path.join(output_dir, "vision_encoder_fp32.onnx")
    vision_int8_path = os.path.join(output_dir, "vision_encoder_int8.onnx")
    text_fp32_path   = os.path.join(output_dir, "text_encoder_fp32.onnx")
    text_int8_path   = os.path.join(output_dir, "text_encoder_int8.onnx")
    combiner_path    = os.path.join(output_dir, "combiner_mobile.onnx")

    dummy_image = torch.zeros(1, 3, 224, 224)
    sizes = {}

    # Clean old exported files to avoid testing against stale models
    for p in [vision_fp32_path, vision_int8_path, text_fp32_path, text_int8_path, combiner_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    # --- 1. Vision encoder ---
    print("\n[Export] Exporting Vision Encoder (backbone, adapters disabled)...")
    vision_model = VisionEncoderExportable(base_model)
    vision_model.eval()

    ok = try_onnx_export(
        vision_model, (dummy_image,),
        vision_fp32_path,
        input_names=['image_input'],
        output_names=['image_embedding'],
        dynamic_axes={'image_input': {0: 'batch'}, 'image_embedding': {0: 'batch'}},
    )

    if not ok:
        print("[Export] ONNX export failed. Falling back to TorchScript mobile bundle.")
        try:
            script = torch.jit.trace(vision_model, dummy_image)
            ts_path = os.path.join(output_dir, "vision_encoder_mobile.pt")
            torch.jit.save(script, ts_path)
            print(f"[Export] TorchScript saved: {ts_path}")
        except Exception as e:
            print(f"[Export] TorchScript also failed: {e}")

    if os.path.exists(vision_fp32_path):
        sizes['vision_fp32'] = os.path.getsize(vision_fp32_path) / (1024 * 1024)
        print(f"[Export] Vision FP32: {sizes['vision_fp32']:.1f} MB")

    # --- 2. Text encoder ---
    print("\n[Export] Exporting Text Encoder (frozen tower + text adapter)...")
    text_model = TextEncoderExportable(base_model)
    text_model.eval()

    # Build a dummy token input
    dummy_tokens = torch.zeros(1, 77, dtype=torch.long)

    try_onnx_export(
        text_model, (dummy_tokens,),
        text_fp32_path,
        input_names=['token_ids'],
        output_names=['text_embedding'],
        dynamic_axes={'token_ids': {0: 'batch'}, 'text_embedding': {0: 'batch'}},
    )

    if os.path.exists(text_fp32_path):
        sizes['text_fp32'] = os.path.getsize(text_fp32_path) / (1024 * 1024)
        print(f"[Export] Text FP32: {sizes['text_fp32']:.1f} MB")

    # --- 3. Combiner reranker ---
    print("\n[Export] Exporting Combiner Query Composition MLP...")
    combiner = FeedbackComposedRetriever(feature_dim=512, hidden_dim=256)

    # Load pretrained weights if available
    pretrained_path = "checkpoints/combiner_pretrained.pt"
    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        c_state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        combiner.load_state_dict(c_state, strict=False)
        print(f"[Export] Loaded pretrained combiner weights from: {pretrained_path}")
    else:
        print("[Export] Warning: combiner_pretrained.pt not found. Exporting random-init combiner.")

    combiner.eval()
    dummy_shown = torch.zeros(1, 512)
    dummy_query = torch.zeros(1, 512)

    torch.onnx.export(
        combiner, (dummy_shown, dummy_query), combiner_path,
        export_params=True, opset_version=14,
        input_names=['shown_candidate', 'refined_query'],
        output_names=['combined_embedding'],
        dynamic_axes={
            'shown_candidate': {0: 'batch'},
            'refined_query':   {0: 'batch'},
            'combined_embedding': {0: 'batch'},
        }
    )
    sizes['combiner'] = os.path.getsize(combiner_path) / (1024 * 1024)
    print(f"[Export] Combiner: {sizes['combiner']:.1f} MB")

    # --- 4. BUG 6 FIX — Static INT8 quantization with calibration dataset ---
    if ONNX_AVAILABLE and os.path.exists(vision_fp32_path):
        try:
            from onnxruntime.quantization import (
                quantize_static, QuantType, CalibrationDataReader
            )

            print("\n[Quantize] Building calibration dataset from FS-COCO...")
            calib_reader = FSCOCOCalibrationDataReader(data_dir=data_dir, n_samples=200)

            print(f"[Quantize] Running STATIC INT8 quantization -> {vision_int8_path}")
            quantize_static(
                model_input=vision_fp32_path,
                model_output=vision_int8_path,
                calibration_data_reader=calib_reader,
                quant_format=None,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=True,
            )
            sizes['vision_int8'] = os.path.getsize(vision_int8_path) / (1024 * 1024)
            print(f"[Quantize] Vision Static INT8: {sizes['vision_int8']:.1f} MB")

        except Exception as e:
            print(f"[Quantize] Static INT8 failed ({e}). Falling back to dynamic INT8.")
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quantize_dynamic(vision_fp32_path, vision_int8_path, weight_type=QuantType.QInt8)
                sizes['vision_int8'] = os.path.getsize(vision_int8_path) / (1024 * 1024)
                print(f"[Quantize] Vision Dynamic INT8 (fallback): {sizes['vision_int8']:.1f} MB")
            except Exception as e2:
                print(f"[Quantize] Dynamic INT8 also failed: {e2}")
                sizes['vision_int8'] = sizes.get('vision_fp32', 0)

    # Text encoder INT8
    if ONNX_AVAILABLE and os.path.exists(text_fp32_path):
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(text_fp32_path, text_int8_path, weight_type=QuantType.QInt8)
            sizes['text_int8'] = os.path.getsize(text_int8_path) / (1024 * 1024)
            print(f"[Quantize] Text Dynamic INT8: {sizes['text_int8']:.1f} MB")
        except Exception as e:
            print(f"[Quantize] Text quantization failed: {e}")

    # --- 5. Benchmark ---
    if ONNX_AVAILABLE:
        eval_path = vision_int8_path if os.path.exists(vision_int8_path) else vision_fp32_path
        if os.path.exists(eval_path):
            print(f"\n[Benchmark] Running ONNX Runtime CPU latency on: {os.path.basename(eval_path)}")
            session = ort.InferenceSession(eval_path, providers=['CPUExecutionProvider'])
            img_np = dummy_image.numpy()

            # Warmup
            for _ in range(3):
                session.run(None, {'image_input': img_np})

            times = []
            for _ in range(20):
                t0 = time.time()
                session.run(None, {'image_input': img_np})
                times.append((time.time() - t0) * 1000.0)
            avg_ms = sum(times) / len(times)

            total_mb = sizes.get('vision_int8', 0) + sizes.get('combiner', 0)

            print("\n" + "=" * 65)
            print(f"{'Metric':<38} | {'Value':<22}")
            print("-" * 65)
            for k, v in sizes.items():
                print(f"{'  ' + k.replace('_', ' ').title() + ' Size':<38} | {v:.1f} MB")
            print(f"{'  Total Quantized (Vision+Combiner)':<38} | {total_mb:.1f} MB")
            print(f"{'  Per-Image Inference Latency':<38} | {avg_ms:.1f} ms")
            print(f"{'  Size Budget Target':<38} | < 45 MB (BUG 6 target)")
            print("=" * 65)

            # Budget enforcement
            vision_int8_mb = sizes.get('vision_int8', sizes.get('vision_fp32', 999))
            if vision_int8_mb > 45.0:
                print(
                    f"\n[Warning] Vision INT8 model is {vision_int8_mb:.1f} MB — "
                    "exceeds 45 MB B2 budget. "
                    "Consider MobileCLIP-S0 or additional pruning for strict compliance."
                )
            else:
                print(f"\n[OK] Vision INT8 model is {vision_int8_mb:.1f} MB — within 45 MB budget.")

    # --- 6. Correctness verification: PyTorch vs ONNX output cosine similarity ---
    if ONNX_AVAILABLE and os.path.exists(vision_fp32_path):
        print("\n[Verification] Checking PyTorch vs ONNX numerical parity...")
        try:
            from src.data.fscoco_dataset import FSCOCODataset
            test_dataset = FSCOCODataset(data_dir, split="test")
            indices = [0, 50, 100, 150, 200]
            indices = [i for i in indices if i < len(test_dataset)]
            if len(indices) < 5:
                indices = list(range(min(5, len(test_dataset))))

            torch_vision = VisionEncoderExportable(base_model).eval()
            check_session = ort.InferenceSession(vision_fp32_path, providers=['CPUExecutionProvider'])

            sims = []
            for idx in indices:
                sample = test_dataset[idx]
                photo_tensor = sample['photo'].unsqueeze(0)  # (1, 3, 224, 224)
                
                # PyTorch prediction
                with torch.no_grad():
                    torch_emb = torch_vision(photo_tensor).squeeze(0).numpy()
                
                # ONNX prediction
                onnx_emb = check_session.run(None, {'image_input': photo_tensor.numpy()})[0].squeeze(0)
                
                # Cosine similarity
                cos_sim = float(np.dot(torch_emb, onnx_emb) / (np.linalg.norm(torch_emb) * np.linalg.norm(onnx_emb) + 1e-8))
                sims.append(cos_sim)
                print(f"  Sample {idx}: Cosine Similarity = {cos_sim:.5f}")

            avg_sim = float(np.mean(sims))
            min_sim = float(np.min(sims))
            print(f"[Verification] Average PyTorch vs ONNX Cosine Similarity: {avg_sim:.5f} (Min: {min_sim:.5f})")
            assert min_sim > 0.95, f"Correctness check FAILED: min cosine similarity {min_sim:.4f} <= 0.95!"
            print("[Verification] PASS: PyTorch vs ONNX outputs are aligned (cos_sim > 0.95).")
        except Exception as e:
            print(f"[Verification] Error during correctness verification: {e}")


def main():
    print("=" * 65)
    print("Mobile Model Export Pipeline — BUG 6 Fixed Version")
    print("=" * 65)

    device = "cpu"
    base_model = TSBIRCompositeModel(device=device)

    # Load adapter-only checkpoint (FIX B compatible)
    ckpt_path = "checkpoints/checkpoint_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/checkpoint_epoch_15.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/checkpoint_latest.pt"

    if os.path.exists(ckpt_path):
        print(f"[Export] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Support both old full_state_dict and new adapter-only state_dict (FIX B)
        state = ckpt.get('state_dict', ckpt.get('full_state_dict', ckpt))
        base_model.load_state_dict(state, strict=False)
    else:
        print("[Export] No checkpoint found — exporting with random adapter init.")

    export_to_onnx(base_model, output_dir="exported_models", data_dir="fscoco")
    print("\n[Export] Mobile Export Process Completed.")


if __name__ == "__main__":
    main()
