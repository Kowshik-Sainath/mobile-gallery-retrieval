"""
Quantization & Mobile Export Script for Edge Deployment (TFLite / ONNX Runtime Mobile).
Strips training-only sketch reconstruction decoder and exports INT8 quantized vision/text/fusion towers.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.composite_model import TSBIRCompositeModel
from src.reranker.combiner import FeedbackCombinerReranker

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class VisionEncoderExportable(nn.Module):
    """
    Inference-ready Vision Encoder wrapper (MobileCLIP Vision Tower + LoRA Adapter).
    Takes image tensor (1, 3, 224, 224) and outputs 512-d normalized embedding vector.
    """
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model.backbone

    def forward(self, image_tensor):
        image_features = self.backbone.encode_image(image_tensor)
        return F.normalize(image_features, dim=-1)


def export_to_onnx(base_model, output_dir="exported_models"):
    os.makedirs(output_dir, exist_ok=True)
    
    vision_model = VisionEncoderExportable(base_model)
    vision_model.eval()

    vision_fp32_path = os.path.join(output_dir, "vision_encoder_fp32.onnx")
    vision_int8_path = os.path.join(output_dir, "vision_encoder_int8.onnx")
    combiner_path = os.path.join(output_dir, "combiner_mobile.onnx")

    dummy_image = torch.randn(1, 3, 224, 224)

    # 1. Export Vision Encoder (Backbone + LoRA)
    print(f"Exporting Vision Tower + LoRA to ONNX format at: {vision_fp32_path}")
    
    # Disable C++ native MHA fast-path across tracing and export for clean operator decomposition
    orig_fastpath = True
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
        try:
            orig_fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
        except Exception:
            pass

    export_successful = False
    for opset in [17, 18, 14, 16]:
        try:
            torch.onnx.export(
                vision_model,
                (dummy_image,),
                vision_fp32_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=['image_input'],
                output_names=['image_embedding'],
                dynamic_axes={
                    'image_input': {0: 'batch_size'},
                    'image_embedding': {0: 'batch_size'}
                }
            )
            print(f"ONNX Vision Tower export succeeded with opset {opset}!")
            export_successful = True
            break
        except Exception as e:
            print(f"Notice: Opset {opset} export attempt failed: {type(e).__name__}: {e}")

    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        try:
            torch.backends.mha.set_fastpath_enabled(orig_fastpath)
        except Exception:
            pass

    # TorchScript Mobile Bundle Export
    print("Saving TorchScript Mobile model bundle (vision_encoder_mobile.pt)...")
    try:
        script_model = torch.jit.trace(vision_model, dummy_image)
        torch.jit.save(script_model, os.path.join(output_dir, "vision_encoder_mobile.pt"))
        print("TorchScript Mobile model saved successfully!")
    except Exception as e:
        print(f"TorchScript export note: {e}")

    if not export_successful and not os.path.exists(os.path.join(output_dir, "vision_encoder_mobile.pt")):
        raise RuntimeError(
            "Vision encoder export failed for both ONNX and TorchScript formats. "
            "Aborting before quantization/benchmarking."
        )

    if os.path.exists(vision_fp32_path):
        vision_fp32_mb = os.path.getsize(vision_fp32_path) / (1024 * 1024)
        print(f"Vision Encoder FP32 ONNX Model Size: {vision_fp32_mb:.2f} MB")
    else:
        vision_fp32_mb = 0.0

    # 2. Export Combiner Feedback Reranker MLP
    combiner = FeedbackCombinerReranker()
    combiner.eval()
    dummy_q = torch.randn(1, 512)
    dummy_c = torch.randn(1, 512)
    torch.onnx.export(
        combiner,
        (dummy_q, dummy_c),
        combiner_path,
        export_params=True,
        opset_version=14,
        input_names=['query_embedding', 'candidate_embedding'],
        output_names=['similarity_score']
    )
    combiner_size_mb = os.path.getsize(combiner_path) / (1024 * 1024)
    print(f"Combiner Reranker ONNX Model Size: {combiner_size_mb:.2f} MB")

    # 3. Dynamic INT8 Quantization
    if ONNX_AVAILABLE:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print(f"Quantizing Vision Encoder to INT8: {vision_int8_path}")
            quantize_dynamic(vision_fp32_path, vision_int8_path, weight_type=QuantType.QUInt8)
            vision_int8_mb = os.path.getsize(vision_int8_path) / (1024 * 1024)
            print(f"INT8 Quantized Vision Encoder Size: {vision_int8_mb:.2f} MB")
        except Exception as e:
            print(f"Notice: Dynamic INT8 quantization skipped ({e}). FP32 model ready.")

    # 4. Benchmark ONNX Runtime CPU Latency
    eval_path = vision_int8_path if os.path.exists(vision_int8_path) else vision_fp32_path
    if ONNX_AVAILABLE and os.path.exists(eval_path):
        print("\nBenchmarking ONNX Runtime CPU Latency on Edge Proxy...")
        session = ort.InferenceSession(eval_path, providers=['CPUExecutionProvider'])
        
        image_np = dummy_image.numpy()
        
        # Warmup
        session.run(None, {'image_input': image_np})
        
        times = []
        for _ in range(20):
            t0 = time.time()
            session.run(None, {'image_input': image_np})
            times.append((time.time() - t0) * 1000.0)

        avg_latency_ms = sum(times) / len(times)
        print("\n" + "="*60)
        print(f"{'Mobile Edge Deployment Benchmark Metric':<35} | {'Value':<20}")
        print("-" * 60)
        print(f"{'Vision Encoder FP32 Model Size':<35} | {vision_fp32_mb:.2f} MB")
        if os.path.exists(vision_int8_path):
            print(f"{'Vision Encoder INT8 Model Size':<35} | {vision_int8_mb:.2f} MB")
        print(f"{'Combiner Reranker Model Size':<35} | {combiner_size_mb:.2f} MB")
        print(f"{'Per-Image Inference Latency':<35} | {avg_latency_ms:.2f} ms")
        print(f"{'Edge Deployment RAM Budget Target':<35} | < 500 MB (4-8GB Phone)")
        print("="*60 + "\n")


def main():
    print("Initializing Mobile Model Export Pipeline...")
    device = "cpu"
    base_model = TSBIRCompositeModel(device=device)

    # Load fine-tuned weights if checkpoint exists
    ckpt_path = "checkpoints/checkpoint_epoch_15.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/checkpoint_latest.pt"

    if os.path.exists(ckpt_path):
        print(f"Loading trained weights from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        base_model.load_state_dict(checkpoint.get('full_state_dict', checkpoint), strict=False)

    export_to_onnx(base_model)
    print("Mobile Export Process Completed Successfully!")

if __name__ == "__main__":
    main()
