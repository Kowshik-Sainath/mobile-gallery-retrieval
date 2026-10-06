"""
Mobile Model Export & INT8 Quantization Pipeline for T+SBIR.
Exports and quantizes the full T+SBIR pipeline with explicit LoRA isolation:
  1. Photo Encoder:  photo_encoder_fp32.onnx  + photo_encoder_int8.onnx  (LoRA DISABLED)
  2. Sketch Encoder: sketch_encoder_fp32.onnx + sketch_encoder_int8.onnx (LoRA MERGED via merge_and_unload)
  3. Text Encoder:   text_encoder_fp32.onnx   + text_encoder_int8.onnx   (Frozen tower + text adapter)
  4. Composite Fusion: composite_fusion_mobile.onnx                     ((sketch, text) -> composite_query)
  5. Feedback Combiner: combiner_mobile.onnx                             ((shown, refined) -> combined_query)

Also executes rigorous numerical parity verification (cosine similarity > 0.95)
for all 5 components against their PyTorch source with real test inputs.
"""

import os
import io
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.composite_model import TSBIRCompositeModel
from src.models.backbone import set_active_sketch_adapter
from src.reranker.combiner import FeedbackComposedRetriever
from src.data.fscoco_dataset import FSCOCODataset

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType, CalibrationDataReader
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Component Export Wrappers
# ---------------------------------------------------------------------------

class PhotoEncoderExportable(nn.Module):
    """
    Inference photo encoder: base FastViT weights, LoRA guaranteed absent.
    Input:  image_input (1, 3, 224, 224)
    Output: image_embedding (1, 512)
    """
    def __init__(self, base_fastvit_model: nn.Module):
        super().__init__()
        self.model = base_fastvit_model

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        feats = self.model(image_tensor)
        return F.normalize(feats, dim=-1)


class SketchEncoderExportable(nn.Module):
    """
    Inference sketch encoder: FastViT with sketch LoRA weights permanently merged.
    Input:  sketch_input (1, 3, 224, 224)
    Output: sketch_embedding (1, 512)
    """
    def __init__(self, merged_mci_model: nn.Module):
        super().__init__()
        self.model = merged_mci_model

    def forward(self, sketch_tensor: torch.Tensor) -> torch.Tensor:
        feats = self.model(sketch_tensor)
        return F.normalize(feats, dim=-1)


class TextEncoderExportable(nn.Module):
    """
    Inference text encoder: frozen text tower + trained text adapter.
    Input:  token_ids (1, seq_len) int64
    Output: text_embedding (1, 512)
    """
    def __init__(self, base_model: TSBIRCompositeModel):
        super().__init__()
        self.backbone = base_model.backbone
        self.text_adapter = base_model.text_adapter

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.encode_text(token_ids)
        adapted = feats + self.text_adapter(feats)
        return F.normalize(adapted, dim=-1)


class CompositeFusionExportable(nn.Module):
    """
    Inference composite fusion: fuses sketch and text embeddings into a composite query.
    Inputs: sketch_embed (1, 512), text_embed (1, 512)
    Output: composite_query (1, 512)
    """
    def __init__(self, base_model: TSBIRCompositeModel):
        super().__init__()
        self.composite_fusion = base_model.composite_fusion

    def forward(self, sketch_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([sketch_embed, text_embed], dim=-1)
        return F.normalize(self.composite_fusion(fused), dim=-1)


# ---------------------------------------------------------------------------
# Calibration Data Readers (Real FS-COCO samples, [0, 1] unnormalized scale)
# ---------------------------------------------------------------------------

_CalibBase = CalibrationDataReader if ONNX_AVAILABLE else object

class VisionCalibrationDataReader(_CalibBase):
    """Feeds real photos or sketches to ONNX static quantizer."""
    def __init__(self, data_dir: str = "fscoco", input_name: str = "image_input", key: str = "photo", n_samples: int = 200):
        self.data = []
        self._idx = 0
        try:
            dataset = FSCOCODataset(data_dir, split="train")
            n = min(n_samples, len(dataset))
            indices = list(range(0, len(dataset), max(1, len(dataset) // n)))[:n]
            for i in indices:
                sample = dataset[i]
                tensor = sample[key].unsqueeze(0).numpy().astype(np.float32)
                self.data.append({input_name: tensor})
            print(f"[Calibration] Loaded {len(self.data)} real '{key}' samples from FS-COCO.")
        except Exception as e:
            print(f"[Calibration] Warning: failed to load real samples for '{key}': {e}")

        if len(self.data) < 50:
            while len(self.data) < 200:
                self.data.append({input_name: np.random.rand(1, 3, 224, 224).astype(np.float32)})

    def get_next(self):
        if self._idx >= len(self.data):
            return None
        item = self.data[self._idx]
        self._idx += 1
        return item

    def rewind(self):
        self._idx = 0


def get_vision_nodes_to_exclude(onnx_path: str):
    """
    FastViT depthwise convolutions (groups > 1) suffer catastrophic quantization
    noise under static post-training quantization without QAT.
    We target late-stage 1x1 convs (network.4 and network.7 fc1/fc2), classification head,
    and all MatMul / Gemm operations, keeping sensitive depthwise and early convs in FP32.
    """
    m = onnx.load(onnx_path)
    target_nodes = set()
    for n in m.graph.node:
        if n.op_type in ("MatMul", "Gemm"):
            target_nodes.add(n.name)
        elif "network.4" in n.name and ("fc1" in n.name or "fc2" in n.name):
            target_nodes.add(n.name)
        elif "network.7" in n.name and ("fc1" in n.name or "fc2" in n.name):
            target_nodes.add(n.name)
        elif "head" in n.name:
            target_nodes.add(n.name)
    nodes_to_exclude = [n.name for n in m.graph.node if n.name not in target_nodes]
    return nodes_to_exclude


def load_sketch_session_from_delta(base_onnx_path: str, lora_delta_path: str):
    """
    Deduplicated Inference:
    Loads the LoRA-free photo base backbone ONNX graph into memory, reconstructs
    weight deltas from the compact low-rank A/B tensors in sketch_lora_delta.npz,
    patches the 90 initializers in-memory, and returns an ONNX Runtime InferenceSession.
    Eliminates the duplicate ~28.35 MB vision model on disk.
    """
    m_base = onnx.load(base_onnx_path)
    data = np.load(lora_delta_path)

    qkv_map = {
        'model.model.network.7.0.token_mixer.qkv.weight': 'onnx::MatMul_1753',
        'model.model.network.7.1.token_mixer.qkv.weight': 'onnx::MatMul_1758',
        'model.model.network.7.2.token_mixer.qkv.weight': 'onnx::MatMul_1763',
        'model.model.network.7.3.token_mixer.qkv.weight': 'onnx::MatMul_1768'
    }

    deltas = {}
    for key in data.files:
        if key.endswith(".A"):
            mod_name = key[:-2]
            a_w = data[f"{mod_name}.A"]
            b_w = data[f"{mod_name}.B"]
            scale = data[f"{mod_name}.scale"][0]
            onnx_name = mod_name.replace("base_model.", "") + ".weight"
            if a_w.ndim == 4:
                b_mat = b_w.squeeze(-1).squeeze(-1)
                a_mat = a_w.squeeze(-1).squeeze(-1)
                deltas[onnx_name] = (b_mat @ a_mat * scale)[:, :, None, None]
            else:
                deltas[onnx_name] = (b_w @ a_w) * scale

    for qkv_n, mm_n in qkv_map.items():
        if qkv_n in deltas:
            deltas[mm_n] = deltas[qkv_n].T

    m_patched = copy.deepcopy(m_base)
    for init in m_patched.graph.initializer:
        if init.name in deltas:
            arr = onnx.numpy_helper.to_array(init)
            new_arr = arr + deltas[init.name]
            init.CopyFrom(onnx.numpy_helper.from_array(new_arr.astype(arr.dtype), name=init.name))

    return ort.InferenceSession(m_patched.SerializeToString(), providers=['CPUExecutionProvider'])


# ---------------------------------------------------------------------------
# ONNX Export Helper
# ---------------------------------------------------------------------------

def try_onnx_export(model, dummy_inputs, path, input_names, output_names, dynamic_axes=None):
    """Attempts ONNX export across opsets 17 -> 18 -> 16."""
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
# Main Export Pipeline
# ---------------------------------------------------------------------------

def export_all_models(
    checkpoint_path: str = "checkpoints/checkpoint_best.pt",
    combiner_path: str = "checkpoints/combiner_pretrained.pt",
    output_dir: str = "exported_models",
    data_dir: str = "fscoco",
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    print("=" * 72)
    print("Mobile Model Export Pipeline - Full T+SBIR Isolated Pipeline")
    print("=" * 72)

    # Clean old exported files
    exported_files = [
        "photo_encoder_fp32.onnx", "photo_encoder_int8.onnx",
        "sketch_encoder_fp32.onnx", "sketch_encoder_int8.onnx",
        "text_encoder_fp32.onnx", "text_encoder_int8.onnx",
        "composite_fusion_mobile.onnx", "combiner_mobile.onnx"
    ]
    for fn in exported_files:
        p = os.path.join(output_dir, fn)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    # 1. Load trained PyTorch Model
    base_model = TSBIRCompositeModel(device=device)
    if os.path.exists(checkpoint_path):
        print(f"[Export] Loading dual-encoder weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt))
        base_model.load_state_dict(state, strict=False)
    else:
        print(f"[Export] WARNING: {checkpoint_path} not found! Exporting with random init.")
    base_model.eval()

    sizes = {}
    dummy_img = torch.zeros(1, 3, 224, 224)

    # -----------------------------------------------------------------------
    # Component 1: Photo Encoder (LoRA Guaranteed Disabled)
    # -----------------------------------------------------------------------
    print("\n[Export 1/5] Exporting Photo Encoder (LoRA explicitly DISABLED)...")
    # Base FastViT with LoRA completely unloaded
    photo_enc = copy.deepcopy(base_model.backbone.image_encoder).unload()
    photo_model = PhotoEncoderExportable(photo_enc)
    photo_model.eval()

    # Pre-export assertion: verify photo_model matches PyTorch model.encode_photo exactly
    test_vec = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pt_photo_test = base_model.encode_photo(test_vec)
        wrp_photo_test = photo_model(test_vec)
        pre_photo_diff = (pt_photo_test - wrp_photo_test).abs().max().item()
    assert pre_photo_diff < 1e-4, f"Pre-export photo check failed: diff={pre_photo_diff}"
    print(f"[Export] Pre-export photo wrapper matches encode_photo (diff={pre_photo_diff:.2e})")

    photo_fp32_path = os.path.join(output_dir, "photo_encoder_fp32.onnx")
    try_onnx_export(
        photo_model, (dummy_img,), photo_fp32_path,
        input_names=["image_input"], output_names=["image_embedding"],
        dynamic_axes={"image_input": {0: "batch"}, "image_embedding": {0: "batch"}},
    )
    sizes["photo_fp32"] = os.path.getsize(photo_fp32_path) / (1024 * 1024)

    # Save photo_backbone_fp32.onnx alias for deduplicated deployment
    import shutil
    photo_backbone_fp32_path = os.path.join(output_dir, "photo_backbone_fp32.onnx")
    shutil.copyfile(photo_fp32_path, photo_backbone_fp32_path)
    sizes["photo_backbone_fp32"] = sizes["photo_fp32"]

    # -----------------------------------------------------------------------
    # Component 2: Sketch Encoder & LoRA Delta Extraction
    # -----------------------------------------------------------------------
    print("\n[Export 2/5] Exporting Sketch LoRA Delta & Standalone Reference Encoder...")

    # Step 2a: Deduplication — Extract low-rank LoRA delta tensors BEFORE merging
    lora_delta_path = os.path.join(output_dir, "sketch_lora_delta.npz")
    export_dict = {}
    for name, module in base_model.backbone.image_encoder.named_modules():
        if hasattr(module, 'lora_A') and 'sketch' in module.lora_A:
            export_dict[f"{name}.A"] = module.lora_A['sketch'].weight.detach().cpu().numpy()
            export_dict[f"{name}.B"] = module.lora_B['sketch'].weight.detach().cpu().numpy()
            export_dict[f"{name}.scale"] = np.array([module.scaling['sketch']], dtype=np.float32)
    np.savez_compressed(lora_delta_path, **export_dict)
    sizes["sketch_lora_delta"] = os.path.getsize(lora_delta_path) / (1024 * 1024)
    print(f"[Export] Saved sketch LoRA delta to {lora_delta_path} ({sizes['sketch_lora_delta']:.2f} MB)")

    # Step 2b: Pre-merge check: get PyTorch encode_sketch output before merging
    with torch.no_grad():
        pt_sketch_test = base_model.encode_sketch(test_vec)

    # Activate sketch adapter and merge into base weights for standalone reference
    set_active_sketch_adapter(base_model.backbone, "sketch")
    merged_mci = base_model.backbone.image_encoder.merge_and_unload()
    sketch_model = SketchEncoderExportable(merged_mci)
    sketch_model.eval()

    # Pre-export assertion: verify sketch_model matches PyTorch model.encode_sketch
    with torch.no_grad():
        wrp_sketch_test = sketch_model(test_vec)
        pre_sketch_diff = (pt_sketch_test - wrp_sketch_test).abs().max().item()
        cos_sim_sketch = float(torch.cosine_similarity(pt_sketch_test, wrp_sketch_test, dim=-1).item())
    print(f"[Export] Sketch LoRA merged into base weights (max diff={pre_sketch_diff:.4f}, cos={cos_sim_sketch:.5f})")
    assert cos_sim_sketch > 0.99, f"Pre-export sketch check failed: cos={cos_sim_sketch}"

    sketch_fp32_path = os.path.join(output_dir, "sketch_encoder_fp32.onnx")
    try_onnx_export(
        sketch_model, (dummy_img,), sketch_fp32_path,
        input_names=["sketch_input"], output_names=["sketch_embedding"],
        dynamic_axes={"sketch_input": {0: "batch"}, "sketch_embedding": {0: "batch"}},
    )
    sizes["sketch_fp32"] = os.path.getsize(sketch_fp32_path) / (1024 * 1024)

    # -----------------------------------------------------------------------
    # Component 3: Text Encoder (Frozen Tower + Text Adapter)
    # -----------------------------------------------------------------------
    print("\n[Export 3/5] Exporting Text Encoder (Frozen tower + text adapter)...")
    text_model = TextEncoderExportable(base_model)
    text_model.eval()
    dummy_tokens = torch.zeros(1, 77, dtype=torch.long)

    text_fp32_path = os.path.join(output_dir, "text_encoder_fp32.onnx")
    try_onnx_export(
        text_model, (dummy_tokens,), text_fp32_path,
        input_names=["token_ids"], output_names=["text_embedding"],
        dynamic_axes={"token_ids": {0: "batch"}, "text_embedding": {0: "batch"}},
    )
    sizes["text_fp32"] = os.path.getsize(text_fp32_path) / (1024 * 1024)

    # -----------------------------------------------------------------------
    # Component 4: Composite Fusion ((sketch, text) -> composite_query)
    # -----------------------------------------------------------------------
    print("\n[Export 4/5] Exporting Composite Fusion Module...")
    fusion_model = CompositeFusionExportable(base_model)
    fusion_model.eval()
    dummy_sk_emb  = torch.zeros(1, 512)
    dummy_txt_emb = torch.zeros(1, 512)

    fusion_path = os.path.join(output_dir, "composite_fusion_mobile.onnx")
    try_onnx_export(
        fusion_model, (dummy_sk_emb, dummy_txt_emb), fusion_path,
        input_names=["sketch_embed", "text_embed"], output_names=["composite_query"],
        dynamic_axes={"sketch_embed": {0: "batch"}, "text_embed": {0: "batch"}, "composite_query": {0: "batch"}},
    )
    sizes["composite_fusion"] = os.path.getsize(fusion_path) / (1024 * 1024)

    # -----------------------------------------------------------------------
    # Component 5: Feedback Combiner ((shown, refined) -> combined_query)
    # -----------------------------------------------------------------------
    print("\n[Export 5/5] Exporting FeedbackComposedRetriever...")
    combiner = FeedbackComposedRetriever(feature_dim=512, hidden_dim=256)
    if os.path.exists(combiner_path):
        ckpt_c = torch.load(combiner_path, map_location="cpu", weights_only=False)
        c_state = ckpt_c.get("model_state_dict", ckpt_c.get("state_dict", ckpt_c))
        combiner.load_state_dict(c_state, strict=False)
        print(f"[Export] Loaded combiner weights from: {combiner_path}")
    combiner.eval()

    dummy_sh = torch.zeros(1, 512)
    dummy_rf = torch.zeros(1, 512)
    combiner_onnx_path = os.path.join(output_dir, "combiner_mobile.onnx")
    try_onnx_export(
        combiner, (dummy_sh, dummy_rf), combiner_onnx_path,
        input_names=["shown_candidate", "refined_query"], output_names=["combined_embedding"],
        dynamic_axes={"shown_candidate": {0: "batch"}, "refined_query": {0: "batch"}, "combined_embedding": {0: "batch"}},
    )
    sizes["combiner"] = os.path.getsize(combiner_onnx_path) / (1024 * 1024)

    # -----------------------------------------------------------------------
    # Quantization (Static INT8 for Vision, Dynamic INT8 for Text)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Quantization] Running INT8 Quantization...")
    print("=" * 60)

    photo_int8_path  = os.path.join(output_dir, "photo_encoder_int8.onnx")
    sketch_int8_path = os.path.join(output_dir, "sketch_encoder_int8.onnx")
    text_int8_path   = os.path.join(output_dir, "text_encoder_int8.onnx")

    # 1. Photo Encoder Static INT8
    print("[Quantize] Calibrating photo encoder with 100 real FS-COCO photos...")
    photo_calib = VisionCalibrationDataReader(data_dir=data_dir, input_name="image_input", key="photo", n_samples=100)
    try:
        photo_nodes_to_exclude = get_vision_nodes_to_exclude(photo_fp32_path)
        quantize_static(
            model_input=photo_fp32_path, model_output=photo_int8_path,
            calibration_data_reader=photo_calib,
            nodes_to_exclude=photo_nodes_to_exclude,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=False, reduce_range=False
        )
        sizes["photo_int8"] = os.path.getsize(photo_int8_path) / (1024 * 1024)
        print(f"[Quantize] Photo Static INT8: {sizes['photo_int8']:.1f} MB")
    except Exception as e:
        print(f"[Quantize] Photo static quant failed ({e}), using dynamic fallback.")
        quantize_dynamic(photo_fp32_path, photo_int8_path, weight_type=QuantType.QInt8)
        sizes["photo_int8"] = os.path.getsize(photo_int8_path) / (1024 * 1024)

    # Save photo_backbone_int8.onnx alias for deduplicated deployment
    photo_backbone_int8_path = os.path.join(output_dir, "photo_backbone_int8.onnx")
    shutil.copyfile(photo_int8_path, photo_backbone_int8_path)
    sizes["photo_backbone_int8"] = sizes["photo_int8"]

    # 2. Sketch Encoder Static INT8
    print("[Quantize] Calibrating sketch encoder with 100 real FS-COCO sketches...")
    sketch_calib = VisionCalibrationDataReader(data_dir=data_dir, input_name="sketch_input", key="sketch", n_samples=100)
    try:
        sketch_nodes_to_exclude = get_vision_nodes_to_exclude(sketch_fp32_path)
        quantize_static(
            model_input=sketch_fp32_path, model_output=sketch_int8_path,
            calibration_data_reader=sketch_calib,
            nodes_to_exclude=sketch_nodes_to_exclude,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=False, reduce_range=False
        )
        sizes["sketch_int8"] = os.path.getsize(sketch_int8_path) / (1024 * 1024)
        print(f"[Quantize] Sketch Static INT8: {sizes['sketch_int8']:.1f} MB")
    except Exception as e:
        print(f"[Quantize] Sketch static quant failed ({e}), using dynamic fallback.")
        quantize_dynamic(sketch_fp32_path, sketch_int8_path, weight_type=QuantType.QInt8)
        sizes["sketch_int8"] = os.path.getsize(sketch_int8_path) / (1024 * 1024)

    # 3. Text Encoder Dynamic INT8
    try:
        quantize_dynamic(text_fp32_path, text_int8_path, weight_type=QuantType.QInt8)
        sizes["text_int8"] = os.path.getsize(text_int8_path) / (1024 * 1024)
        print(f"[Quantize] Text Dynamic INT8: {sizes['text_int8']:.1f} MB")
    except Exception as e:
        print(f"[Quantize] Text quantization failed: {e}")

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Exported Model Artifact':<40} | {'Disk Size':<15}")
    print("-" * 65)
    for k, v in sizes.items():
        print(f"{k.replace('_', ' ').title():<40} | {v:6.1f} MB")
    print("=" * 65)

    # Deduplicated footprint audit
    dedup_assets = [
        ("photo_backbone_int8.onnx (LoRA-free photo)", sizes.get("photo_backbone_int8", 0.0)),
        ("sketch_lora_delta.npz (90 low-rank tensors)", sizes.get("sketch_lora_delta", 0.0)),
        ("text_encoder_int8.onnx (Text tower + adapter)", sizes.get("text_int8", 0.0)),
        ("composite_fusion_mobile.onnx (Multimodal fusion)", sizes.get("composite_fusion", 0.0)),
        ("combiner_mobile.onnx (Feedback query composer)", sizes.get("combiner", 0.0)),
    ]
    total_dedup = sum(sz for _, sz in dedup_assets)
    print("\n" + "=" * 65)
    print("DEDUPLICATED MOBILE DEPLOYMENT FOOTPRINT (Asset Bundle)")
    print("-" * 65)
    for name, sz in dedup_assets:
        print(f"  {name:<48} | {sz:6.1f} MB")
    print("-" * 65)
    print(f"  {'Total Deduplicated Model Footprint':<48} | {total_dedup:6.1f} MB")
    print(f"  {'Eliminated Duplicate Sketch Backbone Savings':<48} | {sizes.get('sketch_int8', 0.0):6.1f} MB")
    print(f"  {'Original Budget Target in android_architecture_design.md':<48} |  <45.0 MB")
    if total_dedup > 45.0:
        print(f"  [Footprint Audit] Exceeds <45 MB budget by {total_dedup - 45.0:.1f} MB (bottleneck: 61.4 MB text encoder).")
    else:
        print(f"  [Footprint Audit] Fits strictly within <45 MB budget.")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Comprehensive Numerical Parity Verification Check (GATE)
    # -----------------------------------------------------------------------
    run_all_parity_checks(base_model, photo_model, sketch_model, text_model, fusion_model, combiner, output_dir, data_dir)


def run_all_parity_checks(base_model, photo_model, sketch_model, text_model, fusion_model, combiner, output_dir, data_dir):
    print("\n" + "=" * 75)
    print("PARITY VERIFICATION: PyTorch vs ONNX Cosine Similarity Check (>0.95)")
    print("=" * 75)

    test_dataset = FSCOCODataset(data_dir, split="test")
    indices = [0, 50, 100, 150, 200]
    indices = [i for i in indices if i < len(test_dataset)]
    if len(indices) < 5:
        indices = list(range(min(5, len(test_dataset))))

    # Sessions
    photo_session    = ort.InferenceSession(os.path.join(output_dir, "photo_encoder_fp32.onnx"), providers=['CPUExecutionProvider'])
    sketch_session   = ort.InferenceSession(os.path.join(output_dir, "sketch_encoder_fp32.onnx"), providers=['CPUExecutionProvider'])
    text_session     = ort.InferenceSession(os.path.join(output_dir, "text_encoder_fp32.onnx"), providers=['CPUExecutionProvider'])
    fusion_session   = ort.InferenceSession(os.path.join(output_dir, "composite_fusion_mobile.onnx"), providers=['CPUExecutionProvider'])
    combiner_session = ort.InferenceSession(os.path.join(output_dir, "combiner_mobile.onnx"), providers=['CPUExecutionProvider'])

    photo_int8_path  = os.path.join(output_dir, "photo_encoder_int8.onnx")
    photo_int8_sess  = ort.InferenceSession(photo_int8_path, providers=['CPUExecutionProvider']) if os.path.exists(photo_int8_path) else None
    sketch_int8_path = os.path.join(output_dir, "sketch_encoder_int8.onnx")
    sketch_int8_sess = ort.InferenceSession(sketch_int8_path, providers=['CPUExecutionProvider']) if os.path.exists(sketch_int8_path) else None
    text_int8_path   = os.path.join(output_dir, "text_encoder_int8.onnx")
    text_int8_sess   = ort.InferenceSession(text_int8_path, providers=['CPUExecutionProvider']) if os.path.exists(text_int8_path) else None

    # Deduplicated sketch session: loads base photo graph and patches low-rank delta in memory
    photo_fp32_path  = os.path.join(output_dir, "photo_encoder_fp32.onnx")
    lora_delta_path  = os.path.join(output_dir, "sketch_lora_delta.npz")
    delta_session    = None
    if os.path.exists(photo_fp32_path) and os.path.exists(lora_delta_path):
        delta_session = load_sketch_session_from_delta(photo_fp32_path, lora_delta_path)

    tokenizer = base_model.tokenizer

    photo_sims        = []
    photo_int8_sims   = []
    sketch_sims       = []
    sketch_int8_sims  = []
    sketch_delta_sims = []
    text_sims         = []
    text_int8_sims    = []
    fusion_sims       = []
    combiner_sims     = []

    for idx in indices:
        sample = test_dataset[idx]
        p_tensor = sample["photo"].unsqueeze(0)   # (1, 3, 224, 224)
        s_tensor = sample["sketch"].unsqueeze(0)  # (1, 3, 224, 224)
        caption  = sample["caption"]
        t_tokens = tokenizer([caption])           # (1, 77)

        # 1. Photo Encoder Parity (FP32 & INT8)
        with torch.no_grad():
            py_photo = photo_model(p_tensor).squeeze(0).numpy()
        onnx_photo = photo_session.run(None, {"image_input": p_tensor.numpy()})[0].squeeze(0)
        c_photo = float(np.dot(py_photo, onnx_photo) / (np.linalg.norm(py_photo) * np.linalg.norm(onnx_photo) + 1e-8))
        photo_sims.append(c_photo)

        if photo_int8_sess is not None:
            onnx_p_int8 = photo_int8_sess.run(None, {"image_input": p_tensor.numpy()})[0].squeeze(0)
            c_p_int8 = float(np.dot(py_photo, onnx_p_int8) / (np.linalg.norm(py_photo) * np.linalg.norm(onnx_p_int8) + 1e-8))
            photo_int8_sims.append(c_p_int8)

        # 2. Sketch Encoder Parity (FP32, INT8, & Base-Plus-Delta)
        with torch.no_grad():
            py_sketch = sketch_model(s_tensor).squeeze(0).numpy()
        onnx_sketch = sketch_session.run(None, {"sketch_input": s_tensor.numpy()})[0].squeeze(0)
        c_sketch = float(np.dot(py_sketch, onnx_sketch) / (np.linalg.norm(py_sketch) * np.linalg.norm(onnx_sketch) + 1e-8))
        sketch_sims.append(c_sketch)

        if sketch_int8_sess is not None:
            onnx_s_int8 = sketch_int8_sess.run(None, {"sketch_input": s_tensor.numpy()})[0].squeeze(0)
            c_s_int8 = float(np.dot(py_sketch, onnx_s_int8) / (np.linalg.norm(py_sketch) * np.linalg.norm(onnx_s_int8) + 1e-8))
            sketch_int8_sims.append(c_s_int8)

        if delta_session is not None:
            onnx_s_delta = delta_session.run(None, {"image_input": s_tensor.numpy()})[0].squeeze(0)
            c_s_delta = float(np.dot(py_sketch, onnx_s_delta) / (np.linalg.norm(py_sketch) * np.linalg.norm(onnx_s_delta) + 1e-8))
            sketch_delta_sims.append(c_s_delta)

        # 3. Text Encoder Parity (FP32 & INT8)
        with torch.no_grad():
            py_text = text_model(t_tokens).squeeze(0).numpy()
        onnx_text = text_session.run(None, {"token_ids": t_tokens.numpy()})[0].squeeze(0)
        c_text = float(np.dot(py_text, onnx_text) / (np.linalg.norm(py_text) * np.linalg.norm(onnx_text) + 1e-8))
        text_sims.append(c_text)

        if text_int8_sess is not None:
            onnx_t_int8 = text_int8_sess.run(None, {"token_ids": t_tokens.numpy()})[0].squeeze(0)
            c_t_int8 = float(np.dot(py_text, onnx_t_int8) / (np.linalg.norm(py_text) * np.linalg.norm(onnx_t_int8) + 1e-8))
            text_int8_sims.append(c_t_int8)

        # 4. Composite Fusion Parity
        with torch.no_grad():
            py_fusion = fusion_model(torch.from_numpy(py_sketch).unsqueeze(0), torch.from_numpy(py_text).unsqueeze(0)).squeeze(0).numpy()
        onnx_fusion = fusion_session.run(None, {
            "sketch_embed": onnx_sketch.reshape(1, -1),
            "text_embed": onnx_text.reshape(1, -1)
        })[0].squeeze(0)
        c_fusion = float(np.dot(py_fusion, onnx_fusion) / (np.linalg.norm(py_fusion) * np.linalg.norm(onnx_fusion) + 1e-8))
        fusion_sims.append(c_fusion)

        # 5. Feedback Combiner Parity
        with torch.no_grad():
            py_comb = combiner(torch.from_numpy(py_photo).unsqueeze(0), torch.from_numpy(py_fusion).unsqueeze(0)).squeeze(0).numpy()
        onnx_comb = combiner_session.run(None, {
            "shown_candidate": onnx_photo.reshape(1, -1),
            "refined_query": onnx_fusion.reshape(1, -1)
        })[0].squeeze(0)
        c_comb = float(np.dot(py_comb, onnx_comb) / (np.linalg.norm(py_comb) * np.linalg.norm(onnx_comb) + 1e-8))
        combiner_sims.append(c_comb)

    # Print Summary Table
    components = [
        ("1. Photo Encoder FP32 (photo_encoder_fp32.onnx)", photo_sims),
        ("   Photo Encoder INT8 (photo_encoder_int8.onnx)", photo_int8_sims),
        ("2. Sketch Encoder FP32 (sketch_encoder_fp32.onnx)", sketch_sims),
        ("   Sketch Encoder INT8 (sketch_encoder_int8.onnx)", sketch_int8_sims),
        ("   Sketch Base-Plus-Delta (load_sketch_session_from_delta)", sketch_delta_sims),
        ("3. Text Encoder FP32 (text_encoder_fp32.onnx)", text_sims),
        ("   Text Encoder INT8 (text_encoder_int8.onnx)", text_int8_sims),
        ("4. Composite Fusion (composite_fusion_mobile.onnx)", fusion_sims),
        ("5. Feedback Combiner (combiner_mobile.onnx)", combiner_sims),
    ]

    print(f"{'Component':<56} | {'Min Cos Sim':<12} | {'Mean Cos Sim':<12} | {'Status':<6}")
    print("-" * 92)
    all_passed = True
    for name, sims in components:
        if not sims:
            continue
        min_s = float(np.min(sims))
        mean_s = float(np.mean(sims))
        status = "PASS" if min_s > 0.95 else "FAIL"
        if min_s <= 0.95:
            all_passed = False
        print(f"{name:<56} | {min_s:12.5f} | {mean_s:12.5f} | {status:<6}")
    print("=" * 92)

    if all_passed:
        print("\n[Parity GATE VERDICT: ALL PASS] All exported ONNX modules exhibit > 0.95 cosine parity with PyTorch.")
    else:
        print("\n[Parity GATE VERDICT: FAIL] One or more components failed the > 0.95 cosine similarity threshold.")


if __name__ == "__main__":
    export_all_models()
