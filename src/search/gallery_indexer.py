"""
Gallery Indexer — Phase 4 Background Indexing Pipeline.

Asynchronously pre-computes and persists all gallery photo embeddings and
patch tokens to disk. This decouples the heavy vision forward passes from
query-time search, enabling instantaneous retrieval.

Two output files:
  gallery_embeddings.npz  — (N, 512) float32 global photo embeddings
  gallery_patches.npz     — (N, 49, 512) float16 patch token matrices

The float16 patch tokens save ~50% storage vs float32.
At N=10,000 images: 10k × 49 × 512 × 2 bytes ≈ 500 MB (manageable).

Usage (offline, on workstation):
    python -m src.search.gallery_indexer \\
        --data_dir fscoco \\
        --checkpoint checkpoints/checkpoint_best.pt \\
        --out_dir gallery_index \\
        --batch_size 512

Reference architecture: fguzman82/CLIP-Finder2 batch processing logic.
"""

import os
import argparse
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.fscoco_dataset import FSCOCODataset
from src.models.composite_model import TSBIRCompositeModel


def build_gallery_index(
    model: TSBIRCompositeModel,
    dataset,
    out_dir: str,
    batch_size: int = 512,
    device: str = "cuda",
    save_patches: bool = True,
) -> dict:
    """
    Pre-computes photo embeddings (and optionally patch tokens) for the
    entire gallery and saves them to disk as compressed numpy arrays.

    Args:
        model:        Loaded and eval-mode TSBIRCompositeModel.
        dataset:      FSCOCODataset instance (any split — typically 'train'
                      for the full gallery, or 'test' for evaluation index).
        out_dir:      Directory to write output .npz files.
        batch_size:   Number of photos to process per GPU forward pass.
                      512 safely fits in 6 GB VRAM with MobileCLIP-S1.
        device:       'cuda' or 'cpu'.
        save_patches: If True, also saves (N, 49, 512) float16 patch tokens.
                      Set False on memory-constrained devices.

    Returns:
        dict with 'embeddings_path' and optionally 'patches_path'.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    all_embeddings = []
    all_patches    = [] if save_patches else None
    all_file_paths = []

    print(f"[GalleryIndexer] Indexing {len(dataset)} photos in batches of {batch_size}...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Indexing gallery"):
            photo = batch["photo"].to(device)
            file_paths = batch.get("file_path", [""] * photo.shape[0])

            # Global photo embedding (LoRA OFF, frozen base weights)
            e_photo = model.encode_photo(photo)                # (B, D) float32
            all_embeddings.append(e_photo.cpu())

            if save_patches:
                # Patch tokens (B, N, D) — hook captures after conv_exp
                with model._disable_adapter_ctx():
                    _ = model.backbone.encode_image(photo)
                patches = model.patch_extractor.extract()      # (B, N, D)
                patches = F.normalize(patches, dim=-1)
                # Save as float16 to halve storage
                all_patches.append(patches.cpu().half())

            all_file_paths.extend(
                fp if isinstance(fp, str) else fp for fp in file_paths
            )

    # Stack into single arrays
    embeddings_np = torch.cat(all_embeddings, dim=0).numpy()   # (N, D) float32
    N, D = embeddings_np.shape
    print(f"[GalleryIndexer] Stacked embeddings: {embeddings_np.shape} ({embeddings_np.nbytes/1e6:.1f} MB)")

    # Save global embeddings
    emb_path = os.path.join(out_dir, "gallery_embeddings.npz")
    np.savez_compressed(emb_path, embeddings=embeddings_np)
    print(f"[GalleryIndexer] Saved: {emb_path}")

    result = {"embeddings_path": emb_path, "n_images": N}

    # Save patch tokens
    if save_patches and all_patches:
        patches_np = torch.cat(all_patches, dim=0).numpy()    # (N, 49, D) float16
        patch_path = os.path.join(out_dir, "gallery_patches.npz")
        np.savez_compressed(patch_path, patches=patches_np)
        patch_mb = patches_np.nbytes / 1e6
        print(f"[GalleryIndexer] Saved patches: {patch_path} ({patch_mb:.1f} MB float16)")
        result["patches_path"] = patch_path

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute and save gallery embeddings + patch tokens."
    )
    parser.add_argument("--data_dir",    type=str,  default="fscoco")
    parser.add_argument("--split",       type=str,  default="test",
                        help="Dataset split to index ('train' for full gallery, 'test' for eval)")
    parser.add_argument("--checkpoint",  type=str,  default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--out_dir",     type=str,  default="gallery_index")
    parser.add_argument("--batch_size",  type=int,  default=512)
    parser.add_argument("--no_patches",  action="store_true",
                        help="Skip patch token extraction (saves time/memory)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GalleryIndexer] Device: {device}")

    # Load model
    model = TSBIRCompositeModel(device=device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[GalleryIndexer] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt))
        model.load_state_dict(state, strict=False)
    else:
        print(f"[GalleryIndexer] WARNING: checkpoint not found at '{args.checkpoint}'. Using random weights.")
    model.to(device)

    # Dataset
    dataset = FSCOCODataset(args.data_dir, split=args.split)
    print(f"[GalleryIndexer] Split '{args.split}': {len(dataset)} images")

    result = build_gallery_index(
        model=model,
        dataset=dataset,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        device=device,
        save_patches=not args.no_patches,
    )

    print(f"\n[GalleryIndexer] Done. Indexed {result['n_images']} images.")
    print(f"  Global embeddings : {result['embeddings_path']}")
    if "patches_path" in result:
        print(f"  Patch tokens      : {result['patches_path']}")
    print(f"\nNext step: build the HNSW search index:")
    print(f"  python -m src.search.hnsw_index --gallery_dir {args.out_dir}")


if __name__ == "__main__":
    main()
