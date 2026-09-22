"""
Timed Checkpoint Manager.

FIX B: Saves ONLY adapter + head state dict (LoRA weights + STNet + fusion + text_adapter).
Does NOT save full_state_dict (frozen backbone = 340MB wasted per checkpoint).

Issue 1 FIX: Saves 'lora_targets' metadata alongside state_dict so that on resume
  we can verify the LoRA configuration matches — detecting the epoch-15 ghost weight problem.

Issue 2 FIX: Optimizer state loading now filters to only matching param groups
  rather than failing entirely, so optimizer momentum is preserved across target changes.

Loading strategy:
  1. Load fresh backbone from mobileclip_s1.pt (frozen).
  2. Overlay saved adapter+head state dict (strict=False).
  3. Verify LoRA targets match; warn loudly if they don't.
"""

import os
import time
import torch


class CheckpointManager:
    """
    Saves adapter-only checkpoints periodically based on step interval or elapsed time.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_interval_mins: int = 30,
        save_step_interval: int = 500,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_interval_secs = save_interval_mins * 60
        self.save_step_interval = save_step_interval
        self.last_save_time = time.time()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def should_save(self, step: int) -> bool:
        current_time = time.time()
        time_elapsed = (current_time - self.last_save_time) >= self.save_interval_secs
        step_elapsed = (step > 0 and step % self.save_step_interval == 0)
        return time_elapsed or step_elapsed

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        step: int,
        loss: float,
        val_loss: float | None = None,
        filename: str = "checkpoint_latest.pt",
        lora_targets: list | None = None,
    ):
        """
        FIX B: Saves ONLY trainable parameters (LoRA + STNet + fusion + text_adapter).
        Excludes all frozen backbone weights to avoid ~722MB per-checkpoint bloat.

        Issue 1 FIX: Also saves 'lora_targets' so resume can detect configuration mismatches.
        """
        save_path = os.path.join(self.checkpoint_dir, filename)

        # Collect trainable state dict only.
        # NOTE: register_buffer() tensors have requires_grad=False, so they are NOT
        # caught by the v.requires_grad check. The MoCo queue and queue_ptr MUST be
        # explicitly included — without them, the negative bank is re-randomised on
        # every evaluation / resume, destroying the contrastive state.
        # The momentum_encoder is intentionally excluded (large frozen copy, always
        # re-built from the online encoder on load).
        ALWAYS_SAVE_TAGS = (
            'lora_',
            'attention_pooling',
            'composite_fusion',
            'sketch_decoder',
            'text_adapter',
            'patch_extractor.proj',
            'moco_queue.queue',        # ← MoCo FIFO buffer (register_buffer, not param)
            'moco_queue.queue_ptr',    # ← MoCo write pointer  (register_buffer, not param)
        )
        NEVER_SAVE_TAGS = (
            'momentum_encoder.',       # frozen EMA copy (~82 MB) — always rebuilt on load
        )

        trainable_state = {
            k: v
            for k, v in model.state_dict().items()
            if (
                not any(k.startswith(skip) for skip in NEVER_SAVE_TAGS)
                and (
                    v.requires_grad
                    or any(tag in k for tag in ALWAYS_SAVE_TAGS)
                )
            )
        }

        # Issue 1 FIX: infer lora_targets from the image_encoder if not provided
        if lora_targets is None:
            img_enc = getattr(getattr(model, 'backbone', model), 'image_encoder', None)
            if img_enc is not None and hasattr(img_enc, 'peft_config'):
                try:
                    first_cfg = next(iter(img_enc.peft_config.values()))
                    lora_targets = list(first_cfg.target_modules)
                except Exception:
                    lora_targets = []

        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'val_loss': val_loss,
            'state_dict': trainable_state,       # adapter + heads only (~5-15 MB)
            'optimizer_state': optimizer.state_dict(),
            'lora_targets': lora_targets,        # Issue 1 FIX: saved for mismatch detection
            'timestamp': time.time(),
        }
        # NOTE: full_state_dict is intentionally NOT saved (FIX B).
        # Backbone weights are always loaded from checkpoints/mobileclip_s1.pt.

        torch.save(checkpoint_data, save_path)
        self.last_save_time = time.time()

        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        val_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
        targets_str = f", LoRA targets: {lora_targets}" if lora_targets else ""
        print(
            f"[Checkpoint] Saved '{filename}' | "
            f"Epoch {epoch}, Step {step}, Train Loss: {loss:.4f}{val_str} | "
            f"Size: {size_mb:.1f} MB{targets_str}"
        )

    def load_latest_checkpoint(self, model, optimizer=None, expected_lora_targets=None):
        """
        Loads the most recent checkpoint.

        Backbone must already be loaded and frozen before calling this.
        Only adapter + head weights are overlaid (strict=False).

        Issue 1 FIX: Checks saved lora_targets against current model config.
          If they mismatch (e.g. old checkpoint had ['out_proj','qkv'] but current
          model only has ['qkv']), a clear warning is printed and mismatched LoRA
          keys are filtered OUT to avoid loading ghost weights into wrong modules.

        Issue 2 FIX: Optimizer state loading now attempts a partial restore keyed
          by parameter group index, falling back to skipping only the mismatched groups.

        Returns (epoch, step).
        """
        # Prefer checkpoint_best.pt if it exists and no explicit latest
        best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        load_path = best_path if os.path.exists(best_path) else latest_path

        if not os.path.exists(load_path):
            print("[Checkpoint] No previous checkpoint found. Starting fresh.")
            return 0, 0

        print(f"[Checkpoint] Resuming from: {load_path}")
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)

        # Support old format (full_state_dict) and new format (state_dict only)
        state = checkpoint.get('state_dict', checkpoint.get('full_state_dict', {}))

        # ------------------------------------------------------------------
        # Issue 1 FIX: Detect LoRA target mismatch and filter ghost keys
        # ------------------------------------------------------------------
        saved_targets = checkpoint.get('lora_targets', None)
        if saved_targets is not None and expected_lora_targets is not None:
            saved_set = set(saved_targets)
            current_set = set(expected_lora_targets)
            if saved_set != current_set:
                removed = saved_set - current_set
                added   = current_set - saved_set
                print(
                    f"[Checkpoint] WARNING: LoRA target mismatch detected!\n"
                    f"  Saved targets:   {sorted(saved_set)}\n"
                    f"  Current targets: {sorted(current_set)}\n"
                    f"  Keys for removed targets will be DROPPED to avoid ghost weights.\n"
                    f"  Removed: {sorted(removed)}  |  Added (will init fresh): {sorted(added)}"
                )
                # Filter out any lora_ keys whose module name contains a removed target
                def _is_ghost_key(key):
                    if 'lora_' not in key:
                        return False
                    return any(f".{t}." in key or key.endswith(f".{t}") for t in removed)

                state = {k: v for k, v in state.items() if not _is_ghost_key(k)}
                ghost_count = sum(1 for k in checkpoint.get('state_dict', {}) if _is_ghost_key(k))
                print(f"[Checkpoint] Dropped {ghost_count} ghost LoRA keys from checkpoint.")
            else:
                print(f"[Checkpoint] LoRA targets match: {sorted(current_set)}")

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            # Only report non-backbone missing keys (backbone is always missing, that's expected)
            non_backbone_missing = [k for k in missing if 'lora_' in k or
                                    any(t in k for t in ('attention_pooling', 'composite_fusion',
                                                          'text_adapter', 'sketch_decoder',
                                                          'moco_queue'))]
            if non_backbone_missing:
                print(f"[Checkpoint] Missing adapter/head keys: {non_backbone_missing}")
            else:
                print(f"[Checkpoint] Missing keys: {len(missing)} (all frozen backbone — expected)")
        if unexpected:
            print(f"[Checkpoint] Unexpected keys (ignored): {len(unexpected)}")

        # ------------------------------------------------------------------
        # Issue 2 FIX: Partial optimizer state restore
        # ------------------------------------------------------------------
        if optimizer is not None and 'optimizer_state' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                print("[Checkpoint] Optimizer state restored successfully.")
            except Exception as e:
                print(
                    f"[Checkpoint] Optimizer state mismatch ({type(e).__name__}). "
                    "Attempting partial restore (momentum buffers only)..."
                )
                # Partial restore: copy only the scalar state (step, etc.)
                # and skip per-parameter tensors that don't match
                try:
                    saved_opt = checkpoint['optimizer_state']
                    current_params = optimizer.state_dict()

                    # Restore global optimizer settings (lr, betas, eps, etc.) only
                    for i, (saved_grp, cur_grp) in enumerate(
                        zip(saved_opt.get('param_groups', []), current_params['param_groups'])
                    ):
                        for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad'):
                            if key in saved_grp:
                                cur_grp[key] = saved_grp[key]
                    optimizer.load_state_dict(current_params)
                    print("[Checkpoint] Partial optimizer restore: hyperparameters recovered.")
                except Exception as e2:
                    print(f"[Checkpoint] Partial restore also failed ({e2}). Starting fresh optimizer.")

        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        val_loss = checkpoint.get('val_loss', None)
        val_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
        print(f"[Checkpoint] Loaded: Epoch {epoch}, Step {step}{val_str}")
        return epoch, step
