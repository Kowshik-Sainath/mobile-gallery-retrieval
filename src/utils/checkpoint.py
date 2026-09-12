"""
Timed Checkpoint Manager for Kaggle & Edge Training Sessions.
"""

import os
import time
import torch

class CheckpointManager:
    """
    Saves model checkpoints periodically based on step intervals or elapsed time limits.
    """
    def __init__(self, checkpoint_dir="checkpoints", save_interval_mins=30, save_step_interval=500):
        self.checkpoint_dir = checkpoint_dir
        self.save_interval_secs = save_interval_mins * 60
        self.save_step_interval = save_step_interval
        self.last_save_time = time.time()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def should_save(self, step):
        current_time = time.time()
        time_elapsed = (current_time - self.last_save_time) >= self.save_interval_secs
        step_elapsed = (step > 0 and step % self.save_step_interval == 0)
        return time_elapsed or step_elapsed

    def save_checkpoint(self, model, optimizer, epoch, step, loss, filename="checkpoint_latest.pt"):
        save_path = os.path.join(self.checkpoint_dir, filename)
        # Extract trainable state dict
        trainable_state = {k: v for k, v in model.state_dict().items() if 'backbone' not in k or 'lora' in k.lower()}
        
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'state_dict': trainable_state,
            'full_state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'timestamp': time.time()
        }
        
        torch.save(checkpoint_data, save_path)
        self.last_save_time = time.time()
        print(f"Checkpoint successfully saved to '{save_path}' at step {step} (Epoch {epoch}, Loss: {loss:.4f})")

    def load_latest_checkpoint(self, model, optimizer=None):
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        
        # Check for highest epoch file if checkpoint_latest is not present or older
        epoch_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
        if epoch_files:
            epoch_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            highest_epoch_file = os.path.join(self.checkpoint_dir, epoch_files[-1])
            if not os.path.exists(latest_path) or os.path.getmtime(highest_epoch_file) > os.path.getmtime(latest_path):
                latest_path = highest_epoch_file

        if os.path.exists(latest_path):
            print(f"Resuming training from checkpoint: {latest_path}")
            checkpoint = torch.load(latest_path, map_location='cpu')
            model.load_state_dict(checkpoint['full_state_dict'], strict=False)
            if optimizer is not None and 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            epoch = checkpoint.get('epoch', 0)
            step = checkpoint.get('step', 0)
            print(f"Loaded checkpoint state: Epoch {epoch}, Global Step {step}")
            return epoch, step
        else:
            print("No previous checkpoint found. Starting fresh training session.")
            return 0, 0
