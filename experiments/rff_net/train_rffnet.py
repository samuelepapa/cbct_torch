import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader
import wandb

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

# Add parent directory to path to import modules if not installed as package
import sys
# Adjust path: current file is in experiments/instant_ngp_torch/experiments/rff_net/train_rffnet.py
# We need to reach experiments/instant_ngp_torch for imports like dataset.load_slices
# Parent -> experiments/rff_net
# Grandparent -> experiments
# Great-Grandparent -> instant_ngp_torch
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.load_slices import get_slice_dataloader
from models.rff_net import RFFNet

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", 
    "configs/rffnet_config.py", 
    "File path to the training hyperparameter configuration.", 
    lock_config=True
)

def train(argv):
    config = FLAGS.config
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Experiment root
    experiment_root = Path(config.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    # Init WandB
    wandb.init(project=config.wandb.project_name, config=config, dir=str(experiment_root))

    # Create specific experiment directory
    experiment_dir = experiment_root / wandb.run.id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    # Data Loading - SINGLE SLICE MODE
    print(f"Initializing Datasets for Single Slice {config.slice_idx}...")
    
    full_dataset = get_slice_dataloader(
        root_dir=config.root_dir,
        stage="train",
        num_vols=config.num_vols,
        batch_size=1,
        num_workers=0, # Debug
        shuffle=False,
        gray_value_scaling=config.gray_value_scaling
    ).dataset
    
    if config.slice_idx >= len(full_dataset):
        print(f"Warning: slice_idx {config.slice_idx} out of range (max {len(full_dataset)-1}). using 0.")
        target_slice_idx = 0
    else:
        target_slice_idx = config.slice_idx
        
    print(f"Selecting slice {target_slice_idx}.")
    
    # Fetch the single sample
    sample = full_dataset[target_slice_idx] 
    # sample keys: 'image', 'coords', ...
    
    # Prepare single batch
    gt_image = sample["image"].unsqueeze(0).to(device)   # [1, H, W]
    coords = sample["coords"].unsqueeze(0).to(device)    # [1, H, W, 2]
    
    B, H, W = gt_image.shape
    gt_flat = gt_image.reshape(-1, 1) # [H*W, 1]
    coords_flat = coords.reshape(-1, 2) # [H*W, 2]
    
    print(f"Slice shape: {gt_image.shape}")

    # Model
    model = RFFNet(
        in_dim=2, 
        out_dim=1, 
        hidden_dim=config.hidden_dim, 
        num_layers=config.num_layers, 
        encoding_size=config.encoding_size, 
        scale=config.scale
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()

    # Training Loop
    start_time = time.time()

    model.train()
    
    # Since we are overfitting one slice, "batches" are just the same slice over and over.
    # But usually we want to count epochs as "passes over the data". 
    # Here data=1 slice. So 1 iteration = 1 epoch basically.
    # Let's just loop epochs.
    
    pbar = tqdm(range(config.epochs), desc="Training")
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(coords_flat)
        
        loss = loss_fn(pred, gt_flat)
        loss.backward()
        optimizer.step()
        
        # Log
        wandb.log({"train_loss": loss.item()})
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
        
        # Validation / Visualization
        if (epoch + 1) % config.val_interval == 0:
            avg_train_loss = loss.item()
            
            # Visualize
            with torch.no_grad():
                pred_image = pred.reshape(B, H, W)
                
                gt_np = gt_image[0].cpu().numpy()
                pred_np = pred_image[0].cpu().numpy()
                diff = np.abs(gt_np - pred_np)
                
                # Log images to wandb
                wandb.log({
                    "val_loss": avg_train_loss,
                    "prediction": wandb.Image(np.clip(pred_np, 0, 1), caption=f"Pred Ep {epoch+1}"),
                    "ground_truth": wandb.Image(np.clip(gt_np, 0, 1), caption="GT"),
                    "difference": wandb.Image(diff, caption=f"Diff Ep {epoch+1}")
                })

            # Save checkpoint
            ckpt_path = experiment_dir / "ckpt_latest.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, ckpt_path)
            
    wandb.finish()


if __name__ == "__main__":
    app.run(train)
