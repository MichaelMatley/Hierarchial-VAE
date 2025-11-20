"""
Checkpointing utilities.
"""

import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        checkpoint_dir (str): Directory to save checkpoints
        is_best (bool): Whether this is the best model
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    # Save latest
    latest_path = checkpoint_dir / 'latest.pth'
    torch.save(checkpoint, latest_path)
    
    # Save best
    if is_best:
        best_path = checkpoint_dir / 'best.pth'
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint
        model: Model to load into
        optimizer: Optional optimizer to load state into
        device: Device to load to
        
    Returns:
        int: Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']