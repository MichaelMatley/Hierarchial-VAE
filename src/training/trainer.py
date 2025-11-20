"""
Main training loop for Hierarchical VAE.

Handles training, validation, early stopping, checkpointing,
and comprehensive metric logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path

from .losses import vae_loss
from .schedulers import BetaScheduler


class VAETrainer:
    """
    Trainer class for Hierarchical VAE.
    
    Manages the complete training pipeline including:
        - Forward/backward passes
        - Loss computation with β-annealing
        - Learning rate scheduling
        - Early stopping
        - Checkpointing
        - Metric tracking
    
    Args:
        model (nn.Module): Hierarchical VAE model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (str): Device to train on ('cuda' or 'cpu')
        lr (float): Initial learning rate
        weight_decay (float): L2 regularization strength
        beta_scheduler (BetaScheduler): β-annealing scheduler
        patience (int): Early stopping patience
        checkpoint_dir (str): Directory for saving checkpoints
        
    Example:
        >>> trainer = VAETrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     device='cuda',
        ...     lr=1e-3
        ... )
        >>> history = trainer.train(epochs=100)
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 lr=1e-3, weight_decay=1e-5, beta_scheduler=None,
                 patience=15, checkpoint_dir='checkpoints'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # β scheduler
        if beta_scheduler is None:
            beta_scheduler = BetaScheduler(mode='linear', warmup_epochs=20)
        self.beta_scheduler = beta_scheduler
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History tracking
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
            'kl_level1': [],
            'kl_level2': [],
            'kl_level3': [],
            'beta_values': [],
            'learning_rates': []
        }
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            dict: Training metrics for this epoch
        """
        self.model.train()
        
        # Get β value for this epoch
        beta = self.beta_scheduler.get_beta(epoch)
        
        # Metrics accumulators
        total_loss = 0
        total_recon = 0
        total_kl = 0
        kl_levels = [0, 0, 0]
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            x = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            recon, latents, params = self.model(x)
            
            # Compute loss
            loss, recon_loss, kl_loss, kl_per_level = vae_loss(
                recon, x, params, beta=beta
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            for i in range(3):
                kl_levels[i] += kl_per_level[i]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'β': f'{beta:.3f}'
            })
        
        # Average metrics
        n_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'kl': total_kl / n_batches,
            'kl_levels': [kl / n_batches for kl in kl_levels],
            'beta': beta
        }
        
        return metrics
    
    def validate_epoch(self, epoch):
        """
        Validate for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            dict: Validation metrics for this epoch
        """
        self.model.eval()
        
        beta = self.beta_scheduler.get_beta(epoch)
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch.to(self.device)
                
                # Forward pass
                recon, latents, params = self.model(x)
                
                # Compute loss
                loss, recon_loss, kl_loss, _ = vae_loss(
                    recon, x, params, beta=beta
                )
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
        
        # Average metrics
        n_batches = len(self.val_loader)
        metrics = {
            'loss': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'kl': total_kl / n_batches
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, epochs):
        """
        Full training loop.
        
        Args:
            epochs (int): Number of epochs to train
            
        Returns:
            dict: Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['val_kl'].append(val_metrics['kl'])
            self.history['kl_level1'].append(train_metrics['kl_levels'][0])
            self.history['kl_level2'].append(train_metrics['kl_levels'][1])
            self.history['kl_level3'].append(train_metrics['kl_levels'][2])
            self.history['beta_values'].append(train_metrics['beta'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            self.lr_scheduler.step(val_metrics['loss'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Recon: {train_metrics['recon']:.4f} | "
                  f"KL: {train_metrics['kl']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | "
                  f"Recon: {val_metrics['recon']:.4f} | "
                  f"KL: {val_metrics['kl']:.4f}")
            print(f"  KL Levels:  L1={train_metrics['kl_levels'][0]:.2f} | "
                  f"L2={train_metrics['kl_levels'][1]:.2f} | "
                  f"L3={train_metrics['kl_levels'][2]:.2f}")
            print(f"  LR: {current_lr:.2e} | β: {train_metrics['beta']:.3f}")
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.patience}")
                self.save_checkpoint(epoch, is_best=False)
            
            if self.patience_counter >= self.patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"{'='*60}\n")
                break
        
        print("\n✓ Training complete")
        return self.history
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.current_epoch = checkpoint['epoch']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch+1}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")