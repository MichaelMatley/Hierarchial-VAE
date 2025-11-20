#!/usr/bin/env python
"""
Training script for Hierarchical VAE.

Usage:
    python scripts/train.py --data path/to/genome.fasta --epochs 100
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.hierarchical_vae import HierarchicalVAE
from data.genomic_dataset import GenomicDataset
from training.trainer import VAETrainer
from training.schedulers import BetaScheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Hierarchical VAE on genomic data'
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to FASTA genome file')
    parser.add_argument('--window-size', type=int, default=1024,
                        help='Sequence window size (default: 1024)')
    parser.add_argument('--stride', type=int, default=512,
                        help='Sliding window stride (default: 512)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use (default: all)')
    
    # Model arguments
    parser.add_argument('--latent-dims', type=int, nargs=3, default=[256, 512, 1024],
                        help='Latent dimensions for 3 levels (default: 256 512 1024)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    
    # Beta scheduling
    parser.add_argument('--beta-mode', type=str, default='linear',
                        choices=['constant', 'linear', 'cosine', 'cyclical', 'sigmoid'],
                        help='Beta annealing schedule (default: linear)')
    parser.add_argument('--max-beta', type=float, default=1.0,
                        help='Maximum beta value (default: 1.0)')
    parser.add_argument('--warmup-epochs', type=int, default=20,
                        help='Beta warmup epochs (default: 20)')
    
    # Other
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints',
                        help='Checkpoint directory (default: outputs/checkpoints)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers (default: 2)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("\n" + "="*60)
    print("HIERARCHICAL VAE TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Window size: {args.window_size}")
    print(f"  Latent dims: {args.latent_dims}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Beta mode: {args.beta_mode}")
    print(f"  Device: {args.device}")
    print()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load dataset
    print("Loading dataset...")
    dataset = GenomicDataset(
        fasta_file=args.data,
        window_size=args.window_size,
        stride=args.stride,
        max_samples=args.max_samples
    )
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Sequences: {stats['num_sequences']:,}")
    print(f"  Mean GC content: {stats['mean_gc_content']:.2f}%")
    print(f"  Base frequencies:")
    for base, freq in stats['base_frequencies'].items():
        print(f"    {base}: {freq:.4f}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset):,} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"  Val:   {len(val_dataset):,} ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"  Test:  {len(test_dataset):,} ({len(test_dataset)/len(dataset)*100:.1f}%)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )
    
    # Create model
    print("\nCreating model...")
    input_dim = args.window_size * 4  # One-hot encoding
    
    model = HierarchicalVAE(
        input_dim=input_dim,
        latent_dims=args.latent_dims,
        dropout=args.dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create beta scheduler
    beta_scheduler = BetaScheduler(
        mode=args.beta_mode,
        max_beta=args.max_beta,
        warmup_epochs=args.warmup_epochs
    )
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta_scheduler=beta_scheduler,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    history = trainer.train(epochs=args.epochs)
    
    # Save final history
    import json
    history_path = Path(args.checkpoint_dir) / 'training_history.json'
    
    # Convert numpy values to python types for JSON serialization
    history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
    
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    
    print(f"\n✓ Training history saved to {history_path}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  Total epochs: {len(history['train_loss'])}")
    print(f"  Checkpoints saved to: {args.checkpoint_dir}")
    print()


if __name__ == '__main__':
    main()