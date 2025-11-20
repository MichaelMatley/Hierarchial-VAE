#!/usr/bin/env python
"""
Evaluation script for trained Hierarchical VAE.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.hierarchical_vae import HierarchicalVAE
from data.genomic_dataset import GenomicDataset
from data.dna_encoder import DNAEncoder
from training.losses import vae_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained Hierarchical VAE'
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to FASTA genome file')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of reconstruction samples to show (default: 10)')
    
    return parser.parse_args()


def evaluate_reconstruction_accuracy(model, dataloader, device, num_samples=10):
    """
    Evaluate per-nucleotide reconstruction accuracy.
    
    Returns:
        dict: Accuracy statistics
    """
    model.eval()
    
    accuracies = []
    samples_shown = 0
    
    print("\nReconstruction Examples:")
    print("-" * 80)
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_shown >= num_samples:
                break
            
            x = batch.to(device)
            recon, _, _ = model(x)
            
            for i in range(min(len(x), num_samples - samples_shown)):
                # Convert to sequences
                original = x[i].cpu().numpy().reshape(4, 1024)
                reconstructed = recon[i].cpu().numpy().reshape(4, 1024)
                
                orig_seq = DNAEncoder.decode_one_hot(original)
                recon_seq = DNAEncoder.decode_one_hot(reconstructed)
                
                # Calculate accuracy
                matches = sum(o == r for o, r in zip(orig_seq, recon_seq))
                accuracy = matches / len(orig_seq)
                accuracies.append(accuracy)
                
                # Show sample
                print(f"\nSample {samples_shown + 1}:")
                print(f"  Original:      {orig_seq[:60]}...")
                print(f"  Reconstructed: {recon_seq[:60]}...")
                print(f"  Accuracy: {accuracy:.2%} ({matches}/{len(orig_seq)} correct)")
                
                samples_shown += 1
    
    print("-" * 80)
    
    return {
        'mean': np.mean(accuracies),
        'median': np.median(accuracies),
        'std': np.std(accuracies),
        'min': np.min(accuracies),
        'max': np.max(accuracies)
    }


def evaluate_loss(model, dataloader, device):
    """
    Evaluate reconstruction and KL losses on dataset.
    
    Returns:
        dict: Loss statistics
    """
    model.eval()
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    kl_levels = [0, 0, 0]
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            
            recon, latents, params = model(x)
            loss, recon_loss, kl_loss, kl_per_level = vae_loss(
                recon, x, params, beta=1.0
            )
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            for i in range(3):
                kl_levels[i] += kl_per_level[i]
            
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon / num_batches,
        'kl_loss': total_kl / num_batches,
        'kl_level1': kl_levels[0] / num_batches,
        'kl_level2': kl_levels[1] / num_batches,
        'kl_level3': kl_levels[2] / num_batches
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("HIERARCHICAL VAE EVALUATION")
    print("="*60)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Create model
    # Infer architecture from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    # Determine input_dim from first layer
    input_dim = state_dict['enc1.0.weight'].shape[1]
    
    # Determine latent dims from mu layers
    latent_dims = [
        state_dict['z1_mu.weight'].shape[0],
        state_dict['z2_mu.weight'].shape[0],
        state_dict['z3_mu.weight'].shape[0]
    ]
    
    print(f"  Model architecture:")
    print(f"    Input dim: {input_dim}")
    print(f"    Latent dims: {latent_dims}")
    
    model = HierarchicalVAE(
        input_dim=input_dim,
        latent_dims=latent_dims
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint['epoch']+1} epochs)")
    
    # Load dataset
    print(f"\nLoading evaluation data from {args.data}...")
    dataset = GenomicDataset(
        fasta_file=args.data,
        window_size=1024,
        stride=512,
        max_samples=10000  # Limit for faster evaluation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Evaluate losses
    print("\nComputing losses...")
    loss_stats = evaluate_loss(model, dataloader, args.device)
    
    print("\nLoss Statistics:")
    print(f"  Total Loss:       {loss_stats['total_loss']:.4f}")
    print(f"  Reconstruction:   {loss_stats['recon_loss']:.4f}")
    print(f"  KL Divergence:    {loss_stats['kl_loss']:.4f}")
    print(f"    Level 1 (256d): {loss_stats['kl_level1']:.4f}")
    print(f"    Level 2 (512d): {loss_stats['kl_level2']:.4f}")
    print(f"    Level 3 (1024d): {loss_stats['kl_level3']:.4f}")
    
    # Evaluate reconstruction accuracy
    accuracy_stats = evaluate_reconstruction_accuracy(
        model, dataloader, args.device, num_samples=args.num_samples
    )
    
    print("\nReconstruction Accuracy Statistics:")
    print(f"  Mean:   {accuracy_stats['mean']:.4f}")
    print(f"  Median: {accuracy_stats['median']:.4f}")
    print(f"  Std:    {accuracy_stats['std']:.4f}")
    print(f"  Min:    {accuracy_stats['min']:.4f}")
    print(f"  Max:    {accuracy_stats['max']:.4f}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
