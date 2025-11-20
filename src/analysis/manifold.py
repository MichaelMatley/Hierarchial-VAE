"""
Manifold continuity and smoothness analysis.

Tests whether the latent space forms a continuous manifold.
"""

import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def test_manifold_continuity(model, dataloader, device, num_tests=100, num_steps=20):
    """
    Test if latent space forms a continuous manifold.
    
    Interpolates between random pairs and measures reconstruction smoothness.
    
    Args:
        model: Trained VAE model
        dataloader: Data loader
        device: Device to use
        num_tests (int): Number of interpolation tests
        num_steps (int): Interpolation granularity
        
    Returns:
        dict: Smoothness metrics
    """
    model.eval()
    
    batch = next(iter(dataloader)).to(device)
    
    smoothness_scores = []
    path_lengths = []
    
    print(f"Testing manifold continuity ({num_tests} interpolations)...")
    
    for test_idx in tqdm(range(min(num_tests, len(batch) - 1))):
        x1 = batch[test_idx:test_idx+1]
        x2 = batch[test_idx+1:test_idx+2]
        
        with torch.no_grad():
            # Encode endpoints
            latents1, _ = model.encode(x1)
            latents2, _ = model.encode(x2)
            
            # Interpolate
            reconstructions = []
            
            for alpha in np.linspace(0, 1, num_steps):
                interp_latents = tuple(
                    (1 - alpha) * z1 + alpha * z2
                    for z1, z2 in zip(latents1, latents2)
                )
                
                recon = model.decode(interp_latents)
                reconstructions.append(recon[0].cpu().numpy())
            
            reconstructions = np.array(reconstructions)
            
            # Measure smoothness: variance of consecutive differences
            diffs = np.diff(reconstructions, axis=0)
            diff_norms = np.linalg.norm(diffs, axis=1)
            
            smoothness = np.var(diff_norms)
            path_length = np.sum(diff_norms)
            
            smoothness_scores.append(smoothness)
            path_lengths.append(path_length)
    
    smoothness_scores = np.array(smoothness_scores)
    path_lengths = np.array(path_lengths)
    
    return {
        'smoothness_scores': smoothness_scores,
        'path_lengths': path_lengths,
        'mean_smoothness': np.mean(smoothness_scores),
        'std_smoothness': np.std(smoothness_scores),
        'mean_path_length': np.mean(path_lengths),
        'std_path_length': np.std(path_lengths)
    }


def plot_manifold_analysis(results, save_path='manifold_continuity.png'):
    """
    Visualize manifold continuity analysis.
    
    Args:
        results (dict): Results from test_manifold_continuity
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Smoothness distribution
    ax = axes[0]
    ax.hist(results['smoothness_scores'], bins=30, color='purple', 
            alpha=0.7, edgecolor='black')
    ax.axvline(results['mean_smoothness'], color='red', linestyle='--', 
              linewidth=2, label=f"Mean: {results['mean_smoothness']:.4f}")
    ax.set_xlabel('Smoothness Score (lower = smoother)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Manifold Smoothness Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Path length distribution
    ax = axes[1]
    ax.hist(results['path_lengths'], bins=30, color='teal', 
            alpha=0.7, edgecolor='black')
    ax.axvline(results['mean_path_length'], color='red', linestyle='--', 
              linewidth=2, label=f"Mean: {results['mean_path_length']:.2f}")
    ax.set_xlabel('Interpolation Path Length', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Path Length Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Manifold analysis plot saved to {save_path}")


def print_manifold_summary(results):
    """Print summary of manifold continuity analysis."""
    print("\n" + "="*60)
    print("MANIFOLD CONTINUITY ANALYSIS")
    print("="*60)
    print(f"\nSmoothness (lower = smoother manifold):")
    print(f"  Mean:   {results['mean_smoothness']:.6f}")
    print(f"  Std:    {results['std_smoothness']:.6f}")
    print(f"  Median: {np.median(results['smoothness_scores']):.6f}")
    print(f"\nPath Length:")
    print(f"  Mean:   {results['mean_path_length']:.4f}")
    print(f"  Std:    {results['std_path_length']:.4f}")
    print(f"  Median: {np.median(results['path_lengths']):.4f}")
    print("\nInterpretation:")
    print("  Low smoothness variance = continuous manifold")
    print("  High smoothness variance = discrete clusters with voids")
    print("="*60)