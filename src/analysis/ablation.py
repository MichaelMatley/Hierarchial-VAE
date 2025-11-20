"""
Ablation studies for dimension importance ranking.

Systematically removes dimensions to measure their contribution.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def dimension_importance_ablation(model, dataloader, device, level=0, num_samples=500):
    """
    Measure importance of each dimension via ablation.
    
    Sets each dimension to zero and measures reconstruction error increase.
    
    Args:
        model: Trained VAE model
        dataloader: Data loader
        device: Device to use
        level (int): Which latent level to ablate (0, 1, or 2)
        num_samples (int): Number of samples to test
        
    Returns:
        dict: Importance scores and rankings
    """
    model.eval()
    
    # Get baseline reconstruction error
    print("Computing baseline reconstruction error...")
    baseline_errors = []
    
    with torch.no_grad():
        samples_collected = 0
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
            
            x = batch.to(device)
            recon, _, _ = model(x)
            error = F.mse_loss(recon, x, reduction='none').mean(dim=1)
            baseline_errors.append(error.cpu().numpy())
            
            samples_collected += len(x)
    
    baseline_errors = np.concatenate(baseline_errors)[:num_samples]
    baseline_mean = np.mean(baseline_errors)
    
    print(f"Baseline error: {baseline_mean:.6f}")
    
    # Get latent dimension
    latent_dim = model.latent_dims[level]
    
    print(f"\nTesting dimension importance (Level {level+1}, {latent_dim} dimensions)...")
    
    importance_scores = []
    
    for dim_idx in tqdm(range(latent_dim), desc="Ablating dimensions"):
        ablation_errors = []
        
        with torch.no_grad():
            samples_collected = 0
            for batch in dataloader:
                if samples_collected >= num_samples:
                    break
                
                x = batch.to(device)
                
                # Encode
                latents, _ = model.encode(x)
                
                # Ablate specific dimension
                latents_list = list(latents)
                latents_list[level] = latents_list[level].clone()
                latents_list[level][:, dim_idx] = 0
                latents_ablated = tuple(latents_list)
                
                # Decode
                recon = model.decode(latents_ablated)
                
                # Measure error
                error = F.mse_loss(recon, x, reduction='none').mean(dim=1)
                ablation_errors.append(error.cpu().numpy())
                
                samples_collected += len(x)
        
        ablation_errors = np.concatenate(ablation_errors)[:num_samples]
        ablation_mean = np.mean(ablation_errors)
        
        # Importance = increase in error when dimension is removed
        importance = ablation_mean - baseline_mean
        importance_scores.append(importance)
    
    importance_scores = np.array(importance_scores)
    ranked_indices = np.argsort(importance_scores)[::-1]  # Descending
    
    return {
        'importance_scores': importance_scores,
        'ranked_indices': ranked_indices,
        'baseline_error': baseline_mean,
        'latent_dim': latent_dim,
        'level': level
    }


def plot_dimension_importance(results, top_k=20, save_path='dimension_importance.png'):
    """
    Visualize dimension importance rankings.
    
    Args:
        results (dict): Results from dimension_importance_ablation
        top_k (int): Number of top dimensions to show
        save_path (str): Path to save figure
    """
    importance = results['importance_scores']
    ranked = results['ranked_indices']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot 1: All dimensions sorted by importance
    ax = axes[0]
    ax.bar(range(len(importance)), importance[ranked], color='crimson', alpha=0.7)
    ax.set_xlabel('Dimension (sorted by importance)', fontsize=11)
    ax.set_ylabel('Importance Score (Δ Error)', fontsize=11)
    ax.set_title(f'Dimension Importance - Level {results["level"]+1}', 
                fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Top K dimensions
    ax = axes[1]
    top_dims = ranked[:top_k]
    top_scores = importance[top_dims]
    
    ax.barh(range(top_k), top_scores, color='darkgreen', alpha=0.7)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f'Dim {d}' for d in top_dims], fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(f'Top {top_k} Most Important Dimensions', 
                fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Dimension importance plot saved to {save_path}")


def print_importance_summary(results, top_k=10):
    """Print summary of most important dimensions."""
    importance = results['importance_scores']
    ranked = results['ranked_indices']
    
    print("\n" + "="*60)
    print(f"TOP {top_k} MOST IMPORTANT DIMENSIONS (LEVEL {results['level']+1})")
    print("="*60)
    
    for rank, dim_idx in enumerate(ranked[:top_k], 1):
        print(f"{rank:2d}. Dimension {dim_idx:3d} | Importance: {importance[dim_idx]:.6f}")
    
    print("="*60)
    
    # Statistics
    print(f"\nImportance Statistics:")
    print(f"  Mean importance:   {np.mean(importance):.6f}")
    print(f"  Std importance:    {np.std(importance):.6f}")
    print(f"  Max importance:    {np.max(importance):.6f}")
    print(f"  Min importance:    {np.min(importance):.6f}")
    print(f"  Baseline error:    {results['baseline_error']:.6f}")