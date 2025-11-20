"""
Intrinsic dimensionality analysis for latent spaces.

Measures how much of the latent capacity is actually utilized
using PCA and other dimensionality estimation techniques.
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def compute_intrinsic_dimensionality(latents, variance_threshold=0.95):
    """
    Compute intrinsic dimensionality using PCA.
    
    Intrinsic dim = minimum number of components needed to explain
    variance_threshold of total variance.
    
    Args:
        latents (np.ndarray): Latent representations [n_samples, latent_dim]
        variance_threshold (float): Cumulative variance threshold (default: 0.95)
        
    Returns:
        dict: Results including intrinsic_dim, explained_variance_ratio, etc.
        
    Example:
        >>> latents = model.encode(data)
        >>> results = compute_intrinsic_dimensionality(latents)
        >>> print(f"Intrinsic dim: {results['intrinsic_dim']}/{results['nominal_dim']}")
    """
    # Fit PCA
    pca = PCA()
    pca.fit(latents)
    
    # Compute cumulative explained variance
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find intrinsic dimensionality
    intrinsic_dim = np.argmax(cumsum_variance >= variance_threshold) + 1
    
    # Utilization percentage
    nominal_dim = latents.shape[1]
    utilization = (intrinsic_dim / nominal_dim) * 100
    
    return {
        'intrinsic_dim': intrinsic_dim,
        'nominal_dim': nominal_dim,
        'utilization': utilization,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumsum_variance': cumsum_variance,
        'pca_model': pca
    }


def analyze_hierarchical_latents(latents_dict, variance_threshold=0.95):
    """
    Analyze intrinsic dimensionality across hierarchical levels.
    
    Args:
        latents_dict (dict): Dictionary with keys 'level1', 'level2', 'level3'
        variance_threshold (float): Variance threshold for intrinsic dim
        
    Returns:
        dict: Results for each level
    """
    results = {}
    
    for level_name, latents in latents_dict.items():
        results[level_name] = compute_intrinsic_dimensionality(
            latents, variance_threshold
        )
    
    return results


def plot_intrinsic_dimensionality(results_dict, save_path='intrinsic_dim.png'):
    """
    Visualize intrinsic dimensionality analysis.
    
    Args:
        results_dict (dict): Results from analyze_hierarchical_latents
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (level_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        cumsum = results['cumsum_variance']
        intrinsic_dim = results['intrinsic_dim']
        nominal_dim = results['nominal_dim']
        
        # Plot cumulative variance
        ax.plot(cumsum, linewidth=2.5, color='darkblue')
        ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='95% threshold')
        ax.axvline(x=intrinsic_dim, color='green', linestyle='--', 
                  linewidth=2, alpha=0.7,
                  label=f'Intrinsic dim: {intrinsic_dim}')
        
        ax.set_xlabel('Number of Components', fontsize=11)
        ax.set_ylabel('Cumulative Explained Variance', fontsize=11)
        ax.set_title(f'{level_name.capitalize()} ({nominal_dim}d)\n'
                    f'Utilization: {results["utilization"]:.1f}%',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Intrinsic dimensionality plot saved to {save_path}")


def print_dimensionality_summary(results_dict):
    """Print formatted summary of dimensionality analysis."""
    print("\n" + "="*60)
    print("INTRINSIC DIMENSIONALITY ANALYSIS")
    print("="*60)
    
    for level_name, results in results_dict.items():
        print(f"\n{level_name.upper()}:")
        print(f"  Nominal dimension:    {results['nominal_dim']}")
        print(f"  Intrinsic dimension:  {results['intrinsic_dim']}")
        print(f"  Utilization:          {results['utilization']:.1f}%")
        print(f"  Top 10 PCs explain:   {results['cumsum_variance'][9]:.2%}")
    
    print("="*60)