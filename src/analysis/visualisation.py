"""
Visualization utilities for latent space analysis.

Provides functions for UMAP, t-SNE projections, heatmaps,
and various other visualization methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from tqdm.auto import tqdm


def plot_umap_projection(latents_dict, n_samples=5000, save_path='latent_umap.png'):
    """
    Create UMAP projections for all hierarchical levels.
    
    Args:
        latents_dict (dict): Dictionary with 'level1', 'level2', 'level3' keys
        n_samples (int): Number of samples to use (subsampled if needed)
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (level_name, latents) in enumerate(latents_dict.items()):
        # Subsample for faster computation
        if len(latents) > n_samples:
            indices = np.random.choice(len(latents), n_samples, replace=False)
            latents_subset = latents[indices]
        else:
            latents_subset = latents
        
        print(f"Computing UMAP for {level_name} ({latents_subset.shape[1]}d -> 2d)...")
        
        # Fit UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42,
            verbose=False
        )
        embedding = reducer.fit_transform(latents_subset)
        
        # Plot
        ax = axes[idx]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=np.arange(len(embedding)),  # Color by sample index
            cmap='viridis',
            s=10,
            alpha=0.6,
            rasterized=True
        )
        
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.set_title(f'{level_name.capitalize()} ({latents.shape[1]}d)',
                    fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sample Index', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ UMAP visualization saved to {save_path}")


def plot_tsne_projection(latents_dict, n_samples=3000, save_path='latent_tsne.png'):
    """
    Create t-SNE projections for all hierarchical levels.
    
    Args:
        latents_dict (dict): Dictionary with hierarchical latent levels
        n_samples (int): Number of samples (t-SNE is slow, keep this low)
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (level_name, latents) in enumerate(latents_dict.items()):
        # Subsample
        if len(latents) > n_samples:
            indices = np.random.choice(len(latents), n_samples, replace=False)
            latents_subset = latents[indices]
        else:
            latents_subset = latents
        
        print(f"Computing t-SNE for {level_name} ({latents_subset.shape[1]}d -> 2d)...")
        
        # Fit t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            random_state=42,
            verbose=0
        )
        embedding = tsne.fit_transform(latents_subset)
        
        # Plot
        ax = axes[idx]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=np.arange(len(embedding)),
            cmap='plasma',
            s=10,
            alpha=0.6,
            rasterized=True
        )
        
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        ax.set_title(f'{level_name.capitalize()} t-SNE',
                    fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sample Index', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ t-SNE visualization saved to {save_path}")


def plot_activation_heatmap(latents_dict, n_samples=500, save_path='activation_heatmap.png'):
    """
    Visualize activation patterns as heatmaps.
    
    Shows which dimensions are active across samples.
    
    Args:
        latents_dict (dict): Hierarchical latent representations
        n_samples (int): Number of samples to visualize
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    for idx, (level_name, latents) in enumerate(latents_dict.items()):
        # Subsample
        if len(latents) > n_samples:
            indices = np.random.choice(len(latents), n_samples, replace=False)
            latents_subset = latents[indices]
        else:
            latents_subset = latents
        
        ax = axes[idx]
        
        # Create heatmap
        im = ax.imshow(
            latents_subset.T,
            aspect='auto',
            cmap='RdBu_r',
            interpolation='nearest',
            vmin=-3,
            vmax=3
        )
        
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Latent Dimension', fontsize=11)
        ax.set_title(f'{level_name.capitalize()} Activation Heatmap ({latents.shape[1]}d)',
                    fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Value', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Activation heatmap saved to {save_path}")


def plot_training_history(history, save_path='training_history.png'):
    """
    Comprehensive visualization of training dynamics.
    
    Args:
        history (dict): Training history from trainer
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train', linewidth=2, alpha=0.8)
    ax.plot(history['val_loss'], label='Validation', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Reconstruction Loss
    ax = axes[0, 1]
    ax.plot(history['train_recon'], label='Train', linewidth=2, alpha=0.8)
    ax.plot(history['val_recon'], label='Validation', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Reconstruction Loss', fontsize=11)
    ax.set_title('Reconstruction Loss (MSE)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: KL Divergence
    ax = axes[0, 2]
    ax.plot(history['train_kl'], label='Train', linewidth=2, alpha=0.8, color='crimson')
    ax.plot(history['val_kl'], label='Validation', linewidth=2, alpha=0.8, color='darkred')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.set_title('KL Divergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Hierarchical KL Levels
    ax = axes[1, 0]
    ax.plot(history['kl_level1'], label='Level 1 (256d)', linewidth=2, alpha=0.8)
    ax.plot(history['kl_level2'], label='Level 2 (512d)', linewidth=2, alpha=0.8)
    ax.plot(history['kl_level3'], label='Level 3 (1024d)', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.set_title('KL by Hierarchical Level', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 5: Beta Schedule
    ax = axes[1, 1]
    ax.plot(history['beta_values'], linewidth=2.5, color='purple')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('β Value', fontsize=11)
    ax.set_title('β-Annealing Schedule', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Plot 6: Learning Rate
    ax = axes[1, 2]
    ax.plot(history['learning_rates'], linewidth=2.5, color='green')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history visualization saved to {save_path}")


def plot_latent_density_distribution(latents_dict, save_path='latent_density.png'):
    """
    Visualize distribution of pairwise distances in latent space.
    
    Helps identify if latent space is densely packed or has voids.
    
    Args:
        latents_dict (dict): Hierarchical latent representations
        save_path (str): Path to save figure
    """
    from scipy.spatial.distance import pdist
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (level_name, latents) in enumerate(latents_dict.items()):
        # Subsample for computational efficiency
        n_samples = min(2000, len(latents))
        indices = np.random.choice(len(latents), n_samples, replace=False)
        latents_subset = latents[indices]
        
        print(f"Computing pairwise distances for {level_name}...")
        
        # Compute pairwise distances
        distances = pdist(latents_subset, metric='euclidean')
        
        # Plot distribution
        ax = axes[idx]
        ax.hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Pairwise Distance', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{level_name.capitalize()}\n'
                    f'Mean: {np.mean(distances):.2f} | Std: {np.std(distances):.2f}',
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add median line
        median = np.median(distances)
        ax.axvline(median, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Median: {median:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Latent density visualization saved to {save_path}")


def plot_correlation_matrix(latents, level_name, save_path='correlation_matrix.png'):
    """
    Plot correlation matrix for a single latent level.
    
    Reveals which dimensions co-activate.
    
    Args:
        latents (np.ndarray): Latent representations [n_samples, latent_dim]
        level_name (str): Name of the level
        save_path (str): Path to save figure
    """
    # Compute correlation matrix
    correlation = np.corrcoef(latents.T)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        correlation,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'}
    )
    
    plt.title(f'{level_name.capitalize()} - Dimension Correlation Matrix',
             fontsize=14, fontweight='bold')
    plt.xlabel('Dimension', fontsize=11)
    plt.ylabel('Dimension', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Correlation matrix saved to {save_path}")