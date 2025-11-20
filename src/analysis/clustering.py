"""
Clustering analysis for latent spaces.

Analyzes how well the model self-organizes data into clusters.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt


def perform_clustering(latents, n_clusters=10, random_state=42):
    """
    Perform k-means clustering on latent space.
    
    Args:
        latents (np.ndarray): Latent representations
        n_clusters (int): Number of clusters
        random_state (int): Random seed
        
    Returns:
        dict: Clustering results and metrics
    """
    print(f"  Performing k-means clustering (k={n_clusters})...")
    
    # Fit k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(latents)
    
    # Compute metrics
    silhouette = silhouette_score(latents, cluster_labels)
    davies_bouldin = davies_bouldin_score(latents, cluster_labels)
    calinski = calinski_harabasz_score(latents, cluster_labels)
    
    return {
        'labels': cluster_labels,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski
    }


def analyze_hierarchical_clustering(latents_dict, n_clusters=10):
    """
    Perform clustering analysis on all hierarchical levels.
    
    Args:
        latents_dict (dict): Dictionary with hierarchical latent levels
        n_clusters (int): Number of clusters for k-means
        
    Returns:
        dict: Clustering results for each level
    """
    results = {}
    
    print("\nPerforming clustering analysis...")
    
    for level_name, latents in latents_dict.items():
        print(f"\n{level_name}:")
        results[level_name] = perform_clustering(latents, n_clusters)
    
    return results


def plot_clustering_results(clustering_results, save_path='clustering_analysis.png'):
    """
    Visualize clustering results for all levels.
    
    Args:
        clustering_results (dict): Results from analyze_hierarchical_clustering
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (level_name, results) in enumerate(clustering_results.items()):
        # Plot 1: Cluster size distribution
        ax = axes[0, idx]
        unique, counts = np.unique(results['labels'], return_counts=True)
        ax.bar(unique, counts, color='steelblue', alpha=0.8)
        ax.set_xlabel('Cluster ID', fontsize=11)
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title(f'{level_name.capitalize()} - Cluster Sizes',
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Plot 2: Metrics
        ax = axes[1, idx]
        metrics = {
            'Silhouette\n(higher=better)': results['silhouette'],
            'Davies-Bouldin\n(lower=better)': results['davies_bouldin'],
            'Calinski-Harabasz\n(higher=better)': results['calinski_harabasz'] / 1000  # Scale for visibility
        }
        
        colors = ['green' if 'higher' in k else 'red' for k in metrics.keys()]
        bars = ax.bar(range(len(metrics)), metrics.values(), color=colors, alpha=0.7)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics.keys(), fontsize=9)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{level_name.capitalize()} - Clustering Quality',
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Clustering visualization saved to {save_path}")


def print_clustering_summary(clustering_results):
    """Print formatted summary of clustering analysis."""
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("="*60)
    
    for level_name, results in clustering_results.items():
        print(f"\n{level_name.upper()}:")
        print(f"  Silhouette score:        {results['silhouette']:.4f}")
        print(f"    (Range: [-1, 1], higher is better)")
        print(f"  Davies-Bouldin score:    {results['davies_bouldin']:.4f}")
        print(f"    (Range: [0, ∞), lower is better)")
        print(f"  Calinski-Harabasz score: {results['calinski_harabasz']:.2f}")
        print(f"    (Range: [0, ∞), higher is better)")
        print(f"  Inertia:                 {results['inertia']:.2f}")
    
    print("="*60)


def find_optimal_clusters(latents, max_clusters=20, save_path='elbow_plot.png'):
    """
    Find optimal number of clusters using elbow method.
    
    Args:
        latents (np.ndarray): Latent representations
        max_clusters (int): Maximum number of clusters to test
        save_path (str): Path to save elbow plot
        
    Returns:
        dict: Inertia and silhouette scores for each k
    """
    print(f"Finding optimal number of clusters (testing k=2 to {max_clusters})...")
    
    inertias = []
    silhouettes = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latents)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(latents, labels))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    ax = axes[0]
    ax.plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Inertia', fontsize=11)
    ax.set_title('Elbow Method', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Silhouette plot
    ax = axes[1]
    ax.plot(k_range, silhouettes, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Silhouette Score vs k', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Elbow plot saved to {save_path}")
    
    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouettes': silhouettes
    }