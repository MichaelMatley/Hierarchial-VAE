"""
Latent space interpolation and arithmetic operations.

Tests if the latent space has smooth, meaningful structure.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ..data.dna_encoder import DNAEncoder


def interpolate_between_samples(model, x1, x2, num_steps=10, device='cuda'):
    """
    Interpolate between two samples in latent space.
    
    Args:
        model: Trained VAE model
        x1, x2: Input tensors [1, input_dim]
        num_steps: Number of interpolation steps
        device: Device to use
        
    Returns:
        list: Interpolated reconstructions
    """
    model.eval()
    
    with torch.no_grad():
        # Encode endpoints
        latents1, _ = model.encode(x1.to(device))
        latents2, _ = model.encode(x2.to(device))
        
        # Interpolate at each hierarchical level
        interpolations = []
        
        for alpha in np.linspace(0, 1, num_steps):
            # Spherical interpolation (slerp) - better for normalized spaces
            # But linear interpolation (lerp) works fine for VAE latents
            interp_latents = tuple(
                (1 - alpha) * z1 + alpha * z2
                for z1, z2 in zip(latents1, latents2)
            )
            
            # Decode
            recon = model.decode(interp_latents)
            interpolations.append(recon.cpu().numpy())
        
        return np.array(interpolations)


def latent_arithmetic(model, x1, x2, x3, device='cuda'):
    """
    Perform latent arithmetic: (x1 - x2) + x3
    
    Tests if latent space supports compositional operations.
    
    Args:
        model: Trained VAE model
        x1, x2, x3: Input tensors
        device: Device to use
        
    Returns:
        Tensor: Result of arithmetic operation
    """
    model.eval()
    
    with torch.no_grad():
        # Encode
        latents1, _ = model.encode(x1.to(device))
        latents2, _ = model.encode(x2.to(device))
        latents3, _ = model.encode(x3.to(device))
        
        # Arithmetic: (x1 - x2) + x3
        result_latents = tuple(
            (z1 - z2) + z3
            for z1, z2, z3 in zip(latents1, latents2, latents3)
        )
        
        # Decode
        result = model.decode(result_latents)
        
        return result


def visualize_interpolation(interpolations, save_path='interpolation.png'):
    """
    Visualize sequence interpolation.
    
    Args:
        interpolations (np.ndarray): Array of interpolated reconstructions
        save_path (str): Path to save figure
    """
    num_steps = len(interpolations)
    
    fig, axes = plt.subplots(num_steps, 1, figsize=(16, num_steps * 1))
    
    for idx, interp in enumerate(interpolations):
        # Decode to sequence
        interp_reshaped = interp[0].reshape(4, 1024)
        seq = DNAEncoder.decode_one_hot(interp_reshaped)
        
        # Show first 80 bases
        display_seq = seq[:80]
        
        ax = axes[idx]
        for pos, base in enumerate(display_seq):
            color_map = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
            ax.text(pos, 0, base, ha='center', va='center', fontsize=7,
                   family='monospace', color=color_map.get(base, 'black'))
        
        alpha = idx / (num_steps - 1)
        ax.set_xlim(-1, len(display_seq))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(f'α={alpha:.2f}', fontsize=9, rotation=0, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    plt.suptitle('Latent Space Interpolation', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Interpolation visualization saved to {save_path}")