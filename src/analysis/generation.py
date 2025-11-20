"""
Generative sampling from prior distribution.

Tests if the model learned a meaningful generative distribution.
"""

import torch
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
from ..data.dna_encoder import DNAEncoder


def generate_from_prior(model, num_samples, device='cuda', temperature=1.0):
    """
    Generate synthetic sequences by sampling from prior N(0,1).
    
    Args:
        model: Trained VAE model
        num_samples (int): Number of sequences to generate
        device: Device to use
        temperature (float): Sampling temperature (>1 = more random, <1 = more deterministic)
        
    Returns:
        list: Generated sequences (strings)
    """
    model.eval()
    
    sequences = []
    
    print(f"Generating {num_samples} sequences from prior (T={temperature})...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Sample from standard normal with temperature scaling
            z1 = torch.randn(1, model.latent_dims[0], device=device) * temperature
            z2 = torch.randn(1, model.latent_dims[1], device=device) * temperature
            z3 = torch.randn(1, model.latent_dims[2], device=device) * temperature
            
            latents = (z1, z2, z3)
            
            # Decode
            generated = model.decode(latents)
            generated_np = generated[0].cpu().numpy().reshape(4, 1024)
            
            # Convert to sequence
            sequence = DNAEncoder.decode_one_hot(generated_np)
            sequences.append(sequence)
    
    return sequences


def analyze_generated_sequences(sequences):
    """
    Analyze statistics of generated sequences.
    
    Args:
        sequences (list): List of DNA sequence strings
        
    Returns:
        dict: Statistics including GC content, base frequencies
    """
    gc_contents = []
    base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'N': 0}
    
    for seq in sequences:
        gc_contents.append(DNAEncoder.compute_gc_content(seq))
        
        for base in seq:
            if base in base_counts:
                base_counts[base] += 1
    
    total_bases = sum(base_counts.values())
    
    return {
        'gc_contents': np.array(gc_contents),
        'mean_gc': np.mean(gc_contents),
        'std_gc': np.std(gc_contents),
        'base_frequencies': {b: c/total_bases for b, c in base_counts.items()},
        'total_bases': total_bases
    }


def save_generated_fasta(sequences, output_path='generated.fasta'):
    """
    Save generated sequences to FASTA file.
    
    Args:
        sequences (list): List of DNA sequences
        output_path (str): Output file path
    """
    records = []
    
    for idx, seq in enumerate(sequences):
        gc = DNAEncoder.compute_gc_content(seq)
        record = SeqRecord(
            Seq(seq),
            id=f"generated_{idx+1:04d}",
            description=f"GC={gc:.2f}%"
        )
        records.append(record)
    
    SeqIO.write(records, output_path, "fasta")
    print(f"✓ Generated sequences saved to {output_path}")


def plot_generated_statistics(stats, save_path='generation_stats.png'):
    """
    Visualize statistics of generated sequences.
    
    Args:
        stats (dict): Statistics from analyze_generated_sequences
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: GC content distribution
    ax = axes[0]
    ax.hist(stats['gc_contents'], bins=30, color='steelblue', 
            alpha=0.7, edgecolor='black')
    ax.axvline(stats['mean_gc'], color='red', linestyle='--', 
              linewidth=2, label=f"Mean: {stats['mean_gc']:.2f}%")
    ax.set_xlabel('GC Content (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('GC Content Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Base frequencies
    ax = axes[1]
    bases = ['A', 'C', 'G', 'T']
    freqs = [stats['base_frequencies'][b] for b in bases]
    colors = ['green', 'blue', 'orange', 'red']
    
    bars = ax.bar(bases, freqs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Base Composition', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(freqs) * 1.1])
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generation statistics plot saved to {save_path}")


def print_generation_summary(stats):
    """Print summary of generated sequence statistics."""
    print("\n" + "="*60)
    print("GENERATED SEQUENCES STATISTICS")
    print("="*60)
    print(f"\nGC Content:")
    print(f"  Mean:   {stats['mean_gc']:.2f}%")
    print(f"  Std:    {stats['std_gc']:.2f}%")
    print(f"  Min:    {np.min(stats['gc_contents']):.2f}%")
    print(f"  Max:    {np.max(stats['gc_contents']):.2f}%")
    print(f"\nBase Frequencies:")
    for base, freq in stats['base_frequencies'].items():
        print(f"  {base}: {freq:.4f} ({freq*100:.2f}%)")
    print(f"\nTotal bases analyzed: {stats['total_bases']:,}")
    print("="*60)