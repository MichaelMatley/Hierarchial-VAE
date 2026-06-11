#!/usr/bin/env python
"""
Generate synthetic sequences from trained VAE prior.

Usage:
    python scripts/generate.py --checkpoint outputs/checkpoints/best_model.pth --num-samples 100
"""

import argparse
import torch
import sys
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.hierarchical_vae import HierarchicalVAE
from data.dna_encoder import DNAEncoder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic sequences from VAE prior'
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of sequences to generate (default: 100)')
    parser.add_argument('--output', type=str, default='generated_sequences.fasta',
                        help='Output FASTA file (default: generated_sequences.fasta)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    
    return parser.parse_args()


def main():
    """Main generation function."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("SYNTHETIC SEQUENCE GENERATION")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    state_dict = checkpoint['model_state_dict']
    input_dim = state_dict['enc1.0.weight'].shape[1]
    latent_dims = [
        state_dict['z1_mu.weight'].shape[0],
        state_dict['z2_mu.weight'].shape[0],
        state_dict['z3_mu.weight'].shape[0]
    ]
    
    model = HierarchicalVAE(input_dim=input_dim, latent_dims=latent_dims)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    print("✓ Model loaded")
    print(f"  Latent dimensions: {latent_dims}")
    
    # Generate sequences
    print(f"\nGenerating {args.num_samples} sequences...")
    
    records = []
    gc_contents = []
    
    with torch.no_grad():
        for i in range(args.num_samples):
            # Sample from prior with temperature scaling
            z1 = torch.randn(1, latent_dims[0], device=args.device) * args.temperature
            z2 = torch.randn(1, latent_dims[1], device=args.device) * args.temperature
            z3 = torch.randn(1, latent_dims[2], device=args.device) * args.temperature
            
            latents = (z1, z2, z3)
            
            # Decode
            generated = model.decode(latents)
            generated_np = generated[0].cpu().numpy().reshape(4, 1024)
            
            # Convert to sequence
            sequence = DNAEncoder.decode_one_hot(generated_np)
            gc_content = DNAEncoder.compute_gc_content(sequence)
            gc_contents.append(gc_content)
            
            # Create record
            record = SeqRecord(
                Seq(sequence),
                id=f"generated_{i+1:04d}",
                description=f"Generated sequence | GC={gc_content:.2f}%"
            )
            records.append(record)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{args.num_samples}")
    
    # Save to FASTA
    print(f"\nSaving sequences to {args.output}...")
    SeqIO.write(records, args.output, "fasta")
    
    # Statistics
    import numpy as np
    print("\nGeneration Statistics:")
    print(f"  Sequences generated: {len(records)}")
    print(f"  Mean GC content: {np.mean(gc_contents):.2f}%")
    print(f"  Std GC content:  {np.std(gc_contents):.2f}%")
    print(f"  Min GC content:  {np.min(gc_contents):.2f}%")
    print(f"  Max GC content:  {np.max(gc_contents):.2f}%")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"  Output: {args.output}")
    print()


if __name__ == '__main__':
    main()