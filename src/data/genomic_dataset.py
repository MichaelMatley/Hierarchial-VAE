"""
PyTorch Dataset for genomic sequences.

Handles loading FASTA files, chunking into fixed-length windows,
and converting to one-hot encoded tensors for model input.
"""

import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import numpy as np
from .dna_encoder import DNAEncoder


class GenomicDataset(Dataset):
    """
    PyTorch Dataset for genomic sequence windows.
    
    Extracts fixed-length windows from FASTA files using a sliding window
    approach. Filters out sequences with excessive ambiguous bases.
    
    Args:
        fasta_file (str): Path to FASTA format genome file
        window_size (int): Length of sequence windows (default: 1024)
        stride (int): Sliding window stride (default: 512)
                     stride < window_size creates overlapping windows
        max_samples (int, optional): Maximum sequences to extract (None = all)
        filter_n_threshold (float): Max proportion of N bases allowed (default: 0.1)
        encoding (str): Encoding scheme - 'one_hot' or 'ordinal' (default: 'one_hot')
        flatten (bool): Whether to flatten one-hot encoding (default: True)
        
    Attributes:
        sequences (list): List of extracted sequence strings
        window_size (int): Length of each sequence
        
    Example:
        >>> dataset = GenomicDataset(
        ...     fasta_file='genome.fasta',
        ...     window_size=1024,
        ...     stride=512
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sequence_tensor = dataset[0]
        >>> print(f"Tensor shape: {sequence_tensor.shape}")
    """
    
    def __init__(self, fasta_file, window_size=1024, stride=512,
                 max_samples=None, filter_n_threshold=0.1,
                 encoding='one_hot', flatten=True):
        
        self.window_size = window_size
        self.stride = stride
        self.encoding = encoding
        self.flatten = flatten
        self.sequences = []
        
        print(f"Loading sequences from {fasta_file}...")
        
        # Parse FASTA file
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence = str(record.seq).upper()
            
            # Extract windows with sliding window
            for i in range(0, len(sequence) - window_size + 1, stride):
                if max_samples and len(self.sequences) >= max_samples:
                    break
                
                chunk = sequence[i:i + window_size]
                
                # Quality filter: reject sequences with too many N's
                n_proportion = chunk.count('N') / len(chunk)
                if n_proportion <= filter_n_threshold:
                    self.sequences.append(chunk)
            
            if max_samples and len(self.sequences) >= max_samples:
                break
        
        overlap = window_size - stride
        print(f"✓ Created dataset:")
        print(f"  Sequences:  {len(self.sequences):,}")
        print(f"  Window:     {window_size} bp")
        print(f"  Stride:     {stride} bp")
        print(f"  Overlap:    {overlap} bp ({overlap/window_size*100:.1f}%)")
    
    def __len__(self):
        """Return number of sequences in dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get encoded sequence tensor by index.
        
        Args:
            idx (int): Sequence index
            
        Returns:
            torch.Tensor: Encoded sequence
                         Shape: (4096,) if flattened one-hot
                                (4, 1024) if not flattened
                                (1024,) if ordinal
        """
        sequence = self.sequences[idx]
        
        if self.encoding == 'one_hot':
            encoded = DNAEncoder.one_hot_encode(sequence)
            
            if self.flatten:
                encoded = encoded.flatten()  # (4, 1024) -> (4096,)
        
        elif self.encoding == 'ordinal':
            encoded = DNAEncoder.ordinal_encode(sequence)
        
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
        
        return torch.tensor(encoded, dtype=torch.float32)
    
    def get_sequence(self, idx):
        """
        Get raw sequence string by index.
        
        Args:
            idx (int): Sequence index
            
        Returns:
            str: DNA sequence string
        """
        return self.sequences[idx]
    
    def get_statistics(self):
        """
        Compute dataset statistics.
        
        Returns:
            dict: Statistics including GC content, base frequencies
        """
        gc_contents = []
        base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        
        for seq in self.sequences[:1000]:  # Sample for speed
            gc_contents.append(DNAEncoder.compute_gc_content(seq))
            
            for base in seq:
                if base in base_counts:
                    base_counts[base] += 1
        
        total_bases = sum(base_counts.values())
        
        return {
            'num_sequences': len(self.sequences),
            'window_size': self.window_size,
            'mean_gc_content': np.mean(gc_contents),
            'std_gc_content': np.std(gc_contents),
            'base_frequencies': {b: c/total_bases for b, c in base_counts.items()}
        }
