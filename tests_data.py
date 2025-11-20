"""
Tests for data loading.
"""

import torch
import pytest
from src.data import DNAEncoder, GenomicDataset
from src.data.synthetic_genome import create_synthetic_genome
import tempfile
import os


def test_dna_encoder():
    """Test DNA encoding/decoding."""
    sequence = "ATCGATCG"
    
    encoded = DNAEncoder.one_hot_encode(sequence)
    decoded = DNAEncoder.decode_one_hot(encoded)
    
    assert decoded == sequence


def test_genomic_dataset():
    """Test dataset creation."""
    # Create temporary genome
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        create_synthetic_genome(10000, f.name)
        temp_file = f.name
    
    try:
        dataset = GenomicDataset(temp_file, window_size=1024, stride=512)
        
        assert len(dataset) > 0
        
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (4096,)
    
    finally:
        os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__])