"""
Tests for model architectures.
"""

import torch
import pytest
from src.models import HierarchicalVAE


def test_hierarchical_vae_forward():
    """Test forward pass works."""
    model = HierarchicalVAE(input_dim=4096, latent_dims=[256, 512, 1024])
    x = torch.randn(4, 4096)
    
    recon, latents, params = model(x)
    
    assert recon.shape == (4, 4096)
    assert len(latents) == 3
    assert len(params) == 3


def test_latent_dimensions():
    """Test latent dimensions are correct."""
    model = HierarchicalVAE(input_dim=4096, latent_dims=[128, 256, 512])
    x = torch.randn(2, 4096)
    
    _, latents, _ = model(x)
    
    assert latents[0].shape == (2, 128)
    assert latents[1].shape == (2, 256)
    assert latents[2].shape == (2, 512)


def test_generation():
    """Test sampling from prior."""
    model = HierarchicalVAE(input_dim=4096, latent_dims=[256, 512, 1024])
    
    samples = model.sample(num_samples=5, device='cpu')
    
    assert samples.shape == (5, 4096)


if __name__ == '__main__':
    pytest.main([__file__])