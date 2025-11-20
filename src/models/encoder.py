"""
Encoder network for Hierarchical VAE.

Extracts hierarchical latent representations from input data.
"""

import torch
import torch.nn as nn


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder with three latent levels.
    
    Args:
        input_dim (int): Input dimension (e.g., 4096 for 1024bp one-hot)
        latent_dims (list): Dimensions for each latent level [L1, L2, L3]
        dropout (float): Dropout probability
        
    Example:
        >>> encoder = HierarchicalEncoder(4096, [256, 512, 1024])
        >>> x = torch.randn(32, 4096)
        >>> latents, params = encoder(x)
    """
    
    def __init__(self, input_dim=4096, latent_dims=None, dropout=0.3):
        super().__init__()
        
        if latent_dims is None:
            latent_dims = [256, 512, 1024]
        
        self.latent_dims = latent_dims
        
        # Encoder stages
        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.enc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.enc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Latent projections (mu and logvar for reparameterization)
        self.z1_mu = nn.Linear(512, latent_dims[0])
        self.z1_logvar = nn.Linear(512, latent_dims[0])
        
        self.z2_mu = nn.Linear(1024, latent_dims[1])
        self.z2_logvar = nn.Linear(1024, latent_dims[1])
        
        self.z3_mu = nn.Linear(2048, latent_dims[2])
        self.z3_logvar = nn.Linear(2048, latent_dims[2])
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Encode input to hierarchical latents.
        
        Returns:
            tuple: (latents, params)
                latents: (z1, z2, z3) sampled latent vectors
                params: [(mu1, logvar1), (mu2, logvar2), (mu3, logvar3)]
        """
        # Forward through encoder
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        
        # Extract latents
        z1_mu = self.z1_mu(h3)
        z1_logvar = self.z1_logvar(h3)
        z1 = self.reparameterize(z1_mu, z1_logvar)
        
        z2_mu = self.z2_mu(h2)
        z2_logvar = self.z2_logvar(h2)
        z2 = self.reparameterize(z2_mu, z2_logvar)
        
        z3_mu = self.z3_mu(h1)
        z3_logvar = self.z3_logvar(h1)
        z3 = self.reparameterize(z3_mu, z3_logvar)
        
        latents = (z1, z2, z3)
        params = [(z1_mu, z1_logvar), (z2_mu, z2_logvar), (z3_mu, z3_logvar)]
        
        return latents, params