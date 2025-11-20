"""
Hierarchical Variational Autoencoder with multi-scale latent spaces.

This module implements a three-level hierarchical VAE designed for 
emergent representation learning on structured, high-entropy data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalVAE(nn.Module):
    """
    Multi-scale Variational Autoencoder with hierarchical latent spaces.
    
    Architecture:
        Input (4096) -> Encoder -> 3 latent spaces [256, 512, 1024]
        Latent spaces -> Decoder -> Reconstruction (4096)
    
    The model learns to represent data at multiple levels of abstraction:
        - Level 1 (256d): Most abstract, highly compressed representation
        - Level 2 (512d): Intermediate structural features
        - Level 3 (1024d): Fine-grained local details
    
    Args:
        input_dim (int): Dimension of input data (default: 4096 for 1024bp one-hot)
        latent_dims (list): Dimensions for each latent level [L1, L2, L3]
        dropout (float): Dropout probability for regularization
        
    Attributes:
        enc1, enc2, enc3: Encoder pathway layers
        z*_mu, z*_logvar: Latent distribution parameters
        dec1, dec2, dec3: Decoder pathway layers
        output: Final reconstruction layer
    
    Example:
        >>> model = HierarchicalVAE(input_dim=4096, latent_dims=[256, 512, 1024])
        >>> x = torch.randn(32, 4096)  # Batch of 32 sequences
        >>> recon, latents, params = model(x)
        >>> print(f"Reconstruction shape: {recon.shape}")
        >>> print(f"Latent levels: {[z.shape for z in latents]}")
    """
    
    def __init__(self, input_dim=4096, latent_dims=None, dropout=0.3):
        super().__init__()
        
        if latent_dims is None:
            latent_dims = [256, 512, 1024]
        
        self.input_dim = input_dim
        self.latent_dims = latent_dims
        
        # ========================
        # ENCODER PATHWAY
        # ========================
        
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
        
        # ========================
        # LATENT SPACE PROJECTIONS
        # ========================
        
        # Level 1: Deepest (most abstract)
        self.z1_mu = nn.Linear(512, latent_dims[0])
        self.z1_logvar = nn.Linear(512, latent_dims[0])
        
        # Level 2: Intermediate
        self.z2_mu = nn.Linear(1024, latent_dims[1])
        self.z2_logvar = nn.Linear(1024, latent_dims[1])
        
        # Level 3: Shallowest (fine details)
        self.z3_mu = nn.Linear(2048, latent_dims[2])
        self.z3_logvar = nn.Linear(2048, latent_dims[2])
        
        # ========================
        # DECODER PATHWAY
        # ========================
        
        total_latent_dim = sum(latent_dims)
        
        self.dec1 = nn.Sequential(
            nn.Linear(total_latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dec2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dec3 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Linear(2048, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization for better gradient flow."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Allows gradients to flow through stochastic sampling operation.
        
        Args:
            mu (Tensor): Mean of latent distribution
            logvar (Tensor): Log variance of latent distribution
            
        Returns:
            Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        """
        Encode input into hierarchical latent representations.
        
        Args:
            x (Tensor): Input data [batch_size, input_dim]
            
        Returns:
            tuple: (latents, params)
                latents: Tuple of (z1, z2, z3) sampled latent vectors
                params: List of (mu, logvar) tuples for KL divergence
        """
        # Forward through encoder stages
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        
        # Extract latent parameters at each level
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
    
    def decode(self, latents):
        """
        Decode from hierarchical latent space to reconstruction.
        
        Args:
            latents (tuple): Tuple of (z1, z2, z3) latent vectors
            
        Returns:
            Tensor: Reconstructed input [batch_size, input_dim]
        """
        z = torch.cat(latents, dim=-1)
        
        h = self.dec1(z)
        h = self.dec2(h)
        h = self.dec3(h)
        
        return self.output(h)
    
    def forward(self, x):
        """
        Full forward pass: encode -> sample -> decode
        
        Args:
            x (Tensor): Input data [batch_size, input_dim]
            
        Returns:
            tuple: (reconstruction, latents, params)
                reconstruction: Reconstructed input
                latents: Sampled latent vectors (z1, z2, z3)
                params: Distribution parameters for loss calculation
        """
        latents, params = self.encode(x)
        reconstruction = self.decode(latents)
        
        return reconstruction, latents, params
    
    def sample(self, num_samples, device='cuda'):
        """
        Generate new samples from prior distribution N(0,1).
        
        Args:
            num_samples (int): Number of samples to generate
            device (str): Device to generate on
            
        Returns:
            Tensor: Generated samples [num_samples, input_dim]
        """
        self.eval()
        
        with torch.no_grad():
            # Sample from standard normal
            z1 = torch.randn(num_samples, self.latent_dims[0], device=device)
            z2 = torch.randn(num_samples, self.latent_dims[1], device=device)
            z3 = torch.randn(num_samples, self.latent_dims[2], device=device)
            
            latents = (z1, z2, z3)
            
            # Decode
            samples = self.decode(latents)
        
        return samples