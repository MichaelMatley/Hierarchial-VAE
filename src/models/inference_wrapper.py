"""
Inference wrapper for easy model usage without training code.
"""

import torch
import torch.nn as nn
from .hierarchical_vae import HierarchicalVAE
from ..data.dna_encoder import DNAEncoder


class InferenceWrapper(nn.Module):
    """
    Wrapper for trained VAE model providing convenient inference methods.
    
    Example:
        >>> # Load trained model
        >>> checkpoint = torch.load('best_model.pth')
        >>> wrapper = InferenceWrapper.from_checkpoint(checkpoint)
        >>> 
        >>> # Encode sequence
        >>> latents = wrapper.encode_sequence("ATCGATCG" * 128)
        >>> 
        >>> # Reconstruct sequence
        >>> reconstructed = wrapper.reconstruct_sequence("ATCG" * 256)
        >>> 
        >>> # Generate new sequence
        >>> synthetic = wrapper.generate_sequence(temperature=1.0)
    """
    
    def __init__(self, vae_model, device='cuda'):
        super().__init__()
        self.model = vae_model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda'):
        """
        Load model from checkpoint file.
        
        Args:
            checkpoint_path (str): Path to .pth checkpoint
            device (str): Device to load on
            
        Returns:
            InferenceWrapper: Initialized wrapper
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Infer architecture from checkpoint
        state_dict = checkpoint['model_state_dict']
        input_dim = state_dict['enc1.0.weight'].shape[1]
        latent_dims = [
            state_dict['z1_mu.weight'].shape[0],
            state_dict['z2_mu.weight'].shape[0],
            state_dict['z3_mu.weight'].shape[0]
        ]
        
        # Create model
        model = HierarchicalVAE(input_dim=input_dim, latent_dims=latent_dims)
        model.load_state_dict(state_dict)
        
        return cls(model, device)
    
    def encode_sequence(self, sequence_str):
        """
        Encode DNA sequence to latent representation.
        
        Args:
            sequence_str (str): DNA sequence (must be 1024 bp)
            
        Returns:
            dict: Latent representations for each level
        """
        if len(sequence_str) != 1024:
            raise ValueError(f"Sequence must be exactly 1024 bp, got {len(sequence_str)}")
        
        # Encode to one-hot
        encoded = DNAEncoder.one_hot_encode(sequence_str)
        encoded_flat = torch.tensor(encoded.flatten(), dtype=torch.float32).unsqueeze(0)
        
        # Get latents
        with torch.no_grad():
            latents, _ = self.model.encode(encoded_flat.to(self.device))
        
        return {
            'level1': latents[0].cpu().numpy(),
            'level2': latents[1].cpu().numpy(),
            'level3': latents[2].cpu().numpy()
        }
    
    def decode_latents(self, z1, z2, z3):
        """
        Decode latent vectors to sequence.
        
        Args:
            z1, z2, z3: Latent vectors (numpy arrays or tensors)
            
        Returns:
            str: Decoded DNA sequence
        """
        # Convert to tensors
        if not isinstance(z1, torch.Tensor):
            z1 = torch.tensor(z1, dtype=torch.float32)
        if not isinstance(z2, torch.Tensor):
            z2 = torch.tensor(z2, dtype=torch.float32)
        if not isinstance(z3, torch.Tensor):
            z3 = torch.tensor(z3, dtype=torch.float32)
        
        latents = (
            z1.to(self.device),
            z2.to(self.device),
            z3.to(self.device)
        )
        
        # Decode
        with torch.no_grad():
            recon = self.model.decode(latents)
        
        # Convert to sequence
        recon_reshaped = recon[0].cpu().numpy().reshape(4, 1024)
        return DNAEncoder.decode_one_hot(recon_reshaped)
    
    def reconstruct_sequence(self, sequence_str):
        """
        Full encode-decode cycle.
        
        Args:
            sequence_str (str): Input DNA sequence
            
        Returns:
            str: Reconstructed sequence
        """
        latents = self.encode_sequence(sequence_str)
        return self.decode_latents(
            latents['level1'],
            latents['level2'],
            latents['level3']
        )
    
    def generate_sequence(self, temperature=1.0):
        """
        Generate synthetic sequence from prior.
        
        Args:
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated DNA sequence
        """
        with torch.no_grad():
            # Sample from prior
            z1 = torch.randn(1, self.model.latent_dims[0], device=self.device) * temperature
            z2 = torch.randn(1, self.model.latent_dims[1], device=self.device) * temperature
            z3 = torch.randn(1, self.model.latent_dims[2], device=self.device) * temperature
            
            latents = (z1, z2, z3)
            
            # Decode
            generated = self.model.decode(latents)
            generated_np = generated[0].cpu().numpy().reshape(4, 1024)
            
            return DNAEncoder.decode_one_hot(generated_np)