# Architecture Documentation

## Hierarchical VAE Structure

### Overview

The Hierarchical VAE consists of three main components:

1. **Encoder**: Maps input (4096d) to three latent spaces
2. **Latent Spaces**: Three hierarchical levels [256d, 512d, 1024d]
3. **Decoder**: Reconstructs input from concatenated latents

### Network Architecture

Input (4096d one-hot encoded sequence)
↓
Encoder Stage 1: Linear(4096 → 2048) + LayerNorm + GELU + Dropout
↓
Encoder Stage 2: Linear(2048 → 1024) + LayerNorm + GELU + Dropout
↓
Encoder Stage 3: Linear(1024 → 512) + LayerNorm + GELU + Dropout
↓
├─→ Latent Level 1 (256d): z1_mu, z1_logvar → reparameterize → z1
├─→ Latent Level 2 (512d): z2_mu, z2_logvar → reparameterize → z2
└─→ Latent Level 3 (1024d): z3_mu, z3_logvar → reparameterize → z3
Concatenate [z1, z2, z3] → (1792d)
↓
Decoder Stage 1: Linear(1792 → 512) + LayerNorm + GELU + Dropout
↓
Decoder Stage 2: Linear(512 → 1024) + LayerNorm + GELU + Dropout
↓
Decoder Stage 3: Linear(1024 → 2048) + LayerNorm + GELU + Dropout
↓
Output Layer: Linear(2048 → 4096)
↓
Reconstruction (4096d)

### Design Choices

### Why LayerNorm instead of BatchNorm?

- More stable with small batch sizes
- No train/eval mode complications
- Better for variable sequence lengths

### Why GELU instead of ReLU?

- Smoother gradients
- Better performance in transformers and modern architectures
- Empirically works better for VAEs

### Why three hierarchical levels?

- Level 1: Global patterns (most abstract)
- Level 2: Local structure (intermediate)
- Level 3: Fine details (least compressed)

### Parameter Count

Default configuration (latent_dims=[256, 512, 1024]):

- Encoder: ~11.5M parameters
- Decoder: ~11.5M parameters
- **Total: ~23M parameters**