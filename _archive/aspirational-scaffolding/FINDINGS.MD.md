  
`
# HVAE v1: Emergent Representation Learning on Genomic Data
  
## Status: COMPLETED — Negative Result

This experiment ran to completion. The model trained for 50 epochs.`

It did not produce emergent representations. This document explains why and what was learned.

  
---

## Research Question

Can a hierarchical variational autoencoder develop self-organised latent structure when trained on genomic sequences without semantic labels or human-imposed categories?

## Architecture

- 3-level Hierarchical VAE: 256d → 512d → 1024d latent spaces
- Input: 4096d (1024bp one-hot encoded DNA)
- 28,471,808 parameters
- β-annealing schedule (linear warmup over 20 epochs)
- Optimiser: AdamW, lr=1e-3
- Data: C. elegans genome (~100Mbp, ~100k windows)

## Results

### Training Convergence

- Final training loss: 748.1270
- Final validation loss: 748.0401
- Loss flatlined; no meaningful convergence achieved

### Latent Space Collapse

- KL divergence (total): 0.0029
- Per-level KL: all ≈ 0
- **Interpretation:** The model learned to bypass the latent distribution.

The reparameterization trick sampled noise that the decoder ignored.

### Reconstruction

- Accuracy: **0.00%**
- The model did not learn to reconstruct DNA sequences.
- Encoder-decoder weights did not encode meaningful inverse operations.

### Generation

- 200 sequences sampled from prior N(0,1)
- Mean GC content: **0.00%** (target: 36%)
- Generated sequences are statistically indistinguishable from random noise  

### "Clustering"

- Silhouette scores: ~0.001 (level 1), -0.003 (level 3)
- No self-organised clusters emerged
- UMAP projections show no manifold structure

## What Went Wrong

1. **Posterior collapse.** β-annealing was insufficient. The KL term was too weak to force structured latents against the high reconstruction pressure of genomic data.

2. **Discrete data in a continuous space.** DNA is categorical (A/C/G/T). One-hot encoding doesn't create a smooth manifold. VAEs assume continuous structure that doesn't exist in genomic sequences.

3. **No inductive bias for sequence.** The architecture used fully-connected layers throughout. No convolutions, no recurrence, no attention — nothing that knows DNA has position-dependent motifs and long-range correlations.

## Scientific Value

This negative result directly motivated the SimAE successor architecture.

The key insight: KL divergence *prevents* structure discovery on this data type. A contrastive objective (NT-Xent) that pushes similar sequences together in latent space is a better inductive bias than KL divergence toward N(0,1).

## Files

- `src/` — Empty. The aspirational modular structure was never populated.
- `HVAE-TRAINING-FIRST_RUN/Hierarchical_VAE.py` — Original monolithic model
- `scripts/` — CLI training/evaluation/generation scripts (reference `src/` modules that don't exist; non-functional without manual path fixes)`
- `notebooks/` — Multiple versions of the same notebook (`_complete`,``_unified`, `colab`, `kaggle`, `v1c`, `v1g`, `v1o`, `v1p`). ``
- `notebooks/hvae_v1c/` contains the actual training outputs. ``
- `HVAE-TRAINING-FIRST_RUN/` Training outputs: loss curves, UMAP plots, intrinsic dimensionality analysis, screenshots

## Conclusion

training dynamics is insufficient to produce self-organised concepts from  raw genomic sequence. Structure requires either:`
- Stronger inductive biases (convolutions, transformers)`
- Different objectives (contrastive learning)`
- Different data modalities (language, images with smooth manifolds)`
 
This experiment is **complete**. No further training will improve the result. The architecture has been superseded by SimAE v1 (contrastive autoencoder, 128d latent, NT-Xent loss).`

---

*Completed: [APRIL 2026]*

*Author: Error-404*

*License: GPLv3*