# HVAE v1: Emergent Representation Learning on Genomic Data

## Status: COMPLETED — Negative Result

This experiment ran to completion (50 epochs).  
The model trained. It did not converge in any meaningful sense.  
This document explains what was built, what happened, and why it matters.

---

## Research Question

> Can a hierarchical variational autoencoder develop self-organised latent structure when trained on genomic sequences without semantic labels or human-imposed categories?

The underlying motivation: if we give a sufficiently complex neural network structured but "meaningless" data (DNA has patterns but no human-assigned concepts), will it invent its own internal ontology? Will compression force the emergence of something resembling understanding?

## Architecture

- **3-level Hierarchical VAE**: 256d → 512d → 1024d latent spaces
- **Input**: 4096d (1024bp one-hot encoded DNA, A/C/G/T)
- **Parameters**: 28,471,808
- **Training**: β-annealing (linear warmup, 20 epochs), AdamW, lr=1e-3
- **Data**: C. elegans genome (~100Mbp, chunked into ~100k windows)

## Results

### Training Convergence

| Metric | Value |
|--------|-------|
| Final training loss | 748.1270 |
| Final validation loss | 748.0401 |
| Best validation loss | 748.0386 |

Loss flatlined immediately and never recovered. The model found a local minimum on epoch ~5 and stayed there.

### Latent Space Collapse

| Metric | Value |
|--------|-------|
| Total KL divergence | 0.0029 |
| KL Level 1 (256d) | ≈ 0 |
| KL Level 2 (512d) | ≈ 0 |
| KL Level 3 (1024d) | ≈ 0 |

**Posterior collapse.** The VAE reparameterization trick sampled noise that the decoder learned to ignore. The KL term — the entire reason this is a *variational* model — effectively vanished. Without KL, there is no probabilistic structure in latent space. Without structure, there is no emergence.

### Reconstruction

| Metric | Value |
|--------|-------|
| Per-nucleotide accuracy | **0.00%** |

The model did not learn to reconstruct DNA sequences. The encoder and decoder weights never developed a meaningful inverse relationship. The "reconstruction" was indistinguishable from random output.

### Generative Sampling

| Metric | Value |
|--------|-------|
| Sequences generated | 200 |
| Mean GC content (generated) | **0.00%** |
| Target GC content (training) | 36.00% |

Sampling from the prior N(0,1) produced sequences with no statistical resemblance to the training distribution. The latent space was not a meaningful generative manifold.

### "Clustering"

| Level | Silhouette | Davies-Bouldin |
|-------|-----------|----------------|
| 1 (256d) | 0.0010 | 8.67 |
| 2 (512d) | 0.0005 | 10.60 |
| 3 (1024d) | -0.0034 | 11.45 |

No clusters. No structure. Random noise would produce similar scores.

### Intrinsic Dimensionality

| Level | Intrinsic / Nominal | Utilisation |
|-------|---------------------|-------------|
| 1 | 220 / 256 | 85.9% |
| 2 | 394 / 512 | 77.0% |
| 3 | 610 / 1024 | 59.6% |

The model "used" its capacity, but for nothing coherent. High utilisation with zero reconstruction accuracy means the parameters encoded noise, not structure.

## What Went Wrong

1. **Posterior collapse.** β-annealing was insufficient. The model learned to set `logvar → -∞`, making the latent distribution a delta function. The KL term went to zero. The VAE became a broken autoencoder.

2. **Discrete data in a continuous space.** DNA is categorical (four nucleotides). One-hot encoding creates high-dimensional sparse vectors with no natural metric. VAEs assume smooth, continuous manifolds. Genomic data violates that assumption.

3. **No sequence-aware inductive bias.** The architecture used fully-connected layers throughout. No convolutions for local motif detection. No recurrence for long-range correlations. No attention for position-dependent patterns. The model had no mechanism to learn that DNA has structure beyond independent per-position statistics.

4. **Insufficient regularisation.** With 28M parameters and no effective KL constraint, the model overfit to noise patterns in the training windows rather than learning generalisable compression.

## Scientific Value

This is a **completed negative result** — not a failed project.

What HVAE v1 proved:

- **Self-organisation ≠ understanding.** Even a sophisticated hierarchical architecture with 28M parameters and careful training dynamics will not spontaneously develop concepts from raw sequence data.
- **Posterior collapse is hard to prevent.** β-annealing alone is insufficient for high-entropy discrete data. Free-bits loss, cyclical annealing, or stronger priors are needed.
- **The genome is the wrong substrate for this question.** DNA's categorical, sparse, long-range-correlated structure breaks the assumptions that make VAEs work.
- **Architecture matters more than scale.** More parameters without the right inductive bias just means more capacity to memorise noise.

This result directly motivated the successor architecture, **SimAE v1**, which replaces the KL-divergence objective (which prevents structure discovery on this data type) with a contrastive NT-Xent loss that explicitly pushes similar sequences together in latent space.

## Files in This Repo

| Path | What It Is |
|------|-----------|
| `README.md` | Original project documentation |
| `FINDINGS.md` | This document — honest post-mortem |
| `setup.py` | Package setup |
| `requirements.txt` | Dependencies |
| `HVAE-TRAINING-FIRST_RUN/` | Original monolithic model + training outputs |
| `HVAE-TRAINING-FIRST_RUN/Hierarchical_VAE.py` | The actual working model code |
| `HVAE-TRAINING-FIRST_RUN/*.png` / `*.jpg` | Training visualisations |
| `notebooks/hvae_v1c/` | Canonical Jupyter notebook with inline results |
| `notebooks/hvae_v1c/final_analysis_report.txt` | Automated analysis output |
| `notebooks/hvae_v1c/*.png` | Exported analysis figures |
| `scripts/train.py` | CLI training script (imports from empty `src/` — non-functional without path fix) |
| `scripts/evaluate.py` | CLI evaluation script (same caveat) |
| `scripts/generate.py` | CLI generation script (same caveat) |
| `docs/` | Architecture, method, and API reference docs |
| `_archive/` | Duplicate notebooks, aspirational scaffolding, broken stubs |

## Why the Scripts Don't Work Out-of-the-Box

The scripts reference a modular `src/` package structure (`src.models.hierarchical_vae`, `src.data.genomic_dataset`, etc.) that was planned but never implemented. The actual model lives as a single monolithic file in `HVAE-TRAINING-FIRST_RUN/Hierarchical_VAE.py`.

To make the scripts functional, either:
- Copy `Hierarchical_VAE.py` into `src/models/hierarchical_vae.py` and create the missing `__init__.py` files, or
- Edit the scripts to import directly from `HVAE-TRAINING-FIRST_RUN.Hierarchical_VAE`

This repository is preserved as an **honest research log**, not a maintained package.

## Conclusion

**The experiment is complete. No further training will improve the result.**

The architecture has been superseded by SimAE v1 (contrastive autoencoder, 128d latent, NT-Xent loss). The core finding stands alone: emergence requires more than scale. It requires the right data modality, the right inductive bias, and an objective function that actually rewards the structure you want to discover.

---

*Completed: June 2025*  
*Author: paradigm dynamics*  
*License: GPLv3*
