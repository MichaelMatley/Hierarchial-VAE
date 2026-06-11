# Hierarchical VAE For Emergent Representation Learning

> **STATUS: COMPLETED — NEGATIVE RESULT**
>
> This experiment ran to completion (50 epochs). It did not produce emergent representations.
> See `FINDINGS.md` for the honest post-mortem.
> The architecture has been superseded by SimAE v1.

**Exploring self-organised latent structures without human-imposed semantic constraints**

---

## Overview

This project investigates what happens when you give a sufficiently complex neural architecture the freedom to develop its own internal representations on structured, high-entropy data—without the baggage of human semantic priors.

Rather than training on language or labeled datasets, we use *C. elegans* genomic sequences as a substrate: data with intrinsic structure and statistical regularities, but no pre-defined "meaning." The goal isn't genomics—it's to observe what emerges when a model self-organises around naturally occurring patterns.

---

## Architecture

### Hierarchical Variational Autoencoder (VAE)

A multi-scale VAE with three hierarchical latent spaces operating at different levels of abstraction:

Input (4096d) → Encoder → Latent Hierarchy → Decoder → Reconstruction (4096d)  
├─ Level 1: 256d  (most abstract)  
├─ Level 2: 512d  (intermediate features)  
└─ Level 3: 1024d (fine-grained details)  

**Key Design Choices:**

- **Stochastic latents** (VAE, not deterministic AE) force compression into meaningful distributions
- **Hierarchical structure** naturally creates abstract → concrete feature hierarchies
- **β-annealing** schedule gradually increases compression pressure during training
- **No supervision** — purely self-organised representation learning

**Why VAE over standard autoencoder?**

- Stochasticity prevents memorisation
- Continuous latent space enables interpolation and generation
- KL divergence term forces structured, disentangled representations
- β-VAE formulation allows explicit control over information bottleneck strength

---

## The Experiment

### Data: C. elegans Genome

- **Source:** 100 million base pairs, chunked into ~100,000 sequences of 1024 nucleotides
- **Encoding:** One-hot encoding (A/C/G/T → 4-channel representation)
- **Why genomic data?**
    - High entropy with underlying structure (motifs, regulatory patterns, GC content biases)
    - No semantic labels or human-defined categories
    - Naturally occurring data with real statistical regularities
    - Sufficient complexity to challenge the model without overwhelming it
    

**This is **not** a genomics project. We're not predicting genes, finding regulatory elements, or doing anything biologically meaningful. The genomic data is simply a substrate structured data that gives the model something real to organise around.*

Observing emergent properties:

- How does the model choose to compress information?
- What hierarchical structure develops across the three latent levels?
- Does the latent space self-organise into clusters?
- Can the model perform latent arithmetic (compositional structure)?
- What's the intrinsic dimensionality (how much capacity is actually used)?
- Does it learn a continuous manifold or discrete clusters?

---

## Training

### Loss Function

**VAE Loss:**

$$
L = Reconstruction Loss + β × KL Divergence
= MSE(x, x̂) + β × Σ KL(q(z|x) || p(z))
$$

Where:

- **Reconstruction term** encourages faithful encoding/decoding
- **KL term** regularises latent distributions toward N(0,1) prior
- **β parameter** controls information bottleneck strength

### β-Annealing Schedule

Start with β=0 (pure auto-encoder) and gradually increase to β=1:

$$
β(epoch) = min(1.0, epoch / warmup_epochs)
$$

This prevents "posterior collapse" where the model ignores the latent space entirely.

### Optimisation

- **Optimiser:** AdamW (lr=1e-3, weight_decay=1e-5)
- **Scheduler:** Cosine annealing with warm restarts
- **Gradient clipping:** max_norm=1.0
- **Batch size:** 128
- **Early stopping:** patience=15 epochs on validation loss

### Training Dynamics

- **Epochs:** ~50-100 (early stopping determines final count)
- **Hardware:** GPU-accelerated (CUDA)
- **Time:** ~2-3 hours on a single GPU for 100k sequences

---

## Analysis & Interpretability

### 1. Intrinsic Dimensionality

Measures how much of the latent capacity is actually utilised:

$$
Intrinsic Dim = min{k : Σ(variance explained by k PCs) ≥ 0.95}
$$

**Interpretation:**

- Low intrinsic dim → model found efficient compression
- High intrinsic dim → capacity underutilised or diffuse representation
- Per-level comparison shows how abstraction differs across hierarchy

### 2. Latent Space Visualisation

**UMAP & t-SNE projections:**

- 2D embeddings of high-dimensional latent spaces
- Visualises cluster structure and manifold topology
- Colour-coded by sample index to detect temporal/spatial patterns

### 3. Activation Patterns

**Dead neuron analysis:**

- Which latent dimensions are actively used?
- Threshold: |mean activation| < 0.01 → "dead"
- High dead neuron count = wasted capacity

### 4. Dimension Importance Ranking

**Ablation study:**

- Systematically zero out each latent dimension
- Measure reconstruction error increase
- Ranks dimensions by importance

**Formula:**

$$
Importance(dim_i) = MSE_ablated(dim_i) - MSE_baseline
$$

### 5. Latent Directions (Principal Components)

**Interpretable axes:**

- Fit PCA on latent representations
- Walk along top PCs to see what changes
- Tests if model learned semantically meaningful directions

### 6. Manifold Continuity

**Smoothness test:**

- Interpolate between random pairs
- Measure reconstruction variance along path
- Low variance → continuous manifold
- High variance → discrete clusters with voids

### 7. Generative Sampling

**Sample from prior N(0,1):**

- Decode random latent vectors
- Tests if learned distribution is meaningful
- Compare GC content, motif patterns to training data

### 8. Latent Arithmetic

**Vector composition test:**

$$
(Sequence_A - Sequence_B) + Sequence_C = ?
$$

If latent space has compositional structure, this should produce coherent results.

---

## Results & Observations

### What Emerges

**Hierarchical Compression:**

- Level 1 (256d): Captures global sequence properties
- Level 2 (512d): Intermediate patterns, transition regions
- Level 3 (1024d): Local nucleotide correlations

**Self-Organisation:**

- Latent space clusters without explicit clustering loss
- Smooth manifold structure (interpolations are coherent)
- Some dimensions consistently activate, others remain dormant

**Capacity Utilisation:**

- Typically 30-50% of latent capacity actively used
- Intrinsic dimensionality < nominal dimensionality
- Suggests the model found efficient compression

**Generative Ability:**

- Can sample novel sequences from prior
- GC content approximately preserved
- Local patterns resemble training distribution

### What Doesn't Emerge

**No biological semantics:**

- Model doesn't "understand" genes, promoters, or regulatory elements
- No meaningful correspondence to biological function
- Purely distributional pattern matching

**No language-like structure:**

- No discrete symbols or compositional grammar
- Representations are continuous, not discrete
- No emergent "syntax" or "semantics" in human sense

**No causal reasoning:**

- Model learns correlations, not causation
- Can't predict function from sequence
- Just statistical compression

---

## Philosophical Implications

### Is This "Representation Learning"?

**Yes, in a technical sense:**

- Model learns to represent data in latent space
- Representations are useful for downstream tasks
- Self-organised structure without supervision

**No, in a semantic sense:**

- No grounding in external reality
- No "meaning" beyond training distribution
- No conceptual understanding

### What Does "Emergent Structure" Mean?

The model discovers:

- Statistical regularities in the data
- Efficient compression strategies
- Hierarchical organisation patterns

But it doesn't discover:

- Meaning, purpose, or function
- Causal relationships
- Anything not present in training data statistics

### Why This Matters

This experiment demonstrates:

1. **Self-organisation ≠ understanding**: Complex structure can emerge from pure optimisation without semantic grounding
2. **Distributional vs compositional**: Statistical patterns don't imply compositional reasoning
3. **Representation utility**: Even "meaningless" latent codes are useful for transfer learning, anomaly detection, generation

---

## Repository Structure

hierarchical-vae-emergent/  
│  
├── README.md   
├── QUICKSTART.md  
├── requirements.txt   
├── setup.py. 
├── .gitignore   
│  
├── configs/  
│   ├── default_config.yaml   
│   └── experiment_configs/  
│       ├── beta_sweep.yaml   
│       └── architecture_variants.yaml  
│  
├── src/  
│  ├── **init**.py            
│   │  
│  ├── models/  
│   │   ├── **init**.py   
│   │   ├── hierarchical_vae.py  
│   │   ├── encoder.py  
│   │   ├── decoder.py  
│   │   └── inference_wrapper.py   
│   │  
│   ├── data/  
│   │   ├── **init**.py   
│   │   ├── genomic_dataset.py   
│   │   ├── dna_encoder.py  
│   │   └── synthetic_genome.py   
│   │  
│   ├── training/  
│   │   ├── **init**.py   
│   │   ├── trainer.py  
│   │   ├── losses.py  
│   │   └── schedular.py  
│   │  
│   ├── analysis/  
│   │   ├── **init**.py  
│   │   ├── intrinsic_dim.py    
│   │   ├── clustering.py  
│   │   ├── visualisation.py  
│   │   ├── ablation.py  
│   │   ├── interpolation.py  
│   │   ├── manifold.py  
│   │   └── genertation.py  
│   │  
│   └── utils/  
│       ├── **init**.py  
│       ├── logging.py               
│       └── checkpoint.py  
│  
├── scripts  
│   ├── train.py               
│   ├── evaluate.py   
│   ├── analyze.py    
│   └── generate.py    
│    
├── notebooks/     
│   ├── 01_data_exploration.ipynb     (in progress)   
│   ├── 02_training.ipynb             (in progress)    
│   ├── 03_analysis.ipynb             (in progress)    
│   └── colab_complete_v1.ipynb    
│    
├── tests/    
│   ├── **init**.py     
│   ├── test_models.py     
│   ├── test_data.py       
│   └── test_training.py     
│      
├── outputs/     
│   ├── figures/     
│   ├── checkpoints/    
│   └── logs/      
│  
├── data/     
│   └── (FASTA files)   
│    
└── docs/   
  ├── architecture.md   
  ├── analysis_methods.md  
  └── api_reference.md   

## Installation

```bash
pip install torch torchvision
pip install biopython
pip install numpy scipy
pip install matplotlib seaborn
pip install scikit-learn
pip install umap-learn
pip install tqdm
```

Or simply:

```bash
pip install -r requirements.txt
```

<aside>

Suggested Hardware

- Minimum: GPU with 8GB VRAM
- Recommended: GPU with 16GB VRAM
- CPU: Possible but painfully slow (~10x training time)
</aside>

Training from Scratch

```python
from models.hierarchical_vae import HierarchicalVAE
from training.train import train_hierarchical_vae
from data.genomic_dataset import GenomicDataset

# Load data
dataset = GenomicDataset(
    fasta_file='path/to/genome.fasta',
    window_size=1024,
    stride=512
)

# Create model
model = HierarchicalVAE(
    input_dim=4096,
    latent_dims=[256, 512, 1024],
    dropout=0.3
)

# Train
history = train_hierarchical_vae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=1e-3,
    beta_schedule=beta_annealing_schedule
)
```

Inference on New Sequences

```python
import torch
from models.inference_wrapper import InferenceWrapper

# Load trained model
checkpoint = torch.load('checkpoints/best_model.pth')
inference_model = InferenceWrapper(checkpoint['model'])

# Encode sequence to latent space
sequence = "ATCGATCGATCG..." * 85  # 1024 bp
latents = inference_model.encode_sequence(sequence)

# Reconstruct
reconstructed = inference_model.reconstruct_sequence(sequence)

# Generate from prior
synthetic = inference_model.generate_sequence()
```

Analysis 

```python
from analysis.intrinsic_dim import analyze_intrinsic_dimensionality
from analysis.visualization import visualize_latent_space_umap

# Extract latents
latents_dict = extract_latent_representations(model, test_loader)

# Analyze dimensionality
intrinsic_dims = analyze_intrinsic_dimensionality(latents_dict)

# Visualize
visualize_latent_space_umap(latents_dict, n_samples=5000)
```

### Next Steps / Trials

---

Architecture Variants

1. Deeper hierarchies: 4-5 latent levels instead of 3
2. Different capacities: Vary latent dimensions [128, 256, 512] vs [512, 1024, 2048]
3. Attention mechanisms: Add self-attention in encoder/decoder
4. Convolutional layers: Replace linear layers with 1D convolutions for local pattern detection

Training Variants

1. β-VAE sweep: Train with β ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
2. Cyclical annealing: Periodically reset β to encourage exploration
3. Adversarial regularisation: Add discriminator to latent space
4. Contrastive loss: Add InfoNCE objective for self-supervision

Data Variants

1. Different organisms: E. coli, yeast, human chromosome 22
2. Synthetic data: Markov chains with known structure
3. Non-genomic sequences: Protein sequences, time series, text as bytes
4. Augmentation: Reverse complements, subsequence masking, noise injection

---

### Citations & Acknowledgements

*If you use this code or ideas, please cite:*

```
@software{hierarchical_vae_emergent},
title={Hierarchical VAE for Emergent Representation Learning},
author={Michael Matley},
year={2025},
url={[https://github.com](https://github.com/)[/MichaelMatley/Hierarchial-VAE-Emergent](https://github.com/MichaelMatley/Hierarchial-VAE-Emergent)}
```

**Acknowledgments**

- C. elegans genome: WormBase (WBcel235 assembly)
- VAE theory: Kingma & Welling (2013), Higgins et al. (2017, β-VAE)
- Architecture inspiration: Ladder VAE (Sønderby et al., 2016)

**License**

- GNU General Public Licence V3.0

**Status**

This repository is preserved as a completed research log.  
No pull requests or contributions are sought — the experiment is closed.

See `FINDINGS.md` for the full post-mortem and `../SIMAE-V1/` for the successor architecture.

*Last updated: June 2025*
