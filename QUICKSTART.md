1. **Absolute beginner**: Click Colab badge, run all cells
2. **Local installation**: Full walkthrough from clone to results
3. **Minimal script**: Copy-paste Python code
4. **Debugging**: Common issues and solutions
5. **Next steps**: Advanced experiments

Get the Hierarchical VAE running in under 10 minutes.

---

## Option 1: Google Colab (Easiest - No Setup Required)

### Step 1: Open Colab

Click this badge to open the complete notebook:

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

Or manually:

1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. Upload `notebooks/colab_complete.ipynb`

### Step 2: Enable GPU

1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4 or better)
3. Save

### Step 3: Run Everything

Click **Runtime → Run all** and wait ~2-3 hours for training to complete.

**That's it.** The notebook handles:

- Installing dependencies
- Generating synthetic genomic data
- Training the model
- Running all analyses
- Visualizing results

All outputs are saved and can be downloaded at the end.

---

## Option 2: Local Installation (For Your Own Data)

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 16GB+ RAM

### Step 1: Clone Repository

```bash
git clone <https://github.com/MichaelMatley/hierarchical-vae-emergent.git>
cd hierarchical-vae-emergent
```

### Step 2: Install Dependencies

Create virtual environment

```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activat
# Install requirements
pip install -r requirements.txt
# Install package in development mode
pip install -e .
```

### Step 3: Prepare Your Data

Use Synthetic Data (Testing)

```python
from src.data.synthetic_genome import create_synthetic_genom

#Generate 5MB synthetic genome
genome_file = create_synthetic_genome(
length=5_000_000,
output_file='data/synthetic_genome.fasta'
)
```

Use Real Genomic Data: Download C. elegans genome (or any FASTA file):

```python
# Create data directory
mkdir -p data

# Download C. elegans genome (~100MB)
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/985/GCF_000002985.6_WBcel235/GCF_000002985.6_WBcel235_genomic.fna.gz

# Uncompress
gunzip GCF_000002985.6_WBcel235_genomic.fna.gz

# Move to data directory
mv GCF_000002985.6_WBcel235_genomic.fna data/celegans_genome.fasta
```

### Step 4: Train Model

Quick training (small dataset):

```python
python scripts/train.py \
    --data data/synthetic_genome.fasta \
    --epochs 50 \
    --max-samples 50000 \
    --batch-size 128 \
    --checkpoint-dir outputs/checkpoints
```

Full training (complete genome):

```python
python scripts/train.py \
--data data/celegans_genome.fasta \
--epochs 100 \
--batch-size 128 \
--beta-mode linear \
--warmup-epochs 20 \
--checkpoint-dir outputs/checkpoints
```

Training will take 2-4 hours depending on your GPU.

### Step 5: Evaluate Model

```python
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --data data/celegans_genome.fasta \
    --num-samples 10
```

### Step 6: Analyse Results

```python
python scripts/analyze.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --data data/celegans_genome.fasta \
    --output-dir outputs/analysis \
    --num-samples 10000
```

### Step 7: Generate Synthetic Sequences

```python
python scripts/generate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --num-samples 100 \
    --output generated_sequences.fasta \
    --temperature 1.0
```

---

## Option 3: Quick Python Script (Minimal)

Save this as quick_train.py:

```python
#!/usr/bin/env python
"""Minimal training script - copy and run."""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Add src to path
import sys
sys.path.append('src')

from models.hierarchical_vae import HierarchicalVAE
from data.genomic_dataset import GenomicDataset
from data.synthetic_genome import create_synthetic_genome
from training.trainer import VAETrainer
from training.schedulers import BetaScheduler

# 1. Generate synthetic data
print("Generating synthetic genome...")
genome_file = create_synthetic_genome(
    length=5_000_000,
    output_file='synthetic_genome.fasta'
)

# 2. Load dataset
print("Loading dataset...")
dataset = GenomicDataset(
    fasta_file=genome_file,
    window_size=1024,
    stride=512,
    max_samples=50000
)

# 3. Split dataset
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 4. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# 5. Create model
print("Creating model...")
model = HierarchicalVAE(
    input_dim=4096,
    latent_dims=[256, 512, 1024],
    dropout=0.3
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 6. Create trainer
beta_scheduler = BetaScheduler(mode='linear', warmup_epochs=20)

trainer = VAETrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    lr=1e-3,
    beta_scheduler=beta_scheduler,
    patience=15,
    checkpoint_dir='checkpoints'
)

# 7. Train
print("Starting training...")
history = trainer.train(epochs=50)

print("\n✓ Training complete!")
print(f"Best val loss: {trainer.best_val_loss:.4f}")
print(f"Checkpoints saved to: checkpoints/")
```

Run it:

```python
python quick_train.py
```

Understanding the Output after training, you’ll have:
Directory Structure

```python
outputs/
├── checkpoints/
│   ├── best_model.pth           # Best model (use this)
│   ├── latest_checkpoint.pth    # Most recent checkpoint
│   └── training_history.json    # Loss curves data
│
├── analysis/
│   ├── intrinsic_dimensionality.png  # How much capacity is used
│   ├── latent_umap.png               # 2D projection of latent space
│   ├── clustering_analysis.png       # Self-organization quality
│   └── analysis_results.json         # Numerical results
│
└── figures/
    ├── training_history.png          # Loss curves
    ├── reconstruction_quality.png    # Input vs output comparison
    ├── latent_interpolation.png      # Smooth transitions
    └── generated_sequences.fasta     # Synthetic sequences
```

---

## Key Files

best_model.pth - Your trained model

Load with: 

```python
torch.load('outputs/checkpoints/best_model.pth')
```

Use for inference, analysis, generation
training_history.json 

Training metrics

- Reconstruction loss over time
- KL divergence per hierarchical level
- β-annealing schedule
- Learning rate changes

analysis_results.json - Latent space analysis

- Intrinsic dimensionality (how much capacity is used)
- Clustering quality metrics
- Utilisation percentages

## What to Look For

### Good Training Signs

✅ Reconstruction loss decreases steadily
•	Should drop from ~0.5 to ~0.05-0.15
✅ KL divergence stabilises (not collapsed)
•	Should be > 0, ideally 10-100 per level
•	If KL = 0, you have posterior collapse (increase β)
✅ Validation loss tracks training loss
•	Gap should be small (<10%)
•	Large gap = overfitting (increase dropout)
✅ Intrinsic dimensionality < nominal dimensionality
•	Level 1 (256d): Intrinsic ~50-150d is good
•	Shows compression is working

### Bad Training Signs

❌ Reconstruction loss stays high (>0.3)

- Model isn’t learning - check data loading
- Try lower learning rate or more epochs

❌ KL divergence = 0 (posterior collapse)

- Model ignoring latent space
- Solution: Increase β, use cyclical annealing

❌ Validation loss much higher than training

- Overfitting - increase dropout to 0.5
- Reduce model size or add more data

❌ All intrinsic dims = nominal dims

- No compression happening
- Model too large or β too small

## Quick Analysis Commands

### View Training Curves

```python
# If you have matplotlib installed
python -c "
import json
import matplotlib.pyplot as plt

with open('outputs/checkpoints/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))
plt.subplot(131); plt.plot(history['train_loss']); plt.title('Total Loss')
plt.subplot(132); plt.plot(history['train_recon']); plt.title('Reconstruction')
plt.subplot(133); plt.plot(history['train_kl']); plt.title('KL Divergence')
plt.tight_layout(); plt.savefig('quick_plots.png'); print('Saved to quick_plots.png')
"
```

### Check Model Size

```python
python -c "
import torch
checkpoint = torch.load('outputs/checkpoints/best_model.pth', map_location='cpu')
total = sum(p.numel() for p in checkpoint['model_state_dict'].values())
print(f'Total parameters: {total:,}')
print(f'Model size: {torch.save(checkpoint, 'temp.pth') or 0} bytes')
import os; os.remove('temp.pth')
"
```

### Test Reconstruction Quality

```python
import torch
from src.models.hierarchical_vae import HierarchicalVAE
from src.data.genomic_dataset import GenomicDataset
from src.data.dna_encoder import DNAEncoder

# Load model
checkpoint = torch.load('outputs/checkpoints/best_model.pth')
model = HierarchicalVAE(input_dim=4096, latent_dims=[256, 512, 1024])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load data
dataset = GenomicDataset('data/synthetic_genome.fasta', window_size=1024, stride=512)

# Test reconstruction
x = dataset[0].unsqueeze(0)
with torch.no_grad():
    recon, _, _ = model(x)

# Compare
orig = x.numpy().reshape(4, 1024)
rec = recon.numpy().reshape(4, 1024)

orig_seq = DNAEncoder.decode_one_hot(orig)
rec_seq = DNAEncoder.decode_one_hot(rec)

accuracy = sum(o == r for o, r in zip(orig_seq, rec_seq)) / len(orig_seq)
print(f"Original:      {orig_seq[:60]}...")
print(f"Reconstructed: {rec_seq[:60]}...")
print(f"Accuracy: {accuracy:.2%}")
```

## Common Issues & Solutions

### “CUDA out of memory”

Solution 1: Reduce batch size

```python
--batch-size 64  # Instead of 128
```

Solution 2: Reduce model size

```python
--latent-dims 128 256 512  # Instead of 256 512 1024
```

### “Training is very slow”

Check GPU usage:

```python
nvidia-smi
```

Should show high GPU utilization (>80%). If not:

- Ensure CUDA is installed:
    - torch.cuda.is_available() returns True
    - Increase batch size (if memory allows)Increase num_workers in data loaders

### “Reconstruction accuracy is low (<50%)”

This is normal for genomic data:

- 60-70% accuracy is good
- 70-80% is excellent
- 90% suggests memorization (too much capacity)

The model learns patterns, not exact sequences.

### “Posterior collapse (KL = 0)”
Solutions:

1. Use cyclical β-annealing:

```python
--beta-mode cyclical
```

2.	Increase β gradually:

```python
--max-beta 2.0 --warmup-epochs 30
```

3.	Use free-bits loss (edit code to use free_bits_vae_loss)

```python
# Train multiple models with different β values
for beta in 0.1 0.5 1.0 2.0 5.0; do
    python scripts/train.py \
        --data data/genome.fasta \
        --max-beta $beta \
        --checkpoint-dir outputs/beta_$beta
done
```

Compare clustering quality and reconstruction accuracy.

---

## Try Different Architectures

```python
# Smaller model
--latent-dims 128 256 512

# Larger model
--latent-dims 512 1024 2048

# Deeper hierarchy (requires code modification)
--latent-dims 128 256 512 1024
```

## Test on Different Data

Protein sequences (amino acids instead of nucleotides)

Time series data (financial, sensor data)

Text as bytes (character-level language modeling)

## Advanced Analysis

```python
# Dimension importance ablation
from src.analysis.ablation import dimension_importance_ablation

results = dimension_importance_ablation(
    model, test_loader, device='cuda', level=0
)

# Manifold continuity
from src.analysis.manifold import test_manifold_continuity

manifold_results = test_manifold_continuity(
    model, test_loader, device='cuda'
)

# Generate from prior
from src.analysis.generation import generate_from_prior

sequences = generate_from_prior(
    model, num_samples=100, temperature=1.5
)
```

---

## Getting Help

### Check Logs

```python
# View training logs
cat outputs/logs/training.log

# Last 50 lines
tail -n 50 outputs/logs/training.log
```

### Debug Mode

Add debug flag to any script (requires modification) or run in Python with:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Questions

Q: How long should training take?
A: 2-4 hours for 100K sequences on a GPU. CPU training: 10-20x slower.

Q: How much data do I need?
A: Minimum 10K sequences, optimal 100K+. More data = better patterns.

Q: Can I train on CPU?
A: Yes, but expect 10-20x slower training. Use --device cpu.

Q: What if I don’t have genomic data?
A: Use create_synthetic_genome() or try with any structured sequential data.

Q: What GPU do I need?
A: Minimum 8GB VRAM (T4, RTX 3070). Ideal: 16GB+ (A100, V100).

---

### Complete Example: Start to Finish

```python
# 1. Setup
git clone https://github.com/yourusername/hierarchical-vae-emergent.git
cd hierarchical-vae-emergent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate test data
python -c "
from src.data.synthetic_genome import create_synthetic_genome
create_synthetic_genome(5_000_000, 'data/test_genome.fasta')
"

# 3. Train model (fast test)
python scripts/train.py \
    --data data/test_genome.fasta \
    --epochs 20 \
    --max-samples 10000 \
    --batch-size 64 \
    --checkpoint-dir outputs/test

# 4. Evaluate
python scripts/evaluate.py \
    --checkpoint outputs/test/best_model.pth \
    --data data/test_genome.fasta \
    --num-samples 5

# 5. Generate synthetic sequences
python scripts/generate.py \
    --checkpoint outputs/test/best_model.pth \
    --num-samples 10 \
    --output test_generated.fasta

# 6. View generated sequences
head -20 test_generated.fasta
```

Expected time: 15-20 minutes on GPU, 2-3 hours on CPU.

### Resources

- Documentation: docs/ directory
- Example notebooks: notebooks/
- Configuration files: configs/
- GitHub Issues: Report bugs or ask questions

You’re ready to explore emergent representations! 🚀