**docs/api_reference.md**

```markdown
# API Reference

## Models

### HierarchicalVAE

```python
from src.models import HierarchicalVAE

model = HierarchicalVAE(
    input_dim=4096,
    latent_dims=[256, 512, 1024],
    dropout=0.3
)
```

### Methods:

- forward(x) - Full encode-decode pass
- encode(x) - Get latent representations
- decode(latents) - Reconstruct from latents
- sample(num_samples) - Generate from prior

---

## Data

### GenomicDataset

```python
from src.data import GenomicDataset

dataset = GenomicDataset(
    fasta_file='genome.fasta',
    window_size=1024,
    stride=512
)
```

### Methods:

- **len**() - Number of sequences
- **getitem**(idx) - Get encoded sequence
- get_sequence(idx) - Get raw sequence string
- get_statistics() - Dataset statistics

---

## Training

### VAETrainer

```python
from src.training import VAETrainer

trainer = VAETrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=1e-3
)

history = trainer.train(epochs=100)
```

### Methods:

- train(epochs) - Full training loop
- train_epoch(epoch) - Single epoch
- validate_epoch(epoch) - Validation
- save_checkpoint(epoch) - Save model

---

### **E. notebooks/ Directory**

Create minimal starter notebooks:
