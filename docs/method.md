# Analysis Methods Documentation

## Intrinsic Dimensionality

**Method:** Principal Component Analysis (PCA)

**Purpose:** Determine how many dimensions the model actually uses

**Calculation:**

```python
pca = PCA()
pca.fit(latents)
cumsum = np.cumsum(pca.explained_variance_ratio_)
intrinsic_dim = np.argmax(cumsum >= 0.95) + 1
```

**Interpretation**:

- Low intrinsic_dim / nominal_dim ratio = efficient compression
- High ratio = underutilised capacity

## Clustering Analysis

**Methods**:

- K-means clustering
- Silhouette score (quality metric)
- Davies-Bouldin index (separation metric)

**Interpretation:**

- High silhouette (>0.5) = well-separated clusters
- Low Davies-Bouldin (<1.0) = compact clusters

## Manifold Continuity

**Method**: Interpolation smoothness test
**Procedure**:

- Sample random pairs of sequences

Interpolate in latent space (20 steps) measure variance of step sizes
Interpretation:

- Low variance = smooth manifold
- High variance = discrete clusters with gaps

## Dimension Importance

**Method**: Ablation study
Procedure:

1. Get baseline reconstruction error
2. For each dimension: set to zero, measure error increase
3. Rank dimensions by importance

**Interpretation**:

- High importance = critical dimension
- Low/negative importance = unused dimension