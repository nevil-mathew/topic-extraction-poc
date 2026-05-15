# TriTopic Cheat Sheet

Quick reference for common tasks and concepts.

---

## 🚀 Quick Start

```python
from tritopic import TriTopic, TriTopicConfig

# For large datasets (recommended)
config = TriTopicConfig(low_memory=True)
model = TriTopic(config)
model.fit(documents)

# Access results
print(f"Topics: {len(np.unique(model.labels_))}")
print(f"Stability: {model.stability_score_:.3f}")
```

---

## 📊 What's Happening?

### The Pipeline (5 Steps)

```
1. EMBEDDING      → Convert text to vectors
2. GRAPH BUILDING → Compute document similarity
3. LEIDEN 10x     → Cluster with different random seeds
4. CONSENSUS      → Find agreement between 10 runs
5. REFINE         → Improve for next iteration
                    (Repeat steps 2-5 for 5 iterations)
```

### Memory Spike Locations

```
Default consensus_method="graph" (Lancichinetti-Fortunato):
  ✓ Sparse co-occurrence + Leiden on thresholded graph
  └─ Peak: ~2-3 GB even at 43k docs
  └─ scipy.linkage is NOT called
  └─ Reference: Sci. Rep. 2:336 (2012)

Legacy consensus_method="hierarchical":
  ✗ DANGEROUS: Co-occurrence matrix (step 4)
    └─ Dense: 43k × 43k = 14 GB ← OOM risk
    └─ Sparse + condensed: ~7 GB ✓ (with low_memory=True)

  ✗ MODERATE: linkage (after step 4)
    └─ fastcluster: 2-4 GB workspace (if installed)
    └─ scipy fallback: 6-10 GB workspace

✓ SAFE: All other steps < 3 GB
```

---

## 🎯 Configuration Cheat Sheet

| Goal | Setting | Code |
|------|---------|------|
| **Large dataset** | low_memory | `low_memory=True` |
| **Fast processing** | max_iterations | `max_iterations=3` |
| **More topics** | resolution | `resolution=1.5` |
| **Fewer topics** | resolution | `resolution=0.5` |
| **Better quality** | n_neighbors | `n_neighbors=25` |
| **Faster clustering** | n_neighbors | `n_neighbors=10` |
| **Stop early** | convergence_threshold | `convergence_threshold=0.85` |
| **More consensus** | n_runs | Increase Leiden runs (memory cost) |
| **Memory-safe consensus** | consensus_method | `consensus_method="graph"` (default) |
| **Stricter agreement** | consensus_threshold_tau | `consensus_threshold_tau=0.7` (more conservative) |
| **Looser agreement** | consensus_threshold_tau | `consensus_threshold_tau=0.3` (more edges retained) |
| **Legacy linkage** | consensus_method | `consensus_method="hierarchical"` (small N only) |

---

## 📈 Interpreting Results

### Stability Score

```
model.stability_score_

0.90-1.0  ✓✓✓ Excellent (Leiden runs very consistent)
0.80-0.89 ✓✓  Good (Leiden runs mostly agree)
0.70-0.79 ✓   OK (Some variation, acceptable)
0.60-0.69 ⚠️  Weak (Concerning variation)
<0.60     ✗✗  Bad (Leiden runs very different)
```

### Iteration History (ARI)

```
model._iteration_history

Iteration 1: ARI = None         (baseline)
Iteration 2: ARI = 0.85         (good jump)
Iteration 3: ARI = 0.92         (convergence starting)
Iteration 4: ARI = 0.93         (small improvement)
Iteration 5: ARI = 0.94         (converged - stop here)

Good sign: ARI increases then plateaus
Bad sign: ARI decreases or stays low
```

---

## 💾 Memory Quick Estimate

```
For N documents with low_memory=False:

Memory = (N × N × 8 bytes) / 1e9 GB

Examples:
  10k docs:   10k × 10k × 8 / 1e9 = 0.8 GB ✓
  20k docs:   20k × 20k × 8 / 1e9 = 3.2 GB ✓
  43k docs:   43k × 43k × 8 / 1e9 = 14.7 GB ✗
  50k docs:   50k × 50k × 8 / 1e9 = 20 GB ✗✗

Rule of thumb: >30k docs → MUST use low_memory=True
```

---

## 🔍 Debugging Checklist

```
Problem: Out of Memory
□ Set low_memory=True
□ Reduce max_iterations
□ Reduce dataset size
□ Close other applications

Problem: Unstable Clustering (stability < 0.7)
□ Check data quality
□ Adjust resolution (try 0.8, 1.0, 1.2)
□ Increase n_neighbors
□ Check if n_clusters seems reasonable

Problem: Too Many Topics
□ Lower resolution
□ Increase min_cluster_size

Problem: Too Few Topics
□ Raise resolution
□ Decrease min_cluster_size
□ Check convergence threshold

Problem: Slow Processing
□ Reduce max_iterations
□ Reduce n_neighbors (15→10)
□ Reduce reduced_dims (50→30)
□ Disable unused features (use_lexical_view=False)
```

---

## 📝 Key Concepts

### Leiden Algorithm
A **clustering algorithm** that groups similar items. Non-deterministic = different runs may produce different results. TriTopic runs it 10 times to find consensus.

### Co-Occurrence Matrix
Tracks "how many times did documents A and B end up in same cluster?" across all 10 Leiden runs. Used to find consensus clustering.

### Stability Score
How well the 10 Leiden runs agree (0-1 scale). Higher = more robust clusters.

### ARI (Adjusted Rand Index)
Compares two clusterings (0-1 scale). Shows how much clustering changed between iterations.

### Iterative Refinement
Pull documents toward their cluster centers after each iteration. Improves embeddings, leads to better clusters next iteration.

### Blend Factor
Controls how much to refine embeddings. Decreases over iterations (start aggressive, end gentle).

---

## 🐛 Common Errors

### `MemoryError: Unable to allocate X GB`
```python
# Fix: Use low_memory=True
config = TriTopicConfig(low_memory=True)
```

### `IndexError in co_occurrence matrix`
```python
# Usually means corrupted graph or duplicate documents
# Try: Remove duplicates, check document validity
```

### Clusters are all the same label
```python
# Resolution too low, graph too weak, or bad embeddings
# Try: Increase resolution, increase n_neighbors
config = TriTopicConfig(resolution=1.5, n_neighbors=25)
```

---

## 📊 Monitoring Memory

```python
import psutil
import os

process = psutil.Process(os.getpid())

def print_memory():
    rss = process.memory_info().rss / 1e9
    print(f"Memory: {rss:.1f} GB")

print_memory()
model = TriTopic(config)
model.fit(documents)
print_memory()
```

---

## ⚡ Performance Tips

```python
# Fastest (but lower quality)
config = TriTopicConfig(
    low_memory=True,
    max_iterations=2,
    n_neighbors=10,
    reduced_dims=30,
    use_lexical_view=False,
)

# Balanced (recommended)
config = TriTopicConfig(
    low_memory=True,
    max_iterations=3,
    n_neighbors=15,
    reduced_dims=50,
)

# Best quality (slower)
config = TriTopicConfig(
    low_memory=True,
    max_iterations=5,
    n_neighbors=30,
    reduced_dims=100,
)
```

---

## 📚 File Reference

```
Where to find what:

Iterative Refinement:
  → tritopic/core/model.py
  → _refine_embeddings() method

Leiden Consensus Clustering:
  → tritopic/core/clustering.py
  → ConsensusLeiden class
  → _compute_consensus() method

Co-occurrence Matrix Building:
  → tritopic/core/clustering.py (line 137-196)
  → Both low_memory=True and False paths

Graph Building:
  → tritopic/core/graph_builder.py
  → kNN, SNN, mutual_knn methods
```

---

## ✅ Pre-Flight Checklist

Before running on large dataset:

```
□ Using low_memory=True?
□ Checked RAM available?
□ Set max_iterations reasonably?
□ Disabled unused features?
□ Set convergence_threshold?
□ Have monitoring ready (psutil)?
□ Know expected number of topics?
□ Checked document quality?
```

---

## 🔗 Links

- **Main README**: [LEARNING_GUIDE/README.md](README.md)
- **Leiden Paper**: https://www.nature.com/articles/s41598-019-41695-0
- **scipy.linkage**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
- **UMAP**: https://umap-learn.readthedocs.io/

---

## 💡 Remember

1. **For 30k+ documents**: ALWAYS use `low_memory=True`
2. **Stability > 0.8**: Good sign ✓
3. **ARI increasing**: Converging correctly ✓
4. **Iterations usually converge**: By iteration 3-4, diminishing returns
5. **Different data = different resolution**: Experiment with resolution parameter
