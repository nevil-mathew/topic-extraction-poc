# TriTopic Visual Guide

Diagrams, flowcharts, and visual explanations.

---

## Pipeline Overview

```
INPUT DOCUMENTS
        ↓
┌───────────────────────────────────────┐
│ STEP 1: EMBEDDING                     │
│ Convert text to vectors               │
│ "Hello world" → [0.23, -0.45, ...]   │
│ Memory: 5-10 GB                       │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│ STEP 2: DIMENSION REDUCTION (UMAP)    │
│ High-dim (768) → Low-dim (50)         │
│ Memory: 2-3 GB temporary              │
└───────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────────────┐
    │   ITERATIVE REFINEMENT LOOP (5 iterations)  │
    │                                             │
    │ ┌───────────────────────────────────────┐  │
    │ │ 2a. BUILD GRAPH                       │  │
    │ │ Compute kNN/SNN similarity            │  │
    │ │ Memory: 2-3 GB                        │  │
    │ └───────────────────────────────────────┘  │
    │                ↓                            │
    │ ┌───────────────────────────────────────┐  │
    │ │ 2b. LEIDEN × 10 RUNS                  │  │
    │ │ Different seeds → 10 clusterings      │  │
    │ │ Memory: 3 GB                          │  │
    │ └───────────────────────────────────────┘  │
    │                ↓                            │
    │ ┌───────────────────────────────────────┐  │
    │ │ 2c. CO-OCCURRENCE MATRIX ⚠️ CRITICAL │  │
    │ │ Dense: 14 GB ✗ (if low_memory=False) │  │
    │ │ Sparse: 2 GB ✓ (if low_memory=True)  │  │
    │ └───────────────────────────────────────┘  │
    │                ↓                            │
    │ ┌───────────────────────────────────────┐  │
    │ │ 2d. SCIPY LINKAGE                     │  │
    │ │ Hierarchical clustering               │  │
    │ │ Memory: 6-10 GB workspace             │  │
    │ └───────────────────────────────────────┘  │
    │                ↓                            │
    │ ┌───────────────────────────────────────┐  │
    │ │ 2e. CONSENSUS + STABILITY             │  │
    │ │ Majority voting on cluster labels     │  │
    │ │ Compute stability_score_              │  │
    │ └───────────────────────────────────────┘  │
    │                ↓                            │
    │ ┌───────────────────────────────────────┐  │
    │ │ 2f. ITERATIVE REFINEMENT              │  │
    │ │ Pull embeddings toward cluster centers│  │
    │ │ Memory: 3-4 GB                        │  │
    │ └───────────────────────────────────────┘  │
    │                ↓                            │
    │ ┌───────────────────────────────────────┐  │
    │ │ 2g. CHECK CONVERGENCE                 │  │
    │ │ Compare with previous iteration (ARI) │  │
    │ │ If converged → EXIT LOOP              │  │
    │ └───────────────────────────────────────┘  │
    │                                             │
    └─────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│ STEP 3: LABEL GENERATION              │
│ Send clusters to LLM for labeling     │
│ Memory: 2-3 GB                        │
└───────────────────────────────────────┘
        ↓
OUTPUT (Labels, Topics, Embeddings)
```

---

## Memory Usage Timeline

### Without Optimization (low_memory=False)

```
Memory (GB)
30  ┌────────────────────────┐
    │                        │  Iteration 1: Spike to 25 GB
25  │        ╱╲              │
    │       ╱  ╲    ╱╲    ╱╲ │  Iteration 2,3,4,5: Same spikes
20  │      ╱    ╲  ╱  ╲  ╱  ╲│
    │     ╱      ╲╱    ╲╱    │
15  │    ╱                    │
    │   ╱                     │
10  │  ╱                      │
    │ ╱                       │
 5  │╱─────────────────────────
    │
 0  └──────────────────────────── →
    Time: Embedding → Loop → Labeling
    
    Peak: 25-30 GB ✗ OOM Risk!
```

### With Optimization (low_memory=True)

```
Memory (GB)
15  ┌─────────────────────────┐
    │                         │  Iteration 1: Spike to 10 GB
12  │     ╱╲        ╱╲    ╱╲  │
    │    ╱  ╲      ╱  ╲  ╱  ╲ │  Iteration 2,3,4,5: Same spikes
10  │   ╱    ╲    ╱    ╲╱    │
    │  ╱      ╲  ╱           │
 7  │ ╱        ╲╱            │
    │╱                       │
 5  │─────────────────────────
    │
 0  └─────────────────────────→
    Time: Embedding → Loop → Labeling
    
    Peak: 10-12 GB ✓ Safe!
```

---

## Co-Occurrence Matrix: Visual Example

### Step 1: Run Leiden 3 Times

```
Documents: [Doc1, Doc2, Doc3, Doc4, Doc5, Doc6]

Run 1 (seed=42):   [A,    A,    B,    B,    C,    C]
                    Doc1-Doc2  Doc3-Doc4  Doc5-Doc6
                    (same cluster)

Run 2 (seed=43):   [A,    B,    A,    B,    C,    C]
                    Doc1-Doc3  Doc2-Doc4  Doc5-Doc6
                    (same cluster)

Run 3 (seed=44):   [A,    A,    B,    B,    C,    C]
                    Doc1-Doc2  Doc3-Doc4  Doc5-Doc6
                    (same cluster)
```

### Step 2: Build Co-Occurrence Matrix

```
Count pairs clustered together:

                Doc1  Doc2  Doc3  Doc4  Doc5  Doc6
           ┌───────────────────────────────────────┐
       Doc1│  -    2     1     0     0     0
       Doc2│  2    -     0     1     0     0
       Doc3│  1    0     -     2     0     0
       Doc4│  0    1     2     -     0     0
       Doc5│  0    0     0     0     -     3
       Doc6│  0    0     0     0     3     -
           └───────────────────────────────────────┘

Legend:
  2 = clustered together 2/3 runs
  1 = clustered together 1/3 runs
  0 = never clustered together
  3 = always clustered together
```

### Step 3: Convert to Distance

```
Distance = 1 - (co-occurrence / total_runs)

                Doc1   Doc2   Doc3   Doc4   Doc5   Doc6
           ┌──────────────────────────────────────────┐
       Doc1│  -     0.33   0.67   1.0    1.0    1.0
       Doc2│ 0.33   -      1.0    0.67   1.0    1.0
       Doc3│ 0.67   1.0    -      0.33   1.0    1.0
       Doc4│ 1.0    0.67   0.33   -      1.0    1.0
       Doc5│ 1.0    1.0    1.0    1.0    -      0.0
       Doc6│ 1.0    1.0    1.0    1.0    0.0    -
           └──────────────────────────────────────────┘

Legend:
  0.0  = Always together (distance = 0)
  0.33 = Often together (distance = 0.33)
  0.67 = Sometimes together
  1.0  = Never together (distance = 1.0)
```

### Step 4: Hierarchical Clustering (Linkage)

```
Tree of similarity (dendrogram):

         All Documents
              |
        ┌─────┴─────┐
        |           |
     Cluster A   Cluster B
     ┌──┴──┐     ┌──┴──┐
   Doc1  Doc2  Doc3  Doc4  Doc5-Doc6
                              |
                          (always
                           together)

Result: Find cut that separates most similar documents
```

### Memory Comparison: Dense vs Sparse

```
DENSE PATH (low_memory=False):
┌─────────────────────────────────────┐
│ Full 6×6 matrix = 36 cells          │
│ For 43,000 docs = 43k×43k = 1.8B    │
│ cells × 8 bytes = 14.4 GB ✗         │
└─────────────────────────────────────┘

SPARSE PATH (low_memory=True):
┌─────────────────────────────────────┐
│ Only non-zero entries stored:       │
│ Doc1-Doc2: 2                        │
│ Doc1-Doc3: 1                        │
│ Doc3-Doc4: 2                        │
│ Doc5-Doc6: 3                        │
│ (6 entries instead of 36)           │
│                                     │
│ For 43,000 docs:                    │
│ Only ~100-200 non-zero entries      │
│ ~100 × 8 bytes = 800 bytes ✓        │
└─────────────────────────────────────┘
```

---

## Iterative Refinement Visualization

### Embedding Evolution

```
ITERATION 1: Original embeddings
Doc1 ●
Doc2 ◆ ← Not quite right, pulled away
Doc3 ▲
Doc4 █
Doc5 ★ ← Topic 1
Doc6 ★

Result: OK clustering, some noise

                           ↓

ITERATION 2: Refined embeddings (pulled toward cluster centers)
Doc1 ●
Doc2 ○ ← Pulled slightly toward Topic 1
Doc3 ▲
Doc4 ◆ ← Pulled slightly toward Topic 2
Doc5 ★ ← Topic 1
Doc6 ★

Result: Better clustering, less noise

                           ↓

ITERATION 3: Further refined
Doc1 ●
Doc2 ● ← Now firmly in Topic 1
Doc3 ▲
Doc4 ▲ ← Now firmly in Topic 2
Doc5 ★ ← Topic 1
Doc6 ★

Result: Clear, tight clusters

                           ↓

ITERATION 4-5: Minor fine-tuning
(Already converged, minimal changes)
```

### ARI Score Progression

```
ARI Score (Convergence)
1.0 ┌────────────────
    │           ▲
0.9 │        ▗▄▄
    │      ▄▀
0.8 │   ▗▄▀
    │ ▄▀
0.7 └──────────────────→
    1    2    3    4    5
         Iteration

Pattern:
  • Iteration 1: Baseline (ARI=None)
  • Iteration 2: Big jump (0.85)
  • Iteration 3: Good improvement (0.92)
  • Iteration 4: Smaller (0.93)
  • Iteration 5: Tiny (0.94) ← Converged
```

---

## Leiden Algorithm: Why Consensus?

### Single Run (Risky)

```
Run Leiden ONCE with random seed:

Cluster 1: [Doc1, Doc2, Doc5]  ← Might be wrong!
Cluster 2: [Doc3, Doc4]
Cluster 3: [Doc6, Doc7]

Problem: One unlucky seed = bad clustering
```

### Consensus (Robust)

```
Run Leiden 10 TIMES with different seeds:

Run 1:  Cluster 1: [Doc1, Doc2, Doc5]
Run 2:  Cluster 1: [Doc1, Doc2]      ← Different!
Run 3:  Cluster 1: [Doc1, Doc2, Doc5]
Run 4:  Cluster 1: [Doc1, Doc2, Doc5]
Run 5:  Cluster 1: [Doc1, Doc2, Doc5]
Run 6:  Cluster 1: [Doc1, Doc2, Doc5]
Run 7:  Cluster 1: [Doc1, Doc2, Doc5]
Run 8:  Cluster 1: [Doc1, Doc2]      ← Different
Run 9:  Cluster 1: [Doc1, Doc2, Doc5]
Run 10: Cluster 1: [Doc1, Doc2, Doc5]

CONSENSUS:
  Doc1 always with Doc2    ← Definite pair
  Doc2 almost always with Doc5 ← Strong pair (8/10)
  
Result: ROBUST clustering (voted by consensus)
```

---

## Stability Score Explained

```
Stability = Average pairwise agreement between 10 runs

Run 1 vs Run 2: ARI = 0.85
Run 1 vs Run 3: ARI = 0.87
Run 1 vs Run 4: ARI = 0.86
...
(all 45 pairs compared: 10×9/2 = 45)

Average = (0.85 + 0.87 + 0.86 + ... ) / 45
        = 0.87

Interpretation:
  0.87 means the 10 runs agree 87% of the time ✓ Good!
```

---

## Configuration Impact

### Resolution Parameter

```
Resolution = 0.3 (VERY LOW)
Result: Few, large clusters

┌──────────────────────────────┐
│     Topic 1 (huge)           │
│ Doc1, Doc2, Doc3, Doc4, ..., │
│ Doc43000                     │
└──────────────────────────────┘

Found: 2 topics (too few!)


Resolution = 1.0 (NORMAL)
Result: Medium clusters

┌─────────────┬────────────────┐
│ Topic 1     │   Topic 2      │
│ Doc1, Doc5  │ Doc2, Doc3, ..│
├─────┬───────┤
│T3   │  T4   │
│ ..  │  ...  │
└─────┴───────┘

Found: 15 topics (good!)


Resolution = 2.5 (VERY HIGH)
Result: Many, small clusters

┌──┬──┬──┬──┬──┬──┬──┬──┬──┐
│T1│T2│T3│T4│T5│T6│T7│T8│..|
└──┴──┴──┴──┴──┴──┴──┴──┴──┘

Found: 200 topics (too many!)
```

---

## Memory vs Speed Trade-off

```
Configuration        Memory  Speed   Quality
─────────────────────────────────────────────
Default              25 GB   10m     High
+ low_memory=True    10 GB   10m     High ✓
+ max_iter=3         10 GB   6m      Good ✓
+ n_neighbors=10     10 GB   5m      Medium
+ reduced_dims=30    10 GB   4m      Medium
+ all above          10 GB   3m      Acceptable

Recommended (balanced):
  low_memory=True
  max_iterations=3
  n_neighbors=15
  Result: 10 GB memory, 6 min, Good quality
```

---

## Convergence Patterns

### Good Convergence

```
ARI Score
1.0  
0.95 │              ────────── Converged (plateau)
0.90 │           ──┐
0.85 │        ──┘
0.80 │     ──┘
0.75 │  ──┘
     │──
     └──────────────→ Iteration
     
Good: ARI increases steadily then plateaus
```

### Problem: No Convergence

```
ARI Score
0.70 │──────────────────────── Stuck (no improvement)
0.65 │──────────────────────── 
0.60 │──────────────────────── 
     │
     └──────────────→ Iteration
     
Bad: ARI doesn't increase
Cause: Wrong resolution, bad embeddings, or graph too weak
```

### Problem: Divergence

```
ARI Score
0.80 │   ──┐
0.70 │──┘  └──┐
0.60 │       └──┐
0.50 │          └───── Going down (refinement failing)
     │
     └──────────────→ Iteration
     
Very Bad: ARI decreases over iterations
Cause: Usually means blend_factor or parameters are wrong
```

---

## Decision Tree: Configuration Help

```
Start here: "How many documents?"

    ↓
    
< 10k docs?
    ├─ YES → Use default config (low_memory not needed)
    └─ NO → Continue
    
< 50k docs?
    ├─ YES → Set low_memory=True
    └─ NO → Set low_memory=True (critical!)
    
Want results in < 5 minutes?
    ├─ YES → max_iterations=2, n_neighbors=10
    └─ NO → max_iterations=5, n_neighbors=25
    
How many topics expected?
    ├─ Few (< 20) → resolution=0.7
    ├─ Medium (20-50) → resolution=1.0
    └─ Many (> 50) → resolution=1.5
    
Have good quality data?
    ├─ YES → use_lexical_view=True
    └─ NO → use_lexical_view=False
```

---

## Summary Diagrams

### Where Memory Goes (43k docs, low_memory=False)

```
Memory Budget: 25 GB peak

Embeddings:         10 GB ███
UMAP:               3 GB  █
Graph building:     2 GB  █
Leiden 10x:         3 GB  █
Co-occurrence:      14 GB ███████ ← BOTTLENECK
scipy.linkage:      6 GB  ███
Refinement:         4 GB  ██
(Some overlap in time)
```

### Where Memory Goes (43k docs, low_memory=True)

```
Memory Budget: 10 GB peak

Embeddings:         10 GB ███
UMAP:               3 GB  █
Graph building:     2 GB  █
Leiden 10x:         3 GB  █
Co-occurrence:      3 GB  █   ← REDUCED (sparse)!
scipy.linkage:      6 GB  ███
Refinement:         4 GB  ██
(Some overlap in time)
```

---

## Checklist for Production Use

```
Before running on large dataset:

Memory:
  ☐ RAM available: Check with `free -h`
  ☐ Set low_memory=True if > 30k docs
  ☐ Monitor with psutil script

Data Quality:
  ☐ No empty documents
  ☐ No duplicate documents
  ☐ Reasonable document length
  ☐ Clean text (no garbage)

Configuration:
  ☐ Set max_iterations sensibly
  ☐ Set convergence_threshold
  ☐ Choose resolution based on expected topics
  ☐ Disable unused features

Monitoring:
  ☐ Memory profiling ready
  ☐ Know what stability score to expect
  ☐ Know what ARI progression should look like
  ☐ Have timeout in case of issues

Testing:
  ☐ Test on small sample first
  ☐ Check results make sense
  ☐ Then run on full dataset
```
