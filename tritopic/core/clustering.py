"""
Consensus Leiden Clustering
============================

Robust community detection with:
- Leiden algorithm (better than Louvain)
- Consensus clustering for stability
- Resolution parameter tuning
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score
from collections import Counter


class ConsensusLeiden:
    """
    Leiden clustering with consensus for stability.
    
    Runs multiple Leiden clusterings with different seeds and combines
    results using consensus clustering. This dramatically improves
    reproducibility and reduces sensitivity to random initialization.
    
    Parameters
    ----------
    resolution : float
        Resolution parameter for Leiden. Higher = more clusters. Default: 1.0
    n_runs : int
        Number of consensus runs. Default: 10
    random_state : int
        Random seed for reproducibility. Default: 42
    consensus_threshold : float
        Minimum agreement ratio for consensus. Default: 0.5
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        n_runs: int = 10,
        random_state: int = 42,
        consensus_threshold: float = 0.5,
        low_memory: bool = False,
    ):
        self.resolution = resolution
        self.n_runs = n_runs
        self.random_state = random_state
        self.consensus_threshold = consensus_threshold
        self.low_memory = low_memory
        
        self.labels_: np.ndarray | None = None
        self.stability_score_: float | None = None
        self._all_partitions: list[np.ndarray] = []
    
    def fit_predict(
        self,
        graph: "igraph.Graph",
        min_cluster_size: int = 5,
        resolution: float | None = None,
    ) -> np.ndarray:
        """
        Fit Leiden clustering with consensus.
        
        Parameters
        ----------
        graph : igraph.Graph
            Input graph with edge weights.
        min_cluster_size : int
            Minimum cluster size. Smaller clusters become outliers.
        resolution : float, optional
            Override default resolution.
            
        Returns
        -------
        labels : np.ndarray
            Cluster assignments. -1 for outliers.
        """
        import leidenalg as la
        
        res = resolution or self.resolution
        n_nodes = graph.vcount()
        
        # Run multiple Leiden clusterings
        self._all_partitions = []
        
        for run in range(self.n_runs):
            seed = self.random_state + run
            
            # Run Leiden
            partition = la.find_partition(
                graph,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=res,
                seed=seed,
            )
            
            # Convert to labels
            labels = np.array(partition.membership)
            self._all_partitions.append(labels)
        
        # Compute consensus
        self.labels_ = self._compute_consensus(self._all_partitions)
        
        # Handle small clusters as outliers
        self.labels_ = self._handle_small_clusters(self.labels_, min_cluster_size)
        
        # Compute stability score
        self.stability_score_ = self._compute_stability()
        
        return self.labels_
    
    def _compute_consensus(self, partitions: list[np.ndarray]) -> np.ndarray:
        """
        Compute consensus partition from multiple runs.

        Uses a sparse co-occurrence matrix and hierarchical clustering.
        Scales to large datasets by building the co-occurrence in sparse
        form and converting only the upper triangle to a condensed distance
        vector for ``linkage``.

        When ``self.low_memory`` is True, the condensed distance is built
        directly from the sparse co-occurrence (float32), skipping the
        N×N dense intermediate.  Output is numerically equivalent.
        """
        from scipy.sparse import csr_matrix as sp_csr
        from scipy.spatial.distance import squareform

        n_nodes = len(partitions[0])
        n_runs = len(partitions)

        # Build co-occurrence matrix efficiently using sparse outer products.
        # For each partition, create a cluster-membership indicator matrix M
        # (n_nodes × n_clusters) and accumulate M @ M.T.  The resulting
        # matrix stores how often each pair of nodes was co-clustered.
        ones_dtype = np.float32 if self.low_memory else np.float64
        co_occur = None
        for partition in partitions:
            unique_ids = np.unique(partition)
            cluster_map = {cid: idx for idx, cid in enumerate(unique_ids)}
            cols = np.array([cluster_map[c] for c in partition])
            rows = np.arange(n_nodes)
            data = np.ones(n_nodes, dtype=ones_dtype)
            M = sp_csr((data, (rows, cols)), shape=(n_nodes, len(unique_ids)))
            # M @ M.T is the co-membership matrix for this partition (sparse)
            co_run = M.dot(M.T)
            if co_occur is None:
                co_occur = co_run
            else:
                co_occur = co_occur + co_run

        if self.low_memory:
            # Build condensed distance directly from sparse co-occurrence.
            # Symmetrize in sparse form (guards against FP asymmetry, same
            # as the dense (distance + distance.T)/2 step below).
            co_occur = (co_occur + co_occur.T) * 0.5
            coo = co_occur.tocoo()
            coo.sum_duplicates()

            # Default 1.0 = max distance for pairs that never co-clustered
            # in any of n_runs partitions.
            # float64 (not float32): scipy.linkage calls _convert_to_double on
            # non-float64 input, creating a hidden copy that doubles peak RAM.
            # float64 from the start lets scipy reuse the array in-place.
            n_pairs = n_nodes * (n_nodes - 1) // 2
            condensed = np.ones(n_pairs, dtype=np.float64)

            # Upper triangle only.  Condensed index for (i,j) with i<j is
            # i*n - i*(i+1)/2 + (j-i-1), matching scipy.squareform's layout.
            mask = coo.row < coo.col
            i = coo.row[mask].astype(np.int64)
            j = coo.col[mask].astype(np.int64)
            v = coo.data[mask].astype(np.float64) / float(n_runs)
            idx = n_nodes * i - i * (i + 1) // 2 + (j - i - 1)
            condensed[idx] = 1.0 - v
            np.clip(condensed, 0.0, 1.0, out=condensed)

            # Free sparse workspace before linkage allocates its own.
            del co_occur, coo
        else:
            # Original dense path.
            # For very large datasets (>50k) this remains the bottleneck;
            # at that scale set ``low_memory=True`` on TriTopicConfig.
            co_occur_dense = co_occur.toarray() / n_runs
            np.fill_diagonal(co_occur_dense, 1.0)
            distance = 1.0 - co_occur_dense

            # Ensure perfect symmetry and no negative values (floating-point)
            distance = np.clip((distance + distance.T) / 2, 0.0, 1.0)

            condensed = squareform(distance, checks=False)

        # Average linkage tends to work well for consensus
        Z = linkage(condensed, method="average")

        # Cut at threshold that matches approximate number of clusters
        # from the most frequent partition
        n_clusters_list = [len(np.unique(p)) for p in partitions]
        median_n_clusters = int(np.median(n_clusters_list))

        # Find optimal cut
        best_labels = None
        best_score = -1

        for n_clusters in range(max(2, median_n_clusters - 2), median_n_clusters + 3):
            try:
                labels = fcluster(Z, n_clusters, criterion="maxclust")
                labels = labels - 1  # 0-indexed

                # Score by average ARI with original partitions
                ari_scores = [adjusted_rand_score(labels, p) for p in partitions]
                avg_ari = np.mean(ari_scores)

                if avg_ari > best_score:
                    best_score = avg_ari
                    best_labels = labels
            except Exception as e:
                warnings.warn(f"Consensus partition failed for n_clusters={n_clusters}: {e}")
                continue

        if best_labels is None:
            # Fallback: pick the partition with the highest average ARI
            # against all others
            best_fallback_score = -1
            for p in partitions:
                avg = np.mean([adjusted_rand_score(p, q) for q in partitions])
                if avg > best_fallback_score:
                    best_fallback_score = avg
                    best_labels = p

        return best_labels
    
    def _handle_small_clusters(
        self,
        labels: np.ndarray,
        min_size: int,
    ) -> np.ndarray:
        """Mark small clusters as outliers (-1)."""
        result = labels.copy()
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
            
            size = np.sum(labels == cluster_id)
            if size < min_size:
                result[labels == cluster_id] = -1
        
        # Relabel to consecutive integers
        unique_labels = sorted([l for l in np.unique(result) if l != -1])
        label_map = {old: new for new, old in enumerate(unique_labels)}
        label_map[-1] = -1
        
        result = np.array([label_map[l] for l in result])
        
        return result
    
    def _compute_stability(self) -> float:
        """Compute stability score as average pairwise ARI."""
        if len(self._all_partitions) < 2:
            return 1.0
        
        ari_scores = []
        for i in range(len(self._all_partitions)):
            for j in range(i + 1, len(self._all_partitions)):
                ari = adjusted_rand_score(
                    self._all_partitions[i],
                    self._all_partitions[j]
                )
                ari_scores.append(ari)
        
        return float(np.mean(ari_scores))
    
    def find_optimal_resolution(
        self,
        graph: "igraph.Graph",
        resolution_range: tuple[float, float] = (0.1, 2.0),
        n_steps: int = 10,
        target_n_topics: int | None = None,
    ) -> float:
        """
        Find optimal resolution parameter.

        When *target_n_topics* is given, uses binary search for much higher
        precision (O(log n) instead of O(n)).  Falls back to a linear sweep
        only when no target is specified.

        Parameters
        ----------
        graph : igraph.Graph
            Input graph.
        resolution_range : tuple
            Range of resolutions to search.
        n_steps : int
            Number of search steps (binary-search iterations when
            *target_n_topics* is given, linear sweep points otherwise).
        target_n_topics : int, optional
            If provided, find resolution closest to this number of topics.

        Returns
        -------
        optimal_resolution : float
            Best resolution parameter.
        """
        import leidenalg as la

        def _n_clusters_at(res: float) -> int:
            partition = la.find_partition(
                graph,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=res,
                seed=self.random_state,
            )
            return len(set(partition.membership))

        if target_n_topics is not None:
            # Binary search: higher resolution → more clusters
            lo, hi = resolution_range
            best_res, best_diff = lo, abs(_n_clusters_at(lo) - target_n_topics)

            for _ in range(n_steps):
                mid = (lo + hi) / 2
                n_clust = _n_clusters_at(mid)
                diff = abs(n_clust - target_n_topics)

                if diff < best_diff:
                    best_diff = diff
                    best_res = mid

                if n_clust == target_n_topics:
                    return mid
                elif n_clust < target_n_topics:
                    lo = mid
                else:
                    hi = mid

            return best_res
        else:
            # Linear sweep for maximum modularity
            resolutions = np.linspace(resolution_range[0], resolution_range[1], n_steps)
            best_res = resolutions[0]
            best_mod = -float("inf")

            for res in resolutions:
                partition = la.find_partition(
                    graph,
                    la.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=res,
                    seed=self.random_state,
                )
                if partition.modularity > best_mod:
                    best_mod = partition.modularity
                    best_res = res

            return best_res


class HDBSCANClusterer:
    """
    Alternative clustering using HDBSCAN.
    
    Useful for datasets with varying density or many outliers.
    """
    
    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        
        self.labels_: np.ndarray | None = None
        self.probabilities_: np.ndarray | None = None
    
    def fit_predict(
        self,
        embeddings: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Fit HDBSCAN clustering.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings (optionally reduced with UMAP first).
            
        Returns
        -------
        labels : np.ndarray
            Cluster assignments. -1 for outliers.
        """
        import hdbscan
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            **kwargs,
        )
        
        self.labels_ = clusterer.fit_predict(embeddings)
        self.probabilities_ = clusterer.probabilities_
        
        return self.labels_
