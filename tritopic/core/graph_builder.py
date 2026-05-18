"""
Graph Builder for TriTopic
============================

Constructs similarity graphs using multiple strategies:
- Mutual kNN: Only keep edges where both nodes are in each other's neighborhood
- SNN (Shared Nearest Neighbors): Weight edges by number of shared neighbors
- Multi-view fusion: Combine semantic, lexical, and metadata graphs
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GraphBuilder:
    """
    Build similarity graphs for topic modeling.
    
    Supports multiple graph construction strategies for robust clustering.
    
    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for kNN graph. Default: 15
    metric : str
        Distance metric. Default: "cosine"
    graph_type : str
        Type of graph: "knn", "mutual_knn", "snn", or "hybrid"
    snn_weight : float
        Weight for SNN edges in hybrid mode. Default: 0.5
    """
    
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "cosine",
        graph_type: Literal["knn", "mutual_knn", "snn", "hybrid"] = "hybrid",
        snn_weight: float = 0.5,
        language: str = "english",
        n_jobs: int = -1,
    ):
        from tritopic.utils.stopwords import get_stopwords

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.graph_type = graph_type
        self.snn_weight = snn_weight
        self.language = language
        self.n_jobs = n_jobs

        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=get_stopwords(language),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,  # log(1+tf) dampens common term dominance
        )
    
    def _compute_knn(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute kNN and return (distances, indices, similarities).

        Shared helper so hybrid graphs avoid duplicate kNN computation.
        """
        k = n_neighbors or self.n_neighbors
        n_samples = embeddings.shape[0]

        nn = NearestNeighbors(
            n_neighbors=min(k + 1, n_samples),
            metric=self.metric,
            algorithm="auto",
            n_jobs=self.n_jobs,
        )
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

        if self.metric == "cosine":
            similarities = 1 - distances
        else:
            similarities = 1 / (1 + distances)

        return distances, indices, similarities

    def build_knn_graph(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
    ) -> csr_matrix:
        """
        Build a basic kNN graph (vectorized).

        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings of shape (n_docs, n_dims).
        n_neighbors : int, optional
            Override default n_neighbors.

        Returns
        -------
        adjacency : csr_matrix
            Sparse adjacency matrix with cosine similarity weights.
        """
        n_samples = embeddings.shape[0]
        _, indices, similarities = self._compute_knn(embeddings, n_neighbors)

        # Vectorized construction: exclude self-loops (col 0 is self)
        rows = np.repeat(np.arange(n_samples), indices.shape[1] - 1)
        cols = indices[:, 1:].ravel()
        data = similarities[:, 1:].ravel()

        adjacency = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        return adjacency
    
    def build_mutual_knn_graph(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
        _precomputed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> csr_matrix:
        """
        Build a mutual kNN graph (vectorized).

        Edge (i, j) exists only if i is in j's neighbors AND j is in i's neighbors.
        This removes "one-way" connections that often represent noise.

        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings.
        n_neighbors : int, optional
            Override default n_neighbors.
        _precomputed : tuple, optional
            Pre-computed (distances, indices, similarities) from _compute_knn.

        Returns
        -------
        adjacency : csr_matrix
            Sparse adjacency matrix.
        """
        n_samples = embeddings.shape[0]

        if _precomputed is not None:
            _, indices, similarities = _precomputed
        else:
            _, indices, similarities = self._compute_knn(embeddings, n_neighbors)

        # Build directed kNN adjacency (excluding self = col 0)
        rows_dir = np.repeat(np.arange(n_samples), indices.shape[1] - 1)
        cols_dir = indices[:, 1:].ravel()
        data_dir = similarities[:, 1:].ravel()

        knn_adj = csr_matrix(
            (data_dir, (rows_dir, cols_dir)), shape=(n_samples, n_samples)
        )

        # Mutual = element-wise minimum of knn_adj and knn_adj.T
        # (non-zero only where both directions exist)
        knn_T = knn_adj.T.tocsr()
        mutual = knn_adj.minimum(knn_T)  # keeps mutual edges, min similarity

        # Average forward and reverse similarities for mutual edges
        mutual_avg = (knn_adj + knn_T).multiply(mutual > 0) / 2

        return mutual_avg.tocsr()
    
    def build_snn_graph(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
        _precomputed_indices: np.ndarray | None = None,
    ) -> csr_matrix:
        """
        Build a Shared Nearest Neighbors (SNN) graph.

        Edge weight = number of shared neighbors between two nodes.
        This is very robust against noise and outliers.

        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings.
        n_neighbors : int, optional
            Override default n_neighbors.
        _precomputed_indices : np.ndarray, optional
            Pre-computed kNN indices (internal use by hybrid graph).

        Returns
        -------
        adjacency : csr_matrix
            Sparse adjacency matrix with SNN weights.
        """
        k = n_neighbors or self.n_neighbors
        n_samples = embeddings.shape[0]

        if _precomputed_indices is not None:
            indices = _precomputed_indices
        else:
            nn = NearestNeighbors(
                n_neighbors=min(k + 1, n_samples),
                metric=self.metric,
                algorithm="auto",
                n_jobs=self.n_jobs,
            )
            nn.fit(embeddings)
            _, indices = nn.kneighbors(embeddings)

        # Vectorized SNN: build binary membership matrix M (n_samples × n_samples)
        # where M[i, j] = 1 iff j is a kNN of i (excluding self at col 0).
        # Shared neighbor count: (M @ M.T)[i, j] = |N(i) ∩ N(j)|.
        k_actual = indices.shape[1] - 1
        row_idx = np.repeat(np.arange(n_samples), k_actual)
        col_idx = indices[:, 1:].ravel()
        M = csr_matrix(
            (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
            shape=(n_samples, n_samples),
        )
        snn = M.dot(M.T) / k_actual  # shared-neighbor fraction in [0, 1]
        snn.setdiag(0)
        snn.eliminate_zeros()

        # Restrict to pairs that are kNN-connected in at least one direction
        knn_union = (M + M.T)
        knn_union.data[:] = 1.0
        adjacency = snn.multiply(knn_union)

        return adjacency.tocsr()
    
    def build_hybrid_graph(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
    ) -> csr_matrix:
        """
        Build a hybrid graph combining mutual kNN and SNN.

        Computes kNN once and shares the result between both sub-graphs.

        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings.
        n_neighbors : int, optional
            Override default n_neighbors.

        Returns
        -------
        adjacency : csr_matrix
            Combined adjacency matrix.
        """
        # Compute kNN ONCE and share
        precomputed = self._compute_knn(embeddings, n_neighbors)
        _, indices, _ = precomputed

        mutual_adj = self.build_mutual_knn_graph(
            embeddings, n_neighbors, _precomputed=precomputed
        )
        snn_adj = self.build_snn_graph(
            embeddings, n_neighbors, _precomputed_indices=indices
        )

        # Normalize both
        mutual_max = mutual_adj.max() if mutual_adj.nnz > 0 else 1
        snn_max = snn_adj.max() if snn_adj.nnz > 0 else 1

        if mutual_max > 0:
            mutual_adj = mutual_adj / mutual_max
        if snn_max > 0:
            snn_adj = snn_adj / snn_max

        # Combine
        combined = (1 - self.snn_weight) * mutual_adj + self.snn_weight * snn_adj

        return combined.tocsr()
    
    def build_lexical_matrix(
        self,
        documents: list[str],
    ) -> csr_matrix:
        """
        Build TF-IDF matrix for lexical similarity.
        
        Parameters
        ----------
        documents : list[str]
            Document texts.
            
        Returns
        -------
        tfidf_matrix : csr_matrix
            TF-IDF sparse matrix.
        """
        tfidf_matrix = self._tfidf_vectorizer.fit_transform(documents)
        return tfidf_matrix
    
    def build_lexical_graph(
        self,
        tfidf_matrix: csr_matrix,
        n_neighbors: int | None = None,
    ) -> csr_matrix:
        """
        Build lexical similarity graph from TF-IDF (vectorized mutual kNN).

        Parameters
        ----------
        tfidf_matrix : csr_matrix
            TF-IDF matrix.
        n_neighbors : int, optional
            Override default n_neighbors.

        Returns
        -------
        adjacency : csr_matrix
            Lexical similarity adjacency matrix.
        """
        k = n_neighbors or self.n_neighbors
        n_samples = tfidf_matrix.shape[0]

        nn = NearestNeighbors(
            n_neighbors=min(k + 1, n_samples),
            metric="cosine",
            algorithm="brute",
            n_jobs=self.n_jobs,
        )
        nn.fit(tfidf_matrix)
        distances, indices = nn.kneighbors(tfidf_matrix)

        similarities = 1 - distances

        # Vectorized mutual kNN
        rows_dir = np.repeat(np.arange(n_samples), indices.shape[1] - 1)
        cols_dir = indices[:, 1:].ravel()
        data_dir = similarities[:, 1:].ravel()

        knn_adj = csr_matrix(
            (data_dir, (rows_dir, cols_dir)), shape=(n_samples, n_samples)
        )
        knn_T = knn_adj.T.tocsr()

        # Mutual edges: average similarities where both directions exist
        mutual = knn_adj.minimum(knn_T)
        mutual_avg = (knn_adj + knn_T).multiply(mutual > 0) / 2

        return mutual_avg.tocsr()
    
    def build_metadata_graph(
        self,
        metadata: "pd.DataFrame",
    ) -> csr_matrix:
        """
        Build metadata similarity graph.

        Documents with matching metadata get connected.
        Categorical columns use exact-match (sparse outer product).
        Numerical columns use kNN on normalized values.

        Parameters
        ----------
        metadata : pd.DataFrame
            Metadata DataFrame with same index as documents.

        Returns
        -------
        adjacency : csr_matrix
            Metadata similarity adjacency matrix.
        """
        import pandas as pd
        from scipy.sparse import csr_matrix as sp_csr

        n_samples = len(metadata)
        adjacency = sp_csr((n_samples, n_samples), dtype=float)

        for col in metadata.columns:
            if pd.api.types.is_string_dtype(metadata[col]) or metadata[col].dtype.name == "category":
                # Categorical: sparse one-hot → M @ M.T gives co-membership
                codes = metadata[col].astype("category").cat.codes.values
                valid = codes >= 0  # -1 for NaN
                if not valid.any():
                    continue
                valid_idx = np.where(valid)[0]
                valid_codes = codes[valid]
                n_cats = valid_codes.max() + 1
                M = sp_csr(
                    (np.ones(len(valid_idx)), (valid_idx, valid_codes)),
                    shape=(n_samples, n_cats),
                )
                co_member = M.dot(M.T)
                co_member.setdiag(0)
                adjacency = adjacency + co_member
            else:
                # Numerical: kNN on normalized values (avoids O(n^2) loop)
                values = metadata[col].values.astype(float)
                valid_mask = ~np.isnan(values)
                if valid_mask.sum() < 2:
                    continue
                valid_idx = np.where(valid_mask)[0]
                v = values[valid_idx]
                v_range = v.max() - v.min()
                if v_range < 1e-10:
                    continue
                v_norm = ((v - v.min()) / v_range).reshape(-1, 1)

                k_meta = min(self.n_neighbors, len(valid_idx) - 1)
                if k_meta < 1:
                    continue
                nn_meta = NearestNeighbors(n_neighbors=k_meta + 1, metric="euclidean")
                nn_meta.fit(v_norm)
                dists, idxs = nn_meta.kneighbors(v_norm)

                # Similarity = 1 - distance, keep only > 0.8
                for local_i in range(len(valid_idx)):
                    for j_pos in range(1, idxs.shape[1]):
                        sim = 1.0 - dists[local_i, j_pos]
                        if sim > 0.8:
                            gi = valid_idx[local_i]
                            gj = valid_idx[idxs[local_i, j_pos]]
                            adjacency[gi, gj] = adjacency[gi, gj] + sim
                            adjacency[gj, gi] = adjacency[gj, gi] + sim

        # Normalize
        max_val = adjacency.max()
        if max_val > 0:
            adjacency = adjacency / max_val

        return adjacency.tocsr()
    
    def build_multiview_graph(
        self,
        semantic_embeddings: np.ndarray,
        lexical_matrix: csr_matrix | None = None,
        metadata_graph: csr_matrix | None = None,
        weights: dict[str, float] | None = None,
        precomputed_lexical_adj: csr_matrix | None = None,
    ) -> "igraph.Graph":
        """
        Build combined multi-view graph.
        
        Fuses semantic, lexical, and metadata views into a single graph
        for robust community detection.
        
        Parameters
        ----------
        semantic_embeddings : np.ndarray
            Document embeddings.
        lexical_matrix : csr_matrix, optional
            TF-IDF matrix for lexical view.
        metadata_graph : csr_matrix, optional
            Pre-computed metadata adjacency.
        weights : dict, optional
            Weights for each view. Keys: "semantic", "lexical", "metadata"
            
        Returns
        -------
        graph : igraph.Graph
            Combined weighted graph.
        """
        import igraph as ig
        
        weights = weights or {"semantic": 0.5, "lexical": 0.3, "metadata": 0.2}
        n_samples = semantic_embeddings.shape[0]

        # Build semantic graph
        if self.graph_type == "knn":
            semantic_adj = self.build_knn_graph(semantic_embeddings)
        elif self.graph_type == "mutual_knn":
            semantic_adj = self.build_mutual_knn_graph(semantic_embeddings)
        elif self.graph_type == "snn":
            semantic_adj = self.build_snn_graph(semantic_embeddings)
        else:  # hybrid
            semantic_adj = self.build_hybrid_graph(semantic_embeddings)

        # Normalize
        if semantic_adj.max() > 0:
            semantic_adj = semantic_adj / semantic_adj.max()

        # Determine which views are active and re-normalize weights
        active_weights = {"semantic": weights.get("semantic", 0.5)}
        has_lexical = lexical_matrix is not None and weights.get("lexical", 0) > 0
        has_metadata = metadata_graph is not None and weights.get("metadata", 0) > 0

        if has_lexical:
            active_weights["lexical"] = weights["lexical"]
        if has_metadata:
            active_weights["metadata"] = weights["metadata"]

        # Re-normalize so active weights sum to 1.0
        weight_sum = sum(active_weights.values())
        if weight_sum > 0:
            active_weights = {k: v / weight_sum for k, v in active_weights.items()}

        # Start with semantic
        combined_adj = active_weights["semantic"] * semantic_adj

        # Add lexical if available
        if has_lexical:
            lexical_adj = (
                precomputed_lexical_adj
                if precomputed_lexical_adj is not None
                else self.build_lexical_graph(lexical_matrix)
            )
            if lexical_adj.max() > 0:
                lexical_adj = lexical_adj / lexical_adj.max()
            combined_adj = combined_adj + active_weights["lexical"] * lexical_adj

            # Consensus bonus: edges present in BOTH semantic and lexical
            # views are more reliable -- give them a small boost.
            # Use element-wise minimum (overlap strength) as the bonus.
            overlap = semantic_adj.minimum(lexical_adj)
            if overlap.nnz > 0:
                combined_adj = combined_adj + 0.1 * overlap

        # Add metadata if available
        if has_metadata:
            if metadata_graph.max() > 0:
                metadata_graph = metadata_graph / metadata_graph.max()
            combined_adj = combined_adj + active_weights["metadata"] * metadata_graph
        
        # Convert to igraph — combined_adj is symmetric, so take upper triangle
        # to avoid duplicate edges without a Python-level dict loop.
        combined_adj = combined_adj.tocoo()
        mask = combined_adj.row < combined_adj.col
        edges = list(zip(combined_adj.row[mask].tolist(), combined_adj.col[mask].tolist()))
        weights_list = combined_adj.data[mask].tolist()

        graph = ig.Graph(n=n_samples, edges=edges, directed=False)
        graph.es["weight"] = weights_list

        return graph
    
    def get_feature_names(self) -> list[str]:
        """Get TF-IDF feature names (for keyword extraction)."""
        if hasattr(self._tfidf_vectorizer, "get_feature_names_out"):
            return list(self._tfidf_vectorizer.get_feature_names_out())
        return list(self._tfidf_vectorizer.get_feature_names())
