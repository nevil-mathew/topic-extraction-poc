"""
TriTopic: Main Model Class
===========================

The core class that orchestrates all components of the topic modeling pipeline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from tritopic.core.embeddings import EmbeddingEngine
from tritopic.core.graph_builder import GraphBuilder
from tritopic.core.clustering import ConsensusLeiden
from tritopic.core.keywords import KeywordExtractor
from tritopic.core.hierarchy import TopicNode, TopicHierarchy
from tritopic.utils.metrics import compute_coherence, compute_diversity, compute_stability


@dataclass
class TopicInfo:
    """Container for topic information."""

    topic_id: int
    size: int
    keywords: list[str]
    keyword_scores: list[float]
    representative_docs: list[int]
    label: str | None = None
    description: str | None = None
    centroid: np.ndarray | None = None
    coherence: float | None = None


@dataclass
class ReportTheme:
    """A high-level meta-theme synthesized from one or more topics.

    Produced by :meth:`TriTopic.generate_report_themes` for inclusion in a
    qualitative research report.
    """

    theme_id: int                       # 1-indexed for human readability
    title: str                          # 5-8 word evocative title
    narrative: str                      # 4-6 sentence paragraph
    topic_ids: list[int]                # constituent topic IDs
    total_size: int                     # sum of constituent topic sizes
    keywords: list[str]                 # aggregated top keywords


@dataclass
class TriTopicConfig:
    """Configuration for TriTopic model."""
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_provider: Literal["local", "google"] = "local"
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_api_batch_size: int = 100
    embedding_output_dim: int | None = None
    embedding_task_type: str | None = None
    embedding_batch_delay: float = 0.0
    embedding_prefix: str | None = None
    language: str = "english"
    
    # Graph settings
    n_neighbors: int = 15
    metric: str = "cosine"
    graph_type: Literal["mutual_knn", "snn", "hybrid"] = "hybrid"
    snn_weight: float = 0.5
    
    # Multi-view settings
    use_lexical_view: bool = True
    use_metadata_view: bool = False
    lexical_weight: float = 0.3
    metadata_weight: float = 0.2
    semantic_weight: float = 0.5
    
    # Clustering settings
    resolution: float = 1.0
    resolution_range: tuple[float, float] | None = None
    n_consensus_runs: int = 10
    min_cluster_size: int = 5
    
    # Iterative refinement
    use_iterative_refinement: bool = True
    max_iterations: int = 5
    convergence_threshold: float = 0.95
    
    # Keyword extraction
    n_keywords: int = 10
    n_representative_docs: int = 5
    keyword_method: Literal["ctfidf", "bm25", "keybert"] = "ctfidf"
    
    # Dimensionality reduction
    use_dim_reduction: bool = True
    reduced_dims: int = 10
    dim_reduction_method: Literal["umap", "pacmap"] = "umap"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0  # 0.0 for clustering (not visualization)

    # Outlier handling
    outlier_threshold: float = 0.35

    # Soft assignment
    soft_assignment_method: Literal["centroid", "graph"] = "centroid"

    # Probability temperature (higher -> sharper distributions)
    softmax_temperature: float = 5.0

    # Misc
    random_state: int = 42
    verbose: bool = True

    # Memory optimization (opt-in).  When True, _compute_consensus builds the
    # condensed distance directly from the sparse co-occurrence (float32)
    # instead of materializing the N×N dense matrix, and _fit_iterative
    # caches the lexical graph across iterations.  Output is numerically
    # equivalent to the default path; enable for large-N OOM avoidance.
    low_memory: bool = False

    # Consensus strategy for ConsensusLeiden.
    # - "graph" (default): Lancichinetti-Fortunato graph consensus.  Threshold
    #   the sparse co-occurrence and run Leiden once on the result.  Avoids
    #   the N×N dense matrix and the scipy.linkage workspace entirely.
    # - "hierarchical": legacy average-linkage on the full co-occurrence
    #   distance.  Uses fastcluster when installed.  ~O(N²) memory; only
    #   appropriate for small N.
    consensus_method: Literal["graph", "hierarchical"] = "graph"

    # τ for graph-consensus: minimum fraction of Leiden runs (in [0, 1])
    # that must co-cluster a pair for it to become a consensus-graph edge.
    # Robust in [0.3, 0.8] per Lancichinetti & Fortunato (Sci. Rep. 2012).
    consensus_threshold_tau: float = 0.5

    # Parallelism: -1 uses all available cores (joblib / sklearn convention).
    n_jobs: int = -1


class TriTopic:
    """
    Tri-Modal Graph Topic Modeling with Iterative Refinement.
    
    A state-of-the-art topic modeling approach that combines semantic embeddings,
    lexical similarity, and optional metadata to create robust, interpretable topics.
    
    Key innovations:
    - Multi-view graph fusion (semantic + lexical + metadata)
    - Leiden clustering with consensus for stability
    - Iterative refinement loop for optimal topic separation
    - Advanced keyword extraction with representative documents
    - Optional LLM-powered topic labeling
    
    Parameters
    ----------
    config : TriTopicConfig, optional
        Configuration object. If None, uses defaults.
    embedding_model : str, optional
        Name of sentence-transformers model. Default: "all-MiniLM-L6-v2"
    n_neighbors : int, optional
        Number of neighbors for graph construction. Default: 15
    n_topics : int or "auto", optional
        Number of topics. "auto" uses Leiden's natural resolution. Default: "auto"
    use_iterative_refinement : bool, optional
        Whether to use the iterative refinement loop. Default: True
    verbose : bool, optional
        Print progress information. Default: True
    
    Attributes
    ----------
    topics_ : list[TopicInfo]
        Information about each discovered topic.
    labels_ : np.ndarray
        Topic assignment for each document.
    embeddings_ : np.ndarray
        Document embeddings.
    graph_ : igraph.Graph
        The constructed similarity graph.
    topic_embeddings_ : np.ndarray
        Centroid embeddings for each topic.
    
    Examples
    --------
    Basic usage:
    
    >>> from tritopic import TriTopic
    >>> model = TriTopic(n_neighbors=15, verbose=True)
    >>> topics = model.fit_transform(documents)
    >>> print(model.get_topic_info())
    
    With metadata:
    
    >>> model = TriTopic()
    >>> model.config.use_metadata_view = True
    >>> topics = model.fit_transform(documents, metadata=df[['source', 'date']])
    
    With LLM labeling:
    
    >>> from tritopic import TriTopic, LLMLabeler
    >>> model = TriTopic()
    >>> model.fit_transform(documents)
    >>> labeler = LLMLabeler(provider="anthropic", api_key="...")
    >>> model.generate_labels(labeler)
    """
    
    def __init__(
        self,
        config: TriTopicConfig | None = None,
        embedding_model: str | None = None,
        n_neighbors: int | None = None,
        n_topics: int | Literal["auto"] = "auto",
        use_iterative_refinement: bool | None = None,
        language: str | None = None,
        verbose: bool | None = None,
        random_state: int | None = None,
    ):
        # Initialize config
        self.config = config or TriTopicConfig()

        # Override config with explicit parameters
        if embedding_model is not None:
            self.config.embedding_model = embedding_model
        if n_neighbors is not None:
            self.config.n_neighbors = n_neighbors
        if use_iterative_refinement is not None:
            self.config.use_iterative_refinement = use_iterative_refinement
        if language is not None:
            self.config.language = language
        if verbose is not None:
            self.config.verbose = verbose
        if random_state is not None:
            self.config.random_state = random_state

        # Auto-select multilingual embedding model
        if self.config.language == "multilingual" and self.config.embedding_model == "all-MiniLM-L6-v2":
            self.config.embedding_model = "BAAI/bge-m3"

        self.n_topics = n_topics

        # Initialize components
        self._embedding_engine = EmbeddingEngine(
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
            provider=self.config.embedding_provider,
            api_key=self.config.embedding_api_key,
            api_batch_size=self.config.embedding_api_batch_size,
            output_dim=self.config.embedding_output_dim,
            task_type=self.config.embedding_task_type,
            batch_delay=self.config.embedding_batch_delay,
            prefix=self.config.embedding_prefix,
        )
        self._graph_builder = GraphBuilder(
            n_neighbors=self.config.n_neighbors,
            metric=self.config.metric,
            graph_type=self.config.graph_type,
            snn_weight=self.config.snn_weight,
            language=self.config.language,
            n_jobs=self.config.n_jobs,
        )
        self._clusterer = ConsensusLeiden(
            resolution=self.config.resolution,
            n_runs=self.config.n_consensus_runs,
            random_state=self.config.random_state,
            low_memory=self.config.low_memory,
            consensus_method=self.config.consensus_method,
            consensus_threshold_tau=self.config.consensus_threshold_tau,
            n_jobs=self.config.n_jobs,
        )
        self._keyword_extractor = KeywordExtractor(
            method=self.config.keyword_method,
            n_keywords=self.config.n_keywords,
            language=self.config.language,
        )
        
        # State
        self.topics_: list[TopicInfo] = []
        self.labels_: np.ndarray | None = None
        self.embeddings_: np.ndarray | None = None
        self.original_embeddings_: np.ndarray | None = None  # unrefined, for transform()
        self.reduced_embeddings_: np.ndarray | None = None
        self.probabilities_: np.ndarray | None = None
        self.lexical_matrix_: Any | None = None
        self.graph_: Any | None = None
        self.topic_embeddings_: np.ndarray | None = None
        self.documents_: list[str] | None = None
        self.hierarchy_: TopicHierarchy | None = None
        self.report_themes_: list[ReportTheme] | None = None
        self._is_fitted: bool = False
        self._iteration_history: list[dict] = []
        self._dim_reducer: Any | None = None
        
    def fit(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> "TriTopic":
        """
        Fit the topic model to documents.
        
        Parameters
        ----------
        documents : list[str]
            List of document texts.
        embeddings : np.ndarray, optional
            Pre-computed embeddings. If None, computed automatically.
        metadata : pd.DataFrame, optional
            Document metadata for the metadata view.
            
        Returns
        -------
        self : TriTopic
            Fitted model.
        """
        # Input validation
        if not documents:
            raise ValueError("documents must be a non-empty list of strings.")
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError(
                f"Embeddings length ({len(embeddings)}) must match "
                f"documents length ({len(documents)})."
            )
        if metadata is not None and len(metadata) != len(documents):
            raise ValueError(
                f"Metadata length ({len(metadata)}) must match "
                f"documents length ({len(documents)})."
            )

        self.documents_ = documents
        n_docs = len(documents)

        # Reset stateful components for clean re-fitting
        self._keyword_extractor.reset()
        self._iteration_history = []

        if self.config.verbose:
            print(f"[TriTopic] Fitting model on {n_docs} documents")
            print(f"   Config: {self.config.graph_type} graph, "
                  f"{'iterative' if self.config.use_iterative_refinement else 'single-pass'} mode"
                  f"{', low_memory=True' if self.config.low_memory else ''}")
        
        # Step 1: Generate embeddings
        if embeddings is not None:
            self.embeddings_ = embeddings
            if self.config.verbose:
                print("   + Using provided embeddings")
        else:
            if self.config.verbose:
                provider_tag = (
                    f"{self.config.embedding_provider}:{self.config.embedding_model}"
                    if self.config.embedding_provider != "local"
                    else self.config.embedding_model
                )
                print(f"   > Generating embeddings ({provider_tag})...")
            self.embeddings_ = self._embedding_engine.encode(documents)

        # Keep unrefined copy so transform() compares new docs in the same space
        self.original_embeddings_ = self.embeddings_.copy()

        # Step 1.5: Dimensionality reduction for graph building
        if self.config.use_dim_reduction:
            self._reduce_dimensions()

        # Step 2: Build lexical representation
        if self.config.use_lexical_view:
            if self.config.verbose:
                print("   > Building lexical similarity matrix...")
            self.lexical_matrix_ = self._graph_builder.build_lexical_matrix(documents)
        
        # Step 3: Build metadata graph (if provided)
        self._metadata_graph = None
        if self.config.use_metadata_view and metadata is not None:
            if self.config.verbose:
                print("   > Building metadata similarity graph...")
            self._metadata_graph = self._graph_builder.build_metadata_graph(metadata)

        # Step 4: Main fitting loop
        if self.config.use_iterative_refinement:
            self._fit_iterative(documents, self._metadata_graph)
        else:
            self._fit_single_pass(documents, self._metadata_graph)
        
        # Step 5: Extract keywords and representative docs
        if self.config.verbose:
            print("   > Extracting keywords and representative documents...")
        self._extract_topic_info(documents)
        
        # Step 6: Compute topic centroids
        self._compute_topic_centroids()

        # Step 7: Compute soft assignments (probabilities)
        self._compute_probabilities()

        self._is_fitted = True

        # Step 8: Apply n_topics target if specified
        if self.n_topics != "auto" and isinstance(self.n_topics, int):
            current_n_topics = len([t for t in self.topics_ if t.topic_id != -1])
            if self.n_topics != current_n_topics:
                # Use resolution search in both directions (fewer or more topics)
                self._auto_resolve_topic_count(
                    documents, self._metadata_graph, current_n_topics
                )

        if self.config.verbose:
            n_topics = len([t for t in self.topics_ if t.topic_id != -1])
            n_outliers = np.sum(self.labels_ == -1) if self.labels_ is not None else 0
            print(f"\n[OK] Fitting complete!")
            print(f"   Found {n_topics} topics")
            print(f"   {n_outliers} outlier documents ({100*n_outliers/n_docs:.1f}%)")

        return self
    
    def _fit_single_pass(
        self,
        documents: list[str],
        metadata_graph: Any | None = None,
    ) -> None:
        """Single-pass fitting without iterative refinement."""
        # Build graph
        if self.config.verbose:
            print("   > Building multi-view graph...")

        # Use reduced embeddings for graph building if available
        graph_embeddings = self.reduced_embeddings_ if self.reduced_embeddings_ is not None else self.embeddings_

        self.graph_ = self._graph_builder.build_multiview_graph(
            semantic_embeddings=graph_embeddings,
            lexical_matrix=self.lexical_matrix_ if self.config.use_lexical_view else None,
            metadata_graph=metadata_graph,
            weights={
                "semantic": self.config.semantic_weight,
                "lexical": self.config.lexical_weight,
                "metadata": self.config.metadata_weight,
            }
        )
        
        # Cluster
        if self.config.verbose:
            print(f"   > Running Leiden consensus clustering ({self.config.n_consensus_runs} runs)...")
            
        self.labels_ = self._clusterer.fit_predict(
            self.graph_,
            min_cluster_size=self.config.min_cluster_size,
        )
    
    def _fit_iterative(
        self,
        documents: list[str],
        metadata_graph: Any | None = None,
    ) -> None:
        """Iterative refinement fitting loop."""
        if self.config.verbose:
            print(f"   > Starting iterative refinement (max {self.config.max_iterations} iterations)...")

        current_embeddings = self.embeddings_.copy()
        # Use reduced embeddings for graph building if available
        if self.reduced_embeddings_ is not None:
            current_reduced = self.reduced_embeddings_.copy()
        else:
            current_reduced = None
        previous_labels = None

        # Precompute the lexical graph once — it is a pure function of the
        # immutable TF-IDF matrix and never changes between iterations.
        precomputed_lexical_adj = None
        if self.config.use_lexical_view and self.lexical_matrix_ is not None:
            precomputed_lexical_adj = self._graph_builder.build_lexical_graph(
                self.lexical_matrix_
            )

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"      Iteration {iteration + 1}...")

            # Build graph with reduced embeddings (or full if no reduction)
            graph_embeddings = current_reduced if current_reduced is not None else current_embeddings
            self.graph_ = self._graph_builder.build_multiview_graph(
                semantic_embeddings=graph_embeddings,
                lexical_matrix=self.lexical_matrix_ if self.config.use_lexical_view else None,
                metadata_graph=metadata_graph,
                weights={
                    "semantic": self.config.semantic_weight,
                    "lexical": self.config.lexical_weight,
                    "metadata": self.config.metadata_weight,
                },
                precomputed_lexical_adj=precomputed_lexical_adj,
            )

            # Cluster — skip stability during refinement; computed once after loop
            self.labels_ = self._clusterer.fit_predict(
                self.graph_,
                min_cluster_size=self.config.min_cluster_size,
                compute_stability=False,
            )

            n_topics_found = len(np.unique(self.labels_[self.labels_ != -1]))

            # Check convergence
            if previous_labels is not None:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(previous_labels, self.labels_)
                self._iteration_history.append({
                    "iteration": iteration + 1,
                    "ari": ari,
                    "n_topics": n_topics_found,
                })

                if self.config.verbose:
                    print(f"         ARI vs previous: {ari:.4f}")

                if ari >= self.config.convergence_threshold:
                    if self.config.verbose:
                        print(f"      Converged at iteration {iteration + 1}")
                    break
            else:
                # Record first iteration baseline
                self._iteration_history.append({
                    "iteration": 1,
                    "ari": None,
                    "n_topics": n_topics_found,
                })

            previous_labels = self.labels_.copy()

            # Refine embeddings with decaying blend factor (aggressive->fine)
            blend = 0.3 - 0.2 * (iteration / max(self.config.max_iterations - 1, 1))
            current_embeddings = self._refine_embeddings(
                current_embeddings, self.labels_, blend_factor=blend
            )

            # Re-reduce refined embeddings for next iteration's graph building
            if current_reduced is not None and self._dim_reducer is not None:
                current_reduced = self._dim_reducer.transform(current_embeddings)

        # Store final refined embeddings
        self.embeddings_ = current_embeddings
        # Update reduced embeddings to match final refined state
        if current_reduced is not None:
            self.reduced_embeddings_ = current_reduced

        # Compute stability once against the final iteration's partitions
        self._clusterer.stability_score_ = self._clusterer._compute_stability()

    def _auto_resolve_topic_count(
        self,
        documents: list[str],
        metadata_graph: Any | None,
        current_n_topics: int,
    ) -> None:
        """Binary-search for a resolution that yields the target n_topics.

        Works in both directions: lowers resolution when we have too many
        topics, raises it when we have too few.  Re-runs a single-pass fit
        at the best resolution found, then refreshes all downstream state.
        """
        target = self.n_topics
        if not isinstance(target, int) or target == current_n_topics:
            return

        if self.config.verbose:
            print(f"\n   > Auto-tuning resolution for {target} topics (currently {current_n_topics})...")

        graph_embeddings = (
            self.reduced_embeddings_
            if self.reduced_embeddings_ is not None
            else self.embeddings_
        )

        # Build graph once (reuse for all resolution probes)
        graph = self._graph_builder.build_multiview_graph(
            semantic_embeddings=graph_embeddings,
            lexical_matrix=self.lexical_matrix_ if self.config.use_lexical_view else None,
            metadata_graph=metadata_graph,
            weights={
                "semantic": self.config.semantic_weight,
                "lexical": self.config.lexical_weight,
                "metadata": self.config.metadata_weight,
            },
        )

        # Set search range based on direction
        if target > current_n_topics:
            # Need more topics → search higher resolutions
            res_range = (self.config.resolution, self.config.resolution * 10)
        else:
            # Need fewer topics → search lower resolutions
            res_range = (0.001, self.config.resolution)

        best_res = self._clusterer.find_optimal_resolution(
            graph,
            resolution_range=res_range,
            n_steps=20,
            target_n_topics=target,
        )

        if self.config.verbose:
            print(f"      Found resolution={best_res:.3f}")

        # Re-cluster at the found resolution
        self.graph_ = graph
        self.labels_ = self._clusterer.fit_predict(
            graph,
            min_cluster_size=self.config.min_cluster_size,
            resolution=best_res,
        )

        new_n = len(np.unique(self.labels_[self.labels_ != -1]))

        # If we overshot, merge down
        if new_n > target:
            # Temporarily mark as fitted so reduce_topics works
            was_fitted = self._is_fitted
            self._is_fitted = True
            self._keyword_extractor.reset()
            self._extract_topic_info(documents)
            self._compute_topic_centroids()
            self.reduce_topics(target)
            self._is_fitted = was_fitted
        else:
            self._keyword_extractor.reset()
            self._extract_topic_info(documents)
            self._compute_topic_centroids()
            self._compute_probabilities()

        final_n = len([t for t in self.topics_ if t.topic_id != -1])
        if self.config.verbose:
            print(f"      Final topic count: {final_n}")

    def _refine_embeddings(
        self,
        original_embeddings: np.ndarray,
        labels: np.ndarray,
        blend_factor: float = 0.2,
    ) -> np.ndarray:
        """
        Refine embeddings by incorporating topic context.

        Uses distance-aware blending: documents close to their topic
        centroid are pulled more strongly, while borderline documents
        are blended more conservatively to avoid misplacement.
        Outlier documents (label == -1) are left unchanged.

        Parameters
        ----------
        blend_factor : float
            Base blend strength (0 = no change, 1 = replace).
        """
        refined = original_embeddings.copy()
        unique_labels = np.unique(labels[labels != -1])

        # Compute topic centroids
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = original_embeddings[mask].mean(axis=0)

        for label in unique_labels:
            mask = labels == label
            centroid = centroids[label]
            topic_embs = refined[mask]

            # Distance-aware: compute cosine similarity to centroid
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
            emb_norms = topic_embs / (np.linalg.norm(topic_embs, axis=1, keepdims=True) + 1e-10)
            cos_sim = emb_norms @ centroid_norm  # shape (n_topic_docs,)

            # Scale blend: core members (high sim) get full blend,
            # borderline members (low sim) get reduced blend
            # Map cos_sim from [min_sim, 1] to [0.3, 1.0] multiplier
            per_doc_scale = np.clip(cos_sim, 0.0, 1.0) ** 0.5  # sqrt for softer scaling
            per_doc_blend = blend_factor * per_doc_scale[:, np.newaxis]

            refined[mask] = (1 - per_doc_blend) * topic_embs + per_doc_blend * centroid

        # Re-normalize (safe against zero-norm)
        norms = np.linalg.norm(refined, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        refined = refined / norms

        return refined
    
    def _reduce_dimensions(self) -> None:
        """Reduce embedding dimensionality for better graph construction."""
        if self.config.verbose:
            print(f"   > Reducing dimensions to {self.config.reduced_dims}d "
                  f"({self.config.dim_reduction_method})...")

        if self.config.dim_reduction_method == "umap":
            from umap import UMAP
            self._dim_reducer = UMAP(
                n_components=self.config.reduced_dims,
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                metric="cosine",
                random_state=self.config.random_state,
            )
        elif self.config.dim_reduction_method == "pacmap":
            from pacmap import PaCMAP
            self._dim_reducer = PaCMAP(
                n_components=self.config.reduced_dims,
                n_neighbors=self.config.umap_n_neighbors,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(f"Unknown dim_reduction_method: {self.config.dim_reduction_method}")

        self.reduced_embeddings_ = self._dim_reducer.fit_transform(self.embeddings_)

    def _compute_probabilities(self) -> None:
        """Compute soft topic assignment probabilities for training documents."""
        if self.config.soft_assignment_method == "graph":
            self._compute_graph_probabilities()
        else:
            self._compute_centroid_probabilities()

    def _compute_centroid_probabilities(self) -> None:
        """Centroid-based soft assignment: softmax over cosine similarity to topic centroids."""
        base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
        if self.topic_embeddings_ is None or base_emb is None:
            return

        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.special import softmax

        sim_matrix = cosine_similarity(base_emb, self.topic_embeddings_)
        # Temperature scaling: higher T -> sharper peaks
        self.probabilities_ = softmax(sim_matrix * self.config.softmax_temperature, axis=1)

    def _compute_graph_probabilities(self) -> None:
        """Graph-based soft assignment: topic distribution of each document's graph neighbours."""
        if self.graph_ is None or self.labels_ is None:
            self._compute_centroid_probabilities()
            return

        # Use same topic order as _compute_topic_centroids / _compute_centroid_probabilities
        non_outlier_topics = [t.topic_id for t in self.topics_ if t.topic_id != -1]
        n_topics = len(non_outlier_topics)
        topic_idx = {tid: i for i, tid in enumerate(non_outlier_topics)}
        n_docs = len(self.labels_)

        proba = np.zeros((n_docs, n_topics), dtype=np.float64)

        for doc_idx in range(n_docs):
            neighbors = self.graph_.neighbors(doc_idx)
            if not neighbors:
                # Fallback: uniform
                proba[doc_idx] = 1.0 / n_topics
                continue

            weights = []
            for nb in neighbors:
                eid = self.graph_.get_eid(doc_idx, nb)
                weights.append(self.graph_.es[eid]["weight"])

            for nb, w in zip(neighbors, weights):
                lab = self.labels_[nb]
                if lab != -1 and lab in topic_idx:
                    proba[doc_idx, topic_idx[lab]] += w

            row_sum = proba[doc_idx].sum()
            if row_sum > 0:
                proba[doc_idx] /= row_sum
            else:
                proba[doc_idx] = 1.0 / n_topics

        self.probabilities_ = proba

    def _extract_topic_info(self, documents: list[str]) -> None:
        """Extract keywords and representative documents for each topic."""
        self.topics_ = []
        unique_labels = np.unique(self.labels_)

        for label in unique_labels:
            mask = self.labels_ == label
            topic_indices = np.where(mask)[0]
            topic_docs = [documents[i] for i in topic_indices]
            
            # Extract keywords
            keywords, scores = self._keyword_extractor.extract(
                topic_docs, 
                all_docs=documents,
                n_keywords=self.config.n_keywords,
            )
            
            # Find representative documents (closest to centroid)
            if self.embeddings_ is not None and label != -1:
                topic_embeddings = self.embeddings_[mask]
                centroid = topic_embeddings.mean(axis=0)
                distances = np.linalg.norm(topic_embeddings - centroid, axis=1)
                top_indices = np.argsort(distances)[:self.config.n_representative_docs]
                representative_docs = [int(topic_indices[i]) for i in top_indices]
            else:
                representative_docs = list(topic_indices[:self.config.n_representative_docs])
            
            topic_info = TopicInfo(
                topic_id=int(label),
                size=int(mask.sum()),
                keywords=keywords,
                keyword_scores=scores,
                representative_docs=representative_docs,
                label=None,
                description=None,
            )
            self.topics_.append(topic_info)
        
        # Sort by size (excluding outliers)
        self.topics_ = sorted(
            self.topics_,
            key=lambda t: (t.topic_id == -1, -t.size)
        )
    
    def _compute_topic_centroids(self) -> None:
        """Compute centroid embeddings for each topic.

        Uses original (unrefined) embeddings so that ``transform()`` on new
        documents operates in the same embedding space as the centroids.
        """
        base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
        if base_emb is None:
            return

        unique_labels = [t.topic_id for t in self.topics_ if t.topic_id != -1]
        self.topic_embeddings_ = np.zeros((len(unique_labels), base_emb.shape[1]))

        topic_lookup = {t.topic_id: t for t in self.topics_}

        for i, label in enumerate(unique_labels):
            mask = self.labels_ == label
            self.topic_embeddings_[i] = base_emb[mask].mean(axis=0)
            if label in topic_lookup:
                topic_lookup[label].centroid = self.topic_embeddings_[i]
    
    def get_document_topics(
        self,
        doc_idx: int,
        top_n: int = 3,
        method: Literal["centroid", "graph"] | None = None,
    ) -> list[tuple[int, float]]:
        """Return top-N topics with probabilities for a single document.

        Parameters
        ----------
        doc_idx : int
            Index of the document.
        top_n : int
            Number of top topics to return.
        method : str, optional
            ``"centroid"`` or ``"graph"``.  If *None*, uses
            ``self.config.soft_assignment_method``.

        Returns
        -------
        topics : list[tuple[int, float]]
            List of ``(topic_id, probability)`` sorted descending.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        method = method or self.config.soft_assignment_method
        # Use same topic order as _compute_topic_centroids (matches topic_embeddings_ columns)
        non_outlier_topics = [t.topic_id for t in self.topics_ if t.topic_id != -1]

        if method == "graph" and self.graph_ is not None:
            topic_idx = {tid: i for i, tid in enumerate(non_outlier_topics)}
            proba = np.zeros(len(non_outlier_topics))
            neighbors = self.graph_.neighbors(doc_idx)
            for nb in neighbors:
                eid = self.graph_.get_eid(doc_idx, nb)
                w = self.graph_.es[eid]["weight"]
                lab = self.labels_[nb]
                if lab != -1 and lab in topic_idx:
                    proba[topic_idx[lab]] += w
            s = proba.sum()
            if s > 0:
                proba /= s
            else:
                proba[:] = 1.0 / len(non_outlier_topics)
        else:
            # Centroid-based
            from sklearn.metrics.pairwise import cosine_similarity
            from scipy.special import softmax

            base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
            sim = cosine_similarity(base_emb[doc_idx:doc_idx+1], self.topic_embeddings_)[0]
            proba = softmax(sim * self.config.softmax_temperature)

        ranked = np.argsort(proba)[::-1][:top_n]
        return [(non_outlier_topics[i], float(proba[i])) for i in ranked]

    def topic_overlap_matrix(self, threshold: float = 0.1) -> pd.DataFrame:
        """Compute a topic co-occurrence matrix from soft assignments.

        For each document, topics whose probability exceeds *threshold* are
        considered "active".  The matrix counts how often each pair of topics
        co-occurs across documents.

        Parameters
        ----------
        threshold : float
            Minimum probability for a topic to count as active.

        Returns
        -------
        overlap : pd.DataFrame
            Symmetric ``(n_topics, n_topics)`` DataFrame of co-occurrence counts.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.probabilities_ is None:
            self._compute_probabilities()

        # Use same topic order as _compute_topic_centroids (matches probabilities_ columns)
        non_outlier_topics = [t.topic_id for t in self.topics_ if t.topic_id != -1]
        n_topics = len(non_outlier_topics)
        overlap = np.zeros((n_topics, n_topics), dtype=np.int64)

        for row in self.probabilities_:
            active = np.where(row >= threshold)[0]
            for i in active:
                for j in active:
                    overlap[i, j] += 1

        labels = [f"Topic {tid}" for tid in non_outlier_topics]
        return pd.DataFrame(overlap, index=labels, columns=labels)

    def visualize_overlap(self, threshold: float = 0.1, **kwargs):
        """Visualize the topic overlap matrix as a heatmap.

        Parameters
        ----------
        threshold : float
            Minimum probability for a topic to count as active.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        from tritopic.visualization.plotter import plot_topic_overlap

        overlap = self.topic_overlap_matrix(threshold)
        topics = [t for t in self.topics_ if t.topic_id != -1]
        return plot_topic_overlap(overlap, topics, **kwargs)

    # ------------------------------------------------------------------
    # Hierarchical Topics
    # ------------------------------------------------------------------

    def build_hierarchy(
        self,
        resolution_levels: list[float] | None = None,
        n_levels: int = 3,
    ) -> TopicHierarchy:
        """Build a multi-resolution topic hierarchy.

        Re-uses the existing graph and clusters it at multiple resolution
        levels.  Coarse levels (low resolution) give broad themes; fine
        levels (high resolution) give specific sub-topics.  Levels are
        linked by majority-vote: each fine-grained node is assigned to
        the coarse-grained node that contains the majority of its
        documents.

        Parameters
        ----------
        resolution_levels : list[float], optional
            Explicit Leiden resolution values from coarse to fine.  If
            *None*, auto-generates *n_levels* values geometrically
            spaced between ``resolution / 4`` and ``resolution * 4``.
        n_levels : int
            Number of levels when *resolution_levels* is *None*.

        Returns
        -------
        hierarchy : TopicHierarchy
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        import leidenalg as la

        if resolution_levels is None:
            base = self.config.resolution
            resolution_levels = list(np.geomspace(base / 4, base * 4, n_levels))

        resolution_levels = sorted(resolution_levels)  # coarse → fine
        graph = self.graph_

        base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_

        all_level_nodes: list[list[TopicNode]] = []

        for level_idx, res in enumerate(resolution_levels):
            partition = la.find_partition(
                graph,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=res,
                seed=self.config.random_state,
            )
            level_labels = np.array(partition.membership)
            unique_ids = sorted(set(level_labels))

            level_nodes: list[TopicNode] = []
            for tid in unique_ids:
                doc_idx = np.where(level_labels == tid)[0]
                centroid = base_emb[doc_idx].mean(axis=0) if base_emb is not None else None

                # Extract keywords for this sub-cluster
                topic_docs = [self.documents_[i] for i in doc_idx]
                kw_extractor = KeywordExtractor(
                    method=self.config.keyword_method,
                    n_keywords=self.config.n_keywords,
                    language=self.config.language,
                )
                keywords, scores = kw_extractor.extract(
                    topic_docs, all_docs=self.documents_,
                    n_keywords=self.config.n_keywords,
                )

                node = TopicNode(
                    node_id=f"L{level_idx}_{tid}",
                    level=level_idx,
                    topic_id=tid,
                    size=len(doc_idx),
                    keywords=keywords,
                    keyword_scores=scores,
                    doc_indices=doc_idx,
                    centroid=centroid,
                )
                level_nodes.append(node)

            all_level_nodes.append(level_nodes)

        # Link levels via majority-vote
        for lvl in range(1, len(all_level_nodes)):
            parent_nodes = all_level_nodes[lvl - 1]
            child_nodes = all_level_nodes[lvl]

            # Build parent lookup: doc_idx → parent node
            parent_of_doc: dict[int, TopicNode] = {}
            for pnode in parent_nodes:
                for di in pnode.doc_indices:
                    parent_of_doc[di] = pnode

            for cnode in child_nodes:
                # Majority vote: which parent has the most overlap?
                votes: dict[str, int] = {}
                for di in cnode.doc_indices:
                    pn = parent_of_doc.get(di)
                    if pn is not None:
                        votes[pn.node_id] = votes.get(pn.node_id, 0) + 1

                if votes:
                    best_parent_id = max(votes, key=votes.get)
                    for pnode in parent_nodes:
                        if pnode.node_id == best_parent_id:
                            cnode.parent = pnode
                            pnode.children.append(cnode)
                            break

        hierarchy = TopicHierarchy(
            roots=all_level_nodes[0],
            levels=all_level_nodes,
            resolution_levels=resolution_levels,
        )
        self.hierarchy_ = hierarchy

        if self.config.verbose:
            sizes = [len(lvl) for lvl in all_level_nodes]
            print(f"[Hierarchy] Built {len(sizes)} levels: {sizes} topics")

        return hierarchy

    def divide(
        self,
        topic_id: int,
        n_subtopics: int = 2,
    ) -> list[TopicInfo]:
        """Split a single topic into *n_subtopics* sub-topics.

        Extracts the subgraph for the given topic and runs Leiden on it
        at a higher resolution to discover finer sub-communities.

        Parameters
        ----------
        topic_id : int
            Topic to divide.
        n_subtopics : int
            Target number of sub-topics.

        Returns
        -------
        subtopics : list[TopicInfo]
            New topic info objects for the sub-topics.  The labels in
            ``self.labels_`` are updated in-place; the original topic is
            replaced by the new sub-topics.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        import leidenalg as la

        mask = self.labels_ == topic_id
        if not np.any(mask):
            raise ValueError(f"Topic {topic_id} not found.")

        doc_indices = np.where(mask)[0]
        subgraph = self.graph_.subgraph(doc_indices.tolist())

        # Find resolution that yields ~n_subtopics
        sub_clusterer = ConsensusLeiden(
            resolution=self.config.resolution,
            n_runs=self.config.n_consensus_runs,
            random_state=self.config.random_state,
            low_memory=self.config.low_memory,
            consensus_method=self.config.consensus_method,
            consensus_threshold_tau=self.config.consensus_threshold_tau,
        )
        best_res = sub_clusterer.find_optimal_resolution(
            subgraph,
            resolution_range=(self.config.resolution, self.config.resolution * 20),
            n_steps=20,
            target_n_topics=n_subtopics,
        )

        sub_labels = sub_clusterer.fit_predict(
            subgraph, min_cluster_size=max(2, self.config.min_cluster_size // 2),
            resolution=best_res,
        )

        # Map sub-labels into global label space
        existing_max = int(self.labels_.max())
        new_topics: list[TopicInfo] = []

        for sub_id in sorted(set(sub_labels[sub_labels != -1])):
            new_label = existing_max + 1 + sub_id
            sub_mask = sub_labels == sub_id
            global_docs = doc_indices[sub_mask]
            self.labels_[global_docs] = new_label

            topic_docs = [self.documents_[i] for i in global_docs]
            kw_ext = KeywordExtractor(
                method=self.config.keyword_method,
                n_keywords=self.config.n_keywords,
                language=self.config.language,
            )
            keywords, scores = kw_ext.extract(
                topic_docs, all_docs=self.documents_,
                n_keywords=self.config.n_keywords,
            )

            base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
            centroid = base_emb[global_docs].mean(axis=0) if base_emb is not None else None

            info = TopicInfo(
                topic_id=new_label,
                size=int(sub_mask.sum()),
                keywords=keywords,
                keyword_scores=scores,
                representative_docs=list(global_docs[:self.config.n_representative_docs]),
                centroid=centroid,
            )
            new_topics.append(info)

        # Docs that became outliers in the subgraph keep original topic_id
        outlier_mask = sub_labels == -1
        if np.any(outlier_mask):
            self.labels_[doc_indices[outlier_mask]] = topic_id

        # Refresh topics list and centroids
        self._keyword_extractor.reset()
        self._extract_topic_info(self.documents_)
        self._compute_topic_centroids()
        self._compute_probabilities()

        if self.config.verbose:
            print(f"Divided topic {topic_id} into {len(new_topics)} sub-topics")

        return new_topics

    def visualize_hierarchy_tree(self, **kwargs):
        """Visualize the topic hierarchy as a tree diagram.

        Requires :meth:`build_hierarchy` to be called first.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        from tritopic.visualization.plotter import plot_hierarchy_tree

        if self.hierarchy_ is None:
            raise ValueError("No hierarchy built. Call build_hierarchy() first.")

        return plot_hierarchy_tree(self.hierarchy_, **kwargs)

    def fit_transform(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Fit the model and return topic assignments.

        Parameters
        ----------
        documents : list[str]
            List of document texts.
        embeddings : np.ndarray, optional
            Pre-computed embeddings.
        metadata : pd.DataFrame, optional
            Document metadata.
            
        Returns
        -------
        labels : np.ndarray
            Topic assignment for each document. -1 indicates outlier.
        """
        self.fit(documents, embeddings, metadata)
        return self.labels_
    
    def transform(self, documents: list[str]) -> np.ndarray:
        """
        Assign topics to new documents.

        Parameters
        ----------
        documents : list[str]
            New documents to classify.

        Returns
        -------
        labels : np.ndarray
            Topic assignments.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        from sklearn.metrics.pairwise import cosine_similarity

        new_embeddings = self._embedding_engine.encode(documents)

        non_outlier_topics = [t for t in self.topics_ if t.topic_id != -1]
        topic_ids = np.array([t.topic_id for t in non_outlier_topics])

        sim_matrix = cosine_similarity(new_embeddings, self.topic_embeddings_)
        nearest_idx = np.argmax(sim_matrix, axis=1)
        max_sim = sim_matrix[np.arange(len(documents)), nearest_idx]

        labels = topic_ids[nearest_idx]
        labels[max_sim < self.config.outlier_threshold] = -1

        return labels

    def transform_proba(self, documents: list[str]) -> np.ndarray:
        """
        Get soft topic assignment probabilities for new documents.

        Parameters
        ----------
        documents : list[str]
            New documents to classify.

        Returns
        -------
        probabilities : np.ndarray
            Shape (n_docs, n_topics) probability matrix. Rows sum to ~1.0.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.special import softmax

        new_embeddings = self._embedding_engine.encode(documents)
        sim_matrix = cosine_similarity(new_embeddings, self.topic_embeddings_)
        return softmax(sim_matrix * self.config.softmax_temperature, axis=1)

    def reduce_outliers(
        self,
        strategy: Literal["embeddings", "neighbors"] = "embeddings",
        threshold: float | None = None,
    ) -> "TriTopic":
        """
        Reassign outlier documents to the nearest topic.

        Parameters
        ----------
        strategy : str
            "embeddings" — assign each outlier to the most similar topic centroid
            (if similarity > threshold).
            "neighbors" — assign each outlier by majority vote of its k nearest
            non-outlier neighbors in embedding space.
        threshold : float, optional
            Minimum cosine similarity for assignment (embeddings strategy only).
            Defaults to ``self.config.outlier_threshold``.

        Returns
        -------
        self : TriTopic
            Updated model (labels_, topics_, topic_embeddings_, probabilities_).
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        outlier_mask = self.labels_ == -1
        if not np.any(outlier_mask):
            if self.config.verbose:
                print("No outliers to reduce.")
            return self

        outlier_indices = np.where(outlier_mask)[0]

        if self.config.verbose:
            print(f"Reducing {len(outlier_indices)} outliers (strategy={strategy})...")

        if strategy == "embeddings":
            from sklearn.metrics.pairwise import cosine_similarity

            thresh = threshold if threshold is not None else self.config.outlier_threshold
            non_outlier_topics = [t for t in self.topics_ if t.topic_id != -1]
            sim_matrix = cosine_similarity(
                self.embeddings_[outlier_indices], self.topic_embeddings_
            )

            for local_idx, global_idx in enumerate(outlier_indices):
                best_topic_idx = int(np.argmax(sim_matrix[local_idx]))
                best_sim = sim_matrix[local_idx, best_topic_idx]
                if best_sim >= thresh:
                    self.labels_[global_idx] = non_outlier_topics[best_topic_idx].topic_id

        elif strategy == "neighbors":
            from sklearn.neighbors import NearestNeighbors

            non_outlier_mask = ~outlier_mask
            non_outlier_indices = np.where(non_outlier_mask)[0]
            non_outlier_embeddings = self.embeddings_[non_outlier_mask]

            nn = NearestNeighbors(
                n_neighbors=min(self.config.n_neighbors, len(non_outlier_indices)),
                metric="cosine",
            )
            nn.fit(non_outlier_embeddings)
            _, neighbor_idx = nn.kneighbors(self.embeddings_[outlier_indices])

            for local_idx, global_idx in enumerate(outlier_indices):
                neighbor_global = non_outlier_indices[neighbor_idx[local_idx]]
                neighbor_labels = self.labels_[neighbor_global]
                # Majority vote
                values, counts = np.unique(neighbor_labels, return_counts=True)
                self.labels_[global_idx] = values[np.argmax(counts)]
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'embeddings' or 'neighbors'.")

        # Refresh downstream state
        self._keyword_extractor.reset()
        self._extract_topic_info(self.documents_)
        self._compute_topic_centroids()
        self._compute_probabilities()

        if self.config.verbose:
            remaining = int(np.sum(self.labels_ == -1))
            print(f"   Outliers remaining: {remaining}")

        return self

    def reduce_topics(self, n_topics: int) -> "TriTopic":
        """
        Iteratively merge the two most similar topics until *n_topics* remain.

        Parameters
        ----------
        n_topics : int
            Target number of non-outlier topics.

        Returns
        -------
        self : TriTopic
            Updated model.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        non_outlier_ids = [t.topic_id for t in self.topics_ if t.topic_id != -1]
        current_count = len(non_outlier_ids)

        if n_topics >= current_count:
            if self.config.verbose:
                print(f"Already at {current_count} topics (requested {n_topics}).")
            return self

        if self.config.verbose:
            print(f"Reducing from {current_count} to {n_topics} topics...")

        while current_count > n_topics:
            # Recompute centroids list aligned with current non-outlier ids
            non_outlier_ids = sorted(set(self.labels_[self.labels_ != -1]))
            sizes = np.array([
                int(np.sum(self.labels_ == tid)) for tid in non_outlier_ids
            ])
            centroids = np.array([
                self.embeddings_[self.labels_ == tid].mean(axis=0)
                for tid in non_outlier_ids
            ])
            sim = cos_sim(centroids)
            np.fill_diagonal(sim, -1)

            # Size-aware merge scoring: prefer merging smaller topics.
            # Scale similarity by a factor that penalises merging two
            # large topics (their combined size would dominate the corpus).
            n_docs = len(self.labels_)
            for pi in range(len(non_outlier_ids)):
                for pj in range(len(non_outlier_ids)):
                    if pi == pj:
                        continue
                    # Penalty = min_size / max_size (small-small -> 1, small-large -> small)
                    min_sz = min(sizes[pi], sizes[pj])
                    max_sz = max(sizes[pi], sizes[pj])
                    size_factor = (min_sz / max_sz) ** 0.3  # mild penalty
                    sim[pi, pj] *= size_factor

            # Find best pair to merge
            flat_idx = int(np.argmax(sim))
            i, j = divmod(flat_idx, len(non_outlier_ids))
            merge_from = non_outlier_ids[j]
            merge_into = non_outlier_ids[i]
            # Keep the larger topic's id
            if sizes[j] > sizes[i]:
                merge_into, merge_from = merge_from, merge_into
            # Relabel
            self.labels_[self.labels_ == merge_from] = merge_into
            current_count -= 1

        # Refresh downstream state
        self._keyword_extractor.reset()
        self._extract_topic_info(self.documents_)
        self._compute_topic_centroids()
        self._compute_probabilities()

        if self.config.verbose:
            final = len([t for t in self.topics_ if t.topic_id != -1])
            print(f"   Now have {final} topics.")

        return self

    def merge_topics(self, topics_to_merge: list[int]) -> "TriTopic":
        """
        Merge the specified topic IDs into one topic.

        The largest topic's ID is kept.

        Parameters
        ----------
        topics_to_merge : list[int]
            Topic IDs to merge together.

        Returns
        -------
        self : TriTopic
            Updated model.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if len(topics_to_merge) < 2:
            raise ValueError("Need at least 2 topic IDs to merge.")

        # Determine which topic to keep (largest)
        sizes = {tid: int(np.sum(self.labels_ == tid)) for tid in topics_to_merge}
        keep_id = max(sizes, key=sizes.get)

        for tid in topics_to_merge:
            if tid != keep_id:
                self.labels_[self.labels_ == tid] = keep_id

        # Refresh downstream state
        self._keyword_extractor.reset()
        self._extract_topic_info(self.documents_)
        self._compute_topic_centroids()
        self._compute_probabilities()

        if self.config.verbose:
            print(f"Merged topics {topics_to_merge} -> {keep_id}")

        return self

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get a DataFrame with topic information.
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns: Topic, Size, Keywords, Label, Coherence
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        data = []
        for topic in self.topics_:
            data.append({
                "Topic": topic.topic_id,
                "Size": topic.size,
                "Keywords": ", ".join(topic.keywords[:5]),
                "All_Keywords": topic.keywords,
                "Keyword_Scores": topic.keyword_scores,
                "Label": topic.label or f"Topic {topic.topic_id}",
                "Description": topic.description,
                "Representative_Docs": topic.representative_docs,
                "Coherence": topic.coherence,
            })
        
        return pd.DataFrame(data)
    
    def get_topic(self, topic_id: int) -> TopicInfo | None:
        """Get information about a specific topic."""
        for topic in self.topics_:
            if topic.topic_id == topic_id:
                return topic
        return None
    
    def get_representative_docs(
        self,
        topic_id: int,
        n_docs: int = 5,
    ) -> list[tuple[int, str]]:
        """
        Get representative documents for a topic.
        
        Parameters
        ----------
        topic_id : int
            Topic ID.
        n_docs : int
            Number of documents to return.
            
        Returns
        -------
        docs : list[tuple[int, str]]
            List of (index, document_text) tuples.
        """
        if not self._is_fitted or self.documents_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topic = self.get_topic(topic_id)
        if topic is None:
            raise ValueError(f"Topic {topic_id} not found.")
        
        indices = topic.representative_docs[:n_docs]
        return [(idx, self.documents_[idx]) for idx in indices]
    
    def generate_labels(
        self,
        labeler: "LLMLabeler",
        topics: list[int] | None = None,
        dedup: bool = True,
        dedup_passes: int = 1,
    ) -> None:
        """
        Generate labels for topics using an LLM.

        Each topic is labeled with awareness of all topics labeled before it
        in the same run, so the LLM avoids producing duplicate or
        near-duplicate labels. After the main pass, an optional cleanup pass
        re-labels any remaining exact-duplicate or shared-prefix collisions
        with explicit "make these distinct" instructions.

        Parameters
        ----------
        labeler : LLMLabeler
            Configured LLM labeler. Use ``style="theme"`` for report-quality
            narrative labels.
        topics : list[int], optional
            Specific topics to label. If None, labels all non-outlier topics.
        dedup : bool
            Pass already-assigned labels to each LLM call. Default True.
        dedup_passes : int
            Number of cleanup passes after the main loop. Default 1.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        target_topics = topics or [t.topic_id for t in self.topics_ if t.topic_id != -1]
        n_total = len(target_topics)
        existing: list[dict] = []

        for i, topic_id in enumerate(tqdm(target_topics, desc="Generating labels", disable=not self.config.verbose), 1):
            topic = self.get_topic(topic_id)
            if topic is None:
                continue

            rep_docs = self.get_representative_docs(topic_id, n_docs=labeler.n_docs)
            doc_texts = [doc for _, doc in rep_docs]

            label, description = labeler.generate_label(
                keywords=topic.keywords,
                representative_docs=doc_texts,
                existing_labels=existing if dedup else None,
            )

            topic.label = label
            topic.description = description
            if dedup:
                existing.append({"label": label, "keywords": topic.keywords})

            if labeler.verbose:
                kw_hint = ", ".join(topic.keywords[:3])
                print(f"[{i}/{n_total}] Topic {topic_id} ({kw_hint}) → {label}")

        # Cleanup pass: regenerate exact-duplicate / shared-prefix labels
        for _ in range(dedup_passes):
            collisions = self._find_label_collisions(target_topics)
            if not collisions:
                break
            self._resolve_label_collisions(labeler, collisions)

    def _find_label_collisions(self, target_topic_ids: list[int]) -> list[list[int]]:
        """Group topics that share an exact or near-duplicate label.

        Two labels collide if they are identical OR share the same first three
        words (case-insensitive). Returns a list of groups, each containing
        2+ topic IDs.
        """
        by_label: dict[str, list[int]] = {}
        by_prefix: dict[str, list[int]] = {}
        for tid in target_topic_ids:
            topic = self.get_topic(tid)
            if topic is None or not topic.label:
                continue
            label = topic.label.strip()
            by_label.setdefault(label.lower(), []).append(tid)
            prefix = " ".join(label.lower().split()[:3])
            by_prefix.setdefault(prefix, []).append(tid)

        # Use prefix groups (they subsume exact duplicates) of size >= 2
        return [ids for ids in by_prefix.values() if len(ids) >= 2]

    def _resolve_label_collisions(
        self, labeler: "LLMLabeler", collisions: list[list[int]]
    ) -> None:
        """Re-label each colliding group with explicit "make distinct" context."""
        for group in collisions:
            # Build a context that tells each member about its siblings
            sibling_summaries: list[dict] = []
            for tid in group:
                t = self.get_topic(tid)
                if t is None:
                    continue
                sibling_summaries.append({"label": t.label or "", "keywords": t.keywords})

            for tid in group:
                topic = self.get_topic(tid)
                if topic is None:
                    continue
                # "existing" = the sibling labels (excluding self) plus everyone else
                others = [s for s in sibling_summaries if s["keywords"] is not topic.keywords]
                rep_docs = self.get_representative_docs(tid, n_docs=labeler.n_docs)
                doc_texts = [doc for _, doc in rep_docs]
                label, description = labeler.generate_label(
                    keywords=topic.keywords,
                    representative_docs=doc_texts,
                    existing_labels=others,
                )
                topic.label = label
                topic.description = description
                if labeler.verbose:
                    print(f"  [dedup] Topic {tid} → {label}")

    # ------------------------------------------------------------------
    # Report meta-themes
    # ------------------------------------------------------------------

    def generate_report_themes(
        self,
        labeler: "LLMLabeler",
        n_themes: int | None = None,
        n_docs_per_theme: int = 12,
    ) -> list[ReportTheme]:
        """Synthesize 5-12 high-level meta-themes from all per-topic labels.

        Designed to produce a small set of report-ready narrative themes (in
        the style of qualitative research reports). Recommended workflow:

            1. Call :meth:`generate_labels` first with ``labeler.style="theme"``
               so each topic has an evocative title + narrative description.
            2. Call this method to consolidate them into a handful of
               cross-cutting meta-themes for the final report.
            3. Call :meth:`export_report` to write the report to a file.

        The method uses two LLM stages:

            * **Proposer**: sees the title, narrative excerpt, and top
              keywords of every non-outlier topic and proposes N meta-themes
              with explicit topic assignments.
            * **Narrative writer** (one call per meta-theme): given the
              meta-theme title and constituent topics, writes the final
              report-style narrative quoting underlying documents.

        Parameters
        ----------
        labeler : LLMLabeler
            Labeler used to call the LLM. ``style`` is irrelevant here — the
            method overrides the prompts for both stages.
        n_themes : int, optional
            Target number of meta-themes. If None (default), the LLM is
            instructed to choose the natural number of meta-themes for the
            data, typically 5-12. When set, the LLM is asked to hit that
            number ±2.
        n_docs_per_theme : int
            Number of representative documents (sampled proportionally
            across constituent topics) passed to the narrative writer.

        Returns
        -------
        list[ReportTheme]
            Stored on ``self.report_themes_``.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        topics = [t for t in self.topics_ if t.topic_id != -1]
        if not topics:
            raise ValueError("No topics available to synthesize from.")
        if any(t.label is None for t in topics):
            raise ValueError(
                "All topics must have labels. Call generate_labels(labeler) "
                "before generate_report_themes()."
            )

        if self.config.verbose:
            target = f"~{n_themes}" if n_themes is not None else "an LLM-chosen count of"
            print(f"   > Proposing {target} meta-themes from {len(topics)} topics...")
        proposals = self._propose_meta_themes(labeler, topics, n_themes)

        if self.config.verbose:
            print(f"   > Writing narratives for {len(proposals)} meta-themes...")
        themes: list[ReportTheme] = []
        for idx, prop in enumerate(tqdm(proposals, desc="Theme narratives", disable=not self.config.verbose), 1):
            theme = self._write_meta_theme_narrative(
                labeler, prop, theme_id=idx, n_docs=n_docs_per_theme
            )
            if theme is not None:
                themes.append(theme)

        self.report_themes_ = themes
        return themes

    def _propose_meta_themes(
        self, labeler: "LLMLabeler", topics: list[TopicInfo], n_themes: int | None
    ) -> list[dict]:
        """Ask the LLM to group all topics into meta-themes.

        If ``n_themes`` is None the LLM chooses the natural number; otherwise
        it is given a target ±2. Returns ``[{"title": str, "topic_ids": list[int]}, ...]``.
        """
        import re

        # Compact per-topic summary block
        lines = []
        for t in topics:
            kws = ", ".join(t.keywords[:6])
            desc = (t.description or "").strip().replace("\n", " ")
            # First sentence of the narrative as the essence
            first_sentence = re.split(r"(?<=[.!?])\s+", desc, maxsplit=1)[0]
            first_sentence = first_sentence[:240]
            lines.append(
                f"  Topic {t.topic_id} (n={t.size}) — {t.label}\n"
                f"    keywords: {kws}\n"
                f"    essence:  {first_sentence}"
            )
        topics_block = "\n".join(lines)

        hint = labeler.domain_hint or ""
        domain_line = f"\nThis is a qualitative study about {hint}.\n" if hint else ""

        system_prompt = (
            "You are a senior qualitative research analyst. You consolidate fine-grained "
            "topic-model output into a small number of EMERGING THEMES suitable for the "
            "Findings section of a research report. You always respond with valid JSON "
            "and nothing else."
        )

        if n_themes is None:
            count_instruction = (
                "Choose the NATURAL number of meta-themes for this data, typically "
                "between 5 and 12. Use exactly as many as the data warrants — do not "
                "pad to reach a round number, and do not force consolidation if the "
                "data clearly spans more distinct patterns. Quality of grouping "
                "matters more than count."
            )
            intro = "Group them into the natural set of EMERGING META-THEMES for a research report."
        else:
            count_instruction = (
                f"Produce approximately {n_themes} meta-themes (±2 acceptable if the "
                f"data clearly demands it). Do not stretch or compress beyond that range."
            )
            intro = f"Group them into approximately {n_themes} EMERGING META-THEMES for a research report."

        user_prompt = f"""Below is the full list of topics from a topic model. {intro}
{domain_line}
TOPICS:
{topics_block}

RULES:
- {count_instruction}
- Every topic ID must be assigned to exactly one meta-theme. Do not drop any.
- A meta-theme should unify topics that share a deeper human concern, mechanism, or pattern — not just surface keyword overlap.
- Each meta-theme title must be 5-8 words, title case, EVOCATIVE (name the lived experience, not a generic category).
- Never use the bare phrases "Systemic Barriers", "Structural Barriers", "Geographic Barriers", "Educational Access" — they are too generic. Name the specific mechanism.
- Order the meta-themes by overall importance (largest constituent size first, ties broken by narrative centrality).

OUTPUT FORMAT (JSON, no other text):
{{
  "themes": [
    {{"title": "Evocative Meta-Theme Title", "topic_ids": [0, 3, 6, 10]}},
    {{"title": "Another Meta-Theme",         "topic_ids": [1, 2, 8]}}
  ]
}}"""

        # Allow extra headroom — the response can be long for many topics
        raw = labeler.call_raw(system_prompt, user_prompt, max_tokens=4000)

        # Parse JSON robustly
        proposals = self._parse_proposer_response(raw, valid_topic_ids={t.topic_id for t in topics})
        if not proposals:
            # Fallback: one meta-theme per topic (degenerate but safe)
            warnings.warn(
                "Meta-theme proposer returned no valid groupings. Falling back to "
                "one-to-one topics → meta-themes."
            )
            proposals = [{"title": t.label or f"Topic {t.topic_id}", "topic_ids": [t.topic_id]} for t in topics]
        return proposals

    def _parse_proposer_response(self, raw: str, valid_topic_ids: set[int]) -> list[dict]:
        """Extract themes from the proposer JSON, dropping invalid IDs."""
        import json, re

        proposals: list[dict] = []
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end <= start:
                return []
            data = json.loads(raw[start:end])
            for theme in data.get("themes", []):
                title = str(theme.get("title", "")).strip()
                ids_raw = theme.get("topic_ids", []) or []
                ids = [int(i) for i in ids_raw if int(i) in valid_topic_ids]
                if title and ids:
                    proposals.append({"title": title, "topic_ids": ids})
        except (json.JSONDecodeError, ValueError, TypeError):
            # Best-effort regex fallback for slightly malformed JSON
            blocks = re.findall(r'"title"\s*:\s*"([^"]+)"\s*,\s*"topic_ids"\s*:\s*\[([0-9,\s]+)\]', raw)
            for title, ids_str in blocks:
                ids = [int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()]
                ids = [i for i in ids if i in valid_topic_ids]
                if title.strip() and ids:
                    proposals.append({"title": title.strip(), "topic_ids": ids})

        # Ensure every topic is assigned (assign orphans to the largest meta-theme)
        assigned = {tid for p in proposals for tid in p["topic_ids"]}
        orphans = valid_topic_ids - assigned
        if orphans and proposals:
            largest = max(proposals, key=lambda p: len(p["topic_ids"]))
            largest["topic_ids"].extend(sorted(orphans))
        return proposals

    def _write_meta_theme_narrative(
        self,
        labeler: "LLMLabeler",
        proposal: dict,
        theme_id: int,
        n_docs: int,
    ) -> ReportTheme | None:
        """Generate the report-style narrative paragraph for one meta-theme."""
        topic_ids: list[int] = proposal["topic_ids"]
        member_topics = [self.get_topic(tid) for tid in topic_ids]
        member_topics = [t for t in member_topics if t is not None]
        if not member_topics:
            return None

        total_size = sum(t.size for t in member_topics)

        # Sample representative docs proportionally to topic size
        docs_per_topic = max(1, n_docs // max(len(member_topics), 1))
        doc_texts: list[str] = []
        for t in member_topics:
            rep = self.get_representative_docs(t.topic_id, n_docs=docs_per_topic)
            doc_texts.extend(doc for _, doc in rep)
        doc_texts = doc_texts[:n_docs]

        # Aggregate keywords (preserve order, dedupe)
        seen: set[str] = set()
        agg_keywords: list[str] = []
        for t in member_topics:
            for kw in t.keywords[:8]:
                if kw not in seen:
                    seen.add(kw)
                    agg_keywords.append(kw)
                if len(agg_keywords) >= 20:
                    break
            if len(agg_keywords) >= 20:
                break

        # Member topic summaries for context
        summaries = []
        for t in member_topics:
            summaries.append(
                f"  - Topic {t.topic_id} (n={t.size}, '{t.label}'): "
                f"{(t.description or '').strip()[:300]}"
            )
        members_block = "\n".join(summaries)

        # Docs block
        docs_text = ""
        for i, doc in enumerate(doc_texts, 1):
            truncated = doc[:1200] + "..." if len(doc) > 1200 else doc
            docs_text += f"\nDocument {i}: {truncated}\n"

        hint = labeler.domain_hint or ""
        domain_line = f"This is a qualitative study about {hint}." if hint else ""

        system_prompt = (
            "You are a qualitative research analyst writing the Findings section of a "
            "report. You write evocative, concrete, narrative themes that quote "
            "participants and synthesize what the data really means. You always "
            "respond with valid JSON and nothing else."
        )

        user_prompt = f"""You are writing one EMERGING THEME for a qualitative research report. The theme's title has already been chosen. Your job is to write the narrative paragraph.

{domain_line}

THEME TITLE (already chosen — do not change):
{proposal['title']}

This meta-theme synthesizes {len(member_topics)} fine-grained topics covering {total_size} documents:
{members_block}

AGGREGATED TOP KEYWORDS:
{', '.join(agg_keywords)}

SAMPLE DOCUMENTS:
{docs_text}

YOUR "narrative" MUST:
- Be 4-6 sentences (60-120 words).
- Open with a concrete observation, not an abstract claim.
- Include at least one short quoted phrase from the documents (use single quotes around the quoted phrase, e.g., parents called the school 'too far', families said admission was 'denied without Aadhar').
- Synthesize the shared pattern across the constituent topics — not list them.
- End with a sentence that names the underlying meaning or why this matters.
- Be written in plain English, third person.
- Do NOT repeat the theme title in the narrative.

REFERENCE STYLE (do not copy content, only match register):

  Title: "A Festive and Welcoming School Environment"
  Narrative: "The PTM was often described as a 'celebration' or a 'festival.' Schools were decorated with student artwork, crafts, and photo booths, creating a vibrant atmosphere. Students actively participated in welcoming guests, singing songs, and organizing events — making the PTM feel like a community-led initiative. This joyful environment made parents feel proud and included."

Respond ONLY with this exact JSON, no other text:
{{"narrative": "Your 4-6 sentence paragraph here."}}"""

        raw = labeler.call_raw(system_prompt, user_prompt, max_tokens=1200)
        narrative = self._parse_narrative_response(raw)

        return ReportTheme(
            theme_id=theme_id,
            title=proposal["title"].strip(),
            narrative=narrative,
            topic_ids=[t.topic_id for t in member_topics],
            total_size=total_size,
            keywords=agg_keywords,
        )

    def _parse_narrative_response(self, raw: str) -> str:
        """Extract the narrative field from the writer's JSON output."""
        import json, re

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                return str(data.get("narrative", "")).strip()
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        m = re.search(r'"narrative"\s*:\s*"([^"]*)"', raw)
        if m:
            return m.group(1).strip()
        # Last-resort: take the longest line
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return max(lines, key=len) if lines else ""

    def regenerate_theme(
        self,
        labeler: "LLMLabeler",
        theme_id: int,
        new_topic_ids: list[int] | None = None,
        n_docs: int = 12,
    ) -> ReportTheme:
        """Regenerate one meta-theme's narrative, optionally moving topics in/out.

        Use after :meth:`generate_report_themes` if a particular theme needs
        editing — either to swap its constituent topics or simply to resample
        a new narrative.
        """
        if self.report_themes_ is None:
            raise ValueError("Call generate_report_themes(labeler) first.")
        target = next((t for t in self.report_themes_ if t.theme_id == theme_id), None)
        if target is None:
            raise ValueError(f"No report theme with id={theme_id}.")
        topic_ids = new_topic_ids if new_topic_ids is not None else target.topic_ids
        proposal = {"title": target.title, "topic_ids": topic_ids}
        new = self._write_meta_theme_narrative(labeler, proposal, theme_id=theme_id, n_docs=n_docs)
        if new is None:
            raise ValueError("Regeneration failed.")
        idx = self.report_themes_.index(target)
        self.report_themes_[idx] = new
        return new

    def export_report(self, path: str, include_appendix: bool = True) -> None:
        """Write the meta-themes to a Markdown report file.

        Format matches the qualitative-research style: a header section,
        then each meta-theme as a numbered H2 with its narrative paragraph.
        With ``include_appendix=True``, an appendix lists the constituent
        topic IDs and sizes under each theme for traceability.
        """
        if self.report_themes_ is None:
            raise ValueError(
                "No report themes available. Call generate_report_themes(labeler) first."
            )

        lines: list[str] = []
        lines.append("# Emerging Themes\n")
        lines.append(
            "The stories collected through this study surfaced recurring patterns. "
            "These themes synthesize the most frequent and meaningful signals across "
            "all collected documents.\n"
        )

        for theme in self.report_themes_:
            lines.append(f"## {theme.theme_id}. {theme.title}\n")
            lines.append(theme.narrative.strip() + "\n")

        if include_appendix:
            lines.append("\n---\n")
            lines.append("## Appendix: Constituent Topics\n")
            lines.append(
                "Each meta-theme is synthesized from the following fine-grained "
                "topics produced by the topic model.\n"
            )
            for theme in self.report_themes_:
                lines.append(
                    f"\n**{theme.theme_id}. {theme.title}** "
                    f"({len(theme.topic_ids)} topics, {theme.total_size} documents)\n"
                )
                for tid in theme.topic_ids:
                    t = self.get_topic(tid)
                    if t is None:
                        continue
                    label = t.label or f"Topic {tid}"
                    lines.append(f"- Topic {tid} (n={t.size}): {label}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if self.config.verbose:
            print(f"   > Report written to {path}")

    def visualize(
        self,
        method: Literal["umap", "pacmap"] = "umap",
        color_by: Literal["topic", "custom"] = "topic",
        custom_labels: list[str] | None = None,
        show_outliers: bool = True,
        interactive: bool = True,
        **kwargs,
    ):
        """
        Visualize topics in 2D.
        
        Parameters
        ----------
        method : str
            Dimensionality reduction method. "umap" or "pacmap".
        color_by : str
            How to color points. "topic" uses topic assignments.
        custom_labels : list[str], optional
            Custom labels for hover text.
        show_outliers : bool
            Whether to show outlier documents.
        interactive : bool
            If True, returns interactive Plotly figure.
        **kwargs
            Additional arguments passed to the visualizer.
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive visualization.
        """
        from tritopic.visualization.plotter import TopicVisualizer
        
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        visualizer = TopicVisualizer(method=method)
        
        return visualizer.plot_documents(
            embeddings=self.embeddings_,
            labels=self.labels_,
            documents=self.documents_,
            topics=self.topics_,
            show_outliers=show_outliers,
            interactive=interactive,
            **kwargs,
        )
    
    def visualize_3d(
        self,
        method: Literal["umap", "pacmap"] = "umap",
        show_outliers: bool = True,
        **kwargs,
    ):
        """
        Visualize topics in 3D.

        Parameters
        ----------
        method : str
            Dimensionality reduction method. "umap" or "pacmap".
        show_outliers : bool
            Whether to show outlier documents.
        **kwargs
            Additional arguments passed to :meth:`TopicVisualizer.plot_documents_3d`.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive 3-D visualization.
        """
        from tritopic.visualization.plotter import TopicVisualizer

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        visualizer = TopicVisualizer(method=method)

        return visualizer.plot_documents_3d(
            embeddings=self.embeddings_,
            labels=self.labels_,
            documents=self.documents_,
            topics=self.topics_,
            show_outliers=show_outliers,
            **kwargs,
        )

    def visualize_hierarchy(self, **kwargs):
        """Visualize topic hierarchy as a dendrogram."""
        from tritopic.visualization.plotter import TopicVisualizer
        
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        visualizer = TopicVisualizer()
        return visualizer.plot_hierarchy(
            topic_embeddings=self.topic_embeddings_,
            topics=self.topics_,
            **kwargs,
        )
    
    def visualize_topics(self, **kwargs):
        """Visualize topics as a heatmap or bar chart."""
        from tritopic.visualization.plotter import TopicVisualizer

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        visualizer = TopicVisualizer()
        return visualizer.plot_topics(
            topics=self.topics_,
            **kwargs,
        )

    def visualize_topic_map(
        self,
        method: Literal["mds", "pca", "umap"] = "mds",
        **kwargs,
    ):
        """Intertopic distance map: 2-D projection of topic centroids.

        Bubbles are sized by topic count and positioned by centroid distance, so
        the overall topic landscape is visible at a glance — much faster than
        :meth:`visualize`, which re-projects every document.

        Parameters
        ----------
        method : {"mds", "pca", "umap"}
            Projection used on the centroid matrix. ``"mds"`` (default) operates
            on cosine distances and matches the pyLDAvis convention.
        **kwargs
            Forwarded to :func:`tritopic.visualization.plotter.plot_intertopic_distance_map`
            (``n_keywords``, ``size_scale``, ``title``, ``width``, ``height``,
            ``random_state``).
        """
        from tritopic.visualization.plotter import plot_intertopic_distance_map

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.topic_embeddings_ is None:
            raise ValueError("Topic centroids are not available.")

        return plot_intertopic_distance_map(
            topic_embeddings=self.topic_embeddings_,
            topics=self.topics_,
            method=method,
            **kwargs,
        )
    
    def export_projector(
        self,
        output_dir: str = ".",
        embeddings: Literal["original", "reduced", "2d", "3d"] = "original",
    ) -> tuple[str, str]:
        """
        Export document embeddings and metadata for the TensorFlow Embedding Projector
        (https://projector.tensorflow.org).

        Writes two TSV files to ``output_dir``:

        * ``vectors.tsv``  – one document per row, tab-separated floats.
        * ``metadata.tsv`` – topic_id, topic_label, keywords, document snippet.

        Parameters
        ----------
        output_dir : str
            Directory to write the TSV files into. Created if it does not exist.
        embeddings : {"original", "reduced", "2d", "3d"}
            Which vectors to export.

            * ``"original"`` – raw high-dimensional embeddings (recommended; lets
              the projector apply PCA / UMAP / t-SNE interactively).
            * ``"reduced"``  – clustering-space reduced embeddings
              (``reduced_embeddings_``).
            * ``"2d"`` / ``"3d"`` – a fresh UMAP projection to 2-D or 3-D
              (same settings as :meth:`visualize`).

        Returns
        -------
        vectors_path, metadata_path : tuple[str, str]
            Absolute paths of the two written files.
        """
        import os

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        os.makedirs(output_dir, exist_ok=True)

        # --- choose vectors ---
        if embeddings == "original":
            vecs = self.embeddings_
        elif embeddings == "reduced":
            if self.reduced_embeddings_ is None:
                raise ValueError("reduced_embeddings_ not available.")
            vecs = self.reduced_embeddings_
        elif embeddings in ("2d", "3d"):
            n_components = 2 if embeddings == "2d" else 3
            from tritopic.visualization.plotter import TopicVisualizer
            viz = TopicVisualizer(method=self.config.dim_reduction_method)
            vecs = viz._reduce_dimensions(self.embeddings_, n_components=n_components)
        else:
            raise ValueError(f"Unknown embeddings value: {embeddings!r}")

        # --- vectors.tsv ---
        vectors_path = os.path.abspath(os.path.join(output_dir, "vectors.tsv"))
        np.savetxt(vectors_path, vecs, delimiter="\t")

        # --- metadata.tsv ---
        topic_map = {t.topic_id: t for t in self.topics_}
        metadata_path = os.path.abspath(os.path.join(output_dir, "metadata.tsv"))
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("topic_id\ttopic_label\tkeywords\tdocument\n")
            for i, doc in enumerate(self.documents_):
                tid = int(self.labels_[i])
                topic = topic_map.get(tid)
                label = topic.label if topic and topic.label else ("Outlier" if tid == -1 else str(tid))
                keywords = ", ".join(topic.keywords[:5]) if topic else ""
                snippet = doc[:150].replace("\t", " ").replace("\n", " ")
                f.write(f"{tid}\t{label}\t{keywords}\t{snippet}\n")

        if self.config.verbose:
            print(f"\n[Projector] Exported {len(self.documents_)} documents")
            print(f"   vectors  → {vectors_path}")
            print(f"   metadata → {metadata_path}")
            print("   Load both files at https://projector.tensorflow.org")

        return vectors_path, metadata_path

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate topic model quality.
        
        Returns
        -------
        metrics : dict
            Dictionary with coherence, diversity, and stability scores.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute coherence for each topic
        coherences = []
        for topic in self.topics_:
            if topic.topic_id != -1:
                coh = compute_coherence(
                    topic.keywords,
                    [self.documents_[i] for i in np.where(self.labels_ == topic.topic_id)[0]]
                )
                topic.coherence = coh
                coherences.append(coh)
        
        # Compute diversity
        all_keywords = [kw for t in self.topics_ if t.topic_id != -1 for kw in t.keywords]
        diversity = compute_diversity(all_keywords, n_topics=len(coherences))
        
        # Get stability from consensus clustering
        stability = self._clusterer.stability_score_ if hasattr(self._clusterer, 'stability_score_') else None
        
        metrics = {
            "coherence_mean": float(np.mean(coherences)) if coherences else 0.0,
            "coherence_std": float(np.std(coherences)) if coherences else 0.0,
            "diversity": diversity,
            "stability": stability,
            "n_topics": len([t for t in self.topics_ if t.topic_id != -1]),
            "outlier_ratio": float(np.mean(self.labels_ == -1)) if self.labels_ is not None else 0.0,
        }
        
        if self.config.verbose:
            print("\n[Metrics] Evaluation:")
            print(f"   Coherence (mean): {metrics['coherence_mean']:.4f}")
            print(f"   Diversity: {metrics['diversity']:.4f}")
            if stability:
                print(f"   Stability: {stability:.4f}")
            print(f"   Outlier ratio: {metrics['outlier_ratio']:.2%}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import pickle

        state = {
            "config": self.config,
            "n_topics": self.n_topics,
            "topics_": self.topics_,
            "labels_": self.labels_,
            "embeddings_": self.embeddings_,
            "original_embeddings_": self.original_embeddings_,
            "reduced_embeddings_": self.reduced_embeddings_,
            "probabilities_": self.probabilities_,
            "lexical_matrix_": self.lexical_matrix_,
            "topic_embeddings_": self.topic_embeddings_,
            "documents_": self.documents_,
            "hierarchy_": self.hierarchy_,
            "_is_fitted": self._is_fitted,
            "_iteration_history": self._iteration_history,
            "_dim_reducer": self._dim_reducer,
            "_keyword_extractor_state": {
                "vectorizer": self._keyword_extractor._vectorizer,
                "vocabulary": self._keyword_extractor._vocabulary,
                "idf": getattr(self._keyword_extractor, "_idf", None),
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        if self.config.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TriTopic":
        """Load model from disk."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        config = state["config"]
        # Backward compat: ensure new config fields exist for models saved before v2.2
        if not hasattr(config, "language"):
            config.language = "english"
        if not hasattr(config, "soft_assignment_method"):
            config.soft_assignment_method = "centroid"
        if not hasattr(config, "embedding_provider"):
            config.embedding_provider = "local"
        if not hasattr(config, "embedding_api_key"):
            config.embedding_api_key = None
        if not hasattr(config, "embedding_api_batch_size"):
            config.embedding_api_batch_size = 100
        if not hasattr(config, "embedding_output_dim"):
            config.embedding_output_dim = None
        if not hasattr(config, "embedding_task_type"):
            config.embedding_task_type = None
        if not hasattr(config, "embedding_batch_delay"):
            config.embedding_batch_delay = 0.0
        if not hasattr(config, "embedding_prefix"):
            config.embedding_prefix = None

        model = cls(config=config)
        model.n_topics = state.get("n_topics", "auto")
        model.topics_ = state["topics_"]
        model.labels_ = state["labels_"]
        model.embeddings_ = state["embeddings_"]
        model.original_embeddings_ = state.get("original_embeddings_")
        model.reduced_embeddings_ = state.get("reduced_embeddings_")
        model.probabilities_ = state.get("probabilities_")
        model.lexical_matrix_ = state.get("lexical_matrix_")
        model.topic_embeddings_ = state["topic_embeddings_"]
        model.documents_ = state["documents_"]
        model.hierarchy_ = state.get("hierarchy_")
        model._is_fitted = state["_is_fitted"]
        model._iteration_history = state["_iteration_history"]
        model._dim_reducer = state.get("_dim_reducer")

        # Restore keyword extractor state
        kw_state = state.get("_keyword_extractor_state")
        if kw_state:
            model._keyword_extractor._vectorizer = kw_state.get("vectorizer")
            model._keyword_extractor._vocabulary = kw_state.get("vocabulary")
            model._keyword_extractor._idf = kw_state.get("idf")

        return model
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        n_topics = len([t for t in self.topics_ if t.topic_id != -1]) if self._is_fitted else "?"
        return f"TriTopic(n_topics={n_topics}, status={status})"
