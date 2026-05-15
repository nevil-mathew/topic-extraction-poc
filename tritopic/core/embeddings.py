"""
Embedding Engine for TriTopic
==============================

Handles document embedding with support for multiple models and providers:
- Local: Sentence-BERT, Instructor, BGE models via sentence-transformers
- API:   Google Gemini (gemini-embedding-2, gemini-embedding-001)
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from tqdm import tqdm


# Native full dimensions for known API models.
# output_dim overrides these when set (Matryoshka truncation).
_KNOWN_API_DIMS: dict[tuple[str, str], int] = {
    ("google", "gemini-embedding-2"): 3072,
    ("google", "gemini-embedding-001"): 768,
    ("google", "text-embedding-004"): 768,   # deprecated but still callable
}

# Models that use prompt-prefix for task context instead of the task_type API param.
_GOOGLE_PROMPT_PREFIX_MODELS: frozenset[str] = frozenset({"gemini-embedding-2"})


class EmbeddingEngine:
    """
    Generate document embeddings using transformer models or cloud APIs.

    Supports local sentence-transformers models and Google Gemini embedding
    APIs. The provider is selected via the ``provider`` parameter; all other
    behaviour (batching, normalization, progress) is uniform across providers.

    Parameters
    ----------
    model_name : str
        For ``provider="local"``: sentence-transformers model name.
        For ``provider="google"``: Gemini model ID (defaults to
        ``"gemini-embedding-2"`` when left at the local sentinel).

        Local model choices:

        - ``"all-MiniLM-L6-v2"``: Fast, good quality (default)
        - ``"all-mpnet-base-v2"``: Higher quality, slower
        - ``"BAAI/bge-base-en-v1.5"``: State-of-the-art English
        - ``"BAAI/bge-m3"``: Multilingual
        - ``"hkunlp/instructor-large"``: Task-specific (use with ``instruction``)

    batch_size : int
        Batch size for local GPU encoding. Default: 32
    device : str or None
        Device for local models (``"cuda"``, ``"cpu"``, or None for auto).
    show_progress : bool
        Show progress bar during encoding. Default: True
    provider : str
        Embedding provider: ``"local"`` (default) or ``"google"``.
    api_key : str or None
        API key for the chosen provider. Not required for ``"local"``.
    api_batch_size : int
        Documents per API request (max 250 for Gemini). Default: 100
    output_dim : int or None
        Matryoshka output dimensionality for Gemini models (128–3072).
        Defaults to 768 for Google provider — same API cost, practical memory.
        Pass ``output_dim=3072`` for full resolution.
    task_type : str or None
        Task hint for ``gemini-embedding-001``: ``"CLUSTERING"``,
        ``"RETRIEVAL_DOCUMENT"``, ``"RETRIEVAL_QUERY"``, etc.
        For ``gemini-embedding-2``, this is automatically converted to a
        prompt prefix (the model does not accept the API param).
        Ignored for local models.
    batch_delay : float
        Seconds to sleep between API batches. Default: 0.0.
        Set ~4.0 on the Gemini free tier (5–15 RPM limit).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
        show_progress: bool = True,
        provider: Literal["local", "google"] = "local",
        api_key: str | None = None,
        api_batch_size: int = 100,
        output_dim: int | None = None,
        task_type: str | None = None,
        batch_delay: float = 0.0,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.show_progress = show_progress
        self.provider = provider
        self.api_key = api_key
        self.api_batch_size = api_batch_size
        self.output_dim = output_dim
        self.task_type = task_type
        self.batch_delay = batch_delay

        self._model: Any = None    # local SentenceTransformer (lazy)
        self._client: Any = None   # API client (lazy)
        self._is_instructor = "instructor" in model_name.lower()

        # If provider is API-based and model_name is still the local sentinel,
        # swap it to the provider's canonical default.
        if self.provider != "local" and model_name == "all-MiniLM-L6-v2":
            self.model_name = self._default_model()

        # Default output_dim to 768 for Google provider.
        # MRL quality loss <10% vs full 3072; matches local BGE/mpnet dims.
        if self.provider == "google" and self.output_dim is None:
            self.output_dim = 768

    # ------------------------------------------------------------------
    # Provider defaults
    # ------------------------------------------------------------------

    def _default_model(self) -> str:
        """Return the canonical default model name for the active provider."""
        if self.provider == "google":
            return "gemini-embedding-2"
        return "all-MiniLM-L6-v2"

    # ------------------------------------------------------------------
    # API client management
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Lazily initialize the API client for the active provider."""
        if self._client is not None:
            return
        if self.provider == "google":
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install with: pip install 'tritopic[llm]'"
                )
        # future providers: elif self.provider == "openai": ...

    def _call_with_retry(self, fn):
        """Call fn() up to 3 times with exponential backoff on transient errors."""
        for attempt in range(3):
            try:
                return fn()
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    # ------------------------------------------------------------------
    # Local model management
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy load the local sentence-transformers model."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Provider-specific encode implementations
    # ------------------------------------------------------------------

    def _encode_google(
        self,
        documents: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode documents via the Google Gemini embedding API.

        Parameters
        ----------
        documents : list[str]
            Texts to embed.
        normalize : bool
            L2-normalize the output. Default: True.
            ``gemini-embedding-2`` auto-normalizes truncated dims; re-normalizing
            already-unit vectors is a no-op and safe for all models.

        Returns
        -------
        embeddings : np.ndarray
            Shape ``(n_docs, embedding_dim)``.
        """
        from google.genai import types

        self._init_client()

        use_prompt_prefix = self.model_name in _GOOGLE_PROMPT_PREFIX_MODELS

        # gemini-embedding-2 does not accept task_type as an API param;
        # inject it as a text prefix instead.
        if use_prompt_prefix and self.task_type:
            task_prefix = f"task: {self.task_type.lower()} | text: "
            contents = [task_prefix + doc for doc in documents]
        else:
            contents = documents

        # Build embed config — task_type only for models that support it.
        if not use_prompt_prefix and self.task_type:
            embed_config = types.EmbedContentConfig(
                task_type=self.task_type,
                output_dimensionality=self.output_dim,
            )
        else:
            embed_config = types.EmbedContentConfig(
                output_dimensionality=self.output_dim,
            )

        all_embeddings: list[list[float]] = []
        batches = [
            contents[i : i + self.api_batch_size]
            for i in range(0, len(contents), self.api_batch_size)
        ]

        for batch in tqdm(batches, disable=not self.show_progress, desc="Embedding (google)"):
            response = self._call_with_retry(
                lambda b=batch: self._client.models.embed_content(
                    model=self.model_name,
                    contents=b,
                    config=embed_config,
                )
            )
            all_embeddings.extend([list(e.values) for e in response.embeddings])
            if self.batch_delay > 0:
                time.sleep(self.batch_delay)

        result = np.array(all_embeddings, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / (norms + 1e-10)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        documents: list[str],
        instruction: str | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode documents to embeddings.

        Parameters
        ----------
        documents : list[str]
            List of document texts.
        instruction : str, optional
            Instruction prefix for Instructor models (local provider only).
        normalize : bool
            Whether to L2-normalize embeddings. Default: True

        Returns
        -------
        embeddings : np.ndarray
            Document embeddings of shape ``(n_docs, embedding_dim)``.
        """
        if self.provider == "google":
            return self._encode_google(documents, normalize=normalize)

        # --- local path (unchanged) ---
        self._load_model()

        if self._is_instructor and instruction:
            documents = [[instruction, doc] for doc in documents]

        return self._model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

    def encode_with_pooling(
        self,
        documents: list[str],
        pooling: Literal["mean", "max", "cls"] = "mean",
    ) -> np.ndarray:
        """
        Encode with custom pooling strategy.

        Parameters
        ----------
        documents : list[str]
            Document texts.
        pooling : str
            Pooling strategy: ``"mean"``, ``"max"``, or ``"cls"``.

        Returns
        -------
        embeddings : np.ndarray
            Pooled embeddings.
        """
        return self.encode(documents)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality for the active provider and model."""
        if self.provider != "local":
            # output_dim (MRL truncation) IS the effective dimension when set.
            if self.output_dim is not None:
                return self.output_dim
            key = (self.provider, self.model_name)
            if key in _KNOWN_API_DIMS:
                return _KNOWN_API_DIMS[key]
            # Fallback: single probe call to discover dimension.
            probe = self.encode(["probe"])
            return probe.shape[1]
        # Local path: unchanged.
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.

        Parameters
        ----------
        embeddings1 : np.ndarray
            First set of embeddings.
        embeddings2 : np.ndarray, optional
            Second set. If None, compute pairwise similarity of embeddings1.

        Returns
        -------
        similarity : np.ndarray
            Similarity matrix.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if embeddings2 is None:
            return cosine_similarity(embeddings1)
        return cosine_similarity(embeddings1, embeddings2)


class MultiModelEmbedding:
    """
    Combine embeddings from multiple models.

    Useful for ensemble approaches where different models capture
    different aspects of document semantics.
    """

    def __init__(
        self,
        model_names: list[str],
        weights: list[float] | None = None,
        batch_size: int = 32,
    ):
        self.model_names = model_names
        self.weights = weights or [1.0 / len(model_names)] * len(model_names)
        self.batch_size = batch_size

        self._engines = [
            EmbeddingEngine(name, batch_size=batch_size)
            for name in model_names
        ]

    def encode(
        self,
        documents: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode using all models and combine.

        Parameters
        ----------
        documents : list[str]
            Document texts.
        normalize : bool
            Normalize final embeddings.

        Returns
        -------
        embeddings : np.ndarray
            Combined embeddings (concatenated and weighted).
        """
        all_embeddings = []

        for engine, weight in zip(self._engines, self.weights):
            emb = engine.encode(documents, normalize=True)
            all_embeddings.append(emb * weight)

        combined = np.hstack(all_embeddings)

        if normalize:
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            combined = combined / (norms + 1e-10)

        return combined
