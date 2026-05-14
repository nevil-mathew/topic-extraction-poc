"""
Keyword Extraction for TriTopic
================================

Extract representative keywords for topics using:
- c-TF-IDF (class-based TF-IDF, like BERTopic)
- BM25 scoring
- KeyBERT (embedding-based)
"""

from __future__ import annotations

from typing import Literal
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class KeywordExtractor:
    """
    Extract keywords for topics.
    
    Supports multiple extraction methods for flexibility.
    
    Parameters
    ----------
    method : str
        Extraction method: "ctfidf", "bm25", or "keybert"
    n_keywords : int
        Number of keywords to extract per topic. Default: 10
    ngram_range : tuple
        N-gram range for keyword extraction. Default: (1, 2)
    """
    
    def __init__(
        self,
        method: Literal["ctfidf", "bm25", "keybert"] = "ctfidf",
        n_keywords: int = 10,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        language: str = "english",
    ):
        self.method = method
        self.n_keywords = n_keywords
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.language = language

        self._vectorizer = None
        self._vocabulary = None
        self._idf = None
        self._keybert_model = None

    def extract(
        self,
        topic_docs: list[str],
        all_docs: list[str] | None = None,
        n_keywords: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """
        Extract keywords from topic documents.
        
        Parameters
        ----------
        topic_docs : list[str]
            Documents belonging to the topic.
        all_docs : list[str], optional
            All documents in corpus (needed for c-TF-IDF).
        n_keywords : int, optional
            Override default n_keywords.
            
        Returns
        -------
        keywords : list[str]
            Top keywords for the topic.
        scores : list[float]
            Keyword scores.
        """
        n = n_keywords or self.n_keywords
        
        if self.method == "ctfidf":
            return self._extract_ctfidf(topic_docs, all_docs or topic_docs, n)
        elif self.method == "bm25":
            return self._extract_bm25(topic_docs, all_docs or topic_docs, n)
        elif self.method == "keybert":
            return self._extract_keybert(topic_docs, n)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def reset(self) -> None:
        """Reset the fitted vectorizer state. Call before re-fitting on new data."""
        self._vectorizer = None
        self._vocabulary = None
        self._idf = None
        self._keybert_model = None

    def _extract_ctfidf(
        self,
        topic_docs: list[str],
        all_docs: list[str],
        n_keywords: int,
    ) -> tuple[list[str], list[float]]:
        """
        Extract keywords using class-based TF-IDF (c-TF-IDF).

        c-TF-IDF treats all documents in a topic as a single "class document"
        and computes TF-IDF against the corpus. This highlights words that
        are distinctive for the topic.
        """
        # Fit vectorizer and cache IDF on all docs (once)
        if self._vectorizer is None:
            from tritopic.utils.stopwords import get_stopwords
            self._vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                stop_words=get_stopwords(self.language),
                min_df=self.min_df,
                max_df=self.max_df,
            )
            all_tf_sparse = self._vectorizer.fit_transform(all_docs)
            self._vocabulary = self._vectorizer.get_feature_names_out()
            # Cache IDF: compute document frequency on the sparse matrix directly
            doc_freq = np.diff(all_tf_sparse.indptr)  # not per-term
            # Per-term document frequency: number of docs where term > 0
            doc_freq = np.asarray((all_tf_sparse > 0).sum(axis=0)).ravel()
            self._idf = np.log(len(all_docs) / (1 + doc_freq))

        # Concatenate topic docs into a single "class document"
        topic_text = " ".join(topic_docs)

        # Get term frequencies for topic (single doc → small dense array is fine)
        topic_tf = self._vectorizer.transform([topic_text]).toarray()[0]

        # c-TF-IDF = normalized_TF * IDF
        topic_tf_normalized = topic_tf / (topic_tf.sum() + 1e-10)
        ctfidf_scores = topic_tf_normalized * self._idf

        # Get top keywords
        top_indices = np.argsort(ctfidf_scores)[::-1][:n_keywords]

        keywords = [self._vocabulary[i] for i in top_indices]
        scores = [float(ctfidf_scores[i]) for i in top_indices]

        return keywords, scores
    
    def _extract_bm25(
        self,
        topic_docs: list[str],
        all_docs: list[str],
        n_keywords: int,
    ) -> tuple[list[str], list[float]]:
        """
        Extract keywords using BM25 scoring.

        Measures topic-specificity by comparing average BM25 relevance of each
        word within the topic vs. across the entire corpus.
        """
        from rank_bm25 import BM25Okapi
        from tritopic.utils.stopwords import get_stopwords_set
        import re

        _stopwords = get_stopwords_set(self.language)

        def tokenize(text: str) -> list[str]:
            tokens = re.findall(r'\b\w+\b', text.lower())
            return [t for t in tokens if t not in _stopwords and len(t) > 2]

        tokenized_all = [tokenize(doc) for doc in all_docs]
        tokenized_topic = [tokenize(doc) for doc in topic_docs]

        # Build vocabulary from topic docs
        topic_vocab = Counter()
        for tokens in tokenized_topic:
            topic_vocab.update(tokens)

        # Fit BM25 on all docs
        bm25 = BM25Okapi(tokenized_all)

        # Build index of which all_docs indices belong to the topic
        topic_doc_set = set()
        for i, doc in enumerate(all_docs):
            if doc in topic_docs:
                topic_doc_set.add(i)
        # More reliable: use positional matching
        n_all = len(all_docs)
        n_topic = len(topic_docs)

        word_scores = {}
        for word, freq in topic_vocab.items():
            scores = bm25.get_scores([word])
            # Compare: average score within topic vs. rest of corpus
            topic_avg = np.mean([scores[i] for i in range(n_all) if i < n_topic]) if n_topic > 0 else 0
            corpus_avg = np.mean(scores) + 1e-10
            # Topic-specificity: ratio of in-topic relevance to corpus average
            specificity = (topic_avg / corpus_avg) * np.log1p(freq)
            word_scores[word] = specificity

        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

        keywords = [w for w, s in sorted_words[:n_keywords]]
        scores = [s for w, s in sorted_words[:n_keywords]]

        max_score = max(scores) if scores else 1
        scores = [s / max_score for s in scores]

        return keywords, scores
    
    def _extract_keybert(
        self,
        topic_docs: list[str],
        n_keywords: int,
    ) -> tuple[list[str], list[float]]:
        """
        Extract keywords using KeyBERT (embedding-based).

        KeyBERT finds keywords by comparing candidate embeddings
        to the document embedding.  The model is cached across calls.
        """
        from keybert import KeyBERT

        # Concatenate topic docs
        topic_text = " ".join(topic_docs)

        # Reuse cached KeyBERT model
        if self._keybert_model is None:
            self._keybert_model = KeyBERT()

        # Extract keywords
        from tritopic.utils.stopwords import get_stopwords
        keywords_with_scores = self._keybert_model.extract_keywords(
            topic_text,
            keyphrase_ngram_range=self.ngram_range,
            stop_words=get_stopwords(self.language),
            top_n=n_keywords,
            use_mmr=True,  # Maximal Marginal Relevance for diversity
            diversity=0.5,
        )
        
        keywords = [kw for kw, score in keywords_with_scores]
        scores = [float(score) for kw, score in keywords_with_scores]
        
        return keywords, scores
    
    def extract_all_topics(
        self,
        documents: list[str],
        labels: np.ndarray,
        n_keywords: int | None = None,
    ) -> dict[int, tuple[list[str], list[float]]]:
        """
        Extract keywords for all topics at once.
        
        Parameters
        ----------
        documents : list[str]
            All documents.
        labels : np.ndarray
            Topic assignments.
        n_keywords : int, optional
            Override default n_keywords.
            
        Returns
        -------
        topic_keywords : dict
            Mapping from topic_id to (keywords, scores).
        """
        result = {}
        
        for topic_id in np.unique(labels):
            if topic_id == -1:
                continue
                
            mask = labels == topic_id
            topic_docs = [documents[i] for i in np.where(mask)[0]]
            
            keywords, scores = self.extract(topic_docs, documents, n_keywords)
            result[int(topic_id)] = (keywords, scores)
        
        return result


class KeyphraseExtractor:
    """
    Extract keyphrases (multi-word) using YAKE or TextRank.
    """
    
    def __init__(
        self,
        method: Literal["yake", "textrank"] = "yake",
        n_keyphrases: int = 10,
        max_ngram: int = 3,
    ):
        self.method = method
        self.n_keyphrases = n_keyphrases
        self.max_ngram = max_ngram
    
    def extract(self, text: str) -> list[tuple[str, float]]:
        """Extract keyphrases from text."""
        if self.method == "yake":
            return self._extract_yake(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _extract_yake(self, text: str) -> list[tuple[str, float]]:
        """Extract using YAKE algorithm."""
        try:
            import yake
        except ImportError:
            # Fallback to simple extraction
            return self._simple_extract(text)
        
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=self.max_ngram,
            dedupLim=0.7,
            top=self.n_keyphrases,
            features=None,
        )
        
        keywords = kw_extractor.extract_keywords(text)
        
        # YAKE returns (keyword, score) where lower score is better
        # Invert for consistency
        max_score = max(s for _, s in keywords) if keywords else 1
        return [(kw, 1 - s/max_score) for kw, s in keywords]
    
    def _simple_extract(self, text: str) -> list[tuple[str, float]]:
        """Simple n-gram frequency extraction."""
        import re
        from collections import Counter
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Generate n-grams
        ngrams = []
        for n in range(1, self.max_ngram + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                ngrams.append(ngram)
        
        # Count and return top
        counts = Counter(ngrams)
        top = counts.most_common(self.n_keyphrases)
        
        max_count = top[0][1] if top else 1
        return [(phrase, count/max_count) for phrase, count in top]
