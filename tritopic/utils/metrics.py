"""
Evaluation Metrics for Topic Models
====================================

Provides standard metrics for evaluating topic model quality:
- Coherence (NPMI, CV)
- Diversity
- Stability (ARI between runs)
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from itertools import combinations


def compute_coherence(
    keywords: list[str],
    documents: list[str],
    method: str = "npmi",
    window_size: int = 10,
) -> float:
    """
    Compute topic coherence based on keyword co-occurrence.
    
    Parameters
    ----------
    keywords : list[str]
        Topic keywords.
    documents : list[str]
        Documents used to compute co-occurrence.
    method : str
        Coherence method: "npmi" (default), "uci", "umass"
    window_size : int
        Window size for co-occurrence. Default: 10
        
    Returns
    -------
    coherence : float
        Coherence score (higher is better).
    """
    if len(keywords) < 2:
        return 0.0
    
    # Tokenize documents
    def tokenize(text):
        import re
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    doc_tokens = [tokenize(doc) for doc in documents]
    n_docs = len(documents)
    
    # Count document frequencies
    word_doc_freq = Counter()
    for tokens in doc_tokens:
        for word in tokens:
            word_doc_freq[word] += 1
    
    # Count co-occurrences (document-level)
    pair_doc_freq = Counter()
    for tokens in doc_tokens:
        for w1, w2 in combinations(keywords, 2):
            if w1.lower() in tokens and w2.lower() in tokens:
                pair_doc_freq[(w1.lower(), w2.lower())] += 1
    
    # Compute coherence
    coherence_scores = []
    
    for w1, w2 in combinations(keywords, 2):
        w1_lower, w2_lower = w1.lower(), w2.lower()
        
        freq_w1 = word_doc_freq.get(w1_lower, 0)
        freq_w2 = word_doc_freq.get(w2_lower, 0)
        freq_pair = pair_doc_freq.get((w1_lower, w2_lower), 0)
        
        if freq_w1 == 0 or freq_w2 == 0:
            continue
        
        if method == "npmi":
            # Normalized Pointwise Mutual Information
            # Use consistent epsilon smoothing on all probabilities
            epsilon = 1e-12
            p_w1 = (freq_w1 + 1) / (n_docs + 1)
            p_w2 = (freq_w2 + 1) / (n_docs + 1)
            p_pair = (freq_pair + 1) / (n_docs + 1)

            pmi = np.log(p_pair / (p_w1 * p_w2))
            npmi = pmi / (-np.log(p_pair) + epsilon)
            coherence_scores.append(npmi)

        elif method == "uci":
            # UCI coherence
            epsilon = 1e-12
            p_pair = (freq_pair + 1) / (n_docs + 1)
            p_w1 = (freq_w1 + 1) / (n_docs + 1)
            p_w2 = (freq_w2 + 1) / (n_docs + 1)

            pmi = np.log(p_pair / (p_w1 * p_w2 + epsilon))
            coherence_scores.append(pmi)
            
        elif method == "umass":
            # UMass coherence
            if freq_w2 > 0:
                score = np.log((freq_pair + 1) / freq_w2)
                coherence_scores.append(score)
    
    return float(np.mean(coherence_scores)) if coherence_scores else 0.0


def compute_diversity(
    all_keywords: list[str],
    n_topics: int,
) -> float:
    """
    Compute topic diversity (proportion of unique keywords).
    
    Diversity measures how different topics are from each other.
    A model where every topic has the same keywords has diversity 0.
    
    Parameters
    ----------
    all_keywords : list[str]
        All keywords from all topics (flattened).
    n_topics : int
        Number of topics.
        
    Returns
    -------
    diversity : float
        Diversity score between 0 and 1 (higher is better).
    """
    if not all_keywords or n_topics == 0:
        return 0.0
    
    unique_keywords = set(kw.lower() for kw in all_keywords)
    
    # Diversity = unique keywords / total keywords
    diversity = len(unique_keywords) / len(all_keywords)
    
    return float(diversity)


def compute_stability(
    partitions: list[np.ndarray],
) -> float:
    """
    Compute clustering stability as average pairwise ARI.
    
    Parameters
    ----------
    partitions : list[np.ndarray]
        Multiple cluster assignments from different runs.
        
    Returns
    -------
    stability : float
        Average Adjusted Rand Index between partitions.
    """
    from sklearn.metrics import adjusted_rand_score
    
    if len(partitions) < 2:
        return 1.0
    
    ari_scores = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            ari = adjusted_rand_score(partitions[i], partitions[j])
            ari_scores.append(ari)
    
    return float(np.mean(ari_scores))


def compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute silhouette score for cluster quality.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings.
    labels : np.ndarray
        Cluster assignments.
        
    Returns
    -------
    silhouette : float
        Silhouette score between -1 and 1 (higher is better).
    """
    from sklearn.metrics import silhouette_score
    
    # Filter out outliers
    mask = labels != -1
    if mask.sum() < 2:
        return 0.0
    
    unique_labels = np.unique(labels[mask])
    if len(unique_labels) < 2:
        return 0.0
    
    return float(silhouette_score(embeddings[mask], labels[mask]))


def compute_downstream_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
    task: str = "classification",
) -> float:
    """
    Evaluate topic model by downstream task performance.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings.
    labels : np.ndarray
        Topic assignments.
    y_true : np.ndarray
        True labels for downstream task.
    task : str
        Task type: "classification" or "clustering"
        
    Returns
    -------
    score : float
        Task-specific score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, adjusted_rand_score
    from sklearn.model_selection import cross_val_score
    
    # Create topic features (one-hot + embedding)
    unique_topics = np.unique(labels[labels != -1])
    n_topics = len(unique_topics)

    # Create mapping from label to contiguous index
    label_to_idx = {label: idx for idx, label in enumerate(unique_topics)}

    # One-hot encode topics
    topic_features = np.zeros((len(labels), n_topics + 1))
    for i, label in enumerate(labels):
        if label == -1:
            topic_features[i, -1] = 1  # Outlier feature
        else:
            topic_features[i, label_to_idx[label]] = 1
    
    # Combine with embeddings
    features = np.hstack([embeddings, topic_features])
    
    if task == "classification":
        # Cross-validated F1
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, features, y_true, cv=5, scoring="f1_macro")
        return float(np.mean(scores))
    else:
        # Clustering ARI
        return float(adjusted_rand_score(labels, y_true))
