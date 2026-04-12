"""
plagiarism_engine.py
--------------------
Multi-method NLP similarity engine:
  1. TF-IDF + Cosine Similarity  (lexical)
  2. Sentence-Transformers        (semantic / paraphrase-aware)
  3. Jaccard on character n-grams (structural / copy-paste detection)

Final score = weighted ensemble of all three methods.
"""

import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lazy-load heavy model to avoid import-time delay
_sentence_model = None
_sentence_model_loaded = False

# ─── Weights ─────────────────────────────────────────────────────────────────
WEIGHT_TFIDF     = 0.40
WEIGHT_SEMANTIC  = 0.45
WEIGHT_JACCARD   = 0.15

# ─── NLP Preprocessing ───────────────────────────────────────────────────────

def _preprocess(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _get_ngrams(text: str, n: int = 3) -> set:
    """Character-level n-grams as a set."""
    text = _preprocess(text)
    return set(text[i:i+n] for i in range(len(text) - n + 1))


# ─── Individual Similarity Methods ───────────────────────────────────────────

def _tfidf_similarity(texts: list[str]) -> np.ndarray:
    """Pairwise TF-IDF cosine similarity matrix."""
    n = len(texts)
    processed = [_preprocess(t) for t in texts]
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams for richer signal
            min_df=1,
            sublinear_tf=True     # log-scale TF to reduce dominance of frequent terms
        )
        tfidf_matrix = vectorizer.fit_transform(processed)
        sim = cosine_similarity(tfidf_matrix)
        return sim
    except Exception:
        return np.zeros((n, n))


def _semantic_similarity(texts: list[str]) -> np.ndarray:
    """Semantic similarity using sentence-transformers (MiniLM)."""
    global _sentence_model, _sentence_model_loaded
    n = len(texts)

    if not _sentence_model_loaded:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _sentence_model = SentenceTransformer("all-mpnet-base-v2", device=device)
            _sentence_model_loaded = True
            print(f"[INFO] Sentence-Transformers model loaded on {device}.")
        except Exception as e:
            print(f"[WARN] Sentence-Transformers unavailable: {e}. Skipping semantic scoring.")
            _sentence_model_loaded = True  # mark as attempted

    if _sentence_model is None:
        return np.zeros((n, n))

    try:
        # Truncate long documents for encoder efficiency (model max is 512 tokens)
        truncated = [t[:4000] for t in texts]
        embeddings = _sentence_model.encode(truncated, show_progress_bar=False, normalize_embeddings=True)
        sim = embeddings @ embeddings.T  # cosine sim since normalized
        sim = np.clip(sim, 0, 1)
        return sim
    except Exception as e:
        print(f"[WARN] Semantic similarity failed: {e}")
        return np.zeros((n, n))


def _jaccard_similarity(texts: list[str]) -> np.ndarray:
    """Pairwise Jaccard similarity on character 2-grams."""
    n = len(texts)
    ngrams = [_get_ngrams(t, n=2) for t in texts]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim[i][j] = 1.0
            elif not ngrams[i] or not ngrams[j]:
                sim[i][j] = 0.0
            else:
                intersection = len(ngrams[i] & ngrams[j])
                union = len(ngrams[i] | ngrams[j])
                sim[i][j] = intersection / union if union > 0 else 0.0
    return sim


# ─── Ensemble ────────────────────────────────────────────────────────────────

def compute_similarities(texts: list[str]) -> dict:
    """
    Compute all pairwise similarity matrices and return an ensemble result.
    Returns a dict with individual and ensemble matrices.
    """
    n = len(texts)
    if n < 2:
        raise ValueError("Need at least 2 documents to compare.")

    print("[INFO] Computing TF-IDF similarity...")
    tfidf_sim    = _tfidf_similarity(texts)

    print("[INFO] Computing semantic similarity...")
    semantic_sim = _semantic_similarity(texts)

    print("[INFO] Computing Jaccard n-gram similarity...")
    jaccard_sim  = _jaccard_similarity(texts)

    # Detect if semantic model is unavailable — rebalance weights
    if np.all(semantic_sim == 0):
        w_tfidf, w_sem, w_jac = 0.70, 0.00, 0.30
    else:
        w_tfidf, w_sem, w_jac = WEIGHT_TFIDF, WEIGHT_SEMANTIC, WEIGHT_JACCARD

    ensemble = w_tfidf * tfidf_sim + w_sem * semantic_sim + w_jac * jaccard_sim
    ensemble = np.clip(ensemble, 0, 1)

    return {
        "tfidf":    tfidf_sim.tolist(),
        "semantic": semantic_sim.tolist(),
        "jaccard":  jaccard_sim.tolist(),
        "ensemble": ensemble.tolist(),
    }


def build_results(filenames: list[str], sim_matrices: dict) -> list[dict]:
    """
    Build a flat list of pairwise results sorted descending by ensemble similarity.
    Each entry: {file1, file2, ensemble_pct, tfidf_pct, semantic_pct, jaccard_pct}
    """
    n = len(filenames)
    ensemble = np.array(sim_matrices["ensemble"])
    tfidf    = np.array(sim_matrices["tfidf"])
    semantic = np.array(sim_matrices["semantic"])
    jaccard  = np.array(sim_matrices["jaccard"])

    results = []
    for i in range(n):
        for j in range(i + 1, n):
            results.append({
                "file1":         filenames[i],
                "file2":         filenames[j],
                "similarity":    round(float(ensemble[i][j]) * 100, 2),
                "tfidf_pct":     round(float(tfidf[i][j])    * 100, 2),
                "semantic_pct":  round(float(semantic[i][j]) * 100, 2),
                "jaccard_pct":   round(float(jaccard[i][j])  * 100, 2),
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results
