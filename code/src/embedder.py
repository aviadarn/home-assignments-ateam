"""
Post Embedder using sentence-transformers all-MiniLM-L6-v2.
Downloads model on first run (~90MB). Subsequent runs use local cache.
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # Lazy-loaded singleton


def _get_model() -> SentenceTransformer:
    """Load model once, reuse across calls."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_posts(posts: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Encode post texts into 384-dim embedding vectors.

    Args:
        posts: List of post dicts (must have 'id' and 'text' fields)

    Returns:
        Dict mapping post_id -> numpy array of shape (384,)
    """
    model = _get_model()
    texts = [p["text"] for p in posts]
    post_ids = [p["id"] for p in posts]

    # Encode all at once (batched internally by sentence-transformers)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    return {pid: emb for pid, emb in zip(post_ids, embeddings)}
