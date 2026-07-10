"""
pipeline/embedder.py
Generates embeddings for score segment summaries and text chunks via
whichever LangChain embeddings backend EMBEDDING_PROVIDER selects
(see pipeline/providers.py).
"""

from __future__ import annotations
from typing import Optional
from pipeline.providers import get_embeddings_model, embedding_provider_ready

_embeddings = None
_embeddings_model: Optional[str] = None


def get_embeddings(model: Optional[str] = None):
    global _embeddings, _embeddings_model
    if _embeddings is None or _embeddings_model != model:
        _embeddings = get_embeddings_model(model=model)
        _embeddings_model = model
    return _embeddings


def embed_texts(texts: list[str], model: Optional[str] = None) -> list[list[float]]:
    """
    Embed a batch of texts. Returns list of embedding vectors.
    """
    if not embedding_provider_ready():
        import random
        vectors = []
        for t in texts:
            random.seed(hash(t))
            vec = [random.uniform(-1, 1) for _ in range(1536)]
            mag = sum(x * x for x in vec) ** 0.5
            vectors.append([x / mag for x in vec])
        return vectors

    return get_embeddings(model=model).embed_documents(texts)


def embed_single(text: str, model: Optional[str] = None) -> list[float]:
    if not embedding_provider_ready():
        return embed_texts([text], model=model)[0]
    return get_embeddings(model=model).embed_query(text)
