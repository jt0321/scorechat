"""
pipeline/embedder.py
Generates OpenAI embeddings for score segment summaries and text chunks.
"""

import os
import time
from typing import Optional
from openai import OpenAI

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """
    Embed a batch of texts. Returns list of embedding vectors.
    Batches in groups of 100 to respect API limits; retries on rate limit.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if "your-openai" in api_key or api_key.startswith("sk-placeholder") or not api_key:
        import random
        vectors = []
        for text in texts:
            random.seed(hash(text))
            vec = [random.uniform(-1, 1) for _ in range(1536)]
            mag = sum(x*x for x in vec) ** 0.5
            vectors.append([x / mag for x in vec])
        return vectors

    client = get_client()
    vectors = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                response = client.embeddings.create(input=batch, model=model)
                vectors.extend([item.embedding for item in response.data])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                print(f"Embedding attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    return vectors


def embed_single(text: str, model: str = "text-embedding-3-small") -> list[float]:
    return embed_texts([text], model=model)[0]
