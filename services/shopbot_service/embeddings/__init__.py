"""ShopBot embeddings package."""
from services.shopbot_service.embeddings.encoder import (
    EmbeddingEncoder,
    batch_cosine_similarities,
    cosine_similarity,
    deserialize_int8_embedding,
    float32_to_pgvector_str,
)

__all__ = [
    "EmbeddingEncoder",
    "batch_cosine_similarities",
    "cosine_similarity",
    "deserialize_int8_embedding",
    "float32_to_pgvector_str",
]
