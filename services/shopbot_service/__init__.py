"""
ShopBot Service — Self-Hosted Catalog-Aware AI Shop Assistant
=============================================================
Production-ready RAG + vLLM service for ShopFeed.
Serves as an intelligent "rayonniste" (stock clerk) for each shop.

Architecture (2026 Best Practices):
- vLLM with Qwen2.5-VL-7B-Instruct-AWQ + Automatic Prefix Caching (APC)
- pgvector + HNSW for vector storage
- Hybrid Search: BM25 (sparse) + Dense Embeddings (multilingual-e5-large-instruct)
- Embedding Quantization: Binary/int8 for 32x memory savings
- Reciprocal Rank Fusion (RRF) for result merging
- Real-time catalog sync via PostgreSQL LISTEN/NOTIFY
"""

__version__ = "1.0.0"
__author__ = "ShopFeed ML Team"
