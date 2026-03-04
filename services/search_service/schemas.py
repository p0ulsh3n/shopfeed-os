"""Pydantic schemas for Search Service — Section 19."""

from __future__ import annotations

from pydantic import BaseModel


class SearchResult(BaseModel):
    id: str
    title: str
    vendor_name: str = ""
    price: float = 0.0
    currency: str = "EUR"
    thumbnail_url: str = ""
    content_type: str = "photo"
    is_live: bool = False                   # True if vendor is streaming NOW
    relevance_score: float = 0.0
    commerce_score: float = 0.0


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_hits: int = 0
    live_results: list[SearchResult] = []   # Separate live context results
    autocomplete_suggestions: list[str] = []
    took_ms: float = 0.0
