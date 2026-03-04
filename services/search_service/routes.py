"""Search Service — FastAPI App + Routes — Section 19."""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import FastAPI, Query as QueryParam

from .schemas import SearchResponse, SearchResult

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Search Service", version="1.0.0")

# In-memory product index (Elasticsearch in production)
_search_index: list[dict] = []


@app.post("/api/v1/search/index")
async def index_product(product: dict):
    """Index a product for search."""
    _search_index.append(product)
    return {"status": "indexed", "total": len(_search_index)}


@app.get("/api/v1/search", response_model=SearchResponse)
async def search(
    q: str = QueryParam(..., min_length=1, max_length=200),
    category: Optional[int] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    content_type: Optional[str] = None,
    sort: str = "relevance",            # relevance | price_asc | price_desc | newest
    page: int = QueryParam(default=1, ge=1),
    limit: int = QueryParam(default=20, le=50),
):
    """Full-text search with context awareness — Section 19."""
    t0 = time.perf_counter()

    query_lower = q.lower()

    results: list[SearchResult] = []
    live_results: list[SearchResult] = []

    for product in _search_index:
        title = product.get("title", "").lower()
        desc = product.get("description_short", "").lower()
        tags = " ".join(product.get("tags", [])).lower()
        brand = product.get("brand", "").lower()

        searchable = f"{title} {desc} {tags} {brand}"
        terms = query_lower.split()
        match_count = sum(1 for term in terms if term in searchable)

        if match_count == 0:
            continue

        relevance = match_count / len(terms) if terms else 0

        price = product.get("base_price", 0)
        if min_price and price < min_price:
            continue
        if max_price and price > max_price:
            continue
        if category and product.get("category_id") != category:
            continue

        result = SearchResult(
            id=product.get("id", ""),
            title=product.get("title", ""),
            price=price,
            relevance_score=relevance,
            content_type=product.get("content_type", "photo"),
        )

        results.append(result)

    # Sort
    if sort == "price_asc":
        results.sort(key=lambda x: x.price)
    elif sort == "price_desc":
        results.sort(key=lambda x: x.price, reverse=True)
    else:
        results.sort(key=lambda x: x.relevance_score, reverse=True)

    # Pagination
    total = len(results)
    start = (page - 1) * limit
    results = results[start:start + limit]

    took_ms = (time.perf_counter() - t0) * 1000

    suggestions = _generate_suggestions(query_lower)

    return SearchResponse(
        results=results,
        total_hits=total,
        live_results=live_results,
        autocomplete_suggestions=suggestions,
        took_ms=round(took_ms, 1),
    )


def _generate_suggestions(query: str) -> list[str]:
    """Generate autocomplete suggestions from indexed products."""
    suggestions = set()
    for product in _search_index[:1000]:
        title = product.get("title", "")
        if query in title.lower():
            suggestions.add(title[:50])
            if len(suggestions) >= 5:
                break
    return list(suggestions)


@app.get("/api/v1/search/autocomplete")
async def autocomplete(
    q: str = QueryParam(..., min_length=1, max_length=100),
    limit: int = QueryParam(default=5, le=10),
):
    """Fast autocomplete suggestions — <10ms with Typesense."""
    suggestions = _generate_suggestions(q.lower())[:limit]
    return {"suggestions": suggestions}
