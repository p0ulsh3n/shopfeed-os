"""Search Service — Full-Text + Context Search — Section 19."""

from .routes import app
from .schemas import SearchResponse, SearchResult

__all__ = ["app", "SearchResult", "SearchResponse"]
