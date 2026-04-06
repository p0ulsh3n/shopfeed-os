"""
Search Module — Visual, Text, Cross-Modal & Re-ranking (2026 best practices)
==============================================================================
Gap 1: Visual Search     → POST /v1/search/visual
Gap 2: Hybrid Text Search → POST /v1/search/text
Gap 3: Cross-Modal Bridge → POST /v1/search/associated
Gap 4: LambdaMART Re-ranking (integrated into all search pipelines)
Gap 5: Category-aware partition routing (Milvus partition_key)

Best practices (verified April 2026):
    - RRF k=60 for hybrid fusion (industry standard)
    - LightGBM LambdaMART with lambdarank objective
    - Milvus partition_key on category_id
    - Pre-filtering by predicted category before vector search
    - Embedding cache in Redis
"""

from ml.search.visual_search import VisualSearchPipeline
from ml.search.hybrid_search import HybridSearchPipeline
from ml.search.cross_modal import CrossModalBridge
from ml.search.reranker import SearchReranker
from ml.search.category_router import CategoryRouter
from ml.search.elasticsearch_backend import ElasticsearchBackend
from ml.search.clip_onnx_inference import CLIPOnnxInference
from ml.search.click_training import SearchClickCollector, LambdaMARTTrainingPipeline

__all__ = [
    "VisualSearchPipeline",
    "HybridSearchPipeline",
    "CrossModalBridge",
    "SearchReranker",
    "CategoryRouter",
    "ElasticsearchBackend",
    "CLIPOnnxInference",
    "SearchClickCollector",
    "LambdaMARTTrainingPipeline",
]
