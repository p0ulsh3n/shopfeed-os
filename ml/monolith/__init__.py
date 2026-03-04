"""
Monolith — Online Streaming Training, Vitesse 2 (Section 14)
==============================================================
Inspired by ByteDance's Monolith paper (2022). 2026 re-implementation
using the modern production stack:

    Kafka + Flink    → streaming event ingestion + feature engineering
    PyTorch + Ray    → distributed online training (replaces PS)
    Redis            → feature store serving (<5ms reads)
    Triton           → model inference serving (updated every 5-15 min)

Modules:
  - cuckoo_table:       CuckooEmbeddingTable — collision-free hash, LRU eviction
  - redis_store:        RedisFeatureStore — <5ms feature serving with auto-sync
  - delta_model:        DeltaModel — real-time score adjustments
  - streaming_trainer:  MonolithStreamingTrainer — Ray-based distributed online SGD
  - triton_publisher:   TritonModelPublisher — pushes weights every 5-15 min
  - ray_integration:    create_ray_trainer — distributed training wrapper
"""

from .cuckoo_table import CuckooEmbeddingTable
from .delta_model import DeltaModel
from .ray_integration import create_ray_trainer
from .redis_store import RedisFeatureStore
from .streaming_trainer import (
    ACTION_WEIGHTS,
    MonolithConfig,
    MonolithStreamingTrainer,
)
from .triton_publisher import TritonModelPublisher

__all__ = [
    # Core data structures
    "CuckooEmbeddingTable",
    "RedisFeatureStore",
    # Models
    "DeltaModel",
    # Training
    "MonolithConfig",
    "MonolithStreamingTrainer",
    "ACTION_WEIGHTS",
    # Serving
    "TritonModelPublisher",
    # Ray
    "create_ray_trainer",
]
