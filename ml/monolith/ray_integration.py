"""
Ray Train Integration — distributed online training (Section 14)
=================================================================
Wraps MonolithStreamingTrainer in Ray's distributed training framework.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .streaming_trainer import MonolithConfig, MonolithStreamingTrainer

logger = logging.getLogger(__name__)


def create_ray_trainer(config: MonolithConfig | None = None) -> Any:
    """Create a Ray TorchTrainer for distributed Monolith training.

    Usage on GPU cluster:
        trainer = create_ray_trainer()
        trainer.fit()

    This wraps MonolithStreamingTrainer in Ray's distributed training
    framework. Each worker independently consumes Kafka partitions
    and trains on its shard.
    """
    cfg = config or MonolithConfig()

    def train_loop_per_worker() -> None:
        """Ray worker entrypoint — each worker runs a streaming trainer."""
        import ray.train as ray_train

        # Each worker gets its own trainer instance
        worker_config = MonolithConfig(
            kafka_bootstrap=cfg.kafka_bootstrap,
            kafka_topics=cfg.kafka_topics,
            kafka_group_id=f"{cfg.kafka_group_id}-{ray_train.get_context().get_world_rank()}",
            redis_url=cfg.redis_url,
            embed_dim=cfg.embed_dim,
            learning_rate=cfg.learning_rate,
        )
        trainer = MonolithStreamingTrainer(worker_config)

        # Run the Kafka consumer loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(trainer.run())
        finally:
            loop.close()

    try:
        import ray
        from ray.train.torch import TorchTrainer
        from ray.train import ScalingConfig

        if not ray.is_initialized():
            ray.init()

        return TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            scaling_config=ScalingConfig(
                num_workers=cfg.ray_num_workers,
                use_gpu=cfg.ray_use_gpu,
            ),
        )
    except ImportError:
        logger.error(
            "Ray not installed. Install with: pip install 'ray[train]'\n"
            "For single-node training, use MonolithStreamingTrainer directly."
        )
        return None
