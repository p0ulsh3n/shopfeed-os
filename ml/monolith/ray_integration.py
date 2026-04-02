"""
Ray Train Integration — distributed online training (Section 14)
=================================================================
Wraps MonolithStreamingTrainer in Ray's distributed training framework.

BUG #12 FIX: Previously, each Ray worker ran a fully independent
MonolithStreamingTrainer with no gradient synchronization between workers.
This produced N siloed models writing to Redis with last-write-wins races —
NOT distributed training.

Fix: wrap delta_model with ray.train.torch.prepare_model() for DDP AllReduce
so that gradient updates to the delta_model are synchronized across workers.
The Cuckoo embedding table remains per-worker (approximation acceptable for
online learning — embeddings converge through shared Redis sync every 60s).
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

    BUG #12 FIX: delta_model is now wrapped with ray.train.torch.prepare_model()
    for DDP AllReduce gradient synchronization. The Cuckoo embedding table
    remains per-worker (each worker consumes a distinct Kafka partition shard)
    but converges through shared Redis sync every 60s per worker.

    Trade-off vs Monolith paper's Parameter Server:
    - PS aggregates gradients exactly at every step (optimal)
    - DDP AllReduce averages gradients every micro-batch (good approximation)
    - Cuckoo table updates remain asynchronous (acceptable for online learning)
    """
    cfg = config or MonolithConfig()

    def train_loop_per_worker() -> None:
        """Ray worker entrypoint — each worker runs a streaming trainer."""
        import ray.train as ray_train
        import ray.train.torch as ray_torch

        rank = ray_train.get_context().get_world_rank()

        # Each worker consumes a distinct Redpanda partition shard via group_id suffix
        worker_config = MonolithConfig(
            redpanda_brokers=cfg.redpanda_brokers,
            kafka_topics=cfg.kafka_topics,
            kafka_group_id=f"{cfg.kafka_group_id}-{rank}",
            redis_url=cfg.redis_url,
            embed_dim=cfg.embed_dim,
            learning_rate=cfg.learning_rate,
            # BUG #12 FIX: increase sync frequency per worker so the shared Redis
            # store converges the distributed embedding updates faster.
            # Previously 600s → now 60s when using multiple workers.
            sync_interval_s=min(cfg.sync_interval_s, 60.0),
        )
        trainer = MonolithStreamingTrainer(worker_config)

        # BUG #12 FIX: wrap delta_model with DDP for AllReduce gradient sync.
        # This ensures delta_model weights are synchronized across all workers
        # after each optimizer step, replacing the Parameter Server from the
        # original Monolith paper.
        trainer.delta_model = ray_torch.prepare_model(trainer.delta_model)
        logger.info(
            "Worker %d: delta_model wrapped with DDP (AllReduce enabled)", rank
        )

        # Run the Kafka consumer loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(trainer.run())
        finally:
            loop.close()

        # Report metrics to Ray Train
        ray_train.report({"total_events": trainer._total_events})

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
