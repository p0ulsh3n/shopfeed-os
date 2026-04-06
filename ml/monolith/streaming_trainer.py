"""
Streaming Trainer — Ray-based distributed online training (Section 14)
========================================================================
Inspired by ByteDance's Monolith paper (2022).

Architecture:
    Redpanda + Flink  → streaming event ingestion + feature engineering
    PyTorch + Ray     → distributed online training (replaces PS)
    Redis             → feature store serving (<5ms reads)
    Triton            → model inference serving (updated every 5-15 min)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .cuckoo_table import CuckooEmbeddingTable
from .delta_model import DeltaModel
from .redis_store import RedisFeatureStore

logger = logging.getLogger(__name__)

# Action weights from Section 02 (commerce scoring)
ACTION_WEIGHTS: dict[str, float] = {
    "buy_now": 12.0, "purchase": 10.0, "add_to_cart": 8.0,
    "save": 6.0, "share": 5.0, "question": 4.5,
    "review": 4.0, "visit_shop": 3.0, "follow": 2.5,
    "comment": 2.0, "like": 1.0,
    "skip": -4.0, "not_interested": -8.0,
}


@dataclass
class MonolithConfig:
    """Configuration for the Monolith streaming training pipeline."""

    # Embedding
    embed_dim: int = 64
    cuckoo_capacity: int = 2_000_000
    eviction_days: int = 30

    # Training
    learning_rate: float = 0.01
    micro_batch_size: int = 512         # Micro-batch from event buffer
    gradient_accumulation: int = 4      # Accumulate before optimizer step

    # Sync (Parameter Server replacement)
    # 2026 best-practice: 60s default (vs old 600s) to close the gap with
    # TikTok Monolith (<1s hotspot sync). For trending items (live shopping,
    # viral new products) a separate aggressive flush path targets 15s.
    sync_interval_s: float = 60         # 1 min default (was 600s)
    trending_sync_interval_s: float = 15  # Aggressive flush for live/viral items
    checkpoint_interval_s: float = 3600  # Hourly checkpoints

    # Trending detection thresholds — an item is "trending" if it receives
    # more than trending_event_threshold events since the last sync.
    trending_event_threshold: int = 50

    # Infrastructure — Redpanda (Kafka-compatible, archi-2026 §5)
    redpanda_brokers: str = "redpanda-1:9092,redpanda-2:9092,redpanda-3:9092"
    kafka_topics: list[str] = field(default_factory=lambda: [
        "shopfeed.user.events",
        "shopfeed.commerce.events",
        "shopfeed.vendor.events",
    ])
    kafka_group_id: str = "monolith-streaming-trainer"
    redis_url: str = "redis://localhost:6379/0"
    checkpoint_dir: str = "checkpoints/monolith"

    # Ray (distributed training)
    ray_num_workers: int = 2
    ray_use_gpu: bool = True


class MonolithStreamingTrainer:
    """Online streaming trainer — 2026 stack.

    Architecture (replaces the archived bytedance/monolith):

        ┌──────────┐    ┌────────────────┐    ┌──────────────┐
        │ Redpanda │ →  │  This Trainer  │ →  │    Redis     │
        │  Topics  │    │  (PyTorch +    │    │  Feature     │
        │          │    │   Ray Train)   │    │  Store       │
        └──────────┘    └────────┬───────┘    └──────┬───────┘
                                 │                    │
                        ┌────────▼───────┐    ┌──────▼───────┐
                        │  Checkpoint    │    │   Triton     │
                        │  (every hour)  │    │   Inference  │
                        └────────────────┘    │  (load new   │
                                              │   weights    │
                                              │   5-15 min)  │
                                              └──────────────┘

    The key insight from Monolith:
      - Events flow in continuously (no hourly batch)
      - Each event updates embeddings immediately via SGD
      - Updated embeddings synced to Redis every 5-15 min
      - Triton loads new model snapshot for serving

    In production, wrap this in a Ray Train worker. The class itself
    is framework-agnostic: it handles the training loop, you provide
    the infrastructure glue.
    """

    def __init__(self, config: MonolithConfig | None = None):
        self.config = config or MonolithConfig()
        cfg = self.config

        # Core components
        self.embedding_table = CuckooEmbeddingTable(
            embed_dim=cfg.embed_dim,
            capacity=cfg.cuckoo_capacity,
            max_eviction_age_days=cfg.eviction_days,
        )
        self.delta_model = DeltaModel(embed_dim=cfg.embed_dim)
        self.optimizer = torch.optim.SGD(
            self.delta_model.parameters(), lr=cfg.learning_rate,
        )
        self.feature_store = RedisFeatureStore(redis_url=cfg.redis_url)

        # Buffers and state
        self._event_buffer: list[dict[str, Any]] = []
        self._last_sync = time.time()
        self._last_trending_sync = time.time()
        self._last_checkpoint = time.time()
        self._events_processed = 0
        self._total_events = 0
        self._running = False
        # BUG #14 FIX: track how many micro-batches have accumulated
        self._accum_steps = 0
        # Trending item counter: tracks event count per item since last sync.
        # Items exceeding cfg.trending_event_threshold get priority flush.
        self._item_event_counts: dict[str, int] = {}

        logger.info(
            "MonolithStreamingTrainer initialized — "
            "embed_dim=%d, sync=%ds, trending_sync=%ds, lr=%.4f, buffer=%d",
            cfg.embed_dim, cfg.sync_interval_s, cfg.trending_sync_interval_s,
            cfg.learning_rate, cfg.micro_batch_size,
        )

    # ── Event Processing ─────────────────────────────────────

    def process_event(self, event: dict[str, Any]) -> None:
        """Process a single event from Kafka.

        Events are buffered in micro-batches for GPU efficiency,
        but conceptually we train "event by event" — no hourly batching.
        """
        self._event_buffer.append(event)
        self._events_processed += 1
        self._total_events += 1

        if len(self._event_buffer) >= self.config.micro_batch_size:
            self._train_micro_batch()

    def _train_micro_batch(self) -> None:
        """Train on buffered events as a micro-batch.

        This is where the actual gradient computation happens.
        Each event contributes to the delta model and item embedding updates.

        Gradient accumulation: optimizer.step() is called only every
        cfg.gradient_accumulation micro-batches, allowing the model to
        accumulate gradients across multiple Kafka message windows before
        updating weights (reduces optimizer noise from small batches).

        Note: Cuckoo table item embeddings are updated immediately per-event
        (online SGD), independent of the delta_model accumulation schedule.
        """
        if not self._event_buffer:
            return

        cfg = self.config
        self.delta_model.train()

        batch_loss = 0.0
        n_valid = 0

        for event in self._event_buffer:
            item_id = str(event.get("item_id", event.get("product_id", "")))
            user_id = str(event.get("user_id", ""))
            action = str(event.get("action", event.get("event_type", "")))

            # Track event frequency per item for trending detection
            if item_id:
                self._item_event_counts[item_id] = (
                    self._item_event_counts.get(item_id, 0) + 1
                )

            if not item_id:
                continue

            # ── Get or create item embedding ──
            stored_emb = self.embedding_table.get(item_id)
            if stored_emb is None:
                clip_data = event.get("clip_embedding")
                clip_tensor = torch.tensor(clip_data, dtype=torch.float32) if clip_data else None
                init_emb = torch.randn(cfg.embed_dim) * 0.01
                self.embedding_table.put(item_id, init_emb, initial_clip=clip_tensor)
                stored_emb = self.embedding_table.get(item_id)

            if stored_emb is None:
                continue

            # BUG #1 FIX: The stored embedding is a detached clone (requires_grad=False).
            # We must create a new leaf tensor with requires_grad=True so that
            # loss.backward() can compute gradients for the embedding update.
            item_emb = stored_emb.detach().requires_grad_(True)

            # ── Compute label from action ──
            weight = ACTION_WEIGHTS.get(action, 0.5)
            label = torch.tensor([[1.0 if weight > 0 else 0.0]])

            # ── Forward pass through delta model ──
            # In production, user_emb comes from Redis feature store
            # For training, we use content from the event or a default
            user_emb_data = event.get("user_embedding")
            if user_emb_data is not None:
                user_emb = torch.tensor(user_emb_data, dtype=torch.float32)[:cfg.embed_dim]
            else:
                user_emb = torch.zeros(cfg.embed_dim)

            pred = torch.sigmoid(
                self.delta_model(item_emb.unsqueeze(0), user_emb.unsqueeze(0))
            )
            loss = F.binary_cross_entropy(pred, label)

            # ── Backward pass ──
            loss.backward()
            batch_loss += loss.item()
            n_valid += 1

            # BUG #1 FIX: item_emb.grad is now guaranteed non-None after backward()
            # because item_emb was created as a leaf tensor with requires_grad=True.
            if item_emb.grad is not None:
                self.embedding_table.update_embedding(
                    item_id, item_emb.grad, lr=cfg.learning_rate,
                )

            # ── Handle stock events ──
            if action == "stock_update" and event.get("stock", 1) <= 0:
                self.embedding_table.mark_inactive(item_id)

        # BUG #14 FIX: Real gradient accumulation.
        # Only call optimizer.step() every cfg.gradient_accumulation micro-batches.
        # This is distinct from the event buffer size — the optimizer accumulates
        # gradients across N consecutive Kafka windows before each weight update.
        if n_valid > 0:
            self._accum_steps += 1
            if self._accum_steps >= cfg.gradient_accumulation:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._accum_steps = 0

        avg_loss = batch_loss / max(n_valid, 1)
        self._event_buffer.clear()

        now = time.time()

        # ── Trending items: aggressive sync every 15s ─────────
        # Items with high event frequency (live shopping, viral products)
        # are flushed on a faster schedule to close the gap with TikTok
        # Monolith's hotspot update latency. Only trending items are flushed,
        # not the entire embedding table — keeps Redis write pressure low.
        trending_items = {
            iid for iid, cnt in self._item_event_counts.items()
            if cnt >= cfg.trending_event_threshold
        }
        if trending_items and now - self._last_trending_sync > cfg.trending_sync_interval_s:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._sync_trending_to_redis(trending_items))
            except RuntimeError:
                logger.debug("Trending Redis sync deferred — not in async context")
            self._last_trending_sync = now
            # Reset counters for flushed items
            for iid in trending_items:
                self._item_event_counts.pop(iid, None)

        # ── Periodic full sync to Redis ──
        if now - self._last_sync > cfg.sync_interval_s:
            # BUG #13 FIX: asyncio.get_event_loop() is deprecated in Python 3.10+.
            # Use get_running_loop() which raises RuntimeError if not in async context.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._sync_to_redis())
            except RuntimeError:
                # Not in an async context (e.g., called from sync test or CLI).
                # Sync will be handled by sync_to_redis_blocking() or next async run().
                logger.debug("Redis sync deferred — not in async context")
            self._last_sync = now
            # Full sync also resets all item event counters
            self._item_event_counts.clear()

        # ── Periodic checkpoint ──
        if now - self._last_checkpoint > cfg.checkpoint_interval_s:
            self._save_checkpoint()
            self._last_checkpoint = now

    # ── Redis Sync (replaces Parameter Server) ───────────────

    async def _sync_trending_to_redis(self, item_ids: set[str]) -> None:
        """Aggressive partial sync — push only trending item embeddings.

        Runs every 15s for items with high event frequency (live shopping
        sessions, viral new products). Keeps write pressure on Redis low
        while dramatically reducing the latency gap with TikTok Monolith
        for the items that matter most commercially.
        """
        await self.feature_store.connect()
        n_synced = 0
        for item_id in item_ids:
            emb = self.embedding_table.get(item_id)
            if emb is not None:
                await self.feature_store.set_embedding(item_id, emb)
                n_synced += 1

        if n_synced:
            logger.info(
                "Trending sync — %d hot-items flushed to Redis (15s path)",
                n_synced,
            )

    async def _sync_to_redis(self) -> None:
        """Push all updated embeddings + delta scores to Redis.

        This is the 2026 replacement for Monolith's Parameter Server.
        Redis gives <5ms reads at serving time.

        2026 update: runs every 60s (was 5-15 min) to close the latency
        gap with TikTok Monolith. Trending items get the faster 15s path
        via _sync_trending_to_redis().
        """
        await self.feature_store.connect()
        n_synced = await self.feature_store.sync_from_cuckoo(self.embedding_table)

        logger.info(
            "Redis full-sync — %d embeddings pushed, %d events since last sync",
            n_synced, self._events_processed,
        )
        self._events_processed = 0

    def sync_to_redis_blocking(self) -> None:
        """Synchronous version for non-async contexts."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._sync_to_redis())
        finally:
            loop.close()

    # ── Scoring (used by ranking service) ────────────────────

    def get_delta_score(self, item_id: str, user_embedding: torch.Tensor) -> float:
        """Score = V3_batch + V2_delta + V1_session.

        This returns V2_delta — the real-time correction from streaming
        events that the batch model hasn't incorporated yet.
        """
        item_emb = self.embedding_table.get(item_id)
        if item_emb is None:
            return 0.0

        with torch.no_grad():
            user_emb = user_embedding[:self.config.embed_dim]
            delta = self.delta_model(
                item_emb.unsqueeze(0), user_emb.unsqueeze(0),
            ).squeeze().item()

        return delta

    # ── Checkpointing (for Triton model updates) ────────────

    def _save_checkpoint(self) -> None:
        """Save delta model + Cuckoo table state for Triton to load.

        Triton Inference Server polls this directory and loads the
        latest model version automatically.
        """
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save delta model (for Triton)
        model_path = ckpt_dir / "delta_model_latest.pt"
        torch.save({
            "model_state_dict": self.delta_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_events": self._total_events,
            "cuckoo_size": self.embedding_table.size,
            "timestamp": time.time(),
        }, model_path)

        # Save embedding snapshot (for recovery)
        emb_path = ckpt_dir / "embeddings_latest.pt"
        torch.save(self.embedding_table.export_all(), emb_path)

        logger.info(
            "Checkpoint saved — model: %s, embeddings: %d items, total events: %d",
            model_path, self.embedding_table.size, self._total_events,
        )

    def load_checkpoint(self, ckpt_dir: str | None = None) -> bool:
        """Load a previous checkpoint for recovery."""
        path = Path(ckpt_dir or self.config.checkpoint_dir)
        model_path = path / "delta_model_latest.pt"
        emb_path = path / "embeddings_latest.pt"

        if not model_path.exists():
            logger.warning("No checkpoint found at %s", path)
            return False

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.delta_model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._total_events = state.get("total_events", 0)

        if emb_path.exists():
            emb_data = torch.load(emb_path, map_location="cpu", weights_only=True)
            for item_id, emb_list in emb_data.items():
                self.embedding_table.put(item_id, torch.tensor(emb_list))

        logger.info(
            "Checkpoint loaded — %d items, %d total events",
            self.embedding_table.size, self._total_events,
        )
        return True

    # ── Kafka Consumer Loop ──────────────────────────────────

    async def run(self) -> None:
        """Main loop: Kafka → train → Redis → Triton.

        This is the entry point for production. In practice, run this
        inside a Ray Train worker for distributed scaling:

            ray.init()
            trainer = ray.train.torch.TorchTrainer(
                train_func,
                scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
            )
        """
        cfg = self.config

        # Connect to Redis
        await self.feature_store.connect()

        # Load previous checkpoint if available
        self.load_checkpoint()

        try:
            from aiokafka import AIOKafkaConsumer

            consumer = AIOKafkaConsumer(
                *cfg.kafka_topics,
                bootstrap_servers=cfg.redpanda_brokers,
                group_id=cfg.kafka_group_id,
                auto_offset_reset="latest",
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                max_poll_records=cfg.micro_batch_size,
            )
            await consumer.start()
            self._running = True
            logger.info(
                "Monolith streaming trainer STARTED — topics=%s, group=%s",
                cfg.kafka_topics, cfg.kafka_group_id,
            )

            try:
                async for msg in consumer:
                    if not self._running:
                        break
                    try:
                        self.process_event(msg.value)
                    except Exception as e:
                        logger.warning("Event processing error: %s", e)
            finally:
                await consumer.stop()
                # Final sync + checkpoint
                await self._sync_to_redis()
                self._save_checkpoint()
                logger.info("Monolith trainer stopped. Total events: %d", self._total_events)

        except ImportError:
            logger.error(
                "aiokafka not installed. Install with: pip install aiokafka\n"
                "For local dev, call process_event() directly."
            )

    def stop(self) -> None:
        """Signal the trainer to stop after current micro-batch."""
        self._running = False
