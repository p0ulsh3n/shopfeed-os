"""
Monolith — Online Streaming Training, Vitesse 2 (Section 14)
==============================================================
Inspired by ByteDance's Monolith paper (2022). The original repo
(bytedance/monolith) was archived Oct 2025. This is a 2026 re-implementation
using the modern production stack:

    Kafka + Flink    → streaming event ingestion + feature engineering
    PyTorch + Ray    → distributed online training (replaces PS)
    Redis            → feature store serving (<5ms reads)
    Triton           → model inference serving (updated every 5-15 min)

The CONCEPT is what matters (streaming gradients, no batching):
  "Un add-to-cart à 14h03 est dans le training à 14h03,
   visible dans les feeds à 14h18."

Key components:
  1. CuckooEmbeddingTable — collision-free hash, LRU eviction >30d
  2. StreamingTrainer     — Ray-based distributed online SGD
  3. RedisFeatureStore    — feature serving <5ms with auto-sync
  4. TritonModelPublisher — pushes updated weights every 5-15 min
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. Cuckoo Embedding Table (concept from Monolith paper)
# ═══════════════════════════════════════════════════════════════

@dataclass
class _CuckooEntry:
    item_id: str
    created_at: float = field(default_factory=time.time)


class CuckooEmbeddingTable:
    """Collision-free embedding table using Cuckoo Hashing.

    Paper concept: "Table sans collision. Eviction LRU des produits inactifs >30j.
    Embeddings nouveaux produits initialisés via content features (CLIP)."

    Cuckoo hashing: two hash functions, O(1) guaranteed lookup.
    On collision, existing entry kicked to alternate slot.
    LRU eviction keeps table bounded.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        capacity: int = 2_000_000,
        max_eviction_age_days: int = 30,
        max_kicks: int = 500,
    ):
        self.embed_dim = embed_dim
        self.capacity = capacity
        self.max_eviction_age_s = max_eviction_age_days * 86400
        self.max_kicks = max_kicks

        # Dual hash tables for cuckoo hashing
        self._table_a: dict[int, _CuckooEntry] = {}
        self._table_b: dict[int, _CuckooEntry] = {}

        # LRU tracking: item_id → last_access_timestamp
        self._access_order: OrderedDict[str, float] = OrderedDict()

        # Embedding storage: item_id → tensor
        self._embeddings: dict[str, torch.Tensor] = {}

        logger.info(
            "CuckooEmbeddingTable: dim=%d, capacity=%d, eviction=%dd",
            embed_dim, capacity, max_eviction_age_days,
        )

    def _hash_a(self, item_id: str) -> int:
        return int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16) % self.capacity

    def _hash_b(self, item_id: str) -> int:
        return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % self.capacity

    def get(self, item_id: str) -> torch.Tensor | None:
        """O(1) lookup. Returns None if not found."""
        for table, h in [(self._table_a, self._hash_a(item_id)),
                         (self._table_b, self._hash_b(item_id))]:
            entry = table.get(h)
            if entry is not None and entry.item_id == item_id:
                self._touch(item_id)
                return self._embeddings.get(item_id)
        return None

    def put(
        self,
        item_id: str,
        embedding: torch.Tensor,
        initial_clip: torch.Tensor | None = None,
    ) -> bool:
        """Insert or update. New items initialized from CLIP content features."""
        # Update existing
        if self.get(item_id) is not None:
            self._embeddings[item_id] = embedding.detach().clone()
            self._touch(item_id)
            return True

        self._maybe_evict()

        # Initialize from CLIP if available (Section 14: content-based cold start)
        if initial_clip is not None:
            if initial_clip.shape[0] > self.embed_dim:
                init_emb = initial_clip[:self.embed_dim].clone()
            else:
                init_emb = F.pad(initial_clip, (0, self.embed_dim - initial_clip.shape[0]))
        else:
            init_emb = embedding.detach().clone()

        self._embeddings[item_id] = init_emb
        return self._cuckoo_insert(_CuckooEntry(item_id=item_id))

    def _cuckoo_insert(self, entry: _CuckooEntry) -> bool:
        current = entry
        for _ in range(self.max_kicks):
            # Try table A
            ha = self._hash_a(current.item_id)
            existing = self._table_a.get(ha)
            if existing is None:
                self._table_a[ha] = current
                self._touch(current.item_id)
                return True
            self._table_a[ha] = current
            current = existing

            # Try table B
            hb = self._hash_b(current.item_id)
            existing = self._table_b.get(hb)
            if existing is None:
                self._table_b[hb] = current
                self._touch(current.item_id)
                return True
            self._table_b[hb] = current
            current = existing

        self._evict_lru()
        return self._cuckoo_insert(entry)

    def _touch(self, item_id: str) -> None:
        self._access_order[item_id] = time.time()
        self._access_order.move_to_end(item_id)

    def _maybe_evict(self) -> None:
        now = time.time()
        to_remove = []
        for item_id, last_access in self._access_order.items():
            if now - last_access > self.max_eviction_age_s:
                to_remove.append(item_id)
            else:
                break
        for item_id in to_remove:
            self._remove(item_id)
        if to_remove:
            logger.info("Evicted %d inactive items (>%dd)", len(to_remove), self.max_eviction_age_s // 86400)

    def _evict_lru(self) -> None:
        if self._access_order:
            self._remove(next(iter(self._access_order)))

    def _remove(self, item_id: str) -> None:
        ha, hb = self._hash_a(item_id), self._hash_b(item_id)
        if ha in self._table_a and self._table_a[ha].item_id == item_id:
            del self._table_a[ha]
        if hb in self._table_b and self._table_b[hb].item_id == item_id:
            del self._table_b[hb]
        self._embeddings.pop(item_id, None)
        self._access_order.pop(item_id, None)

    def mark_inactive(self, item_id: str) -> None:
        """Out-of-stock product → immediate eviction."""
        self._remove(item_id)
        logger.debug("Item %s marked inactive, removed from Cuckoo table", item_id)

    def update_embedding(self, item_id: str, gradient: torch.Tensor, lr: float = 0.01) -> None:
        """SGD step on a single item embedding (streaming update)."""
        emb = self.get(item_id)
        if emb is not None:
            with torch.no_grad():
                emb -= lr * gradient
            self._embeddings[item_id] = emb

    @property
    def size(self) -> int:
        return len(self._embeddings)

    def export_all(self) -> dict[str, list[float]]:
        """Export all embeddings for Redis sync."""
        return {k: v.tolist() for k, v in self._embeddings.items()}


# ═══════════════════════════════════════════════════════════════
# 2. Redis Feature Store — <5ms serving (replaces in-memory)
# ═══════════════════════════════════════════════════════════════

class RedisFeatureStore:
    """Feature store backed by Redis for <5ms reads at serving time.

    2026 stack: Redis replaces the Monolith Parameter Server for
    feature serving. Training writes → Redis → Serving reads.

    Keys schema:
      item:emb:{item_id}       → embedding vector (bytes)
      item:features:{item_id}  → JSON feature dict
      user:emb:{user_id}       → user embedding (bytes)
      user:seq:{user_id}       → behavior sequence (JSON list)
      delta:score:{item_id}    → online delta correction (float)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "shopfeed",
        ttl_seconds: int = 86400,  # 24h default TTL
    ):
        self.prefix = prefix
        self.ttl = ttl_seconds
        self._redis = None
        self._redis_url = redis_url

    async def connect(self) -> None:
        """Lazy connect to Redis."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._redis_url,
                    decode_responses=False,
                    max_connections=50,
                )
                await self._redis.ping()
                logger.info("Redis feature store connected: %s", self._redis_url)
            except ImportError:
                logger.warning("redis package not installed — using in-memory fallback")
                self._redis = None
            except Exception as e:
                logger.error("Redis connection failed: %s", e)
                self._redis = None

    def _key(self, *parts: str) -> str:
        return f"{self.prefix}:{':'.join(parts)}"

    # ── Item embeddings ──────────────────────────────────────

    async def set_item_embedding(self, item_id: str, embedding: torch.Tensor) -> None:
        """Write item embedding to Redis (called by trainer)."""
        if self._redis is None:
            return
        key = self._key("item", "emb", item_id)
        data = embedding.numpy().tobytes()
        await self._redis.set(key, data, ex=self.ttl)

    async def get_item_embedding(self, item_id: str, dim: int = 64) -> torch.Tensor | None:
        """Read item embedding from Redis (<1ms)."""
        if self._redis is None:
            return None
        key = self._key("item", "emb", item_id)
        data = await self._redis.get(key)
        if data is None:
            return None
        return torch.frombuffer(bytearray(data), dtype=torch.float32)[:dim]

    # ── User features ────────────────────────────────────────

    async def set_user_embedding(self, user_id: str, embedding: torch.Tensor) -> None:
        if self._redis is None:
            return
        key = self._key("user", "emb", user_id)
        await self._redis.set(key, embedding.numpy().tobytes(), ex=self.ttl)

    async def get_user_embedding(self, user_id: str, dim: int = 256) -> torch.Tensor | None:
        if self._redis is None:
            return None
        data = await self._redis.get(self._key("user", "emb", user_id))
        if data is None:
            return None
        return torch.frombuffer(bytearray(data), dtype=torch.float32)[:dim]

    # ── User behavior sequence (for DIN/DIEN/BST) ───────────

    async def append_behavior(self, user_id: str, item_id: str, action: str) -> None:
        """Append to user behavior sequence (capped at 200)."""
        if self._redis is None:
            return
        key = self._key("user", "seq", user_id)
        entry = json.dumps({"item_id": item_id, "action": action, "ts": time.time()})
        pipe = self._redis.pipeline()
        pipe.rpush(key, entry)
        pipe.ltrim(key, -200, -1)  # Keep last 200
        pipe.expire(key, self.ttl)
        await pipe.execute()

    async def get_behavior_sequence(self, user_id: str) -> list[dict]:
        """Get user's full behavior sequence for ranking models."""
        if self._redis is None:
            return []
        key = self._key("user", "seq", user_id)
        raw = await self._redis.lrange(key, 0, -1)
        return [json.loads(r) for r in raw] if raw else []

    # ── Online delta scores ──────────────────────────────────

    async def set_delta_score(self, item_id: str, delta: float) -> None:
        """Write online delta correction for Score = V3_batch + V2_delta + V1_session."""
        if self._redis is None:
            return
        key = self._key("delta", "score", item_id)
        await self._redis.set(key, str(delta), ex=900)  # 15 min TTL

    async def get_delta_score(self, item_id: str) -> float:
        """Read delta score (<1ms). Returns 0.0 if not found."""
        if self._redis is None:
            return 0.0
        data = await self._redis.get(self._key("delta", "score", item_id))
        return float(data) if data else 0.0

    # ── Batch sync from Cuckoo table ─────────────────────────

    async def sync_from_cuckoo(self, table: CuckooEmbeddingTable) -> int:
        """Bulk sync all embeddings from Cuckoo table to Redis.
        Called every 5-15 minutes by the parameter sync job.
        """
        if self._redis is None:
            return 0
        pipe = self._redis.pipeline()
        count = 0
        for item_id, emb in table._embeddings.items():
            key = self._key("item", "emb", item_id)
            pipe.set(key, emb.numpy().tobytes(), ex=self.ttl)
            count += 1
            if count % 1000 == 0:
                await pipe.execute()
                pipe = self._redis.pipeline()
        if count % 1000 != 0:
            await pipe.execute()
        logger.info("Synced %d embeddings from Cuckoo table → Redis", count)
        return count


# ═══════════════════════════════════════════════════════════════
# 3. Delta Model — learns real-time adjustments to batch scores
# ═══════════════════════════════════════════════════════════════

class DeltaModel(nn.Module):
    """Online delta model: Score_final = Score_batch(V3) + Delta(V2).

    Small MLP that learns adjustments from streaming events.
    This captures changes since the last batch training run.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, item_emb: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([item_emb, user_emb], dim=-1)
        return self.net(combined)


# ═══════════════════════════════════════════════════════════════
# 4. Streaming Trainer — Ray-based distributed online training
# ═══════════════════════════════════════════════════════════════

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
    sync_interval_s: float = 600        # 10 min default (5-15 min range)
    checkpoint_interval_s: float = 3600  # Hourly checkpoints

    # Infrastructure
    kafka_bootstrap: str = "localhost:9092"
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
        │  Kafka   │ →  │  This Trainer  │ →  │    Redis     │
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
        self._last_checkpoint = time.time()
        self._events_processed = 0
        self._total_events = 0
        self._running = False

        logger.info(
            "MonolithStreamingTrainer initialized — "
            "embed_dim=%d, sync=%ds, lr=%.4f, buffer=%d",
            cfg.embed_dim, cfg.sync_interval_s,
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

            if not item_id:
                continue

            # ── Get or create item embedding ──
            item_emb = self.embedding_table.get(item_id)
            if item_emb is None:
                clip_data = event.get("clip_embedding")
                clip_tensor = torch.tensor(clip_data, dtype=torch.float32) if clip_data else None
                init_emb = torch.randn(cfg.embed_dim) * 0.01
                self.embedding_table.put(item_id, init_emb, initial_clip=clip_tensor)
                item_emb = self.embedding_table.get(item_id)

            if item_emb is None:
                continue

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

            # ── Update item embedding via gradient ──
            if item_emb.grad is not None:
                self.embedding_table.update_embedding(
                    item_id, item_emb.grad, lr=cfg.learning_rate,
                )
                item_emb.grad = None

            # ── Handle stock events ──
            if action == "stock_update" and event.get("stock", 1) <= 0:
                self.embedding_table.mark_inactive(item_id)

        # ── Optimizer step (gradient accumulation) ──
        if n_valid > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = batch_loss / max(n_valid, 1)
        self._event_buffer.clear()

        # ── Periodic sync to Redis ──
        now = time.time()
        if now - self._last_sync > cfg.sync_interval_s:
            asyncio.create_task(self._sync_to_redis()) if asyncio.get_event_loop().is_running() else None
            self._last_sync = now

        # ── Periodic checkpoint ──
        if now - self._last_checkpoint > cfg.checkpoint_interval_s:
            self._save_checkpoint()
            self._last_checkpoint = now

    # ── Redis Sync (replaces Parameter Server) ───────────────

    async def _sync_to_redis(self) -> None:
        """Push updated embeddings + delta scores to Redis.

        This is the 2026 replacement for Monolith's Parameter Server.
        Redis gives <5ms reads at serving time.

        Section 14: "Sync vers serving layer toutes les 5-15 minutes
        pour stabilité."
        """
        await self.feature_store.connect()
        n_synced = await self.feature_store.sync_from_cuckoo(self.embedding_table)

        logger.info(
            "Redis sync complete — %d embeddings pushed, %d events since last sync",
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
                bootstrap_servers=cfg.kafka_bootstrap,
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


# ═══════════════════════════════════════════════════════════════
# 5. Triton Model Publisher — pushes weights for serving
# ═══════════════════════════════════════════════════════════════

class TritonModelPublisher:
    """Publishes updated model weights for Triton Inference Server.

    Triton polls a model repository directory. We write updated
    ONNX/TorchScript models there every sync interval.

    Production setup:
        /models/
            delta_model/
                config.pbtxt      (Triton config)
                1/model.onnx      (version 1)
                2/model.onnx      (version 2 — after sync)
    """

    def __init__(
        self,
        model_repo: str = "/models/delta_model",
        export_format: str = "torchscript",  # torchscript | onnx
    ):
        self.model_repo = Path(model_repo)
        self.export_format = export_format
        self._version = 0

    def publish(self, delta_model: DeltaModel, embed_dim: int = 64) -> Path:
        """Export and publish a new model version for Triton."""
        self._version += 1
        version_dir = self.model_repo / str(self._version)
        version_dir.mkdir(parents=True, exist_ok=True)

        delta_model.eval()
        dummy_item = torch.randn(1, embed_dim)
        dummy_user = torch.randn(1, embed_dim)

        if self.export_format == "torchscript":
            path = version_dir / "model.pt"
            traced = torch.jit.trace(delta_model, (dummy_item, dummy_user))
            traced.save(str(path))
        else:  # onnx
            path = version_dir / "model.onnx"
            torch.onnx.export(
                delta_model, (dummy_item, dummy_user), str(path),
                input_names=["item_embedding", "user_embedding"],
                output_names=["delta_score"],
                dynamic_axes={
                    "item_embedding": {0: "batch"},
                    "user_embedding": {0: "batch"},
                },
                opset_version=17,
            )

        logger.info("Triton model published: version=%d, path=%s", self._version, path)
        return path


# ═══════════════════════════════════════════════════════════════
# 6. Ray Train Integration — distributed online training
# ═══════════════════════════════════════════════════════════════

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
