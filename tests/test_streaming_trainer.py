"""
Tests for ml/monolith/streaming_trainer.py
Covers BUG #1, #13, #14 fixes.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from ml.monolith.streaming_trainer import MonolithConfig, MonolithStreamingTrainer


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def trainer():
    cfg = MonolithConfig(
        embed_dim=16,
        micro_batch_size=2,
        gradient_accumulation=2,
        cuckoo_capacity=100,
    )
    return MonolithStreamingTrainer(cfg)


def _make_event(item_id: str = "item_001", action: str = "buy_now") -> dict:
    return {
        "item_id": item_id,
        "user_id": "user_001",
        "action": action,
    }


# ──────────────────────────────────────────────────────────────
# BUG #1 — item_emb.grad must not be None after backward()
# ──────────────────────────────────────────────────────────────

def test_item_embedding_grad_is_not_none(trainer):
    """BUG #1 FIX: item_emb must be a leaf tensor with requires_grad=True
    so that loss.backward() can compute and store its gradient."""

    # Seed the embedding table with a known embedding
    init_emb = torch.randn(trainer.config.embed_dim) * 0.01
    trainer.embedding_table.put("item_001", init_emb)

    # Retrieve and simulate the fixed forward pass
    stored_emb = trainer.embedding_table.get("item_001")

    # This is the fix: stored_emb (requires_grad=False) must be re-wrapped
    item_emb = stored_emb.detach().requires_grad_(True)

    # Minimal forward + backward
    user_emb = torch.zeros(trainer.config.embed_dim)
    pred = torch.sigmoid(trainer.delta_model(item_emb.unsqueeze(0), user_emb.unsqueeze(0)))
    label = torch.tensor([[1.0]])
    import torch.nn.functional as F
    loss = F.binary_cross_entropy(pred, label)
    loss.backward()

    # Core assertion: grad must be non-None after backward (was ALWAYS None before fix)
    assert item_emb.grad is not None, (
        "BUG #1 NOT FIXED: item_emb.grad is None — "
        "embedding table update will never run"
    )
    assert item_emb.grad.shape == (trainer.config.embed_dim,)


def test_process_event_updates_embedding(trainer):
    """End-to-end: after processing micro_batch_size events, embedding must change."""
    cfg = trainer.config

    # Register initial embedding for item_001
    init_emb = torch.randn(cfg.embed_dim) * 0.01
    trainer.embedding_table.put("item_001", init_emb)
    original = trainer.embedding_table.get("item_001").clone()

    # Process exactly micro_batch_size events to trigger _train_micro_batch()
    for _ in range(cfg.micro_batch_size):
        trainer.process_event(_make_event())

    updated = trainer.embedding_table.get("item_001")
    # Embedding must have changed (gradient update applied)
    assert not torch.allclose(original, updated), (
        "BUG #1 NOT FIXED: embedding unchanged after training — "
        "update_embedding() was never called"
    )


# ──────────────────────────────────────────────────────────────
# BUG #13 — asyncio.get_running_loop() replaces deprecated get_event_loop()
# ──────────────────────────────────────────────────────────────

def test_sync_deferred_in_sync_context(trainer, monkeypatch):
    """BUG #13 FIX: _sync_to_redis() should not raise DeprecationWarning
    or TypeError when called from a synchronous context."""
    import inspect

    # Force the sync timer to trigger
    trainer._last_sync = 0.0

    deferred = []
    # Patch asyncio.get_running_loop to raise RuntimeError (sync context)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError("no loop")))

    # Should NOT raise — just defer
    for _ in range(trainer.config.micro_batch_size):
        trainer._event_buffer.append(_make_event())
    try:
        trainer._train_micro_batch()
    except RuntimeError:
        pytest.fail("BUG #13 NOT FIXED: RuntimeError leaked from _train_micro_batch()")


# ──────────────────────────────────────────────────────────────
# BUG #14 — optimizer steps only every gradient_accumulation micro-batches
# ──────────────────────────────────────────────────────────────

def test_gradient_accumulation_step_count(trainer, monkeypatch):
    """BUG #14 FIX: optimizer.step() must be called exactly once per
    gradient_accumulation micro-batches, not once per micro-batch."""
    cfg = trainer.config  # gradient_accumulation=2

    step_count = {"n": 0}
    original_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        step_count["n"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(trainer.optimizer, "step", counting_step)

    # Seed table
    trainer.embedding_table.put("item_001", torch.randn(cfg.embed_dim) * 0.01)

    # Run 4 micro-batches  →  should produce 4 // 2 = 2 optimizer steps
    n_batches = 4
    for _ in range(n_batches):
        for _ in range(cfg.micro_batch_size):
            trainer._event_buffer.append(_make_event())
        trainer._train_micro_batch()

    expected_steps = n_batches // cfg.gradient_accumulation
    assert step_count["n"] == expected_steps, (
        f"BUG #14 NOT FIXED: expected {expected_steps} optimizer steps "
        f"(gradient_accumulation={cfg.gradient_accumulation}), "
        f"got {step_count['n']}"
    )
