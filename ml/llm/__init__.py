"""
ShopFeed LLM Module — Llama 4 Scout Unified Client
=====================================================
UN SEUL modèle pour TOUTES les tâches :

    Llama 4 Scout 17B-16E Instruct (Meta, 2025)
        → 17B params actifs / 109B total (MoE 16 experts)
        → Multimodal natif : texte + image
        → Contexte 10M tokens
        → Tourne sur 1 seul A100 80GB
        → Remplace Qwen2.5-VL-7B + Phi-4 Mini + Maverick

Tâches :
    - Vision    : quality scoring, ad creative, product enrichment, moderation
    - Texte     : ad copy, search conversationnel, descriptions SEO
    - Reasoning : vendor insights, audience analysis, campaign strategy

Deployment (vLLM) :
    vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \\
        --port 8200 \\
        --gpu-memory-utilization 0.85 \\
        --max-model-len 32768 \\
        --kv-cache-dtype fp8

Environment :
    SCOUT_URL=http://localhost:8200
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────

SCOUT_URL = os.environ.get("SCOUT_URL", "http://localhost:8200")
SCOUT_MODEL = os.environ.get(
    "SCOUT_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct"
)


# ── vLLM Client ───────────────────────────────────────────────────

class VLLMClient:
    """Async client for vLLM-served Llama 4 Scout (OpenAI-compatible API).

    Handles retry, timeout, multimodal messages, and JSON extraction.
    Single model does everything: vision, text gen, and reasoning.
    """

    def __init__(
        self,
        base_url: str = SCOUT_URL,
        model: str = SCOUT_MODEL,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError:
                logger.error("httpx required: pip install httpx")
                raise
        return self._client

    async def health(self) -> bool:
        """Check if the vLLM server is reachable."""
        try:
            client = await self._get_client()
            r = await client.get("/health")
            return r.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Text-only chat completion."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._chat(messages, max_tokens, temperature)

    async def generate_vision(
        self,
        prompt: str,
        image_url: str,
        system_prompt: str = "You are a professional visual analysis AI.",
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> str:
        """Multimodal chat completion (image + text).

        Llama 4 Scout handles images natively — no separate vision model needed.
        """
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return await self._chat(messages, max_tokens, temperature)

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. Always respond in valid JSON.",
        max_tokens: int = 500,
        temperature: float = 0.3,
        image_url: str | None = None,
    ) -> dict | list:
        """Generate and parse JSON response."""
        if image_url:
            raw = await self.generate_vision(prompt, image_url, system_prompt, max_tokens, temperature)
        else:
            raw = await self.generate(prompt, system_prompt, max_tokens, temperature)
        return self._extract_json(raw)

    async def _chat(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Execute chat completion with retries."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(self.max_retries):
            try:
                client = await self._get_client()
                r = await client.post("/v1/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return ""
            except Exception as e:
                wait = min(2 ** attempt, 8)
                logger.warning(
                    "Scout attempt %d/%d failed: %s (retry in %ds)",
                    attempt + 1, self.max_retries, e, wait,
                )
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(wait)

        logger.error("Scout: all %d attempts failed", self.max_retries)
        return ""

    @staticmethod
    def _extract_json(text: str) -> dict | list:
        """Extract JSON from LLM response (handles ```json blocks)."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            inner = "\n".join(
                l for l in lines[1:] if not l.strip().startswith("```")
            )
            text = inner.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start = text.find(start_char)
                end = text.rfind(end_char)
                if start != -1 and end > start:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        continue
            logger.warning("Failed to extract JSON from: %s", text[:200])
            return {}

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# ── Model Router (unified — all tasks → Scout) ───────────────────

class ModelRouter:
    """Unified model router — ALL tasks go to Llama 4 Scout.

    Scout is multimodal (image + text) with 17B active params (MoE).
    No need for separate vision/text/reasoning models.

    Usage:
        router = get_router()
        result = await router.vision(prompt, image_url)   # Scout
        result = await router.text(prompt)                  # Scout
        result = await router.reason(prompt)                # Scout
    """

    def __init__(self):
        self.scout = VLLMClient(SCOUT_URL, SCOUT_MODEL, timeout=30)

    # ── Vision tasks (product photos, ad creatives, moderation) ──

    async def vision(
        self,
        prompt: str,
        image_url: str,
        system_prompt: str = "You are a professional product photography and visual analysis AI expert.",
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> str:
        """Route vision tasks to Scout (native multimodal)."""
        return await self.scout.generate_vision(
            prompt, image_url, system_prompt, max_tokens, temperature,
        )

    async def vision_json(
        self,
        prompt: str,
        image_url: str,
        system_prompt: str = "You are a professional visual analysis AI. Always respond in valid JSON only.",
        max_tokens: int = 500,
    ) -> dict | list:
        """Route vision+JSON tasks to Scout."""
        return await self.scout.generate_json(
            prompt, system_prompt, max_tokens, temperature=0.2, image_url=image_url,
        )

    # ── Text generation (ad copy, search, descriptions) ──────────

    async def text(
        self,
        prompt: str,
        system_prompt: str = "You are a professional e-commerce copywriter.",
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Route text gen tasks to Scout."""
        return await self.scout.generate(
            prompt, system_prompt, max_tokens, temperature,
        )

    async def text_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. Always respond with valid JSON only.",
        max_tokens: int = 500,
    ) -> dict | list:
        """Route text+JSON tasks to Scout."""
        return await self.scout.generate_json(
            prompt, system_prompt, max_tokens,
        )

    # ── Reasoning (vendor insights, campaign strategy) ───────────

    async def reason(
        self,
        prompt: str,
        system_prompt: str = "You are an expert analyst.",
        max_tokens: int = 1000,
        temperature: float = 0.5,
    ) -> str:
        """Route complex reasoning to Scout (17B MoE = strong reasoning)."""
        return await self.scout.generate(
            prompt, system_prompt, max_tokens, temperature,
        )

    async def close(self):
        await self.scout.close()


# ── Singleton ─────────────────────────────────────────────────────

_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Get the global model router singleton."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
