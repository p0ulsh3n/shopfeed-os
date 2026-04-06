"""SGLang Serving Configuration for ShopFeed LLM (Llama 4 Scout).
================================================================
Item 2 — 2026 serving upgrade.

Why SGLang over vLLM (verified March 2026):
  - RadixAttention: automatic KV-cache prefix reuse across requests.
    Massive win for ShopFeed because many LLM calls share the same
    system prompt (same product enrichment / moderation context).
  - ~29% higher throughput than vLLM on H100 for multi-turn workloads.
  - Structured output (JSON mode) natively supported — matches all
    llm_enrichment.py callers that expect JSON responses.
  - Better for RAG/agentic workflows (LLM-as-RS next-item prediction).
  Reference: premai.io, localaimaster.com benchmarks (March 2026).

When NOT to choose SGLang:
  - You need the widest Day-1 model compatibility → vLLM
  - Simple single-turn batch completion → vLLM or LMDeploy
  - Absolute max throughput on a fixed model, no update needed → TensorRT-LLM

ShopFeed usage: multi-turn enrichment + RAG + LLM-as-RS → SGLang wins.

Usage:
    # Start the server
    python -m ml.serving.sglang_config start --model meta-llama/Llama-4-Scout-17B-16E

    # Or use the client
    from ml.serving.sglang_config import SGLangClient
    client = SGLangClient()
    result = await client.generate_json(prompt, system)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# SGLang Server Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class SGLangConfig:
    """Production SGLang server configuration for Llama 4 Scout.

    Optimised for ShopFeed workloads (2026):
      - JSON structured output for enrichment tasks
      - Multi-turn support for conversational search / LLM-as-RS
      - RadixAttention prefix caching (auto — huge win when system prompts repeat)

    See: https://sgl-project.github.io/start/install.html
    """
    # Model
    model: str = "meta-llama/Llama-4-Scout-17B-16E"
    tokenizer: str = ""           # defaults to model
    revision: str = ""            # model revision / branch

    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"       # bf16 on H100; use "float16" on A100
    tensor_parallel_size: int = 1  # Set to 4 for multi-GPU (4× H100)
    gpu_memory_utilization: float = 0.90

    # Quantization (use for A100 40GB to fit the model)
    # Options: None, "fp8", "int4", "awq", "gptq"
    quantization: str | None = None

    # Server
    host: str = "127.0.0.1"
    port: int = 30000
    max_running_requests: int = 256
    max_total_tokens: int = 131072  # 128K context (reduce for more throughput)

    # RadixAttention (the performance difference vs vLLM)
    # This is ON by default in SGLang — no config needed.
    # It automatically caches KV for repeated prefixes (system prompts, etc.)

    # Logging
    log_level: str = "info"

    # Additional sglang args as list (passed directly to CLI)
    extra_args: list[str] = field(default_factory=list)

    def to_cli_args(self) -> list[str]:
        """Convert config to sglang.launch_server() CLI arguments."""
        args = [
            "--model-path", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-running-requests", str(self.max_running_requests),
            "--max-total-tokens", str(self.max_total_tokens),
            "--log-level", self.log_level,
        ]
        if self.tokenizer:
            args += ["--tokenizer-path", self.tokenizer]
        if self.quantization:
            args += ["--quantization", self.quantization]
        if self.revision:
            args += ["--revision", self.revision]
        args += self.extra_args
        return args

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# Default production config for ShopFeed
DEFAULT_SGLANG_CONFIG = SGLangConfig(
    model=os.environ.get("SGLANG_MODEL", "meta-llama/Llama-4-Scout-17B-16E"),
    port=int(os.environ.get("SGLANG_PORT", "30000")),
    tensor_parallel_size=int(os.environ.get("SGLANG_TP", "1")),
    gpu_memory_utilization=float(os.environ.get("SGLANG_GPU_MEM", "0.90")),
)


# ──────────────────────────────────────────────────────────────
# SGLang Client (async, for use in llm_enrichment.py & LLM-as-RS)
# ──────────────────────────────────────────────────────────────

class SGLangClient:
    """Async HTTP client for the SGLang inference server.

    Provides the same interface as the existing LLM router so that
    llm_enrichment.py functions work unchanged.

    RadixAttention benefit: when the same system prompt is used across
    multiple requests (e.g., all photo-scoring calls use the same
    "product photography assessor" system prompt), SGLang automatically
    caches and reuses the KV, slashing TTFT from ~500ms to ~50ms.
    """

    def __init__(self, config: SGLangConfig | None = None):
        self.config = config or DEFAULT_SGLANG_CONFIG
        self._session = None

    async def _get_session(self):
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                raise ImportError("aiohttp required: pip install aiohttp")
        return self._session

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion via SGLang OpenAI-compatible API."""
        session = await self._get_session()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        try:
            async with session.post(
                f"{self.config.base_url}/v1/chat/completions",
                json=payload,
                timeout=30,
            ) as resp:
                if resp.status != 200:
                    logger.warning("SGLang returned %d", resp.status)
                    return ""
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("SGLang generate error: %s", e)
            return ""

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.05,  # Low temp for JSON reliability
    ) -> dict | list | None:
        """Generate structured JSON output.

        SGLang supports constrained decoding (JSON mode) natively,
        which avoids the retry overhead that vLLM requires for bad JSON.
        """
        session = await self._get_session()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},  # SGLang JSON mode
        }

        try:
            async with session.post(
                f"{self.config.base_url}/v1/chat/completions",
                json=payload,
                timeout=30,
            ) as resp:
                if resp.status != 200:
                    logger.warning("SGLang JSON generate returned %d", resp.status)
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("SGLang returned invalid JSON: %s", content[:200])
                    return None
        except Exception as e:
            logger.warning("SGLang generate_json error: %s", e)
            return None

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None


# ──────────────────────────────────────────────────────────────
# Server Launcher
# ──────────────────────────────────────────────────────────────

def launch_server(config: SGLangConfig | None = None) -> subprocess.Popen:
    """Launch SGLang inference server as a subprocess.

    Usage:
        config = SGLangConfig(model="meta-llama/Llama-4-Scout-17B-16E", port=30000)
        proc = launch_server(config)
        # ... wait for server ready ...
        proc.terminate()
    """
    cfg = config or DEFAULT_SGLANG_CONFIG
    cmd = [sys.executable, "-m", "sglang.launch_server"] + cfg.to_cli_args()

    logger.info("Launching SGLang server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


async def wait_for_server(
    config: SGLangConfig | None = None,
    timeout_s: float = 120.0,
    poll_interval: float = 2.0,
) -> bool:
    """Wait until the SGLang server is ready to accept requests.

    Returns True if server is ready, False if timeout exceeded.
    """
    cfg = config or DEFAULT_SGLANG_CONFIG
    client = SGLangClient(cfg)
    deadline = asyncio.get_event_loop().time() + timeout_s

    while asyncio.get_event_loop().time() < deadline:
        try:
            session = await client._get_session()
            async with session.get(f"{cfg.base_url}/v1/models", timeout=3) as resp:
                if resp.status == 200:
                    logger.info("SGLang server ready at %s", cfg.base_url)
                    await client.close()
                    return True
        except Exception:
            pass
        await asyncio.sleep(poll_interval)

    logger.warning("SGLang server did not start within %.0fs", timeout_s)
    await client.close()
    return False


# ──────────────────────────────────────────────────────────────
# Docker / docker-compose snippet (printed as documentation)
# ──────────────────────────────────────────────────────────────

DOCKER_COMPOSE_SNIPPET = """
# Add to docker-compose.yml for SGLang LLM serving:
#
# sglang-llm:
#   image: lmsysorg/sglang:latest
#   runtime: nvidia
#   environment:
#     - NVIDIA_VISIBLE_DEVICES=0
#   volumes:
#     - ~/.cache/huggingface:/root/.cache/huggingface
#   ports:
#     - "30000:30000"
#   command: >
#     python -m sglang.launch_server
#     --model-path meta-llama/Llama-4-Scout-17B-16E
#     --host 0.0.0.0
#     --port 30000
#     --dtype bfloat16
#     --gpu-memory-utilization 0.90
#     --max-running-requests 256
#   healthcheck:
#     test: ["CMD", "curl", "-f", "http://localhost:30000/v1/models"]
#     interval: 30s
#     timeout: 10s
#     retries: 5
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ShopFeed SGLang server launcher")
    parser.add_argument("action", choices=["start", "config"], help="Action to perform")
    parser.add_argument("--model", default=DEFAULT_SGLANG_CONFIG.model)
    parser.add_argument("--port", type=int, default=DEFAULT_SGLANG_CONFIG.port)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--quant", default=None, help="Quantization (fp8|int4|awq)")
    args = parser.parse_args()

    if args.action == "config":
        print(DOCKER_COMPOSE_SNIPPET)
    elif args.action == "start":
        cfg = SGLangConfig(
            model=args.model,
            port=args.port,
            tensor_parallel_size=args.tp,
            quantization=args.quant,
        )
        print(f"Starting SGLang server: {cfg.base_url}")
        print(f"CLI: {' '.join(cfg.to_cli_args())}")
        proc = launch_server(cfg)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
