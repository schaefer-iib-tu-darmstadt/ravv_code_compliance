"""GWDG LLM and embedding clients gated on the process-wide limiter at the HTTP layer.

Every HTTP attempt, including SDK and tenacity retries, consumes one limiter slot.
"""
from __future__ import annotations

from llama_index.llms.openai_like import OpenAILike

from src.gwdg_settings.vllm_embeddings import VLLMEmbedding
from src.utils.llm_api.rate_limit import (
    gated_async_http_client,
    gated_http_client,
)


class RateLimitedOpenAILike(OpenAILike):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("http_client", gated_http_client())
        kwargs.setdefault("async_http_client", gated_async_http_client())
        super().__init__(*args, **kwargs)


class RateLimitedVLLMEmbedding(VLLMEmbedding):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("http_client", gated_http_client())
        kwargs.setdefault("async_http_client", gated_async_http_client())
        super().__init__(*args, **kwargs)
