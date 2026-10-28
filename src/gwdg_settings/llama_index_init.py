"""Set the global LlamaIndex LLM and embedding model.

Providers: "gwdg" (rate-limited), "openrouter" (pinned hosting provider), anything else -> OpenAI.
:author: Jonathan Decker https://gitlab-ce.gwdg.de/hpc-team-public/chat-ai-llamaindex-examples
"""

import os
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .constants import (
    LLM_MODEL_NAME,
    LLM_API_ENDPOINT_KEY,
    EMBEDDING_MODEL_NAME,
    GWDG_API_KEY
)

from llama_index.core import Settings

from .rate_limited import RateLimitedOpenAILike, RateLimitedVLLMEmbedding
from src.utils.llm_api.client import OPENROUTER_BASE, openrouter_routing
from src.utils.llm_api.rate_limit import ensure_default_limiter


def init(
        api_key: str = GWDG_API_KEY,
        end_point: str = LLM_API_ENDPOINT_KEY,
        llm_model_name: str = LLM_MODEL_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        temperature: float = 0.0,
        embedding_provider: str = "gwdg",
        llm_provider: str = "gwdg"
) -> Settings:
    _llm_timeout = float(os.getenv("GWDG_LLM_TIMEOUT", "90"))
    _llm_max_retries = int(os.getenv("GWDG_LLM_MAX_RETRIES", "2"))

    _embed_batch_size = int(os.getenv("GWDG_EMBED_BATCH_SIZE", "256"))

    if llm_provider == "gwdg" or embedding_provider == "gwdg":
        ensure_default_limiter()

    if llm_provider == "gwdg":
        llm = RateLimitedOpenAILike(
            model=llm_model_name,
            is_chat_model=True,
            temperature=temperature,
            api_key=api_key,
            api_base=end_point,
            timeout=_llm_timeout,
            max_retries=_llm_max_retries,
            default_headers={"inference-service": llm_model_name}
        )
    elif llm_provider == "openrouter":
        from src.utils.llm_api.timing_log import timing_only_http_client
        provider = openrouter_routing(llm_model_name)
        llm = OpenAILike(
            model=llm_model_name,
            is_chat_model=True,
            context_window=131072,
            temperature=temperature,
            api_key=(os.environ.get("OPENROUTER_API_KEY") or "").strip(),
            api_base=OPENROUTER_BASE,
            timeout=float(os.getenv("OPENROUTER_LLM_TIMEOUT", "600")),
            max_retries=6,
            additional_kwargs={"extra_body": {"provider": provider}} if provider else {},
            http_client=timing_only_http_client(),
        )
    else:
        llm = OpenAI(
            model=llm_model_name,
            temperature=temperature,
            api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=_llm_timeout,
            max_retries=_llm_max_retries,
        )

    if embedding_provider == "gwdg":
        embed_model = RateLimitedVLLMEmbedding(
            model=embedding_model_name,
            api_key=api_key,
            api_base=end_point,
            embed_batch_size=_embed_batch_size,
        )
    else:
        from src.utils.llm_api.timing_log import timing_only_http_client
        embed_model = OpenAIEmbedding(
            model=embedding_model_name,
            api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=timing_only_http_client(),
        )

    Settings.llm = llm
    Settings.embed_model = embed_model
    return Settings
