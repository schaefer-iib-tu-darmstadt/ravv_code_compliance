"""OpenAI-compatible chat clients (GWDG rate-limited, OpenAI, OpenRouter) for the non-LlamaIndex paths."""
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.config_helpers import load_config
from src.utils.llm_api.rate_limit import ensure_default_limiter, gated_http_client

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

OPENROUTER_PROVIDER_PINS = {
    "google/gemma-3-27b-it": "novita",
    "openai/gpt-oss-120b": "deepinfra",
}


def openrouter_routing(model: str) -> dict | None:
    slug = OPENROUTER_PROVIDER_PINS.get(model)
    if slug is None:
        return None
    return {"order": [slug], "allow_fallbacks": False}


def create_client(mode: str = "gwdg") -> OpenAI:
    if mode == "gwdg":
        print("Initializing GWDG client...")
        return create_gwdg_client()
    if mode == "openai":
        print("Initializing OpenAI client...")
        return create_openai_client()
    if mode == "openrouter":
        print("Initializing OpenRouter client...")
        return create_openrouter_client()
    raise ValueError(f"Unknown client mode {mode!r} (expected 'gwdg', 'openai' or 'openrouter')")


def create_gwdg_client() -> OpenAI:
    load_dotenv()
    config = load_config()
    api_key = os.environ.get("GWDG_API_KEY")
    base_url = os.environ.get("GWDG_API_BASE") or config['api']['base_url']
    ensure_default_limiter()
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=gated_http_client(),
    )


def create_openai_client() -> OpenAI:
    load_dotenv()
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def create_openrouter_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing from the environment/.env")
    from src.utils.llm_api.timing_log import timing_only_http_client
    client = OpenAI(
        api_key=api_key.strip(),
        base_url=OPENROUTER_BASE,
        max_retries=6,
        timeout=600,
        http_client=timing_only_http_client(),
    )
    client.openrouter_routing = openrouter_routing
    return client
