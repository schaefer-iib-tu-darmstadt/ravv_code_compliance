"""Chat-completion helpers for the direct (non-LlamaIndex) code paths."""
import re
from src.utils.config_helpers import load_config

ai_config = load_config()
reasoning_model_list = ai_config['llm']['reasoning_model_list']

_REASONING_BLOCK = re.compile(r"<(think|thought)>.*?</\1>", re.DOTALL)
_REASONING_UNCLOSED = re.compile(r"<(think|thought)>.*\Z", re.DOTALL)


def strip_reasoning(text: str) -> str:
    stripped = _REASONING_BLOCK.sub("", text)
    stripped = _REASONING_UNCLOSED.sub("", stripped)
    return stripped.strip()


def _build_api_params(client, model: str, messages: list[dict], temperature, top_p,
                      max_tokens, extra_body) -> dict:
    api_params = {
        "model": model,
        "messages": messages,
        "seed": 10,
    }
    if temperature is not None:
        api_params["temperature"] = temperature
    if top_p is not None:
        api_params["top_p"] = top_p
    if max_tokens is not None:
        api_params["max_tokens"] = max_tokens
    routing = getattr(client, "openrouter_routing", None)
    provider = routing(model) if routing else None
    if provider is not None:
        extra_body = {"provider": provider, **(extra_body or {})}
    if extra_body is not None:
        api_params["extra_body"] = extra_body
    return api_params


def llm_chat_completion_message_content(client, model: str, messages: list[dict], temperature: float | None = None, top_p: float | None = None,
        max_tokens: int | None = None, extra_body: dict | None = None) -> str:
    try:
        api_params = _build_api_params(client, model, messages, temperature, top_p, max_tokens, extra_body)

        response = client.chat.completions.create(**api_params)
        response_content = response.choices[0].message.content

        if model in reasoning_model_list:
            response_content = strip_reasoning(response_content or "")
            if not response_content:
                raise ValueError(
                    f"{model} returned only a (likely truncated) reasoning block; "
                    "consider raising max_tokens."
                )

        print("Response received")
        return response_content
    except Exception as e:
        print(f"Error while calling the LLM-API for Chat Completion: {e}")
        raise

