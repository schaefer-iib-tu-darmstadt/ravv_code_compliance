"""Embedding client for vLLM's OpenAI-compatible embedding API (GWDG).

Adapted under the MIT License from
https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-openai/llama_index/embeddings/openai/base.py
(helpers copied from openai-python ``embeddings_utils``).
:author: Jonathan Decker https://gitlab-ce.gwdg.de/hpc-team-public/chat-ai-llamaindex-examples
"""
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from typing import Any, List, Optional, Dict
import httpx
from openai import OpenAI, AsyncOpenAI
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.openai.utils import create_retry_decorator
from llama_index.core.base.embeddings.base import BaseEmbedding

embedding_retry_decorator = create_retry_decorator(
    max_retries=6,
    random_exponential=True,
    stop_after_delay_seconds=60,
    min_seconds=1,
    max_seconds=60,
)


@embedding_retry_decorator
def get_embedding(client: OpenAI, text: str, engine: str, **kwargs: Any) -> List[float]:
    text = text.replace("\n", " ")

    return (
        client.embeddings.create(input=[text], model=engine, **kwargs).data[0].embedding
    )


@embedding_retry_decorator
async def aget_embedding(
    aclient: AsyncOpenAI, text: str, engine: str, **kwargs: Any
) -> List[float]:
    text = text.replace("\n", " ")

    return (
        (await aclient.embeddings.create(input=[text], model=engine, **kwargs))
        .data[0]
        .embedding
    )


@embedding_retry_decorator
def get_embeddings(
    client: OpenAI, list_of_text: List[str], engine: str, **kwargs: Any
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = client.embeddings.create(input=list_of_text, model=engine, **kwargs).data
    return [d.embedding for d in data]


@embedding_retry_decorator
async def aget_embeddings(
    aclient: AsyncOpenAI,
    list_of_text: List[str],
    engine: str,
    **kwargs: Any,
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = (
        await aclient.embeddings.create(input=list_of_text, model=engine, **kwargs)
    ).data
    return [d.embedding for d in data]


class VLLMEmbedding(BaseEmbedding):
    api_key: str
    api_base: str
    max_retries: int
    reuse_client: bool
    timeout: float

    _client: Optional[OpenAI] = PrivateAttr()
    _aclient: Optional[AsyncOpenAI] = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()
    _async_http_client: Optional[httpx.AsyncClient] = PrivateAttr()

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the ChatAI API."
    )
    default_headers: Optional[Dict[str, str]]

    dimensions: Optional[int] = Field(
        default=None,
        description=(
            "The number of dimensions on the output embedding vectors. "
        ),
    )

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        dimensions: Optional[int] = None,
        max_retries: int = 10,
        reuse_client: bool = True,
        timeout: float = 60.0,
        embed_batch_size: int = 100,
        num_workers: Optional[int] = None,
        additional_kwargs: Optional[dict] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        if dimensions is not None:
            additional_kwargs["dimensions"] = dimensions

        super().__init__(
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            callback_manager=callback_manager,
            model_name=model,
            api_key=api_key,
            api_base=api_base,
            max_retries=max_retries,
            reuse_client=reuse_client,
            timeout=timeout,
            num_workers=num_workers,
            default_headers=default_headers,
            additional_kwargs=additional_kwargs,
            **kwargs,
        )

        self._client = None
        self._aclient = None
        self._http_client = http_client
        self._async_http_client = async_http_client

    def _get_credential_kwargs(self, is_async: bool = False) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "http_client": self._async_http_client if is_async else self._http_client,
        }

    def _get_client(self) -> OpenAI:
        if not self.reuse_client:
            return OpenAI(**self._get_credential_kwargs())

        if self._client is None:
            self._client = OpenAI(**self._get_credential_kwargs())
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:
        if not self.reuse_client:
            return AsyncOpenAI(**self._get_credential_kwargs(is_async=True))

        if self._aclient is None:
            self._aclient = AsyncOpenAI(**self._get_credential_kwargs(is_async=True))
        return self._aclient

    def _get_query_embedding(self, query: str) -> List[float]:
        client = self._get_client()
        return get_embedding(
            client,
            query,
            engine=self.model_name,
            **self.additional_kwargs,
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        aclient = self._get_aclient()
        return await aget_embedding(
            aclient,
            query,
            engine=self.model_name,
            **self.additional_kwargs,
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        client = self._get_client()
        return get_embedding(
            client,
            text,
            engine=self.model_name,
            **self.additional_kwargs,
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        aclient = self._get_aclient()
        return await aget_embedding(
            aclient,
            text,
            engine=self.model_name,
            **self.additional_kwargs,
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        return get_embeddings(
            client,
            texts,
            engine=self.model_name,
            **self.additional_kwargs,
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        aclient = self._get_aclient()
        return await aget_embeddings(
            aclient,
            texts,
            engine=self.model_name,
            **self.additional_kwargs,
        )
