"""No RAG vision baseline: full rulebook text + the query image to a VLM, no retrieval."""

from pathlib import Path
from llama_index.core.base.response.schema import Response
from .base_pipeline import BasePipeline
from src.utils.llm_api.messages import create_messages_with_reference_and_image
from src.utils.llm_api.chat_completions import llm_chat_completion_message_content
from src.utils.llm_api.client import create_client


class NoRAGVisionPipeline(BasePipeline):
    def __init__(self, config, pipeline_name):
        super().__init__(config, pipeline_name)

        general_config = config.get("general", {})
        llm_config = config.get("llm", {})

        self.data_path = Path(general_config.get("data_path"))
        self.llm_model = llm_config.get("llm_model")
        self.temperature = llm_config.get("temperature", 0.0)
        self.llm_provider = llm_config.get("llm_provider", "gwdg")
        self.max_output_tokens = llm_config.get("max_tokens")
        self.client = create_client(self.llm_provider)

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.full_context = f.read()

        print(f"Loaded text context ({len(self.full_context)} chars)")
        print(f"Vision model: {self.llm_model}")

    def query(self, query_str: str, image_path) -> Response:
        if not image_path:
            raise ValueError("NoRAGVisionPipeline requires 'image_path' argument in query()")

        system_prompt = "You are a helpful assistant answering questions based on provided documents and images."

        messages = create_messages_with_reference_and_image(
            system_prompt=system_prompt,
            question=query_str,
            reference=self.full_context,
            image_path=str(image_path)
        )

        answer = llm_chat_completion_message_content(
            client=self.client,
            model=self.llm_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )

        return Response(response=answer, source_nodes=[])

    def get_query_engine(self):
        print("No-RAG Vision Engine is ready.")
        return self