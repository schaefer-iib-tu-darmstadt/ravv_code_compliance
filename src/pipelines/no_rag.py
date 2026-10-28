"""No RAG baseline (in-context retrieval): the full rulebook text is the context of every query."""


from pathlib import Path
from llama_index.core import Settings
from llama_index.core.base.response.schema import Response
from src.gwdg_settings import llama_index_init
from .base_pipeline import BasePipeline
from src.prompts import QA_PROMPT


class NoRAGPipeline(BasePipeline):
    def __init__(self, config, pipeline_name):
        super().__init__(config, pipeline_name)

        general_config = config.get("general", {})
        llm_config = config.get("llm", {})

        self.data_path = Path(general_config.get("data_path"))
        self.llm_model = llm_config.get("llm_model")
        self.temperature = llm_config.get("temperature")

        self._initialize_settings()
        self.llm = Settings.llm

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.full_context = f.read()
        print(f"Loaded full document context ({len(self.full_context)} chars).")
        estimated_tokens = int(len(self.full_context) / 4)
        print(f"   Estimated Tokens (4 chars/token): ~{estimated_tokens} tokens.")

    def _initialize_settings(self):
        print(f"\n--- Initializing No-RAG Pipeline for: {self.pipeline_name} ---")
        llama_index_init.init(llm_model_name=self.llm_model, temperature=self.temperature)

    def query(self, query_str: str) -> Response:
        prompt = QA_PROMPT.format(
            context_str=self.full_context,
            query_str=query_str
        )
        llm_response = self.llm.complete(prompt)
        return Response(response=str(llm_response), source_nodes=[])

    def get_query_engine(self):
        print("No-RAG Engine is ready.")
        return self