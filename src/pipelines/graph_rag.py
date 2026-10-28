"""Graph RAG on an LLM-extracted in-memory property graph (zero-shot or FSAE-ontology-guided extractor),
retrieved by vector search plus graph traversal. The extracted graph is persisted and reloaded for reproducibility.
"""

import os
import json
import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor, SimpleLLMPathExtractor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode

from src.gwdg_settings import llama_index_init
from src.prompts import GRAPH_COMPILATION_PROMPT
from .base_pipeline import BasePipeline

FSAE_ENTITIES = Literal["RULE", "TOPIC", "COMPONENT", "MATERIAL", "SYSTEM"]
FSAE_RELATIONS = Literal["HAS_SUBRULE", "CITES", "CONTAINS_TOPIC", "APPLIES_TO"]
FSAE_VALIDATION_SCHEMA = [
    ("RULE", "HAS_SUBRULE", "RULE"),
    ("RULE", "CITES", "RULE"),
    ("RULE", "CONTAINS_TOPIC", "TOPIC"),
    ("RULE", "APPLIES_TO", "COMPONENT"),
    ("RULE", "APPLIES_TO", "MATERIAL"),
    ("RULE", "APPLIES_TO", "SYSTEM"),
]


load_dotenv()
logger = logging.getLogger(__name__)


class GraphRAGPipeline(BasePipeline):
    def __init__(self, config, pipeline_name):
        super().__init__(config, pipeline_name)

        general_config = config.get("general", {})
        retrieval_config = config.get("retrieval", {})
        llm_config = config.get("llm", {})
        graph_config = config.get("graph", {})

        self.data_path = Path(general_config.get("data_path"))
        self.embed_model = retrieval_config.get("embed_model")
        self.llm_model = llm_config.get("llm_model")
        self.temperature = llm_config.get("temperature", 0.0)
        self.llm_provider = llm_config.get("llm_provider", "gwdg")
        self.embed_provider = retrieval_config.get("embed_provider", "gwdg")

        self.extractor = graph_config.get("extractor", "simple")
        self.override_graph = graph_config.get("override_graph", False)
        self.persist_dir = graph_config.get("persist_dir")

        self.similarity_top_k = retrieval_config.get("similarity_top_k", 20)
        self.path_depth = retrieval_config.get("path_depth", 2)

        self.graph_store = SimplePropertyGraphStore()

    def _extraction_llm(self):
        from llama_index.llms.openai import OpenAI
        from src.gwdg_settings.constants import GWDG_API_KEY, LLM_API_ENDPOINT_KEY
        from src.gwdg_settings.rate_limited import RateLimitedOpenAILike
        if self.llm_provider == "gwdg":
            return RateLimitedOpenAILike(
                model=self.llm_model,
                is_chat_model=True,
                is_function_calling_model=True,
                temperature=self.temperature,
                api_key=GWDG_API_KEY,
                api_base=LLM_API_ENDPOINT_KEY,
                default_headers={"inference-service": self.llm_model},
            )
        return OpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY"),
            is_function_calling_model=True,
        )

    def _build_extractor(self):
        if self.extractor == "schema":
            logger.info("Using SchemaLLMPathExtractor (ontology-guided, FSAE schema)")
            return SchemaLLMPathExtractor(
                llm=self._extraction_llm(),
                possible_entities=FSAE_ENTITIES,
                possible_relations=FSAE_RELATIONS,
                kg_validation_schema=FSAE_VALIDATION_SCHEMA,
                strict=True,
                num_workers=1,
            )
        logger.info("Using SimpleLLMPathExtractor (zero-shot, no ontology)")
        return SimpleLLMPathExtractor(
            llm=Settings.llm,
            num_workers=1,
        )

    def _create_graph(self):
        logger.info("Creating new Property Graph (extractor=%s)...", self.extractor)
        documents = SimpleDirectoryReader(
            input_files=[self.data_path]
        ).load_data(show_progress=True, num_workers=1)

        index = PropertyGraphIndex.from_documents(
            documents,
            embed_model=Settings.embed_model,
            kg_extractors=[self._build_extractor()],
            property_graph_store=self.graph_store,
            show_progress=True,
        )
        logger.info("[ok] Graph created")

        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            tmp = self._persist_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(self.graph_store.graph.model_dump_json())
            os.replace(tmp, self._persist_file)
            logger.info("[ok] Graph persisted to %s", self._persist_file)
        return index

    @property
    def _persist_file(self):
        return os.path.join(self.persist_dir, "property_graph_store.json")

    def _load_graph(self):
        logger.info("Loading persisted Property Graph from %s ...", self._persist_file)
        with open(self._persist_file, "r", encoding="utf-8") as f:
            store = SimplePropertyGraphStore.from_dict(json.loads(f.read()))
        index = PropertyGraphIndex.from_existing(
            property_graph_store=store,
            embed_model=Settings.embed_model,
            llm=Settings.llm,
            embed_kg_nodes=False,
        )
        self.graph_store = store
        logger.info("[ok] Graph loaded from disk")
        return index

    def get_query_engine(self):
        print("\n--- Setting up Graph RAG Query Engine ---")

        llama_index_init.init(
            llm_model_name=self.llm_model,
            embedding_model_name=self.embed_model,
            temperature=self.temperature,
            embedding_provider=self.embed_provider,
            llm_provider=self.llm_provider,
        )

        if not self.override_graph and self.persist_dir and os.path.exists(self._persist_file):
            index = self._load_graph()
        else:
            index = self._create_graph()

        retriever = index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            path_depth=self.path_depth,
            include_text=True,
        )

        synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=GRAPH_COMPILATION_PROMPT,
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_synthesizer=synthesizer,
        )
        print("[ok] Graph RAG Query Engine ready")
        return query_engine
