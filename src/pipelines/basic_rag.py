"""Basic RAG: SentenceSplitter chunks, dense Qdrant retrieval (OpenAI embeddings), no reranking."""

import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode

from src.utils.rag.naming_utils import build_collection_name
from src.utils.rag.qdrant_utils import create_qdrant_client, collection_is_empty
from src.gwdg_settings import llama_index_init
from .base_pipeline import BasePipeline
from src.prompts import QA_PROMPT

logger = logging.getLogger(__name__)

class BasicRAGPipeline(BasePipeline):
    def __init__(self, config, pipeline_name):
        super().__init__(config, pipeline_name)

        general_config = config.get("general", {})
        retrieval_config = config.get("retrieval", {})
        llm_config = config.get("llm", {})

        self.data_path = Path(general_config.get("data_path"))
        self.embed_model = retrieval_config.get("embed_model")
        self.retriever_kwargs = retrieval_config.get("retriever_kwargs", {"similarity_top_k": 5})
        self.llm_model = llm_config.get("llm_model")
        self.temperature = llm_config.get("temperature")

        self.parser_config = retrieval_config.get("parser")

        self.collection_name = build_collection_name(pipeline_name=pipeline_name, embed_model=self.embed_model)
        self.qdrant_client = create_qdrant_client()

    def _resolve_parser(self):
        cfg = self.parser_config

        default_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        if cfg is None:
            print("No parser configured, using default SentenceSplitter(512, 50)")
            return default_parser

        if isinstance(cfg, SentenceSplitter):
            print(f"Using provided parser instance: {cfg}")
            return cfg

        if isinstance(cfg, dict):
            class_name = cfg.get("class")
            kwargs = cfg.get("kwargs", {})

            if class_name == "SentenceSplitter":
                try:
                    parser = SentenceSplitter(**kwargs)
                    print(f"Created SentenceSplitter with: {kwargs}")
                    return parser
                except Exception as e:
                    print(f"Error creating parser: {e}. Using default.")
                    return default_parser
            else:
                print(f"Unknown parser class '{class_name}'. Using default.")
                return default_parser

        print(f"Invalid parser config type: {type(cfg)}. Using default.")
        return default_parser

    def get_query_engine(self):
        print("\n--- Setting up Basic RAG Query Engine ---")

        llama_index_init.init(
            llm_model_name=self.llm_model,
            embedding_model_name=self.embed_model,
            temperature=self.temperature,
            embedding_provider="openai",
        )

        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        parser = self._resolve_parser()
        print(f"Using parser: {parser.__class__.__name__}")

        if not collection_is_empty(self.qdrant_client, self.collection_name):
            print(f"Loading existing index from collection '{self.collection_name}'.")
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )
        else:
            print(f"Building index for collection '{self.collection_name}'.")
            documents = SimpleDirectoryReader(input_files=[self.data_path]).load_data()
            nodes = parser.get_nodes_from_documents(documents)
            print(f"Created {len(nodes)} nodes (parser: {parser.__class__.__name__}).")
            index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

        retriever = index.as_retriever(**self.retriever_kwargs)
        print(f"Retriever kwargs: {self.retriever_kwargs}")

        synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=QA_PROMPT,
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_synthesizer=synthesizer,
        )
        print("Basic RAG Query Engine ready.")
        return query_engine
