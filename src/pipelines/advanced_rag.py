"""Advanced RAG: Qdrant retrieval over section nodes, node postprocessors (RankGPT top-8) and
expansion to unique parent chapters, which are kept in a local docstore (.storage/<pipeline>/).
"""

import os
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client.http import models as rest

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import NodeRelationship

from src.utils.rag.naming_utils import build_collection_name, ensure_list, make_instance
from src.utils.rag.qdrant_utils import create_qdrant_client, collection_is_empty, delete_collection_if_exists

from src.gwdg_settings import llama_index_init
from src.rag.postprocessors.custom.swap_to_unique_parents import SwapToUniqueParentChaptersPostprocessor
from .base_pipeline import BasePipeline
from src.prompts import QA_PROMPT

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AdvancedRAGPipeline(BasePipeline):
    def __init__(self, config, pipeline_name):
        super().__init__(config, pipeline_name)

        general_config = config.get("general")
        retrieval_config = config.get("retrieval")
        llm_config = config.get("llm")

        if general_config is None:
            raise ValueError("Missing 'general' section in pipeline config.")
        if retrieval_config is None:
            raise ValueError("Missing 'retrieval' section in pipeline config.")
        if llm_config is None:
            raise ValueError("Missing 'llm' section in pipeline config.")

        self.data_path = Path(general_config.get("data_path"))
        self.parser_selection = retrieval_config.get("parser")
        self.hybrid_search = bool(retrieval_config.get("hybrid_search"))
        self.embed_model = retrieval_config.get("embed_model")
        self.fastembed_sparse_model = retrieval_config.get("fastembed_sparse_model")
        self.retriever_kwargs = retrieval_config.get("retriever_kwargs")
        self.node_postprocessors = retrieval_config.get("node_postprocessor")
        self.llm_model = llm_config.get("llm_model")
        self.temperature = llm_config.get("temperature")

        index_source = retrieval_config.get("index_source_pipeline", pipeline_name)
        self.persist_dir = Path(".storage") / index_source
        self.skip_parent_expansion = bool(retrieval_config.get("skip_parent_expansion", False))

        self.qdrant_client = create_qdrant_client()

        self.query_mode = self.retriever_kwargs.get("vector_store_query_mode")
        if self.query_mode == "sparse" and not self.hybrid_search:
            raise ValueError("Config: vector_store_query_mode='sparse' requires retrieval.hybrid_search=True.")
        self.collection_name = build_collection_name(pipeline_name=index_source, embed_model=self.embed_model, fastembed_sparse_model=self.fastembed_sparse_model, hybrid_search=self.hybrid_search, query_mode=self.query_mode)

    def _initialize_settings(self):
        print(f"\n--- Initializing Advanced RAG Pipeline for: {self.pipeline_name} ---")
        llama_index_init.init(llm_model_name=self.llm_model, embedding_model_name=self.embed_model, temperature=self.temperature)

    def _connect_vector_store(self):
        logger.info(f"Connecting to Qdrant at: {QDRANT_URL} (collection: {self.collection_name})")
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            enable_hybrid=self.hybrid_search,
            fastembed_sparse_model=(self.fastembed_sparse_model if self.hybrid_search else None),
        )

        docstore_path = self.persist_dir / "docstore.json"
        if docstore_path.exists():
            self.storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(self.persist_dir))
            self._docstore_available = True
            logger.info(f"Loaded local docstore from {docstore_path}")
        else:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._docstore_available = False
            logger.warning("No local docstore found. Parent lookups may not work. Consider rebuilding the index.")

    def _ensure_qdrant_payload_indexes(self, enabled: bool = False):
        if not enabled:
            return
        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="node_type",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            logger.info("Created payload index on 'node_type'.")
        except Exception as e:
            logger.info(f"Payload index on 'node_type' already exists or could not be created: {e}")

    def _should_rebuild_index(self):
        has_local_docstore = (self.persist_dir / "docstore.json").exists()
        collection_empty = collection_is_empty(self.qdrant_client, self.collection_name)
        return (not has_local_docstore) or collection_empty

    def _reset_storage_context(self):
        storage_context = getattr(self, "storage_context", None)
        if storage_context is None:
            return
        vector_store = storage_context.vector_store
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._docstore_available = False

    def _rebuild_index(self):
        print("\n--- Rebuilding Index from Scratch ---")

        delete_collection_if_exists(self.qdrant_client, self.collection_name)
        shutil.rmtree(self.persist_dir, ignore_errors=True)
        self._reset_storage_context()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        documents = SimpleDirectoryReader(input_files=[self.data_path]).load_data()
        print(f"Loaded {len(documents)} document(s)")

        if isinstance(self.parser_selection, dict) and "class" in self.parser_selection:
            from llama_index.core.node_parser import SentenceSplitter
            cls_name = self.parser_selection["class"]
            kwargs = self.parser_selection.get("kwargs", {})
            if cls_name == "SentenceSplitter":
                parser = SentenceSplitter(**kwargs)
            else:
                raise ValueError(f"Unsupported parser class: {cls_name}")
        else:
            parser = make_instance(self.parser_selection)
        nodes = parser.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} nodes")

        nodes_to_index = self._store_parent_nodes(nodes)

        self.index = VectorStoreIndex(
            nodes_to_index,
            storage_context=self.storage_context,
            show_progress=True
        )
        self.index.storage_context.persist(persist_dir=str(self.persist_dir))
        self._docstore_available = True
        print(f"Index stored in collection: {self.collection_name}")

    def _store_parent_nodes(self, nodes):
        has_parent_rel = any(
            bool(n.relationships.get(NodeRelationship.PARENT)) for n in nodes
        )

        if not has_parent_rel:
            print("No parent relationships found. Indexing all nodes.")
            return nodes

        parents = [n for n in nodes if NodeRelationship.PARENT not in n.relationships]
        children = [n for n in nodes if NodeRelationship.PARENT in n.relationships]

        if parents:
            self.storage_context.docstore.add_documents(parents)
            print(f"Stored {len(parents)} parent nodes in docstore")

        print(f"Indexing {len(children)} child nodes")
        return children

    def _load_existing_index(self):
        print(f"\n--- Loading Existing Index: {self.collection_name} ---")
        try:
            collection = self.qdrant_client.get_collection(collection_name=self.collection_name)
            if collection.points_count > 0:
                print(f"Loading existing index from collection '{self.collection_name}'.")
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.storage_context.vector_store,
                    storage_context=self.storage_context
                )
                print("Index loaded successfully")
                return
            raise ValueError("Collection empty.")
        except Exception:
            print(f"Collection '{self.collection_name}' empty or missing - forcing rebuild.")
            self._rebuild_index()

    def _build_or_load_vector_index(self):
        if self._should_rebuild_index():
            self._rebuild_index()
        else:
            self._load_existing_index()

        self._ensure_qdrant_payload_indexes(enabled=False)

    def get_query_engine(self):
        print("\n--- Setting up Advanced RAG Query Engine ---")
        self._initialize_settings()
        self._connect_vector_store()
        self._build_or_load_vector_index()

        retriever = self.index.as_retriever(**self.retriever_kwargs)
        print(f"Retriever configured with arguments: {self.retriever_kwargs}")

        raw_postprocessors = ensure_list(self.node_postprocessors)
        node_postprocessors = []
        reranker_top_n = None

        for pp in raw_postprocessors:
            inst = make_instance(pp)
            if inst is None: continue
            top_n = getattr(inst, "top_n", None)
            if isinstance(top_n, int) and top_n > 0:
                reranker_top_n = top_n
            node_postprocessors.append(inst)

        if self._docstore_available and not self.skip_parent_expansion:
            auto_limit = reranker_top_n or self.retriever_kwargs.get("similarity_top_k") or self.retriever_kwargs.get("sparse_top_k")
            node_postprocessors.append(SwapToUniqueParentChaptersPostprocessor(docstore=self.storage_context.docstore, limit_unique_parents=auto_limit))

        synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=QA_PROMPT,
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=node_postprocessors,
        )

        print("Advanced RAG Query Engine is ready.")
        return query_engine
