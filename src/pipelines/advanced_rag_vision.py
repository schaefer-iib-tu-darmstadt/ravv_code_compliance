"""RAVV pipeline: (0) LLM extracts the rule id, (1) Advanced RAG retrieves the rule text,
(2) a VLM judges compliance from rule text + image (instead of the full rulebook as in No RAG vision).
"""

import os
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client.http import models as rest

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import NodeRelationship
from llama_index.core.base.response.schema import Response
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine

from src.utils.rag.naming_utils import build_collection_name, ensure_list, make_instance
from src.utils.rag.qdrant_utils import create_qdrant_client, collection_is_empty, delete_collection_if_exists
from src.utils.llm_api.messages import create_messages_with_reference_and_image
from src.utils.llm_api.chat_completions import llm_chat_completion_message_content
from src.utils.llm_api.client import create_client
from src.utils.llm_api import timing_log

from src.gwdg_settings import llama_index_init
from src.rag.postprocessors.custom.swap_to_unique_parents import SwapToUniqueParentChaptersPostprocessor
from src.prompts import QA_PROMPT, VISION_STAGE2_SYSTEM_PROMPT
from .base_pipeline import BasePipeline

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AdvancedRAGVisionPipeline(BasePipeline):
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

        index_source = retrieval_config.get("index_source_pipeline", pipeline_name)
        self.persist_dir = Path(".storage") / index_source

        self.retrieval_llm_model = retrieval_config.get("llm_model")
        self.retrieval_llm_provider = retrieval_config.get("llm_provider", "gwdg")
        self.vision_llm_model = llm_config.get("llm_model")
        self.temperature = llm_config.get("temperature", 0.0)
        self.llm_provider = llm_config.get("llm_provider", "gwdg")
        self.max_output_tokens = llm_config.get("max_tokens", 1024)

        self.retrieval_client = create_client(self.retrieval_llm_provider)
        self.vision_client = (
            self.retrieval_client
            if self.llm_provider == self.retrieval_llm_provider
            else create_client(self.llm_provider)
        )
        self.rule_extraction_model = retrieval_config.get("llm_model")

        self.qdrant_client = create_qdrant_client()

        self.query_mode = self.retriever_kwargs.get("vector_store_query_mode")
        if self.query_mode == "sparse" and not self.hybrid_search:
            raise ValueError("Config: vector_store_query_mode='sparse' requires retrieval.hybrid_search=True.")

        self.collection_name = build_collection_name(
            pipeline_name=index_source,
            embed_model=self.embed_model,
            fastembed_sparse_model=self.fastembed_sparse_model,
            hybrid_search=self.hybrid_search,
            query_mode=self.query_mode
        )

        print(f"Rule Extraction LLM (Stage 0): {self.rule_extraction_model} [{self.retrieval_llm_provider}]")
        print(f"Retrieval LLM (Stage 1): {self.retrieval_llm_model} [{self.retrieval_llm_provider}]")
        print(f"Vision LLM (Stage 2): {self.vision_llm_model} [{self.llm_provider}]")

    def _initialize_settings(self):
        print(f"\n--- Initializing Advanced RAG Vision Pipeline for: {self.pipeline_name} ---")
        llama_index_init.init(
            llm_model_name=self.retrieval_llm_model,
            embedding_model_name=self.embed_model,
            temperature=self.temperature,
            llm_provider=self.retrieval_llm_provider,
        )

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
            self.storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(self.persist_dir)
            )
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

    def _setup_retriever_and_synthesizer(self):
        retriever = self.index.as_retriever(**self.retriever_kwargs)
        print(f"Retriever configured with arguments: {self.retriever_kwargs}")

        raw_postprocessors = ensure_list(self.node_postprocessors)
        node_postprocessors = []
        reranker_top_n = None

        for pp in raw_postprocessors:
            inst = make_instance(pp)
            if inst is None:
                continue
            top_n = getattr(inst, "top_n", None)
            if isinstance(top_n, int) and top_n > 0:
                reranker_top_n = top_n
            node_postprocessors.append(inst)

        if self._docstore_available:
            auto_limit = reranker_top_n or self.retriever_kwargs.get("similarity_top_k") or self.retriever_kwargs.get("sparse_top_k")
            node_postprocessors.append(
                SwapToUniqueParentChaptersPostprocessor(
                    docstore=self.storage_context.docstore,
                    limit_unique_parents=auto_limit
                )
            )

        synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=QA_PROMPT,  
        )

        self.text_query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=node_postprocessors,
        )

        print(f"Configured {len(node_postprocessors)} postprocessors")
        print("Configured query engine with QA_PROMPT")

    def _extract_rule_number(self, question: str) -> str:
        print(f"Stage 0: Extracting rule number with {self.rule_extraction_model}")

        system_prompt = """You are a specialized assistant that extracts FSAE rule numbers from questions.

                        TASK:
                        Extract ONLY the rule number from the question.

                        OUTPUT FORMAT:
                        - Output ONLY the rule number (e.g., V.1.2, D.12.3, F.5.7.5, T.4.3)
                        - No explanations, no extra text
                        - If no rule number is found, output exactly: NO_RULE_FOUND"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        rule_number = llm_chat_completion_message_content(
            client=self.retrieval_client,
            model=self.rule_extraction_model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )

        print(f"Extracted rule number: {rule_number}")

        rule_number = (rule_number or "").strip()
        if not rule_number:
            raise ValueError(
                f"LLM returned empty/None response when extracting rule number from: {question[:100]}..."
            )
        if rule_number.rstrip(".").upper() in {"NO_RULE_FOUND", "API-ERROR"}:
            raise ValueError(f"Could not extract valid rule number from question. Got: {rule_number!r}")

        return rule_number

    def _build_rag_question(self, rule_number: str) -> str:
        return f"We are a student engineering team designing a vehicle for the FSAE competition. Attached is the FSAE rules document. What does rule {rule_number} state exactly? Answer with only the text of the rule and no other words."

    def _extract_rule_text(self, rag_question: str) -> str:
        print(f"Stage 1: Extracting rule text with {self.retrieval_llm_model}")

        response = self.text_query_engine.query(rag_question)
        rule_text = str(response)

        print(f"Extracted rule text ({len(rule_text)} chars)")
        print("\n" + "="*80)
        print("EXTRACTED RULE TEXT:")
        print("="*80)
        print(rule_text)
        print("="*80 + "\n")

        return rule_text

    def query(self, query_str: str, image_path=None) -> Response:
        if not image_path:
            raise ValueError("AdvancedRAGVisionPipeline requires 'image_path' argument in query()")

        with timing_log.stage("stage0_rule_extract"):
            rule_number = self._extract_rule_number(query_str)

        rag_question = self._build_rag_question(rule_number)
        print(f"Built RAG question: {rag_question}")

        with timing_log.stage("stage1_rag_rule_text"):
            rule_text = self._extract_rule_text(rag_question)

        print(f"Stage 2: Analyzing with vision model {self.vision_llm_model}")

        messages = create_messages_with_reference_and_image(
            system_prompt=VISION_STAGE2_SYSTEM_PROMPT,
            question=query_str,
            reference=rule_text,
            image_path=str(image_path)
        )

        with timing_log.stage("stage2_vision"):
            answer = llm_chat_completion_message_content(
                client=self.vision_client,
                model=self.vision_llm_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )

        print(f"Vision LLM response ({len(answer)} chars)")
        print("\n" + "="*80)
        print("VISION LLM RESPONSE:")
        print("="*80)
        print(answer)
        print("="*80 + "\n")

        return Response(response=answer, source_nodes=[])

    def get_query_engine(self):
        print("\n--- Setting up Advanced RAG Vision Query Engine ---")
        self._initialize_settings()
        self._connect_vector_store()
        self._build_or_load_vector_index()
        self._setup_retriever_and_synthesizer()

        print("Advanced RAG Vision Engine is ready.")
        return self