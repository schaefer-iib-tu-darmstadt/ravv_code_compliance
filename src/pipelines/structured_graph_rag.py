"""Structure-aware graph RAG for the compilation task: the clause-id set retrieved from the
deterministic clause graph (src/rag/graph/structured_graph.py) is the answer; no LLM is called.
"""
import logging
from pathlib import Path

from src.rag.graph.structured_graph import (
    build_graph_from_markdown,
    format_answer,
    lexical_seeds,
    topic_synonyms,
    expand,
    REL_SUBCLAUSE,
    REL_CROSSREF,
)
from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class _Response:
    def __init__(self, text: str, ids):
        self.text = text
        self.ids = ids
        self.source_nodes = []

    def __str__(self) -> str:
        return self.text


class _DeterministicEngine:
    def __init__(self, graph, *, use_subtree: bool, use_crossref: bool):
        self.graph = graph
        self.use_subtree = use_subtree
        self.use_crossref = use_crossref

    def query(self, question: str) -> _Response:
        seeds = lexical_seeds(self.graph, topic_synonyms(question))
        ids = expand(
            self.graph, seeds,
            use_subtree=self.use_subtree, use_crossref=self.use_crossref,
        )
        return _Response(format_answer(ids), ids)


class StructuredGraphRAGPipeline(BasePipeline):
    def __init__(self, config, pipeline_name):
        super().__init__(config, pipeline_name)
        general = config.get("general", {})
        graph_cfg = config.get("graph", {})

        self.data_path = Path(general.get("data_path"))
        self.use_subtree = graph_cfg.get("use_subtree", True)
        self.use_crossref = graph_cfg.get("use_crossref", True)
        self.generation = graph_cfg.get("generation", "deterministic")
        if self.generation != "deterministic":
            raise ValueError(
                f"Unknown structured_graph_rag generation mode: {self.generation!r} "
                "(only 'deterministic' is supported)"
            )

    def get_query_engine(self):
        print("\n--- Setting up Structured Graph RAG Query Engine ---")
        graph = build_graph_from_markdown(self.data_path)
        n_sub = sum(1 for *_e, k in graph.edges(keys=True) if k == REL_SUBCLAUSE)
        n_xref = sum(1 for *_e, k in graph.edges(keys=True) if k == REL_CROSSREF)
        print(
            f"Built property graph: {graph.number_of_nodes()} clauses, "
            f"{n_sub} {REL_SUBCLAUSE}, {n_xref} {REL_CROSSREF} edges "
            f"(subtree={self.use_subtree}, crossref={self.use_crossref})"
        )
        engine = _DeterministicEngine(
            graph, use_subtree=self.use_subtree, use_crossref=self.use_crossref,
        )
        print("[ok] Structured Graph RAG Query Engine ready")
        return engine
