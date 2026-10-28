"""Two-stage parser of the Advanced RAG hierarchy: H1 chapters as parent nodes, their sections as child nodes."""
from typing import List
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship
from src.rag.parser.custom.chapter_markdown_node_parser import ChapterMarkdownNodeParser
from src.rag.parser.custom.h1_markdown_parser import H1OnlyMarkdownNodeParser

class TwoStageH1ChapterParser:
    def __init__(self, include_metadata: bool = True, include_prev_next_rel: bool = True):
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel
        self.h1_parser = H1OnlyMarkdownNodeParser(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )
        self.chapter_parser = ChapterMarkdownNodeParser(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )

    def get_nodes_from_documents(self, documents) -> List[BaseNode]:
        nodes: List[BaseNode] = []

        parents = self.h1_parser.get_nodes_from_documents(documents)
        for parent in parents:
            pmeta = dict(parent.metadata or {})
            pmeta["node_type"] = "md_parent"
            parent.metadata = pmeta
            nodes.append(parent)

            children: List[TextNode] = self.chapter_parser.get_nodes_from_node(parent)

            for idx, child in enumerate(children):
                cmeta = {**parent.metadata, **(child.metadata or {})}
                cmeta["node_type"] = "child"
                cmeta["parent_id"] = getattr(parent, "node_id", getattr(parent, "id_", None))
                cmeta["section_idx"] = idx
                child.metadata = cmeta

                child.relationships[NodeRelationship.PARENT] = parent.as_related_node_info()

                nodes.append(child)

        return nodes
