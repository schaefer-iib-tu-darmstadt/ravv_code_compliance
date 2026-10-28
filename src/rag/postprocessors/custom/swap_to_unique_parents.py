"""Postprocessor that replaces retrieved section nodes by their unique parent chapters in reranked order."""
from collections import OrderedDict
from typing import List, Optional, Any

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, NodeRelationship
from llama_index.core.bridge.pydantic import ConfigDict, Field

class SwapToUniqueParentChaptersPostprocessor(BaseNodePostprocessor):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    docstore: Any = Field(...)
    limit_unique_parents: Optional[int] = None

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle=None) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        by_parent: "OrderedDict[str, NodeWithScore]" = OrderedDict()

        get_node_fn = getattr(self.docstore, "get_node", None)
        get_document_fn = getattr(self.docstore, "get_document", None)

        for nws in nodes:
            node = nws.node
            score = nws.score

            parent_id = None
            if node.relationships and NodeRelationship.PARENT in node.relationships:
                parent_rel = node.relationships[NodeRelationship.PARENT]
                parent_id = getattr(parent_rel, "node_id", None)

            if parent_id:
                parent_node = None
                try:
                    if callable(get_node_fn):
                        parent_node = get_node_fn(parent_id, raise_error=False)
                    elif callable(get_document_fn):
                        parent_node = get_document_fn(parent_id, raise_error=False)
                except Exception:
                    parent_node = None
                if parent_node is None:
                    parent_node = node
                    parent_id = getattr(node, "node_id", getattr(node, "id_", None))
                    if parent_id is None:
                        continue

                if (parent_id not in by_parent) or (score or 0.0) > (by_parent[parent_id].score or 0.0):
                    by_parent[parent_id] = NodeWithScore(node=parent_node, score=score)
            else:
                pid = getattr(node, "node_id", getattr(node, "id_", None))
                if pid is None:
                    continue
                if (pid not in by_parent) or (score or 0.0) > (by_parent[pid].score or 0.0):
                    by_parent[pid] = NodeWithScore(node=node, score=score)

        result = list(by_parent.values())

        print(f"[SwapToUniqueParentChaptersPostprocessor] {len(result)} unique parent nodes found and loaded.")

        if self.limit_unique_parents is not None:
            result = result[: self.limit_unique_parents]

        return result

    def postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle=None) -> List[NodeWithScore]:
        return self._postprocess_nodes(nodes, query_bundle)