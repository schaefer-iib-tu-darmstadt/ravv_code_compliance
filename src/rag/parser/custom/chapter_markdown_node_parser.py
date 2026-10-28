"""Markdown parser with one node per heading section; stores the enclosing H1 chapter as metadata."""
import re
from typing import List, Optional

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.file.markdown import MarkdownNodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode


class ChapterMarkdownNodeParser(MarkdownNodeParser):
    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        header_path_separator: str = "/",
        callback_manager: Optional[CallbackManager] = None,
    ) -> "ChapterMarkdownNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            header_path_separator=header_path_separator,
            callback_manager=callback_manager,
        )

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        markdown_nodes: List[TextNode] = []
        lines = text.split("\n")
        current_section = ""
        header_stack: List[tuple[int, str]] = []
        code_block = False

        for line in lines:
            if line.lstrip().startswith("```"):
                code_block = not code_block
                current_section += line + "\n"
                continue

            if not code_block:
                header_match = re.match(r"^(#+)\s(.*)", line)
                if header_match:
                    if current_section.strip():
                        header_path = self.header_path_separator.join(h[1] for h in header_stack[:-1])
                        chapter_text = header_stack[0][1] if header_stack else ""
                        markdown_nodes.append(
                            self._build_node_from_split(
                                current_section.strip(), node, header_path, chapter_text
                            )
                        )

                    header_level = len(header_match.group(1))
                    header_text = header_match.group(2)

                    while header_stack and header_stack[-1][0] >= header_level:
                        header_stack.pop()

                    header_stack.append((header_level, header_text))
                    current_section = "#" * header_level + f" {header_text}\n"
                    continue

            current_section += line + "\n"

        if current_section.strip():
            header_path = self.header_path_separator.join(h[1] for h in header_stack[:-1])
            chapter_text = header_stack[0][1] if header_stack else ""
            markdown_nodes.append(
                self._build_node_from_split(
                    current_section.strip(), node, header_path, chapter_text
                )
            )

        return markdown_nodes

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        header_path: str,
        chapter: str = "",
    ) -> TextNode:
        new_node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

        if self.include_metadata:
            sep = self.header_path_separator
            new_node.metadata["header_path"] = (sep + header_path + sep) if header_path else sep
            new_node.metadata["chapter"] = chapter or ""

        return new_node
