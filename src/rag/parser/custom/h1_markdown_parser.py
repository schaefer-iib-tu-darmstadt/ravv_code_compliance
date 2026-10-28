"""Markdown parser with one node per H1 chapter of the rulebook."""
import re
from typing import List, Optional

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.file.markdown import MarkdownNodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode


class H1OnlyMarkdownNodeParser(MarkdownNodeParser):
    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        header_path_separator: str = "/",
        callback_manager: Optional[CallbackManager] = None,
    ) -> "H1OnlyMarkdownNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            header_path_separator=header_path_separator,
            callback_manager=callback_manager,
        )

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        lines = text.split("\n")

        markdown_nodes: List[TextNode] = []
        current_section = ""
        current_chapter = ""
        saw_first_h1 = False
        code_block = False

        for line in lines:
            if line.lstrip().startswith("```"):
                code_block = not code_block
                current_section += line + "\n"
                continue

            if not code_block:
                m_h1 = re.match(r"^#\s+(.*)", line)
                if m_h1:
                    h1_text = m_h1.group(1)

                    if not saw_first_h1:
                        saw_first_h1 = True
                        current_chapter = h1_text
                        header_line = f"# {current_chapter}\n"
                        if current_section.strip():
                            current_section = current_section.strip() + "\n\n" + header_line
                        else:
                            current_section = header_line
                        continue
                    else:
                        if current_section.strip():
                            markdown_nodes.append(
                                self._build_node_from_split(
                                    current_section.strip(), node, chapter=current_chapter
                                )
                            )
                        current_chapter = h1_text
                        current_section = f"# {current_chapter}\n"
                        continue

            current_section += line + "\n"

        if current_section.strip():
            markdown_nodes.append(
                self._build_node_from_split(
                    current_section.strip(), node, chapter=current_chapter if saw_first_h1 else ""
                )
            )

        return markdown_nodes

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        chapter: str = "",
    ) -> TextNode:
        new_node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

        if self.include_metadata:
            sep = self.header_path_separator
            new_node.metadata["header_path"] = sep  
            new_node.metadata["chapter"] = chapter or ""

        return new_node