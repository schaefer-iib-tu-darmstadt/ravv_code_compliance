"""Parser factory for the Advanced RAG hierarchy (PARSER_CONFIGS["hierarchical_h1"])."""

from src.rag.parser.custom.two_stage_h1_chapter_parser import TwoStageH1ChapterParser


def create_two_stage_h1_chapter_parser():
    return TwoStageH1ChapterParser(include_metadata=True, include_prev_next_rel=True)
