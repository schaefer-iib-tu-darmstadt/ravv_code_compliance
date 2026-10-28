"""Map a config's "type" to its pipeline class."""

from .pipelines.base_pipeline import BasePipeline
from .pipelines.no_rag import NoRAGPipeline
from .pipelines.basic_rag import BasicRAGPipeline
from .pipelines.advanced_rag import AdvancedRAGPipeline
from .pipelines.graph_rag import GraphRAGPipeline
from .pipelines.structured_graph_rag import StructuredGraphRAGPipeline
from .pipelines.no_rag_vision import NoRAGVisionPipeline
from .pipelines.advanced_rag_vision import AdvancedRAGVisionPipeline

def create_pipeline(pipeline_name: str, config: dict) -> BasePipeline:
    pipeline_type = config.get("type")
    if not pipeline_type:
        raise ValueError(f"Configuration for '{pipeline_name}' is missing a 'type' key.")

    if pipeline_type == "no_rag":
        return NoRAGPipeline(config, pipeline_name)

    elif pipeline_type == "basic_rag":
        return BasicRAGPipeline(config, pipeline_name)
    elif pipeline_type == "advanced_rag":
        return AdvancedRAGPipeline(config, pipeline_name)
    elif pipeline_type == "graph_rag":
        return GraphRAGPipeline(config, pipeline_name)
    elif pipeline_type == "structured_graph_rag":
        return StructuredGraphRAGPipeline(config, pipeline_name)

    elif pipeline_type == "no_rag_vision":
        return NoRAGVisionPipeline(config, pipeline_name)
    elif pipeline_type == "advanced_rag_vision":
        return AdvancedRAGVisionPipeline(config, pipeline_name)

    else:
        raise ValueError(f"Unknown pipeline type: '{pipeline_type}'")