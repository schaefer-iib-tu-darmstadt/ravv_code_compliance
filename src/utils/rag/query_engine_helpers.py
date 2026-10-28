"""Build the query engine of a named pipeline configuration."""
from src.strategies import PIPELINE_CONFIGS
from src.pipeline_factory import create_pipeline

def get_query_engine(pipeline_name: str):
    config = PIPELINE_CONFIGS.get(pipeline_name)
    if not config:
        print(f"Error: Pipeline configuration '{pipeline_name}' not found.")
        print("Available configurations:", list(PIPELINE_CONFIGS.keys()))
        return None

    pipeline = create_pipeline(pipeline_name, config)
    return pipeline.get_query_engine()