"""Abstract base class of all pipelines."""

from abc import ABC, abstractmethod

class BasePipeline(ABC):
    def __init__(self, config: dict, pipeline_name: str):
        self.config: dict = config
        self.pipeline_name: str = pipeline_name

    @abstractmethod
    def get_query_engine(self):
        pass