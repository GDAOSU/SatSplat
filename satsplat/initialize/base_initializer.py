from abc import ABC, abstractmethod
from ..pipeline.base_pipeline import BasePipeline
from ..module.gaussians import GaussianModel


class BaseInitializer(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def initialize(self, pipeline: BasePipeline) -> GaussianModel:
        pass
