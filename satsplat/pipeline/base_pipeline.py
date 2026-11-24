from abc import ABC, abstractmethod

import torch

from ..dataset import SatDataset
from ..module import GaussianModel


class BasePipeline(torch.nn.Module, ABC):
    dataset: SatDataset
    model: GaussianModel
    color_dim: int
    feature_dim: int

    def __init__(self, dataset: SatDataset, color_dim: int, feature_dim: int):
        super().__init__()
        self.color_dim = color_dim
        self.feature_dim = feature_dim
        self.dataset = dataset

    def to(self, device):
        super().to(device)
        self.dataset.to(device)
        self.model.to(device)
        return self

    @abstractmethod
    def on_train_start(self):
        pass

    @abstractmethod
    def training_step(self, step):
        pass

    @abstractmethod
    def evaluation_step(self, step):
        pass
