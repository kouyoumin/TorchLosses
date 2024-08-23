from typing import Sequence
from inspect import signature
import torch


class WeightedMultiLoss(torch.nn.Module):
    def __init__(self, criterions:Sequence[torch.nn.Module], weights:Sequence[float]):
        super(WeightedMultiLoss, self).__init__()
        assert(len(criterions) == len(weights))
        self.criterions = criterions
        self.weights = weights
    
    
    def __call__(self, *args):
        loss = 0
        for criterion, weight in zip(self.criterions, self.weights):
            if isinstance(criterion, torch.nn.CosineEmbeddingLoss):
                loss += criterion(*args[:3]) * weight
            else:
                loss += criterion(*args[:2]) * weight
        return loss


if __name__ == "__main__":
    import torch.nn as nn
    criterion1 = nn.MSELoss() # two inputs
    criterion2 = nn.CosineEmbeddingLoss() # three inputs
    mixed = WeightedMultiLoss([criterion1, criterion2], [0.5, 0.5])
    mixed(torch.randn(1,5), torch.randn(1,5), torch.ones(1))
