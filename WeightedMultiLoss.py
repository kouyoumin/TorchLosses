from typing import Sequence
import torch


class WeightedMultiLoss(torch.nn.Module):
    def __init__(self, criterions:Sequence[torch.nn.Module], weights:Sequence[float]):
        super(WeightedMultiLoss, self).__init__()
        assert(len(criterions) == len(weights))
        self.criterions = criterions
        self.weights = weights
    
    
    def __call__(self, output, target):
        loss = 0
        for criterion, weight in zip(self.criterions, self.weights):
            loss += criterion(output, target) * weight
        return loss