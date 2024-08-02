from typing import Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class MRL(nn.Module):
    def __init__(self, criterion, max_dimensions:int, normalize=False, groups:Union[Sequence[int], int]=1, linear=False, reduction='mean'):
        super(MRL, self).__init__()
        self.criterion = criterion
        self.normalize = normalize
        self.linear = linear
        if not isinstance(groups, Sequence):
            self.groups = []
            for g in range(groups):
                if self.linear:
                    dimensions = max_dimensions * (g+1) // groups
                else:
                    dimensions = max_dimensions // 2**g
                self.groups.append(dimensions)
        else:
            self.groups = groups
        self.reduction = reduction
    
    
    def __call__(self, input, target):
        losses = []
        for dimensions in self.groups:
            partial_input = input[:, :dimensions]
            partial_target = target[:, :dimensions]
            if self.normalize:
                partial_input = F.normalize(partial_input, dim=1)
                partial_target = F.normalize(partial_target, dim=1)
            loss = self.criterion(partial_input, partial_target)
            if loss.ndim > 1:
                loss = loss.mean(tuple(range(1, loss.ndim)))
            losses += [loss]
        
        if self.reduction == 'none':
            return torch.stack(losses)
        elif self.reduction == 'mean':
            return torch.stack(losses).mean()
        elif self.reduction == 'sum':
            return torch.stack(losses).sum()
        else:
            return torch.stack(losses).mean()


def test_mrl():
    criterion = nn.MSELoss()
    mrl1 = MRL(criterion, 8, normalize=True, groups=1)
    mrl2 = MRL(criterion, 8, normalize=True, groups=2)
    
    input = torch.zeros(4,8)
    target = torch.zeros(4,8)
    input[:,:4] = 1
    target[:,:8] = 1
    mseloss = criterion(F.normalize(input), F.normalize(target))
    mrl1loss = mrl1(input, target)
    mrl2loss = mrl2(input, target)
    print('MSELoss:', mseloss)
    print('MRL g1:', mrl1loss)
    assert(torch.isclose(mseloss, mrl1loss))
    print('MRL g2:', mrl2loss)
    assert(torch.isclose((0+mseloss)/2, mrl2loss))


if __name__ == '__main__':
    test_mrl()