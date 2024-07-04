import torch
import torch.nn as nn
import torch.nn.functional as F


class MRL(nn.Module):
    def __init__(self, criterion, groups=1, linear=False, reduction='mean'):
        super(MRL, self).__init__()
        self.criterion = criterion
        self.groups = groups
        self.linear = linear
        self.reduction = reduction
    
    
    def __call__(self, input, target):
        losses = []
        for g in range(self.groups):
            if self.linear:
                dimensions = input.shape[1] * (g+1) // self.groups
            else:
                dimensions = input.shape[1] // 2**g
            partial_input = F.normalize(input[:, :dimensions], dim=1)
            partial_target = F.normalize(target[:, :dimensions], dim=1)
            loss = self.criterion(partial_input, partial_target)
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
    mrl1 = MRL(criterion, groups=1)
    mrl2 = MRL(criterion, groups=2)
    
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