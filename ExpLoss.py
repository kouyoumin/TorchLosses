import torch
import torch.nn as nn


class Exp():
    def __init__(self, exp=2.0, normalize=True, reduction='mean'):
        self.exp = exp
        self.normalize = normalize
        self.reduction = reduction
    
    
    def __call__(self, loss):
        exploss = loss ** self.exp
        if self.normalize:
            exploss = exploss * loss.sum() / exploss.sum()
        
        if self.reduction == 'none':
            return exploss
        elif self.reduction == 'mean':
            return exploss.mean()
        elif self.reduction == 'sum':
            return exploss.sum()
        else:
            return exploss.mean()
        

class ExpLoss(Exp):
    def __init__(self, criterion, exp=2.0, normalize=False, reduction='mean'):
        super().__init__(exp=exp, normalize=normalize, reduction=reduction)
        self.criterion = criterion
    
    
    def __call__(self, *args):
        loss = self.criterion(*args)
        return super(ExpLoss, self).__call__(loss)


def test_exploss():
    criterion = nn.L1Loss(reduction='none')
    exploss = ExpLoss(criterion, exp=2.0, normalize=True, reduction='none')
    
    input = torch.zeros(8)
    target = torch.zeros(8)
    target[:4] = 0.5
    target[4:8] = 1
    
    l1loss = criterion(input, target)
    exploss = exploss(input, target)
    
    assert(l1loss.sum().item() == exploss.sum().item())
    assert((l1loss[-1]/l1loss[0]) ** 2 == (exploss[-1]/exploss[0]))


if __name__ == '__main__':
    test_exploss()