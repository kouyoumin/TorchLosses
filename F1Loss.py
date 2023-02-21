import torch
import torch.nn as nn


class F1Loss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
        super(F1Loss, self).__init__()
        self.reduction = reduction
        self.eps = eps
    

    def __call__(self, output, target):
    
        tp = (target * output).sum(0)
        fp = ((1 - target) * output).sum(0)
        fn = (target * (1 - output)).sum(0)
        p = tp / (tp + fp + self.eps)
        r = tp / (tp + fn + self.eps)

        f1 = 2 * p * r / (p + r + self.eps)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        if self.reduction == 'mean':
            return 1 - f1.mean()
        elif self.reduction == 'sum':
            return (1-f1).sum()
        else:
            return 1 - f1


def test():
    target = torch.zeros((20, 2))
    target[18:] = 1
    pred = torch.rand((20,2))
    criterion = F1Loss(reduction='mean')
    loss = criterion(target, pred)
    print(loss.item())


if __name__ == '__main__':
    test()

