import torch
import math

class OHEMLoss():
    def __init__(self, criterion, reduction_dim=None, prop=0.1, verbose=False):
        if verbose:
            print('OHEMLoss criterion:', criterion)
            print('OHEMLoss reduction_dim:', reduction_dim)
            print('OHEMLoss prop:', prop)
        self.criterion = criterion
        self.reduction_dim = reduction_dim
        self.prop = prop
    

    def __call__(self, input, target):
        loss = self.criterion(input, target)
        if self.reduction_dim is not None:
            loss = torch.mean(loss, self.reduction_dim)
        topn = math.ceil(loss.numel() * self.prop)
        val, _ = torch.topk(loss.flatten(), topn)
        return val.mean()

def test():
    from IoULoss import IoULoss
    dummy_input = torch.randn(2, 6, 256, 256)
    dummy_target = torch.randn(2, 6, 256, 256)
    dummy_criterion = IoULoss(pos_weight=torch.ones(6), neg_weight=torch.ones(6), reduction='none')
    ohem = OHEMLoss(dummy_criterion)#, reduction_dim=(1,))
    loss = ohem(dummy_input, dummy_target)
    print(loss)


if __name__ == '__main__':
    test()
