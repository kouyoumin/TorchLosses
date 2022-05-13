import torch
import numpy as np
import math

class OHEMLoss():
    def __init__(self, criterion, reduction_dim=None, init_prop=1.0, final_prop=0.15, steps=1e8, verbose=False):
        if verbose:
            print('OHEMLoss criterion:', criterion)
            print('OHEMLoss reduction_dim:', reduction_dim)
            print('OHEMLoss init_prop:', init_prop)
            print('OHEMLoss final_prop:', final_prop)
            print('OHEMLoss steps:', steps)
        self.criterion = criterion
        self.reduction_dim = reduction_dim
        self.init_prop = min(init_prop, 1)
        self.final_prop = max(final_prop, 0)
        self.steps = steps
        self.curr_prop = init_prop
        self.curr_step = 0
    

    def __call__(self, input, target):
        loss = self.criterion(input, target)
        
        # Skip
        if self.curr_prop == 1:
            return loss.mean()
        
        if self.reduction_dim is not None:
            loss = torch.mean(loss, self.reduction_dim)
        
        topn = math.ceil(loss.numel() * self.curr_prop)
        val, _ = torch.topk(loss.flatten(), topn)
        return val.mean()
    

    def step(self):
        self.curr_step += 1
        if self.init_prop != self.final_prop:
            self.curr_prop = self.final_prop + (self.init_prop - self.final_prop) * math.cos(min(self.curr_step / self.steps, 1)*(math.pi / 2))
        print('OHEM: curr_step = %d/%d' % (self.curr_step, self.steps))
        print('OHEM: curr_prop =', self.curr_prop)

def test():
    from IoULoss import IoULoss
    steps = 50
    dummy_input = torch.randn(2, 6, 256, 256)
    dummy_target = torch.randn(2, 6, 256, 256)
    dummy_criterion = IoULoss(pos_weight=torch.ones(6), neg_weight=torch.ones(6), reduction='none')
    ohem = OHEMLoss(dummy_criterion, init_prop=1.0, final_prop=0.15, steps=steps, verbose=True)
    loss = ohem(dummy_input, dummy_target)
    print(loss)
    for i in range(steps+10):
        ohem.step()


if __name__ == '__main__':
    test()
