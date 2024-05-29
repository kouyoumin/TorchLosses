from typing import Sequence, Union, Literal

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    DiceLoss
    """

    def __init__(self, dim=(2,3), pos_weight:Union[torch.Tensor, float]=torch.tensor(1.), neg_weight:Union[torch.Tensor, float]=torch.tensor(1.), class_weight:Union[torch.Tensor, Sequence]=None, mode:Literal['pos_only', 'pos_or_neg', 'both', 'neg_only']='pos_only', preprocess=nn.Identity(), ignore_empty=True, ignore_full=True, eps=1e-8, exp=1., reduction='none', verbose=False):
        """
        Constructor for DiceLoss
        
        Parameters
        ----------
        dim : tuple, optional
            Dimensions to reduce, by default (2,3)
        pos_weight : torch.Tensor or float, optional
            Positive weight, by default 1.
        neg_weight : torch.Tensor of float, optional
            Negative weight, by default 0.
        class_weight : torch.Tensor, optional
            Class weight, by default None
        mode : str, optional
            Mode of operation, by default 'pos_only'
        preprocess : nn.Module, optional
            Preprocessing layer, by default nn.Identity()
            Set to nn.Sigmoid() if your model outputs logits
        ignore_empty : bool, optional
            Ignore empty masks, by default True
        ignore_full : bool, optional
            Ignore full masks, by default True
        eps : float, optional
            Epsilon, by default 0.
        exp : float, optional
            Exponent, by default 1.
        reduction : str, optional
            Reduction method, by default 'none'
        verbose : bool, optional
            Print debug information, by default False
        """
        super(DiceLoss, self).__init__()
        if verbose:
            print('DiceLoss dim:', dim)
            print('DiceLoss pos_weight:', pos_weight)
            print('DiceLoss neg_weight:', neg_weight)
            print('DiceLoss class_weight:', class_weight)
            print('DiceLoss mode:', mode)
            print('DiceLoss reduction:', reduction)
        self.dim = dim
        self.pos_weight = pos_weight
        if not isinstance(self.pos_weight, torch.Tensor):
            self.pos_weight = torch.tensor(self.pos_weight)
        self.neg_weight = neg_weight
        if not isinstance(self.neg_weight, torch.Tensor):
            self.neg_weight = torch.tensor(self.neg_weight)
        self.class_weight = class_weight
        if isinstance(self.class_weight, type(None)):
            self.class_weight = torch.tensor(1.)
        elif isinstance(self.class_weight, Sequence):
            self.class_weight = torch.Tensor(self.class_weight)
        if 0 not in self.dim and self.class_weight.ndim == 1:
            self.class_weight = self.class_weight.unsqueeze(0)
        self.mode = mode
        self.preprocess = preprocess
        self.ignore_empty = ignore_empty
        self.ignore_full = ignore_full
        self.eps = eps
        self.exp = exp
        self.reduction = reduction
        

    def __call__(self, input, target):
        th_input = self.preprocess(input)
        if isinstance(self.preprocess, nn.Sigmoid):
            assert(input.min()<0)
        assert(th_input.max()<=1)
        assert(th_input.min()>=0)
        th_target = target
        
        intersection_p = torch.sum(th_input * th_target, self.dim)
        sum_p = torch.sum(th_input + th_target, self.dim) + self.eps

        if self.mode != 'pos_only':
            intersection_n = torch.sum((1-th_input) * (1-th_target), self.dim)
            sum_n = torch.sum((1-th_input) + (1-th_target), self.dim) + self.eps
        
        if self.mode == 'pos_or_neg':
            pos = torch.sum(th_target, self.dim).clamp_(0,1)
            dice_loss = self.pos_weight * self.class_weight * pos * ((1 - 2 * intersection_p / sum_p).clamp_(0,1)) + self.neg_weight * self.class_weight * (1-pos) * ((1 - 2 * intersection_n / sum_n).clamp_(0,1))
        elif self.mode == 'both':
            #class_weight = class_weight * (torch.amax(th_target, self.dim) > 0.5).float()
            #class_weight[class_weight == 0] = 1
            pos = torch.sum(th_target, self.dim).clamp_(0,1) if self.ignore_empty else torch.ones_like(intersection_p)
            neg = 1-torch.sum(th_target, self.dim).clamp_(0,1) if self.ignore_full else torch.ones_like(intersection_n)
            #dice_loss = torch.zeros(pos.shape).cuda()
            dice_loss = self.pos_weight * self.class_weight * pos * ((1 - 2 * intersection_p / sum_p).clamp_(0,1)) + self.neg_weight * self.class_weight * neg * ((1 - 2 * intersection_n / sum_n).clamp_(0,1))
        elif self.mode == 'neg_only':
            #class_weight = class_weight * (torch.amax(th_target, self.dim) > 0.5).float()
            #class_weight[class_weight == 0] = 1
            #assert(class_weight.shape == th_target.shape[:2]) # apply only to positive samples
            neg = 1-torch.sum(th_target, self.dim).clamp_(0,1) if self.ignore_full else torch.ones_like(intersection_n)
            dice_loss = self.class_weight * neg * ((1 - 2 * intersection_n / sum_n).clamp_(0,1))
            #dice_loss = class_weight * ((1 - intersection_n / sum_n).clamp_(0,1))
        else:
            # pos_only
            #class_weight = self.class_weight * (torch.amax(th_target, self.dim) > 0.5).float()
            #class_weight[class_weight == 0] = 1
            pos = torch.sum(th_target, self.dim).clamp_(0,1) if self.ignore_empty else torch.ones_like(intersection_p)
            dice_loss = self.class_weight * pos * ((1 - 2 * intersection_p / sum_p).clamp_(0,1))
            #dice_loss = class_weight * ((1 - intersection_p / sum_p).clamp_(0,1))

        dice_loss = dice_loss ** self.exp
        
        if self.reduction == 'none':
            final_loss = dice_loss
        elif self.reduction == 'mean':
            final_loss = dice_loss.nanmean()
        elif self.reduction == 'sum':
            final_loss = dice_loss.nansum()
        else:
            final_loss = dice_loss
        
        return final_loss


def test_diceloss_pos_only_dim23():
    """
    Test DiceLoss with dim=(2,3)
    """
    loss = DiceLoss(dim=(2,3), mode='pos_only')
    input = torch.zeros(4,2,10,10)
    target = torch.zeros(4,2,10,10)
    input[0,1,0:5,0:5] = 1
    target[0,1,:,:] = 1
    output = loss(input, target)
    #print(output)
    assert(output.shape == torch.Size([4,2]))
    assert(output[0,1] == 0.6)


def test_diceloss_pos_or_neg_dim23():
    """
    Test DiceLoss with dim=(2,3)
    """
    loss = DiceLoss(dim=(2,3), mode='pos_or_neg')
    input = torch.zeros(4,2,10,10)
    target = torch.zeros(4,2,10,10)
    input[0,1,0:5,0:5] = 1
    input[1,0,0,0] = 1
    target[0,1,:,:] = 1
    output = loss(input, target)
    assert(output.shape == torch.Size([4,2]))
    assert(output[0,1] == 0.6)
    assert(torch.isclose(output[1,0], torch.tensor(1-2*99/199)))


def test_diceloss_pos_only_dim023():
    """
    Test DiceLoss with dim=(0,2,3)
    """
    loss = DiceLoss(dim=(0,2,3), mode='pos_only')
    input = torch.zeros(4,2,10,10)
    target = torch.zeros(4,2,10,10)
    input[0,1,0:5,0:5] = 1
    input[1,1,0:2,0:2] = 1
    target[0,1,:,:] = 1
    target[1,1,0:5,0:5] = 1
    output = loss(input, target)
    #print(output)
    assert(output.shape == torch.Size([2]))
    assert(torch.isclose(output[1], torch.tensor(1-2*(25+4)/(25+100+4+25))))


def test_diceloss_pos_only_class_weight_dim023():
    """
    Test DiceLoss with dim=(0,2,3)
    """
    loss = DiceLoss(dim=(0,2,3), class_weight=torch.Tensor([1,2]), mode='pos_only')
    input = torch.zeros(4,2,10,10)
    target = torch.zeros(4,2,10,10)
    input[0,1,0:5,0:5] = 1
    input[1,1,0:2,0:2] = 1
    target[0,1,:,:] = 1
    target[1,1,0:5,0:5] = 1
    output = loss(input, target)
    #print(output)
    assert(output.shape == torch.Size([2]))
    assert(torch.isclose(output[1], torch.tensor(2*(1-2*(25+4)/(25+100+4+25)))))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_diceloss_"):
            print(name)
            fn()
    print("All tests passed")