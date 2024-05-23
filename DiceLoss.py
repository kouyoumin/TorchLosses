import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    DiceLoss
    """

    def __init__(self, dim=(2,3), pos_weight=1., neg_weight=0., class_weight=None, mode='pos_only', preprocess=nn.Identity(), ignore_empty=True, ignore_full=True, eps=1e-8, exp=1., reduction='none', verbose=False):
        """
        Constructor for DiceLoss
        
        Parameters
        ----------
        dim : tuple, optional
            Dimensions to reduce, by default (2,3)
        pos_weight : float, optional
            Positive weight, by default 1.
        neg_weight : float, optional
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
        self.neg_weight = neg_weight
        self.class_weight = class_weight
        self.mode = mode
        self.preprocess = preprocess
        self.ignore_empty = ignore_empty
        self.ignore_full = ignore_full
        self.eps = eps
        self.exp = exp
        self.reduction = reduction
        

    def __call__(self, input, target):
        th_input = self.preprocess(input)
        th_target = target
        if isinstance(self.class_weight, torch.Tensor):
            class_weight = self.class_weight.reshape(1, target.shape[1])
        else:
            class_weight = torch.ones(1, target.shape[1])
        
        intersection_p = torch.sum(th_input * th_target + self.eps, self.dim)
        sum_p = torch.sum(th_input + th_target + self.eps, self.dim)# + 1e-8

        if self.mode != 'pos_only':
            intersection_n = torch.sum((1-th_input) * (1-th_target), self.dim)
            sum_n = torch.sum((1-th_input) + (1-th_target), self.dim)# + 1e-8
        
        if self.mode == 'pos_or_neg':
            pos = (torch.amax(th_target, self.dim) > 0.5).float()
            iou_loss = self.pos_weight.unsqueeze(0) * class_weight * pos * ((1 - 2 * intersection_p / sum_p).clamp_(0,1)) + self.neg_weight.unsqueeze(0) * class_weight * (1-pos) * ((1 - 2 * intersection_n / sum_n).clamp_(0,1))
        elif self.mode == 'both':
            class_weight = class_weight * (torch.amax(th_target, self.dim) > 0.5).float()
            class_weight[class_weight == 0] = 1
            pos = (torch.amax(th_target, self.dim) > 0.5).float() if self.ignore_empty else torch.ones_like(intersection_p)
            neg = (torch.amin(th_target, self.dim) < 0.5).float() if self.ignore_full else torch.ones_like(intersection_n)
            iou_loss = torch.zeros(pos.shape).cuda()
            iou_loss = self.pos_weight.unsqueeze(0) * class_weight * pos * ((1 - 2 * intersection_p / sum_p).clamp_(0,1)) + self.neg_weight.unsqueeze(0) * class_weight * neg * ((1 - 2 * intersection_n / sum_n).clamp_(0,1))
        elif self.mode == 'neg_only':
            class_weight = class_weight * (torch.amax(th_target, self.dim) > 0.5).float()
            class_weight[class_weight == 0] = 1
            #assert(class_weight.shape == th_target.shape[:2]) # apply only to positive samples
            neg = (torch.amin(th_target, self.dim) < 0.5).float() if self.ignore_full else torch.ones_like(intersection_n)
            iou_loss = class_weight * neg * ((1 - 2 * intersection_n / sum_n).clamp_(0,1))
            #iou_loss = class_weight * ((1 - intersection_n / sum_n).clamp_(0,1))
        else:
            # pos_only
            class_weight = class_weight * (torch.amax(th_target, self.dim) > 0.5).float()
            class_weight[class_weight == 0] = 1
            pos = (torch.amax(th_target, self.dim) > 0.5).float() if self.ignore_empty else torch.ones_like(intersection_p)
            iou_loss = class_weight * pos * ((1 - 2 * intersection_p / sum_p).clamp_(0,1))
            #iou_loss = class_weight * ((1 - intersection_p / sum_p).clamp_(0,1))

        iou_loss = iou_loss ** self.exp
        
        if self.reduction == 'none':
            final_loss = iou_loss
        elif self.reduction == 'mean':
            final_loss = iou_loss.mean()
        elif self.reduction == 'sum':
            final_loss = iou_loss.sum()
        else:
            final_loss = iou_loss
        
        return final_loss
