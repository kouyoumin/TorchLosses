import torch
import torch.nn as nn

class IntersectionLoss(nn.Module):
    """
    IntersectionLoss
    """

    def __init__(self, dim=(2,3), eps=0., exp=1., reduction='none', verbose=False):
        super(IntersectionLoss, self).__init__()
        if verbose:
            print('IntersectionLoss dim:', dim)
            print('IntersectionLoss reduction:', reduction)
        self.dim = dim
        self.eps = eps
        self.exp = exp
        self.reduction = reduction
        

    def __call__(self, mask_obj, mask_base):
        th_obj = mask_obj.sigmoid()
        th_base = mask_base.sigmoid()
        
        intersection_p = torch.sum(th_obj * th_base + self.eps, self.dim)
        obj_p = torch.sum(th_obj + self.eps, self.dim)# + 1e-8

        inter_loss = ((1 - intersection_p / obj_p).clamp_(0,1))

        inter_loss = inter_loss ** self.exp
        
        if self.reduction == 'none':
            final_loss = inter_loss
        elif self.reduction == 'mean':
            final_loss = inter_loss.mean()
        elif self.reduction == 'sum':
            final_loss = inter_loss.sum()
        else:
            final_loss = inter_loss
        
        return final_loss
