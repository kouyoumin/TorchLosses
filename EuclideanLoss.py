import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanLoss(nn.Module):
    def __init__(self, normalize:bool=False, reduction:str='mean'):
        """
        Initializes a new instance of the `EuclideanLoss` class.

        Args:
            normalize (bool, optional): Whether to normalize the input tensors. Defaults to False.
            reduction (str, optional): The reduction method to apply to the loss. Possible values are 'none', 'mean', and 'sum'. Defaults to 'mean'.

        Returns:
            None
        """
        self.normalize = normalize
        self.reduction = reduction
    
    
    def __call__(self, tensor1, tensor2):
        if self.normalize:
            tensor1 = F.normalize(tensor1, dim=1)
            tensor2 = F.normalize(tensor2, dim=1)
        dist = (tensor1 - tensor2).pow(2).sum(tuple(range(1,tensor1.ndim))).sqrt()
        if self.reduction == 'mean':
            return dist.mean()
        elif self.reduction == 'sum':
            return dist.sum()
        elif self.reduction == 'none':
            return dist
        else:
            return dist.mean()


def test_euclidean_loss():
    loss_fn = EuclideanLoss()
    tensor1 = torch.tensor([[0, 0]])
    tensor2 = torch.tensor([[3, 4]])
    loss = loss_fn(tensor1, tensor2)
    assert(loss.item() == 5.0)


if __name__ == '__main__':
    test_euclidean_loss()
