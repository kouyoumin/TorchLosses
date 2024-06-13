import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    
    def __init__(self,
                feature_norm=True,
                temperature=1.0):
        """Compute loss for model.

        Args:
            feature_norm: whether or not to use normalization on the feature vector.
            temperature: a `floating` number for temperature scaling.

        Returns:
            A loss scalar.
        """
        self.feature_norm = feature_norm
        self.temperature = temperature


    def __call__(self, feature1, feature2):
        # Get (normalized) feature1 and feature2.
        if self.feature_norm:
            f1 = F.normalize(feature1, dim=1)
            f2 = F.normalize(feature2, dim=1)
        else:
            f1 = feature1
            f2 = feature2

        batch_size = f1.shape[0]

        # Create labels.
        labels = torch.eye(batch_size)#F.one_hot(torch.arange(batch_size), batch_size)

        logits_ab = torch.matmul(f1, f2.T) / self.temperature
        logits_ba = torch.matmul(f2, f1.T) / self.temperature

        loss_a = F.cross_entropy(logits_ab, labels)
        loss_b = F.cross_entropy(logits_ba, labels)
        loss = (loss_a + loss_b)/2

        return loss


def test_contrastive_loss():
    # Create instance of ContrastiveLoss
    loss_fn = ContrastiveLoss()

    # Create dummy input features
    feature1 = torch.randn((4, 10))
    feature2 = torch.randn((4, 10))

    # Compute the loss
    loss = loss_fn(feature1, feature2)

    # Compute the expected loss
    f1 = F.normalize(feature1, dim=1)
    f2 = F.normalize(feature2, dim=1)
    batch_size = f1.shape[0]
    labels = torch.eye(batch_size)
    logits_ab = torch.matmul(f1, f2.T) / loss_fn.temperature
    logits_ba = torch.matmul(f2, f1.T) / loss_fn.temperature
    loss_a = F.cross_entropy(logits_ab, labels)
    loss_b = F.cross_entropy(logits_ba, labels)
    expected_loss = (loss_a + loss_b) / 2

    # Check if the computed loss matches the expected loss
    assert torch.allclose(loss, expected_loss)


if __name__ == "__main__":
    test_contrastive_loss()
    print("ContrastiveLoss test passed.")
