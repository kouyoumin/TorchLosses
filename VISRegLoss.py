import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VISRegLoss(nn.Module):

    def __init__(self, num_proj=256, scale_weight=1.0, shape_weight=1.0, center_weight=1.0):
        """
        Variance-Invariance-Sketching Regularization (VISReg) loss, from
        "VISReg: Variance-Invariance-Sketching Regularization for JEPA
        training" (arXiv:2606.02572).  Reference implementation:
        https://github.com/HaiyuWu/visreg/blob/main/visreg/losses/visreg.py

        Like SIGRegLoss, this pushes the embedding distribution toward an
        isotropic standard Gaussian using random 1D sketches, but it matches
        the *quantiles* of each sketch instead of its characteristic function.
        Three terms are summed:

            center: E[z]  ->  0        (mean of every channel is zeroed)
            scale:  Std[z] -> 1        (per-channel std is pulled to one)
            shape:  sort(z @ w)  ->  Gaussian order statistics

        The shape term is a sliced quantile (1D Wasserstein-2) distance: the
        projected values are sorted and compared against the theoretical
        standard-normal quantiles at q_i = i / (N + 1), i.e.

            target_i = sqrt(2) * erfinv(2 * q_i - 1) = Phi^-1(q_i)

        Sorting makes the shape term insensitive to the mean/scale of the
        projection, so the three terms stay largely decoupled (the scale
        normalization applied before projection is `detach`ed for the same
        reason - only the scale term drives the scale gradient).

        Compared with SIGRegLoss this is cheaper (one sort, no frequency
        grid) and its gradient does not vanish for far-from-Gaussian inputs,
        but it only constrains the marginals of the sketches rather than
        producing a proper goodness-of-fit test statistic.

        Takes a single (N, C) batch, matching SIGRegLoss/WeakSIGRegLoss. The
        reference implementation accepts (V, N, C) to regularize V JEPA
        prediction heads independently; since each view is handled separately
        and the terms are plain means, that is just V independent calls, so it
        is left to the caller.

        Args:
            num_proj (int, optional): Number of random unit-vector projection
                directions (sketches) sampled per forward pass. Defaults to 256.
            scale_weight (float, optional): Weight of the per-channel unit
                variance term. Defaults to 1.0.
            shape_weight (float, optional): Weight of the sorted-quantile
                Gaussian matching term. Defaults to 1.0.
            center_weight (float, optional): Weight of the zero-mean term.
                Defaults to 1.0.

        Returns:
            None
        """
        super().__init__()
        self.num_proj = num_proj
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight

        # Gaussian quantile targets, lazily built and cached per batch size.
        # Non-persistent: it is derived from N alone, nothing to checkpoint.
        self.register_buffer('target', None, persistent=False)


    def _get_target(self, num_samples, device, dtype):
        """
        Standard-normal quantiles at q_i = i / (num_samples + 1), cached
        across calls with the same batch size.

        Args:
            num_samples (int): Number of samples (order statistics) N.
            device (torch.device): Device to return the targets on.
            dtype (torch.dtype): Dtype to return the targets in.

        Returns:
            torch.Tensor: Ascending quantile targets of shape (num_samples,).
        """
        if self.target is None or self.target.numel() != num_samples:
            # erfinv is evaluated in float32 for accuracy near the tails,
            # then cast to the embedding dtype below (matters under AMP)
            q = torch.linspace(1, num_samples, num_samples, device=device, dtype=torch.float32) / (num_samples + 1)
            self.target = torch.erfinv(2.0 * q - 1.0).mul_(math.sqrt(2.0))

        return self.target.to(device=device, dtype=dtype)


    def __call__(self, embedding):
        """
        Args:
            embedding (torch.Tensor): Embedding tensor of shape (N, C).

        Returns:
            torch.Tensor: Scalar regularization loss.
        """
        N, C = embedding.shape
        device, dtype = embedding.device, embedding.dtype

        # Centering term: per-channel mean should be zero
        # mu shape: (1, C)
        mu = embedding.mean(dim=0, keepdim=True)
        center_loss = mu.square().mean()

        # Scale term: per-channel std should be one.
        # norm(dim=0) / sqrt(N) is the biased std of the centred values.
        # shape: (C,)
        centered = embedding - mu
        std = centered.norm(dim=0).div(math.sqrt(N)).clamp_min(1e-6)
        scale_loss = (std - 1.0).square().mean()

        # Normalize to unit variance before sketching so the shape term sees a
        # scale-free cloud. `detach` keeps the scale gradient in scale_loss
        # only, so the two terms cannot fight each other.
        normed = centered / std.detach()

        # Sample random projection directions and normalize to unit length
        # shape: (C, num_proj)
        W = F.normalize(torch.randn(C, self.num_proj, device=device, dtype=dtype), dim=0)

        # Project, then sort along the sample dimension to get the empirical
        # order statistics of each sketch; shape: (N, num_proj)
        proj_sorted = (normed @ W).sort(dim=0).values

        # Sliced quantile distance against the standard-normal order statistics
        target = self._get_target(N, device, dtype).view(N, 1)
        shape_loss = (proj_sorted - target).square().mean()

        return self.scale_weight * scale_loss + \
               self.shape_weight * shape_loss + \
               self.center_weight * center_loss


def test_visreg_loss():
    torch.manual_seed(0)
    loss_fn = VISRegLoss(num_proj=256)

    # Standard Gaussian samples should yield near-zero loss (residual is the
    # O(1/N) variance of the order statistics themselves)
    gaussian = torch.randn(512, 128)
    gaussian_loss = loss_fn(gaussian)
    print(f'Gaussian loss:     {gaussian_loss.item():.4f}  (expect ~0)')

    # Uniform samples are non-Gaussian and should yield higher loss
    uniform = (torch.rand(512, 128) - 0.5) * 2.0
    uniform_loss = loss_fn(uniform)
    print(f'Uniform loss:      {uniform_loss.item():.4f}  (expect > Gaussian)')

    # Shifted Gaussian is penalized by the centering term
    shifted = torch.randn(512, 128) + 2.0
    shifted_loss = loss_fn(shifted)
    print(f'Shifted loss:      {shifted_loss.item():.4f}  (expect > Gaussian)')

    # Scaled Gaussian is penalized by the scale term
    scaled = torch.randn(512, 128) * 3.0
    scaled_loss = loss_fn(scaled)
    print(f'Scaled loss:       {scaled_loss.item():.4f}  (expect > Gaussian)')

    # Constant (degenerate) distribution should yield very high loss
    constant = torch.ones(512, 128)
    constant_loss = loss_fn(constant)
    print(f'Constant loss:     {constant_loss.item():.4f}  (expect >> Gaussian)')

    # Rank-collapsed embedding (every channel a copy of the same direction)
    # is Gaussian per-channel but not isotropic: only the shape term catches it
    collapsed = torch.randn(512, 1).expand(-1, 128).contiguous()
    collapsed_loss = loss_fn(collapsed)
    print(f'Rank-1 loss:       {collapsed_loss.item():.4f}  (expect > Gaussian)')

    assert uniform_loss > gaussian_loss, "Uniform should score higher loss than Gaussian"
    assert shifted_loss > gaussian_loss, "Shifted should score higher loss than Gaussian"
    assert scaled_loss > gaussian_loss, "Scaled should score higher loss than Gaussian"
    assert constant_loss > gaussian_loss, "Constant should score higher loss than Gaussian"
    assert collapsed_loss > gaussian_loss, "Rank-1 should score higher loss than Gaussian"
    print("VISRegLoss test passed.")


def test_visreg_loss_grad():
    loss_fn = VISRegLoss(num_proj=256)

    # Gradients must reach the input
    embedding = torch.randn(64, 32, requires_grad=True)
    loss_fn(embedding).backward()
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all(), "Gradient must be finite"
    print('Gradient:          finite, reaches the input')

    # Batch size may vary between calls: the quantile target cache is keyed on N
    for n in (128, 256, 128):
        loss = loss_fn(torch.randn(n, 32))
        assert loss_fn.target.numel() == n, "quantile target cache did not follow the batch size"
        print(f'N={n:<4}             loss {loss.item():.4f}, cached target {tuple(loss_fn.target.shape)}')

    # Anything other than (N, C) is a caller error, not something to broadcast
    try:
        loss_fn(torch.randn(2, 64, 32))
        print("SHAPE CHECK FAILED: 3D input should not be accepted")
        raise AssertionError
    except ValueError:
        print('3D input:          rejected (unpack of (N, C) fails)')
    print("VISRegLoss gradient/shape test passed.")


if __name__ == '__main__':
    test_visreg_loss()
    test_visreg_loss_grad()
