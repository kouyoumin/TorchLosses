import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):

    def __init__(self, knots=17, num_proj=256):
        """
        Sketched Isotropic Gaussian Regularization (SIGReg) loss, from LeJEPA.
        Reference implementation:
        https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
        The defaults below match it: 17 knots over t in [0, 3], 256 projections.

        Pushes the embedding distribution toward an isotropic standard Gaussian
        by comparing the empirical characteristic function of random 1D projections
        against the Gaussian characteristic function phi(t) = exp(-t^2 / 2).

        For X ~ N(0, 1):
            E[cos(tX)] = phi(t)    (real part of characteristic function)
            E[sin(tX)] = 0         (imaginary part, zero by symmetry)

        The loss integrates the squared deviation over the frequency domain [0, 3]
        using trapezoidal weights, averaged over multiple random projection directions.

        Args:
            knots (int, optional): Number of frequency integration points in [0, 3].
                More knots give a finer approximation of the integral. Defaults to 17.
            num_proj (int, optional): Number of random unit-vector projection
                directions sampled per forward pass. Defaults to 256.

        Returns:
            None
        """
        super().__init__()
        self.num_proj = num_proj

        # Frequency domain integration points t in [0, 3]
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        # Trapezoidal integration weights: 2*dt for interior, dt for endpoints
        trap_weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        trap_weights[[0, -1]] = dt

        # Gaussian characteristic function values at each frequency point
        phi = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        # Precompute trap_weights * phi so the matmul in forward computes the
        # phi-weighted integral in one step
        self.register_buffer("weights", trap_weights * phi)


    def __call__(self, embedding):
        """
        Args:
            embedding (torch.Tensor): Embedding tensor of shape (N, C).

        Returns:
            torch.Tensor: Scalar regularization loss.
        """
        N, C = embedding.shape
        device, dtype = embedding.device, embedding.dtype

        # Sample random projection directions and normalize to unit length
        # shape: (C, num_proj)
        A = torch.randn(C, self.num_proj, device=device, dtype=dtype)
        A.div_(A.norm(p=2, dim=0, keepdim=True))

        # Move buffers to the same device/dtype as the input (no-op if already matching)
        #t = self.t.to(device=device, dtype=dtype)
        #phi = self.phi.to(device=device, dtype=dtype)
        #weights = self.weights.to(device=device, dtype=dtype)
        self.to(device=device)

        # Project onto each direction, then broadcast with frequency points
        # (embedding @ A): (N, num_proj)
        # unsqueeze(-1) * t: (N, num_proj, knots)
        proj = (embedding @ A).unsqueeze(-1) * self.t

        # Empirical characteristic function vs. Gaussian target
        # Both mean(0) terms average over the batch dimension
        cos_err = proj.cos().mean(0) - self.phi    # shape: (num_proj, knots)
        sin_err = proj.sin().mean(0)                # should be ~0 for Gaussian
        err = cos_err.square() + sin_err.square()

        # phi-weighted integral over the frequency domain, scaled by batch size
        # to form a proper test statistic; shape: (num_proj,)
        statistic = (err @ self.weights) * N

        return statistic.mean()


class WeakSIGRegLoss(nn.Module):

    def __init__(self, num_proj=64):
        """
        Weak Sketched Isotropic Gaussian Regularization (Weak SIGReg) loss.

        Pushes the embedding covariance toward the identity matrix, matching
        only the 2nd moment (spherical cloud).  Cheaper than SIGRegLoss, which
        matches all moments via the empirical characteristic function.

        For X ~ N(0, I):
            Cov(X A) = Aᵀ A ≈ I    (A scaled so E[Aᵀ A] = I)

        A random Gaussian sketch A ∈ R^{C × num_proj} (scaled by 1/√C) is
        drawn each forward pass.  The loss is the Frobenius norm distance of
        the empirical covariance of the projected embeddings from the identity:

            L = ‖Cov(embedding · A) − I‖_F

        Args:
            num_proj (int, optional): Number of random sketch projection
                directions sampled per forward pass. Defaults to 64.

        Returns:
            None
        """
        super().__init__()
        self.num_proj = num_proj


    def __call__(self, embedding):
        """
        Args:
            embedding (torch.Tensor): Embedding tensor of shape (N, C).

        Returns:
            torch.Tensor: Scalar regularization loss.
        """
        N, C = embedding.shape
        device, dtype = embedding.device, embedding.dtype

        # Random sketch matrix scaled by 1/√C; shape: (C, num_proj)
        A = torch.randn(C, self.num_proj, device=device, dtype=dtype) / (C ** 0.5)

        # Project onto each direction and centre; shape: (N, num_proj)
        proj = embedding @ A
        proj = proj - proj.mean(dim=0, keepdim=True)

        # Empirical covariance; shape: (num_proj, num_proj)
        cov = (proj.T @ proj) / (N - 1 + 1e-6)

        return torch.norm(cov - torch.eye(self.num_proj, device=device, dtype=dtype), p='fro')


def test_sigreg_loss():
    loss_fn = SIGRegLoss(knots=17, num_proj=256)

    # Standard Gaussian samples should yield near-zero loss
    gaussian = torch.randn(512, 128)
    gaussian_loss = loss_fn(gaussian)
    print(f'Gaussian loss:     {gaussian_loss.item():.4f}  (expect ~0)')

    # Uniform samples are non-Gaussian and should yield higher loss
    uniform = (torch.rand(512, 128) - 0.5) * 2.0
    uniform_loss = loss_fn(uniform)
    print(f'Uniform loss:      {uniform_loss.item():.4f}  (expect > Gaussian)')

    # Constant (degenerate) distribution should yield very high loss
    constant = torch.ones(512, 128)
    constant_loss = loss_fn(constant)
    print(f'Constant loss:     {constant_loss.item():.4f}  (expect >> Gaussian)')

    assert uniform_loss > gaussian_loss, "Uniform should score higher loss than Gaussian"
    assert constant_loss > gaussian_loss, "Constant should score higher loss than Gaussian"
    print("SIGRegLoss test passed.")


def test_weak_sigreg_loss():
    # C=512 >> num_proj=64 keeps sketch noise O(num_proj/√C) small
    loss_fn = WeakSIGRegLoss(num_proj=64)

    # Standard Gaussian samples should yield near-zero loss
    gaussian = torch.randn(512, 512)
    gaussian_loss = loss_fn(gaussian)
    print(f'Gaussian loss:     {gaussian_loss.item():.4f}  (expect ~0)')

    # Scaled Gaussian (cov ≈ 9I) should yield higher loss
    scaled = torch.randn(512, 512) * 3.0
    scaled_loss = loss_fn(scaled)
    print(f'Scaled loss:       {scaled_loss.item():.4f}  (expect > Gaussian)')

    # Constant (degenerate) distribution should yield higher loss
    constant = torch.ones(512, 512)
    constant_loss = loss_fn(constant)
    print(f'Constant loss:     {constant_loss.item():.4f}  (expect > Gaussian)')

    assert scaled_loss > gaussian_loss, "Scaled should score higher loss than Gaussian"
    assert constant_loss > gaussian_loss, "Constant should score higher loss than Gaussian"
    print("WeakSIGRegLoss test passed.")


if __name__ == '__main__':
    test_sigreg_loss()
    test_weak_sigreg_loss()
