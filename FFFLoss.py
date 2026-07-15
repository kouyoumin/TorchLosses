import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFFLoss(nn.Module):
    """Multi-positive sigmoid loss from "FFF: Fixing Flawed Foundations in
    contrastive pre-training" (arXiv:2405.10286, CVPR 2024).

    Combines batch text augmentation (k captions per image as true positives)
    with on-the-fly false-negative re-assignment and a SigLIP-style sigmoid
    loss that supports an arbitrary number of positives per sample.
    """

    def __init__(self,
                normalize=True,
                # Defaults below were calibrated on the M2R_FRONTAL training
                # set (frozen 202606221403/epoch25, full-set similarity stats
                # + initial-loss minimization). Paper (arXiv:2405.10286) /
                # SigLIP values are noted per argument.
                logit_scale=4.19,   # paper/SigLIP init: 10
                logit_bias=-4.18,   # paper/SigLIP init: -10
                p1=0.768,           # paper: 0.27
                p1_prime=0.712,     # paper: 0.24
                p2=0.848,           # paper: 0.92
                p3=0.9,             # paper: 0.99
                p_dup=0.995,        # not in paper (exact-duplicate extension)
                reduction='mean'):
        """
        Args:
            normalize: L2-normalize features before computing similarity.
            logit_scale: Initial value of the learnable logit scale (1/tau).
                SigLIP initializes this to 10.
            logit_bias: Initial value of the learnable bias beta. SigLIP
                initializes this to -10 so training starts from a low loss
                given the negative-dominated label matrix.
            p1: Image-text similarity threshold for re-assigning a negative
                pair as positive. Set to 1.0 to disable.
            p1_prime: Weaker image-text threshold that gates the text-text
                re-assignment (p1_prime < p1). Set to 1.0 to disable.
            p2: Image-image similarity threshold. Set to 1.0 to disable.
            p3: Text-text similarity threshold. Set to 1.0 to disable.
            p_dup: Caption-level exact-duplicate threshold. Any caption pair
                with cosine similarity above this is marked positive without
                the p1_prime image-text gate (identical captions, e.g.
                'No abnormal findings', are trustworthy matches even before
                the image tower has converged). Set to 1.0 to disable.
            reduction: 'mean' | 'sum' | 'none'. 'mean' follows the paper
                (sum over all pairs divided by N_txt); 'none' returns
                per-image losses of shape (B,) whose mean equals 'mean'.
        """
        super().__init__()
        self.normalize = normalize
        self.log_logit_scale = nn.Parameter(torch.tensor(float(logit_scale)).log())
        self.logit_bias = nn.Parameter(torch.tensor(float(logit_bias)))
        self.p1 = p1
        self.p1_prime = p1_prime
        self.p2 = p2
        self.p3 = p3
        self.p_dup = p_dup
        self.reduction = reduction

    @property
    def logit_scale(self):
        return self.log_logit_scale.exp()

    @property
    def temperature(self):
        return 1. / self.logit_scale

    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (B, D) image embeddings.
            text_features: (B, k, D) text embeddings, k captions per image
                (all treated as positives of their image). (B, D) is accepted
                and treated as k=1.
        """
        if text_features.ndim == 2:
            text_features = text_features.unsqueeze(1)
        B, k, D = text_features.shape

        if self.normalize:
            img = F.normalize(image_features, dim=-1)
            txt = F.normalize(text_features, dim=-1)
        else:
            img = image_features
            txt = text_features

        txt_flat = txt.reshape(B * k, D)  # caption j of image i at index i*k+j

        s_it = torch.matmul(img, txt_flat.T)  # (B, B*k)
        logits = s_it * self.log_logit_scale.exp() + self.logit_bias

        with torch.no_grad():
            # Base positives: captions belonging to the image
            m = torch.zeros(B, B * k, dtype=torch.bool, device=img.device)
            rows = torch.arange(B, device=img.device).repeat_interleave(k)
            cols = torch.arange(B * k, device=img.device)
            m[rows, cols] = True

            # Re-assign false negatives (Eq. 1 of the paper)
            if self.p1 < 1.0:
                m |= s_it > self.p1
            if self.p2 < 1.0:
                s_ii = torch.matmul(img, img.T).repeat_interleave(k, dim=1)  # (B, B*k)
                m |= s_ii > self.p2
            if self.p3 < 1.0:
                # Mean similarity between the k captions of image i and each
                # caption j; dot product is linear, so this equals the dot
                # product with the (unnormalized) mean caption embedding.
                s_tt = torch.matmul(txt.mean(dim=1), txt_flat.T)  # (B, B*k)
                m |= (s_tt > self.p3) & (s_it > self.p1_prime)
            if self.p_dup < 1.0:
                # Exact-duplicate captions are positives regardless of the
                # image-text gate: image i matches caption j if any of image
                # i's own captions duplicates caption j.
                s_cc = torch.matmul(txt_flat, txt_flat.T)  # (B*k, B*k)
                m |= (s_cc > self.p_dup).view(B, k, B * k).any(dim=1)

            m = m.float() * 2 - 1  # {-1, +1}

        loss = -F.logsigmoid(m * logits)  # (B, B*k)

        if self.reduction == 'mean':
            return loss.sum() / (B * k)
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none': per-image, mean of these equals 'mean' reduction
            return loss.sum(dim=1) / k


def test_fff_loss():
    torch.manual_seed(0)
    B, k, D = 4, 3, 16
    img = torch.randn(B, D)
    txt = torch.randn(B, k, D)

    # k=1, all fixing terms disabled -> plain SigLIP sigmoid loss
    loss_fn = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0, p_dup=1.0)
    loss = loss_fn(img, txt[:, 0])
    f_i = F.normalize(img, dim=-1)
    f_t = F.normalize(txt[:, 0], dim=-1)
    logits = f_i @ f_t.T * loss_fn.log_logit_scale.exp() + loss_fn.logit_bias
    labels = 2 * torch.eye(B) - 1
    expected = -F.logsigmoid(labels * logits).sum() / B
    assert torch.allclose(loss, expected), f"{loss} != {expected}"

    # k=3: base positive mask must cover exactly columns i*k..(i+1)*k of row i
    loss_fn3 = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0, p_dup=1.0)
    loss3 = loss_fn3(img, txt)
    f_t3 = F.normalize(txt, dim=-1).reshape(B * k, D)
    logits3 = f_i @ f_t3.T * loss_fn3.log_logit_scale.exp() + loss_fn3.logit_bias
    m = -torch.ones(B, B * k)
    for i in range(B):
        m[i, i * k:(i + 1) * k] = 1
    expected3 = -F.logsigmoid(m * logits3).sum() / (B * k)
    assert torch.allclose(loss3, expected3), f"{loss3} != {expected3}"

    # 'none' reduction: mean over images equals 'mean' reduction
    loss_fn_none = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0, p_dup=1.0, reduction='none')
    per_image = loss_fn_none(img, txt)
    assert per_image.shape == (B,)
    assert torch.allclose(per_image.mean(), loss3), f"{per_image.mean()} != {loss3}"

    # Re-assignment: duplicate images/captions get marked positive
    img_dup = img.clone()
    img_dup[1] = img[0]
    txt_dup = txt.clone()
    txt_dup[1] = txt[0]
    loss_fix = FFFLoss()(img_dup, txt_dup)
    assert torch.isfinite(loss_fix)

    # Exact-duplicate rule: a shared caption (e.g. 'No abnormal findings')
    # becomes a positive for both images even when the image-text gate
    # (p1_prime) would block the text-text rule
    txt_shared = txt.clone()
    txt_shared[1, 0] = txt[0, 0]  # image 1's caption 0 duplicates image 0's caption 0
    dup_only = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0, p_dup=0.999)
    no_dup = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0, p_dup=1.0)
    no_dup.load_state_dict(dup_only.state_dict())
    l_dup = dup_only(img, txt_shared)
    l_no = no_dup(img, txt_shared)
    assert not torch.allclose(l_dup, l_no), "duplicate rule had no effect"
    # verify the flipped entries are exactly (row 0, col k+0) and (row 1, col 0)
    f_ts = F.normalize(txt_shared, dim=-1).reshape(B * k, D)
    logits_ts = f_i @ f_ts.T * no_dup.log_logit_scale.exp() + no_dup.logit_bias
    m_ts = -torch.ones(B, B * k)
    for i in range(B):
        m_ts[i, i * k:(i + 1) * k] = 1
    m_ts[0, k] = 1  # image 0 <-> image 1's caption 0 (the duplicate)
    m_ts[1, 0] = 1  # image 1 <-> image 0's caption 0
    expected_dup = -F.logsigmoid(m_ts * logits_ts).sum() / (B * k)
    assert torch.allclose(l_dup, expected_dup), f"{l_dup} != {expected_dup}"

    # Gradients flow to features and to the learnable scale/bias
    img_g = img.clone().requires_grad_(True)
    txt_g = txt.clone().requires_grad_(True)
    loss_fn_g = FFFLoss()
    loss_fn_g(img_g, txt_g).backward()
    assert img_g.grad is not None and txt_g.grad is not None
    assert loss_fn_g.log_logit_scale.grad is not None
    assert loss_fn_g.logit_bias.grad is not None

    print("k=1 siglip:", loss.item())
    print("k=3 multi-positive:", loss3.item())


if __name__ == "__main__":
    test_fff_loss()
    print("FFFLoss test passed.")
