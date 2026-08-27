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
                # logit scale/bias were calibrated on the M2R_FRONTAL training
                # set (frozen 202606221403/epoch25, duplicate-only M,
                # initial-loss minimization). Paper (arXiv:2405.10286) /
                # SigLIP values are noted per argument.
                logit_scale=4.35,   # paper/SigLIP init: 10
                logit_bias=-4.24,   # paper/SigLIP init: -10
                # Similarity-threshold re-assignment is DISABLED by default
                # (1.0). Two independent reasons, either of which is sufficient:
                #
                # 1. The paper computes these matrices with a FROZEN pretrained
                #    model; computing them online from the model being trained
                #    creates a positive feedback loop (drift -> more
                #    re-assignments -> collapse). Pass ref_image_features /
                #    ref_text_features from a frozen or EMA reference.
                # 2. On M2R_FRONTAL/epoch25 the thresholds have no usable
                #    operating point at all, frozen reference or not.
                #    FFFThresholdCalibration.py over 43.3M candidate pairs found
                #    no threshold reaching 90% precision for any of p1/p2/p3: on
                #    the residual that dup_mask does not already cover, the mean
                #    positive similarity sits at or below the negatives' 99th
                #    percentile. p1 and p2 could not even recover the
                #    string-identical pairs.
                #
                # Paper values (p1=0.27, p1'=0.24, p2=0.92, p3=0.99) are an
                # operating point on a different encoder and do not transfer --
                # re-calibrate per checkpoint rather than copying any number.
                p1=1.0,
                p1_prime=1.0,
                p2=1.0,
                p3=1.0,
                p_dup=1.0,          # embedding-based duplicates (superseded by dup_mask)
                soft_dup_weight=0.0,
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
            p_dup: Caption-level exact-duplicate threshold on embedding
                similarity. Ignored when a `dup_mask` is passed to forward()
                (the preferred, collapse-proof way: string equality computed
                by the caller). Set to 1.0 to disable.
            soft_dup_weight: how much of a NEGATIVE a soft duplicate still
                counts as, in [0, 1] -- the same knob as
                ContrastiveLoss.soft_dup_weight, so one CLI flag can drive
                both. Soft duplicates are marked by `soft_dup_mask` in
                forward() and are meant for related-but-not-identical pairs
                (e.g. same abnormal findings, different normal ones). This
                loss is a sigmoid/BCE loss, not a softmax one, so a partial
                negative is expressed as a partial LABEL: target = 1 - w.
                w = 0 therefore reproduces the hard `dup_mask` behaviour
                (target 1, a full positive) and w = 1 leaves the pair an
                ordinary negative (target 0).
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
        self.soft_dup_weight = float(soft_dup_weight)
        self.reduction = reduction

    @property
    def logit_scale(self):
        return self.log_logit_scale.exp()

    @property
    def temperature(self):
        return 1. / self.logit_scale

    def forward(self, image_features, text_features, dup_mask=None,
                ref_image_features=None, ref_text_features=None,
                soft_dup_mask=None):
        """
        Args:
            image_features: (B, D) image embeddings.
            text_features: (B, k, D) text embeddings, k captions per image
                (all treated as positives of their image). (B, D) is accepted
                and treated as k=1.
            dup_mask: optional (B, B*k) bool tensor; True where caption j is
                an exact duplicate (string equality) of any caption of image
                i. Marked positive unconditionally. Preferred over p_dup
                because it does not depend on the embeddings being trained.
            soft_dup_mask: optional (B, B*k) bool tensor of PARTIAL duplicates,
                given the target 1 - soft_dup_weight instead of 0. Entries that
                are already positive (the image's own captions, dup_mask, a
                threshold re-assignment) keep target 1 -- a soft mark can only
                lift a negative part-way, never demote a positive.
            ref_image_features, ref_text_features: optional embeddings from a
                reference model that is NOT the weights being trained (same
                shapes as image_features/text_features), used instead of the
                trainable embeddings to compute the p1/p1_prime/p2/p3
                similarity-threshold masks. Required to safely enable those
                thresholds (see class docstring) — pass both or neither. The
                sigmoid loss itself still uses the trainable image_features/
                text_features.
                Either a frozen copy or a slow-moving average (EMA) works: the
                requirement is that the reference cannot respond to this step's
                gradient, which is what closes the drift -> re-assignment ->
                collapse loop. An EMA lags by roughly 1/(1-decay) steps, so it
                keeps improving with training instead of going stale, but it
                gives weaker protection than a frozen copy early on, when the
                average is still close to the live weights.
                Both must come from the SAME snapshot — an image tower and a
                text tower of different ages are not in a shared embedding
                space, and p1/p1_prime compare one against the other.
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

        use_ref = ref_image_features is not None and ref_text_features is not None
        if use_ref:
            with torch.no_grad():
                ref_img = F.normalize(ref_image_features, dim=-1) if self.normalize else ref_image_features
                ref_txt = ref_text_features
                if ref_txt.ndim == 2:
                    ref_txt = ref_txt.unsqueeze(1)
                ref_txt = F.normalize(ref_txt, dim=-1) if self.normalize else ref_txt
                ref_txt_flat = ref_txt.reshape(B * k, D)
        else:
            ref_img, ref_txt, ref_txt_flat = img, txt, txt_flat

        with torch.no_grad():
            # Base positives: captions belonging to the image
            m = torch.zeros(B, B * k, dtype=torch.bool, device=img.device)
            rows = torch.arange(B, device=img.device).repeat_interleave(k)
            cols = torch.arange(B * k, device=img.device)
            m[rows, cols] = True

            # Re-assign false negatives (Eq. 1 of the paper). Uses the frozen
            # reference embeddings when provided, never the trainable ones
            # directly — see the class docstring for why.
            ref_s_it = torch.matmul(ref_img, ref_txt_flat.T) if use_ref else s_it
            if self.p1 < 1.0:
                m |= ref_s_it > self.p1
            if self.p2 < 1.0:
                s_ii = torch.matmul(ref_img, ref_img.T).repeat_interleave(k, dim=1)  # (B, B*k)
                m |= s_ii > self.p2
            if self.p3 < 1.0:
                # Mean similarity between the k captions of image i and each
                # caption j; dot product is linear, so this equals the dot
                # product with the (unnormalized) mean caption embedding.
                s_tt = torch.matmul(ref_txt.mean(dim=1), ref_txt_flat.T)  # (B, B*k)
                m |= (s_tt > self.p3) & (ref_s_it > self.p1_prime)
            if dup_mask is not None:
                m |= dup_mask.to(m.device)
            elif self.p_dup < 1.0:
                # Embedding-based fallback: image i matches caption j if any
                # of image i's own captions duplicates caption j. Prefer
                # dup_mask — this variant drifts with the trained embeddings.
                s_cc = torch.matmul(txt_flat, txt_flat.T)  # (B*k, B*k)
                m |= (s_cc > self.p_dup).view(B, k, B * k).any(dim=1)

            if soft_dup_mask is not None and self.soft_dup_weight <= 0.0:
                # w = 0 is exactly a hard duplicate; folding it in here keeps
                # the fast +/-1 path below for the common case. Deliberately
                # AFTER the dup_mask/p_dup chain above -- slotting it in as
                # another branch of that if/elif would have let p_dup fire in
                # cases where dup_mask already covered the pairs.
                m |= soft_dup_mask.to(m.device)
                soft_dup_mask = None

            if soft_dup_mask is None or self.soft_dup_weight >= 1.0:
                m = m.float() * 2 - 1  # {-1, +1}
                target = None
            else:
                # Soft labels. -logsigmoid(m*z) with m in {-1,+1} IS binary
                # cross-entropy with target in {0,1}, so writing it out in BCE
                # form generalises to a fractional target without changing any
                # existing result: target 1 -> -logsigmoid(z), target 0 ->
                # -logsigmoid(-z), both exactly as before.
                target = m.float()
                target = torch.where(soft_dup_mask.to(m.device) & ~m,
                                     torch.full_like(target, 1.0 - self.soft_dup_weight),
                                     target)

        if target is None:
            loss = -F.logsigmoid(m * logits)  # (B, B*k)
        else:
            loss = -(target * F.logsigmoid(logits) + (1 - target) * F.logsigmoid(-logits))

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

    # dup_mask (string-equality) path must match the embedding-based result
    # for the same duplicate structure, and must win over p_dup=1.0
    dup_mask = torch.zeros(B, B * k, dtype=torch.bool)
    dup_mask[0, k] = True
    dup_mask[1, 0] = True
    l_mask = no_dup(img, txt_shared, dup_mask=dup_mask)
    assert torch.allclose(l_mask, expected_dup), f"{l_mask} != {expected_dup}"

    # default construction = pure multi-positive sigmoid (no re-assignment)
    default_loss = FFFLoss()
    assert default_loss.p1 == default_loss.p2 == default_loss.p3 == default_loss.p_dup == 1.0

    # Gradients flow to features and to the learnable scale/bias
    img_g = img.clone().requires_grad_(True)
    txt_g = txt.clone().requires_grad_(True)
    loss_fn_g = FFFLoss()
    loss_fn_g(img_g, txt_g).backward()
    assert img_g.grad is not None and txt_g.grad is not None
    assert loss_fn_g.log_logit_scale.grad is not None
    assert loss_fn_g.logit_bias.grad is not None

    # ref_image_features/ref_text_features: thresholds must be evaluated
    # against the reference embeddings, not the trainable ones — construct a
    # case where the trainable embeddings alone would NOT trigger p1, but the
    # reference embeddings (near-identical for two different images) do.
    torch.manual_seed(1)
    img_r = torch.randn(B, D)
    txt_r = torch.randn(B, k, D)
    ref_img_r = torch.randn(B, D)
    ref_img_r[1] = ref_img_r[0] + 1e-4 * torch.randn(D)  # ref: images 0,1 near-identical
    ref_txt_r = torch.randn(B, k, D)  # unrelated reference text, p3 stays inactive

    p1_only = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0)  # p1 disabled: baseline, ref irrelevant
    with_ref = FFFLoss(p1=0.5, p1_prime=1.0, p2=1.0, p3=1.0)
    with_ref.load_state_dict(p1_only.state_dict())
    l_no_ref = p1_only(img_r, txt_r)  # ref-features not needed here (p1 disabled)
    l_with_ref = with_ref(img_r, txt_r, ref_image_features=ref_img_r, ref_text_features=ref_txt_r)
    # Passing ref features must not change which embeddings the sigmoid loss uses
    ref_img_n = F.normalize(ref_img_r, dim=-1)
    ref_txt_n = F.normalize(ref_txt_r, dim=-1).reshape(B * k, D)
    ref_s_it = ref_img_n @ ref_txt_n.T
    assert (ref_s_it > 0.5).any(), "test setup should trigger p1 via ref embeddings"
    f_i_r = F.normalize(img_r, dim=-1)
    f_t_r = F.normalize(txt_r, dim=-1).reshape(B * k, D)
    trainable_s_it = f_i_r @ f_t_r.T
    m_ref = torch.zeros(B, B * k, dtype=torch.bool)
    for i in range(B):
        m_ref[i, i * k:(i + 1) * k] = True
    m_ref |= ref_s_it > 0.5  # mask built from REFERENCE similarities
    m_ref = m_ref.float() * 2 - 1
    logits_r = trainable_s_it * with_ref.log_logit_scale.exp() + with_ref.logit_bias  # logits from TRAINABLE features
    expected_ref = -F.logsigmoid(m_ref * logits_r).sum() / (B * k)
    assert torch.allclose(l_with_ref, expected_ref), f"{l_with_ref} != {expected_ref}"
    # sanity: using the trainable embeddings' own S_it for the p1=0.5 mask would very
    # likely disagree with the ref-driven mask (different random embeddings) —
    # confirms ref features, not trainable ones, actually drove the re-assignment
    assert not torch.allclose((trainable_s_it > 0.5), (ref_s_it > 0.5)), \
        "test setup coincidentally identical — not a meaningful check"

    print("k=1 siglip:", loss.item())
    print("k=3 multi-positive:", loss3.item())


def test_fff_soft_dup():
    """A soft duplicate is a fractional label, and w=0 / w=1 hit the two ends."""
    torch.manual_seed(0)
    B, k, D, w = 3, 2, 16, 0.25
    img = torch.randn(B, D)
    txt = torch.randn(B, k, D)
    soft = torch.zeros(B, B * k, dtype=torch.bool)
    soft[0, k:2 * k] = True          # image 0 shares its abnormal findings with study 1
    soft[1, :k] = True

    def logits_of(fn):
        f_i = F.normalize(img, dim=-1)
        f_t = F.normalize(txt, dim=-1).reshape(B * k, D)
        return f_i @ f_t.T * fn.logit_scale + fn.logit_bias

    fn = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0, p_dup=1.0, soft_dup_weight=w)
    loss = fn(img, txt, soft_dup_mask=soft)

    z = logits_of(fn)
    pos = torch.zeros(B, B * k, dtype=torch.bool)
    pos[torch.arange(B).repeat_interleave(k), torch.arange(B * k)] = True
    target = torch.where(soft & ~pos, torch.full_like(z, 1.0 - w), pos.float())
    expected = -(target * F.logsigmoid(z) + (1 - target) * F.logsigmoid(-z))
    expected = expected.sum() / (B * k)
    assert torch.allclose(loss, expected), f"{loss} != {expected}"

    # w = 0 -> identical to passing the same mask as a hard dup_mask
    hard0 = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0, p_dup=1.0, soft_dup_weight=0.0)
    hard1 = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0, p_dup=1.0)
    assert torch.allclose(hard0(img, txt, soft_dup_mask=soft),
                          hard1(img, txt, dup_mask=soft))

    # w = 1 -> the soft pairs are ordinary negatives, i.e. no mask at all
    off = FFFLoss(p1=1.0, p1_prime=1.0, p2=1.0, p3=1.0, p_dup=1.0, soft_dup_weight=1.0)
    assert torch.allclose(off(img, txt, soft_dup_mask=soft), hard1(img, txt))

    # A soft mark never demotes a real positive: the diagonal block is untouched
    self_soft = pos.clone()
    assert torch.allclose(fn(img, txt, soft_dup_mask=self_soft), hard1(img, txt))

    # and the loss must sit between the two extremes it interpolates
    lo = hard0(img, txt, soft_dup_mask=soft).item()   # fully positive
    hi = off(img, txt, soft_dup_mask=soft).item()     # fully negative
    assert min(lo, hi) <= loss.item() <= max(lo, hi), f"{lo} {loss.item()} {hi}"

    print("FFF soft dup tests passed")


if __name__ == "__main__":
    test_fff_loss()
    test_fff_soft_dup()
    print("FFFLoss test passed.")
