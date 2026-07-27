import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self,
                normalize=True,
                temperature=0.07,
                train_temp=False,
                false_negative_threshold=0.98,
                queue_size=4096,
                dedup=False,
                reduction='mean'):
        """
        Args:
            normalize: L2-normalize features before computing similarity.
            temperature: Temperature for logit scaling (CLIP uses 0.07).
            train_temp: If True, temperature is a learnable parameter (CLIP-style),
                updated via backprop instead of staying fixed.
            false_negative_threshold: Cosine similarity above this is masked as
                a potential false negative. Set to 1.0 to disable.
            queue_size: Number of past text embeddings kept as extra negatives
                (MoCo-style FIFO queue). Set to 0 to disable.
            dedup: If True, the caller passes per-sample text ids (e.g. string
                hashes) via dup_ids; pairs with identical text are excluded
                from the negatives — both within the batch and against the
                queue (queue entries remember their text id). Removes exact
                false negatives without relying on the trained embeddings.
            reduction: 'mean' | 'sum' | 'none'.  Use 'none' to get per-sample
                losses of shape (B,) for use with OHEM wrappers.
        """
        super().__init__()
        self.normalize = normalize
        self.false_negative_threshold = false_negative_threshold
        self.train_temp = train_temp
        log_temperature = torch.tensor(temperature).log()
        self.log_temperature = nn.Parameter(log_temperature) if train_temp else log_temperature
        self.queue_size = queue_size
        self.dedup = dedup
        self.reduction = reduction
        self._queue: torch.Tensor | None = None
        self._queue_ids: torch.Tensor | None = None
        self._queue_ptr = 0
        self._queue_filled = 0

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(0.01, 0.5)

    def _enqueue(self, embeddings: torch.Tensor, ids: torch.Tensor | None = None):
        B, D = embeddings.shape
        if self._queue is None:
            self._queue = torch.zeros(self.queue_size, D, device=embeddings.device)
        if ids is not None and self._queue_ids is None:
            self._queue_ids = torch.zeros(self.queue_size, dtype=torch.int64, device=embeddings.device)
        end = self._queue_ptr + B
        if end <= self.queue_size:
            self._queue[self._queue_ptr:end] = embeddings
            if ids is not None:
                self._queue_ids[self._queue_ptr:end] = ids
        else:
            first = self.queue_size - self._queue_ptr
            self._queue[self._queue_ptr:] = embeddings[:first]
            self._queue[:B - first] = embeddings[first:]
            if ids is not None:
                self._queue_ids[self._queue_ptr:] = ids[:first]
                self._queue_ids[:B - first] = ids[first:]
        self._queue_ptr = end % self.queue_size
        self._queue_filled = min(self._queue_filled + B, self.queue_size)

    def __call__(self, feature1, feature2, dup_ids=None):
        """
        Args:
            feature1: (B, D) image embeddings.
            feature2: (B, D) text embeddings.
            dup_ids: optional (B,) int64 tensor of per-sample text identifiers
                (e.g. hash of the report string). Samples sharing an id are
                exact text duplicates: they are excluded from each other's
                negatives, and queue entries with the same id are excluded too.
        """
        if self.normalize:
            f1 = F.normalize(feature1, dim=1)
            f2 = F.normalize(feature2, dim=1)
        else:
            f1 = feature1
            f2 = feature2

        B = f1.shape[0]
        temperature = self.log_temperature.to(f1.device).exp().clamp(0.01, 0.5)

        eye = torch.eye(B, device=f1.device, dtype=torch.bool)

        # Build false-negative mask for the current batch
        fn_mask = None
        if self.false_negative_threshold < 1.0:
            with torch.no_grad():
                sim = torch.matmul(f1, f2.T)
                fn_mask = (sim > self.false_negative_threshold) & ~eye

        # Exact-duplicate mask from text ids (string equality, training-independent)
        if dup_ids is not None:
            dup_ids = dup_ids.to(f1.device)
            dup_mask = (dup_ids.view(-1, 1) == dup_ids.view(1, -1)) & ~eye
            fn_mask = dup_mask if fn_mask is None else (fn_mask | dup_mask)

        # --- image → text (with queue for extra negatives) ---
        queue_dup_mask = None
        if self.queue_size > 0 and self._queue is not None and self._queue_filled > 0:
            queue = self._queue[:self._queue_filled].detach().to(f2.device)
            all_f2 = torch.cat([f2, queue], dim=0)   # (B+K) x D
            if dup_ids is not None and self._queue_ids is not None:
                queue_ids = self._queue_ids[:self._queue_filled].to(f1.device)
                queue_dup_mask = dup_ids.view(-1, 1) == queue_ids.view(1, -1)  # B x K
        else:
            all_f2 = f2

        logits_a = torch.matmul(f1, all_f2.T) / temperature  # B x (B+K)
        if fn_mask is not None:
            logits_a[:, :B] = logits_a[:, :B].masked_fill(fn_mask, -1e4)
        if queue_dup_mask is not None:
            logits_a[:, B:] = logits_a[:, B:].masked_fill(queue_dup_mask, -1e4)
        labels_a = torch.arange(B, device=f1.device)
        loss_a = F.cross_entropy(logits_a, labels_a, reduction='none')  # (B,)

        # --- text → image (current batch only) ---
        logits_b = torch.matmul(f2, f1.T) / temperature      # B x B
        if fn_mask is not None:
            logits_b = logits_b.masked_fill(fn_mask.T, -1e4)
        labels_b = torch.arange(B, device=f2.device)
        loss_b = F.cross_entropy(logits_b, labels_b, reduction='none')  # (B,)

        # Enqueue current text embeddings after loss is computed
        if self.queue_size > 0:
            self._enqueue(f2.detach(), dup_ids.detach() if dup_ids is not None else None)

        per_sample = (loss_a + loss_b) / 2  # (B,)
        if self.reduction == 'mean':
            return per_sample.mean()
        elif self.reduction == 'sum':
            return per_sample.sum()
        else:  # 'none'
            return per_sample


def test_contrastive_loss():
    loss_fn = ContrastiveLoss(temperature=0.2, false_negative_threshold=1.0, queue_size=0)
    feature1 = torch.randn((4, 10))
    feature2 = torch.randn((4, 10))

    loss = loss_fn(feature1, feature2)

    f1 = F.normalize(feature1, dim=1)
    f2 = F.normalize(feature2, dim=1)
    temperature = loss_fn.log_temperature.exp().clamp(0.01, 0.5)
    logits_ab = torch.matmul(f1, f2.T) / temperature
    logits_ba = torch.matmul(f2, f1.T) / temperature
    labels = torch.arange(4)
    expected = (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels)) / 2
    assert torch.allclose(loss, expected), f"{loss} != {expected}"

    print("f1, f2:", loss_fn(feature1, feature2).item())
    print("queue test:", ContrastiveLoss(queue_size=16)(feature1, feature2).item())


def test_dedup():
    torch.manual_seed(0)
    B, D = 4, 16
    f1 = torch.randn(B, D)
    f2 = torch.randn(B, D)
    ids = torch.tensor([100, 200, 100, 300])  # samples 0 and 2 share the same text

    # In-batch dedup: masked CE must match manual computation
    loss_fn = ContrastiveLoss(temperature=0.2, false_negative_threshold=1.0, queue_size=0, dedup=True)
    loss = loss_fn(f1, f2, dup_ids=ids)
    n1 = F.normalize(f1, dim=1)
    n2 = F.normalize(f2, dim=1)
    t = loss_fn.log_temperature.exp().clamp(0.01, 0.5)
    la = n1 @ n2.T / t
    lb = n2 @ n1.T / t
    dup = (ids.view(-1, 1) == ids.view(1, -1)) & ~torch.eye(B, dtype=torch.bool)
    la = la.masked_fill(dup, -1e4)
    lb = lb.masked_fill(dup.T, -1e4)
    labels = torch.arange(B)
    expected = (F.cross_entropy(la, labels) + F.cross_entropy(lb, labels)) / 2
    assert torch.allclose(loss, expected), f"{loss} != {expected}"

    # dup_ids=None -> identical to legacy behavior
    legacy = ContrastiveLoss(temperature=0.2, false_negative_threshold=1.0, queue_size=0)
    dedup_off = ContrastiveLoss(temperature=0.2, false_negative_threshold=1.0, queue_size=0, dedup=True)
    assert torch.allclose(legacy(f1, f2), dedup_off(f1, f2))

    # Queue dedup: queue entries with the same id must be excluded
    qloss = ContrastiveLoss(temperature=0.2, false_negative_threshold=1.0, queue_size=8, dedup=True)
    qloss(f1, f2, dup_ids=ids)                     # fills queue with ids [100,200,100,300]
    g1, g2 = torch.randn(B, D), torch.randn(B, D)
    gids = torch.tensor([100, 999, 998, 997])      # sample 0 duplicates two queue entries
    loss_q = qloss(g1, g2, dup_ids=gids)
    m1 = F.normalize(g1, dim=1)
    m2 = F.normalize(g2, dim=1)
    queue = F.normalize(f2, dim=1)                 # queue currently holds first batch's texts
    all2 = torch.cat([m2, queue], dim=0)
    la = m1 @ all2.T / t
    dup2 = (gids.view(-1, 1) == gids.view(1, -1)) & ~torch.eye(B, dtype=torch.bool)
    la[:, :B] = la[:, :B].masked_fill(dup2, -1e4)
    la[:, B:] = la[:, B:].masked_fill(gids.view(-1, 1) == ids.view(1, -1), -1e4)
    lb = m2 @ m1.T / t
    lb = lb.masked_fill(dup2.T, -1e4)
    expected_q = (F.cross_entropy(la, labels) + F.cross_entropy(lb, labels)) / 2
    assert torch.allclose(loss_q, expected_q), f"{loss_q} != {expected_q}"

    print("dedup tests passed")


if __name__ == "__main__":
    test_contrastive_loss()
    test_dedup()
    print("ContrastiveLoss test passed.")
