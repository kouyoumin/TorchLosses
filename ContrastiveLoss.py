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
                a potential false negative. Set to 1.0 to disable. Applied to
                both the in-batch negatives and the queue.
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
        # Registered as buffers so .to()/.cuda() move the queue with the module.
        # persistent=False: the queue holds embeddings produced by whatever the
        # text encoder looked like a few hundred steps ago, so carrying it
        # across a checkpoint save/load would only inject stale negatives.
        # state_dict() therefore stays {log_temperature}, as before.
        self.register_buffer('_queue', None, persistent=False)
        self.register_buffer('_queue_ids', None, persistent=False)
        self._queue_ptr = 0
        self._queue_filled = 0

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(0.01, 0.5)

    def _enqueue(self, embeddings: torch.Tensor, ids: torch.Tensor | None = None):
        B, D = embeddings.shape
        if self._queue is None:
            # fp32 regardless of the embedding dtype; cast back on read
            self._queue = torch.zeros(self.queue_size, D, dtype=torch.float32, device=embeddings.device)
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
        queue_mask = None
        if self.queue_size > 0 and self._queue is not None and self._queue_filled > 0:
            # The queue is kept in fp32; read it back in the embedding dtype so
            # torch.cat below does not silently promote the whole logit matrix
            # when the towers run in a reduced precision.
            queue = self._queue[:self._queue_filled].detach().to(device=f2.device, dtype=f2.dtype)
            all_f2 = torch.cat([f2, queue], dim=0)   # (B+K) x D
            if dup_ids is not None and self._queue_ids is not None:
                queue_ids = self._queue_ids[:self._queue_filled].to(f1.device)
                queue_mask = dup_ids.view(-1, 1) == queue_ids.view(1, -1)  # B x K
        else:
            all_f2 = f2

        logits_a = torch.matmul(f1, all_f2.T) / temperature  # B x (B+K)
        if fn_mask is not None:
            logits_a[:, :B] = logits_a[:, :B].masked_fill(fn_mask, -1e4)

        # The similarity threshold has to cover the queue as well, not just the
        # in-batch block. The queue is where nearly all the negatives are
        # (train.py uses queue_size = batch_size * 128), and dup_ids only
        # catches reports that are byte-identical after canonicalising —
        # paraphrases of the same finding are exactly what the threshold is
        # for. Masking only the in-batch block left the model penalised for
        # correctly recognising a text that happened to sit in the queue.
        #
        # The similarities are recovered from the logits rather than from a
        # second matmul against the queue: at the training queue size that
        # matmul costs as much again as the main one (~2.2ms/step measured),
        # while the multiply is elementwise. It costs one extra rounding, and
        # the in-batch mask above is already evaluated at matmul precision
        # (bf16 under autocast), so the two blocks stay consistent.
        if all_f2.shape[0] > B and self.false_negative_threshold < 1.0:
            with torch.no_grad():
                q_fn_mask = (logits_a[:, B:] * temperature) > self.false_negative_threshold
            queue_mask = q_fn_mask if queue_mask is None else (queue_mask | q_fn_mask)

        if queue_mask is not None:
            logits_a[:, B:] = logits_a[:, B:].masked_fill(queue_mask, -1e4)
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


def test_queue_false_negative_threshold():
    """The similarity threshold must mask queue negatives, not just in-batch ones."""
    torch.manual_seed(1)
    B, D, thr = 4, 16, 0.5
    f1, f2 = torch.randn(B, D), torch.randn(B, D)
    g1, g2 = torch.randn(B, D), torch.randn(B, D)

    loss_fn = ContrastiveLoss(temperature=0.2, false_negative_threshold=thr, queue_size=16)
    loss_fn(f1, f2)                       # queue now holds batch 1's texts
    loss = loss_fn(g1, g2)

    m1, m2 = F.normalize(g1, dim=1), F.normalize(g2, dim=1)
    queue = F.normalize(f2, dim=1)
    t = loss_fn.log_temperature.exp().clamp(0.01, 0.5)
    eye = torch.eye(B, dtype=torch.bool)
    labels = torch.arange(B)

    la = m1 @ torch.cat([m2, queue], dim=0).T / t
    la[:, :B] = la[:, :B].masked_fill((m1 @ m2.T > thr) & ~eye, -1e4)
    la[:, B:] = la[:, B:].masked_fill(m1 @ queue.T > thr, -1e4)
    lb = (m2 @ m1.T / t).masked_fill(((m1 @ m2.T > thr) & ~eye).T, -1e4)
    expected = (F.cross_entropy(la, labels) + F.cross_entropy(lb, labels)) / 2
    assert torch.allclose(loss, expected), f"{loss} != {expected}"

    # Targeted check: a queue entry that duplicates a sample's own positive
    # must not count as a negative for that sample. Sample 0's image and text
    # are both `u`, and the queue holds one entry equal to `u`; its row of the
    # loss must therefore be identical to the no-queue baseline, while the
    # other rows (which gain one genuine extra negative) must not be.
    u = F.normalize(torch.randn(1, D), dim=1)
    x1 = torch.cat([u, torch.randn(B - 1, D)])
    x2 = torch.cat([u, torch.randn(B - 1, D)])

    def queued(threshold):
        fn = ContrastiveLoss(temperature=0.2, false_negative_threshold=threshold,
                             queue_size=4, reduction='none')
        fn(u, u)                                     # queue = [u]
        return fn(x1, x2)

    baseline = ContrastiveLoss(temperature=0.2, false_negative_threshold=thr,
                               queue_size=0, reduction='none')(x1, x2)
    masked = queued(thr)
    assert torch.allclose(masked[0], baseline[0]), f"{masked[0]} != {baseline[0]}"
    assert not torch.allclose(masked[1:], baseline[1:])

    # With the threshold disabled the duplicate is scored as a hard negative —
    # this is what the queue used to do unconditionally. It carries the same
    # logit as the positive, so it doubles the dominant term of the image→text
    # softmax denominator: +log(2) on loss_a, i.e. +log(2)/2 after averaging
    # the two directions.
    unmasked = queued(1.0)
    delta = unmasked[0] - baseline[0]
    assert torch.allclose(delta, torch.log(torch.tensor(2.0)) / 2, atol=0.05), delta

    print("queue false-negative threshold tests passed")


def test_queue_buffers():
    """Queue lives in buffers: moved by .to(), absent from state_dict."""
    loss_fn = ContrastiveLoss(temperature=0.2, false_negative_threshold=1.0, queue_size=8, train_temp=True)
    f1, f2 = torch.randn(4, 16), torch.randn(4, 16)
    ids = torch.tensor([1, 2, 3, 4])
    loss_fn(f1, f2, dup_ids=ids)

    buffers = dict(loss_fn.named_buffers())
    assert '_queue' in buffers and '_queue_ids' in buffers, buffers.keys()
    assert set(loss_fn.state_dict().keys()) == {'log_temperature'}, loss_fn.state_dict().keys()

    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
        assert loss_fn._queue.is_cuda and loss_fn._queue_ids.is_cuda
        loss_fn(f1.cuda(), f2.cuda(), dup_ids=ids.cuda())

        # bf16 embeddings must not promote the logits back to fp32
        bf = ContrastiveLoss(temperature=0.2, false_negative_threshold=0.9, queue_size=8).cuda()
        a = torch.randn(4, 16, device='cuda', dtype=torch.bfloat16)
        b = torch.randn(4, 16, device='cuda', dtype=torch.bfloat16)
        bf(a, b)
        assert bf._queue.dtype == torch.float32
        bf(a, b)

    print("queue buffer tests passed")


if __name__ == "__main__":
    test_contrastive_loss()
    test_dedup()
    test_queue_false_negative_threshold()
    test_queue_buffers()
    print("ContrastiveLoss test passed.")
