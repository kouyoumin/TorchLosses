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
        self.reduction = reduction
        self._queue: torch.Tensor | None = None
        self._queue_ptr = 0
        self._queue_filled = 0

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(0.01, 0.5)

    def _enqueue(self, embeddings: torch.Tensor):
        B, D = embeddings.shape
        if self._queue is None:
            self._queue = torch.zeros(self.queue_size, D, device=embeddings.device)
        end = self._queue_ptr + B
        if end <= self.queue_size:
            self._queue[self._queue_ptr:end] = embeddings
        else:
            first = self.queue_size - self._queue_ptr
            self._queue[self._queue_ptr:] = embeddings[:first]
            self._queue[:B - first] = embeddings[first:]
        self._queue_ptr = end % self.queue_size
        self._queue_filled = min(self._queue_filled + B, self.queue_size)

    def __call__(self, feature1, feature2):
        if self.normalize:
            f1 = F.normalize(feature1, dim=1)
            f2 = F.normalize(feature2, dim=1)
        else:
            f1 = feature1
            f2 = feature2

        B = f1.shape[0]
        temperature = self.log_temperature.to(f1.device).exp().clamp(0.01, 0.5)

        # Build false-negative mask for the current batch
        fn_mask = None
        if self.false_negative_threshold < 1.0:
            with torch.no_grad():
                sim = torch.matmul(f1, f2.T)
                eye = torch.eye(B, device=f1.device, dtype=torch.bool)
                fn_mask = (sim > self.false_negative_threshold) & ~eye

        # --- image → text (with queue for extra negatives) ---
        if self.queue_size > 0 and self._queue is not None and self._queue_filled > 0:
            queue = self._queue[:self._queue_filled].detach().to(f2.device)
            all_f2 = torch.cat([f2, queue], dim=0)   # (B+K) x D
        else:
            all_f2 = f2

        logits_a = torch.matmul(f1, all_f2.T) / temperature  # B x (B+K)
        if fn_mask is not None:
            logits_a[:, :B] = logits_a[:, :B].masked_fill(fn_mask, -1e4)
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
            self._enqueue(f2.detach())

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


if __name__ == "__main__":
    test_contrastive_loss()
    print("ContrastiveLoss test passed.")
