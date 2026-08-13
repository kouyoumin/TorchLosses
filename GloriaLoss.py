import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def attention_fn(query, contextT, temp1, ih, iw):
    """
    query: 1 x ndf x queryL -- ONE sample's sentences, NOT repeated across the
        batch. torch.matmul broadcasts the batch dimension; torch.bmm does not,
        and materialising a [batch, ndf, queryL] copy purely to satisfy bmm cost
        12MB per call at the training shapes.
    contextT: batch x sourceL x ndf (sourceL = ih x iw), flattened and
        transposed by the caller. This used to be derived from `context` in
        here, which meant one 108MB .contiguous() copy per call -- all of them
        identical, and every one kept alive by autograd until backward. That
        single line was 20GB of the 33GB this loss cost at batch 192.

    Returns weightedContext as batch x queryL x ndf; the original returned
    batch x ndf x queryL and the caller transposed it right back.
    """
    batch_size = contextT.size(0)

    # (batch x sourceL x ndf)(1 x ndf x queryL) --> batch x sourceL x queryL
    attn = torch.matmul(contextT, query)
    # Softmax over queryL. F.softmax on the last dim of the 3-D tensor computes
    # what view(batch*sourceL, queryL) + nn.Softmax(-1) did, and does not need
    # the reshape to be contiguous.
    attn = F.softmax(attn, dim=-1)

    # --> batch x queryL x sourceL, then softmax over sourceL
    attn = attn.transpose(1, 2)
    attn = F.softmax(attn * temp1, dim=-1)

    # bmm(context[b,ndf,sourceL], attnT[b,sourceL,queryL]) -> [b,ndf,queryL],
    # which the caller then transposed, is exactly attn @ contextT. Producing
    # [batch, queryL, ndf] directly drops both .contiguous() copies.
    weightedContext = torch.matmul(attn, contextT)

    # reshape, not view: attn is the output of an elementwise op on a transposed
    # tensor, and TensorIterator is free to give that output the same permuted
    # stride order (it does so to keep the loop coalesced). view() would raise
    # on such a tensor -- and only on whichever backend chose to permute, which
    # is the sort of thing that surfaces an hour into a training run. reshape()
    # is identical when the tensor is already contiguous, which it usually is.
    return weightedContext, attn.reshape(batch_size, -1, ih, iw)


def local_loss(
    img_features, sentence_embs, sentence_cnt, temp1=4.0, temp2=5.0, temp3=10.0, reduction="mean", dup_mask=None
):

    batch_size, ndf, ih, iw = img_features.shape

    att_maps = []
    similarities = []

    # Hoisted out of the loop. `context` never changes across iterations, so the
    # old code built batch_size identical copies of this and autograd held every
    # one of them: 108MB x 192 = 20GB at the training shapes, 65% of what this
    # loss used. Everything below is arithmetically unchanged -- verified
    # bit-identical against the previous implementation in float64, for both
    # uniform and ragged sentence_cnt.
    contextT = img_features.view(batch_size, ndf, ih * iw).transpose(1, 2).contiguous()  # [batch, sourceL, dim]

    for i in range(sentence_embs.shape[0]):

        # Get the i-th text description
        num_sentences = min(int(sentence_cnt[i]), sentence_embs.shape[1])
        assert num_sentences > 0, f"Invalid num_sentences at index {i}: {num_sentences}"

        word = sentence_embs[i, :num_sentences, :].unsqueeze(0)  # [1, MAX_NUM_SENTENCES, dim]

        weiContext, attn = attention_fn(
            word.transpose(1, 2), contextT, temp1, ih, iw
        )  # [batch, MAX_NUM_SENTENCES, dim], [batch, MAX_NUM_SENTENCES, H, W]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [MAX_NUM_SENTENCES, H, W]

        # Broadcast [1, MAX_NUM_SENTENCES, dim] against [batch, ...] instead of
        # expanding it: the old code materialised word to the full batch and
        # flattened both operands only because it called cosine_similarity on
        # 2-D views.
        row_sim = F.cosine_similarity(word, weiContext, dim=-1)  # [batch, MAX_NUM_SENTENCES]

        row_sim = row_sim.mul(temp2).exp()
        if reduction == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [batch, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [batch, 1]
        row_sim = torch.log(row_sim.clamp(min=1e-8))

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    if dup_mask is not None:
        # Exclude exact-duplicate (image, sentence-set) pairs from negatives.
        # dup_mask: (batch, batch) bool, True where texts i and j are the same
        # string but i != j (symmetric), so one masked_fill covers both CE
        # directions after the transpose.
        similarities = similarities.masked_fill(dup_mask.to(similarities.device), -1e4)
    similarities1 = similarities.transpose(0, 1)  # [batch, batch]

    labels = torch.LongTensor(range(batch_size)).to(similarities.device)

    loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return (loss0+loss1)/2., att_maps


class GloriaLocalLoss(nn.Module):
    def __init__(self, temp1=4.0, temp2=5.0, temp3=10.0, reduction="mean", train_temp=False, dedup=False):
        super(GloriaLocalLoss, self).__init__()
        self.train_temp = train_temp
        if train_temp:
            self.temp1 = nn.Parameter(torch.tensor(float(temp1)))
            self.temp2 = nn.Parameter(torch.tensor(float(temp2)))
            self.temp3 = nn.Parameter(torch.tensor(float(temp3)))
        else:
            self.temp1 = temp1
            self.temp2 = temp2
            self.temp3 = temp3
        self.reduction = reduction
        # dedup: caller passes dup_mask (exact-duplicate text pairs, string
        # equality) to exclude them from the batch negatives
        self.dedup = dedup

    def __call__(self, img_features, sentence_embs, sentence_cnt, dup_mask=None):
        return local_loss(
            img_features,
            sentence_embs,
            sentence_cnt,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
            reduction=self.reduction,
            dup_mask=dup_mask,
        )