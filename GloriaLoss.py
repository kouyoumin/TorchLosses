import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def attention_fn(query, context, temp1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


def local_loss(
    img_features, sentence_embs, sentence_cnt, temp1=4.0, temp2=5.0, temp3=10.0, reduction="mean"
):

    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    
    for i in range(sentence_embs.shape[0]):

        # Get the i-th text description
        num_sentences = min(int(sentence_cnt[i]), sentence_embs.shape[1])
        assert num_sentences > 0, f"Invalid num_sentences at index {i}: {num_sentences}"
        
        word = sentence_embs[i, :num_sentences, :].unsqueeze(0).transpose(1, 2).contiguous()  # [1, dim, MAX_NUM_SENTENCES]
        word = word.repeat(batch_size, 1, 1)  # [batch, dim, MAX_NUM_SENTENCES]
        context = img_features  # [batch, dim, H, W]
        #print(word.shape, context.shape)

        weiContext, attn = attention_fn(
            word, context, temp1
        )  # [batch, dim, MAX_NUM_SENTENCES], [batch, MAX_NUM_SENTENCES, H, W]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [MAX_NUM_SENTENCES, H, W]
        word = word.transpose(1, 2).contiguous()  # [batch, MAX_NUM_SENTENCES, dim]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [batch, MAX_NUM_SENTENCES, dim]

        word = word.view(batch_size * num_sentences, -1)  # [batch * MAX_NUM_SENTENCES, dim]
        weiContext = weiContext.view(batch_size * num_sentences, -1)  # [batch * MAX_NUM_SENTENCES, dim]

        #row_sim = cosine_similarity(word, weiContext)
        row_sim = F.cosine_similarity(word, weiContext, dim=1)  # [batch * MAX_NUM_SENTENCES]
        row_sim = row_sim.view(batch_size, num_sentences)  # [batch, MAX_NUM_SENTENCES]

        row_sim.mul_(temp2).exp_()
        if reduction == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [batch, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [batch, 1]
        row_sim = torch.log(row_sim.clamp(min=1e-8))

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [batch, batch]

    labels = torch.LongTensor(range(batch_size)).to(similarities.device)

    loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return (loss0+loss1)/2., att_maps


class GloriaLocalLoss(nn.Module):
    def __init__(self, temp1=4.0, temp2=5.0, temp3=10.0, reduction="mean", train_temp=False):
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

    def __call__(self, img_features, sentence_embs, sentence_cnt):
        return local_loss(
            img_features,
            sentence_embs,
            sentence_cnt,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
            reduction=self.reduction,
        )