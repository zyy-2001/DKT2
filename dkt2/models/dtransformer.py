import torch
import random
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn.functional import one_hot

MIN_SEQ_LEN = 5

class DTransformer(nn.Module):
    def __init__(self,  mask_response,
                        pred_last,
                        mask_future,
                        length,
                        trans,
                        num_skills,
                        num_questions, 
                        embedding_size=64, 
                        d_ff=256, 
                        num_attn_heads=8, 
                        n_know=16,
                        num_blocks=3, 
                        dropout=0.3,
                        lambda_cl=0.1, 
                        proj=False, 
                        hard_neg=False, 
                        window=1, 
                        shortcut=False,
                        separate_qr= False):
        super().__init__()
        self.mask_response = mask_response
        self.pred_last = pred_last
        self.mask_future = mask_future
        self.length = length
        self.trans = trans
        self.num_skills = num_skills
        self.dropout_rate = dropout
        self.lambda_cl = lambda_cl
        self.hard_neg = hard_neg
        self.shortcut = shortcut
        self.n_layers = num_blocks
        self.window = window
        d_fc = d_ff
        self.num_questions = num_questions

        self.separate_qr = separate_qr
        embed_l = embedding_size
        if self.num_questions > 0:
            self.q_diff_embed = nn.Embedding(num_skills+1,embedding_size)
            self.s_diff_embed = nn.Embedding(2, embedding_size) 
            self.p_diff_embed = nn.Embedding(num_questions+1,1)
        # num_skills+1 ,embedding_size
        self.q_embed = nn.Embedding(self.num_skills, embed_l)
        if self.separate_qr: 
            self.s_embed = nn.Embedding(2*self.num_skills+1, embed_l) # interaction emb
        else: # false default
            self.s_embed = nn.Embedding(2, embed_l)

        # Transformer Encoder
        self.n_heads = num_attn_heads 
        self.block1 = DTransformerLayer(embedding_size, self.n_heads, dropout)
        self.block2 = DTransformerLayer(embedding_size, self.n_heads, dropout)
        self.block3 = DTransformerLayer(embedding_size, self.n_heads, dropout)
        self.block4 = DTransformerLayer(embedding_size, self.n_heads, dropout, kq_same=False)

        # Knowledge Encoder
        self.n_know = n_know
        self.know_params = nn.Parameter(torch.empty(n_know, embedding_size))
        torch.nn.init.uniform_(self.know_params, -1.0, 1.0) 
        if self.trans:
            # Output Layer
            self.out = nn.Sequential(
                nn.Linear(embedding_size * 2, d_fc),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_fc, d_fc // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_fc // 2, self.num_skills),
            )
        else:
            # Output Layer
            self.out = nn.Sequential(
                nn.Linear(embedding_size * 2, d_fc),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_fc, d_fc // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_fc // 2, 1),
            )
        self.reset()

        # CL Linear Layer
        if proj:
            self.proj = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.GELU())
        else:
            self.proj = None

        self.loss_fn = nn.BCELoss(reduction="mean")
    
    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions+1 and self.num_questions>0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_emb, s_emb, lens):
        if self.shortcut:
            # AKT
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, scores = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            return self.block3(hq, hq, hs, lens, peek_cur=False), scores, None

        if self.n_layers == 1:
            hq = q_emb
            p, q_scores = self.block1(q_emb, q_emb, s_emb, lens, peek_cur=True)
        elif self.n_layers == 2:
            hq = q_emb
            hs, _ = self.block1(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block2(hq, hq, hs, lens, peek_cur=True)
        else: # default
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)

        bs, seqlen, embedding_size = p.size()
        n_know = self.n_know

        query = (
            self.know_params[None, :, None, :]
            .expand(bs, -1, seqlen, -1)
            .contiguous()
            .view(bs * n_know, seqlen, embedding_size)
        )
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)

        z, k_scores = self.block4(
            query, hq, p, torch.repeat_interleave(lens, n_know), peek_cur=False
        )
        z = (
            z.view(bs, n_know, seqlen, embedding_size)  # unpack dimensions
            .transpose(1, 2)  # (bs, seqlen, n_know, embedding_size)
            .contiguous()
            .view(bs, seqlen, -1)
        )
        k_scores = (
            k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)  # unpack dimensions
            .permute(0, 2, 3, 1, 4)  # (bs, n_heads, seqlen, n_know, seqlen)
            .contiguous()
        )
        return z, q_scores, k_scores

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  embedding_size# c_ct
        if self.separate_qr:
            qa_data = q_data + self.num_skills * target
            qa_embed_data = self.s_embed(qa_data)
        else:
            # BS, seqlen, embedding_size # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.s_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data
    
    def embedding(self,  q_data, target, pid_data=None):
        lens = (target >= 0).sum(dim=1)
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.num_questions > 0: # have problem id
            q_embed_diff_data = self.q_diff_embed(q_data)
            pid_embed_data = self.p_diff_embed(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.s_diff_embed(
                target)  # f_(ct,rt) or #h_rt (qt, rt)
            if self.separate_qr:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)
        return q_embed_data, qa_embed_data, lens, pid_embed_data

    def readout(self, z, query):
        bs, seqlen, _ = query.size()
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        ) # (bs * seqlen, self.n_know, -1)
        value = z.reshape(bs * seqlen, self.n_know, -1) # (bs * seqlen, self.n_know, -1)

        beta = torch.matmul(
            key,
            query.reshape(bs * seqlen, -1, 1),
        ).view(bs * seqlen, 1, self.n_know)  #  (bs * seqlen, 1, self.n_know)
        alpha = torch.softmax(beta, -1)
        return torch.matmul(alpha, value).view(bs, seqlen, -1)  # (bs, seqlen, -1)

    def predict(self, q, s, pid=None, n=1):
        if pid is None:
            pass
        else:
            if pid.nelement() == 0:
                # print("pid.nelement()")
                pid = None
            else:
                pass
        
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid)
        # print(f"q_emb {q_emb.shape} q_emb {q_emb.shape} lens:{len}")
        z, q_scores, k_scores = self(q_emb, s_emb, lens)
        # print(f"z {z.shape} q_scores {q_scores.shape} k_scores:{k_scores.shape}")
        
        # predict T+N
        if self.shortcut:
            assert n == 1, "AKT does not support T+N prediction"
            h = z
        else:
            query = q_emb[:, n - 1 :, :]
            h = self.readout(z[:, : query.size(1), :], query)
        

        # import sys
        # sys.exit()
        concat_q = torch.cat([q_emb, h], dim=-1)
        if self.trans:
            output = self.out(concat_q)
        else:
            output = self.out(concat_q).squeeze(-1)
        # print(f"yshape:{y.size()}")
       
        if pid is not None:
            return output, concat_q, z, q_emb, (p_diff**2).mean() * 1e-3, (q_scores, k_scores)
        else:
            return output, concat_q, z, q_emb, 0.0, (q_scores, k_scores)


    def get_loss(self, device, q, s, pids=None, q_cl=False):
        if pid.size(1) ==0:
            pid = None
        
        q = q.to(device)
        s = s.to(device)
        if pid is not None:
            pid = pid.to(device)
        output, concat_q, _, _, reg_loss, _ = self.predict(q, s, pid)
        m = nn.Sigmoid()
        preds = m(output)
        if q_cl:
            masked_labels = s[s>=0].float()
            masked_logits = logits[s>=0]
            return (
            F.binary_cross_entropy_with_logits(masked_logits, masked_labels, reduction="mean")+ reg_loss 
            )
        else:
            return preds, reg_loss
    

    def get_cl_loss(self, feed_dict):
        
        pid = feed_dict['questions']
        r = feed_dict['responses']
        q = feed_dict['skills']
        attention_mask = feed_dict['attention_mask']
        s = r * (r > -1).long()

        if self.trans:
            pid = pid[:, :-self.length]
            s = s[:, :-self.length]
            cshft = q[:, self.length:]
            q = q[:, :-self.length]
        elif self.mask_future:
            attention_mask[:, -self.length:] = 0
            pid = pid * attention_mask
            s = s * attention_mask
            q = q * attention_mask
        elif self.mask_response:
            attention_mask[:, -self.length:] = 0
            s = s * attention_mask
            
        bs = s.size(0)

        # Input data preprocess
        if pid.size(1) ==0 or self.num_questions == 0:
            pid = None


        # skip CL for batches that are too short
        lens = (s >= 0).sum(dim=1)
        minlen = lens.min().item()
        # if minlen < MIN_SEQ_LEN:

        #     return self.get_loss(device, q, s, pid,True)

        # augmentation
        q_ = q.clone()
        s_ = s.clone()

        if pid is not None:
            pid_ = pid.clone()
        else:
            pid_ = None

        # manipulate order
        for b in range(bs):
            idx = random.sample(
                range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                q_[b, i], q_[b, i + 1] = q_[b, i + 1], q_[b, i]
                s_[b, i], s_[b, i + 1] = s_[b, i + 1], s_[b, i]
                if pid_ is not None:
                    pid_[b, i], pid_[b, i + 1] = pid_[b, i + 1], pid_[b, i]

        # hard negative
        s_flip = s.clone() if self.hard_neg else s_
        for b in range(bs):
            # manipulate score
            idx = random.sample(
                range(lens[b]), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]
        if not self.hard_neg:
            s_ = s_flip

    #     # model
        # logits, z_1, q_emb, reg_loss, _ = self.predict(q, s, pid)
        logits, concat_q, z_1, q_emb, reg_loss, _ = self.predict(q, s, pid)
        # masked_logits = logits[s >= 0]

        # extract forward
        # print(f"q_ shape:{q_.shape} s_ shape:{s_.shape}")
        _, _,  z_2, *_ = self.predict(q_, s_, pid_)

        if self.hard_neg:
           _, _, z_3, *_ = self.predict(q, s_flip, pid)

        # CL loss
        # print(f"z1 shape:{z_1.shape} z2 shape:{z_2.shape}")
        # import sys
        # sys.exit()
        input = self.sim(z_1[:, :minlen, :], z_2[:, :minlen, :])
        if self.hard_neg:
            hard_neg = self.sim(z_1[:, :minlen, :], z_3[:, :minlen, :])
            input = torch.cat([input, hard_neg], dim=1)
        target = (
            torch.arange(s.size(0))[:, None]
            .to(self.know_params.device)
            .expand(-1, minlen)
        )
        cl_loss = F.cross_entropy(input, target)

        #prediction loss
        # masked_labels = s[s >= 0].float()
        # pred_loss = F.binary_cross_entropy_with_logits(
        #     masked_logits, masked_labels, reduction="mean"
        # )

        for i in range(1, self.window):
            label = s[:, i:]
            query = q_emb[:, i:, :]
            h = self.readout(z_1[:, : query.size(1), :], query)
            if self.trans:
                y = self.out(torch.cat([query, h], dim=-1))
            else:
                y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

            pred_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )

        m = nn.Sigmoid()
        preds = m(logits)
        if self.trans:
            preds = (preds * one_hot(cshft.long(), self.num_skills)).sum(-1)
            true = r[:, self.length:].float()
        elif self.mask_future or self.pred_last or self.mask_response:
            preds = preds[:, -self.length:]
            true = r[:, -self.length:].float()
        else:
            preds = preds[:, self.length:]
            true = r[:, self.length:].float()

        out_dict = {
            "pred": preds,
            "true": true,
            "reg_loss": cl_loss * self.lambda_cl + reg_loss
        }

        return out_dict
    
    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        reg_loss = out_dict["reg_loss"]
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss + reg_loss, len(pred[mask]), true[mask].sum().item()

    def sim(self, z1, z2):
        bs, seqlen, _ = z1.size()
        z1 = z1.unsqueeze(1).view(bs, 1, seqlen, self.n_know, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seqlen, self.n_know, -1)
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / 0.05


class DTransformerLayer(nn.Module):
    def __init__(self, embedding_size, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(embedding_size, n_heads, kq_same)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False):
        # construct mask
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.device())

        # mask manipulation
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()
            # print(f"mask: {mask.shape}")
            # for b in range(query.size(0)):
            #     # sample for each batch
            #     if lens[b] < MIN_SEQ_LEN:
            #         # skip for short sequences
            #         continue
            #     idx = random.sample(
            #         range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
            #     )
            #     for i in idx:
            #         mask[b, :, i + 1 :, i] = 0

        # apply transformer layer
        query_, scores = self.masked_attn_head(
            query, key, values, mask, maxout=not peek_cur
        )
        query = query + self.dropout(query_)
        return self.layer_norm(query), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.d_k = embedding_size // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.v_linear = nn.Linear(embedding_size, embedding_size, bias=bias)

        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, maxout=False):
        bs = q.size(0)

        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        v_, scores = attention(
            q,
            k,
            v,
            mask,
            self.gammas,
            maxout,
        )

        # concatenate heads and put through final linear layer
        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.embedding_size)

        output = self.out_proj(concat)

        return output, scores


def attention(q, k, v, mask, gamma=None, maxout=False):
    # attention score with scaled dot production
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    # include temporal effect
    if gamma is not None:
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            # ones = torch.ones(head // 2, 1, 1).to(gamma.device)
            # sign = torch.concat([ones, -ones])
            # scores_ = (scores * sign).masked_fill(mask == 0, -1e32)
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)

            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        gamma = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        scores *= total_effect

    # normalize attention score
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)  # set to hard zero to avoid leakage

    # max-out scores (bs, n_heads, seqlen, seqlen)
    if maxout:
        scale = torch.clamp(1.0 / (scores.max(dim=-1, keepdim=True)[0] + 1e-8), max=5.0)
        scores *= scale

    # calculate output
    output = torch.matmul(scores, v)
    return output, scores