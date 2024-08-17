import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout, Sequential, ReLU
import copy
from torch.nn.functional import one_hot


class SAKT(Module):
    def __init__(self, joint, mask_future, length, trans, num_skills, seq_len, embedding_size, num_attn_heads, dropout, num_blocks=2):
        super().__init__()
        self.joint = joint
        self.mask_future = mask_future
        self.length = length
        self.trans = trans
        self.num_skills = num_skills
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_blocks = num_blocks

        # num_skills, seq_len, embedding_size, num_attn_heads, dropout, emb_path="")
        self.interaction_emb = Embedding(num_skills * 2, embedding_size)
        self.exercise_emb = Embedding(num_skills, embedding_size)
        # self.P = Parameter(torch.Tensor(self.seq_len, self.embedding_size))
        self.position_emb = Embedding(seq_len, embedding_size)

        self.blocks = get_clones(Blocks(embedding_size, num_attn_heads, dropout), self.num_blocks)

        self.dropout_layer = Dropout(dropout)
        if self.trans:
            self.pred = Linear(self.embedding_size, self.num_skills)
        else:
            self.pred = Linear(self.embedding_size, 1)
        self.loss_fn = nn.BCELoss(reduction="mean")

    def base_emb(self, q, r, qry):
        x = q + self.num_skills * r
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
        device = q.device
        posemb = self.position_emb(pos_encode(xemb.shape[1], device))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, feed_dict):


        q = feed_dict['skills']
        r = feed_dict['responses']
        masked_r = r * (r > -1).long()
        qry = q[:, self.length:]
        if self.trans:
            cshft = q[:, self.length:]
        q = q[:, :-self.length]
        masked_r = masked_r[:, :-self.length]
        qshftemb, xemb = None, None
        qshftemb, xemb = self.base_emb(q, masked_r, qry)
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
        for i in range(self.num_blocks):
            xemb = self.blocks[i](qshftemb, xemb, xemb)
        if self.trans:
            if self.joint:
                p = self.pred(self.dropout_layer(xemb))
                seq_len = p.size(1)
                mid = seq_len // 2
                p[:, mid:, :] = p[:, mid:mid+1, :].expand(-1, seq_len - mid, -1)
                p = torch.sigmoid(p)
                p = (p * one_hot(cshft.long(), self.num_skills)).sum(-1)
                p = p[:, mid:]
                rshft = r[:, self.length:]
                true = rshft[:, mid:].float()
            else:
                p = torch.sigmoid(self.pred(self.dropout_layer(xemb)))
                p = (p * one_hot(cshft.long(), self.num_skills)).sum(-1)
                true = r[:, self.length:].float()
        elif self.mask_future:
            p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
            p = p[:, -self.length:]
            true = r[:, -self.length:].float()
        else:
            p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
            true = r[:, self.length:].float()
        out_dict = {
            "pred": p,
            "true": true
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss, len(pred[mask]), true[mask].sum().item()

class Blocks(Module):
    def __init__(self, embedding_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(embedding_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(embedding_size)

        self.FFN = transformer_FFN(embedding_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(embedding_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default
        device = q.device
        causal_mask = ut_mask(seq_len = k.shape[0], device = device)
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb
    

class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(seq_len, device):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)

def pos_encode(seq_len, device):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0).to(device)


def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])