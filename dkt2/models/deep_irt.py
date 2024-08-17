import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot


class DeepIRT(Module):
    def __init__(self, mask_response, pred_last, mask_future, length, trans, num_skills, dim_s, size_m, dropout=0.2):
        super().__init__()
        self.mask_response = mask_response
        self.pred_last = pred_last
        self.mask_future = mask_future
        self.length = length
        self.trans = trans
        self.num_skills = num_skills
        self.dim_s = dim_s
        self.size_m = size_m

        self.k_emb_layer = Embedding(self.num_skills, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_skills * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)
        
        if self.trans:
            self.diff_layer = nn.Sequential(Linear(self.dim_s,self.num_skills),nn.Tanh())
            self.ability_layer = nn.Sequential(Linear(self.dim_s,self.num_skills),nn.Tanh())
        else:
            self.diff_layer = nn.Sequential(Linear(self.dim_s,1),nn.Tanh())
            self.ability_layer = nn.Sequential(Linear(self.dim_s,1),nn.Tanh())

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, feed_dict):
        q = feed_dict['skills']
        r = feed_dict['responses']
        masked_r = r * (r > -1).long()
        if self.trans:
            cshft = q[:, self.length:]
            q = q[:, :-self.length]
            masked_r = masked_r[:, :-self.length]
        elif self.mask_future:
            attention_mask = feed_dict["attention_mask"]
            attention_mask[:, -self.length:] = 0
            q = q * attention_mask
            masked_r = r * attention_mask
        elif self.mask_response:
            attention_mask = feed_dict["attention_mask"]
            attention_mask[:, -self.length:] = 0
            masked_r = r * attention_mask
            

        batch_size = q.shape[0]
        x = q + self.num_skills * masked_r
        k = self.k_emb_layer(q)#question embedding
        v = self.v_emb_layer(x)#q,a embedding
        
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
 
        stu_ability = self.ability_layer(self.dropout_layer(f))#equ 12
        que_diff = self.diff_layer(self.dropout_layer(k))#equ 13

        p = torch.sigmoid(3.0*stu_ability-que_diff)#equ 14
        if self.trans:
            p = (p * one_hot(cshft.long(), self.num_skills)).sum(-1)
            true = r[:, self.length:].float()
        elif self.mask_future or self.pred_last or self.mask_response:
            p = p.squeeze(-1)
            p = p[:, -self.length:]
            true = r[:, -self.length:].float()
        else:
            p = p.squeeze(-1)
            p = p[:, self.length:]
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