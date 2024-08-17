import os

import numpy as np
import torch
from torch import nn


from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
from torch import Tensor
from typing import Optional
from enum import IntEnum



import math
import torch.nn.functional as fn
from torch.nn.functional import one_hot

class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            self.default_weight_init(linear.weight)
            self.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

        self.net = torch.nn.Sequential(*modules)

    # trunk model init
    def default_weight_init(self, tensor):
        torch.nn.init.xavier_uniform(tensor)
        # torch.nn.init.kaiming_normal_(tensor)


    def default_bias_init(self, tensor):
        torch.nn.init.constant_(tensor, 0)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class Mamba4KT(Module):
    def __init__(self, joint, length, num_skills, num_questions, embedding_size, num_attn_heads, num_blocks, d_state, d_conv, expand, dropout=0.1):
        super().__init__()
        self.joint = joint
        self.len = length
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.num_blocks = num_blocks
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout= dropout
        self.mamba_states = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                bimamba_type='none',
                dropout=self.dropout,
                num_blocks=self.num_blocks,
            ) for _ in range(self.num_blocks)
        ])

        if self.num_questions > 0:
            self.question_difficult = nn.Embedding(self.num_questions + 1, self.embedding_size)
            self.concept_diff = nn.Embedding(self.num_skills + 1, self.embedding_size)
            self.answer_diff = nn.Embedding(2 * self.num_skills + 1, self.embedding_size)
            
        self.concept_encoder = nn.Embedding(self.num_skills, self.embedding_size)
        self.answer_encoder = nn.Embedding(2, self.embedding_size)
        # self.state_encoder = nn.Embedding(2 * self.num_skills + 1, self.embedding_size)
        self._mlp_trans = StackedDense(
            self.embedding_size,
            [self.hidden_size] * 2,
            ([torch.nn.Tanh] * (1)) + [None]
        )
        self.dropout_layer = Dropout(self.dropout)
        self.out_layer = Linear(self.hidden_size, self.num_skills)
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, feed_dict):
        '''
        c: [batch_size, seq_len]
        r: [batch_sze, seq_len]
        q: [batch_sze, seq_len]
        '''
        q = feed_dict['questions']
        c = feed_dict['skills']
        r = feed_dict['responses']
        cshft = c[:, self.len:]
        rshft = r[:, self.len:]
        masked_r = r * (r > -1).long()
        q_input = q[:, :-self.len]
        c_input = c[:, :-self.len]
        r_input = masked_r[:, :-self.len]
        concept_emb = self.concept_encoder(c_input)
        state = self.answer_encoder(r_input) + concept_emb
        if self.num_questions > 0: # have problem id
            concept_diff = self.concept_diff(c_input) 
            question_difficult = self.question_difficult(q_input) 
            concept_emb = concept_emb + question_difficult * concept_diff  # uq *d_ct + c_ct # question encoder

            answer_difficult = self.answer_diff(r_input)
            state = state + question_difficult * answer_difficult
            reg_loss = (question_difficult ** 2.0).sum()
        else:
            reg_loss = 0.
  
        y = state
        for i in range(self.num_blocks):
            y = self.mamba_states[i](y)
        
        # y = self.dropout_layer(y)
        y = self.out_layer(y)
        if self.joint:
            seq_len = y.size(1)
            mid = seq_len // 2
            y[:, mid:, :] = y[:, mid:mid+1, :].expand(-1, seq_len - mid, -1)
        y = torch.sigmoid(y)
        y = (y * one_hot(cshft.long(), self.num_skills)).sum(-1)
        if self.joint:
            y = y[:, mid:]
            rshft = rshft[:, mid:]
        out_dict = {
            "pred": y,
            "true": rshft.float(),
            "reg_loss": reg_loss,
        }
        return out_dict
    
    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        reg_loss = out_dict["reg_loss"]
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask]) + reg_loss
        return loss, len(pred[mask]), true[mask].sum().item()

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, bimamba_type, dropout, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.bimamba_type = bimamba_type
        self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type=bimamba_type,
            )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = RMSNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
    
    def forward(self, input_tensor):

        hidden_states = self.mamba(input_tensor)
        if self.num_blocks == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:                           # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states
    
class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = RMSNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    