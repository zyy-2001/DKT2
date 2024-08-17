import os

import numpy as np
import torch
import torch.nn as nn


from torch.nn import Module, Embedding, LSTM, Linear, Dropout, ReLU, ModuleList, Sequential, LayerNorm

from torch.nn.functional import one_hot, pairwise_distance
import torch.nn.functional as F
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig

# from .xLSTM.xLSTM import xLSTM
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class DKT2(Module):
    def __init__(self, joint, mask_future, length, num_skills, num_questions, batch_size, seq_len, device, factor=1.3, num_blocks=2, num_heads=2, slstm_at=[1], conv1d_kernel_size=4, qkv_proj_blocksize=4, embedding_size=64, dropout=0.1):
        super().__init__()
        self.joint = joint
        self.mask_future = mask_future
        self.length = length
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.dropout = dropout
        self.device = device
        self.factor = factor
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.slstm_at = slstm_at
        self.conv1d_kernel_size = conv1d_kernel_size
        self.qkv_proj_blocksize = qkv_proj_blocksize

        # self.interaction_emb = Embedding(self.num_skills * 2, self.embedding_size)

        # self.lstm_layer = RNN(self.embedding_size, self.hidden_size, batch_first=True)

        if self.num_questions > 0:
            self.difficult_param = nn.Embedding(self.num_questions+1, 1)
            self.q_embed_diff = nn.Embedding(self.num_skills+1, self.embedding_size)
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, self.embedding_size)
        else:
            self.difficult_param = nn.Embedding(self.num_skills+1, 1)
            self.q_embed_diff = nn.Embedding(self.num_skills+1, self.embedding_size)
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, self.embedding_size) 

        
        # num_skills+1 ,embedding_size
        self.q_embed = nn.Embedding(self.num_skills, self.embedding_size)
        # if self.separate_qr: 
        #     self.qa_embed = nn.Embedding(2*self.num_skills+1, self.embedding_size) # interaction emb
        # else: # false default
        self.qa_embed = nn.Embedding(2, self.embedding_size)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=self.conv1d_kernel_size,
                    qkv_proj_blocksize=self.qkv_proj_blocksize,
                    num_heads=self.num_heads,
                    proj_factor=self.factor,
                    dropout=self.dropout,
                    embedding_dim=self.embedding_size,
                    _inner_embedding_dim=2*self.embedding_size,
                    _num_blocks=1,
                    round_proj_up_dim_up=True,
                    # round_proj_up_to_multiple_of=64,
                    # internal
                    _proj_up_dim=None,  # will be computed from embedding_dim and proj_factor
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=self.num_heads,
                    conv1d_kernel_size=self.conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent", # "powerlaw_blockdependent", "small_init", "standard"
                    recurrent_weight_init="zeros", # "zeros", "standard"
                    embedding_dim=self.embedding_size,
                    # hidden_size=3*self.embedding_size,
                    dropout=self.dropout,
                    group_norm_weight=True,
                    # num_gates=4,
                    # this option cuts of the gradient for recurrent connection, i.e. no exploding gradient if False
                    gradient_recurrent_cut=False,
                    # this option clips the gradient values for recurrent connections at dy
                    gradient_recurrent_clipval=None,
                    # this option clips the y value
                    forward_clipval=None,
                    # this can be ignored internally, but may be used to optimize kernels
                    batch_size=self.batch_size,
                ),
                feedforward=FeedForwardConfig(proj_factor=self.factor, act_fn="relu"),
            ),
            context_length=seq_len-1,
            num_blocks=self.num_blocks,
            embedding_dim=self.embedding_size,
            add_post_blocks_norm=True,
            bias=True,
            dropout=self.dropout,
            # The block indices at which sLSTM blocks are placed.
            # Indexing starts from 0.
            slstm_at=self.slstm_at,
            # _block_map is a string that specifies which block is used at which position
            # 0: use the mLSTM block
            # 1: use the sLSTM block
        )
        
        self.xlstm_stack = xLSTMBlockStack(cfg).to(device)


        # self.xlstm_layers = nn.ModuleList([
        #     xLSTM("m", torch.zeros(batch_size, seq_len, self.hidden_size).to(device),
        #           factor=factor, depth=depth).to(device)
        # ])
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_skills)
        self.loss_fn = nn.BCELoss(reduction="mean")

        self.out = Sequential(
            Linear(2 * self.embedding_size + 2 * self.hidden_size, 2 * self.hidden_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(2 * self.hidden_size, self.hidden_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.hidden_size, self.num_skills),
        )
        self.lambda_r = 0.01
        self.lambda_w1 = 0.003
        self.lambda_w2 = 3.0
        

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions+1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  embedding_size# c_ct
        # if self.separate_qr:
        #     qa_data = q_data + self.num_skills * target
        #     qa_embed_data = self.qa_embed(qa_data)
        # else:
        # BS, seqlen, embedding_size # c_ct+ g_rt =e_(ct,rt)
        qa_embed_data = self.qa_embed(target)+q_embed_data
        # qa_embed_data = self.qa_embed(target)
        return q_embed_data, qa_embed_data


    def forward(self, feed_dict):
        pid_data = feed_dict['questions'][:, :-self.length]
        r = feed_dict['responses']
        c = feed_dict['skills']
        attention_mask = feed_dict['attention_mask'][:, self.length:]
        q_data = c[:, :-self.length]
        q_shft = c[:, self.length:]
        r_shft = r[:, self.length:]
        target = (r * (r > -1).long())[:, :-self.length]
        # Batch First
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        q_embed_diff_data = self.q_embed_diff(q_data) 
        pid_embed_data = self.difficult_param(pid_data) 
        q_embed_data = q_embed_data + pid_embed_data * \
            q_embed_diff_data 

        qa_embed_diff_data = self.qa_embed_diff(
            target) 
        qa_embed_data = qa_embed_data + pid_embed_data * \
            qa_embed_diff_data

        
        qa_embed_data = self.dropout_layer(qa_embed_data)
        d_output = self.xlstm_stack(qa_embed_data)
        familiar_ability = torch.zeros_like(d_output)
        unfamiliar_ability = torch.zeros_like(d_output)
        familiar_position = target == 1
        unfamiliar_position = target == 0
        familiar_ability[familiar_position] = d_output[familiar_position]
        unfamiliar_ability[unfamiliar_position] = d_output[unfamiliar_position]



        d_output = (d_output - pid_embed_data)
        concat_q = torch.cat([d_output, q_embed_data, familiar_ability, unfamiliar_ability], dim=-1) 
        output = self.out(concat_q)
        if self.joint:
            seq_len = q_data.size(1)
            mid = seq_len // 2
            output[:, mid:, :] = output[:, mid:mid+1, :].expand(-1, seq_len - mid, -1)
            

        output = torch.sigmoid(output)

        output = (output * one_hot(q_shft.long(), self.num_skills)).sum(-1)
        if self.mask_future:
            output = output[:, -self.length:]
            r_shft = r_shft[:, -self.length:]
        elif self.joint:
            output = output[:, mid:]
            r_shft = r_shft[:, mid:]
        out_dict = {
                "pred": output,
                "true": r_shft.float(),
            }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss, len(pred[mask]), true[mask].sum().item()


class Architecture(nn.Module):
    def __init__(self, xlstm_layer, d_model, dropout=0.2):
        super().__init__()
        self.xlstm_block = xlstm_layer
        self.w_1 = nn.Linear(d_model, d_model)
        self.w_2 = nn.Linear(d_model, d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):

        hidden_states1 = self.activation(self.w_1(self.LayerNorm(input_tensor)))
        hidden_states2 = self.xlstm_block(self.w_2(self.LayerNorm(input_tensor)))
        return self.dropout(hidden_states2 * hidden_states1) + input_tensor