import os

import numpy as np
import torch
import torch.nn as nn


from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot


class DKTPlus(Module):
    def __init__(self, mask_future, length, num_skills, lambda_r=0.01, lambda_w1=0.003, lambda_w2=3.0, embedding_size=64, dropout=0.1):
        super().__init__()
        self.mask_future = mask_future
        self.length = length
        self.num_skills = num_skills
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2
        self.emb_size = embedding_size
        self.hidden_size = embedding_size
        self.interaction_emb = Embedding(self.num_skills * 2, self.emb_size)

        # self.lstm_layer = RNN(self.emb_size, self.hidden_size, batch_first=True)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_skills)

        self.loss_fn = nn.BCELoss(reduction="mean")
        

    def forward(self, feed_dict):
        q = feed_dict['skills']
        r = feed_dict['responses']
        attention_mask = feed_dict['attention_mask'][:, self.length:]
        masked_r = r * (r > -1).long()
        q_input = q[:, :-self.length]
        r_input = masked_r[:, :-self.length]
        x = q_input + self.num_skills * r_input
        xemb = self.interaction_emb(x)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        if self.mask_future:
            pred = (y * one_hot(q[:, self.length:].long(), self.num_skills)).sum(-1)
            pred = pred[:, -self.length:]
            r_shft = r[:, self.length:]
            true = r_shft[:, -self.length:].float()
        else:
            pred = (y * one_hot(q[:, self.length:].long(), self.num_skills)).sum(-1)
            true = r[:, self.length:].float()
        
        out_dict = {
            "pred": pred,
            "true": true,
            "y": y,
            "y_curr": (y * one_hot(q_input.long(), self.num_skills)).sum(-1),
            "y_next": (y * one_hot(q[:, self.length:].long(), self.num_skills)).sum(-1),
            "r_curr": r_input,
            "r_next": r[:, self.length:],
            "attention_mask": attention_mask,
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        sm = out_dict["attention_mask"].bool()
        y = out_dict["y"]
        y_curr = out_dict["y_curr"]
        y_next = out_dict["y_next"]
        r_curr = out_dict["r_curr"]
        r_next = out_dict["r_next"]
        y_curr = torch.masked_select(y_curr, sm)
        y_next = torch.masked_select(y_next, sm)
        r_curr = torch.masked_select(r_curr, sm)
        r_next = torch.masked_select(r_next, sm)
        loss = self.loss_fn(y_next.double(), r_next.double())

        loss_r = self.loss_fn(y_curr.double(), r_curr.double()) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(y[:, self.length:] - y[:, :-self.length], p=1, dim=-1), sm[:, self.length:])
        loss_w1 = loss_w1.mean() / self.num_skills
        loss_w2 = torch.masked_select(torch.norm(y[:, self.length:] - y[:, :-self.length], p=2, dim=-1) ** 2, sm[:, self.length:])
        loss_w2 = loss_w2.mean() / self.num_skills

        loss = loss + self.lambda_r * loss_r + self.lambda_w1 * loss_w1 + self.lambda_w2 * loss_w2
        return loss, len(y_next), r_next.sum().item()