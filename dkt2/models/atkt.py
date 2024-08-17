# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.functional import one_hot
from torch.autograd import Variable, grad



class ATKT(nn.Module):
    def __init__(self, joint, mask_future, length, num_skills, skill_dim, answer_dim, hidden_dim, attention_dim=80, epsilon=10, beta=0.2, dropout=0.2):
        super(ATKT, self).__init__()
        self.joint = joint
        self.mask_future = mask_future
        self.length = length
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.num_skills = num_skills
        self.epsilon = epsilon
        self.beta = beta
        self.rnn = nn.LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim*2, self.num_skills)
        self.sig = nn.Sigmoid()
        
        self.skill_emb = nn.Embedding(self.num_skills+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0
        
        self.attention_dim = attention_dim
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        self.loss_fn = nn.BCELoss(reduction="mean")

    
    def attention_module(self, lstm_output):
        # lstm_output = lstm_output[0:1, :, :]
        # print(f"lstm_output: {lstm_output.shape}")
        att_w = self.mlp(lstm_output)
        # print(f"att_w: {att_w.shape}")
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        # print(f"att_w: {att_w.shape}")
        device = lstm_output.device
        attn_mask = ut_mask(lstm_output.shape[1], device)
        att_w = att_w.transpose(1,2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
        att_w = att_w.masked_fill_(attn_mask, float("-inf"))
        alphas = torch.nn.functional.softmax(att_w, dim=-1)
        attn_ouput = torch.bmm(alphas, lstm_output)

        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        # print(f"attn_ouput: {attn_ouput}")
        # print(f"attn_output_cum: {attn_output_cum}")
        attn_output_cum_1=attn_output_cum-attn_ouput
        # print(f"attn_output_cum_1: {attn_output_cum_1}")
        # print(f"lstm_output: {lstm_output}")

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        # import sys
        # sys.exit()

        return final_output


    def forward(self, feed_dict, perturbation=None):
        c = feed_dict['skills']
        r = feed_dict['responses']
        masked_r = r * (r > -1).long()
        skill = c[:, :-self.length]
        answer = masked_r[:, :-self.length]
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)
        
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)
        
        # print(skill_answer_embedding)
        
        skill_answer_embedding1=skill_answer_embedding
        if  perturbation is not None:
            skill_answer_embedding += perturbation
            
        out,_ = self.rnn(skill_answer_embedding)
        # print(f"out: {out.shape}")
        out=self.attention_module(out)
        # print(f"after attn out: {out.shape}")
        output = self.fc(self.dropout_layer(out))
        if self.joint:
            seq_len = output.size(1)
            mid = seq_len // 2
            output[:, mid:, :] = output[:, mid:mid+1, :].expand(-1, seq_len - mid, -1)

        res = self.sig(output)
        if self.joint:
            preds = (res * one_hot(c[:, self.length:].long(), self.num_skills)).sum(-1)
            rshft = r[:, self.length:]
            true = rshft[:, mid:].float()
            preds = preds[:, mid:]
        else:
            preds = (res * one_hot(c[:, self.length:].long(), self.num_skills)).sum(-1)
            true = r[:, self.length:].float()
        if self.mask_future:
            preds = preds[:, -self.length:]
            true = r[:, -self.length:].float()
        # res = res[:, :-1, :]
        # pred_res = self._get_next_pred(res, skill)
        out_dict = {
            "pred": preds,
            "true": true,
            "features": skill_answer_embedding1
        }
        return out_dict
    
    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        features = out_dict["features"]
        device = features.device
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(self.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        new_out_dict = self(feed_dict, p_adv)
        pred_res = new_out_dict["pred"].flatten()
        # second loss
        adv_loss = self.loss_fn(pred_res[mask], true[mask])
        loss = loss + self.beta * adv_loss
        return loss, len(pred[mask]), true[mask].sum().item()

from torch.autograd import Variable

def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)

def ut_mask(seq_len, device):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)