import torch

from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, CrossEntropyLoss
from torch.nn.functional import one_hot


class ATDKT(Module):
    def __init__(self, joint, mask_future, length, num_skills, num_questions, embedding_size, dropout=0.1, 
            num_layers=1, num_attn_heads=8, l1=0.5, l2=0.5, l3=0.5, start=50):
        super().__init__()
        self.joint = joint
        self.mask_future = mask_future
        self.length = length
        self.num_questions = num_questions
        self.num_skills = num_skills
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size

        self.interaction_emb = Embedding(self.num_skills * 2, self.embedding_size)

        self.lstm_layer = LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
            Linear(self.hidden_size//2, self.num_skills))


        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        if self.num_questions > 0:
            self.question_emb = Embedding(self.num_questions, self.embedding_size) # 1.2
        self.nhead = num_attn_heads
        d_model = self.hidden_size# * 2
        encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
        encoder_norm = LayerNorm(d_model)
        self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)

        # self.qlstm = LSTM(self.embedding_size, self.hidden_size, batch_first=True)

        self.qclasifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
            Linear(self.hidden_size//2, self.num_skills))
        self.concept_emb = Embedding(self.num_skills, self.embedding_size) # add concept emb

        self.closs = CrossEntropyLoss()
        self.start = start
        self.hisclasifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_size//2, 1))
        self.hisloss = nn.MSELoss()

        self.loss_fn = nn.BCELoss(reduction="mean")

    def predcurc(self, joint, dcur, q, c, r, xemb, train):
        y2, y3 = 0, 0
        qemb = self.question_emb(q)
        cemb = self.concept_emb(c)
        catemb = qemb + cemb

        mask = torch.triu(torch.ones(catemb.shape[1],catemb.shape[1]),diagonal=1).to(dtype=torch.bool).to(catemb.device)
        qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
        # qh, _ = self.qlstm(catemb)

        if train:
            sm = dcur["attention_mask"][:, self.length:].long()
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:])
            flag = sm[:,start:]==1
            y2 = self.closs(cpreds[flag], c[:,start:][flag])

        # predict response
        xemb = xemb + qh + cemb
        # xemb = xemb+qemb
        h, _ = self.lstm_layer(xemb)

        # predict history correctness rates
        rpreds = None
        if train:
            sm = dcur["attention_mask"][:, self.length:].long()
            start = self.start
            rpreds = torch.sigmoid(self.hisclasifier(h)).squeeze(-1)
            rsm = sm[:,start:]
            rflag = rsm==1
            rtrues = dcur["historycorrs"][:, :-self.length][:,start:]
            y3 = self.hisloss(rpreds[:,start:][rflag], rtrues[rflag])

        # predict response
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        if joint:
            seq_len = y.size(1)
            mid = seq_len // 2
            y[:, mid:, :] = y[:, mid:mid+1, :].expand(-1, seq_len - mid, -1) 
        y = torch.sigmoid(y)
        return y, y2, y3

    def forward(self, feed_dict): ## F * xemb
        # print(f"keys: {dcur.keys()}")
        q, c, r = feed_dict["questions"][:, :-self.length], feed_dict["skills"][:, :-self.length], feed_dict["responses"]
        cshft = feed_dict['skills'][:, self.length:]
        masked_r = r * (r > -1).long()
        r_input = masked_r[:, :-self.length]

        y2, y3 = 0, 0

        x = c + self.num_skills * r_input
        xemb = self.interaction_emb(x)
        y, y2, y3 = self.predcurc(self.joint, feed_dict, q, c, r_input, xemb, self.training)

        if self.joint:
            seq_len = y.size(1)
            mid = seq_len // 2
            preds = (y * one_hot(cshft, self.num_skills)).sum(-1)
            rshft = r[:, self.length:]
            preds = preds[:, mid:]
            true = rshft[:, mid:].float()
        else:
            preds = (y * one_hot(cshft, self.num_skills)).sum(-1)
            true = r[:, self.length:].float()
        if self.mask_future:
            preds = preds[:, -self.length:]
            true = r[:, -self.length:].float()

        out_dict = {
            "pred": preds,
            "true": true,
            "y": y,
            "y2": y2,
            "y3": y3,
        }
        return out_dict
    
    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        y2 = out_dict["y2"]
        y3 = out_dict["y3"]
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        loss = self.l1*loss+self.l2*y2+self.l3*y3
        return loss, len(pred[mask]), true[mask].sum().item()
  
