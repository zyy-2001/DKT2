import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, BCELoss
from torch.nn.functional import one_hot

# device = "cpu" if not torch.cuda.is_available() else "cuda"

class DKTForget(Module):
    def __init__(self, mask_future, length, device, num_skills, num_rgap, num_sgap, num_pcount, embedding_size, dropout=0.1):
        super().__init__()
        self.mask_future = mask_future
        self.length = length
        self.device = device

        self.num_skills = num_skills
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size

        self.interaction_emb = Embedding(self.num_skills * 2, self.embedding_size)

        self.c_integration = CIntegration(self.device, num_rgap, num_sgap, num_pcount, embedding_size).to(self.device)
        ntotal = num_rgap + num_sgap + num_pcount
    
        self.lstm_layer = LSTM(self.embedding_size + ntotal, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size + ntotal, self.num_skills)

        self.loss_fn = BCELoss(reduction="mean")
        

    def forward(self, feed_dict):
        q = feed_dict['skills']
        r = feed_dict['responses']
        attention_mask = feed_dict['attention_mask']
        masked_r = r * (r > -1).long()
        q_input = q[:, :-self.length]
        r_input = masked_r[:, :-self.length]
        q_shft = q[:, self.length:]
        r_shft = r[:, self.length:]
        q_input, r_input = q_input.to(self.device), r_input.to(self.device)
        x = q_input + self.num_skills * r_input
        rgaps, sgaps, pcounts = feed_dict["rgaps"], feed_dict["sgaps"], feed_dict["pcounts"]
        rgaps_input = rgaps[:, :-self.length]
        sgaps_input = sgaps[:, :-self.length]
        pcounts_input = pcounts[:, :-self.length]
        rgaps_shft = rgaps[:, self.length:]
        sgaps_shft = sgaps[:, self.length:]
        pcounts_shft = pcounts[:, self.length:]
        xemb = self.interaction_emb(x)
        theta_in = self.c_integration(xemb, rgaps_input.to(self.device).long(), sgaps_input.to(self.device).long(), pcounts_input.to(self.device).long())

        h, _ = self.lstm_layer(theta_in)
        theta_out = self.c_integration(h, rgaps_shft.to(self.device).long(), sgaps_shft.to(self.device).long(), pcounts_shft.to(self.device).long())
        theta_out = self.dropout_layer(theta_out)
        y = self.out_layer(theta_out)
        y = torch.sigmoid(y)

        y = (y * one_hot(q_shft.long(), self.num_skills)).sum(-1)

        if self.mask_future:
            y = y[:, -self.length:]
            r_shft = r_shft[:, -self.length:]
        
        out_dict = {
            "pred": y,
            "true": r_shft.float(),
        }
        return out_dict
    
    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss, len(pred[mask]), true[mask].sum().item()


class CIntegration(Module):
    def __init__(self, device, num_rgap, num_sgap, num_pcount, emb_dim) -> None:
        super().__init__()
        self.device = device
        self.rgap_eye = torch.eye(num_rgap).to(self.device)
        self.sgap_eye = torch.eye(num_sgap)
        self.pcount_eye = torch.eye(num_pcount)

        ntotal = num_rgap + num_sgap + num_pcount
        self.cemb = Linear(ntotal, emb_dim, bias=False)
        print(f"num_sgap: {num_sgap}, num_rgap: {num_rgap}, num_pcount: {num_pcount}, ntotal: {ntotal}")
        # print(f"total: {ntotal}, self.cemb.weight: {self.cemb.weight.shape}")

    def forward(self, vt, rgap, sgap, pcount):
        rgap, sgap, pcount = self.rgap_eye[rgap].to(self.device), self.sgap_eye[sgap].to(self.device), self.pcount_eye[pcount].to(self.device)
        # print(f"vt: {vt.shape}, rgap: {rgap.shape}, sgap: {sgap.shape}, pcount: {pcount.shape}")
        ct = torch.cat((rgap, sgap, pcount), -1) # bz * seq_len * num_fea
        # print(f"ct: {ct.shape}, self.cemb.weight: {self.cemb.weight.shape}")
        # element-wise mul
        Cct = self.cemb(ct) # bz * seq_len * emb
        # print(f"ct: {ct.shape}, Cct: {Cct.shape}")
        theta = torch.mul(vt, Cct)
        theta = torch.cat((theta, ct), -1)
        return theta