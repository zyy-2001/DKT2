from torch import cat,squeeze,unsqueeze,sum
from torch.nn import Embedding,Module,Sigmoid,Tanh,Dropout,Linear,Parameter
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class DIMKT(Module):
    def __init__(self,mask_future, length,trans,num_skills,num_questions,embedding_size,dropout,batch_size,difficult_levels=100):
        super().__init__()
        self.mask_future = mask_future
        self.length = length
        self.trans = trans
        self.num_questions = num_questions  
        self.num_skills = num_skills
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.difficult_levels = difficult_levels
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.dropout = Dropout(dropout)
        
        self.interaction_emb = Embedding(self.num_skills * 2, self.embedding_size)

        self.knowledge = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.embedding_size)), requires_grad=True)

        self.q_emb = Embedding(self.num_questions+1,self.embedding_size,padding_idx=0)
        self.c_emb = Embedding(self.num_skills+1,self.embedding_size,padding_idx=0)
        self.sd_emb = Embedding(self.difficult_levels+2,self.embedding_size,padding_idx=0)
        self.qd_emb = Embedding(self.difficult_levels+2,self.embedding_size,padding_idx=0)
        self.a_emb = Embedding(2,self.embedding_size)
        
        self.linear_1 = Linear(4*self.embedding_size,self.embedding_size)
        self.linear_2 = Linear(1*self.embedding_size,self.embedding_size)
        self.linear_3 = Linear(1*self.embedding_size,self.embedding_size)
        self.linear_4 = Linear(2*self.embedding_size,self.embedding_size)
        self.linear_5 = Linear(2*self.embedding_size,self.embedding_size)
        self.linear_6 = Linear(4*self.embedding_size,self.embedding_size)

        self.loss_fn = nn.BCELoss(reduction="mean")

        if self.trans or self.mask_future:
            self.out = nn.Sequential(
                nn.Linear(self.embedding_size,
                        self.embedding_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(self.embedding_size, 256), nn.ReLU(
                ), nn.Dropout(dropout),
                nn.Linear(256, self.num_skills)
            )
        
            
    def forward(self,feed_dict):
        q, c = feed_dict['questions'], feed_dict['skills']
        qd, sd = feed_dict['question_difficulty'], feed_dict['skill_difficulty']
        r = feed_dict['responses']

        q_input = q[:, :-self.length]
        c_input = c[:, :-self.length]
        sd_input = sd[:, :-self.length]
        qd_input = qd[:, :-self.length]
        masked_r = r * (r > -1).long()
        r_input = masked_r[:, :-self.length]
        rshft = masked_r[:, self.length:]
        qshft = q[:, self.length:]
        cshft = c[:, self.length:]
        qdshft = qd[:, self.length:]
        sdshft = sd[:, self.length:]

        if self.batch_size != len(q_input):
            self.batch_size = len(q_input)
        q_emb = self.q_emb(Variable(q_input))
        c_emb = self.c_emb(Variable(c_input))
        sd_emb = self.sd_emb(Variable(sd_input))
        qd_emb = self.qd_emb(Variable(qd_input))
        a_emb = self.a_emb(Variable(r_input))
        device = q_emb.device

        target_q = self.q_emb(Variable(qshft))
        target_c = self.c_emb(Variable(cshft))
        target_sd = self.sd_emb(Variable(sdshft))
        target_qd = self.qd_emb(Variable(qdshft))
       
        input_data = cat((q_emb,c_emb,sd_emb,qd_emb),-1)
        input_data = self.linear_1(input_data)

        target_data = cat((target_q,target_c,target_sd,target_qd),-1)
        target_data = self.linear_1(target_data)

        
        shape = list(sd_emb.shape)
        padd = torch.zeros(shape[0],1,shape[2],device=device)
        sd_emb = cat((padd,sd_emb),1)
        slice_sd_embedding = sd_emb.split(1,dim=1)

        shape = list(a_emb.shape)
        padd = torch.zeros(shape[0],1,shape[2],device=device)
        a_emb = cat((padd,a_emb),1)
        slice_a_embedding = a_emb.split(1,dim=1)

        shape = list(input_data.shape)
        padd = torch.zeros(shape[0],1,shape[2],device=device)
        input_data = cat((padd,input_data),1)
        slice_input_data = input_data.split(1,dim=1)

        qd_emb = cat((padd,qd_emb),1)
        slice_qd_embedding = qd_emb.split(1,dim=1)
        
        k = self.knowledge.repeat(self.batch_size,1).cuda()
        
        h = list()
        seqlen = q_input.size(1)
        for i in range(1,seqlen+1):
            
            sd_1 = squeeze(slice_sd_embedding[i],1)
            a_1 = squeeze(slice_a_embedding[i],1)
            qd_1 = squeeze(slice_qd_embedding[i],1)
            input_data_1 = squeeze(slice_input_data[i],1)
            
            qq = k-input_data_1

            gates_SDF = self.linear_2(qq)
            gates_SDF = self.sigmoid(gates_SDF)
            SDFt = self.linear_3(qq)
            SDFt = self.tanh(SDFt) 
            SDFt = self.dropout(SDFt)

            SDFt = gates_SDF*SDFt

            x = cat((SDFt,a_1),-1)
            gates_PKA = self.linear_4(x)
            gates_PKA = self.sigmoid(gates_PKA)

            PKAt = self.linear_5(x)
            PKAt = self.tanh(PKAt)

            PKAt = gates_PKA*PKAt

            ins = cat((k,a_1,sd_1,qd_1),-1) 
            gates_KSU = self.linear_6(ins)
            gates_KSU = self.sigmoid(gates_KSU)

            k = gates_KSU*k + (1-gates_KSU)*PKAt

            h_i = unsqueeze(k,dim=1)
            h.append(h_i)

        output = cat(h,axis = 1)

        if self.trans:
            logits = target_data*output
            logits = self.out(logits)
            logits = self.sigmoid(logits)
            logits = (logits * one_hot(cshft.long(), self.num_skills)).sum(-1)
            true = r[:, self.length:].float()
        elif self.mask_future:
            logits = target_data*output
            logits = self.out(logits)
            logits = self.sigmoid(logits)
            logits = (logits * one_hot(cshft.long(), self.num_skills)).sum(-1)
            logits = logits[:, -self.length:]
            true = r[:, -self.length:].float()
        else:
            logits = sum(target_data*output,dim = -1)
            logits = self.sigmoid(logits)
            true = r[:, self.length:].float()
        # y = self.sigmoid(logits)
        out_dict = {
            "pred": logits,
            "true": true,
        }
        return out_dict
    
    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss, len(pred[mask]), true[mask].sum().item()