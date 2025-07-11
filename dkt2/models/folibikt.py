import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from einops import rearrange, repeat
from torch.nn.functional import one_hot


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class folibiKT(nn.Module):
    def __init__(self, mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, seq_len, embedding_size, num_blocks, dropout, d_ff=256, 
            kq_same=True, final_fc_dim=512, num_attn_heads=8, separate_qr=False, l2=1e-5, emb_type="qid_alibi", num_buckets=32,max_distance=100):
        super().__init__()
        """
        Input:
            embedding_size: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.mask_response = mask_response
        self.pred_last = pred_last
        self.mask_future = mask_future
        self.length = length
        self.trans = trans
        self.model_name = "folibikt"
        self.num_skills = num_skills
        self.dropout = dropout
        self.kq_same = kq_same
        self.num_questions = num_questions
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qr = separate_qr
        self.emb_type = emb_type
        embed_l = embedding_size

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if self.num_questions > 0:
            self.difficult_param = nn.Embedding(self.num_questions+1, 1)
            self.q_embed_diff = nn.Embedding(self.num_skills+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, embed_l)
        
        if emb_type.startswith("qid"):
            # num_skills+1 ,embedding_size
            self.q_embed = nn.Embedding(self.num_skills, embed_l)
            if self.separate_qr: 
                self.qa_embed = nn.Embedding(2*self.num_skills+1, embed_l) # interaction emb
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(num_skills=num_skills, num_blocks=num_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    embedding_size=embedding_size, d_feature=embedding_size / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=self.emb_type,num_buckets = self.num_buckets,max_distance = self.max_distance)

        if self.trans:
            self.out = nn.Sequential(
                nn.Linear(embedding_size + embed_l,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, self.num_skills)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(embedding_size + embed_l,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )
        self.loss_fn = nn.BCELoss(reduction="mean")
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions+1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  embedding_size# c_ct
        if self.separate_qr:
            qa_data = q_data + self.num_skills * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, embedding_size # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, feed_dict):
        pid_data = feed_dict['questions']
        r = feed_dict['responses']
        c = feed_dict['skills']
        attention_mask = feed_dict['attention_mask']
        q_data = c
        target = r * (r > -1).long()
        if self.trans:
            pid_data = pid_data[:, :-self.length]
            q_data = q_data[:, :-self.length]
            target = target[:, :-self.length]
            attention_mask = attention_mask[:, :-self.length]
            cshft = c[:, self.length:]
        elif self.mask_future:
            attention_mask[:, -self.length:] = 0
            pid_data = pid_data * attention_mask
            q_data = q_data * attention_mask
            target = target * attention_mask
        elif self.mask_response:
            attention_mask[:, -self.length:] = 0
            target = target * attention_mask

        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.num_questions > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data) 
            pid_embed_data = self.difficult_param(pid_data) 
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                target)  # f_(ct,rt) or #h_rt (qt, rt)
            if self.separate_qr:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # (q-response emb diff + question emb diff)

            c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2
        else:
            c_reg_loss = 0.

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        pooled_ques_score = (self.q_embed(q_data) * attention_mask.unsqueeze(-1)).sum(
            1
        ) / attention_mask.sum(-1).unsqueeze(-1)
        pooled_inter_score = (qa_embed_data * attention_mask.unsqueeze(-1)).sum(
            1
        ) / attention_mask.sum(-1).unsqueeze(-1)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        if self.trans:
            output = self.out(concat_q)
        else:
            output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        output = m(output)
        if self.trans:
            output = (output * one_hot(cshft.long(), self.num_skills)).sum(-1)
            true = r[:, self.length:].float()
        elif self.mask_future or self.pred_last or self.mask_response:
            output = output[:, -self.length:]
            true = r[:, -self.length:].float()
        else:
            output = output[:, self.length:]
            true = r[:, self.length:].float()
        if self.training:
            out_dict = {
                "pred": output,
                "true": true,
                "c_reg_loss": c_reg_loss,
            }
        else:
            out_dict = {
                "pred": output,
                "true": true,
                "c_reg_loss": c_reg_loss,
                "q_embed": pooled_ques_score,
                "qr_embed": pooled_inter_score,
            }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        c_reg_loss = out_dict["c_reg_loss"]
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss + c_reg_loss, len(pred[mask]), true[mask].sum().item()

class Architecture(nn.Module):
    def __init__(self, num_skills,  num_blocks, embedding_size, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len, emb_type,num_buckets,max_distance):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            embedding_size : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = embedding_size
        """
        self.embedding_size = embedding_size
        self.model_type = model_type
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.emb_type = emb_type

        if self.emb_type.find("sin") != -1:
            self.position_emb = SinePositionalEncoding(d_hid=self.embedding_size, n_position=seq_len)

        if model_type in {'folibikt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(embedding_size=embedding_size, d_feature=embedding_size // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type,num_buckets = self.num_buckets,max_distance = self.max_distance)
                for _ in range(num_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(embedding_size=embedding_size, d_feature=embedding_size // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type,num_buckets = self.num_buckets,max_distance = self.max_distance)
                for _ in range(num_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        if self.emb_type.find("sin") != -1:
            q_posemb = self.position_emb(q_embed_data)
            q_embed_data = q_embed_data + q_posemb
            qa_posemb = self.position_emb(qa_embed_data)
            qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1: 
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data) # yt^
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data) # False
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data) # True
                flag_first = True
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embedding_size, d_feature,
                 d_ff, n_heads, dropout,  kq_same, emb_type,num_buckets,max_distance):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            embedding_size, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type,num_buckets = self.num_buckets,max_distance = self.max_distance)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embedding_size, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, embedding_size)

        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        device = query.device
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, d_feature, n_heads, dropout, kq_same, num_buckets,max_distance, bias=True, emb_type="qid"):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.embedding_size = embedding_size
        self.d_k = d_feature
        self.emb_type = emb_type

        if emb_type.find("t5") != -1:
            self.rel_pos_bias = T5RelativePositionBias(scale = embedding_size ** 0.5, causal = True, num_buckets=num_buckets, max_distance=max_distance)
        else:
            self.rel_pos_bias = None
        if emb_type.find("rotary") != -1:
            self.rotary_pe = RotaryPositionalEmbeddings(self.d_k)
        else:
            self.rotary_pe=None

        if emb_type.endswith("avgpool"):
            # pooling
            #self.pool =  nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            pool_size = 3
            self.pooling =  nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            self.out_proj = nn.Linear(embedding_size, embedding_size, bias=bias)
        elif emb_type.endswith("linear"):
            # linear
            self.linear = nn.Linear(embedding_size, embedding_size, bias=bias)
            self.out_proj = nn.Linear(embedding_size, embedding_size, bias=bias)
        elif emb_type.startswith("qid"):
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
            self.k_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(embedding_size, embedding_size, bias=bias)
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)
            self._reset_parameters()

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))  # 2*(-(8 / n))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        
        # calculate linear bias
        maxpos = 1000
        attn_heads = n_heads  
        context_position = torch.arange(maxpos)[:, None].cuda()
        memory_position = torch.arange(maxpos)[None, :].cuda()
        relative_position = memory_position - context_position 
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(attn_heads, -1,-1)

        self.slopes = torch.Tensor(get_slopes(attn_heads)).cuda()*-1
        self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * relative_position
        self.alibi = self.alibi.view(1, attn_heads, maxpos, maxpos)


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            # constant_(self.attnlinear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):

        bs = q.size(0)

        if self.emb_type.endswith("avgpool"):
            # v = v.transpose(1,2)
            scores = self.pooling(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)#.contiguous().view(bs, -1, self.embedding_size)
        elif self.emb_type.endswith("linear"):
            # v = v.transpose(1,2)
            scores = self.linear(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)
        elif self.emb_type.startswith("qid"):
            # perform linear operation and split into h heads

            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # transpose to get dimensions bs * h * sl * embedding_size

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            # calculate attention using function we will define next
            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None

            scores = attention(q, k, v, self.d_k,mask, self.dropout, zero_pad, gammas, pdiff, alibi = self.alibi,emb_type=self.emb_type, rel_pos_bias=self.rel_pos_bias, rotary_pe=self.rotary_pe)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.embedding_size)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        device = scores.device
        if zero_pad:
            # # need: torch.Size([64, 1, 200]), scores: torch.Size([64, 200, 200]), v: torch.Size([64, 200, 32])
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1) 
        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None, alibi=None, emb_type=None, rel_pos_bias=None, rotary_pe=None):
    """
    This is called by Multi-head atention object to find the values.
    """

    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    device = q.device

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    if emb_type.find("alibi") != -1:
        seq_len = scores.size()[-1]
        scores = scores+alibi[:, :, :seq_len, :seq_len]
        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
            scores_ = scores_ * mask.float().to(device)
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
            disttotal_scores = torch.sum(
                scores_, dim=-1, keepdim=True)
            # print(f"distotal_scores: {disttotal_scores}")
            position_effect = torch.abs(
                x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)
            # bs, 8, sl, sl positive distance
            dist_scores = torch.clamp(
                (disttotal_scores-distcum_scores)*position_effect, min=0.)
            dist_scores = dist_scores.sqrt().detach()
        m = nn.Softplus()
        gamma = -1. * m(gamma).unsqueeze(0) 
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        if pdiff == None:
            total_effect = torch.clamp(torch.clamp(
                (dist_scores*gamma).exp(), min=1e-5), max=1e5)
        else:
            diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
            diff = diff.sigmoid().exp()
            total_effect = torch.clamp(torch.clamp(
                (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5)
        scores = scores * total_effect

    else:
        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
            scores_ = scores_ * mask.float().to(device)
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
            disttotal_scores = torch.sum(
                scores_, dim=-1, keepdim=True) 
            # print(f"distotal_scores: {disttotal_scores}")
            position_effect = torch.abs(
                x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)
            # bs, 8, sl, sl positive distance
            dist_scores = torch.clamp(
                (disttotal_scores-distcum_scores)*position_effect, min=0.) 
            dist_scores = dist_scores.sqrt().detach()
        m = nn.Softplus()
        gamma = -1. * m(gamma).unsqueeze(0)  
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        if pdiff == None:
            total_effect = torch.clamp(torch.clamp(
                (dist_scores*gamma).exp(), min=1e-5), max=1e5) 
        else:
            diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
            diff = diff.sigmoid().exp()
            total_effect = torch.clamp(torch.clamp(
                (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5) 
        scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) 
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, embedding_size)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() *
                             -(math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)



class SinePositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(SinePositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        n_position = 1000

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = True,
        num_buckets = 16,
        max_distance = 50
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 16,
        max_distance = 50
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale
    
class RotaryPositionalEmbeddings(nn.Module):
    """
    ## [RoPE embeddings](../rope/index.html)

    *We use rotary position embeddings in self-attention layers.
    We assume the positional information gets embedded in embeddings
    and therefore not use them in causal attention.
    [Non-causal self-attention needs explicit positional information
     because it cannot infer it](https://papers.labml.ai/paper/3999902edc8511eba3db37f65e372566).*
    """

    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        self.theta = nn.Parameter(1. / (base ** (torch.arange(0, d, 2).float() / d)), requires_grad=False)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[ batch_size, seq_len, n_heads, d]`
        """
        # Extract the shape
        batch_size, seq_len, n_heads, d = x.shape

        # $\frac{d}{2}$
        d_2 = d // 2

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).type_as(self.theta)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum('n,d->nd', seq_idx, self.theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta 0, m \theta 1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., -x^{(\frac{d}{2})}]$
        neg_half_x = torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

        # Calculate
        #
        # \begin{align}
        # \begin{pmatrix}
        # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
        # \end{pmatrix} \\
        # \end{align}
        #
        # for $i \in {1, 2, ..., \frac{d}{2}}$
        rx = (x * idx_theta2.cos()[None, :, None, :]) + (neg_half_x * idx_theta2.sin()[None, :, None, :])

        #
        return rx