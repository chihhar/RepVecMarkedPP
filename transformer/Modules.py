import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        #q,k,v [B,n,l,div]
        #attn:[B,n_head,len,len] !q.shape torch.Size([256, 8, 32, 16])
        #mask.shape torch.Size([256, 1, 32, 32])
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))#attn.shape torch.Size([256, 4, 32, 32])
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)#maskがtrueのところが全て-1e9 headごとに同じ

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
