import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
import transformer.Constants as Constants
from transformer.Modules import ScaledDotProductAttention
# import Constants
# from Modules import ScaledDotProductAttention

import pdb
class ModSelfAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self,select_mod_n, n_head, d_model, d_k, d_v,input_n=29, S_n=3, mod_n=2, dropout=0.1, normalize_before=True,device=None):
        super().__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.input_n = input_n
        self.S_n=S_n
        self.mod_n=mod_n
        self.select_mod_n=select_mod_n
        self.list_w_qs = ModuleList( [nn.Linear(d_model, n_head * d_k, bias=False,device=device) for _ in range(mod_n)])#64をn*kにする重み。
        self.list_w_ks = ModuleList([nn.Linear(d_model, n_head * d_k, bias=False,device=device) for _ in range(mod_n)])
        self.list_w_vs = ModuleList([nn.Linear(d_model, n_head * d_v, bias=False,device=device) for _ in range(mod_n)])
        # modalごとにWが定義されているとして、意味があるんだっけ？
        [nn.init.xavier_uniform_(wq.weight) for wq in self.list_w_qs]
        [nn.init.xavier_uniform_(wk.weight)for wk in self.list_w_ks]
        [nn.init.xavier_uniform_(wv.weight)for wv in self.list_w_vs]
        
        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    def find_use_weight(self,weight_list,nume):
        if self.select_mod_n==nume-1:#0,1,2==(0,1,2,3) -1
            return weight_list[0]
        elif self.select_mod_n<nume-1:
            return weight_list[nume-1]
        else:
            return weight_list[nume]
    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        input_size, Sn =self.input_n , self.S_n
        mod_n=self.mod_n
        select_mod_n=self.select_mod_n
        sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)
        
        residual = x
        ##k,vのnormalizeは? 
        if self.normalize_before:
            x = self.layer_norm(x)
        # x shape: b x (lq+ lS1 + lS2 + ... + lSmod_n) x d
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        
        list_separate_x= [tmp_x for tmp_x in [x[:,:input_size],x[:,input_size:input_size+Sn], x[:,input_size+Sn:input_size+Sn*2],x[:,input_size+Sn*2:]]]
        
        # for i, tmp_wq in enumerate(self.list_w_qs):
        #     q = tmp_wq(list_separate_x[i]).view(sz_b, len_q, n_head, d_k)
        # select_mod_n=0,tau select_mod_n==1 x , select_mod_n==2 y
        q= (torch.cat([ self.find_use_weight(self.list_w_qs,i)(tmp_x) for i, tmp_x in enumerate(list_separate_x)],dim=1)).view(sz_b, len_q, n_head, d_k)
        k= (torch.cat([ self.find_use_weight(self.list_w_ks,i)(tmp_x) for i, tmp_x in enumerate(list_separate_x)],dim=1)).view(sz_b, len_k, n_head, d_k)
        v= (torch.cat([ self.find_use_weight(self.list_w_vs,i)(tmp_x) for i, tmp_x in enumerate(list_separate_x)],dim=1)).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))#
        output += residual
        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

class Rep_Separate_SelfAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self,select_mod_n, n_head, d_model, d_k, d_v,input_n=29, S_n=3, mod_n=2, dropout=0.1, normalize_before=True,device=None):
        super().__init__()
        
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.input_n = input_n
        self.S_n=S_n# 
        self.mod_n=mod_n#mod数 2 経過時間とマーク
        self.select_mod_n=select_mod_n# 今のmod
        # ModuleList が抜けると model.eval() の判定から外される　。　そっちは多分大丈夫
        self.list_w_qs = ModuleList([nn.Linear(d_model, n_head * d_k, bias=False,device=device) for _ in range(mod_n)])#64をn*kにする重み。
        self.list_w_ks = ModuleList([nn.Linear(d_model, n_head * d_k, bias=False,device=device) for _ in range(mod_n)])
        self.list_w_vs = ModuleList([nn.Linear(d_model, n_head * d_v, bias=False,device=device) for _ in range(mod_n)])
        # modalごとにWが定義されている
        [nn.init.xavier_uniform_(wq.weight) for wq in self.list_w_qs]
        [nn.init.xavier_uniform_(wk.weight)for wk in self.list_w_ks]
        [nn.init.xavier_uniform_(wv.weight)for wv in self.list_w_vs]
        
        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x [256,35,64]
        
        history_input=x[:,:-(self.mod_n*self.S_n),:]
        rep=x[:,-(self.mod_n*self.S_n):,:]
        which_rep=[rep[:,:self.S_n,:],rep[:,self.S_n:,:]]
        mod_rep=which_rep[self.select_mod_n]
        other_rep=which_rep[int((self.select_mod_n+1)%2)]
        
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        input_size, Sn =self.input_n , self.S_n
        mod_n=self.mod_n
        select_mod_n=self.select_mod_n
        sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)
        
        residual = x
        ##k,vのnormalizeは? 
        if self.normalize_before:
            x = self.layer_norm(x)
        # x shape: b x (lq+ lS1 + lS2 + ... + lSmod_n) x d
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        
        list_separate_x= [tmp_x for tmp_x in [x[:,:input_size],x[:,input_size:input_size+Sn], x[:,input_size+Sn:]]]
        select_weight=[self.select_mod_n,0,1]
        # for i, tmp_wq in enumerate(self.list_w_qs):
        #     q = tmp_wq(list_separate_x[i]).view(sz_b, len_q, n_head, d_k)
        # select_mod_n=0,tau select_mod_n==1 x , select_mod_n==2 y
        
        q= (torch.cat([ self.list_w_qs[select_weight[i]](tmp_x) for i, tmp_x in enumerate(list_separate_x)],dim=1)).view(sz_b, len_q, n_head, d_k)
        k= (torch.cat([ self.list_w_ks[select_weight[i]](tmp_x) for i, tmp_x in enumerate(list_separate_x)],dim=1)).view(sz_b, len_k, n_head, d_k)
        v= (torch.cat([ self.list_w_vs[select_weight[i]](tmp_x) for i, tmp_x in enumerate(list_separate_x)],dim=1)).view(sz_b, len_v, n_head, d_v)
        #pdb.set_trace()
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))#
        output += residual
        if not self.normalize_before:
            output = self.layer_norm(output)
        # with open(f"./param_grad/res1.log", 'w') as f:
        #             f.write(f"\n residual\n{residual}")
        #             f.write(f"\n output\n{output}")
        #             f.write(f"\n q\n{q}")
        #             #   f.write(f"\n enc_split\n{enc_split}")
        #             #   f.write(f"\n out\n{output}")
        # pdb.set_trace()
        return output, attn


class ModMultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self,DL_name, n_head, d_model, d_k, d_v, S_n=3, mod_n=3, dropout=0.1, normalize_before=True,device=None):
        super().__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.S_n=S_n
        self.mod_n=mod_n
        
        self.list_w_qs = ModuleList([nn.Linear(d_model, n_head * d_k, bias=False,device=device) for _ in range(mod_n)])#64をn*kにする重み。
        
        self.list_w_ks = ModuleList([nn.Linear(d_model, n_head * d_k, bias=False,device=device) for _ in range(mod_n)])
        
        self.list_w_vs = ModuleList([nn.Linear(d_model, n_head * d_v, bias=False,device=device) for _ in range(mod_n)])
        self.DL_name=DL_name
        [nn.init.xavier_uniform_(wq.weight) for wq in self.list_w_qs]
        [nn.init.xavier_uniform_(wk.weight)for wk in self.list_w_ks]
        [nn.init.xavier_uniform_(wv.weight)for wv in self.list_w_vs]
        
        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q,k,v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        mod_n=self.mod_n
        select_mod_n=self.select_mod_n
        sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)
        DL_name=self.DL_name
        
        residual = q
        ##k,vのnormalizeは? 
        if self.normalize_before:
            q = self.layer_norm(q)
        # x shape: b x (lq+ lS1 + lS2 + ... + lSmod_n) x d
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if DL_name=="phi":
            list_separate_q= [tmp_q for tmp_q in [q[:,0],q[:,1], q[:,2]]]
            list_separate_k= [tmp_k for tmp_k in [k[:,0],k[:,1], k[:,2]]]
            list_separate_v= [tmp_v for tmp_v in [v[:,0],v[:,1], v[:,2]]]
        else:
            list_separate_q= [tmp_q for tmp_q in [q[:,:Sn], q[:,Sn:Sn*2],q[:,Sn*2:]]]
            list_separate_k= [tmp_k for tmp_k in [k[:,:Sn], k[:,Sn:Sn*2],k[:,Sn*2:]]]
            list_separate_v= [tmp_v for tmp_v in [v[:,:Sn], v[:,Sn:Sn*2],v[:,Sn*2:]]]
        
        # for i, tmp_wq in enumerate(self.list_w_qs):
        #     q = tmp_wq(list_separate_x[i]).view(sz_b, len_q, n_head, d_k)
        # select_mod_n=0,tau select_mod_n==1 x , select_mod_n==2 y
        q= (torch.cat([ self.list_w_qs[i](tmp_x) for i, tmp_x in enumerate(list_separate_q)],dim=1)).view(sz_b, len_q, n_head, d_k)
        k= (torch.cat([ self.list_w_ks[i](tmp_x) for i, tmp_x in enumerate(list_separate_k)],dim=1)).view(sz_b, len_k, n_head, d_k)
        v= (torch.cat([ self.list_w_vs[i](tmp_x) for i, tmp_x in enumerate(list_separate_v)],dim=1)).view(sz_b, len_v, n_head, d_v)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))#
        output += residual
        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

class SelfAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)

        residual = x
        ##k,vのnormalizeは? 
        if self.normalize_before:
            x = self.layer_norm(x)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(x).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))#
        output += residual
        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        ##k,vのnormalizeは? 
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual
        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))#activation 
        x = self.dropout(x)# dropout1
        x = self.w_2(x)# linear projection
        x = self.dropout(x)# dropout2
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x

if __name__ == "__main__":
    #test mod SA
    mod_sa=ModSelfAttention(n_head=8,d_model=64,d_k=8,d_v=8)
    B=128
    len=29
    mod_n=3
    rep_n=3
    input=torch.ones((B,(len+mod_n*rep_n),64))
    mod_sa(input)
    