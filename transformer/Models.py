import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn import ModuleList
from torch import zeros,ones
import kmeans
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, modEncLayer,CrossEncoderLayer
from transformer.Layers import DecoderLayer, modDecoderLayer

# import Constants
# from Layers import EncoderLayer
# from Layers import DecoderLayer

from matplotlib import pyplot as plt
import pickle
import gc
def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

class Encoder(nn.Module):
    # 最もベースのEncoder 廃止済み
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, device, normalize_before,mod_sample,is_bottle_neck=True,
            time_linear=False,train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0,n_dataset=2,start_fusion_layers=1,allcat=None):
        super().__init__()
        
        self.n_layers=n_layers
        self.start_fusion_layers=start_fusion_layers
        # self.d_model = d_model
        self.device=device
        # # position vector, used for temporal encoding
        
        if allcat:
            self.layer_stack_modal = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
            for _ in range(n_layers)])
        else:
            self.layer_stack_modal = ModuleList([
                ModuleList([
                modEncLayer(j,d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before,device=device) if start_fusion_layers==i+1 else EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
                for i in np.arange(n_layers)]) for j in np.arange(n_dataset)
            ])


        self.is_bottle_neck=is_bottle_neck
        
        btl_num=3
        self.bottle_neck = nn.Parameter(torch.randn(1,btl_num,d_model))
        
        
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            ])


    def forward(self, enc_input, rep_Mat=None,non_pad_mask=None,plot=None,imp=None,enc_plot=None,gene=None,allcat=None,emb_list=None):
        
        if allcat:
            batch_len=enc_input[0].shape[0]
            event_num=29
            if rep_Mat is not None:
                rep_num = rep_Mat[0].shape[1]# [B,rep_num]
                for mod_ind in np.arange(3):
                
                    if mod_ind==0:
                        tmp_input=torch.cat((enc_input[mod_ind], rep_Mat[mod_ind]) ,dim=1) # inputをimaginary rep vecとcat
                        tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                        mod_embs=tmp_emb
                    else:
                        tmp_emb=torch.cat((enc_input[mod_ind], rep_Mat[mod_ind]) ,dim=1)
                        for emb in emb_list[mod_ind]:
                            tmp_emb=emb(tmp_emb)
                        mod_embs=torch.cat([mod_embs,tmp_emb],dim=1)#[B,29+rep_num+29+rep_num+29+rep_num]
                #mod_embs=torch.cat(([emb_list[mod_ind](torch.cat((enc_input[mod_ind], rep_Mat[mod_ind]) ,dim=1)) for mod_ind in np.arange(3)]),dim=1)
            if non_pad_mask is not None:
                if rep_Mat is not None:
                    represent_mod_number=0
                    slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number])# input 合わせ
                    slf_attn_mask_subseq=torch.cat((torch.cat((slf_attn_mask_subseq,zeros((batch_len,rep_num,event_num),device=self.device)),dim=1),zeros((batch_len,rep_num+event_num,rep_num),device=self.device)),dim=2)
                    slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                    slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                    slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
            tmp_enc_output=zeros(mod_embs.shape,device=self.device)
            non_pad_mask=torch.cat((non_pad_mask,non_pad_mask,non_pad_mask),dim=1)
            slf_attn_mask=torch.cat((slf_attn_mask,slf_attn_mask,slf_attn_mask),dim=1)
            slf_attn_mask=torch.cat((slf_attn_mask,slf_attn_mask,slf_attn_mask),dim=2)
            
            for lyr_ind in np.arange(self.n_layers):
                tmp_enc_output+=mod_embs
                tmp_enc_output, _ = self.layer_stack_modal[lyr_ind](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
            
            mod_num=int(tmp_enc_output.shape[1]/3)
            outputlist=[tmp_enc_output[:,0:mod_num]]
            outputlist.append(tmp_enc_output[:,mod_num:mod_num+mod_num])
            outputlist.append(tmp_enc_output[:,mod_num+mod_num:])
            mod_enc_output=outputlist
            if self.normalize_before==True:
                for mod_ind in np.arange(3):
                    tmp_enc_output=mod_enc_output[mod_ind]
                    tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                    if mod_ind==0:
                        next_enc_output=[tmp_enc_output]
                        next_enc_output_append=next_enc_output.append
                    else:
                        next_enc_output_append(tmp_enc_output)
                enc_output=next_enc_output
            else:
                enc_output=mod_enc_output
            return enc_output
            
            
            
            
        #enc_input [modal ,[Batch; seq_len, Dim=1]]
        #rep_Mat [modal; [Batch; rep_num; Dim=1]]
        #        [modal; [B; rep_num;Dim=2]
        
        mod_len=len(enc_input)
        batch_len=enc_input[0].shape[0]
        event_num=enc_input[0].shape[1]
        if rep_Mat is not None:
            rep_num = rep_Mat[0].shape[1]# [B,rep_num]
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=torch.cat((enc_input[mod_ind], rep_Mat[mod_ind]) ,dim=1) # inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    mod_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=torch.cat((enc_input[mod_ind], rep_Mat[mod_ind]) ,dim=1)
                    for emb in emb_list[mod_ind]:
                            tmp_emb=emb(tmp_emb)
                    mod_embs=torch.cat([mod_embs,tmp_emb.unsqueeze(0)],dim=0)
        else:
            rep_num = 0
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=enc_input[mod_ind]# inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    mod_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=enc_input[mod_ind]
                    
                    for emb in emb_list[mod_ind]:
                            tmp_emb=emb(tmp_emb)
                    mod_embs=torch.cat([mod_embs,tmp_emb.unsqueeze(0)],dim=0)
            
        # Encoding inputs
        
        #initial_S=mod_embs[0][0,-3:,:]
        
        # Generating mask
        # """ Encode event sequences via masked self-attention. """
        # #入力の時間エンコーディング
        
        # tem_enc = torch.cat((tem_enc,tem_rep), dim=1)#(16,seqence-1,M)->(16,seq-1+rep_vec_num,M)
        
        # modal 別でもイベントの見えている関係性は同じだから...
        if non_pad_mask is not None:
            if rep_Mat is not None:
                represent_mod_number=0
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number])# input 合わせ
                slf_attn_mask_subseq=torch.cat((torch.cat((slf_attn_mask_subseq,zeros((batch_len,rep_num,event_num),device=self.device)),dim=1),zeros((batch_len,rep_num+event_num,rep_num),device=self.device)),dim=2)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
            else:
                represent_mod_number=0
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number])# input 合わせ
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=enc_input[represent_mod_number], seq_q=enc_input[represent_mod_number])
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None
        
        # enc_output = zeros(tem_enc.shape,device=self.device)
        
        # Attention
        mod_enc_output=[zeros(mod_embs[mod_i].shape,device=self.device) for mod_i in np.arange(mod_len)]
        
        #mod_enc_output: [[128,32,64],[128,64,32]]
        if self.is_bottle_neck==True:
            btl=self.bottle_neck.repeat([batch_len,1,1])
            btl_num=btl.shape[1]
        
        mask_change_flg=0
        #
        if plot==True and enc_plot==True:
            event_time_history=[enc_input[0][0]]
            event_x_history=[enc_input[1][0]]
            event_y_history=[enc_input[2][0]]
            with open(f"./pickled/enc_output/{gene}_event_time",mode="wb") as file:
                pickle.dump(event_time_history,file)
            with open(f"./pickled/enc_output/{gene}_event_x",mode="wb") as file:
                pickle.dump(event_x_history,file)
            with open(f"./pickled/enc_output/{gene}_event_y",mode="wb") as file:
                pickle.dump(event_y_history,file)
            time_enc_output_history=[[mod_embs[0],mod_embs[1],mod_embs[2]]]#(3=mod,256=batch,32=29+3,64=dim)
            if self.is_bottle_neck==True:
                share_enc_output_his=[btl]
            for lyr_ind in np.arange(self.n_layers):
                
                sum_btl=None
                lyr_ind_i = lyr_ind
                for mod_ind in np.arange(mod_len):
                    # modal to tmp
                    tmp_enc_output=mod_enc_output[mod_ind]
                    
                    tmp_enc_output+=mod_embs[mod_ind]
                    if (self.is_bottle_neck==True) and (lyr_ind_i>=(self.start_fusion_layers - 1)):
                        if self.is_bottle_neck==True:
                            #btlがあるときの入力作り直し btlが各モーダル入力の右に追加
                            tmp_enc_output=torch.cat((tmp_enc_output,btl),dim=1)
                        if mask_change_flg==0:
                            #btlがあるときのmask作り直し [B,L+S,L+S]->[B,L+S+btl,L+S+btl]
                            mask_change_flg=1
                            mask_Bts=zeros((batch_len,btl_num,slf_attn_mask_subseq.shape[2]),device=self.device)#0がマスクなし、1があり。最終行は0で構成.[B,btl,L] 
                            if rep_Mat is None:
                                mask_Bts[:,:,-1]+=1 #最後のイベントを見えないようにしないとbtlから間接的に真値情報がわたってしまう。
                            slf_attn_mask_subseq=torch.cat((torch.cat((slf_attn_mask_subseq,mask_Bts),dim=1),zeros((batch_len,event_num+rep_num+btl_num,btl_num),device=self.device)),dim=2)
                            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1))
                            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                            non_pad_mask = get_non_pad_mask(torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1))
                        
                    tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
                    
                    if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                        tmp_btl=tmp_enc_output[:,-btl_num:]
                        tmp_enc_output=tmp_enc_output[:,:-btl_num]
                    if mod_ind==0:
                        next_mod_enc_output=[tmp_enc_output]
                        next_mod_enc_output_append=next_mod_enc_output.append
                    else:
                        next_mod_enc_output_append(tmp_enc_output)
                    
                    if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                        if mod_ind==0:
                            sum_btl=tmp_btl
                        else:
                            sum_btl=sum_btl+tmp_btl
                
                if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                    btl=sum_btl/mod_len
                mod_enc_output=next_mod_enc_output
                time_enc_output_history.append(mod_enc_output)
                if self.is_bottle_neck==True:
                    share_enc_output_his.append(btl)
                
            if self.is_bottle_neck==True:
                with open(f"./pickled/enc_output/{imp}share",mode="wb") as file:
                    pickle.dump(share_enc_output_his,file)
            with open(f"./pickled/enc_output/{imp}proposed_encout",mode="wb") as file:
                pickle.dump(time_enc_output_history,file)
        else:
            for lyr_ind in np.arange(self.n_layers):
                # Transformer のLayer分繰り返し
                sum_btl=None
                lyr_ind_i = lyr_ind
                for mod_ind in np.arange(mod_len):
                    # modal 数分繰り返し
                    tmp_enc_output=mod_enc_output[mod_ind]
                    
                    tmp_enc_output+=mod_embs[mod_ind]
                    if (self.is_bottle_neck==True) and (lyr_ind_i>=(self.start_fusion_layers - 1)):
                        if self.is_bottle_neck==True:
                            #btlがあるときの入力作り直し btlが各モーダル入力の右に追加
                            tmp_enc_output=torch.cat((tmp_enc_output,btl),dim=1)
                        if mask_change_flg==0:
                            #btlがあるときのmask作り直し [B,L+S,L+S]->[B,L+S+btl,L+S+btl]
                            mask_change_flg=1
                            mask_Bts=zeros((batch_len,btl_num,slf_attn_mask_subseq.shape[2]),device=self.device)#0がマスクなし、1があり。最終行は0で構成.[B,btl,L] 
                            if rep_Mat is None:
                                mask_Bts[:,:,-1]+=1 #最後のイベントを見えないようにしないとbtlから間接的に真値情報がわたってしまう。
                            slf_attn_mask_subseq=torch.cat((torch.cat((slf_attn_mask_subseq,mask_Bts),dim=1),zeros((batch_len,event_num+rep_num+btl_num,btl_num),device=self.device)),dim=2)
                            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1))
                            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                            non_pad_mask = get_non_pad_mask(torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1))
                        
                    tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
                    
                    if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                        tmp_btl=tmp_enc_output[:,-btl_num:]
                        tmp_enc_output=tmp_enc_output[:,:-btl_num]
                    if mod_ind==0:
                        next_mod_enc_output=[tmp_enc_output]
                        next_mod_enc_output_append=next_mod_enc_output.append
                    else:
                        next_mod_enc_output_append(tmp_enc_output)
                    
                    if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                        if mod_ind==0:
                            sum_btl=tmp_btl
                        else:
                            sum_btl=sum_btl+tmp_btl
                if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                    btl=sum_btl/mod_len
                mod_enc_output=next_mod_enc_output
            
        # for enc_layer in self.layer_stack:
        #     enc_output += tem_enc
        #     enc_output, _ = enc_layer(
        #         enc_input=enc_output,
        #         non_pad_mask=non_pad_mask,
        #         slf_attn_mask=slf_attn_mask)
        
        if self.normalize_before==True:
            for mod_ind in np.arange(mod_len):
                tmp_enc_output=mod_enc_output[mod_ind]
                tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                if mod_ind==0:
                    next_enc_output=[tmp_enc_output]
                    next_enc_output_append=next_enc_output.append
                else:
                    next_enc_output_append(tmp_enc_output)
            enc_output=next_enc_output
        else:
            enc_output=mod_enc_output
        return enc_output#[[256,32,64],[],[]]

# early fusion方式
class EarlyMoveEncoder(nn.Module):
    # early-fusionのためのEncoder
    def __init__(
            self,
            d_model, d_inner, n_dataset,
            n_layers, n_head, d_k, d_v, dropout, device, normalize_before,mod_sample,is_bottle_neck=True,
            time_linear=False,train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0):
        super().__init__()
        
        self.n_layers=n_layers#4
        
        self.device=device
        
        
        self.n_marks=len(mod_sample)#2
        
        self.layer_stack_modal = ModuleList([
            ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
            for i in np.arange(n_layers)]) for j in np.arange(n_dataset)
        ])
        
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            for i in np.arange(n_layers)])
        self.modal_cat_list=ModuleList([nn.Linear(d_model,d_model, bias=False,device=device).to(device).double() for i in range(self.n_marks)])
    def forward(self, enc_input, rep_Mat=None,non_pad_mask=None,plot=None,imp=None,enc_plot=None,gene=None,allcat=None,emb_list=None):
            
        #enc_input [modal ,[Batch; seq_len, Dim=1]]
        # tau [ Batch, seq, dim=1]
        # marks[ Batch, seq, dim=marks(x,y,...)]
        #rep_Mat [modal; [Batch; rep_num; Dim=1]]
        #        [modal; [B; rep_num;Dim=2]
        mod_len=len(enc_input)
        batch_len=enc_input[0].shape[0]
        event_num=enc_input[0].shape[1]
        
        # input embedding
        for mod_ind in np.arange(mod_len):
            if mod_ind==0:
                tmp_input=enc_input[mod_ind] # inputをimaginary rep vecとcat
                tmp_emb=emb_list[mod_ind](tmp_input).to(torch.double) # embedding
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=tmp_emb.unsqueeze(0)
            else:
                tmp_emb=enc_input[mod_ind]
                for emb in emb_list[mod_ind]:
                    tmp_emb=emb(tmp_emb).to(torch.double)
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                
                input_embs=torch.cat([input_embs,tmp_emb.unsqueeze(0)],dim=0)
        dim=input_embs.shape[-1]
        input_embs=torch.concat([*input_embs],dim=1)#concat
        
        # repも入力と別で、embeddingして、置いておく
        if rep_Mat is not None:
            rep_num = rep_Mat[0].shape[1]# [B,rep_num]
            for mod_ind in np.arange(mod_len):
                if mod_ind==0:
                    tmp_input=rep_Mat[mod_ind] # inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=rep_Mat[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)
        else:
            rep_num = 0
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=enc_input[mod_ind]# inputをimaginary rep vecとcat
                    pdb.set_trace()
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=enc_input[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)#3,128,3,64
        rep_embs=torch.concat([*rep_embs],dim=1)#concat
        
        # Embedding 
        # input_embs[[256,29,64][][]]
        # rep_embs[[256,3,64][][]]
        # Encoding inputs
        
        #initial_S=mod_embs[0][0,-3:,:]
        
        # Generating mask
        # """ Encode event sequences via masked self-attention. """
        # #入力の時間エンコーディング
        
        # tem_enc = torch.cat((tem_enc,tem_rep), dim=1)#(16,seqence-1,M)->(16,seq-1+rep_vec_num,M)
        
        # modal 別でもイベントの見えている関係性は同じだから...
        if non_pad_mask is not None:
            if rep_Mat is not None:
                represent_mod_number=0
                #pdb.set_trace()
                
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                slf_attn_mask_subseq=slf_attn_mask_subseq.tile(2,2)
                
                slf_attn_mask_subseq=torch.cat(
                    (torch.cat((slf_attn_mask_subseq,zeros((batch_len,rep_num*mod_len,event_num*mod_len),device=self.device)),dim=1)
                     ,zeros((batch_len,(rep_num+event_num)*mod_len,rep_num*mod_len)
                    ,device=self.device))
                    ,dim=2)
                # slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                # slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                #slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                slf_attn_mask = (slf_attn_mask_subseq).gt(0)

        else:
            slf_attn_mask = None
        non_pad_mask=torch.cat([non_pad_mask,non_pad_mask],dim=1)
        # enc_output = zeros(tem_enc.shape,device=self.device)
        
        # Attention
        mod_enc_output=[zeros(input_embs.shape,device=self.device)]
        rep_enc_output=[zeros(rep_embs.shape,device=self.device)]
        
        rep_num=rep_embs.shape[1]
        rep_all_num=rep_embs.shape[1]
        mod_len=len(mod_enc_output)
        for lyr_ind in np.arange(self.n_layers):#Transformer layers Number
            
            lyr_ind_i = lyr_ind
            for mod_ind in np.arange(mod_len):
                # modal to tmp
                tmp_enc_output=mod_enc_output[mod_ind]
                tmp_rep_output=rep_enc_output[mod_ind]
                #enc_output[mod_i][:,-self.rep_n:,:]
                
                tmp_enc_output+=input_embs[mod_ind]
                tmp_rep_output+=rep_embs[mod_ind]
                # repの結合
                tmp_enc_output=torch.cat((tmp_enc_output,tmp_rep_output),dim=1)
                if plot==True:
                    if (mod_ind==0) and (lyr_ind_i==0):
                        attn_score_time_list=[]
                        attn_score_mark_list=[]
                    tmp_enc_output, attn_score = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
                    if mod_ind==0:
                        attn_score_time_list.append(attn_score)
                    else:
                        attn_score_mark_list.append(attn_score)
                else:
                    tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
                
                # enc_outとrepを分解
                mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_num]
                rep_enc_output[mod_ind]=tmp_enc_output[:,-rep_num:]
        mod_len=len(enc_input)
        enc_output_len=mod_enc_output[0].shape[1]
        rep_enc_output_len=rep_enc_output[0].shape[1]
        for mod_ind in np.arange(mod_len):
            tmp_list=[torch.cat((mod_enc_output[0][:,int((enc_output_len/2)*mod_ind):int((enc_output_len/2)*(mod_ind+1))], rep_enc_output[0][:,int((rep_enc_output_len/2)*mod_ind):int((rep_enc_output_len/2)*(mod_ind+1))]) ,dim=1) ]
                                 
            if mod_ind==0:
                # inputをimaginary rep vecとcat
                list_output=tmp_list.copy()
            else:
                list_output.extend(tmp_list)
        if self.normalize_before==True:
            for mod_ind in np.arange(mod_len):
                tmp_enc_output=list_output[mod_ind]
                tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                if mod_ind==0:
                    next_enc_output=[tmp_enc_output]
                    next_enc_output_append=next_enc_output.append
                else:
                    next_enc_output_append(tmp_enc_output)
            enc_output=next_enc_output
        else:
            enc_output=list_output
        if plot==True:
            # print(attn_score_time_list)
            # print(attn_score_mark_list)
            return enc_output, attn_score_time_list, attn_score_mark_list
        return enc_output#enc_output[[256,32,64][][]]

# late-fusion方式
class LateEncoder(nn.Module):
    #late-fusion方式のためのEncoder
    def __init__(
            self,
            d_model, d_inner, n_dataset,
            n_layers, n_head, d_k, d_v, dropout, device, normalize_before,mod_sample,is_bottle_neck=True,
            time_linear=False,train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0,start_fusion_layers=1):
        super().__init__()
        
        self.n_layers=n_layers
        self.start_fusion_layers=start_fusion_layers
        # self.d_model = d_model
        self.device=device
        # # position vector, used for temporal encoding
        
        self.n_marks=len(mod_sample)
        self.layer_stack_modal = ModuleList([
            ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before) for i in np.arange(n_layers)])  for j in np.arange(n_dataset)
            ])

        
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            ])

        self.modal_cat_list=ModuleList([nn.Linear(d_model,d_model, bias=False,device=device).to(device).double() for i in range(self.n_marks)])
        for ns_Linear in self.modal_cat_list:
            nn.init.xavier_uniform_(ns_Linear.weight)
    
    def forward(self, enc_input, rep_Mat=None,non_pad_mask=None,plot=None,imp=None,enc_plot=None,gene=None,allcat=None,emb_list=None):
            
        #enc_input [modal ,[Batch; seq_len, Dim=1]]
        # tau [ Batch, seq, dim=1]
        # marks[ Batch, seq, dim=marks(x,y,...)]
        #rep_Mat [modal; [Batch; rep_num; Dim=1]]
        #        [modal; [B; rep_num;Dim=2]
        mod_len=len(enc_input)
        batch_len=enc_input[0].shape[0]
        event_num=enc_input[0].shape[1]
        
        # input embedding
        for mod_ind in np.arange(mod_len):
            if mod_ind==0:
                tmp_input=enc_input[mod_ind] # inputをimaginary rep vecとcat
                tmp_emb=emb_list[mod_ind](tmp_input).to(torch.double) # embedding
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=tmp_emb.unsqueeze(0)
            else:
                tmp_emb=enc_input[mod_ind]
                for emb in emb_list[mod_ind]:
                    tmp_emb=emb(tmp_emb).to(torch.double)
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                
                input_embs=torch.cat([input_embs,tmp_emb.unsqueeze(0)],dim=0)
        dim=input_embs.shape[-1]
        
        # repも入力と別で、embeddingして、置いておく
        if rep_Mat is not None:
            rep_num = rep_Mat[0].shape[1]# [B,rep_num]
            for mod_ind in np.arange(mod_len):
                if mod_ind==0:
                    tmp_input=rep_Mat[mod_ind] # inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=rep_Mat[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)
        else:
            rep_num = 0
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=enc_input[mod_ind]# inputをimaginary rep vecとcat
                    pdb.set_trace()
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=enc_input[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)#3,128,3,64

        if non_pad_mask is not None:
            if rep_Mat is not None:
                represent_mod_number=0
                #pdb.set_trace()
                
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                
                
                slf_attn_mask_subseq=torch.cat(
                    (torch.cat((slf_attn_mask_subseq,zeros((batch_len,rep_num,event_num),device=self.device)),dim=1)
                     ,zeros((batch_len,rep_num+event_num,rep_num)
                    ,device=self.device))
                    ,dim=2)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                
            else:
                pdb.set_trace()
                represent_mod_number=0
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input 合わせ
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=enc_input[represent_mod_number].squeeze(-1), seq_q=enc_input[represent_mod_number].squeeze(-1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None
        
        # enc_output = zeros(tem_enc.shape,device=self.device)
        
        # Attention
        mod_enc_output=[zeros((input_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        rep_enc_output=[zeros((rep_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        
        mask_change_flg=0
        #
        rep_num=rep_embs.shape[2]
        rep_all_num=rep_embs.shape[0]*rep_embs.shape[2]
    
        for lyr_ind in np.arange(self.n_layers):#Transformer layers Number
            lyr_ind_i = lyr_ind
            for mod_ind in np.arange(mod_len):
                # 前の階層のenc出力を獲得
                tmp_enc_output=mod_enc_output[mod_ind]
                #
                tmp_enc_output+=input_embs[mod_ind]
                
                
                tmp_rep_output=rep_enc_output[mod_ind]
                tmp_rep_output+=rep_embs[mod_ind]
                # repの結合
                tmp_enc_output=torch.cat((tmp_enc_output,tmp_rep_output),dim=1)
                    
                tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                    enc_input=tmp_enc_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask)
                
                
                    # enc_outとrepを分解
                mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_num]
                rep_enc_output[mod_ind]=tmp_enc_output[:,-rep_num:]
                
        
        for mod_ind in np.arange(mod_len):
            tmp_list=[torch.cat((mod_enc_output[mod_ind], rep_enc_output[mod_ind]) ,dim=1) ]
            if mod_ind==0:
                # inputをimaginary rep vecとcat
                list_output=tmp_list.copy()
            else:
                list_output.extend(tmp_list)
        if self.normalize_before==True:
            for mod_ind in np.arange(mod_len):
                tmp_enc_output=list_output[mod_ind]
                tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                if mod_ind==0:
                    next_enc_output=[tmp_enc_output]
                    next_enc_output_append=next_enc_output.append
                else:
                    next_enc_output_append(tmp_enc_output)
            enc_output=next_enc_output
        else:
            enc_output=list_output
        return enc_output#enc_output[[256,32,64][][]]

# mod-attention + late-fusion 用
class attnLateEncoder(nn.Module):
    #本質的にはlate-fusionと同等？
    def __init__(
            self,
            d_model, d_inner, n_dataset,
            n_layers, n_head, d_k, d_v, dropout, device, normalize_before,mod_sample,is_bottle_neck=True,
            time_linear=False,train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0,start_fusion_layers=1):
        super().__init__()
        
        self.n_layers=n_layers
        self.start_fusion_layers=start_fusion_layers
        # self.d_model = d_model
        self.device=device
        # # position vector, used for temporal encoding
        
        self.n_marks=len(mod_sample)
        self.layer_stack_modal = ModuleList([
            ModuleList([
                modEncLayer( select_mod_n=j, d_model=d_model, d_inner=d_inner,
                             n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                             normalize_before=normalize_before) for i in np.arange(n_layers)])  for j in np.arange(n_dataset)
        ])

        
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            ])

        self.modal_cat_list=ModuleList([nn.Linear(d_model,d_model, bias=False,device=device).to(device).double() for i in range(self.n_marks)])
        for ns_Linear in self.modal_cat_list:
            nn.init.xavier_uniform_(ns_Linear.weight)
    
    def forward(self, enc_input, rep_Mat=None,non_pad_mask=None,plot=None,imp=None,enc_plot=None,gene=None,allcat=None,emb_list=None):
            
        #enc_input [modal ,[Batch; seq_len, Dim=1]]
        # tau [ Batch, seq, dim=1]
        # marks[ Batch, seq, dim=marks(x,y,...)]
        #rep_Mat [modal; [Batch; rep_num; Dim=1]]
        #        [modal; [B; rep_num;Dim=2]
        mod_len=len(enc_input)
        batch_len=enc_input[0].shape[0]
        event_num=enc_input[0].shape[1]
        
        # input embedding
        for mod_ind in np.arange(mod_len):
            if mod_ind==0:
                tmp_input=enc_input[mod_ind] # inputをimaginary rep vecとcat
                tmp_emb=emb_list[mod_ind](tmp_input).to(torch.double) # embedding
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=tmp_emb.unsqueeze(0)
            else:
                tmp_emb=enc_input[mod_ind]
                for emb in emb_list[mod_ind]:
                    tmp_emb=emb(tmp_emb).to(torch.double)
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                
                input_embs=torch.cat([input_embs,tmp_emb.unsqueeze(0)],dim=0)
        dim=input_embs.shape[-1]
        
        # repも入力と別で、embeddingして、置いておく
        if rep_Mat is not None:
            rep_num = rep_Mat[0].shape[1]# [B,rep_num]
            for mod_ind in np.arange(mod_len):
                if mod_ind==0:
                    tmp_input=rep_Mat[mod_ind] # inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=rep_Mat[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)
        else:
            rep_num = 0
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=enc_input[mod_ind]# inputをimaginary rep vecとcat
                    pdb.set_trace()
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=enc_input[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)#3,128,3,64

        if non_pad_mask is not None:
            if rep_Mat is not None:
                represent_mod_number=0
                #pdb.set_trace()
                
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                
                
                slf_attn_mask_subseq=torch.cat(
                    (torch.cat((slf_attn_mask_subseq,zeros((batch_len,rep_num,event_num),device=self.device)),dim=1)
                     ,zeros((batch_len,rep_num+event_num,rep_num)
                    ,device=self.device))
                    ,dim=2)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                
            else:
                pdb.set_trace()
                represent_mod_number=0
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input 合わせ
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=enc_input[represent_mod_number].squeeze(-1), seq_q=enc_input[represent_mod_number].squeeze(-1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None
        
        # enc_output = zeros(tem_enc.shape,device=self.device)
        
        # Attention
        mod_enc_output=[zeros((input_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        rep_enc_output=[zeros((rep_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        
        mask_change_flg=0
        #
        rep_num=rep_embs.shape[2]
        rep_all_num=rep_embs.shape[0]*rep_embs.shape[2]
    
        for lyr_ind in np.arange(self.n_layers):#Transformer layers Number
            lyr_ind_i = lyr_ind
            for mod_ind in np.arange(mod_len):
                # 前の階層のenc出力を獲得
                tmp_enc_output=mod_enc_output[mod_ind]
                #
                tmp_enc_output+=input_embs[mod_ind]
                
                
                tmp_rep_output=rep_enc_output[mod_ind]
                tmp_rep_output+=rep_embs[mod_ind]
                # repの結合
                tmp_enc_output=torch.cat((tmp_enc_output,tmp_rep_output),dim=1)
                    
                tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                    enc_input=tmp_enc_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask)
                
                
                    # enc_outとrepを分解
                mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_num]
                rep_enc_output[mod_ind]=tmp_enc_output[:,-rep_num:]
                
        
        for mod_ind in np.arange(mod_len):
            tmp_list=[torch.cat((mod_enc_output[mod_ind], rep_enc_output[mod_ind]) ,dim=1) ]
            if mod_ind==0:
                # inputをimaginary rep vecとcat
                list_output=tmp_list.copy()
            else:
                list_output.extend(tmp_list)
        if self.normalize_before==True:
            for mod_ind in np.arange(mod_len):
                tmp_enc_output=list_output[mod_ind]
                tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                if mod_ind==0:
                    next_enc_output=[tmp_enc_output]
                    next_enc_output_append=next_enc_output.append
                else:
                    next_enc_output_append(tmp_enc_output)
            enc_output=next_enc_output
        else:
            enc_output=list_output
        return enc_output#enc_output[[256,32,64][][]]

# crossattention-fusion方式
class CrossEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner, n_dataset,
            n_layers, n_head, d_k, d_v, dropout, device, normalize_before,mod_sample,is_bottle_neck=True,
            time_linear=False,train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0):
        super().__init__()
        
        self.n_layers=n_layers#4
        
        self.device=device
        
        
        self.n_marks=len(mod_sample)#2
        
        self.layer_stack_modal = ModuleList([
            ModuleList([
            CrossEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
            for i in np.arange(n_layers)]) for j in np.arange(n_dataset)
        ])
        
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            for i in np.arange(n_layers)])
        self.modal_cat_list=ModuleList([nn.Linear(d_model,d_model, bias=False,device=device).to(device).double() for i in range(self.n_marks)])
    

    def forward(self, enc_input, rep_Mat=None,non_pad_mask=None,plot=None,imp=None,enc_plot=None,gene=None,allcat=None,emb_list=None):
            
        #enc_input [modal ,[Batch; seq_len, Dim=1]]
        # tau [ Batch, seq, dim=1]
        # marks[ Batch, seq, dim=marks(x,y,...)]
        #rep_Mat [modal; [Batch; rep_num; Dim=1]]
        #        [modal; [B; rep_num;Dim=2]
        mod_len=len(enc_input)
        batch_len=enc_input[0].shape[0]
        event_num=enc_input[0].shape[1]
        
        # input embedding
        for mod_ind in np.arange(mod_len):
            if mod_ind==0:
                tmp_input=enc_input[mod_ind] # inputをimaginary rep vecとcat
                tmp_emb=emb_list[mod_ind](tmp_input).to(torch.double) # embedding
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=tmp_emb.unsqueeze(0)
            else:
                tmp_emb=enc_input[mod_ind]
                for emb in emb_list[mod_ind]:
                    tmp_emb=emb(tmp_emb).to(torch.double)
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                
                input_embs=torch.cat([input_embs,tmp_emb.unsqueeze(0)],dim=0)
        dim=input_embs.shape[-1]
        input_embs=torch.concat([*input_embs],dim=1)#concat
        
        # repも入力と別で、embeddingして、置いておく
        if rep_Mat is not None:
            rep_num = rep_Mat[0].shape[1]# [B,rep_num]
            for mod_ind in np.arange(mod_len):
                if mod_ind==0:
                    tmp_input=rep_Mat[mod_ind] # inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=rep_Mat[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)
        else:
            rep_num = 0
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=enc_input[mod_ind]# inputをimaginary rep vecとcat
                    pdb.set_trace()
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=enc_input[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)#3,128,3,64
        rep_embs=torch.concat([*rep_embs],dim=1)#concat
        
        # Embedding 
        # input_embs[[256,29,64][][]]
        # rep_embs[[256,3,64][][]]
        # Encoding inputs
        
        #initial_S=mod_embs[0][0,-3:,:]
        
        # Generating mask
        # """ Encode event sequences via masked self-attention. """
        # #入力の時間エンコーディング
        
        # tem_enc = torch.cat((tem_enc,tem_rep), dim=1)#(16,seqence-1,M)->(16,seq-1+rep_vec_num,M)
        
        # modal 別でもイベントの見えている関係性は同じだから...
        if non_pad_mask is not None:
            if rep_Mat is not None:
                represent_mod_number=0
                #pdb.set_trace()
                
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                slf_attn_mask_subseq=slf_attn_mask_subseq.tile(2,2)#256,58,58
                
                slf_attn_mask_subseq=torch.cat(
                    (
                        torch.cat(
                            (slf_attn_mask_subseq,
                            zeros((batch_len,rep_num*mod_len,event_num*mod_len),device=self.device))
                            ,dim=1),
                        zeros(
                            (batch_len,(rep_num+event_num)*mod_len,rep_num*mod_len),device=self.device)
                        )
                    ,dim=2)
                # slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                # slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                #slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                slf_attn_mask = (slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None
        non_pad_mask=torch.cat([non_pad_mask,non_pad_mask],dim=1)
        # enc_output = zeros(tem_enc.shape,device=self.device)
        
        # Attention
        mod_enc_output=[zeros(input_embs.shape,device=self.device)]
        rep_enc_output=[zeros(rep_embs.shape,device=self.device)]
        
        rep_num=rep_embs.shape[1]
        rep_all_num=rep_embs.shape[1]
        mod_len=len(mod_enc_output)
        for lyr_ind in np.arange(self.n_layers):#Transformer layers Number
            
            lyr_ind_i = lyr_ind
            for mod_ind in np.arange(mod_len):
                # modal to tmp
                tmp_enc_output=mod_enc_output[mod_ind]
                tmp_rep_output=rep_enc_output[mod_ind]
                #enc_output[mod_i][:,-self.rep_n:,:]
                
                tmp_enc_output+=input_embs[mod_ind]
                tmp_rep_output+=rep_embs[mod_ind]
                # repの結合
                tmp_enc_output=torch.cat((tmp_enc_output,tmp_rep_output),dim=1)
                
                tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                    enc_input=tmp_enc_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask,
                    rep_num=rep_all_num)
                
                
                # enc_outとrepを分解
                mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_num]
                rep_enc_output[mod_ind]=tmp_enc_output[:,-rep_num:]
        mod_len=len(enc_input)
        enc_output_len=mod_enc_output[0].shape[1]
        rep_enc_output_len=rep_enc_output[0].shape[1]
        for mod_ind in np.arange(mod_len):
            tmp_list=[torch.cat((mod_enc_output[0][:,int((enc_output_len/2)*mod_ind):int((enc_output_len/2)*(mod_ind+1))], rep_enc_output[0][:,int((rep_enc_output_len/2)*mod_ind):int((rep_enc_output_len/2)*(mod_ind+1))]) ,dim=1) ]
                                 
            if mod_ind==0:
                # inputをimaginary rep vecとcat
                list_output=tmp_list.copy()
            else:
                list_output.extend(tmp_list)
        if self.normalize_before==True:
            for mod_ind in np.arange(mod_len):
                tmp_enc_output=list_output[mod_ind]
                tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                if mod_ind==0:
                    next_enc_output=[tmp_enc_output]
                    next_enc_output_append=next_enc_output.append
                else:
                    next_enc_output_append(tmp_enc_output)
            enc_output=next_enc_output
        else:
            enc_output=list_output
        return enc_output#enc_output[[256,32,64][][]]

# bottleneck-fusion方式
class BtlEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner, n_dataset,
            n_layers, n_head, d_k, d_v, dropout, device, normalize_before,mod_sample,is_bottle_neck=True,
            time_linear=False,train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0,start_fusion_layers=1):
        super().__init__()
        
        self.n_layers=n_layers
        self.start_fusion_layers=1
        # self.d_model = d_model
        self.device=device
        # # position vector, used for temporal encoding
        
        
        self.n_marks=len(mod_sample)
        
        
        self.layer_stack_modal = ModuleList([
            ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before) for i in np.arange(n_layers)])  for j in np.arange(n_dataset)
        ])


        
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            ])

        self.modal_cat_list=ModuleList([nn.Linear(d_model,d_model, bias=False,device=device).to(device).double() for i in range(self.n_marks)])
        for ns_Linear in self.modal_cat_list:
            nn.init.xavier_uniform_(ns_Linear.weight)
        self.is_bottle_neck=True
        btl_num=3
        self.bottle_neck = nn.Parameter(torch.randn(1,btl_num,d_model))
        
    def forward(self, enc_input, rep_Mat=None,non_pad_mask=None,plot=None,imp=None,enc_plot=None,gene=None,allcat=None,emb_list=None):
            
        #enc_input [modal ,[Batch; seq_len, Dim=1]]
        # tau [ Batch, seq, dim=1]
        # marks[ Batch, seq, dim=marks(x,y,...)]
        #rep_Mat [modal; [Batch; rep_num; Dim=1]]
        #        [modal; [B; rep_num;Dim=2]
        mod_len=len(enc_input)
        batch_len=enc_input[0].shape[0]
        event_num=enc_input[0].shape[1]
        
        # input embedding
        if rep_Mat is not None:
            rep_num = rep_Mat[0].shape[1]# [B,rep_num]
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=torch.cat((enc_input[mod_ind], rep_Mat[mod_ind]) ,dim=1) # inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    mod_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=torch.cat((enc_input[mod_ind], rep_Mat[mod_ind]) ,dim=1)
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    mod_embs=torch.cat([mod_embs,tmp_emb.unsqueeze(0)],dim=0)
        else:
            rep_num = 0
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=enc_input[mod_ind]# inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    mod_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=enc_input[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    mod_embs=torch.cat([mod_embs,tmp_emb.unsqueeze(0)],dim=0)
        
        # Embedding 
        # input_embs[[256,29,64][][]]
        # Encoding inputs
        
        #initial_S=mod_embs[0][0,-3:,:]
        
        # Generating mask
        # """ Encode event sequences via masked self-attention. """
        # #入力の時間エンコーディング
        
        # tem_enc = torch.cat((tem_enc,tem_rep), dim=1)#(16,seqence-1,M)->(16,seq-1+rep_vec_num,M)
        
        # modal 別でもイベントの見えている関係性は同じだから...
        if non_pad_mask is not None:
            if rep_Mat is not None:
                represent_mod_number=0
                #pdb.set_trace()
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                slf_attn_mask_subseq=torch.cat(
                    (torch.cat((slf_attn_mask_subseq,zeros((batch_len,rep_num,event_num),device=self.device)),dim=1)
                     ,zeros((batch_len,rep_num+event_num,rep_num)
                    ,device=self.device))
                    ,dim=2)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                
            else:
                pdb.set_trace()
                represent_mod_number=0
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input 合わせ
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=enc_input[represent_mod_number].squeeze(-1), seq_q=enc_input[represent_mod_number].squeeze(-1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None
        
        # enc_output = zeros(tem_enc.shape,device=self.device)
        
        # Attention
        mod_enc_output=[zeros(mod_embs[mod_i].shape,device=self.device) for mod_i in np.arange(mod_len)]
        #mod_enc_output: [[128,29,64],[128,29,64],[128,29,64]]
        if self.is_bottle_neck==True:
            btl=self.bottle_neck.repeat([batch_len,1,1])
            btl_num=btl.shape[1]
        
        mask_change_flg=0
        #
        
        for lyr_ind in np.arange(self.n_layers):#Transformer layers Number
            sum_btl=None
            lyr_ind_i = lyr_ind
            for mod_ind in np.arange(mod_len):
                # 前の階層のenc出力を獲得
                tmp_enc_output=mod_enc_output[mod_ind]
                #
                tmp_enc_output+=mod_embs[mod_ind]
                if (self.is_bottle_neck==True) and (lyr_ind_i>=(self.start_fusion_layers - 1)):
                        if self.is_bottle_neck==True:
                            #btlがあるときの入力作り直し btlが各モーダル入力の右に追加
                            tmp_enc_output=torch.cat((tmp_enc_output,btl),dim=1)
                        if mask_change_flg==0:
                            #btlがあるときのmask作り直し [B,L+S,L+S]->[B,L+S+btl,L+S+btl]
                            mask_change_flg=1
                            mask_Bts=zeros((batch_len,btl_num,slf_attn_mask_subseq.shape[2]),device=self.device)#0がマスクなし、1があり。最終行は0で構成.[B,btl,L] 
                            if rep_Mat is None:
                                mask_Bts[:,:,-1]+=1 #最後のイベントを見えないようにしないとbtlから間接的に真値情報がわたってしまう。
                            
                            slf_attn_mask_subseq=torch.cat((torch.cat((slf_attn_mask_subseq,mask_Bts),dim=1),zeros((batch_len,event_num+rep_num+btl_num,btl_num),device=self.device)),dim=2)
                            
                            slf_attn_mask = (slf_attn_mask_subseq).gt(0)
                            non_pad_mask = get_non_pad_mask(torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num,1),device=self.device)),dim=1).squeeze(-1))
                    
                
                    
                tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                    enc_input=tmp_enc_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask)
                
                if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                        tmp_btl=tmp_enc_output[:,-btl_num:]
                        tmp_enc_output=tmp_enc_output[:,:-btl_num]
                if mod_ind==0:
                        next_mod_enc_output=[tmp_enc_output]
                        next_mod_enc_output_append=next_mod_enc_output.append
                else:
                        next_mod_enc_output_append(tmp_enc_output)
                    
                if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                        if mod_ind==0:
                            sum_btl=tmp_btl
                        else:
                            sum_btl=sum_btl+tmp_btl
            if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                btl=sum_btl/mod_len
            mod_enc_output=next_mod_enc_output

        list_output=mod_enc_output
        if self.normalize_before==True:
            for mod_ind in np.arange(mod_len):
                tmp_enc_output=list_output[mod_ind]
                tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                if mod_ind==0:
                    next_enc_output=[tmp_enc_output]
                    next_enc_output_append=next_enc_output.append
                else:
                    next_enc_output_append(tmp_enc_output)
            enc_output=next_enc_output
        else:
            enc_output=list_output
        return enc_output#enc_output[[256,32,64][][]]

# rep-fusion方式
class MoveEncoder(nn.Module):
    #rep-fusion方式用

    def __init__(
            self,
            d_model, d_inner, n_dataset,
            n_layers, n_head, d_k, d_v, dropout, device, normalize_before,mod_sample,is_bottle_neck=True,
            time_linear=False,train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0,start_fusion_layers=1):
        super().__init__()
        
        self.n_layers=n_layers
        self.start_fusion_layers=start_fusion_layers
        # self.d_model = d_model
        self.device=device
        # # position vector, used for temporal encoding
        
        
        self.n_marks=len(mod_sample)
        
        
        self.layer_stack_modal = ModuleList([
            ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before) for i in np.arange(n_layers)])  for j in np.arange(n_dataset)
        ])

        
        
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            ])

        self.modal_cat_list=ModuleList([nn.Linear(d_model,d_model, bias=False,device=device).to(device).double() for i in range(self.n_marks)])
        for ns_Linear in self.modal_cat_list:
            nn.init.xavier_uniform_(ns_Linear.weight)
    
    
    def forward(self, enc_input, rep_Mat=None,non_pad_mask=None,plot=None,imp=None,enc_plot=None,gene=None,allcat=None,emb_list=None):
            
        #enc_input [modal ,[Batch; seq_len, Dim=1]]
        # tau [ Batch, seq, dim=1]
        # marks[ Batch, seq, dim=marks(x,y,...)]
        #rep_Mat [modal; [Batch; rep_num; Dim=1]]
        #        [modal; [B; rep_num;Dim=2]
        mod_len=len(enc_input)
        batch_len=enc_input[0].shape[0]
        event_num=enc_input[0].shape[1]
        
        # input embedding
        for mod_ind in np.arange(mod_len):
            if mod_ind==0:
                tmp_input=enc_input[mod_ind] # inputをimaginary rep vecとcat
                tmp_emb=emb_list[mod_ind](tmp_input).to(torch.double) # embedding
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=tmp_emb.unsqueeze(0)
            else:
                tmp_emb=enc_input[mod_ind]
                for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=torch.cat([input_embs,tmp_emb.unsqueeze(0)],dim=0)
        dim=input_embs.shape[-1]

        # repも入力と別で、embeddingして、置いておく
        if rep_Mat is not None:
            rep_num = rep_Mat[0].shape[1]# [B,rep_num]
            for mod_ind in np.arange(mod_len):
                if mod_ind==0:
                    tmp_input=rep_Mat[mod_ind] # inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=rep_Mat[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)
        else:
            rep_num = 0
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=enc_input[mod_ind]# inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=enc_input[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)#3,128,3,64
        # Embedding 
        # input_embs[[256,29,64][][]]
        # rep_embs[[256,3,64][][]]
        # Encoding inputs
        
        #initial_S=mod_embs[0][0,-3:,:]
        
        # Generating mask
        # """ Encode event sequences via masked self-attention. """
        # #入力の時間エンコーディング
        
        # tem_enc = torch.cat((tem_enc,tem_rep), dim=1)#(16,seqence-1,M)->(16,seq-1+rep_vec_num,M)
        
        # modal 別でもイベントの見えている関係性は同じだから...
        if non_pad_mask is not None:
            if rep_Mat is not None:
                represent_mod_number=0
                #pdb.set_trace()
                
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                
                
                slf_attn_mask_subseq=torch.cat(
                    (torch.cat((slf_attn_mask_subseq,zeros((batch_len,rep_num,event_num),device=self.device)),dim=1)
                     ,zeros((batch_len,rep_num+event_num,rep_num)
                    ,device=self.device))
                    ,dim=2)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                
            else:
                pdb.set_trace()
                represent_mod_number=0
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input 合わせ
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=enc_input[represent_mod_number].squeeze(-1), seq_q=enc_input[represent_mod_number].squeeze(-1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None
        
        # enc_output = zeros(tem_enc.shape,device=self.device)
        
        # Attention
        mod_enc_output=[zeros((input_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        rep_enc_output=[zeros((rep_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        #mod_enc_output: [[128,29,64],[128,29,64],[128,29,64]]
        # if self.is_bottle_neck==True:
        #     btl=self.bottle_neck.repeat([batch_len,1,1])
        #     btl_num=btl.shape[1]
        
        mask_change_flg=0
        #
        rep_num=rep_embs.shape[2]
        rep_all_num=rep_embs.shape[0]*rep_embs.shape[2]
        if plot==True:
            time_enc_output_history=[[mod_embs[0],mod_embs[1],mod_embs[2]]]#(3=mod,256=batch,32=29+3,64=dim)
            if self.is_bottle_neck==True:
                share_enc_output_his=[btl]
            for lyr_ind in np.arange(self.n_layers):
                
                sum_btl=None
                lyr_ind_i = lyr_ind
                for mod_ind in np.arange(mod_len):
                    # modal to tmp
                    tmp_enc_output=mod_enc_output[mod_ind]
                    
                    tmp_enc_output+=mod_embs[mod_ind]
                    if (self.is_bottle_neck==True) and (lyr_ind_i>=(self.start_fusion_layers - 1)):
                        if self.is_bottle_neck==True:
                            #btlがあるときの入力作り直し btlが各モーダル入力の右に追加
                            tmp_enc_output=torch.cat((tmp_enc_output,btl),dim=1)
                        if mask_change_flg==0:
                            #btlがあるときのmask作り直し [B,L+S,L+S]->[B,L+S+btl,L+S+btl]
                            mask_change_flg=1
                            mask_Bts=zeros((batch_len,btl_num,slf_attn_mask_subseq.shape[2]),device=self.device)#0がマスクなし、1があり。最終行は0で構成.[B,btl,L] 
                            if rep_Mat is None:
                                mask_Bts[:,:,-1]+=1 #最後のイベントを見えないようにしないとbtlから間接的に真値情報がわたってしまう。
                            slf_attn_mask_subseq=torch.cat((torch.cat((slf_attn_mask_subseq,mask_Bts),dim=1),zeros((batch_len,event_num+rep_num+btl_num,btl_num),device=self.device)),dim=2)
                            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1))
                            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                            non_pad_mask = get_non_pad_mask(torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1))
                        
                    tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
                    
                    if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                        tmp_btl=tmp_enc_output[:,-btl_num:]
                        tmp_enc_output=tmp_enc_output[:,:-btl_num]
                    if mod_ind==0:
                        next_mod_enc_output=[tmp_enc_output]
                        next_mod_enc_output_append=next_mod_enc_output.append
                    else:
                        next_mod_enc_output_append(tmp_enc_output)
                    
                    if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                        if mod_ind==0:
                            sum_btl=tmp_btl
                        else:
                            sum_btl=sum_btl+tmp_btl
                
                if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
                    btl=sum_btl/mod_len
                mod_enc_output=next_mod_enc_output
                time_enc_output_history.append(mod_enc_output)
                if self.is_bottle_neck==True:
                    share_enc_output_his.append(btl)
                
            if self.is_bottle_neck==True:
                with open(f"./pickled/enc_output/{imp}share",mode="wb") as file:
                    pickle.dump(share_enc_output_his,file)
            with open(f"./pickled/enc_output/{imp}proposed_encout",mode="wb") as file:
                pickle.dump(time_enc_output_history,file)
            pdb.set_trace()
        else:
            for lyr_ind in np.arange(self.n_layers):#Transformer layers Number
                lyr_ind_i = lyr_ind
                for mod_ind in np.arange(mod_len):
                    # 前の階層のenc出力を獲得
                    tmp_enc_output=mod_enc_output[mod_ind]
                    #
                    tmp_enc_output+=input_embs[mod_ind]
                    
                    if (lyr_ind_i>=(self.start_fusion_layers - 1)):
                        
                        for i, list_rep_output in enumerate(rep_enc_output):
                            list_rep_output+=rep_embs[i]
                            if i==0:
                                m_rep_output=list_rep_output.unsqueeze(0)
                            else:
                                m_rep_output=torch.cat([m_rep_output,list_rep_output.unsqueeze(0)],dim=0)
                                
                        
                        m_rep_output=torch.cat([m_rep_output[0],m_rep_output[1]],dim=1)
                        tmp_enc_output=torch.cat((tmp_enc_output,m_rep_output),dim=1)
                        if mask_change_flg==0:
                            # mv4=start2 CUDA_VISIBLE_DEVICES=1 python Main.py -gene=jisin --phase --pre_attn --vec -imp=mv4   --movevec -start_fusion_layers=2 --train
                            # イベントから該当マスクのみ見えるvisible_mask_for_event、非該当仮想はイベントとrepを見ないnot_visible_mask_for_rep=torch.cat((not_visible_mask_for_rep,ones((batch_len,rep_num,rep_all_num),device=self.device)),dim=2)#mv4
                            #CUDA_VISIBLE_DEVICES=1 python Main.py -gene=jisin --phase --pre_attn --vec -imp=mv4   --movevec
                            #CUDA_VISIBLE_DEVICES=1 python Main.py -gene=jisin --phase --pre_attn --vec -imp=mv1   --movevec
                            # mv2=イベントから該当マスクのみ見えるvisible_mask_for_event,非該当は仮想をみる
                            # mv1=イベントから全てのrepが見える。
                            #btlがあるときのmask作り直し [B,L+S,L+S]->[B,L+S+btl,L+S+btl]
                            # イベント履歴に仮想ベクトルを追加するマスク
                            # 対応する仮想ベクトルは0他は１
                            # 仮想ベクトル
                            mask_change_flg=1
                            
                            for make_allmod_mask_modal_ind in range(rep_embs.shape[0]):# あるmodイベント履歴が該当仮想ベクトルのみ見えるようにしている。#該当仮想ベクトルが仮想ベクトルと該当イベントのみ見えるようにしている。
                                # あるmodイベント履歴が該当仮想ベクトルのみ見えるようにしている。
                                slf_attn_mask_subseq = get_subsequent_mask(enc_input[0].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                                # place1 event to repS
                                visible_mask_for_event=zeros((batch_len,slf_attn_mask_subseq.shape[1],rep_num,
                                             ),device=self.device)#0がマスクなし、1があり。[B,L,rep] 
                                not_visible_mask_for_event=ones((batch_len,slf_attn_mask_subseq.shape[1],rep_num,
                                             ),device=self.device)#0がマスクなし、1があり。[B,L,rep] 
                                # if imp=="mv1" or imp=="mv3"or imp=="mv12"or imp=="mv32" or imp=="mvC1" or imp=="mvC3"or imp=="mvC12"or imp=="mvC32":
                                #         # イベントから見た仮想ベクトルのマスクの追加 全部見えてる
                                #     for make_mask_modal_ind in range(rep_embs.shape[0]):
                                #         slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_event),dim=2)
                                # else:#mv2  mv4
                                    # イベントから見た仮想ベクトルのマスクの追加　該当以外はみえてない
                                for make_mask_modal_ind in range(rep_embs.shape[0]):
                                    if make_mask_modal_ind==make_allmod_mask_modal_ind:
                                        slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_event),dim=2)
                                    else:
                                        slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,not_visible_mask_for_event),dim=2)
                                
                                
                                # place2 rep to event
                                #仮想ベクトルから見たマスクの作成
                                # B,29,29+(mod*rep_num)
                                # イベントが見えてよいか同課の違いだけ。仮想ベクトル同士は見えている。
                                # rep がイベントを見てもよいか。
                                # visible_mask_for_rep=zeros((batch_len,rep_num,event_num),device=self.device)
                                # not_visible_mask_for_rep=zeros((batch_len,rep_num,event_num),device=self.device)
                                
                                # # place3 rep to rep 
                                # visible_mask_for_rep2rep=zeros((batch_len,slf_attn_mask_subseq.shape[1],rep_num),device=self.device)
                                # not_visible_mask_for_rep2rep=ones((batch_len,slf_attn_mask_subseq.shape[1],rep_num),device=self.device)
                                # place2 and 3
                                
                                # for repind in range(rep_embs.shape[0]):
                                #     for reprepind in range(rep_embs.shape[0]):
                                #         if repind==0:
                                #             if make_allmod_mask_modal_ind==repind:
                                #                 torch.cat((visible_mask_for_rep))
                                #                 visible_mask_for_rep=torch.cat((visible_mask_for_rep))
                                #             else:
                                #                 not_visible_mask_for_rep=
                                            
                                #         else:
                                #             visible_mask_for_mod_rep=torch.cat((visible_mask_for_mod_rep,))

                                # concat 2 and 3
                                visible_mask_for_rep=zeros((batch_len,rep_num,event_num+rep_all_num),device=self.device)
                                not_visible_mask_for_rep=ones((batch_len,rep_num,event_num+rep_all_num),device=self.device)

                                # if imp=="mv1" or imp=="mv2" or imp=="mv12" or imp=="mv22" or imp=="mvC1" or imp=="mvC2" or imp=="mvC12" or imp=="mvC22":
                                #     for make_mask_modal_ind in range(rep_embs.shape[0]):
                                #         slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_rep),dim=1)
                                # else:
                                    
                                for make_mask_modal_ind in range(rep_embs.shape[0]):
                                    if make_mask_modal_ind==make_allmod_mask_modal_ind:
                                        slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_rep),dim=1)
                                    else:
                                        slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,not_visible_mask_for_rep),dim=1)
                                
                                # visible_mask_for_rep=torch.cat((visible_mask_for_rep,zeros((batch_len,rep_num,rep_all_num),device=self.device)),dim=2)
                                # if imp=="mv1":
                                #     not_visible_mask_for_rep=ones((batch_len,rep_num,event_num,
                                #              ),device=self.device)
                                # else:
                                    
                                # if imp=="mv4":#repが"他の"repをみない
                                #     for make_mask_modal_ind in range(rep_embs.shape[0]):
                                #         if make_mask_modal_ind==0:
                                #             if make_mask_modal_ind==make_allmod_mask_modal_ind:#対応している
                                #                 not_visible_mask_for_rep_rep=zeros((batch_len,rep_num,rep_num))
                                #             else: #対応していない
                                #                 not_visible_mask_for_rep_rep=ones((batch_len,rep_num,rep_num))
                                #         else:
                                #             if make_mask_modal_ind==make_allmod_mask_modal_ind:#対応している
                                #                 not_visible_mask_for_rep_rep=torch.cat((not_visible_mask_for_rep_rep,zeros((batch_len,rep_num,rep_num))),dim=2)
                                #             else: #対応してない
                                #                 not_visible_mask_for_rep_rep=torch.cat((not_visible_mask_for_rep_rep,ones((batch_len,rep_num,rep_num))),dim=2)
                                #         if make_mask_modal_ind==0:
                                #             not_visible_mask_for_rep_rep_matome=not_visible_mask_for_rep_rep
                                #         else:
                                #             not_visible_mask_for_rep_rep_matome=not_visible_mask_for_rep_rep_matome
                                #         not_visible_mask_for_rep=torch.cat((not_visible_mask_for_rep,ones((batch_len,rep_num,rep_num),device=self.device)),dim=2)#mv4
                                #         #not_visible_mask_for_rep=torch.cat((not_visible_mask_for_rep,ones((batch_len,rep_num,rep_all_num),device=self.device)),dim=2)#mv4
                                # else:
                                #     not_visible_mask_for_rep=torch.cat((not_visible_mask_for_rep,zeros((batch_len,rep_num,rep_all_num),device=self.device)),dim=2)#mv2

                                # for make_mask_modal_ind in range(rep_embs.shape[0]):
                                #     if make_mask_modal_ind==make_allmod_mask_modal_ind:
                                #         slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_rep),dim=1)
                                #     else:
                                #         slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,not_visible_mask_for_rep[make_mask_modal_ind]),dim=1)
                                # mask_Bts=zeros((batch_len,rep_all_num-rep_num,
                                #             slf_attn_mask_subseq.shape[2]),device=self.device)#0がマスクなし、1があり。最終行は0で構成.[B,btl,L] 
                                if rep_Mat is None:
                                    pdb.set_trace()
                                    mask_Bts[:,:,-1]+=1 #最後のイベントを見えないようにしないとbtlから間接的に真値情報がわたってしまう。
                                
                                
                                
                                #slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                                slf_attn_mask_keypad = zeros(slf_attn_mask_subseq.shape,device=self.device)#slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                                if make_allmod_mask_modal_ind==0:
                                    slf_attn_mask_all=slf_attn_mask.unsqueeze(0)
                                else:
                                    slf_attn_mask_all=torch.cat([slf_attn_mask_all,slf_attn_mask.unsqueeze(0)],dim=0)
                                del visible_mask_for_rep,not_visible_mask_for_rep
                                gc.collect()
                                non_pad_mask = get_non_pad_mask(torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_all_num),device=self.device)),dim=1))
                            #     slf_attn_mask_subseq=torch.cat(
                            #         (torch.cat((slf_attn_mask_subseq,mask_Bts),dim=1),
                            #         zeros((batch_len,event_num+rep_all_num,rep_all_num-rep_num)
                            #         ,device=self.device))
                            #         ,dim=2)
                            # slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_all_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_all_num),device=self.device)),dim=1))
                            # slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                            # slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                                
                        slf_attn_mask=slf_attn_mask_all[mod_ind]
                    else:
                        tmp_rep_output=rep_enc_output[mod_ind]
                        tmp_rep_output+=rep_embs[mod_ind]
                        # repの結合
                        tmp_enc_output=torch.cat((tmp_enc_output,tmp_rep_output),dim=1)
                        
                    tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
                    
                    if lyr_ind_i>=(self.start_fusion_layers - 1):
                        #rep_embs[mod_ind]
                        # enc_outからrep*3を分解
                        tmp_rep_output=tmp_enc_output[:,-rep_all_num:]
                        
                        rep_enc_output[mod_ind]=tmp_rep_output[:,mod_ind*rep_num:(mod_ind+1)*rep_num]
                        mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_all_num]
                    else:
                        # enc_outとrepを分解
                        mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_num]
                        rep_enc_output[mod_ind]=tmp_enc_output[:,-rep_num:]
                    
                    # if mod_ind==0:
                    #     # 分解したものでenc_outを更新
                    #     next_mod_enc_output=[tmp_enc_output]
                    #     next_mod_enc_output_append=next_mod_enc_output.append
                    # else:
                    #     next_mod_enc_output_append(tmp_enc_output)
                    
                    # if lyr_ind_i>=(self.start_fusion_layers - 1):
                    #     # repの更新するために和を保存
                    #     if mod_ind==0:
                    #         sum_btl=tmp_btl
                    #     else:
                    #         sum_btl=sum_btl+tmp_btl
                    # else:
                    #     # repを更新
                    #     rep_num[mod_ind]=tmp_btl
                # if lyr_ind_i>=(self.start_fusion_layers - 1):
                #     # 更新
                    
                
                #mod_enc_output=next_mod_enc_output
            
        # for enc_layer in self.layer_stack:
        #     enc_output += tem_enc
        #     enc_output, _ = enc_layer(
        #         enc_input=enc_output,
        #         non_pad_mask=non_pad_mask,
        #         slf_attn_mask=slf_attn_mask)
        
        for mod_ind in np.arange(mod_len):
            tmp_list=[torch.cat((mod_enc_output[mod_ind], rep_enc_output[mod_ind]) ,dim=1) ]
            if mod_ind==0:
                # inputをimaginary rep vecとcat
                list_output=tmp_list.copy()
            else:
                list_output.extend(tmp_list)
        if self.normalize_before==True:
            for mod_ind in np.arange(mod_len):
                tmp_enc_output=list_output[mod_ind]
                tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                if mod_ind==0:
                    next_enc_output=[tmp_enc_output]
                    next_enc_output_append=next_enc_output.append
                else:
                    next_enc_output_append(tmp_enc_output)
            enc_output=next_enc_output
        else:
            enc_output=list_output
        return enc_output#enc_output[[256,32,64][][]]

# 提案法用 mod-attention + rep-fusion方式
class modMoveEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner, n_dataset,
            n_layers, n_head, d_k, d_v, dropout, device, normalize_before,mod_sample,is_bottle_neck=True,
            time_linear=False,train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0,start_fusion_layers=1):
        super().__init__()
        
        self.n_layers=n_layers
        self.start_fusion_layers=start_fusion_layers
        # self.d_model = d_model
        self.device=device
        # # position vector, used for temporal encoding
        
        
        self.n_marks=len(mod_sample)
        
        
        self.layer_stack_modal = ModuleList([
            ModuleList([
                modEncLayer( select_mod_n=j, d_model=d_model, d_inner=d_inner,
                             n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                             normalize_before=normalize_before) for i in np.arange(n_layers)])  for j in np.arange(n_dataset)
        ])

        
        
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            ])

        self.modal_cat_list=ModuleList([nn.Linear(d_model,d_model, bias=False,device=device).to(device).double() for i in range(self.n_marks)])
        for ns_Linear in self.modal_cat_list:
            nn.init.xavier_uniform_(ns_Linear.weight)
    
    
    def forward(self, enc_input, rep_Mat=None,non_pad_mask=None,plot=None,imp=None,enc_plot=None,gene=None,allcat=None,emb_list=None):
            
        #enc_input [modal ,[Batch; seq_len, Dim=1]]
        # tau [ Batch, seq, dim=1]
        # marks[ Batch, seq, dim=marks(x,y,...)]
        #rep_Mat [modal; [Batch; rep_num; Dim=1]]
        #        [modal; [B; rep_num;Dim=2]
        mod_len=len(enc_input)
        batch_len=enc_input[0].shape[0]
        event_num=enc_input[0].shape[1]
        
        # input embedding
        for mod_ind in np.arange(mod_len):
            if mod_ind==0:
                tmp_input=enc_input[mod_ind] # inputをimaginary rep vecとcat
                tmp_emb=emb_list[mod_ind](tmp_input).to(torch.double) # embedding
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=tmp_emb.unsqueeze(0)
            else:
                tmp_emb=enc_input[mod_ind]
                for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=torch.cat([input_embs,tmp_emb.unsqueeze(0)],dim=0)
        dim=input_embs.shape[-1]

        # repも入力と別で、embeddingして、置いておく
        if rep_Mat is not None:
            rep_num = rep_Mat[0].shape[1]# [B,rep_num]
            for mod_ind in np.arange(mod_len):
                if mod_ind==0:
                    tmp_input=rep_Mat[mod_ind] # inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=rep_Mat[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)
        else:
            rep_num = 0
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=enc_input[mod_ind]# inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=enc_input[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)#3,128,3,64
        # Embedding 
        # input_embs[[256,29,64][][]]
        # rep_embs[[256,3,64][][]]
        # Encoding inputs
        
        #initial_S=mod_embs[0][0,-3:,:]
        
        # Generating mask
        # """ Encode event sequences via masked self-attention. """
        # #入力の時間エンコーディング
        
        # tem_enc = torch.cat((tem_enc,tem_rep), dim=1)#(16,seqence-1,M)->(16,seq-1+rep_vec_num,M)
        
        # modal 別でもイベントの見えている関係性は同じだから...
        if non_pad_mask is not None:
            if rep_Mat is not None:
                represent_mod_number=0
                #pdb.set_trace()
                
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                
                
                slf_attn_mask_subseq=torch.cat(
                    (torch.cat((slf_attn_mask_subseq,zeros((batch_len,rep_num,event_num),device=self.device)),dim=1)
                     ,zeros((batch_len,rep_num+event_num,rep_num)
                    ,device=self.device))
                    ,dim=2)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                
            else:
                pdb.set_trace()
                represent_mod_number=0
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input 合わせ
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=enc_input[represent_mod_number].squeeze(-1), seq_q=enc_input[represent_mod_number].squeeze(-1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None
        
        # enc_output = zeros(tem_enc.shape,device=self.device)
        
        # Attention
        mod_enc_output=[zeros((input_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        rep_enc_output=[zeros((rep_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        #mod_enc_output: [[128,29,64],[128,29,64],[128,29,64]]
        # if self.is_bottle_neck==True:
        #     btl=self.bottle_neck.repeat([batch_len,1,1])
        #     btl_num=btl.shape[1]
        
        mask_change_flg=0
        #
        rep_num=rep_embs.shape[2]
        rep_all_num=rep_embs.shape[0]*rep_embs.shape[2]
        # fig
        # if plot==True:
        #     time_enc_output_history=[[mod_embs[0],mod_embs[1],mod_embs[2]]]#(3=mod,256=batch,32=29+3,64=dim)
        #     if self.is_bottle_neck==True:
        #         share_enc_output_his=[btl]
        #     for lyr_ind in np.arange(self.n_layers):
                
        #         sum_btl=None
        #         lyr_ind_i = lyr_ind
        #         for mod_ind in np.arange(mod_len):
        #             # modal to tmp
        #             tmp_enc_output=mod_enc_output[mod_ind]
                    
        #             tmp_enc_output+=mod_embs[mod_ind]
        #             if (self.is_bottle_neck==True) and (lyr_ind_i>=(self.start_fusion_layers - 1)):
        #                 if self.is_bottle_neck==True:
        #                     #btlがあるときの入力作り直し btlが各モーダル入力の右に追加
        #                     tmp_enc_output=torch.cat((tmp_enc_output,btl),dim=1)
        #                 if mask_change_flg==0:
        #                     #btlがあるときのmask作り直し [B,L+S,L+S]->[B,L+S+btl,L+S+btl]
        #                     mask_change_flg=1
        #                     mask_Bts=zeros((batch_len,btl_num,slf_attn_mask_subseq.shape[2]),device=self.device)#0がマスクなし、1があり。最終行は0で構成.[B,btl,L] 
        #                     if rep_Mat is None:
        #                         mask_Bts[:,:,-1]+=1 #最後のイベントを見えないようにしないとbtlから間接的に真値情報がわたってしまう。
        #                     slf_attn_mask_subseq=torch.cat((torch.cat((slf_attn_mask_subseq,mask_Bts),dim=1),zeros((batch_len,event_num+rep_num+btl_num,btl_num),device=self.device)),dim=2)
        #                     slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1))
        #                     slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        #                     slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        #                     non_pad_mask = get_non_pad_mask(torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num+btl_num),device=self.device)),dim=1))
                        
        #             tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
        #                 enc_input=tmp_enc_output,
        #                 non_pad_mask=non_pad_mask,
        #                 slf_attn_mask=slf_attn_mask)
                    
        #             if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
        #                 tmp_btl=tmp_enc_output[:,-btl_num:]
        #                 tmp_enc_output=tmp_enc_output[:,:-btl_num]
        #             if mod_ind==0:
        #                 next_mod_enc_output=[tmp_enc_output]
        #                 next_mod_enc_output_append=next_mod_enc_output.append
        #             else:
        #                 next_mod_enc_output_append(tmp_enc_output)
                    
        #             if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
        #                 if mod_ind==0:
        #                     sum_btl=tmp_btl
        #                 else:
        #                     sum_btl=sum_btl+tmp_btl
                
        #         if self.is_bottle_neck==True and lyr_ind_i>=(self.start_fusion_layers - 1):
        #             btl=sum_btl/mod_len
        #         mod_enc_output=next_mod_enc_output
        #         time_enc_output_history.append(mod_enc_output)
        #         if self.is_bottle_neck==True:
        #             share_enc_output_his.append(btl)
                
        #     if self.is_bottle_neck==True:
        #         with open(f"./pickled/enc_output/{imp}share",mode="wb") as file:
        #             pickle.dump(share_enc_output_his,file)
        #     with open(f"./pickled/enc_output/{imp}proposed_encout",mode="wb") as file:
        #         pickle.dump(time_enc_output_history,file)
        #     pdb.set_trace()
        # else:
        for lyr_ind in np.arange(self.n_layers):#Transformer layers Number
            lyr_ind_i = lyr_ind
            for mod_ind in np.arange(mod_len):
                # 前の階層のenc出力を獲得
                tmp_enc_output=mod_enc_output[mod_ind]
                #
                tmp_enc_output+=input_embs[mod_ind]
                # tmp_enc_output [B:L,M]
                if (lyr_ind_i>=(self.start_fusion_layers - 1)):
                    
                    for i, list_rep_output in enumerate(rep_enc_output):
                        list_rep_output+=rep_embs[i]#B,rep_n,D
                        if i==0:
                            m_rep_output=list_rep_output.unsqueeze(0)
                        else:
                            m_rep_output=torch.cat([m_rep_output,list_rep_output.unsqueeze(0)],dim=0)
                            
                    
                    m_rep_output=torch.cat([m_rep_output[0],m_rep_output[1]],dim=1)
                    # 2, B, rep, Dim
                    tmp_enc_output=torch.cat((tmp_enc_output,m_rep_output),dim=1)
                    # 256, B , rep*2,Dim
                    if mask_change_flg==0:
                        # mv4=start2 CUDA_VISIBLE_DEVICES=1 python Main.py -gene=jisin --phase --pre_attn --vec -imp=mv4   --movevec -start_fusion_layers=2 --train
                        # イベントから該当マスクのみ見えるvisible_mask_for_event、非該当仮想はイベントとrepを見ないnot_visible_mask_for_rep=torch.cat((not_visible_mask_for_rep,ones((batch_len,rep_num,rep_all_num),device=self.device)),dim=2)#mv4
                        #CUDA_VISIBLE_DEVICES=1 python Main.py -gene=jisin --phase --pre_attn --vec -imp=mv4   --movevec
                        #CUDA_VISIBLE_DEVICES=1 python Main.py -gene=jisin --phase --pre_attn --vec -imp=mv1   --movevec
                        # mv2=イベントから該当マスクのみ見えるvisible_mask_for_event,非該当は仮想をみる
                        # mv1=イベントから全てのrepが見える。
                        #btlがあるときのmask作り直し [B,L+S,L+S]->[B,L+S+btl,L+S+btl]
                        # イベント履歴に仮想ベクトルを追加するマスク
                        # 対応する仮想ベクトルは0他は１
                        # 仮想ベクトル
                        mask_change_flg=1
                        
                        for make_allmod_mask_modal_ind in range(rep_embs.shape[0]):# あるmodイベント履歴が該当仮想ベクトルのみ見えるようにしている。#該当仮想ベクトルが仮想ベクトルと該当イベントのみ見えるようにしている。
                            # あるmodイベント履歴が該当仮想ベクトルのみ見えるようにしている。
                            slf_attn_mask_subseq = get_subsequent_mask(enc_input[0].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                            # place1 event to repS
                            visible_mask_for_event=zeros((batch_len,slf_attn_mask_subseq.shape[1],rep_num,
                                            ),device=self.device)#0がマスクなし、1があり。[B,L,rep] 
                            not_visible_mask_for_event=ones((batch_len,slf_attn_mask_subseq.shape[1],rep_num,
                                            ),device=self.device)#0がマスクなし、1があり。[B,L,rep] 
                            # if imp=="mv1" or imp=="mv3"or imp=="mv12"or imp=="mv32" or imp=="mvC1" or imp=="mvC3"or imp=="mvC12"or imp=="mvC32":
                            #         # イベントから見た仮想ベクトルのマスクの追加 全部見えてる
                            #     for make_mask_modal_ind in range(rep_embs.shape[0]):
                            #         slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_event),dim=2)
                            # else:#mv2  mv4
                                # イベントから見た仮想ベクトルのマスクの追加　該当以外はみえてない
                            for make_mask_modal_ind in range(rep_embs.shape[0]):
                                if make_mask_modal_ind==make_allmod_mask_modal_ind:
                                    slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_event),dim=2)
                                else:
                                    slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,not_visible_mask_for_event),dim=2)
                            
                            
                            # place2 rep to event
                            #仮想ベクトルから見たマスクの作成
                            # B,29,29+(mod*rep_num)
                            # イベントが見えてよいか同課の違いだけ。仮想ベクトル同士は見えている。
                            # rep がイベントを見てもよいか。
                            # visible_mask_for_rep=zeros((batch_len,rep_num,event_num),device=self.device)
                            # not_visible_mask_for_rep=zeros((batch_len,rep_num,event_num),device=self.device)
                            
                            # # place3 rep to rep 
                            # visible_mask_for_rep2rep=zeros((batch_len,slf_attn_mask_subseq.shape[1],rep_num),device=self.device)
                            # not_visible_mask_for_rep2rep=ones((batch_len,slf_attn_mask_subseq.shape[1],rep_num),device=self.device)
                            # place2 and 3
                            
                            # for repind in range(rep_embs.shape[0]):
                            #     for reprepind in range(rep_embs.shape[0]):
                            #         if repind==0:
                            #             if make_allmod_mask_modal_ind==repind:
                            #                 torch.cat((visible_mask_for_rep))
                            #                 visible_mask_for_rep=torch.cat((visible_mask_for_rep))
                            #             else:
                            #                 not_visible_mask_for_rep=
                                        
                            #         else:
                            #             visible_mask_for_mod_rep=torch.cat((visible_mask_for_mod_rep,))

                            # concat 2 and 3
                            visible_mask_for_rep=zeros((batch_len,rep_num,event_num+rep_all_num),device=self.device)
                            not_visible_mask_for_rep=ones((batch_len,rep_num,event_num+rep_all_num),device=self.device)

                            # if imp=="mv1" or imp=="mv2" or imp=="mv12" or imp=="mv22" or imp=="mvC1" or imp=="mvC2" or imp=="mvC12" or imp=="mvC22":
                            #     for make_mask_modal_ind in range(rep_embs.shape[0]):
                            #         slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_rep),dim=1)
                            # else:
                                
                            for make_mask_modal_ind in range(rep_embs.shape[0]):
                                if make_mask_modal_ind==make_allmod_mask_modal_ind:
                                    slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_rep),dim=1)
                                else:
                                    slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,not_visible_mask_for_rep),dim=1)
                            
                            # visible_mask_for_rep=torch.cat((visible_mask_for_rep,zeros((batch_len,rep_num,rep_all_num),device=self.device)),dim=2)
                            # if imp=="mv1":
                            #     not_visible_mask_for_rep=ones((batch_len,rep_num,event_num,
                            #              ),device=self.device)
                            # else:
                                
                            # if imp=="mv4":#repが"他の"repをみない
                            #     for make_mask_modal_ind in range(rep_embs.shape[0]):
                            #         if make_mask_modal_ind==0:
                            #             if make_mask_modal_ind==make_allmod_mask_modal_ind:#対応している
                            #                 not_visible_mask_for_rep_rep=zeros((batch_len,rep_num,rep_num))
                            #             else: #対応していない
                            #                 not_visible_mask_for_rep_rep=ones((batch_len,rep_num,rep_num))
                            #         else:
                            #             if make_mask_modal_ind==make_allmod_mask_modal_ind:#対応している
                            #                 not_visible_mask_for_rep_rep=torch.cat((not_visible_mask_for_rep_rep,zeros((batch_len,rep_num,rep_num))),dim=2)
                            #             else: #対応してない
                            #                 not_visible_mask_for_rep_rep=torch.cat((not_visible_mask_for_rep_rep,ones((batch_len,rep_num,rep_num))),dim=2)
                            #         if make_mask_modal_ind==0:
                            #             not_visible_mask_for_rep_rep_matome=not_visible_mask_for_rep_rep
                            #         else:
                            #             not_visible_mask_for_rep_rep_matome=not_visible_mask_for_rep_rep_matome
                            #         not_visible_mask_for_rep=torch.cat((not_visible_mask_for_rep,ones((batch_len,rep_num,rep_num),device=self.device)),dim=2)#mv4
                            #         #not_visible_mask_for_rep=torch.cat((not_visible_mask_for_rep,ones((batch_len,rep_num,rep_all_num),device=self.device)),dim=2)#mv4
                            # else:
                            #     not_visible_mask_for_rep=torch.cat((not_visible_mask_for_rep,zeros((batch_len,rep_num,rep_all_num),device=self.device)),dim=2)#mv2

                            # for make_mask_modal_ind in range(rep_embs.shape[0]):
                            #     if make_mask_modal_ind==make_allmod_mask_modal_ind:
                            #         slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_rep),dim=1)
                            #     else:
                            #         slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,not_visible_mask_for_rep[make_mask_modal_ind]),dim=1)
                            # mask_Bts=zeros((batch_len,rep_all_num-rep_num,
                            #             slf_attn_mask_subseq.shape[2]),device=self.device)#0がマスクなし、1があり。最終行は0で構成.[B,btl,L] 
                            if rep_Mat is None:
                                pdb.set_trace()
                                mask_Bts[:,:,-1]+=1 #最後のイベントを見えないようにしないとbtlから間接的に真値情報がわたってしまう。
                            
                            
                            
                            #slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                            slf_attn_mask_keypad = zeros(slf_attn_mask_subseq.shape,device=self.device)#slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                            if make_allmod_mask_modal_ind==0:
                                slf_attn_mask_all=slf_attn_mask.unsqueeze(0)
                            else:
                                slf_attn_mask_all=torch.cat([slf_attn_mask_all,slf_attn_mask.unsqueeze(0)],dim=0)
                            del visible_mask_for_rep,not_visible_mask_for_rep
                            gc.collect()
                            non_pad_mask = get_non_pad_mask(torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_all_num),device=self.device)),dim=1))
                        #     slf_attn_mask_subseq=torch.cat(
                        #         (torch.cat((slf_attn_mask_subseq,mask_Bts),dim=1),
                        #         zeros((batch_len,event_num+rep_all_num,rep_all_num-rep_num)
                        #         ,device=self.device))
                        #         ,dim=2)
                        # slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_all_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_all_num),device=self.device)),dim=1))
                        # slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                        # slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                            
                    slf_attn_mask=slf_attn_mask_all[mod_ind]
                else:
                    tmp_rep_output=rep_enc_output[mod_ind]
                    tmp_rep_output+=rep_embs[mod_ind]
                    # repの結合
                    tmp_enc_output=torch.cat((tmp_enc_output,tmp_rep_output),dim=1)
                if plot==True:
                    if (mod_ind==0) and (lyr_ind_i==0):
                        attn_score_time_list=[]
                        attn_score_mark_list=[]
                    tmp_enc_output, attn_score = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
                    if mod_ind==0:
                        attn_score_time_list.append(attn_score)
                    else:
                        attn_score_mark_list.append(attn_score)
                    
                else:
                    tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)

                
                if lyr_ind_i>=(self.start_fusion_layers - 1):
                    #rep_embs[mod_ind]
                    # enc_outからrep*3を分解
                    tmp_rep_output=tmp_enc_output[:,-rep_all_num:]
                    
                    rep_enc_output[mod_ind]=tmp_rep_output[:,mod_ind*rep_num:(mod_ind+1)*rep_num]
                    mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_all_num]
                else:
                    # enc_outとrepを分解
                    mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_num]
                    rep_enc_output[mod_ind]=tmp_enc_output[:,-rep_num:]
                
                # if mod_ind==0:
                #     # 分解したものでenc_outを更新
                #     next_mod_enc_output=[tmp_enc_output]
                #     next_mod_enc_output_append=next_mod_enc_output.append
                # else:
                #     next_mod_enc_output_append(tmp_enc_output)
                
                # if lyr_ind_i>=(self.start_fusion_layers - 1):
                #     # repの更新するために和を保存
                #     if mod_ind==0:
                #         sum_btl=tmp_btl
                #     else:
                #         sum_btl=sum_btl+tmp_btl
                # else:
                #     # repを更新
                #     rep_num[mod_ind]=tmp_btl
            # if lyr_ind_i>=(self.start_fusion_layers - 1):
            #     # 更新
                
            
            #mod_enc_output=next_mod_enc_output
        
        # for enc_layer in self.layer_stack:
        #     enc_output += tem_enc
        #     enc_output, _ = enc_layer(
        #         enc_input=enc_output,
        #         non_pad_mask=non_pad_mask,
        #         slf_attn_mask=slf_attn_mask)
        
        for mod_ind in np.arange(mod_len):
            tmp_list=[torch.cat((mod_enc_output[mod_ind], rep_enc_output[mod_ind]) ,dim=1) ]
            if mod_ind==0:
                # inputをimaginary rep vecとcat
                list_output=tmp_list.copy()
            else:
                list_output.extend(tmp_list)
        if self.normalize_before==True:
            for mod_ind in np.arange(mod_len):
                tmp_enc_output=list_output[mod_ind]
                tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                if mod_ind==0:
                    next_enc_output=[tmp_enc_output]
                    next_enc_output_append=next_enc_output.append
                else:
                    next_enc_output_append(tmp_enc_output)
            enc_output=next_enc_output
        else:
            enc_output=list_output
        if plot==True:
            # print(attn_score_time_list)
            # print(attn_score_mark_list)
            return enc_output, attn_score_time_list, attn_score_mark_list
        return enc_output#enc_output[[256,32,64][][]]

# rep-fusion方式 + mod attentionなし
class nomalattnMoveEncoder(nn.Module):
    # rep-fusion方式+mod-attentionなし(modalごとにWを用意しない一般的なAttention)
    def __init__(
            self,
            d_model, d_inner, n_dataset,
            n_layers, n_head, d_k, d_v, dropout, device, normalize_before,mod_sample,is_bottle_neck=True,
            time_linear=False,train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0,start_fusion_layers=1):
        super().__init__()
        
        self.n_layers=n_layers
        self.start_fusion_layers=start_fusion_layers
        # self.d_model = d_model
        self.device=device
        # # position vector, used for temporal encoding
        
        
        self.n_marks=len(mod_sample)
        
        
        self.layer_stack_modal = ModuleList([
            ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before) for i in np.arange(n_layers)])  for j in np.arange(n_dataset)
        ])
        
        
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            ])

        self.modal_cat_list=ModuleList([nn.Linear(d_model,d_model, bias=False,device=device).to(device).double() for i in range(self.n_marks)])
        for ns_Linear in self.modal_cat_list:
            nn.init.xavier_uniform_(ns_Linear.weight)
    
    
    def forward(self, enc_input, rep_Mat=None,non_pad_mask=None,plot=None,imp=None,enc_plot=None,gene=None,allcat=None,emb_list=None):
            
        #enc_input [modal ,[Batch; seq_len, Dim=1]]
        # tau [ Batch, seq, dim=1]
        # marks[ Batch, seq, dim=marks(x,y,...)]
        #rep_Mat [modal; [Batch; rep_num; Dim=1]]
        #        [modal; [B; rep_num;Dim=2]
        mod_len=len(enc_input)
        batch_len=enc_input[0].shape[0]
        event_num=enc_input[0].shape[1]
        
        # input embedding
        for mod_ind in np.arange(mod_len):
            if mod_ind==0:
                tmp_input=enc_input[mod_ind] # inputをimaginary rep vecとcat
                tmp_emb=emb_list[mod_ind](tmp_input).to(torch.double) # embedding
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=tmp_emb.unsqueeze(0)
            else:
                tmp_emb=enc_input[mod_ind]
                for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                tmp_emb=self.modal_cat_list[mod_ind](tmp_emb)
                input_embs=torch.cat([input_embs,tmp_emb.unsqueeze(0)],dim=0)
        dim=input_embs.shape[-1]

        # repも入力と別で、embeddingして、置いておく
        if rep_Mat is not None:
            rep_num = rep_Mat[0].shape[1]# [B,rep_num]
            for mod_ind in np.arange(mod_len):
                if mod_ind==0:
                    tmp_input=rep_Mat[mod_ind] # inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=rep_Mat[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)
        else:
            rep_num = 0
            for mod_ind in np.arange(mod_len):
            
                if mod_ind==0:
                    tmp_input=enc_input[mod_ind]# inputをimaginary rep vecとcat
                    tmp_emb=emb_list[mod_ind](tmp_input) # embedding
                    rep_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=enc_input[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    rep_embs=torch.cat([rep_embs,tmp_emb.unsqueeze(0)],dim=0)#3,128,3,64
        # Embedding 
        # input_embs[[256,29,64][][]]
        # rep_embs[[256,3,64][][]]
        # Encoding inputs
        
        #initial_S=mod_embs[0][0,-3:,:]
        
        # Generating mask
        # """ Encode event sequences via masked self-attention. """
        # #入力の時間エンコーディング
        
        # tem_enc = torch.cat((tem_enc,tem_rep), dim=1)#(16,seqence-1,M)->(16,seq-1+rep_vec_num,M)
        
        # modal 別でもイベントの見えている関係性は同じだから...
        if non_pad_mask is not None:
            if rep_Mat is not None:
                represent_mod_number=0
                #pdb.set_trace()
                
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                
                
                slf_attn_mask_subseq=torch.cat(
                    (torch.cat((slf_attn_mask_subseq,zeros((batch_len,rep_num,event_num),device=self.device)),dim=1)
                     ,zeros((batch_len,rep_num+event_num,rep_num)
                    ,device=self.device))
                    ,dim=2)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                
            else:
                pdb.set_trace()
                represent_mod_number=0
                slf_attn_mask_subseq = get_subsequent_mask(enc_input[represent_mod_number].squeeze(-1))# input 合わせ
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=enc_input[represent_mod_number].squeeze(-1), seq_q=enc_input[represent_mod_number].squeeze(-1))
                slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None
        
        # enc_output = zeros(tem_enc.shape,device=self.device)
        
        # Attention
        mod_enc_output=[zeros((input_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        rep_enc_output=[zeros((rep_embs[mod_i].shape),device=self.device) for mod_i in np.arange(mod_len)]
        #mod_enc_output: [[128,29,64],[128,29,64],[128,29,64]]
        # if self.is_bottle_neck==True:
        #     btl=self.bottle_neck.repeat([batch_len,1,1])
        #     btl_num=btl.shape[1]
        
        mask_change_flg=0
        #
        rep_num=rep_embs.shape[2]
        rep_all_num=rep_embs.shape[0]*rep_embs.shape[2]
        
        for lyr_ind in np.arange(self.n_layers):#Transformer layers Number
            lyr_ind_i = lyr_ind
            for mod_ind in np.arange(mod_len):
                # 前の階層のenc出力を獲得
                tmp_enc_output=mod_enc_output[mod_ind]
                #
                tmp_enc_output+=input_embs[mod_ind]
                # tmp_enc_output [B:L,M]
                if (lyr_ind_i>=(self.start_fusion_layers - 1)):
                    
                    for i, list_rep_output in enumerate(rep_enc_output):
                        list_rep_output+=rep_embs[i]#B,rep_n,D
                        if i==0:
                            m_rep_output=list_rep_output.unsqueeze(0)
                        else:
                            m_rep_output=torch.cat([m_rep_output,list_rep_output.unsqueeze(0)],dim=0)
                            
                    
                    m_rep_output=torch.cat([m_rep_output[0],m_rep_output[1]],dim=1)
                    # 2, B, rep, Dim
                    tmp_enc_output=torch.cat((tmp_enc_output,m_rep_output),dim=1)
                    # 256, B , rep*2,Dim
                    if mask_change_flg==0:
                        # mv4=start2 CUDA_VISIBLE_DEVICES=1 python Main.py -gene=jisin --phase --pre_attn --vec -imp=mv4   --movevec -start_fusion_layers=2 --train
                        # イベントから該当マスクのみ見えるvisible_mask_for_event、非該当仮想はイベントとrepを見ないnot_visible_mask_for_rep=torch.cat((not_visible_mask_for_rep,ones((batch_len,rep_num,rep_all_num),device=self.device)),dim=2)#mv4
                        #CUDA_VISIBLE_DEVICES=1 python Main.py -gene=jisin --phase --pre_attn --vec -imp=mv4   --movevec
                        #CUDA_VISIBLE_DEVICES=1 python Main.py -gene=jisin --phase --pre_attn --vec -imp=mv1   --movevec
                        # mv2=イベントから該当マスクのみ見えるvisible_mask_for_event,非該当は仮想をみる
                        # mv1=イベントから全てのrepが見える。
                        #btlがあるときのmask作り直し [B,L+S,L+S]->[B,L+S+btl,L+S+btl]
                        # イベント履歴に仮想ベクトルを追加するマスク
                        # 対応する仮想ベクトルは0他は１
                        # 仮想ベクトル
                        mask_change_flg=1
                        
                        for make_allmod_mask_modal_ind in range(rep_embs.shape[0]):# あるmodイベント履歴が該当仮想ベクトルのみ見えるようにしている。#該当仮想ベクトルが仮想ベクトルと該当イベントのみ見えるようにしている。
                            # あるmodイベント履歴が該当仮想ベクトルのみ見えるようにしている。
                            slf_attn_mask_subseq = get_subsequent_mask(enc_input[0].squeeze(-1))# input [B,29,29]　1が見えないように、0は見えてる。
                            # place1 event to repS
                            visible_mask_for_event=zeros((batch_len,slf_attn_mask_subseq.shape[1],rep_num,
                                            ),device=self.device)#0がマスクなし、1があり。[B,L,rep] 
                            not_visible_mask_for_event=ones((batch_len,slf_attn_mask_subseq.shape[1],rep_num,
                                            ),device=self.device)#0がマスクなし、1があり。[B,L,rep] 
                            # if imp=="mv1" or imp=="mv3"or imp=="mv12"or imp=="mv32" or imp=="mvC1" or imp=="mvC3"or imp=="mvC12"or imp=="mvC32":
                            #         # イベントから見た仮想ベクトルのマスクの追加 全部見えてる
                            #     for make_mask_modal_ind in range(rep_embs.shape[0]):
                            #         slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_event),dim=2)
                            # else:#mv2  mv4
                                # イベントから見た仮想ベクトルのマスクの追加　該当以外はみえてない
                            for make_mask_modal_ind in range(rep_embs.shape[0]):
                                if make_mask_modal_ind==make_allmod_mask_modal_ind:
                                    slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_event),dim=2)
                                else:
                                    slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,not_visible_mask_for_event),dim=2)
                            
                            # concat 2 and 3
                            visible_mask_for_rep=zeros((batch_len,rep_num,event_num+rep_all_num),device=self.device)
                            not_visible_mask_for_rep=ones((batch_len,rep_num,event_num+rep_all_num),device=self.device)
                                
                            for make_mask_modal_ind in range(rep_embs.shape[0]):
                                if make_mask_modal_ind==make_allmod_mask_modal_ind:
                                    slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,visible_mask_for_rep),dim=1)
                                else:
                                    slf_attn_mask_subseq=torch.cat((slf_attn_mask_subseq,not_visible_mask_for_rep),dim=1)
                            
                            if rep_Mat is None:
                                pdb.set_trace()
                                mask_Bts[:,:,-1]+=1 #最後のイベントを見えないようにしないとbtlから間接的に真値情報がわたってしまう。
                            
                            
                            
                            #slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1), seq_q=torch.cat((enc_input[represent_mod_number],torch.ones((batch_len,rep_num),device=self.device)),dim=1))
                            slf_attn_mask_keypad = zeros(slf_attn_mask_subseq.shape,device=self.device)#slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
                            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                            if make_allmod_mask_modal_ind==0:
                                slf_attn_mask_all=slf_attn_mask.unsqueeze(0)
                            else:
                                slf_attn_mask_all=torch.cat([slf_attn_mask_all,slf_attn_mask.unsqueeze(0)],dim=0)
                            del visible_mask_for_rep,not_visible_mask_for_rep
                            gc.collect()
                            non_pad_mask = get_non_pad_mask(torch.cat((enc_input[represent_mod_number].squeeze(-1),torch.ones((batch_len,rep_all_num),device=self.device)),dim=1))
                        
                    slf_attn_mask=slf_attn_mask_all[mod_ind]
                else:
                    tmp_rep_output=rep_enc_output[mod_ind]
                    tmp_rep_output+=rep_embs[mod_ind]
                    # repの結合
                    tmp_enc_output=torch.cat((tmp_enc_output,tmp_rep_output),dim=1)
                if plot==True:
                    if (mod_ind==0) and (lyr_ind_i==0):
                        attn_score_time_list=[]
                        attn_score_mark_list=[]
                    tmp_enc_output, attn_score = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
                    if mod_ind==0:
                        attn_score_time_list.append(attn_score)
                    else:
                        attn_score_mark_list.append(attn_score)
                    
                else:
                    tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        enc_input=tmp_enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)

                
                if lyr_ind_i>=(self.start_fusion_layers - 1):
                    #rep_embs[mod_ind]
                    # enc_outからrep*3を分解
                    tmp_rep_output=tmp_enc_output[:,-rep_all_num:]
                    
                    rep_enc_output[mod_ind]=tmp_rep_output[:,mod_ind*rep_num:(mod_ind+1)*rep_num]
                    mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_all_num]
                else:
                    # enc_outとrepを分解
                    mod_enc_output[mod_ind]=tmp_enc_output[:,:-rep_num]
                    rep_enc_output[mod_ind]=tmp_enc_output[:,-rep_num:]
                
        
        for mod_ind in np.arange(mod_len):
            tmp_list=[torch.cat((mod_enc_output[mod_ind], rep_enc_output[mod_ind]) ,dim=1) ]
            if mod_ind==0:
                # inputをimaginary rep vecとcat
                list_output=tmp_list.copy()
            else:
                list_output.extend(tmp_list)
        if self.normalize_before==True:
            for mod_ind in np.arange(mod_len):
                tmp_enc_output=list_output[mod_ind]
                tmp_enc_output = self.layer_norm[mod_ind](tmp_enc_output)
                if mod_ind==0:
                    next_enc_output=[tmp_enc_output]
                    next_enc_output_append=next_enc_output.append
                else:
                    next_enc_output_append(tmp_enc_output)
            enc_output=next_enc_output
        else:
            enc_output=list_output
        if plot==True:
            # print(attn_score_time_list)
            # print(attn_score_mark_list)
            #pdb.set_trace()
            return enc_output, attn_score_time_list, attn_score_mark_list
        return enc_output#enc_output[[256,32,64][][]]


class Decoder(nn.Module):
    """ A encoder model with self attention mechanism. """
    def __init__(
            self,
            d_model, d_inner,n_dataset,
            n_layers, n_head, d_k, d_v, dropout,device,normalize_before,
            mod_sample,is_bottle_neck=True,time_linear=False,
            train_max=0,train_x_max=0,train_x_min=0,train_y_max=0,train_y_min=0,start_fusion_layers=1,dec_btl=False):
        super().__init__()
        self.normalize_before=normalize_before
        self.d_model = d_model
        self.device = device
        self.n_layers=n_layers
        self.n_marks=len(mod_sample)
        
        self.layer_stack_modal = ModuleList([
            ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
            for _ in np.arange(n_layers)]) for _ in np.arange(n_dataset)
        ])
        
        if normalize_before:
            self.layer_norm = ModuleList([
                nn.LayerNorm(d_model, eps=1e-6)
                for _ in np.arange(n_dataset)
            ])
        if normalize_before:
            self.philayer_norm = nn.LayerNorm(d_model, eps=1e-6)#check
        self.train_max=train_max
        self.is_bottle_neck=dec_btl
        btl_num=3
        #self.bottle_neck = nn.Parameter(torch.randn(1,btl_num,d_model))
        
    


    def forward(self, dec_input,k,v,temp_enc=True,mode="else",imp=None,dec_plot=None,anc_plot=None,emb_list=None):
        if mode=="else":
            print("check your code")
            return 0
        elif mode=="phi":
            """
            dec_input [[128,1(1500)],[128,1(1500)],[128,1(1500)]]
            k[128,9,64]
            
            output([128, 3(4500), 64])
            """
            # 本のは128,3,64(anchor),128,500,64(anchor)
            #dec_input modal[time[B,L,M],x[],y[]...,[]]
            #k,v [B,time_s+x_s+y_s:M]
            #Encoding inputs
            mod_len=len(dec_input)
            batch_len=dec_input[0].shape[0]
            #---- inputs encoding
            
            for mod_ind in np.arange(mod_len):
                tmp_emb=dec_input[mod_ind]
                if mod_ind==0:
                    tmp_emb=emb_list[mod_ind](tmp_emb)#[B,1,M]
                    #mod_embs=tmp_emb
                    mod_embs=tmp_emb.unsqueeze(0)
                else:
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb.to(torch.double)).to(torch.double)
                
                    
                    #mod_embs=torch.cat([mod_embs,tmp_emb],dim=1)#[B,mod_n,M]
                    mod_embs=torch.cat([mod_embs,tmp_emb.unsqueeze(0)],dim=0)
            
            # -----\ inputs encoded
            dec_emb=mod_embs
            #output = zeros(dec_emb.shape,device=self.device)#([B,mod_n,D])

            # for dec_layer in self.layer_stack_phi:
            #     output += dec_emb #residual
            #     output, _ = dec_layer(
            #         output,
            #         k,#([128, 384])
            #         v,
            #         non_pad_mask=None,
            #         slf_attn_mask=None)
            # if self.normalize_before==True:##check
            #     output=self.philayer_norm(output)
            output = [zeros(mod_embs[mod_i].shape,device=self.device) for mod_i in np.arange(mod_len)]
            for lyr_ind in np.arange(self.n_layers):
                lyr_ind_i = lyr_ind
                for mod_ind in np.arange(mod_len):
                    # modal to tmp
                    tmp_enc_output=output[mod_ind]

                    tmp_enc_output+=mod_embs[mod_ind]
                    tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        tmp_enc_output,k[mod_ind],v[mod_ind],
                        non_pad_mask=None,
                        slf_attn_mask=None)
                    if mod_ind==0:
                        next_mod_enc_output=[tmp_enc_output]
                    else:
                        next_mod_enc_output.append(tmp_enc_output)
                output=next_mod_enc_output
            if self.normalize_before==True:
                next_enc_output=[self.layer_norm[mod_ind](output[mod_ind]) for mod_ind in np.arange(mod_len)]
                
                output=next_enc_output
            else:
                output=output
            return output

        elif mode=="anc":
            #x:temp_enb(B,L,M)
            #k,v:[B,S*mod_n*D]
            #dec_input {grand_truth, or anchor batch}
            #   GT [mordal; B; 1; Dim=1]
            #   anc_batch [modal; Batch; anc; Dim=1]
            
            
            mod_len=len(dec_input)
            batch_len=dec_input[0].shape[0]

            """ Encode event sequences via masked self-attention. """
            # prepare attention masks
            # slf_attn_mask is where we cannot look, i.e., the future and the padding
            
            #Encoding inputs
            for mod_ind in np.arange(mod_len):
                
                if mod_ind==0:
                    tmp_input=dec_input[mod_ind]
                    tmp_emb=emb_list[mod_ind](tmp_input)
                    mod_embs=tmp_emb.unsqueeze(0)
                else:
                    tmp_emb=dec_input[mod_ind]
                    for emb in emb_list[mod_ind]:
                        tmp_emb=emb(tmp_emb).to(torch.double)
                
                    mod_embs=torch.cat([mod_embs,tmp_emb.unsqueeze(0)],dim=0)


            # else:#初期値がencされない（いらない）
            #     x_tem_enc = dec_input.repeat([k.shape[0],1,1])

            # Attention
            mod_enc_output = [zeros(mod_embs[mod_i].shape,device=self.device) for mod_i in np.arange(mod_len)]
            for lyr_ind in np.arange(self.n_layers):
                lyr_ind_i = lyr_ind
                for mod_ind in np.arange(mod_len):
                    # modal to tmp
                    tmp_enc_output=mod_enc_output[mod_ind]

                    tmp_enc_output+=mod_embs[mod_ind]
                    tmp_enc_output, _ = self.layer_stack_modal[mod_ind][lyr_ind_i](
                        tmp_enc_output,k[mod_ind],v[mod_ind],
                        non_pad_mask=None,
                        slf_attn_mask=None)
                    if mod_ind==0:
                        next_mod_enc_output=[tmp_enc_output]
                    else:
                        next_mod_enc_output.append(tmp_enc_output)
                mod_enc_output=next_mod_enc_output
            if self.normalize_before==True:
                next_enc_output=[self.layer_norm[mod_ind](mod_enc_output[mod_ind]) for mod_ind in np.arange(mod_len)]
                
                dec_output=next_enc_output
            else:
                dec_output=mod_enc_output
            return dec_output


class Linear_layers(nn.Module):
    def __init__(
        self,d_model,d_out,dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_model,d_out)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)#check
    def forward(self,x):
        #[256,192] [256,1,128]
        out = self.linear(x)
        out = self.gelu(out)
        out = self.dropout(out)
        return out

class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim,linear_layers):
        super().__init__()
        layer_stack=[Linear_layers(dim,96),Linear_layers(96,48)]#[Linear_layers(int(dim/(2**i))) for i in np.arange(linear_layers-1)]
        layer_stack.append(nn.Linear(48,1))#(nn.Linear(int(dim/(2**(linear_layers-1))),1))
        self.layer_stack = ModuleList(layer_stack)
        for i in np.arange(len(self.layer_stack)-1):
            nn.init.xavier_uniform_(self.layer_stack[i].linear.weight)
        nn.init.xavier_uniform_(self.layer_stack[len(self.layer_stack)-1].weight)
    def forward(self, data):
        for linear_layer in self.layer_stack:
            data = linear_layer(data)
        return data

class modPredictor(nn.Module):
    """ 経過時間以外の mark \in R^{markの次元数} の予測のための関数"""

    def __init__(self, dim,linear_layers):
        super().__init__()
        xlayer_stack=[Linear_layers(dim,96),Linear_layers(96,48)]#[Linear_layers(int(dim/(2**i))) for i in np.arange(linear_layers-1)]
        xlayer_stack.append(nn.Linear(48,1))#(nn.Linear(int(dim/(2**(linear_layers-1))),1))
        self.xlayer_stack = ModuleList(xlayer_stack)
        for i in np.arange(len(self.xlayer_stack)-1):
            nn.init.xavier_uniform_(self.xlayer_stack[i].linear.weight)
        nn.init.xavier_uniform_(self.xlayer_stack[len(self.xlayer_stack)-1].weight)
        
        ylayer_stack=[Linear_layers(dim,96),Linear_layers(96,48)]#[Linear_layers(int(dim/(2**i))) for i in np.arange(linear_layers-1)]
        ylayer_stack.append(nn.Linear(48,1))#(nn.Linear(int(dim/(2**(linear_layers-1))),1))
        self.ylayer_stack = ModuleList(ylayer_stack)
        for i in np.arange(len(self.ylayer_stack)-1):
            nn.init.xavier_uniform_(self.ylayer_stack[i].linear.weight)
        nn.init.xavier_uniform_(self.ylayer_stack[len(self.ylayer_stack)-1].weight)
        
    def forward(self, data):
        
        xdata=data
        ydata=data
        for xlinear_layer in self.xlayer_stack:
            xdata = xlinear_layer(xdata)
        for ylinear_layer in self.ylayer_stack:
            ydata = ylinear_layer(ydata)
        return torch.cat([xdata,ydata],dim=1)

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            n_dataset,
            d_model=256, d_inner=1024,
            n_marks=2,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,
            rep_vec_num=3,anc_vec_num=3,time_step=20,device="cuda:0",
            method="normal",train_max=0,train_min=0,train_med=0, 
            linear_layers=1, normalize_before=True,mod_sample=None,
            is_bottle_neck=True,notMoveVec=False,
            train=None,opt=None):
        super().__init__()
        self.method = method
        if notMoveVec:#廃止
            self.encoder = Encoder(
                d_model=d_model,
                d_inner=d_inner,
                n_layers=n_layers,
                n_head=n_head,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                device=device,
                normalize_before=normalize_before,
                mod_sample=mod_sample,
                is_bottle_neck=is_bottle_neck,
                time_linear=False,
                train_max=train_max,
                train_x_max=opt.train_x_max,
                train_x_min=opt.train_x_min,
                train_y_max=opt.train_y_max,
                train_y_min=opt.train_y_min,
                n_dataset=n_dataset,
                start_fusion_layers=opt.start_fusion_layers,
                allcat=opt.allcat
            )
            self.decoder = Decoder(
                d_model=d_model,
                d_inner=d_inner,
                n_layers=n_layers,
                n_head=n_head,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                device=device,
                normalize_before=normalize_before,
                mod_sample=mod_sample,
                is_bottle_neck=is_bottle_neck,
                time_linear = False,
                train_max=train_max,
                train_x_max=opt.train_x_max,
                train_x_min=opt.train_x_min,
                train_y_max=opt.train_y_max,
                train_y_min=opt.train_y_min,
                n_dataset=n_dataset,
                start_fusion_layers=opt.start_fusion_layers
            )
            pdb.set_trace()
        else:
            if method=="early":#early fusion方式。
                self.encoder = EarlyMoveEncoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear=False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                )
                self.decoder = Decoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear = False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
            elif method=="late":# late fusion方式
                self.encoder = LateEncoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear=False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                )
                self.decoder = Decoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear = False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
            elif method=="cross":#cross attention fusion方式
                self.encoder = CrossEncoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear=False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                )
                self.decoder = Decoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear = False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
            elif method=="btl":#bottleneck fusion方式
                self.encoder = BtlEncoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear=False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                )
                self.decoder = Decoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear = False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
            elif method=="all":#提案法　(modal-attention + rep-fusion方式 (+- contrastive loss ))
                self.encoder = modMoveEncoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear=False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
                self.decoder = Decoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear = False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
            elif method=="ab_atn":#提案法のablationstudy.ver1　(modal-attention + (late-fusion) (+- contrastive loss )) 
                self.encoder = attnLateEncoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear=False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
                self.decoder = Decoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear = False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
            elif method=="ab_mov":#提案法のablationstudy.ver2　((attention) + rep-fusion (+- contrastive loss )) 
                self.encoder = nomalattnMoveEncoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear=False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
                self.decoder = Decoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear = False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
            else:
                self.encoder = MoveEncoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear=False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
                self.decoder = Decoder(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    device=device,
                    normalize_before=normalize_before,
                    mod_sample=mod_sample,
                    is_bottle_neck=is_bottle_neck,
                    time_linear = False,
                    train_max=train_max,
                    train_x_max=opt.train_x_max,
                    train_x_min=opt.train_x_min,
                    train_y_max=opt.train_y_max,
                    train_y_min=opt.train_y_min,
                    n_dataset=n_dataset,
                    start_fusion_layers=opt.start_fusion_layers
                )
        
        self.n_marks=n_marks
        ## embedding
        self.emb_list=[]
        self.marks_emb_linear1 = nn.Linear(self.n_marks,d_model)
        nn.init.xavier_uniform_(self.marks_emb_linear1.weight)
        self.marks_emb_linear2 = nn.Linear(d_model,d_model)
        nn.init.xavier_uniform_(self.marks_emb_linear2.weight)
        self.softmax=nn.Softmax(dim=2)
        emb_append=self.emb_list.append
        for mod_ind in mod_sample:
            if mod_ind=="time":
                emb_append(self.temporal_enc)
            elif mod_ind=="marks":
                emb_append(ModuleList([self.marks_emb_linear1.to(device).double(),self.marks_emb_linear2.to(device).double(),self.softmax]))
        self.position_vec = torch.tensor(
            [math.pow(train_max*1.5, 2.0 * (i // 2) / d_model) for i in np.arange(d_model)],
            device=device)
        ####
        
        mod_max=[train_max,opt.train_x_max,opt.train_y_max]
        mod_min=[0,opt.train_x_min,opt.train_y_min]
        
        #
        layer_stack=[Linear_layers(d_model*len(mod_sample),32) ,Linear_layers(32,16)]
        layer_stack.append(nn.Linear(16,1))
        self.layer_stack = ModuleList(layer_stack)
        
        for i in np.arange(len(self.layer_stack)-1):
            nn.init.xavier_uniform_(self.layer_stack[i].linear.weight)
        nn.init.xavier_uniform_(self.layer_stack[len(self.layer_stack)-1].weight)

        self.relu = nn.ReLU()
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.contrasive_tempareture = nn.Parameter(torch.tensor(1.0))
        
        
        if opt.method=="both_scalar":
            self.rep_vector=nn.ParameterList()
            for mod_ind in np.arange(n_dataset):
                # if mod_ind==0:#[1,1,rep_num]
                if opt.mod_sample[mod_ind]=="marks":
                    self.rep_vector.append(nn.Parameter(torch.Tensor(kmeans.Set_data_kmeans(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(opt.device).reshape((2, rep_vec_num))))
                else:
                    self.rep_vector.append(nn.Parameter(torch.Tensor(kmeans.Set_data_kmeans(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(opt.device).reshape((1, rep_vec_num))))
                #self.rep_vector.append(tmp)# B=1,vecnum
                # else:#[modal,1,rep_num]
            self.anchor_vector=nn.ParameterList()
            for mod_ind in np.arange(n_dataset):
                # if mod_ind==0:
                if opt.mod_sample[mod_ind]=="marks":
                    self.anchor_vector.append(nn.Parameter(torch.Tensor(kmeans.Set_data_kmeans(input=train, n_clusters=anc_vec_num,mod_ind=mod_ind,opt=opt)).to(opt.device).reshape((2,anc_vec_num))))# 
                else:
                    self.anchor_vector.append(nn.Parameter(torch.Tensor(kmeans.Set_data_kmeans(input=train, n_clusters=anc_vec_num,mod_ind=mod_ind,opt=opt)).to(opt.device).reshape((1,anc_vec_num))))# 
                # else:#[modal,1,rep_num]
                # self.anchor_vector = nn.ParameterList([self.anchor_vector,nn.Parameter(torch.Tensor(kmeans.Set_data_kmeans(input=train, n_clusters=anc_vec_num,mod_ind=mod_ind)).to(opt.device).reshape((1,anc_vec_num)))])


        
        elif opt.method=="rep_quan" or opt.method=="btl"or opt.method=="late" or opt.method=="all" or opt.method=="cross" or opt.method=="ab_atn" or opt.method=="ab_mov":
            self.rep_vector=nn.ParameterList()
            for mod_ind in np.arange(n_dataset):#[time,marks]で2回
                if opt.mod_sample[mod_ind]=="marks":
                    self.rep_vector.append(nn.Parameter(torch.Tensor(kmeans.choice_initial_means(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(torch.double).to(opt.device)))
                else:
                    self.rep_vector.append((nn.Parameter(torch.Tensor(kmeans.choice_initial_means(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(opt.device).reshape((rep_vec_num,1)))))
            self.anchor_vector=nn.ParameterList()
            for mod_ind in np.arange(n_dataset):
                if opt.mod_sample[mod_ind]=="marks":
                    self.anchor_vector.append(nn.Parameter(torch.Tensor(kmeans.choice_initial_means(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(opt.device).to(torch.double)))
                else:
                    self.anchor_vector.append(nn.Parameter(torch.Tensor(kmeans.choice_initial_means(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(opt.device).to(torch.double).reshape((rep_vec_num,1))))
        elif opt.method=="early":
            self.rep_vector=nn.ParameterList()            
            for mod_ind in np.arange(n_dataset):#[time,marks]で2回
                
                if opt.mod_sample[mod_ind]=="marks":
                    self.rep_vector.append(nn.Parameter(torch.Tensor(kmeans.choice_initial_means(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(torch.double).to(opt.device)))
                else:
                    self.rep_vector.append(nn.Parameter(torch.Tensor(kmeans.choice_initial_means(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(torch.double).to(opt.device).reshape((rep_vec_num,1))))
            self.anchor_vector=nn.ParameterList()
            for mod_ind in np.arange(n_dataset):
                if opt.mod_sample[mod_ind]=="marks":
                    self.anchor_vector.append(nn.Parameter(torch.Tensor(kmeans.choice_initial_means(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(torch.double).to(opt.device)))
                else:
                    self.anchor_vector.append(nn.Parameter(torch.Tensor(kmeans.choice_initial_means(input=train, n_clusters=rep_vec_num,mod_ind=mod_ind,opt=opt)).to(opt.device).to(torch.double).reshape((rep_vec_num,1))))
        if anc_vec_num>0:
            self.mod_predictor = ModuleList([
            
                Predictor(d_model*anc_vec_num,linear_layers),
                modPredictor(d_model*anc_vec_num,linear_layers)
                
            ])
        else:
            self.mod_predictor = ModuleList([
                Predictor(d_model,linear_layers,mod_max[i],mod_min[i])
                for _ in np.arange(n_dataset)
            ])
        self.rep_n=rep_vec_num
        self.anc_n=anc_vec_num
        
        self.rep_repeat=[]
        self.anc_repeat=[]

        self.train_std=opt.train_time_std
        self.train_mean=opt.train_time_mean
        #self.isNormalize=opt.normalize
        
        self.gene=opt.gene
        self.allcat=opt.allcat
    def temporal_enc(self, time):
        """
        Input: batch,seq_len,1.
                batch,1,3.
        Output: batch,seq_len,d_model.
        """
        result = time / self.position_vec
        after_result = zeros(result.shape,device=result.device)
        after_result[:,:, 0::2] = torch.sin(result[:, :,0::2])
        after_result[:, :, 1::2] = torch.cos(result[:, :,1::2])
        return after_result.to(result.device)
    
    def forward(self, model_input, model_target,plot=None,imp=None,enc_plot=None,reverse=False):
        ####
        # model_input[time_input[B,len,1],place_input[B,len,2]]
        # Return the hidden representations and predictions.
        # For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        # Input:  event_time: mordal_len*batch*seq_len
        #         Target:     mordal_len*batch*1
        # Output: enc_output: mordal_len*batch*(seq_len-1)*model_dim;
        #         time_prediction: mordal_len*batch*seq_len.
        ####
        # opt.vec判定は考慮されていない
        
        # input_time [Modal,B,len]
        # target [Modal,B,1]
        modal_num=len(model_input)
        batch_num=model_input[0].shape[0]
        if self.method=="early":
            non_pad_mask = get_non_pad_mask(torch.cat((model_input[0].squeeze(-1),torch.ones((batch_num,self.rep_n),device=model_input[0].device)),dim=1))
            # [modal,1,rep_num]->[modal,B,rep_num]
            # rep_batch = self.rep_vector[0].repeat([model_input.shape[0],1])y
            rep_expand_batch=[self.rep_vector[mod_i].repeat([batch_num,1,1]) for mod_i in np.arange(modal_num)]
                #rep_expand_batch=(rep_expand_batch,self.rep_vector[mod_i].repeat([batch_num,1,1]))
                #torch.cat(rep_expand_batch,self.rep_vector[mod_i].repeat([batch_num,1,1]),dim=0)
            enc_output = self.encoder(model_input,rep_Mat=rep_expand_batch,non_pad_mask=non_pad_mask,gene=self.gene,allcat=self.allcat,imp=imp,emb_list=self.emb_list)#,plot=plot
            
            enc_output_split=[enc_output[mod_i][:,-self.rep_n:,:] for mod_i in np.arange(modal_num)]
            

            # predict
            if self.anc_n>0:
                cat_outputMat=None
                # 
                for mod_i in np.arange(modal_num):
                    if mod_i == 0:
                        cat_outputMat=enc_output_split[mod_i]
                    else:
                        cat_outputMat=torch.cat((cat_outputMat,enc_output_split[mod_i]),dim=1)#[B,S1+S2,D]
                if reverse:
                    return 1,1,cat_outputMat
                anc_expand_batch=[self.anchor_vector[mod_i].repeat([batch_num,1,1]) for mod_i in np.arange(modal_num)]
                time_hidden = self.decoder(anc_expand_batch,k=enc_output_split,v=enc_output_split,imp=imp,mode="anc",emb_list=self.emb_list)
                time_pred_decout_flatten = [torch.flatten(time_hidden[mod_i],1) for mod_i in np.arange(modal_num)]
                time_prediction = [self.mod_predictor[mod_i](time_pred_decout_flatten[mod_i]) for mod_i in np.arange(modal_num)]
                # /predict 
                #---------------------------------
                #---------------------------------
                # hazard
                
                dec_output = self.decoder(model_target,k=enc_output_split,v=enc_output_split,mode="phi",dec_plot=plot,imp=imp,emb_list=self.emb_list)
                return dec_output, time_prediction, cat_outputMat,enc_output_split
        elif self.method=="btl":
            if self.rep_n>0:
                non_pad_mask = get_non_pad_mask(torch.cat((model_input[0].squeeze(-1),torch.ones((batch_num,self.rep_n),device=model_input[0].device)),dim=1))
                rep_expand_batch=[self.rep_vector[mod_i].repeat([batch_num,1,1]) for mod_i in np.arange(modal_num)]                    
                enc_output = self.encoder(model_input,rep_Mat=rep_expand_batch,non_pad_mask=non_pad_mask,gene=self.gene,allcat=self.allcat,imp=imp,emb_list=self.emb_list)#,plot=plot
                enc_output_split=[enc_output[mod_i][:,-self.rep_n:,:] for mod_i in np.arange(modal_num)]
            else:
                non_pad_mask = get_non_pad_mask(torch.cat((model_input[0].squeeze(-1),model_target[0]),dim=1))
                for modal_ind in np.arange(modal_num):
                    model_input[modal_ind]=torch.cat((model_input[modal_ind],model_target[modal_ind]),dim=1)
                
                enc_output = self.encoder(model_input,rep_Mat=None,non_pad_mask=non_pad_mask,gene=self.gene,emb_list=self.emb_list)#,plot=plot,imp=imp
                enc_output_split=[enc_output[mod_i][:,-2:-1,:] for mod_i in np.arange(modal_num)]
            #---------------------------------------------
            # predict
            if self.anc_n>0:
                cat_outputMat=None
                # 
                for mod_i in np.arange(modal_num):
                    if mod_i == 0:
                        cat_outputMat=enc_output_split[mod_i]
                    else:
                        cat_outputMat=torch.cat((cat_outputMat,enc_output_split[mod_i]),dim=1)#[B,S1+S2,D]
                if reverse:
                    return 1,1,cat_outputMat
                anc_expand_batch=[self.anchor_vector[mod_i].repeat([batch_num,1,1]) for mod_i in np.arange(modal_num)]
                time_hidden = self.decoder(anc_expand_batch,k=enc_output_split,v=enc_output_split,imp=imp,mode="anc",emb_list=self.emb_list)
                time_pred_decout_flatten = [torch.flatten(time_hidden[mod_i],1) for mod_i in np.arange(modal_num)]
                time_prediction = [self.mod_predictor[mod_i](time_pred_decout_flatten[mod_i]) for mod_i in np.arange(modal_num)]
                # /predict 
                #---------------------------------
                #---------------------------------
                # hazard
                dec_output = self.decoder(model_target,k=enc_output_split,v=enc_output_split,mode="phi",dec_plot=plot,imp=imp,emb_list=self.emb_list)
                return dec_output, time_prediction, cat_outputMat,enc_output_split
        
        elif self.method=="rep_quan" or self.method=="late" or self.method=="all" or self.method=="cross" or self.method=="ab_atn" or self.method=="ab_mov":
            if self.rep_n>0:
                non_pad_mask = get_non_pad_mask(torch.cat((model_input[0].squeeze(-1),torch.ones((batch_num,self.rep_n),device=model_input[0].device)),dim=1))
                # [modal,1,rep_num]->[modal,B,rep_num]
                # rep_batch = self.rep_vector[0].repeat([model_input.shape[0],1])y
                rep_expand_batch=[self.rep_vector[mod_i].repeat([batch_num,1,1]) for mod_i in np.arange(modal_num)]
                    #rep_expand_batch=(rep_expand_batch,self.rep_vector[mod_i].repeat([batch_num,1,1]))
                    #torch.cat(rep_expand_batch,self.rep_vector[mod_i].repeat([batch_num,1,1]),dim=0)
                enc_output = self.encoder(model_input,rep_Mat=rep_expand_batch,non_pad_mask=non_pad_mask,gene=self.gene,allcat=self.allcat,imp=imp,emb_list=self.emb_list)#,plot=plot
                enc_output_split=[enc_output[mod_i][:,-self.rep_n:,:] for mod_i in np.arange(modal_num)]
            else:
                non_pad_mask = get_non_pad_mask(torch.cat((model_input[0].squeeze(-1),model_target[0]),dim=1))
                for modal_ind in np.arange(modal_num):
                    model_input[modal_ind]=torch.cat((model_input[modal_ind],model_target[modal_ind]),dim=1)
                
                enc_output = self.encoder(model_input,rep_Mat=None,non_pad_mask=non_pad_mask,gene=self.gene,emb_list=self.emb_list)#,plot=plot,imp=imp
                enc_output_split=[enc_output[mod_i][:,-2:-1,:] for mod_i in np.arange(modal_num)]
            #enc_output = enc_output[:,-(rep_expand_batch[0].shape[1]):,:]
            #predict /
            #anchor_batch = self.anchor_vector.repeat([enc_output.shape[0],1])#Batch expand
            #---------------------------------------------
            # predict
            if self.anc_n>0:
                cat_outputMat=None
                # 
                for mod_i in np.arange(modal_num):
                    if mod_i == 0:
                        cat_outputMat=enc_output_split[mod_i]
                    else:
                        cat_outputMat=torch.cat((cat_outputMat,enc_output_split[mod_i]),dim=1)#[B,S1+S2,D]
                if reverse:
                    return 1,1,cat_outputMat
                anc_expand_batch=[self.anchor_vector[mod_i].repeat([batch_num,1,1]) for mod_i in np.arange(modal_num)]
                time_hidden = self.decoder(anc_expand_batch,k=enc_output_split,v=enc_output_split,imp=imp,mode="anc",emb_list=self.emb_list)
                time_pred_decout_flatten = [torch.flatten(time_hidden[mod_i],1) for mod_i in np.arange(modal_num)]
                time_prediction = [self.mod_predictor[mod_i](time_pred_decout_flatten[mod_i]) for mod_i in np.arange(modal_num)]
                # /predict 
                #---------------------------------
                #---------------------------------
                # hazard
                
                dec_output = self.decoder(model_target,k=enc_output_split,v=enc_output_split,mode="phi",dec_plot=plot,imp=imp,emb_list=self.emb_list)
                return dec_output, time_prediction, cat_outputMat,enc_output_split
        
        else:
            
            time_prediction = [self.mod_predictor[mod_i](enc_output_split[mod_i]) for mod_i in np.arange(modal_num)]            
            return enc_output, time_prediction,non_pad_mask,enc_output_split
            
if __name__=="__main__":
    #test

    encoder = MoveEncoder(
            d_model=64,
            d_inner=64,
            n_layers=4,
            n_head=8,
            d_k=8,
            d_v=8,
            dropout=0.1,
            normalize_before=True,
            mod_sample=None,
            is_bottle_neck=None,
            n_dataset=3,
            start_fusion_layers=2
            )