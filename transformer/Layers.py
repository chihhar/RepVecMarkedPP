import torch.nn as nn

from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, SelfAttention, ModSelfAttention,ModMultiHeadAttention,Rep_Separate_SelfAttention
import pdb
import torch
class modEncLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self,select_mod_n, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True,device=None):
        super(modEncLayer, self).__init__()
        self.slf_attn = Rep_Separate_SelfAttention(
            select_mod_n, n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before,device=device)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, mask=slf_attn_mask)##q=k=v
        if non_pad_mask is None:
            enc_output = self.pos_ffn(enc_output)
        else:
            enc_output *= non_pad_mask
            enc_output = self.pos_ffn(enc_output)
            enc_output *= non_pad_mask
        return enc_output, enc_slf_attn

class modDecoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, DL_name,d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(DecoderLayer, self).__init__()
        #self.slf_attn = MultiHeadAttention(
        #    n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.mlt_attn = ModMultiHeadAttention(
            DL_name,n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, dec_input, k, v, non_pad_mask=None, slf_attn_mask=None):
        #enc_output, enc_slf_attn = self.slf_attn(
        #    enc_input, enc_input, enc_input, mask=slf_attn_mask)##q=k=v
        #dec_output, _ = self.slf_attn(
        #    dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, attn = self.mlt_attn(
            dec_input, k, v, mask=slf_attn_mask)##q!=k=v
        if non_pad_mask is None:
            dec_output = self.pos_ffn(dec_output)
        else:
            dec_output *= non_pad_mask
            dec_output = self.pos_ffn(dec_output)
            dec_output *= non_pad_mask
        
        return dec_output, attn

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = SelfAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, mask=slf_attn_mask)##q=k=v
        
        if non_pad_mask is None:
            enc_output = self.pos_ffn(enc_output)
        else:
            enc_output *= non_pad_mask
            enc_output = self.pos_ffn(enc_output)
            enc_output *= non_pad_mask
        return enc_output, enc_slf_attn
class CrossEncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(CrossEncoderLayer, self).__init__()
        self.slf_attn = SelfAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.time_mlt_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.mark_mlt_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None,rep_num=None):
        
        batch_num,all_len,vec_dims=enc_input.shape
        time_mark_len=all_len-rep_num
        time_len=int(time_mark_len/2)
        mark_len=int(time_mark_len/2)
        
        # enc_input=[B, (time 29;mark 29;time_rep3;mark_rep3), dim]
        
        time_mark_eve=enc_input[:,:-rep_num]
        time_mark_rep=enc_input[:,-rep_num:]
        time_enc=torch.cat([time_mark_eve[:,:time_len],time_mark_rep[:,:int(rep_num/2)]],dim=1)
        mark_enc=torch.cat([time_mark_eve[:,time_len:],time_mark_rep[:,int(rep_num/2):]],dim=1)
        
        one_slf_attn_mask=slf_attn_mask[:,time_len:time_mark_len+int(rep_num/2),time_len:time_mark_len+int(rep_num/2)]
        time_enc_output, attn = self.time_mlt_attn(
            time_enc, mark_enc, mark_enc, mask=one_slf_attn_mask)##q!=k=v
        mark_enc_output, attn = self.mark_mlt_attn(
            mark_enc, time_enc, time_enc, mask=one_slf_attn_mask)##q!=k=v
        
        time_eve_enc=time_enc_output[:,:time_len]
        time_rep_enc=time_enc_output[:,time_len:]
        mark_eve_enc=mark_enc_output[:,:mark_len]
        mark_rep_enc=mark_enc_output[:,mark_len:]
        
        
        cat_output=torch.cat([time_eve_enc,mark_eve_enc,time_rep_enc,mark_rep_enc],dim=1)
        
        enc_output, enc_slf_attn = self.slf_attn(
            cat_output, mask=slf_attn_mask)##q=k=v
        
        if non_pad_mask is None:
            enc_output = self.pos_ffn(enc_output)
        else:
            enc_output *= non_pad_mask
            enc_output = self.pos_ffn(enc_output)
            enc_output *= non_pad_mask
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(DecoderLayer, self).__init__()
        #self.slf_attn = MultiHeadAttention(
        #    n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.mlt_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, dec_input, k, v, non_pad_mask=None, slf_attn_mask=None):
        #enc_output, enc_slf_attn = self.slf_attn(
        #    enc_input, enc_input, enc_input, mask=slf_attn_mask)##q=k=v
        #dec_output, _ = self.slf_attn(
        #    dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, attn = self.mlt_attn(
            dec_input, k, v, mask=slf_attn_mask)##q!=k=v
        if non_pad_mask is None:
            dec_output = self.pos_ffn(dec_output)
        else:
            dec_output *= non_pad_mask
            dec_output = self.pos_ffn(dec_output)
            dec_output *= non_pad_mask
        
        return dec_output, attn