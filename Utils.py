import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask
import pdb
import numpy as np
from scipy.spatial import distance
 
@torch.jit.script
def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

@torch.jit.script
def compute_event(event):
    """ Log-likelihood of events. """
    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    result = torch.log(event)
    return result

def THP_compute_integral_unbiased(model, data, input, target,opt):#, gap_f):
    """ Log-likelihood of non-events, using Monte Carlo integration. 
        data(B,1,M)
        input[B,seq-1]
        target[B,1]
        enc_out (B,seq-1,M)
    """
    
    #THP用
    #tauのrandom値を渡して、encoder2にtempencしてやる必要があるのでは？x
    num_samples = 1000
    mod_num=len(target)
    for mod_i in np.arange(mod_num):
        tmp_target=target[mod_i]
        if mod_i == 0:
            
            tmp_rand = tmp_target * torch.rand([target[mod_i].shape[0],num_samples], device=opt.device)
            cat_rand=[tmp_rand[:,np.newaxis,:]]
            cat_rand_append=cat_rand.append
        elif mod_i==1:
            tmp_rand = (torch.rand([target[mod_i].shape[0],num_samples], device=opt.device)*(opt.train_x_max-opt.train_x_min))+opt.train_x_min
            cat_rand_append(tmp_rand[:,np.newaxis,:])
        elif mod_i==2:
            tmp_rand = (torch.rand([target[mod_i].shape[0],num_samples], device=opt.device)*(opt.train_y_max-opt.train_y_min))+opt.train_y_min
            cat_rand_append(tmp_rand[:,np.newaxis,:])
    #random samples 
    # rand_time = target.unsqueeze(2) * \
    #             torch.rand([*target.size(), num_samples], device=opt.device)#[B,1,num_samples]
    
    #rand_time /= (target + 1)#[B,M]
    #B,100,M
    
    if opt.trainvec_num==0:
        
        for mod_i in np.arange(mod_num):
            tmp_target=target[mod_i]
            if mod_i == 0:
                tmp_output=data[mod_i][:,-2:-1,:]#time[B,L,D], x[B,L,D], y[B,L,D]-> time[B,-1,D]
            else:
                tmp_output=torch.cat((tmp_output,data[mod_i][:,-2:-1,:]),dim=1)
        
    for linear_layer in model.layer_stack:
        tmp_output = linear_layer(tmp_output)
    temp_hid = tmp_output#[B,1,1]
    
    temp_lambda = softplus(temp_hid + model.alpha * cat_rand[0]+cat_rand[1]+ cat_rand[2],model.beta)#[B,1,samples]
    all_lambda = torch.sum(temp_lambda,dim=2)/num_samples#[B,1]
    unbiased_integral = all_lambda * target[0]*(opt.train_x_max-opt.train_x_min)*(opt.train_y_max-opt.train_y_min) #/target
    
    
    return unbiased_integral

def compute_integral_unbiased(model, data, input, target, enc_output_split,opt):#, gap_f):
    """ Log-likelihood of non-events, using Monte Carlo integration. 
        data(B,1,M)
        input[B,seq-1]
        target[B,1]
        enc_out (B,1,M)
    """
    
    #if model.anchor_vector.requires_grad==True:
    #    torch.manual_seed(42)
    num_samples = 1500
    #random samples 
    
    # target=target[0]
    # rand_time = target * torch.rand([target.shape[0],num_samples], device=opt.device)

    mod_num=len(target)
    for mod_i in np.arange(mod_num):
        tmp_target=target[mod_i]
        
        # if mod_i == 0:
        #     tmp_rand = tmp_target * torch.rand([target[mod_i].shape[0],num_samples], device=opt.device)
        #     cat_rand=[tmp_rand]
        #     cat_rand_append=cat_rand.append
        # elif mod_i==1:
        #     tmp_rand = (torch.rand([target[mod_i].shape[0],num_samples], device=opt.device)*(opt.train_x_max-opt.train_x_min))+opt.train_x_min
        #     cat_rand_append(tmp_rand)
        # elif mod_i==2:
        #     tmp_rand = (torch.rand([target[mod_i].shape[0],num_samples], device=opt.device)*(opt.train_y_max-opt.train_y_min))+opt.train_y_min
        #     cat_rand_append(tmp_rand)
        if mod_i == 0:
            tmp_rand = tmp_target.squeeze(-1)  * torch.rand([target[mod_i].shape[0],num_samples], device=opt.device)#256,1 * 256,1500
            cat_rand=[tmp_rand.unsqueeze(-1)]
            cat_rand_append=cat_rand.append#256,1500
        elif mod_i==1:
            #targetは B,1,2
            tmp_rand=torch.rand([target[mod_i].squeeze().shape[0],num_samples,target[mod_i].squeeze().shape[1]], device=opt.device)#256,1500,2
            for i in range(opt.n_marks):
                if i==0:
                    tmp_rand[:,:,0]=tmp_rand[:,:,0]*(opt.train_x_max-opt.train_x_min)+opt.train_x_min
                elif i==1:
                    tmp_rand[:,:,1]=tmp_rand[:,:,1]*(opt.train_y_max-opt.train_y_min)+opt.train_y_min

            #tmp_rand = (torch.rand([*target[mod_i].squeeze().shape,num_samples], device=opt.device).transpose(1,2)*([opt.train_x_max-opt.train_x_min,opt.train_y_max-opt.train_y_min]))+opt.train_x_min
            cat_rand_append(tmp_rand)
    
    #B,500
    # dec_input [[128,1(1500)],[128,1(1500)],[128,1(1500)]]
    # k[128,9,64]
            
    # output([128, 3(4500), 64])

    #pdb.set_trace()
    # cat_rand [[256,1500,1][256,1500,2]]
    # split [[256, 3, 64],[256, 3, 64]]
    temp_output = model.decoder(cat_rand,k=enc_output_split,v=enc_output_split,mode="phi",emb_list=model.emb_list)
    # temp_output=torch.cat((temp_output[:,0:num_samples,:],
    #                        temp_output[:,num_samples:2*num_samples,:],
    #                        temp_output[:,2*num_samples:3*num_samples,:]),dim=2)
    temp_output=torch.cat([tmp for tmp in temp_output],dim=2)

    #pdb.set_trace()
    #[tau64;x64;y64
    #output=torch.cat((output[:,0:1,:],output[:,1:2,:],output[:,2:3,:]), dim=2)
    #B,500,M
    # temp_hid = model.linear2(temp_output)
    # relu_hid = model.relu(temp_hid)
    # a_lambda = model.linear(relu_hid)
    for linear_layer in model.layer_stack:
        temp_output = linear_layer(temp_output)
    a_lambda = temp_output
    #B,500,1
    temp_lambda = softplus(a_lambda,model.beta)
    #B,500,1
    all_lambda = torch.sum(temp_lambda,dim=1)/num_samples    
    #B,1
    
    unbiased_integral = all_lambda * target[0]*(opt.train_x_max-opt.train_x_min)*(opt.train_y_max-opt.train_y_min) #/target
    
    return unbiased_integral

def log_likelihood(model, output, input, target, enc_out,enc_output_split, opt):#, gap_f):
    #if model.isNormalize:
    #    input = (input-model.train_mean)/model.train_std
    #    target = (target-model.train_mean)/model.train_std
        
    """ Log-likelihood of sequence. """
    if opt.anc_vec_num==0:
        integral_output=output
        mod_num=len(target)
        for mod_i in np.arange(mod_num):
            tmp_target=target[mod_i]
            if mod_i == 0:
                tmp_output=output[mod_i][:,-1:,:]#time[B,L,D], x[B,L,D], y[B,L,D]-> time[B,-1,D]
                
            else:
                tmp_output=torch.cat((tmp_output,output[mod_i][:,-1:,:]),dim=1)
        output=tmp_output
        #output 256,1,64
        #B*mod_n*M output
        output_tensor=torch.cat((output[:,0:1,:],output[:,1:2,:],output[:,2:3,:]), dim=2)
    else:
        output_tensor=torch.cat([tmp for tmp in output],dim=2)# 256 1 192
    
    for linear_layer in model.layer_stack:
        
        output_tensor = linear_layer(output_tensor)
          
    all_hid = output_tensor
    
    #B*3*1->[B,3,1]
    all_lambda = softplus(all_hid,model.beta)
    all_lambda = torch.sum(all_lambda,dim=2)#(B,sequence,type)の名残
    #[B*1]
    
    # event log-likelihood
    event_ll = compute_event(all_lambda)#[B,1]
    event_ll = torch.sum(event_ll,dim=-1)#[B]
    #B*1*1
    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(, time, non_pad_mask)
    if opt.anc_vec_num==0:
        #non_event_ll = THP_compute_integral_unbiased(model, integral_output, input, target,opt)#[16,1]
        non_event_ll = compute_integral_unbiased(model, output, input, target, enc_output_split,opt)#[16,1]
    else:
        non_event_ll = compute_integral_unbiased(model, output, input, target, enc_output_split,opt)#[16,1]
    non_event_ll = torch.sum(non_event_ll, dim=-1)#[B]
    
    return event_ll, non_event_ll

def time_loss_se(prediction, target,mask):
    #prediction : (B,L-1,1)
    #event_time: (B,L)
    prediction = prediction[:,-1].squeeze(-1)
    target = target.reshape(prediction.shape)
    #t_1~t_L
    """ Time prediction loss. """
    # event time gap prediction
    diff = prediction - target
    
    mask=mask.reshape(diff.shape)
    se = torch.sum((diff * diff)*mask)
    return se
def time_se_list(prediction, target,mask):
    #prediction : (B,L-1,1)
    #event_time: (B,L)
    prediction = prediction[:,-1].squeeze(-1)
    target = target.reshape(prediction.shape)
    #t_1~t_L
    """ Time prediction loss. """
    # event time gap prediction
    diff = prediction - target
    
    mask=mask.reshape(diff.shape)
    seMat = ((diff * diff))
    return seMat.cpu()
def time_loss_ae(prediction, input, target,mask):
    prediction = prediction[:,-1].squeeze(-1)  
    target = target.reshape(prediction.shape)  
    # event time gap prediction
    diff = prediction - target
    mask=mask.reshape(diff.shape)
    ae = torch.sum(torch.abs(diff)*mask)
    return ae

def mark_se(prediction, target,mask):
    #prediction : (B,2)
    #event_time: (B,L)
    target = target.reshape(prediction.shape)
    #t_1~t_L
    """ Time prediction loss. """
    # event time gap prediction
    diff = prediction - target
    
    se = torch.sum((torch.sum(diff * diff,dim=1)))
    return se
def marks_se_list(prediction, target,mask):
    #prediction : (B,2)
    #event_time: (B,L)
    target = target.reshape(prediction.shape)
    #t_1~t_L
    """ Time prediction loss. """
    # event time gap prediction
    diff = prediction - target
    
    seMat = (torch.sum(diff * diff,dim=1))
    return seMat.cpu()
def mark_ae(prediction, input, target,mask):
    target = target.reshape(prediction.shape)
    #t_1~t_L
    """ Time prediction loss. """
    # event time gap prediction
    diff = prediction - target
    ae = torch.sum(torch.abs(diff))
    return ae

def time_mean_prediction(model, output, input, target, enc_out, opt):
    #output[B,1,M], input[B,seq]
    left=opt.train_mean*0.0001*torch.ones(target.shape,dtype=torch.float64,device=opt.device)
    #[B,1]
    right=opt.train_mean*100*torch.ones(target.shape,dtype=torch.float64,device=opt.device)
    #[B,1]
    #input = torch.cat((input,target),dim=1)#THP用
    for _ in np.arange(0,13):
        """
        #THP用
        center=(left+right)/2

        center = center.reshape(target.shape)
        output, _, enc_out = model(input,center)
        _, non_event_ll = log_likelihood(model, output, input, center, enc_out)
        value= non_event_ll-np.log(2)
        value = value.reshape(target.shape)#B,1
        left = (torch.where(value<0,center,left))#.unsqueeze(1)
        right = (torch.where(value>=0, center, right))#.unsqueeze(1)
        """
        
        
        #vec.pool用
        center=(left+right)/2
        output, _, enc_out = model(input,center)
        _, non_event_ll = log_likelihood(model, output, input, center, enc_out)
        value= non_event_ll-np.log(2)
        value = value.reshape(target.shape)#B,1
        left = (torch.where(value<0,center,left))
        right = (torch.where(value>=0, center, right))
        
    return (left+right)/2

def cosine_similarity_matrix(x1, x2, eps=1e-08): # dimは単純化のため省略
    w12 = torch.sum(x1 * x2,dim=2)
    w1 = torch.sum(x1 * x1,dim=2)
    w2 = torch.sum(x2 * x2,dim=2)
    n12 = (w1 * w2).clamp_min_(eps * eps).sqrt_()
    return w12 / n12
def uqulid_matrix(u, v):
    delta = u - v.detach()#256,3 - 256,3 = 256,3 自分ー相手の3次元距離が[i,:]で格納 #[N,] delta256,3 3,3
    m = ((u - v)**2).sum(dim=2)#本来は[1,3]*[3,3]*[3,1] [N,1]# (torch.mul(delta,x)).sum(1)
    return torch.sqrt(m)
def all_rep_d_sim_posiquan_contrastive_loss(model, enc_out, train_target, pred, opt,temparture=1.0):
    #all_contra03
    all_contrastive_loss=0
    batch, rep_n, dim=enc_out.shape
    n_marks = model.n_marks
    rep_n=model.rep_n
    for mod_n in range(n_marks):
        #opt.mod_sample
        
        time_enc_out=enc_out[:,mod_n*rep_n:(mod_n+1)*rep_n,:]
        if mod_n==0:
            tmp_input=train_target[mod_n]# inputをimaginary rep vecとcat
            tmp_emb=model.emb_list[mod_n](tmp_input) # embedding
        else:
            tmp_emb=train_target[mod_n]
            for emb in model.emb_list[mod_n]:
                tmp_emb=emb(tmp_emb).to(torch.double)
                
        #target_enc=model.emb_list[mod_n](train_target[mod_n])#256,1,64
        cos_tau_GT=cosine_similarity_matrix(tmp_emb,time_enc_out)#256,3
        #train_target#[B,1]
        #pdb.set_trace()
        
        target_mask= torch.Tensor(train_target[mod_n].shape)#256,1,1 
        # target_mask[((train_target<opt.rep12))]=0
        # target_mask[((train_target>=opt.rep12)*(train_target<opt.rep23))]=1
        # target_mask[((train_target>=opt.rep23))]=2
        if opt.mod_sample[mod_n]=="time":
            tmp_list=opt.rep_list
            for i in range(rep_n):
                # modの中でもrep3種
                if i ==0:
                    target_mask[((train_target[mod_n]<tmp_list[i]))]=i
                elif i==(rep_n-1):
                    target_mask[((train_target[mod_n]>=tmp_list[i-1]))]=i
                else: 
                    target_mask[((train_target[mod_n]>=tmp_list[i-1])*(train_target[mod_n]<tmp_list[i]))]=i
            
        elif opt.mod_sample[mod_n]=="marks":
            model.rep_vector[mod_n]#1番近いrepが担当
            tile_rep=torch.tile(model.rep_vector[mod_n],(batch,1,1))
            train_target[mod_n]#に一番近いrep#256,1,2に対して、256,3,2のどれが近いか
            
            target_mask=uqulid_matrix(tile_rep,train_target[mod_n]).min(dim=1)[1].unsqueeze(-1).unsqueeze(-1)
        near_Sd_number=target_mask.squeeze(-1).to(torch.long)#torch.max(cos_tau_GT,dim=1)[1]
        near_tau_for_S_mask=torch.nn.functional.one_hot(near_Sd_number,num_classes=rep_n).bool()
        
        bunsi=torch.sum(torch.exp(cos_tau_GT/temparture)*(near_tau_for_S_mask.squeeze(1).detach().to(opt.device)),dim=1)
        bunbo= torch.exp(cos_tau_GT/temparture).sum(dim=1)
        all_contrastive_loss-=compute_event(bunsi/bunbo).sum()

        
    return all_contrastive_loss
def used_contra(model, model_output, train_input, train_target, pred, opt,enc_out):
    if opt.imp =="all_contra":
        return all_rep_d_sim_posiquan_contrastive_loss(model, enc_out, train_target, pred, opt,temparture=1.0)
    elif opt.imp =="all_contra03":
        return all_rep_d_sim_posiquan_contrastive_loss(model, enc_out, train_target, pred, opt,temparture=0.3)
    elif opt.imp=="ncl":
        return torch.zeros(1).to(opt.device)
    else:
        return torch.zeros(1).to(opt.device)
        
