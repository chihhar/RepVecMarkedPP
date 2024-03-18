from matplotlib import pyplot as plt
import torch
import os
import numpy as np
import pdb
import pickle
import transformer.Models as trm
from tqdm import tqdm
from sklearn.manifold import TSNE
import Main
import math
import Utils
import pandas as pd

#import umap

dir_file=os.getcwd()

def generate_hawkes_modes():
    np.random.seed(seed=32)
    [T,LL,L_TRG1] = simulate_hawkes_modes(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return [T,score,L_TRG1]

def simulate_hawkes_modes(n,mu,alpha,beta,short_thre=1,long_thre=5):
    T = []
    LL = []
    L_TRG1 = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    is_long_mode = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        #step = np.random.exponential(scale=1)/l

        if l_trg1 > long_thre:
            is_long_mode = 1

        if l_trg1 < short_thre:
            is_long_mode = 0

        if is_long_mode: # long mode
            step = step = np.random.exponential(scale=2)/l
        else: # short mode
            step = np.random.exponential(scale=0.5)/l

        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            L_TRG1.append(l_trg1)
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
        
        if count == n:
            break
        
    return [np.array(T),np.array(LL),np.array(L_TRG1)]

def get_generate(data_type):
    if data_type == 'sp':
        return Main.generate_stationary_poisson()
    elif data_type == 'nsp':
        return Main.generate_nonstationary_poisson()
    elif data_type == 'sr':
        return Main.generate_stationary_renewal()
    elif data_type == 'nsr':
        return Main.generate_nonstationary_renewal()
    elif data_type == 'sc':
        return Main.generate_self_correcting()
    elif data_type == 'h1':
        return Main.generate_hawkes1()
    elif data_type == 'h2':
        return Main.generate_hawkes2()
    elif data_type == "h_fix":
        return generate_hawkes_modes()
    elif data_type == 'h_fix05':
        return Main.generate_hawkes_modes05()
def rolling_matrix(x,time_step):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

# 廃止
def Compare_event_GT_pred(model, test_data, opt):
    model.eval()
    GT_his=[]
    pred_his=[]
    
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,dynamic_ncols=True,
                          desc='  - (Validation) ', leave=False):
            mask=None
            if len(batch)==2:
                mask=batch[1]
                batch=batch[0]
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            test_input = event_time[:,:-1]
            test_target = event_time[:,-1:]
            if mask is not(None):
                mask=mask[:,-1].to(opt.device)
            else:
                mask=torch.ones(event_time.shape[0]).to(opt.device)
            mask=mask.reshape(test_target.shape)

            GT_his = np.append(GT_his,np.array(test_target.squeeze(-1).cpu()))
            
            _, pred, enc_out = model(test_input,test_target)
            pred = pred.reshape(test_target.shape)
            pred_his=np.append(pred_his,pred.cpu())
    print("Compare event and GT")
    
    plt.clf()
    plt.figure(figsize=(10,4)) 
    plt.xlabel("event ID",fontsize=18)
    plt.ylabel("elapsed time",fontsize=18)
    plt.plot(range(100),GT_his[200:300],label="ground-truth")
    plt.plot(range(100),pred_his[200:300],c="r",label="k-means anchor",linestyle="dashed")
    
    plt.legend(fontsize=18, loc='upper right')
        
        #plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+opt.method+opt.imp+".png", bbox_inches='tight', pad_inches=0)
        #plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+opt.method+opt.imp+".pdf", bbox_inches='tight', pad_inches=0)
        #np.save("THP_pred_his_64_jisin.npy",pred_his)
        #np.save("THP_pred_his_64preh1l4.npy",pred_his)
        #np.save("Transformer_FT-PPh164pre33phl4_tau_pred.npy",pred_his)
        #np.save("Transformer_FT-PP64jisin_tau_pred.npy",pred_his)
        #np.save("THP_pred_his_64_jisin.npy",pred_his)


def save_npy_synthetic(model, testloader, opt):
    """
    synthetic_dataのGT_intensityとhat_hazardを比較するもの
    
        hat_hazardをnpy形式にて保存する関数
    Args:
        
        ...

    Vars
        
    """
    
    model.eval()
    #select data:
    test_data=testloader.__iter__()
    test_datax = test_data.next()[20:29]
    
    #prepare data:
    event_time = test_datax.to(opt.device)
    input = event_time[:,:-1]
    target = test_datax[:,-1:]
    [T,score]=get_generate(opt.gene)
    test_datat=T[80000:]
    dT_test = np.ediff1d(test_datat)
    dT_test[10020:]
    rt_test = torch.tensor(rolling_matrix(dT_test,opt.time_step)).to(torch.double)

    t_min=0
    t_max=target.sum()+math.pow(10,-9)
    loop_start=0
    loop_num=5000
    loop_delta = (t_max-t_min)/loop_num
    print_progress_num = 1000
    
    cumsum_tau = torch.cumsum(target,dim=0).to(opt.device)
    log_likelihood_history = []
    non_log_likelihood_history = []
    target_history = []
    calc_log_l_history = []
    with torch.no_grad():
        for t in np.arange(loop_start,loop_num):
            if t % print_progress_num == 0:
                print(t)
            now_row_number = (target.size(0) - ( cumsum_tau > t*loop_delta+math.pow(10,-9)).sum().item())
            if now_row_number >= target.size(0):
                break
            
            now_input = input[now_row_number:now_row_number+1]
            now_target = target[now_row_number:now_row_number+1] 
            
            minus_target_value = cumsum_tau[now_row_number-1] if now_row_number >0 else 0
            variation_target = torch.tensor((t*loop_delta+math.pow(10,-9)),device=input.device)- minus_target_value
            
            variation_target = variation_target.reshape(now_target.shape)
            output, prediction, enc_out = model(now_input,variation_target)
            event_ll, non_event_ll = Utils.log_likelihood(model, output, now_input, variation_target,enc_out)            
            
            all_t = T[90020+opt.time_step]+t*loop_delta+math.pow(10,-9)
            if opt.gene =="sp":
                log_l_t = np.log(1)
            elif opt.gene =="nsp":
                log_l_t = np.log(0.99*np.sin((2*np.pi*all_t.cpu().numpy())/20000)+1)
            elif opt.gene=="h1":
                log_l_t = np.log(0.2 + (0.8*np.exp(-(all_t.cpu().numpy() - T[T<all_t.cpu().numpy()]))).sum())
            
            elif opt.gene=="h2":
                log_l_t = np.log(0.2 + (0.4*np.exp(-(all_t.cpu().numpy()-T[T<all_t.cpu().numpy()]))).sum() + (0.4*20*np.exp(-20*(all_t.cpu().numpy()-T[T<all_t.cpu().numpy()]))).sum())
            elif opt.gene=="sc":
                past_event_num = ((T<all_t.cpu().numpy()).sum())
                
                log_l_t = np.log(np.exp(all_t.cpu().numpy() - past_event_num))
            elif opt.gene=="sr":
                log_l_t = 0
            elif opt.gene=="nsr":
                log_l_t=0  

            calc_log_l_history = np.append(calc_log_l_history,log_l_t)
            log_likelihood_history = np.append(log_likelihood_history,event_ll.cpu().detach().numpy())
            non_log_likelihood_history =np.append(non_log_likelihood_history,non_event_ll.cpu().detach().numpy())
            target_history+=[t*loop_delta+math.pow(10,-9)]
    #np.save("npy_Matome/"+opt.method+opt.imp+opt.gene+"_calc_intensity.npy",log_likelihood_history)
    #np.save("npy_Matome/GT_intensity.npy",calc_log_l_history)
    #np.save("npy_Matome/target_history.npy",target_history)
    
    plt.clf()
    plt.figure(figsize=(8,8),dpi=300)
    plt.plot(target_history,calc_log_l_history,label=r"ground-truth",color="r")
    plt.scatter(cumsum_tau.cpu(),torch.zeros(cumsum_tau.shape)-2,marker='x',color="k",label="event-time")
    #THPSLOG=np.load("THP.npy")
    THP_ll=np.load(f"{dir_file}/npy_Matome/THPh164pre_l4h1_calc_intensity.npy")
    plt.plot(target_history,THP_ll,label=r"THP",color="b",linestyle="dashdot")
    #plt.plot(target_history,THPSLOG,label=r"THP",linestyle="dashed")
    plt.plot(target_history,log_likelihood_history,label=r"proposed method",color="g",linestyle="dotted")
    plt.ylim(-2.5,1.5)
    plt.xlabel(r"time", fontsize=18)
    plt.ylabel(r"log-intensity", fontsize=18)
    #メモリの数値
    #×の大きさ
    #線の太さは1?
    #線の種類
    #GTを太目黒
    #THPを青　マーク付きかどっと
    #proposedを赤　破線
    #plt.title("toy data", fontsize=18)
    plt.legend(fontsize=18)
    #for i in np.arange(input.shape[0]-1):
    #    plt.scatter(input[i+1].cpu()+cumsum_tau.cpu()[i],np.zeros(opt.time_step-1),color="y")
    #plt.plot(cumsum_tau.cpu(),to_plot_log_l.cpu(),label="true_log_l")
    #plt.plot(cumsum_tau.cpu(),to_plot_Int_l.cpu(),label="true_Int_l")
    #plt.xlabel("Time t")
    #plt.ylabel(r"Conditional_intensity log $\lambda(t|H_t)$")
    
    #plt.rc("pdf", fonttype="none")
    print("atest")
    plt.savefig("atest.pdf",bbox_inches='tight', pad_inches=0)
    
    with open(f"{dir_file}/pickled/proposed/{opt.gene}/proposed_{opt.imp}{opt.gene}_intensity", 'wb') as file:
        pickle.dump(log_likelihood_history , file)
    with open(f"{dir_file}/pickled/proposed/{opt.gene}/target_intensity", 'wb') as file:
        pickle.dump(target_history , file)
    with open(f"{dir_file}/pickled/proposed/{opt.gene}/True_intensity", 'wb') as file:
        pickle.dump(calc_log_l_history , file)
    with open(f"{dir_file}/pickled/proposed/{opt.gene}/eventtime_intensity", 'wb') as file:
        pickle.dump(cumsum_tau , file)
    pdb.set_trace()
    print("end")

#廃止
def near_tau_and_vector(model,opt):
    print("系列代表ベクトルに近いtauの探索(廃止)")
    #Division_Num=int(opt.train_max*4)
    Division_Num=50000
    """
    Division_time=[opt.train_max / Division_Num * i for i in np.arange(Division_Num)]
    tensor_time=torch.tensor(Division_time).to(opt.device).unsqueeze(0)
    encoded_time=model.encoder.temporal_enc(tensor_time)
    
    rep_near=np.zeros([opt.rep_vec_num,Division_Num])
    anchor_near=np.zeros([opt.anc_vec_num,Division_Num])
    
    for rep_n in np.arange(opt.rep_vec_num):
        #rep_near[rep_n]=np.argsort((torch.cosine_similarity(model.train_parameter[:,rep_n,:],encoded_time[0,:,:])).cpu().detach().numpy())[::-1]
        rep_near[rep_n]=np.argsort((torch.cosine_similarity(model.rep_vector[:,rep_n,:],encoded_time[0,:,:])).cpu().detach().numpy())[::-1]
    for anc_n in np.arange(opt.anc_vec_num):
        #anchor_near[anc_n]=np.argsort((torch.cosine_similarity(model.ancarvector[:,anc_n,:],encoded_time[0,:,:])).cpu().detach().numpy())[::-1]
        anchor_near[anc_n]=np.argsort((torch.cosine_similarity(model.anchor_vector[:,anc_n],encoded_time[0,:,:])).cpu().detach().numpy())[::-1]
    rep_near=rep_near.astype(int)
    anchor_near=anchor_near.astype(int)
    [((torch.cosine_similarity(model.anchor_vector[:,0,:],encoded_time[0,:,:])).cpu().detach().numpy()).max(),((torch.cosine_similarity(model.anchor_vector[:,1,:],encoded_time[0,:,:])).cpu().detach().numpy()).max(),((torch.cosine_similarity(model.ancarvector[:,2,:],encoded_time[0,:,:])).cpu().detach().numpy()).max()]
    print((np.array(Division_time)[rep_near])[:,:20])
    print((np.array(Division_time)[anchor_near])[:,:20])
    """
    #torch.cosine_similarity(model.ancarvector[:,:,:],model.ancarvector[:,::])  

#廃止
def synthetic_plot(model, plot_data, opt):
    print("synthetic plot")
    print("要 改善")
    with torch.no_grad():
        motonotime=[opt.train_max / 1000 * i for i in np.arange(1000)]
        tensortime=torch.tensor(motonotime).to(opt.device).unsqueeze(0)
        temptime=model.encoder.temporal_enc(tensortime)
        event_num=0
        ae=0
        all_num=0
        target_history=[]
        prediction_history=[]
        LLpred_history=[]
        for batch in tqdm(plot_data, mininterval=2,dynamic_ncols=True,
                          desc='  - (Validation) ', leave=False):
            if len(batch)==2:
                mask=batch[1]
                batch=batch[0]
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            train_input = event_time[:,:-1]
            train_target = event_time[:,-1:]
            
            target_history=np.append(target_history,train_target.cpu())
            """ forward """
            
            model_output, prediction,enc_out = model(train_input,train_target)
            
            #LLpred=Utils.time_mean_prediction(model,model_output,train_input,train_target,enc_out,opt)
            #LLpred_history=np.append(LLpred_history,LLpred.cpu())
            prediction = prediction.reshape(train_target.shape)
            prediction_history=np.append(prediction_history,prediction.cpu())
            ae+=np.abs(prediction.cpu() - train_target.cpu()).sum()
            
            event_num+=event_time.shape[0]
            plt.scatter(train_target.cpu(),prediction.cpu(),c='blue')                
        
        gosa=abs(target_history-prediction_history)
        len_e=gosa.shape[0]
        gosa.sort()
        hako=np.array([0.25,0.5,0.75])
        print(gosa[(len_e*hako).astype(int)])
        
        print(ae/event_num)
        #print((abs(target_history-LLpred_history).sum())/event_num)
        plt.xlabel('event index')
        plt.ylabel('prediction')
        plt.title('toy data')
        plt.savefig("plot/syn_tru/THP_event_index"+opt.imp+'.png', bbox_inches='tight', pad_inches=0)
        
        plt.clf()
        plt.xlabel(r'elapsed time$\tau$')
        plt.ylabel('count')
        #plt.title('tau count histgram')
        hist_min = np.append(prediction_history,target_history).min()
        hist_max = np.append(prediction_history,target_history).max()
        bins = np.linspace(hist_min, hist_max, 100)
        plt.ylim(0,2000)
        
        plt.hist([target_history,prediction_history],range=(0,3), bins=100, alpha = 0.5, label=['True',"Prediction"])
        #plt.hist(prediction_history, bins,alpha = 0.5, label='b')
        plt.legend()
        
        plt.savefig("plot/syn_hist/hist"+opt.gene+'_'+opt.imp+'_'+str(opt.time_step)+"_"+str(opt.epoch)+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig("plot/syn_hist/hist"+opt.gene+'_'+opt.imp+'_'+str(opt.time_step)+"_"+str(opt.epoch)+'.svg', bbox_inches='tight', pad_inches=0)
        plt.clf()
        
        plt.figure(figsize=(10,4))
        plt.xlabel("event iD",fontsize=18)
        plt.ylabel("elapsed time",fontsize=18)
        
        plt.plot(range(100),target_history[200:300],label="ground-truth")
        plt.plot(range(100),prediction_history[200:300],c="r",label="pred",linestyle="dashed")
        plt.legend(fontsize=18, loc='upper right')
        #plt.rc("pdf", fonttype="none")
        plt.savefig("plot/syn_hist/ID_time"+opt.wp+".pdf", bbox_inches='tight', pad_inches=0)
        plt.savefig("plot/syn_hist/seismic_id_timetau_hat_flatafnorm5_5+ID_time"+opt.imp+".pdf", bbox_inches='tight', pad_inches=0)
        print(np.sqrt((((target_history-target_history.mean())**2).sum())/target_history.shape[0]))
        gosa=abs(prediction_history-target_history)
        
        len_e=gosa.shape[0]
        gosa.sort()
        hako=np.array([0.25,0.5,0.75])
        print(gosa[(len_e*hako).astype(int)])
        print(np.sqrt((((gosa-gosa.mean())**2).sum())/gosa.shape[0]))
        
        print(np.sqrt((((prediction_history-prediction_history.mean())**2).sum())/prediction_history.shape[0]))
        sxx=(((target_history-target_history.mean())**2).sum())/target_history.shape[0]  
        syy=(((prediction_history-prediction_history.mean())**2).sum())/prediction_history.shape[0]
        sxy=(((target_history-target_history.mean())*(prediction_history-prediction_history.mean())).sum())/prediction_history.shape[0]
        corr=sxy/(np.sqrt(sxx)*np.sqrt(syy))
        
        print(corr)     

# 図8のための
def plot_phi(model, plot_data, opt):
    import seaborn as sns
    model.eval()
    time_all_input=torch.tensor([]).to(torch.double).to(opt.device, non_blocking=True)
    time_all_target=torch.tensor([]).to(torch.double).to(opt.device, non_blocking=True)
    mark_all_input=torch.tensor([]).to(torch.double).to(opt.device, non_blocking=True)
    mark_all_target=torch.tensor([]).to(torch.double).to(opt.device, non_blocking=True)
    all_input=None
    all_target=None
    with torch.no_grad():
        for batch in tqdm(plot_data, mininterval=2,dynamic_ncols=True,
                          desc='-(Valid)  ', leave=False):
            for mod_i in np.arange(len(batch)):
                tmp_batch=batch[mod_i].to(opt.device, non_blocking=True)
                tmp_input=tmp_batch[:,:-1]
                tmp_target=tmp_batch[:,-1:]
                if mod_i==0:
                    train_input=[tmp_input]
                    train_input_append=train_input.append
                    train_target=[tmp_target]
                    train_target_append=train_target.append
                    
                    time_all_input=torch.cat([time_all_input,tmp_input])
                    time_all_target=torch.cat([time_all_target,tmp_target])
                else:
                    train_input_append(tmp_input)#[[][]],[]
                    train_target_append(tmp_target)
                    mark_all_input=torch.cat([mark_all_input,tmp_input],dim=0)
                    mark_all_target=torch.cat([mark_all_target,tmp_target],dim=0)
            # #tmp_event=torch.stack(batch,dim=2)
            if all_input is None:
                all_input=[train_input]
                all_target=[train_target]
            else:
                all_input.append([train_input])
                all_target.append([train_target])

        for eve in range(150):#[[N,29,1],[N,29,2]]
            plt.clf()
            use_event_number=eve
            xtics=[]
            ytics=[]
            train_input=[time_all_input[eve:eve+1],mark_all_input[eve:eve+1]]#list(train_input[use_event_number].T.to(opt.device).reshape(3,1,29))#[[154,29,1][]]
            train_target=[time_all_target[eve:eve+1],mark_all_target[eve:eve+1]]#list(train_target[use_event_number].T.to(opt.device).reshape(3,1,1))
            model_output, prediction, enc_out,enc_split = model(train_input,train_target,imp=opt.imp)
            #event_ll, non_event_ll = Utils.log_likelihood(model, model_output, now_input, variation_target,enc_out)            
            bins=50
            x_minmax=opt.train_x_max-opt.train_x_min
            y_minmax=opt.train_y_max-opt.train_y_min
            
            phi_matome=torch.zeros([bins,bins])
            for loop_i in range(bins):
                    
                tmp_x=opt.train_x_min+loop_i*(x_minmax/bins)# x座標　最小値から最大値まで
                xtics.append(tmp_x)
                
                if loop_i%100==0:
                    print(loop_i)
                # train_target [[1,1,1],[1,1,2]]
                #tmp_target=torch.tile(torch.tensor([train_target[0],torch.tensor(tmp_x)]),(bins,1))
                tmp_target=torch.tile(torch.tensor([train_target[0]]),(bins,1)).to(opt.device)
                range_tmp_x=torch.tile(torch.tensor([torch.tensor(tmp_x)]),(bins,1)).to(opt.device)
                enc_split_tile=[x.repeat(bins,1,1) for x in enc_split]
                range_tmp_y=torch.range(start=opt.train_y_min, end=opt.train_y_max, step=y_minmax/bins)[:bins].reshape(bins,1).to(opt.device)
                #pdb.set_trace()
                cat_rand=torch.cat([range_tmp_x,range_tmp_y],dim=1).to(opt.device)
                cat_rand=[tmp_target.unsqueeze(1),cat_rand.unsqueeze(1)]
                #pdb.set_trace()
                temp_output = model.decoder(cat_rand,k=enc_split_tile,v=enc_split_tile,mode="phi",emb_list=model.emb_list)
                #5000,1 #[[5000,3,64],[5000,3,64]]
                output_tensor=torch.cat([tmp for tmp in temp_output],dim=2)
                for linear_layer in model.layer_stack:
                    output_tensor = linear_layer(output_tensor)
                all_hid = output_tensor
        
                #B*3*1->[B,3,1]
                all_lambda = Utils.softplus(all_hid,model.beta)
                all_lambda = torch.sum(all_lambda,dim=2)#(B,sequence,type)の名残
                # event log-likelihood
                event_ll = Utils.compute_event(all_lambda)#[B,1]
                event_ll = torch.sum(event_ll,dim=-1)#[B]
                phi_matome[loop_i]=event_ll

            sns.heatmap(phi_matome.flipud(),xticklabels=np.round(np.array(xtics),2),yticklabels= np.round(range_tmp_y.cpu().numpy(),1)[::-1])
            #plt.scatter(np.round(np.array(xtics),2)>train_target[1].cpu().numpy()[0][0][0],train_target[1].cpu().numpy()[0][0][1])
            plt.scatter( (np.round(np.array(xtics),2)>train_target[1].cpu().numpy()[0][0][0]).sum(), (np.round(range_tmp_y.cpu().numpy(),1)>train_target[1].cpu().numpy()[0][0][1]).sum(),marker="x")
            plt.title(f"{train_target[0].cpu().numpy()},{train_target[1].cpu().numpy()}")
            if not os.path.exists(f"plot/heat/{opt.method}__{opt.imp}/"):# 無ければ
                os.makedirs(f"plot/heat/{opt.method}__{opt.imp}/")
            plt.savefig(f"plot/heat/{opt.method}__{opt.imp}/{opt.gene}_{opt.imp}_{eve}.pdf")
            plt.clf()
            plt.close()
            #plt.savefig(f"./heat/heat{eve}.svg")

#図10？
def plot_attn(model, plot_data, opt):
    model.eval()
    time_all_input=torch.tensor([]).to(torch.double).to(opt.device, non_blocking=True)
    time_all_target=torch.tensor([]).to(torch.double).to(opt.device, non_blocking=True)
    mark_all_input=torch.tensor([]).to(torch.double).to(opt.device, non_blocking=True)
    mark_all_target=torch.tensor([]).to(torch.double).to(opt.device, non_blocking=True)
    all_input=None
    all_target=None
    with torch.no_grad():
        for batch in tqdm(plot_data, mininterval=2,dynamic_ncols=True,
                          desc='-(Valid)  ', leave=False):
            for mod_i in np.arange(len(batch)):
                tmp_batch=batch[mod_i].to(opt.device, non_blocking=True)
                tmp_input=tmp_batch[:,:-1]
                tmp_target=tmp_batch[:,-1:]
                if mod_i==0:
                    train_input=[tmp_input]
                    train_input_append=train_input.append
                    train_target=[tmp_target]
                    train_target_append=train_target.append
                    time_all_input=torch.cat([time_all_input,tmp_input])
                    time_all_target=torch.cat([time_all_target,tmp_target])
                else:
                    train_input_append(tmp_input)#[[][]],[]
                    train_target_append(tmp_target)
                    mark_all_input=torch.cat([mark_all_input,tmp_input],dim=0)
                    mark_all_target=torch.cat([mark_all_target,tmp_target],dim=0)
            # #tmp_event=torch.stack(batch,dim=2)
            if all_input is None:
                all_input=[train_input]
                all_target=[train_target]
            else:
                all_input.append([train_input])
                all_target.append([train_target])

        for eve in range(20):#[[N,29,1],[N,29,2]]
            plt.clf()
            use_event_number=eve
            xtics=[]
            ytics=[]
            train_input=[time_all_input[eve:eve+1],mark_all_input[eve:eve+1]]#list(train_input[use_event_number].T.to(opt.device).reshape(3,1,29))#[[154,29,1][]]
            train_target=[time_all_target[eve:eve+1],mark_all_target[eve:eve+1]]#list(train_target[use_event_number].T.to(opt.device).reshape(3,1,1))
            
            batch_num=train_input[0].shape[0]
            non_pad_mask = get_non_pad_mask(torch.cat((train_input[0].squeeze(-1),torch.ones((batch_num,model.rep_n),device=train_input[0].device)),dim=1))
            rep_expand_batch=[model.rep_vector[mod_i].repeat([batch_num,1,1]) for mod_i in np.arange(len(train_input))]
            
            enc_output, time_attn, mark_attn =model.encoder(train_input,rep_Mat=rep_expand_batch,non_pad_mask=non_pad_mask,gene=model.gene,allcat=model.allcat,imp=opt.imp,emb_list=model.emb_list,plot=True)
            import seaborn as sns
            if opt.method=="early":
                for i in range(len(time_attn)):
                    sns.heatmap(time_attn[i][0][0][58:,58:].cpu())
                    if not os.path.exists(f"{dir_file}/plot/attn/{opt.method}/{opt.gene}/{opt.imp}"):
                        os.makedirs(f"{dir_file}/plot/attn/{opt.method}/{opt.gene}/{opt.imp}")
                    plt.savefig(f"{dir_file}/plot/attn/{opt.method}/{opt.gene}/{opt.imp}/{eve}_heatcheak_{opt.imp}_{opt.method}{i}")
                    plt.clf()
                plt.close()
            elif opt.method=="all":
                
                for i in range(len(time_attn)):
                    sns.heatmap(torch.cat([time_attn[i][0][0][29:32,29:],mark_attn[i][0][0][32:35,29:]]).cpu())
                    if not os.path.exists(f"{dir_file}/plot/attn/{opt.method}/{opt.gene}/{opt.imp}"):
                        os.makedirs(f"{dir_file}/plot/attn/{opt.method}/{opt.gene}/{opt.imp}")
                    plt.savefig(f"{dir_file}/plot/attn/{opt.method}/{opt.gene}/{opt.imp}/{eve}_heatcheak_{opt.imp}_{opt.method}{i}")
                    plt.clf()
                plt.close()

# 学習曲線の保存
def plot_learning_curve(train_loss_his,valid_loss_his,opt,name=""):
    l_curve_folder_path=f"plot/loss_lc/{opt.imp}/"
    if not os.path.exists(l_curve_folder_path):# 無ければ
        os.makedirs(l_curve_folder_path) 
    plt.clf()
    plt.plot(range(len(train_loss_his)),train_loss_his,label="train_curve")
    plt.plot(range(len(valid_loss_his)),valid_loss_his,label="valid_curve")
    plt.legend()
    plt.savefig(f"plot/loss_lc/{opt.imp}/{opt.wp}_{name}.png", bbox_inches='tight', pad_inches=0)

# prediction plot
# 図6 図7をplotするためのコード
def phase_eventGT_prediction_plot(model, test_data,opt,plot=None,imp=None,enc_plot=None):
    time_GT_history=[]
    marks_GT_history=[]
    
    time_pred_history=[]
    marks_pred_history=[]
    
    all_event_num=0
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,dynamic_ncols=True,
                          desc='-(Test_plot)  ', leave=False):
            for mod_i in np.arange(len(batch)):
                # data 入力とターゲット分割
                tmp_batch=batch[mod_i].to(opt.device, non_blocking=True)
                tmp_input=tmp_batch[:,:-1]# [mod,過去系列]
                tmp_target=tmp_batch[:,-1:]
                
                if mod_i==0:
                    test_input=[tmp_input]
                    test_target=[tmp_target]
                    test_input_append=test_input.append
                    test_target_append=test_target.append
                else:
                    test_input_append(tmp_input)
                    test_target_append(tmp_target)
                    
            if opt.do_mask==False:
                for mod_i in np.arange(len(batch)):
                    if mod_i==0:
                        mask=[torch.ones(test_target[mod_i].shape).to(opt.device)]
                        mask_append=mask.append
                    else:
                        mask_append(test_target[mod_i].shape)
            #forward
            output, prediction, enc_out,enc_split = model(test_input,test_target,plot,imp=opt.imp,enc_plot=enc_plot)
            
            pred_time=prediction[0]
            pred_marks=prediction[1]
            
            time_GT=test_target[0]
            marks_GT=test_target[1]
            
            mask=mask[0]
            
            
            time_pred_history = np.append(time_pred_history,pred_time.cpu())
            marks_pred_history = np.append(marks_pred_history,pred_marks.cpu())

            time_GT_history = np.append(time_GT_history,time_GT.cpu())
            marks_GT_history = np.append(marks_GT_history,marks_GT.cpu())
            all_event_num+=mask.sum().item()
    #pdb.set_trace()  
    marks_pred_history=marks_pred_history.reshape(-1,2)
    marks_GT_history=marks_GT_history.reshape(-1,2)
    
    time_gosa=abs(time_pred_history-time_GT_history)
    marks_gosa=abs(marks_pred_history-marks_GT_history)
    
    time_se=((time_pred_history-time_GT_history)**2)
    marks_se=np.sqrt(((marks_pred_history-marks_GT_history)**2).sum(1))
    
    dir=f"{dir_file}/pickled/proposed/{opt.gene}/"
    if not os.path.exists(f"{dir}{opt.method}_{opt.imp}/"):# 無ければ
        os.makedirs(f"{dir}{opt.method}_{opt.imp}/") 

    # with open(f"{dir}{opt.wp}_ABS_Error", 'wb') as file:
    #     pickle.dump(time_gosa , file)

    with open(f"{dir}{opt.method}_{opt.imp}/{opt.gene}_timeTrue", 'wb') as file:
        pickle.dump(time_GT_history , file)
    with open(f"{dir}{opt.method}_{opt.imp}/{opt.gene}_marksTrue", 'wb') as file:
        pickle.dump(marks_GT_history , file)
    #/data1/nishizawa/RepVecMarkedPP/pickled/proposed/jisin
    
    #/data1/nishizawa/RepVecMarkedPP/pickled/proposed/jisin/'

    with open(f"{dir}{opt.method}_{opt.imp}/{opt.wp}{opt.gene}_timepred", 'wb') as file:
        pickle.dump(time_pred_history , file)
    with open(f"{dir}{opt.method}_{opt.imp}/{opt.wp}{opt.gene}_markspred", 'wb') as file:
        pickle.dump(marks_pred_history , file)
    
    
    print("plotting time pred and GT ")
    plt.clf()
    plt.figure(figsize=(20,4))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("event iD",fontsize=18)
    plt.ylabel("elapsed time",fontsize=18)
    max_event_num=300
    plt.plot(range(time_GT_history[0:max_event_num].shape[0]),time_GT_history[0:max_event_num],label="ground-truth")
    plt.plot(range(time_pred_history[0:max_event_num].shape[0]),time_pred_history[0:max_event_num],label="timepred",linestyle="dashdot")
    plt.legend(fontsize=18, loc='upper right')
    
    dir=f"{dir_file}/plot/event_GT/{opt.gene}/{opt.rep_vec_num}_{opt.anc_vec_num}/{opt.method}_{opt.imp}/"
    if not os.path.exists(dir):# 無ければ
        os.makedirs(dir) 
    plt.savefig(f"{dir}ID_time_{opt.wp}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}ID_time_{opt.wp}.png", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}ID_time_{opt.wp}.svg", bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    
    print("plotting lonlat GT ")
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)    
    
    plt.xlabel("GT lontitude",fontsize=18)
    plt.ylabel("GT latitude",fontsize=18)
    
    plt.xlim(opt.train_x_min, opt.train_x_max)
    plt.ylim(opt.train_y_min, opt.train_y_max)
    plt.scatter(marks_GT_history[:,0],marks_GT_history[:,1],label="ground-truth")
    plt.legend(fontsize=18, loc='upper right')
    
    dir=f"{dir_file}/plot/place_GT_map/{opt.gene}/{opt.rep_vec_num}_{opt.anc_vec_num}/{opt.method}_{opt.imp}/"
    if not os.path.exists(dir):# 無ければ
        os.makedirs(dir) 
    plt.savefig(f"{dir}GTplacemap_{opt.wp}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}GTplacemap_{opt.wp}.png", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}GTplacemap_{opt.wp}.svg", bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    print("plotting lonlat pred ")
    plt.clf()
    plt.figure(figsize=(6, 4))
    plt.xlim(opt.train_x_min, opt.train_x_max)
    plt.ylim(opt.train_y_min, opt.train_y_max)
    plt.xlabel("pred latitude",fontsize=18)
    plt.ylabel("pred longitude",fontsize=18)
    #pdb.set_trace()
    plt.scatter(marks_pred_history[:,0],marks_pred_history[:,1],label="prediction",c=marks_se)
    plt.colorbar()
    plt.legend(fontsize=18, loc='upper right')
    
    dir=f"{dir_file}/plot/place_pred_map/{opt.gene}/{opt.rep_vec_num}_{opt.anc_vec_num}/{opt.method}_{opt.imp}/"
    if not os.path.exists(dir):# 無ければ
        os.makedirs(dir) 
    plt.savefig(f"{dir}predmarks_{opt.wp}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}predmarks_{opt.wp}.png", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}predmarks_{opt.wp}.svg", bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    #上位10％
    sort_time=np.sort(time_GT_history)
    joui_compare_time=sort_time[int(len(sort_time)*0.9)]
    joui_pred_time=time_pred_history[time_GT_history>joui_compare_time]
    joui_GT_time=time_GT_history[time_GT_history>joui_compare_time]
    joui_time_se=(joui_pred_time-joui_GT_time)**2
    print(f"{opt.method}_{opt.imp} 上位10%:{np.sqrt(joui_time_se.mean()):8.3f}({np.sqrt(np.std(joui_time_se)):4.3f})")
    # 下位10％
    sort_time=np.sort(time_GT_history)
    kai_compare_time=sort_time[int(len(sort_time)*0.1)]
    kai_pred_time=time_pred_history[time_GT_history<kai_compare_time]
    kai_GT_time=time_GT_history[time_GT_history<kai_compare_time]
    kai_time_se=(kai_pred_time-kai_GT_time)**2
    print(f"{opt.method}_{opt.imp} 下位10%:{np.sqrt(kai_time_se.mean()):8.3f}({np.sqrt(np.std(kai_time_se)):4.3f})")
    
    
def plot_data_hist(data,opt):
    plt.clf()
    timeGT_his=[]
    lonGT_his=[]
    latGT_his=[]
    with torch.no_grad():
        for batch in tqdm(data, mininterval=2,dynamic_ncols=True,
                          desc='-(Test_plot)  ', leave=False):
            for mod_i in np.arange(len(batch)):
                tmp_batch=batch[mod_i].to(opt.device, non_blocking=True)
                tmp_input=tmp_batch[:,:-1]
                tmp_target=tmp_batch[:,-1:]
                if mod_i==0:
                    train_input=[tmp_input]
                    train_target=[tmp_target]
                else:
                    train_input.append(tmp_input)
                    train_target.append(tmp_target)
            if opt.do_mask==False:
                for mod_i in np.arange(len(batch)):
                    if mod_i==0:
                        mask=[torch.ones(train_target[mod_i].shape).to(opt.device)]
                    else:
                        mask.append(train_target[mod_i].shape)
            """ prepare data """
            
            timeGT_his=np.append(timeGT_his,tmp_target[0].cpu())
            lonGT_his=np.append(lonGT_his,tmp_target[1].cpu())
            latGT_his=np.append(latGT_his,tmp_target[2].cpu())
    plt.clf()
    plt.title(f"{opt.gene}-timehist")
    plt.hist(timeGT_his,bins=100)#911
    dir=f"{dir_file}/plot/timehist/"
    
    if not os.path.exists(dir):# 無ければ
        os.makedirs(dir) 
    plt.savefig(f"{dir}/data{opt.gene}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}/data{opt.gene}.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}/data{opt.gene}.png", bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.title(f"{opt.gene}-lonhist")
    plt.hist(lonGT_his,bins=100)#911
    dir=f"{dir_file}/plot/lonhist/"
    if not os.path.exists(dir):# 無ければ
        os.makedirs(dir) 
    plt.savefig(f"{dir}/data{opt.gene}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}/data{opt.gene}.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}/data{opt.gene}.png", bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.title(f"{opt.gene}-lathist")
    plt.hist(latGT_his,bins=100)#911
    dir=f"{dir_file}/plot/lathist/"
    if not os.path.exists(dir):# 無ければ
        os.makedirs(dir) 
    plt.savefig(f"{dir}/data{opt.gene}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}/data{opt.gene}.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir}/data{opt.gene}.png", bbox_inches='tight', pad_inches=0)

def cosine_similarity_matrix(x1, x2, eps=1e-08): # dimは単純化のため省略
    w12 = torch.sum(x1 * x2,dim=2)
    w1 = torch.sum(x1 * x1,dim=2)
    w2 = torch.sum(x2 * x2,dim=2)
    n12 = (w1 * w2).clamp_min_(eps * eps).sqrt_()
    return w12 / n12

#
def plot_con(model, test_data,opt,plot=None,imp=None,enc_plot=None):
    
    with torch.no_grad():
        for mod_n in range(3):
            cos_tau_GT_his=[]
            near_his=[]
            for batch in tqdm(test_data, mininterval=2,dynamic_ncols=True,
                            desc='-(Test_plot)  ', leave=False):
                for mod_i in np.arange(len(batch)):
                    # data 入力とターゲット分割
                    tmp_batch=batch[mod_i].to(opt.device, non_blocking=True)
                    tmp_input=tmp_batch[:,:-1]# [mod,過去系列]
                    tmp_target=tmp_batch[:,-1:]
                    
                    if mod_i==0:
                        test_input=[tmp_input]
                        test_target=[tmp_target]
                        test_input_append=test_input.append
                        test_target_append=test_target.append
                    else:
                        test_input_append(tmp_input)
                        test_target_append(tmp_target)
                        
                if opt.do_mask==False:
                    for mod_i in np.arange(len(batch)):
                        if mod_i==0:
                            mask=[torch.ones(test_target[mod_i].shape).to(opt.device)]
                            mask_append=mask.append
                        else:
                            mask_append(test_target[mod_i].shape)
                #forward
                output, prediction, enc_out,enc_split = model(test_input,test_target,plot,imp=opt.imp,enc_plot=enc_plot)
            
                batchSize, rep_n, dim=enc_out.shape
                rep_n = opt.rep_vec_num
                
                time_enc_out=enc_out[:,mod_n*rep_n:(mod_n+1)*rep_n,:]
                target_enc=model.decoder.emb_list[mod_n](test_target[mod_n])#256,1,64
                mod_target=test_target[mod_n]
                cos_tau_GT=cosine_similarity_matrix(target_enc,time_enc_out)#256,3
                cos_tau_GT_his=np.append(cos_tau_GT_his,cos_tau_GT.cpu())
                #train_target#[B,1]
                #pdb.set_trace()
        
                target_mask= torch.Tensor(test_target[mod_n].shape)
                # target_mask[((train_target<opt.rep12))]=0
                # target_mask[((train_target>=opt.rep12)*(train_target<opt.rep23))]=1
                # target_mask[((train_target>=opt.rep23))]=2
                
                if opt.mod_sample[mod_n]=="time":
                    tmp_list=opt.rep_list
                elif opt.mod_sample[mod_n]=="x_place":
                    tmp_list=opt.rep_list_x
                elif opt.mod_sample[mod_n]=="y_place":
                    tmp_list=opt.rep_list_y
                
                
                for i in range(rep_n):
                    if i ==0:
                        target_mask[((mod_target<tmp_list[i]))]=i
                    elif i==(rep_n-1):
                        target_mask[((mod_target>=tmp_list[i-1]))]=i
                    else: 
                        target_mask[((mod_target>=tmp_list[i-1])*(mod_target<tmp_list[i]))]=i
                near_Sd_number=target_mask.squeeze(-1).to(torch.long)#torch.max(cos_tau_GT,dim=1)[1]
                #pdb.set_trace()
                near_tau_for_S_mask=torch.nn.functional.one_hot(near_Sd_number,num_classes=rep_n).bool()
                near_his=np.append(near_his,near_tau_for_S_mask.cpu())
            
            cos_tau_GT_his=cos_tau_GT_his.reshape(-1,3)
            near_his=near_his.reshape(-1,3).astype(bool)
            dir=f"{dir_file}/plot/similarity/{opt.gene}/{opt.rep_vec_num}_{opt.anc_vec_num}/{opt.mod_sample[mod_n]}/"
            if not os.path.exists(dir):# 無ければ
                os.makedirs(dir) 
            
            
            plt.clf()
            plt.xlabel(r"$positive sim S^\prime_{1}$",fontsize=18)
            plt.ylabel(r"$negative sim S^\prime_{2} or S^\prime_{3}$",fontsize=18)
            
            posi=cos_tau_GT_his[near_his[:,0]][:,0]
            nega1=cos_tau_GT_his[near_his[:,0]][:,1]
            nega2=cos_tau_GT_his[near_his[:,0]][:,2]
            plt.scatter(posi,nega1,color="#ff7f0e",label=r"$nega S^\prime_2$")
            plt.scatter(posi,nega2,color="#2ca02c",label=r"$nega S^\prime_3$")
            plt.plot([-1,1],[-1,1],linestyle='dashed',color="k")
            plt.plot([posi.mean(),posi.mean()],[-1,1],linestyle='dashed',color="#1f77b4")
            plt.plot([-1,1],[nega1.mean(),nega1.mean()],linestyle='dashed',color="#ff7f0e")
            plt.plot([-1,1],[nega2.mean(),nega2.mean()],linestyle='dashed',color="#2ca02c")
            plt.savefig(f"{dir}p1VSn23{opt.imp}_{opt.mod_sample[mod_n]}_{opt.wp}.pdf", bbox_inches='tight', pad_inches=0)
            plt.savefig(f"{dir}p1VSn23{opt.imp}_{opt.mod_sample[mod_n]}_{opt.wp}.svg", bbox_inches='tight', pad_inches=0)

            
            plt.clf()
            plt.xlabel(r"$positive sim S^\prime_{2}$",fontsize=18)
            plt.ylabel(r"$negative sim S^\prime_{1} or S^\prime_{3}$",fontsize=18)
            
            posi=cos_tau_GT_his[near_his[:,1]][:,1]
            nega1=cos_tau_GT_his[near_his[:,1]][:,0]
            nega2=cos_tau_GT_his[near_his[:,1]][:,2]
            plt.scatter(posi,nega1,color="#1f77b4",label=r"$nega S^\prime_1$")
            plt.scatter(posi,nega2,color="#2ca02c",label=r"$nega S^\prime_3$")
            plt.plot([-1,1],[-1,1],linestyle='dashed',color="k")
            plt.plot([posi.mean(),posi.mean()],[-1,1],linestyle='dashed',color="#1f77b4")
            plt.plot([-1,1],[nega1.mean(),nega1.mean()],linestyle='dashed',color="#ff7f0e")
            plt.plot([-1,1],[nega2.mean(),nega2.mean()],linestyle='dashed',color="#2ca02c")
            plt.savefig(f"{dir}p2VSn13{opt.imp}_{opt.mod_sample[mod_n]}_{opt.wp}.pdf", bbox_inches='tight', pad_inches=0)
            plt.savefig(f"{dir}p2VSn13{opt.imp}_{opt.mod_sample[mod_n]}_{opt.wp}.svg", bbox_inches='tight', pad_inches=0)
    
            plt.clf()
            plt.xlabel(r"$positive sim S^\prime_{3}$",fontsize=18)
            plt.ylabel(r"$negative sim S^\prime_{1} or S^\prime_{2}$",fontsize=18)
            posi=cos_tau_GT_his[near_his[:,2]][:,2]
            nega1=cos_tau_GT_his[near_his[:,2]][:,0]
            nega2=cos_tau_GT_his[near_his[:,2]][:,1]
            plt.scatter(posi,nega1,color="#1f77b4",label=r"$nega S^\prime_1$")
            plt.scatter(posi,nega2,color="#ff7f0e",label=r"$nega S^\prime_2$")
            plt.plot([-1,1],[-1,1],linestyle='dashed',color="k")
            plt.plot([posi.mean(),posi.mean()],[-1,1],linestyle='dashed',color="#1f77b4")
            plt.plot([-1,1],[nega1.mean(),nega1.mean()],linestyle='dashed',color="#ff7f0e")
            plt.plot([-1,1],[nega2.mean(),nega2.mean()],linestyle='dashed',color="#2ca02c")
            plt.savefig(f"{dir}p3VSn12{opt.imp}_{opt.mod_sample[mod_n]}_{opt.wp}.pdf", bbox_inches='tight', pad_inches=0)
            plt.savefig(f"{dir}p3VSn12{opt.imp}_{opt.mod_sample[mod_n]}_{opt.wp}.svg", bbox_inches='tight', pad_inches=0)
