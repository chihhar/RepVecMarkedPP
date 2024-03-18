import os

dir_name=os.getcwd()

import argparse

import numpy as np
import pandas as pd


################################### for generating synthetic data
#from scipy.stats import lognorm,gamma
#from scipy.optimize import brentq
###################################
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import transformer.Constants as Constants
import Utils
from dataloadFolder import set_data

from matplotlib import pyplot as plt
import plot_code
from data import toy_data_generater as toy_dg
from transformer.Models import Transformer
from tqdm import tqdm
from functools import partial
from datetime import datetime as dt


def generate_data(data_type,opt):
    def rolling_matrix(x,time_step):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()
    def transform_data(T,n_train,n_validation,n_test,time_step,batch_size,opt):

        T_train = T[:n_train]
        T_valid = T[n_train:n_train+n_validation]
        T_test = T[n_train+n_validation:n_train+n_validation+n_test]
        
        dT_train = np.ediff1d(T_train)
        opt.train_mean=dT_train.mean()
        opt.train_std=np.std(T_train)
        train_data = torch.tensor(rolling_matrix(dT_train,time_step)).to(torch.double)
        dT_valid = np.ediff1d(T_valid)
        valid_data = torch.tensor(rolling_matrix(dT_valid,time_step)).to(torch.double)
        dT_test = np.ediff1d(T_test)
        test_data = torch.tensor(rolling_matrix(dT_test,time_step)).to(torch.double)
        print(f"shape: {test_data[:,-1:].shape}")
        
        with open(f'numpy_{opt.gene}_GT_his.pkl', 'wb') as file:
            pickle.dump(test_data[:,-1:].cpu().numpy(),file)
        
        return torch.utils.data.DataLoader(train_data,num_workers=2,batch_size=batch_size,pin_memory=True,shuffle=True),\
        torch.utils.data.DataLoader(valid_data,num_workers=2,batch_size=batch_size,pin_memory=True,shuffle=False), \
        torch.utils.data.DataLoader(test_data,num_workers=2,batch_size=batch_size,pin_memory=True,shuffle=False)\
        ,dT_train.max(),dT_train.min(),np.median(dT_train)
    
    if data_type == 'sp':
        [T,score_ref] = toy_dg.generate_stationary_poisson()
    elif data_type == 'nsp':
        [T,score_ref] = toy_dg.generate_nonstationary_poisson()
    elif data_type == 'sr':
        [T,score_ref] = toy_dg.generate_stationary_renewal()
    elif data_type == 'nsr':
        [T,score_ref] = toy_dg.generate_nonstationary_renewal()
    elif data_type == 'sc':
        [T,score_ref] = toy_dg.generate_self_correcting()
    elif data_type == 'h1':
        [T,score_ref] = toy_dg.generate_hawkes1()
    elif data_type == 'h2':
        [T,score_ref] = toy_dg.generate_hawkes2()
    elif data_type == 'ee':
        [T,score_ref] = toy_dg.generate_eahawkes()
    elif data_type == 'h_fix':
        [T,score_ref] = toy_dg.generate_hawkes_modes()
    elif data_type == 'h_fix05':
        [T,score_ref] = toy_dg.generate_hawkes_modes05()
    n = T.shape[0]
    time_step=opt.time_step
    batch_size=opt.batch_size

    trainloader, validloader, testloader ,train_max, train_min,train_med= transform_data(T,int(n*0.8),int(n*0.1),int(n*0.1),time_step,batch_size,opt) # A sequence is divided into training and test data.
    
    return trainloader, validloader,testloader,train_max,train_min,train_med
def set_data_function(df,opt):
    def rolling_matrix(x,time_step):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()
    
    df["Time_second"] = df["DateTime"].map(pd.Timestamp.timestamp)/3600##UNIX変換
    df_train=df[:int(len(df)*0.8)]
    dT_train=np.ediff1d(df_train["Time_second"])
    train_data = torch.tensor(rolling_matrix(dT_train,opt.time_step)).to(torch.double)
    train_dataset = torch.utils.data.TensorDataset(train_data)
    df_valid=df[int(len(df)*0.8):int(len(df)*0.9)]
    dT_valid=np.ediff1d(df_valid["Time_second"])
    rT_valid = torch.tensor(rolling_matrix(dT_valid,opt.time_step)).to(torch.double)
    df_test = df[int(len(df)*0.9):]
    df_test = df_test.reset_index()
    
    dT_test = np.ediff1d(df_test["Time_second"])
    rT_test = torch.tensor(rolling_matrix(dT_test,opt.time_step)).to(torch.double)
    
    trainloader = torch.utils.data.DataLoader(train_data,num_workers=2,batch_size=opt.batch_size,pin_memory=True,shuffle=True)
    validloader = torch.utils.data.DataLoader(rT_valid,num_workers=2,batch_size=opt.batch_size,pin_memory=True,shuffle=False)
    testloader = torch.utils.data.DataLoader(rT_test,num_workers=2,batch_size=opt.batch_size,pin_memory=True,shuffle=False)

    opt.train_mean=dT_train.mean()
    opt.train_std=np.std(dT_train)
    train_max=dT_train.max()
    train_min=dT_train.min()
    train_med=np.median(dT_train)

    return trainloader, validloader, testloader,train_max,train_min,train_med

################
### Early Stop
################
class EarlyStopping:
    def __init__(self,patience=10, verbose=False, path='c_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1
            if self.verbose:  #表示を有b効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0
    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

################
### Train
################
def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """
    model.train()
    
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_place_se = 0
    total_num_event = 0  # number of total events
    total_time_ae = 0
    total_place_ae = 0
    
    total_contrastive=0
    time_se_history=[]
    place_se_history=[]
    marks_pred_history=[]
    for batch in tqdm(training_data, mininterval=2,dynamic_ncols=True,
                          desc='-(Training)  ', leave=False):
        for mod_i in np.arange(len(batch)):
            tmp_batch=batch[mod_i].to(opt.device, non_blocking=True)
            tmp_input=tmp_batch[:,:-1]
            tmp_target=tmp_batch[:,-1:]
            if mod_i==0:
                train_input=[tmp_input]
                train_input_append=train_input.append
                train_target=[tmp_target]
                train_target_append=train_target.append
            else:
                train_input_append(tmp_input)
                train_target_append(tmp_target)
        if opt.do_mask==False:
            for mod_i in np.arange(len(batch)):
                if mod_i==0:
                    mask=[torch.ones(train_target[mod_i].shape).to(opt.device)]
                    mask_append=mask.append
                else:
                    mask_append(train_target[mod_i].shape)
        
        
        
        """ forward """
        optimizer.zero_grad()
        model_output, prediction, enc_out,enc_split = model(train_input,train_target,imp=opt.imp)
        
        torch.autograd.set_detect_anomaly(True)
        """ backward """
        event_ll, non_event_ll = Utils.log_likelihood(model, model_output, train_input, train_target, enc_out,enc_split,opt)
        contrastive_loss = Utils.used_contra(model, model_output, train_input, train_target, prediction,opt,enc_out)
        #contrastive_loss = Utils.used_contra(model, model_output, train_input, train_target, enc_out,opt,event_ll,enc_split,prediction)
        
        event_loss = -torch.sum((event_ll - non_event_ll)*mask[0].reshape(event_ll.shape))
        
        se = Utils.time_loss_se(prediction[0],  train_target[0],mask[0])#[]
        place_se = Utils.mark_se(prediction[1],  train_target[1],mask[0])
        
        with torch.no_grad():
            ae = Utils.time_loss_ae(prediction[0], train_input, train_target[0],mask[0])#[]
            place_ae = Utils.mark_ae(prediction[1], train_input, train_target[1],mask[0])#[]
            
        # SE is usually large, scale it to stabilize training
        scale_time_loss = opt.loss_scale
        loss = event_loss + se + place_se + (contrastive_loss *scale_time_loss) #minimize
        loss.backward()
        
        if opt.grad_log==True:
            for name, param in model.named_parameters():
                with open(f"./param_grad/{opt.imp}/{name}.log","a") as f:
                    f.write(f'{param.grad}\n')#[name for name, param in model.named_parameters()]
        """ update parameters """
        optimizer.step()
        """ note keeping """
        with torch.no_grad():
            total_event_ll += float(-event_loss.item())
            total_time_se += float(se.item())
            total_place_se += float(place_se.item())
            total_time_ae += float(ae.item())
            total_place_ae += float(place_ae.item())
            total_num_event += int(mask[0].sum().item())
            total_contrastive += float(scale_time_loss*contrastive_loss.item())
            time_se_history=np.append(time_se_history,Utils.time_se_list(prediction[0],train_target[0],mask[0]))
            place_se_history=np.append(place_se_history,Utils.marks_se_list(prediction[1],train_target[1],mask[0]))
            del loss
            del model_output
            del enc_out
            torch.cuda.empty_cache()
            pred_marks=prediction[1]
            marks_pred_history = np.append(marks_pred_history,pred_marks.cpu())
        
    time_mse = total_time_se / total_num_event
    place_mse = total_place_se / total_num_event
    time_mae = total_time_ae / total_num_event
    place_mae = total_place_ae / total_num_event
    contrastive=total_contrastive/total_num_event
    
    marks_pred_history=marks_pred_history.reshape(-1,2)
    if opt.now_epoch%2 == 0:
        plt.clf()
        plt.figure(figsize=(6, 4))
        plt.xlim(opt.train_x_min, opt.train_x_max)
        plt.ylim(opt.train_y_min, opt.train_y_max)
        plt.xlabel("pred latitude",fontsize=18)
        plt.ylabel("pred longitude",fontsize=18)
        plt.scatter(marks_pred_history[:,0],marks_pred_history[:,1],label="prediction",c=np.sqrt(place_se_history))
        plt.colorbar()
        plt.legend(fontsize=18, loc='upper right')
        dir_file=os.getcwd()
        dir=f"{dir_file}/tmpplot/place_pred_map/{opt.gene}/{opt.rep_vec_num}_{opt.anc_vec_num}/{opt.method}_{opt.imp}/train/"
        if not os.path.exists(dir):# 無ければ
            os.makedirs(dir) 
        plt.savefig(f"{dir}_train_predmarks_{opt.now_epoch}.png", bbox_inches='tight', pad_inches=0)
        
        plt.clf()
    
    return total_event_ll / total_num_event, time_se_history, place_se_history, contrastive

################
### Evaluation
################
def eval_epoch(model, validation_data, opt, plot=None,imp=None,enc_plot=None):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_place_se = 0
    total_time_ae = 0
    total_place_ae = 0
    
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    
    time_se_history=[]
    place_se_history=[]
    marks_pred_history=[]
    total_contrastive=0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,dynamic_ncols=True,
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
                else:
                    train_input_append(tmp_input)
                    train_target_append(tmp_target)
            if opt.do_mask==False:
                for mod_i in np.arange(len(batch)):
                    if mod_i==0:
                        mask=[torch.ones(train_target[mod_i].shape).to(opt.device)]
                        mask_append=mask.append
                    else:
                        mask_append(train_target[mod_i].shape)

            """ forward """
            output, prediction, enc_out,enc_split = model(train_input,train_target,plot,imp=opt.imp,enc_plot=enc_plot)
            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, output, train_input, train_target, enc_out,enc_split, opt)
            
            
            contrastive_loss = Utils.used_contra(model, output, train_input, train_target, prediction,opt,enc_out)

            
            event_loss = -torch.sum((event_ll - non_event_ll)*mask[0].reshape(event_ll.shape))
            # time prediction
            se = Utils.time_loss_se(prediction[0],  train_target[0], mask[0])
            place_se = Utils.mark_se(prediction[1],  train_target[1], mask[0])#[]
        
            ae = Utils.time_loss_ae(prediction[0], train_input, train_target[0], mask[0])#[]
            place_ae = Utils.mark_ae(prediction[1], train_input, train_target[1], mask[0])#[]
            
            """ note keeping """
            total_event_ll += float(-event_loss.item())
            total_time_se += float(se.item())
            total_place_se += float(place_se.item())
            total_time_ae += float(ae.item())
            total_place_ae += float(place_ae.item())
            total_num_event += int(mask[0].sum().item())
            total_contrastive += float(contrastive_loss.item())
            time_se_history=np.append(time_se_history,Utils.time_se_list(prediction[0],train_target[0],mask[0]))
            place_se_history=np.append(place_se_history,Utils.marks_se_list(prediction[1],train_target[1],mask[0]))
            
            pred_marks=prediction[1]
            marks_pred_history = np.append(marks_pred_history,pred_marks.cpu())
    contrastive =total_contrastive/total_num_event
    
    marks_pred_history=marks_pred_history.reshape(-1,2)
    if opt.now_epoch%5 == 0:
        plt.clf()
        plt.figure(figsize=(6, 4))
        plt.xlim(opt.train_x_min, opt.train_x_max)
        plt.ylim(opt.train_y_min, opt.train_y_max)
        plt.xlabel("pred latitude",fontsize=18)
        plt.ylabel("pred longitude",fontsize=18)
        plt.scatter(marks_pred_history[:,0],marks_pred_history[:,1],label="prediction",c=np.sqrt(place_se_history))
        plt.colorbar()
        plt.legend(fontsize=18, loc='upper right')
        dir_file=os.getcwd()
        dir=f"{dir_file}/tmpplot/place_pred_map/{opt.gene}/{opt.rep_vec_num}_{opt.anc_vec_num}/{opt.method}_{opt.imp}/"
        if not os.path.exists(dir):# 無ければ
            os.makedirs(dir) 
        plt.savefig(f"{dir}predmarks_{opt.now_epoch}.png", bbox_inches='tight', pad_inches=0)
        
        plt.clf()
        plt.close()
    
    
    return total_event_ll / total_num_event, time_se_history, place_se_history, contrastive

################
### train-eval-plot-earlystop
################
def train(model, training_data, validation_data ,test_data,optimizer, scheduler, opt):
    """ Start training. """
    train_loss_his = []
    train_loss_eve = []
    train_loss_time_mse = []
    train_loss_place_mse = []
    train_loss_contra = []
    
    valid_loss_his = []
    valid_loss_eve = []  # validation log-likelihood
    valid_loss_time_mse = []  # validation event time prediction MSE
    valid_loss_place_mse = []
    valid_loss_contra = []
    
    if not os.path.exists(f"checkpoint/{opt.gene}"):# 無ければ
        os.makedirs(f"checkpoint/{opt.gene}") 
    if opt.epoch==0:
        torch.save(model.state_dict(), f"checkpoint/{opt.gene}/{opt.wp}.pth") 
        epoch = 0
    if opt.train==True:
        torch.backends.cudnn.benchmark = True
        es = EarlyStopping(verbose=True,path=f"checkpoint/{opt.gene}/{opt.wp}.pth")
        
        for epoch_i in np.arange(opt.epoch):
            
            torch.cuda.empty_cache()
            epoch = epoch_i + 1
            opt.now_epoch=epoch
            print(f'[ Epoch, {epoch}]')
            ################### train
            start = time.time()
            train_event, train_time_se_history, train_place_se_history, train_contrastive = train_epoch(model, training_data, optimizer, opt)

            train_time_mse=train_time_se_history.mean()
            train_place_mse=train_place_se_history.mean()
            print(f'  - (Training)    Loss:{-train_event+train_time_mse+train_place_mse+train_contrastive: 8.3f},'
                f' loglikelihood: {train_event: 8.3f}, '
                f' timeRMSE: {np.sqrt(train_time_mse): 8.3f} ({np.sqrt(np.std(train_time_se_history)): 4.3f}),'
                f' placeRMSE: {np.sqrt(train_place_mse): 8.3f} ({np.sqrt(np.std(train_place_se_history)): 4.3f}),'
                f' contra: {train_contrastive: 8.3f},'
                f' elapse: {((time.time() - start) / 60):3.3f} min')
            train_loss_his += [-train_event+train_time_mse+train_place_mse+train_contrastive]
            train_loss_eve += [-train_event]
            train_loss_time_mse += [train_time_mse]
            train_loss_place_mse += [train_place_mse]
            train_loss_contra += [train_contrastive]
            
            ################### valid ##################
            start = time.time()
            valid_event, valid_time_se_history, valid_place_se_history, valid_contrastive = eval_epoch(model, validation_data, opt)
            
            valid_time_mse=valid_time_se_history.mean()
            valid_place_mse=valid_place_se_history.mean()
            
            print(f'  - (Valid   )    Loss:{-valid_event+ valid_time_mse+valid_place_mse+valid_contrastive: 8.3f},'
                f' loglikelihood: {valid_event: 8.3f}, '
                f' timeRMSE: {np.sqrt(valid_time_mse): 8.3f} ({np.sqrt(np.std(valid_time_se_history)):4.3f}),'
                f' placeRMSE: {np.sqrt(valid_place_mse): 8.3f} ({np.sqrt(np.std(valid_place_se_history)):4.3f}),'
                f' contra: {valid_contrastive: 8.3f},'
                f' elapse: {((time.time() - start) / 60):3.3f} min')
            comp_loss=-valid_event+valid_time_mse+valid_place_mse+valid_contrastive
            
            valid_loss_his +=[comp_loss]
            valid_loss_eve += [valid_event]
            valid_loss_time_mse += [valid_time_mse]
            valid_loss_place_mse += [valid_place_mse]
            valid_loss_contra += [valid_contrastive]
            
            print('  - [Info] Loss: {loss:8.3f}, Maximum ll: {event: 8.3f}, Minimum RMSE:{mse: 8.3f}'
                .format(loss=min(valid_loss_his), event=max(valid_loss_eve), mse=np.sqrt(min(valid_loss_time_mse))))
            
            ############### test
            start = time.time()
            test_event, test_time_se_history, test_place_se_history, test_contrastive = eval_epoch(model, test_data, opt)
            test_time_mse=test_time_se_history.mean()
            test_place_mse=test_place_se_history.mean()
            
            print(f'  - (testing   )    Loss:{-test_event+ test_time_mse+test_place_mse+test_contrastive: 8.3f}, '
                f' loglikelihood: {test_event: 8.3f}, '
                f' RMSE: {np.sqrt(test_time_mse): 8.3f} ({np.sqrt(np.std(test_time_se_history)):4.3f}),'
                f' placeRMSE: {np.sqrt(test_place_mse): 8.3f} ({np.sqrt(np.std(test_place_se_history)):4.3f}),'
                f" contra: {test_contrastive:8.3f}, "
                f' elapse: {(time.time() - start) / 60:3.3f} min')
            # logging
            with open(opt.log, 'a') as f:
                f.write(f" epoch {epoch}\n"
                        f"  - (Train   )    Loss: {-train_event+train_time_mse+train_place_mse+train_contrastive: 8.3f},"
                        f" ll: {train_event: 8.3f},"
                        f" timeRMSE: {np.sqrt(train_time_mse): 8.3f}({np.sqrt(np.std(train_time_se_history)): 4.3f}),"
                        f" placeRMSE: {np.sqrt(train_place_mse): 8.3f}({np.sqrt(np.std(train_place_se_history)): 4.3f}),"
                        f" contra: {train_contrastive:4.3f}\n")
                
                f.write(f'  - (Valid   )    Loss:{-valid_event+ valid_time_mse+valid_place_mse+valid_contrastive: 8.3f}'
                        f' ll: {valid_event: 8.3f}, '
                        f' timeRMSE: {np.sqrt(valid_time_mse): 8.3f} ({np.sqrt(np.std(valid_time_se_history)):4.3f}),'
                        f' placeRMSE: {np.sqrt(valid_place_mse): 8.3f} ({np.sqrt(np.std(valid_place_se_history)):4.3f}),'
                        f" contra: {valid_contrastive:4.3f}\n\n"
                        )

            scheduler.step()
            
            
            es( -valid_event+valid_time_mse+valid_place_mse ,model)
            if es.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
                print("Early Stopping!")
                break

        model_path=f"checkpoint/{opt.gene}/{opt.wp}.pth"
        model.load_state_dict(torch.load(model_path))
        model.eval()
        start = time.time()
        test_event, test_time_se_history,test_place_se_history, test_contrastive = eval_epoch(model, test_data, opt)
        test_time_mse=test_time_se_history.mean()
        test_place_mse=test_place_se_history.mean()
        print(f'  - (testing   )    Loss:{-test_event+ test_time_mse+test_place_mse+test_contrastive: 8.3f},'
                f'loglikelihood: {test_event: 8.3f}, '
                f' timeRMSE: {np.sqrt(test_time_mse): 8.3f} ({np.sqrt(np.std(test_time_se_history)):4.3f}), '
                f' placeRMSE: {np.sqrt(test_place_mse): 8.3f} ({np.sqrt(np.std(test_place_se_history)):4.3f}), '
                f' contra: {(test_contrastive): 8.5f}, '
                f'elapse: {(time.time() - start) / 60:3.3f} min')
            
        with open(opt.log, 'a') as f:
            f.write(f'  - (Test:   )    Loss:{-test_event+ test_time_mse+ test_place_mse+test_contrastive: 8.3f}'
                f',loglikelihood: {test_event: 8.3f}, '
                f' timeRMSE: {np.sqrt(test_time_mse): 8.3f} ({np.sqrt(np.std(test_time_se_history)):4.3f}), '
                f' placeRMSE: {np.sqrt(test_place_mse): 8.3f} ({np.sqrt(np.std(test_place_se_history)):4.3f}), '
                f' contra: {(test_contrastive): 8.5f}, '
                f'\n\n')
        
        with open(opt.log, 'a') as f:
            if opt.vec==True:
                for i in range(len(model.rep_vector)):
                    f.write(f'rep values {model.rep_vector[i]}\n')
                for i in range(len(model.anchor_vector)):
                    f.write(f'anchor values {model.anchor_vector[i]}\n')
        #plot_code.plot_learning_curve(train_loss_his,valid_loss_his,opt)
        
        plot_code.plot_learning_curve(train_loss_his,valid_loss_his,opt,name=f"{opt.phase_num}-loss")
        plot_code.plot_learning_curve(train_loss_eve,valid_loss_eve,opt,name=f"{opt.phase_num}-ll")
        plot_code.plot_learning_curve(train_loss_time_mse,valid_loss_time_mse,opt,name=f"{opt.phase_num}-timemse")
        plot_code.plot_learning_curve(train_loss_place_mse,valid_loss_place_mse,opt,name=f"{opt.phase_num}-placemse")
        plot_code.plot_learning_curve(train_loss_contra,valid_loss_contra,opt,name=f"{opt.phase_num}-contra")
    
def test(model, training_data, validation_data ,test_data,optimizer, scheduler, opt):
    
    path=opt.wp
    opt.now_epoch=0
    opt.wp+="train"
    plot_code.plot_data_hist(training_data, opt)
    opt.wp=path
    
    opt.wp+="test"
    plot_code.plot_data_hist(test_data, opt)
    opt.wp=path
    
    if opt.phase==True:
        model_path=f"checkpoint/{opt.gene}/{opt.wp}phase3.pth"
    else:
        model_path=f"checkpoint/{opt.gene}/{opt.wp}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # with open(f"./param_grad/modelparam2.log", 'w') as f:
    #  for param in model.parameters():
    #   f.write(f"{param}")
    # with open(f"./param_grad/enc_layer2.log", 'w') as f:
    #         for i in model.encoder.layer_stack_modal.state_dict():
    #             f.write(f"\n enc_laer\n{i}")
    #   f.write(f"\n enc_split\i{enc_split}")
    #   f.write(f"\n out\n{output}")

    start = time.time()
    
    test_event, test_time_se_history,test_place_se_history, test_contrastive = eval_epoch(model, test_data, opt)
    test_time_mse=test_time_se_history.mean()
    test_place_mse=test_place_se_history.mean()
    print(f'  - (testing   )    Loss:{-test_event+ test_time_mse+test_place_mse+test_contrastive: 8.3f},loglikelihood: {test_event: 8.3f}, '
                f' timeRMSE: {np.sqrt(test_time_mse): 8.3f} ({np.sqrt(np.std(test_time_se_history)):4.3f}), '
                f' placeRMSE: {np.sqrt(test_place_mse): 8.3f} ({np.sqrt(np.std(test_place_se_history)):4.3f}), '
                f'elapse: {(time.time() - start) / 60:3.3f} min')
        
    
    opt.wp=path
    
    plot_code.phase_eventGT_prediction_plot(model,test_data,opt)
    plot_code.plot_attn(model,test_data,opt)
    #plot_code.plot_con(model,test_data,opt)
    plot_code.plot_phi(model,test_data,opt)
    
    #plot_code.t_SNE(model,test_data,opt)
    #plot_code.phase_eventGT_prediction_plot(model,training_data,opt,train=True)

    
def make_path(opt):
    if opt.batch_size==256:
        opt.log = f"{opt.d_model}_{opt.d_inner_hid}_{opt.d_k}_{opt.d_v}_{opt.n_head}_{opt.gene}_{opt.method}_{opt.imp}_{opt.epoch}_{opt.time_step}"
    else:
        opt.log = f"{opt.batch_size}_{opt.d_model}_{opt.d_inner_hid}_{opt.d_k}_{opt.d_v}_{opt.n_head}_{opt.gene}_{opt.method}_{opt.imp}_{opt.epoch}_{opt.time_step}"
    
    # imaginary vec
    if opt.vec==True:
        opt.log+=f"_{opt.rep_vec_num}_{opt.anc_vec_num}"
    else:
        opt.rep_vec_num=0
        opt.anc_vec_num=0
        opt.log+=f"_novec"
    # bottle or allcat or repfusion 
    if opt.btl==True:
        opt.log+=f"_btllayer{opt.start_fusion_layers}"
    else:
        opt.log+=f"_nofusion"

    # phase
    if opt.phase==True:
        opt.log+="_phase"

    # layer norm
    if opt.pre_attn == True:
        opt.log+="_preLN"
    else:
        opt.log+="_postLN"
    if opt.nomovevec == True:
        opt.log+="_nomodmove"
    opt.wp = opt.log
    opt.log="log/"+opt.log+"_log.txt"
#################
### Main
#################

def main():
    """ Main function. """
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=256)#32
    parser.add_argument('-loss_scale',type=int,default=1)
    
    parser.add_argument('-d_model', type=int, default=64)#512
    parser.add_argument('-d_inner_hid', type=int, default=64)#1024
    parser.add_argument('-d_k', type=int, default=8)#512
    parser.add_argument('-d_v', type=int, default=8)#512

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-start_fusion_layers', type=int, default=1)
    parser.add_argument('-linear_num', type=int, default=3)
    parser.add_argument('-now_phase', type=int, default=1)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)#1e-5
    #parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-gene', type=str, default='h1')
    parser.add_argument('-mod_sample', default=["time","place"])
    parser.add_argument('-log', type=str, default='log/log.txt')

    
    parser.add_argument('-imp', type=str, default='_')
    
    # set_data　を介して更新
    parser.add_argument("-n_dataset", type=float, default=0)# dataset の data分解数 2 [time,marks,...] 2以外はいらない
    parser.add_argument("-n_marks", type=float, default=0)#[x,y,...]2でしょ
    parser.add_argument("-train_time_mean", type=float, default=0)
    parser.add_argument("-train_time_max", type=float, default=0)
    parser.add_argument("-train_time_min", type=float, default=0)
    parser.add_argument("-train_time_med", type=float, default=0)
    parser.add_argument("-train_time_std", type=float, default=0)
    parser.add_argument("-train_x_max", type=float, default=0)
    parser.add_argument("-train_x_min", type=float, default=0)
    parser.add_argument("-train_y_max", type=float, default=0)
    parser.add_argument("-train_y_min", type=float, default=0)
    
    #parser.add_argument("-train_cov_i", type=float, default=0)
    
    parser.add_argument("-test_time_mean", type=float, default=0)

    parser.add_argument("-time_step", type=int, default=30)
    parser.add_argument("-rep_vec_num",type=int, default=3)
    parser.add_argument("-anc_vec_num",type=int, default=3)
    
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--vec",action="store_true")
    parser.add_argument("--btl",action="store_true")
    parser.add_argument("--nomovevec",action="store_true")
    parser.add_argument("--do_mask",action="store_true")
    parser.add_argument("--pickle_F",action='store_true')
    parser.add_argument("--pre_attn",action='store_true')
    parser.add_argument("--miman",action='store_true')
    parser.add_argument("--phase",action="store_true")
    parser.add_argument("--sped",action="store_true")
    parser.add_argument("--grad_log",action="store_true")
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--enc_plot",action="store_true")
    parser.add_argument("--allcat",action="store_true")
    parser.add_argument("--modmove",action="store_true")
    
    parser.add_argument("-method",type=str, default="rep_quan")
    
    parser.add_argument("-wp",type=str, default="_")
    parser.add_argument("-device_num",type=int, default=0)
    
    opt = parser.parse_args()
    
    # default device is CUDA
    opt.device = torch.device('cuda:'+str(opt.device_num))
    make_path(opt)
    
    if opt.sped==True:
        
        torch.autograd.detect_anomaly=False
        torch.autograd.set_detect_anomaly(False)
    
        torch.autograd.profiler.profile=False

        torch.autograd.profiler.emit_nvtx=False

        torch.autograd.gradcheck=False
        torch.autograd.gradgradcheck=False
    #torch.autograd.set_detect_anomaly(True)

    
    print('[Info] parameters: {}'.format(opt))
    pickle_Flag=False
    Call_num=0
    Call_type=""
    train_mask_path=None
    """ prepare dataloader """
    trainloader, validloader, testloader = set_data.generate_loader(opt)
    if not os.path.exists(f"checkpoint/{opt.gene}"):# 無ければ
        os.makedirs(f"checkpoint/{opt.gene}") 
    if not os.path.exists(f"log"):
        os.makedirs(f"log") 
    # setup the log file
    if opt.train==True:
        with open(opt.log, 'w') as f:
            f.write(f'[Info] parameters: {opt}\n')
            f.write('Loss,  Epoch,  Log-likelihood,  MAE,  MSE\n')
    else:
        print("Not training")
    """ prepare model """
    model = Transformer(
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_dataset=opt.n_dataset,
        n_marks=opt.n_marks,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        rep_vec_num=opt.rep_vec_num,
        anc_vec_num=opt.anc_vec_num,
        time_step=opt.time_step,
        device=opt.device,
        method=opt.method,
        train_max=opt.train_time_max,
        train_min=opt.train_time_min,
        train_med=opt.train_time_med,
        linear_layers=opt.linear_num,
        normalize_before=opt.pre_attn,
        mod_sample=opt.mod_sample,
        is_bottle_neck=opt.btl,
        train=trainloader,
        notMoveVec=opt.nomovevec,
        opt=opt
    )
    model.to(opt.device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))
    if opt.train == True:
        if opt.vec ==True:
            with open(opt.log, 'a') as f:
                f.write(f"all_parameter_num {num_params}\n")
                f.write(f'anchor values {model.anchor_vector[0]}\n')
                f.write(f'rep values {model.rep_vector[0]}\n')
        if opt.grad_log==True:
            os.makedirs(f'./param_grad/{opt.imp}', exist_ok = True)
            for name, param in model.named_parameters():#初期open
                with open(f"./param_grad/{opt.imp}/{name}.log", 'w') as f:
                    f.write('st')
    
    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    """ number of parameters """
    
    
    """ train the model """
    if opt.train==True:
        opt.phase_num="0"
        train(model, trainloader, validloader, testloader, optimizer, scheduler, opt)
    else:
        
        test(model, trainloader, validloader, testloader, optimizer, scheduler, opt)
if __name__ == '__main__':
    plt.switch_backend('agg')
    plt.figure()
    main()