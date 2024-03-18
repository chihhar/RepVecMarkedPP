import numpy as np
import torch
import torch.utils.data
import pdb
import pandas as pd
from tqdm import tqdm
from dataloadFolder import generate_pinwheel as genePIN
def set_batch_sequence_loader(mod_data,opt,df=None):
    # mod_data:データによって入ってくるデータ列は変化　generate_loader参照
    # 例 地震
    #  mod_data[0]
    # 0            20.963217
    # 1            50.866144
    #             ...      
    # 18469    456376.886206
    # Name: DateTime, Length: 18470, dtype: float64
    # 
    # mod_data[1]
    # array([-121.38533, -122.07117, -122.07333, ..., -118.344  , -125.35767,-121.27016])
    # 
    # mod_data[2]
    # array([36.77833, 37.319  , 37.3165 , ..., 38.26383, 40.40567, 36.66516])
    
    def rolling_matrix(x,time_step):
        if len(x.shape)==1:
            x = x.flatten()
            n = x.shape[0]
            
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()
        elif len(x.shape)>1:
            n_marks=len(x.shape)
            n = x.shape[0]
            in_step=time_step*n_marks
            stride = x.strides[0] # x=8byte,y=8byte =16byteで次の行
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step,n_marks), strides=(stride,stride,int(stride/n_marks)) ).copy()
                   
    data_n=mod_data[0].shape[0]
    n_marks=len(mod_data)-1
    opt.n_marks=n_marks
    opt.n_dataset=len(opt.mod_sample)
    n_train=int(data_n*0.8)
    n_validation=int(data_n*0.1)
    n_test=data_n-(n_train+n_validation)
    time_step=opt.time_step

    for mod_ind in range(len(opt.mod_sample)):# time and marksの2種
        if opt.mod_sample[mod_ind]=="time":
            # event_data [B,]
            event_data=mod_data[0]
            train_event=event_data[:n_train]
            valid_event=event_data[n_train:n_train+n_validation]
            test_event=event_data[n_train+n_validation:]
        elif opt.mod_sample[mod_ind]=="marks":
            # event_data [B,2]
            event_data = np.stack(mod_data[1:],axis=1)
            train_event=event_data[:n_train]
            valid_event=event_data[n_train:n_train+n_validation]
            test_event=event_data[n_train+n_validation:]
        else:
            print("anything wrong")
            exit(1)
        
        if opt.mod_sample[mod_ind] =="time":
            # seq_eps_train[B_size,L,1]
            elapsed_train=np.ediff1d(train_event)
            elapsed_valid=np.ediff1d(valid_event)
            elapsed_test=np.ediff1d(test_event)
            
            seq_eps_train=torch.tensor(rolling_matrix(elapsed_train,time_step)).to(torch.double).unsqueeze(-1)
            seq_eps_valid=torch.tensor(rolling_matrix(elapsed_valid,time_step)).to(torch.double).unsqueeze(-1)
            seq_eps_test=torch.tensor(rolling_matrix(elapsed_test,time_step)).to(torch.double).unsqueeze(-1)
            
            opt.train_time_mean=elapsed_train.mean()
            opt.train_time_max=elapsed_train.max()
            opt.train_time_min=elapsed_train.min()
            opt.train_time_med=np.median(elapsed_train)
            opt.train_time_std=np.std(elapsed_train)
            mod_train=[seq_eps_train]
            mod_valid=[seq_eps_valid]
            mod_test=[seq_eps_test]
        elif opt.mod_sample[mod_ind]=="marks":
            # seq_eps_train[B_size,L,n_marks]
            future_train_event=np.array(train_event[1:])
            future_valid_event=np.array(valid_event[1:])
            future_test_event=np.array(test_event[1:])
            
            seq_ftr_train=torch.tensor(rolling_matrix(future_train_event,time_step)).to(torch.double)
            seq_ftr_valid=torch.tensor(rolling_matrix(future_valid_event,time_step)).to(torch.double)
            seq_ftr_test=torch.tensor(rolling_matrix(future_test_event,time_step)).to(torch.double)
            
            mod_train.append(seq_ftr_train)
            mod_valid.append(seq_ftr_valid)
            mod_test.append(seq_ftr_test)
    if opt.gene=="911_1_Address" or opt.gene=="911_2_Address" or opt.gene=="911_3_Address":
        train_df=df[:n_train]["mask"].values
        future_train_df=np.array(train_df[1:])
        seq_train_df=torch.tensor(rolling_matrix(future_train_df,time_step)).to(torch.double)
        mod_train[0]=mod_train[0][(seq_train_df[:,-1]).bool().numpy()]
        mod_train[1]=mod_train[1][(seq_train_df[:,-1]).bool().numpy()]
        
        valid_df=df[n_train:n_train+n_validation]["mask"].values
        future_valid_df=np.array(valid_df[1:])
        seq_valid_df=torch.tensor(rolling_matrix(future_valid_df,time_step)).to(torch.double)
        mod_valid[0]=mod_valid[0][(seq_valid_df[:,-1]).bool().numpy()]
        mod_valid[1]=mod_valid[1][(seq_valid_df[:,-1]).bool().numpy()]
        
        
        test_df=df[n_train+n_validation:]["mask"].values
        future_test_df=np.array(test_df[1:])
        seq_test_df=torch.tensor(rolling_matrix(future_test_df,time_step)).to(torch.double)
        mod_test[0]=mod_test[0][(seq_test_df[:,-1]).bool().numpy()]
        mod_test[1]=mod_test[1][(seq_test_df[:,-1]).bool().numpy()]
        
    trainloader = torch.utils.data.DataLoader(list(zip(*mod_train)),num_workers=2,batch_size=opt.batch_size,pin_memory=True,shuffle=True)
    validloader = torch.utils.data.DataLoader(list(zip(*mod_valid)),num_workers=2,batch_size=opt.batch_size,pin_memory=True,shuffle=False)
    testloader = torch.utils.data.DataLoader(list(zip(*mod_test)),num_workers=2,batch_size=opt.batch_size,pin_memory=True,shuffle=False)
    
    return trainloader, validloader, testloader

#def set_loader_train_velid_test(train_data,test_data,valid_data,batch_size):
    
def generate_loader(opt):
    
    if opt.gene=="jisin":
        # event_time        :[DateTime]
        # event_magnitude   :[Magnitude]
        df = pd.read_csv("data/date_jisin.90016")
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df["DateTime"] = df["DateTime"].map(pd.Timestamp.timestamp)/3600##UNIX変換
        
        opt.mod_sample=["time","marks"]
        
        opt.train_x_max=max(df["Longitude"].values)
        opt.train_x_min=min(df["Longitude"].values)
        opt.train_y_max=max(df["Latitude"].values)
        opt.train_y_min=min(df["Latitude"].values)
        
        train_loader, valid_loader, test_loader = set_batch_sequence_loader([df["DateTime"],df["Longitude"].values,df["Latitude"].values],opt)
    elif opt.gene=="911_1_Address":
        opt.mod_sample=["time","marks"]
        df = pd.read_csv('data/kaggle/police-department-incidents.csv',                           dtype={"IncidntNum":str})
        df2=df[["Date","Time","Address","X","Y"]]
        df3=df2.assign(counts=0).groupby(["Date","Time","Address","X","Y"]).count().reset_index()
        df3["Date"] = pd.to_datetime(df3.Date) 
        df3["DateTime"] = pd.to_datetime(df3.Date.dt.date.astype(str)+ " " + df3.Time)
        df3.drop("Time", axis=1, inplace=True)
        df3.drop("Date", axis=1, inplace=True)
        df3.drop("counts", axis=1, inplace=True)
        df4 = df3.groupby('Address')['Address'].count().reset_index(name='counts')
        df5 =  df4.sort_values('counts',ascending=False)[0:1]
        address_list=df5["Address"].tolist()
        df6 = df3[df3["Address"].isin(address_list)]
        df6 = df6.sort_values("DateTime")
        df6["Unix"]=df6["DateTime"].map(pd.Timestamp.timestamp)/3600##UNIX変換 単位[h]
        df6["mask"]=((df6.DateTime.dt.hour>=15)+(df6.DateTime.dt.hour<1)).astype(int)
        df7=df6[df6["mask"]==1]
        
        opt.train_x_max=max(df["X"].values)
        opt.train_x_min=min(df["X"].values)
        opt.train_y_max=max(df["Y"].values)
        opt.train_y_min=min(df["Y"].values)
        train_loader, valid_loader, test_loader = set_batch_sequence_loader([df6["Unix"],df6["X"].values,df6["Y"].values],opt,df=df6)
    elif opt.gene=="911_2_Address":
        opt.mod_sample=["time","marks"]
        df = pd.read_csv('data/kaggle/police-department-incidents.csv',                           dtype={"IncidntNum":str})
        df2=df[["Date","Time","Address","X","Y"]]
        df3=df2.assign(counts=0).groupby(["Date","Time","Address","X","Y"]).count().reset_index()
        df3["Date"] = pd.to_datetime(df3.Date) 
        df3["DateTime"] = pd.to_datetime(df3.Date.dt.date.astype(str)+ " " + df3.Time)
        df3.drop("Time", axis=1, inplace=True)
        df3.drop("Date", axis=1, inplace=True)
        df3.drop("counts", axis=1, inplace=True)
        df4 = df3.groupby('Address')['Address'].count().reset_index(name='counts')
        df5 =  df4.sort_values('counts',ascending=False)[1:2]
        address_list=df5["Address"].tolist()
        df6 = df3[df3["Address"].isin(address_list)]
        df6 = df6.sort_values("DateTime")
        df6["Unix"]=df6["DateTime"].map(pd.Timestamp.timestamp)/3600##UNIX変換 単位[h]
        df6["mask"]=((df6.DateTime.dt.hour>=15)+(df6.DateTime.dt.hour<1)).astype(int)
        df7=df6[df6["mask"]==1]
        
        opt.train_x_max=max(df["X"].values)
        opt.train_x_min=min(df["X"].values)
        opt.train_y_max=max(df["Y"].values)
        opt.train_y_min=min(df["Y"].values)
        train_loader, valid_loader, test_loader = set_batch_sequence_loader([df6["Unix"],df6["X"].values,df6["Y"].values],opt,df=df6)
    elif opt.gene=="911_3_Address":
        opt.mod_sample=["time","marks"]
        df = pd.read_csv('data/kaggle/police-department-incidents.csv',                           dtype={"IncidntNum":str})
        df2=df[["Date","Time","Address","X","Y"]]
        df3=df2.assign(counts=0).groupby(["Date","Time","Address","X","Y"]).count().reset_index()
        df3["Date"] = pd.to_datetime(df3.Date) 
        df3["DateTime"] = pd.to_datetime(df3.Date.dt.date.astype(str)+ " " + df3.Time)
        df3.drop("Time", axis=1, inplace=True)
        df3.drop("Date", axis=1, inplace=True)
        df3.drop("counts", axis=1, inplace=True)
        df4 = df3.groupby('Address')['Address'].count().reset_index(name='counts')
        df5 =  df4.sort_values('counts',ascending=False)[2:3]
        address_list=df5["Address"].tolist()
        df6 = df3[df3["Address"].isin(address_list)]
        df6 = df6.sort_values("DateTime")
        df6["Unix"]=df6["DateTime"].map(pd.Timestamp.timestamp)/3600##UNIX変換 単位[h]
        df6["mask"]=((df6.DateTime.dt.hour>=15)+(df6.DateTime.dt.hour<1)).astype(int)
        df7=df6[df6["mask"]==1]
        
        opt.train_x_max=max(df["X"].values)
        opt.train_x_min=min(df["X"].values)
        opt.train_y_max=max(df["Y"].values)
        opt.train_y_min=min(df["Y"].values)
        train_loader, valid_loader, test_loader = set_batch_sequence_loader([df6["Unix"],df6["X"].values,df6["Y"].values],opt,df=df6)
    elif opt.gene=="911_1_3":
        opt.mod_sample=["time","marks"]
        df = pd.read_csv('/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/kaggle/police-department-incidents.csv',                           dtype={"IncidntNum":str})
        df2=df[["Date","Time","Address","X","Y"]]
        df3=df2.assign(counts=0).groupby(["Date","Time","Address","X","Y"]).count().reset_index()
        df3["Date"] = pd.to_datetime(df3.Date) 
        df3["DateTime"] = pd.to_datetime(df3.Date.dt.date.astype(str)+ " " + df3.Time)
        df3.drop("Time", axis=1, inplace=True)
        df3.drop("Date", axis=1, inplace=True)
        df3.drop("counts", axis=1, inplace=True)
        df4 = df3.groupby('Address')['Address'].count().reset_index(name='counts')
        df5 =  df4.sort_values('counts',ascending=False)[0:3]
        address_list=df5["Address"].tolist()
        df6 = df3[df3["Address"].isin(address_list)]
        df6 = df6.sort_values("DateTime")
        df6["Unix"]=df6["DateTime"].map(pd.Timestamp.timestamp)/3600##UNIX変換 単位[h]
        df6["mask"]=((df6.DateTime.dt.hour>=15)+(df6.DateTime.dt.hour<1)).astype(int)
        df7=df6[df6["mask"]==1]
        
        opt.train_x_max=max(df["X"].values)
        opt.train_x_min=min(df["X"].values)
        opt.train_y_max=max(df["Y"].values)
        opt.train_y_min=min(df["Y"].values)
        train_loader, valid_loader, test_loader = set_batch_sequence_loader([df6["Unix"],df6["X"].values,df6["Y"].values],opt,df=df6)
    
    elif opt.gene=="pin":
        opt.mod_sample=["time","marks"]
        seq=genePIN.pin_wheel_generate()
        opt.train_x_max=max(seq[:,1])
        opt.train_x_min=min(seq[:,1])
        opt.train_y_max=max(seq[:,2])
        opt.train_y_min=min(seq[:,2])
        train_loader, valid_loader, test_loader = set_batch_sequence_loader([seq[:,0],seq[:,1],seq[:,2]],opt)
    elif opt.gene=="pin2":
        opt.mod_sample=["time","marks"]
        seq=genePIN.pin_wheel_generate2()
        opt.train_x_max=max(1000*seq[:,1])
        opt.train_x_min=min(1000*seq[:,1])
        opt.train_y_max=max(1000*seq[:,2])
        opt.train_y_min=min(1000*seq[:,2])
        train_loader, valid_loader, test_loader = set_batch_sequence_loader([seq[:,0],1000*seq[:,1],1000*seq[:,2]],opt)
    elif opt.gene=="pin3":
        opt.mod_sample=["time","marks"]
        seq=genePIN.pin_wheel_generate3()
        opt.train_x_max=max(seq[:,1])
        opt.train_x_min=min(seq[:,1])
        opt.train_y_max=max(seq[:,2])
        opt.train_y_min=min(seq[:,2])
        train_loader, valid_loader, test_loader = set_batch_sequence_loader([seq[:,0],seq[:,1],seq[:,2]],opt)
    
    elif opt.gene=="pin4":
        opt.mod_sample=["time","marks"]
        seq=genePIN.pin_wheel_generate3()
        opt.train_x_max=max(seq[:,1])
        opt.train_x_min=min(seq[:,1])
        opt.train_y_max=max(seq[:,2])
        opt.train_y_min=min(seq[:,2])
        train_loader, valid_loader, test_loader = set_batch_sequence_loader([seq[:,0],seq[:,1],seq[:,2]],opt)
    return train_loader, valid_loader, test_loader
def event_count(data_loader):
    all_event_num=0
    for batch in tqdm(data_loader, mininterval=2,dynamic_ncols=True, leave=False):        
        mask=None
        if len(batch)==2:
            mask=batch[1]
            all_event_num+=mask[:,-1:].sum().item()
        else:
            all_event_num+=batch[0].shape[0]
        
    return all_event_num

def mask_load(data_loader):
    mask_list=[]
    for batch in tqdm(data_loader, mininterval=2,dynamic_ncols=True, leave=False):        
        mask=None
        np.append(mask_list,mask)
        if len(batch)==2:
            mask=batch[1]
        else:
            mask=np.ones(batch[0][:,-1:].shape)
        mask_list=np.append(mask_list,mask[:,-1])
    return mask_list

