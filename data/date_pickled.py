import numpy as np
import pandas as pd
import pickle
import pdb
import os
from matplotlib import pyplot as plt
def event_count_function(event_window):
    window_num=event_window.shape[0]#スライディングウィンドウの数
    window_len=event_window.shape[1]#履歴＋予測＝30想定
    event_count=0
    first_count=True
    for win_ind in range(window_num):
        now_wind=event_window[win_ind]
        if now_wind[-1]==1:
            event_count+=window_len
            if first_count:
                first_count=False
                
            else:
                max_ind=np.where(now_wind[:-1]%2==1)[0][-1]
                event_count-=(max_ind+1)
    return event_count
time_step=30
#スライディングウィンドウ
def rolling_matrix(x,time_step):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()


df = pd.read_csv('/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/kaggle/police-department-incidents.csv', usecols=lambda x: x not in ["Location"],
                           dtype={"IncidntNum":str})
#df.groupby(["IncidntNum","Date"])["Date"].count()[df.groupby(["IncidntNum","Date"])["Date"].count()>4]
# "Address" "Date" "location"
#df.groupby(["IncidntNum", "Date"])
df2=df[["Date","Time","location","Address"]]

df3=df2.assign(counts=0).groupby(["Date","Time","location","Address"]).count().reset_index()
Call_log='/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/log/'
if not os.path.exists(Call_log):# 無ければ
        os.makedirs(Call_log) 
# with open(f'{Call_log}911_Address.log', 'w') as f:
#             f.write('911_Address\n')

df3["Date"] = pd.to_datetime(df3.Date) 
df3["DateTime"] = pd.to_datetime(df3.Date.dt.date.astype(str)+ " " + df3.Time)
df3.drop("Time", axis=1, inplace=True)
df3.drop("Date", axis=1, inplace=True)
df3.drop("counts", axis=1, inplace=True)
df4 = df3.groupby('Address')['Address'].count().reset_index(name='counts')

#上位100種の住所
df5 = df4.sort_values('counts',ascending=False)[:100]
address_list=df5["Address"].tolist()
len_address = len(address_list)
#上位100種の住所で発生した記録
df6 = df3[df3["Address"].isin(address_list)]
#df5 = df4.sort_values("Call Date Time")
#df5["date_time64"] = pd.to_datetime(df5["Call Date Time"])
#df5["hour_date_time"] = df5["date_time64"].map(pd.Timestamp.timestamp)/3600##UNIX変換 単位[h]
df6 = df6.sort_values("DateTime")

df6["Unix"]=df6["DateTime"].map(pd.Timestamp.timestamp)/3600##UNIX変換 単位[h]
#pdb.set_trace()
df6["mask"]=((df6.DateTime.dt.hour>=15)+(df6.DateTime.dt.hour<1)).astype(int)
# with open(f'{Call_log}911_Address.log', 'a') as f:
#     f.write("911-All -Address\n")
#     f.write(f"DateSize:{len(df6)}  //mask:{df6['mask'].sum()}\n\n")
df7 = df6.groupby("Address")

separate_sequences = [df7.get_group(address_list[i]) for i in range(len_address)]
plot_hist_list=[1,2,3]
for i in range(len(separate_sequences)):
    if i+1 in plot_hist_list:
        plt.clf()
        plt.hist(separate_sequences[i].DateTime.dt.hour,bins=24)

        plt.title(f"Address{i+1}")
        plt.savefig(f"hour_hist_{i+1}_Address")

train_list=[]
test_list=[]
valid_list=[]
train_mask_list=[]
valid_mask_list=[]
test_mask_list=[]
pdb.set_trace()
# separate_sequences[0]["diff"]=np.ediff1d(seqtime,to_begin=seqtime[0])
# tsts=separate_sequences[0][35390:]
for address_num in range(len(separate_sequences)):
    now_separate_sequence=separate_sequences[address_num]
    len_sequences = len(separate_sequences[address_num])
    
    df_train = now_separate_sequence[:int(len_sequences*0.8)]
    df_valid = now_separate_sequence[int(len_sequences*0.8):int(len_sequences*0.9)]
    df_test = now_separate_sequence[int(len_sequences*0.9):]
    
    dT_train=np.ediff1d(df_train["Unix"])
    train_list+=[rolling_matrix(dT_train,time_step)]

    dT_valid=np.ediff1d(df_valid["Unix"])
    valid_list+=[rolling_matrix(dT_valid,time_step)]

    dT_test=np.ediff1d(df_test["Unix"])
    test_list+=[rolling_matrix(dT_test,time_step)]
    
    train_mask_list+=[rolling_matrix(np.array(df_train["mask"][1:]),time_step)]
    valid_mask_list+=[rolling_matrix(np.array(df_valid["mask"][1:]),time_step)]
    test_mask_list+=[rolling_matrix(np.array(df_test["mask"][1:]),time_step)]
    used_train=event_count_function(train_mask_list[address_num])
    used_val=event_count_function(valid_mask_list[address_num])
    used_tst=event_count_function(test_mask_list[address_num])

    # with open(f"{Call_log}911_Address.log", 'a') as f:
    #     f.write(f"911-{address_num+1} -Address\n")

    #     f.write(f"Train:{len(dT_train)}  //used_event:{used_train}//win:{(train_mask_list[address_num][:,-1]).sum()-(train_mask_list[address_num][0,-1:].sum())} //\n")
    #     f.write(f"Valid:{len(dT_valid)}  //used_event:{used_val}//win:{(valid_mask_list[address_num][:,-1]).sum()-(valid_mask_list[address_num][0,-1:].sum())} //\n")
    #     f.write(f"Test :{len(dT_test)}   //used_event:{used_tst}//win:{(test_mask_list[address_num][:,-1]).sum()-(test_mask_list[address_num][0,-1:].sum())} //\n")

    #     f.write(f"Train_mean:{train_list[address_num][:,-1:].mean()}  //train_std:{train_list[address_num][:,-1:].std()} //\n")
    #     f.write(f"Valid_mean:{valid_list[address_num][:,-1:].mean()}  //valid_std:{valid_list[address_num][:,-1:].std()} //\n")
    #     f.write(f"Test_mean :{test_list[address_num][:,-1:].mean()}   //test_std:{test_list[address_num][:,-1:].std()} //\n")

    #     f.write(f"Sum  :{used_train+used_val+used_tst}  //mask:{((train_mask_list[address_num][:,-1]).sum()+(valid_mask_list[address_num][:,-1]).sum()+(test_mask_list[address_num][:,-1]).sum())}\n\n")
pkl_list_data=[1, 2, 3]
pkl_list_mask=[1, 2, 3]
# for rank in range(len(separate_sequences)):
#     if rank+1 in pkl_list_data:
#         with open(f'Address/Call_{rank+1}_freq_Address_sliding_train.pkl', mode='wb') as f:
#             pickle.dump(train_list[rank], f)
#         with open(f'Address/Call_{rank+1}_freq_Address_sliding_valid.pkl', mode='wb') as f:
#             pickle.dump(valid_list[rank], f)
#         with open(f'Address/Call_{rank+1}_freq_Address_sliding_test.pkl', mode='wb') as f:
#             pickle.dump(test_list[rank], f)
#     if rank+1 in pkl_list_mask:
#         with open(f'Address/Call_{rank+1}_freq_Address_sliding_train_mask.pkl', mode='wb') as f:
#             pickle.dump(train_mask_list[rank], f)
#         with open(f'Address/Call_{rank+1}_freq_Address_sliding_valid_mask.pkl', mode='wb') as f:
#             pickle.dump(valid_mask_list[rank], f)
#         with open(f'Address/Call_{rank+1}_freq_Address_sliding_test_mask.pkl', mode='wb') as f:
#             pickle.dump(test_mask_list[rank], f) 
