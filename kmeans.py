import numpy as np
import itertools 
import pdb
import tqdm
from sklearn.cluster import KMeans
from tqdm import tqdm
def Set_data_kmeans( input, n_clusters,mod_ind,opt):
    
    #input [[~B,L],[1B~2B,L],[~N,L]] :input_time
    #place_input      [[~B,x,y],
    event_his_list=[batch[mod_ind].cpu() for batch in input]
    event_his=np.concatenate(event_his_list)
    target_event_his=event_his[:,-1,:]    
    model = KMeans(n_clusters,random_state=42)
    model.fit(target_event_his)
    
    return model.cluster_centers_# [n_clusters,1] or [n_clusters,2]
def Set_data_quatile( input, n_clusters,mod_ind,opt=None):
    #input [[~B,L],[1B~2B,L],[~N,L]] :input_time
    #      [[~B,x,y],
    event_his_list=[batch[mod_ind].cpu() for batch in input]
    event_his=np.concatenate(event_his_list)
    target_data_sort=np.sort(event_his[:,-1,0])
    
    data_len=len(target_data_sort)
    print(f"data_len of {opt.mod_sample[mod_ind]}:{data_len}")    
    opt.rep_list=target_data_sort[(np.array(data_len*np.array([i/(n_clusters*2) for i in range(n_clusters*2)][2::2]))).astype(int)]
    rep=target_data_sort[(np.array(data_len*np.array([i/(n_clusters*2) for i in range(n_clusters*2)][1::2]))).astype(int)].reshape(n_clusters,1)
    return rep
def choice_initial_means(input, n_clusters,mod_ind,opt=None):
    if opt.mod_sample[mod_ind]=="time":
        return Set_data_quatile(input, n_clusters,mod_ind,opt=opt)
    elif opt.mod_sample[mod_ind]=="marks":
        return Set_data_kmeans(input, n_clusters,mod_ind,opt=opt)