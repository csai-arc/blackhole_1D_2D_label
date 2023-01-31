from TIRE_image import DenseTIRE as TIRE
import torch
import numpy as np
import matplotlib.pyplot as plt
from TIRE_image import utils
from scipy import integrate
from scipy.fft import fft
from scipy.stats import entropy
from scipy.stats import variation
import pandas as pd
import csv
import sys
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim = 1

model = TIRE(dim,window_size=10,intermediate_dim_TD=10,intermediate_dim_FD=10).to(device)

checkpoint = torch.load('./weights/checkpoint_encode_ws10.pth.tar')
model.load_state_dict(checkpoint)
size=128
dpi=40


grs_txt_list=['sac_ascf_alpha','sac_ascf_beta','sac_ascf_delta','sac_ascf_gamma','sac_ascf_kai','sac_ascf_kappa','sac_ascf_lambda','sac_ascf_mu','sac_ascf_nu','sac_ascf_phi','sac_ascf_rho','sac_ascf_theta']
path='/mnt/sdb/code_repo/blackhole_ijcnn/data/GRS1915/'
out_path='/mnt/sdb/code_repo/blackhole_ijcnn/data/GRS1915_images/'


for idx in range(len(grs_txt_list)):

    input_name=grs_txt_list[idx]
    txtfilename1=path+input_name
    ts1=np.loadtxt(txtfilename1)

    minimum=(min(ts1))
    ts1=ts1-minimum
    maximum=(max(ts1))
    ts1=ts1/maximum
    
    #plt.figure(figsize=(20, 5),dpi=dpi)
    #plt.plot(ts1)
    #plt.savefig(out_path+input_name.split('.txt')[0]+'_ts'+'.jpg')
    #plt.close()

    shared_features_TD, shared_features_FD = model.predict(ts1)
    print(shared_features_TD.shape,shared_features_FD.shape)
    plt.figure(figsize=(size, size),dpi=dpi)
    plt.scatter(shared_features_TD, shared_features_FD, c ="black")
    plt.savefig(out_path+input_name.split('.txt')[0]+'.jpg')
    plt.close()




    
path='/mnt/sdb/code_repo/blackhole_ijcnn/codes/data_generator/'
out_path='/mnt/sdb/code_repo/blackhole_ijcnn/data/Synthetic_data_images/'
ns_path='Non-Stochastic/'
s_path='Stochastic/'
import os
print(len(os.listdir(path+ns_path)))
ns_txt_list=os.listdir(path+ns_path)
print(len(os.listdir(path+s_path)))
s_txt_list=os.listdir(path+s_path)


for idx in range(len(ns_txt_list)):

    input_name=ns_txt_list[idx]
    txtfilename1=path+ns_path+input_name
    ts1=np.loadtxt(txtfilename1)

    minimum=(min(ts1))
    ts1=ts1-minimum
    maximum=(max(ts1))
    ts1=ts1/maximum
    
    #plt.figure(figsize=(20, 5),dpi=dpi)
    #plt.plot(ts1)
    #plt.savefig(out_path+ns_path+input_name.split('.txt')[0]+'_ts'+'.jpg')
    #plt.close()

    shared_features_TD, shared_features_FD = model.predict(ts1)
    print(shared_features_TD.shape,shared_features_FD.shape)
    plt.figure(figsize=(size, size), dpi=dpi)
    plt.scatter(shared_features_TD, shared_features_FD, c ="black")
    plt.savefig(out_path+ns_path+input_name.split('.txt')[0]+'.jpg')
    plt.close()

for idx in range(len(s_txt_list)):

    input_name=s_txt_list[idx]
    txtfilename1=path+s_path+input_name
    ts1=np.loadtxt(txtfilename1)

    minimum=(min(ts1))
    ts1=ts1-minimum
    maximum=(max(ts1))
    ts1=ts1/maximum
    
    #plt.figure(figsize=(20, 5),dpi=dpi)
    #plt.plot(ts1)
    #plt.savefig(out_path+s_path+input_name.split('.txt')[0]+'_ts'+'.jpg')
    #plt.close()

    shared_features_TD, shared_features_FD = model.predict(ts1)
    #print(shared_features_TD.shape,shared_features_FD.shape)
    plt.figure(figsize=(size, size), dpi=dpi)
    plt.scatter(shared_features_TD, shared_features_FD, c ="black")
    plt.savefig(out_path+s_path+input_name.split('.txt')[0]+'.jpg')
    plt.close()


























