# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:38:06 2020
result summary and plot
spectral fitting sensitivity study 
@author: junwan
"""

import numpy as np
import scipy.io as spio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal
from   scipy.optimize import curve_fit
from   scipy.optimize import least_squares
import re
import pandas as pd
from   datetime import datetime 
from   datetime import timedelta

from scipy.stats import binned_statistic
from scipy import stats
import matplotlib.dates as mdates
import json
import os


# round time to 1 minute
def round_time(dt=None, round_to=60):
   if dt == None: 
       dt = datetime.now()
   seconds = (dt - dt.min).seconds
   rounding = (seconds+round_to/2) // round_to * round_to
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)
  

cases = pd.read_excel('Storm events.xlsx', sheet_name='Clustered wind storms sort (t)')
delta_1h         = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))

#data = np.zeros((10,np.size(temp,0)))


U_Osp1   = np.empty((0, 3))
Dir_Osp1 = np.empty((0, 3))
Iu_Osp1  = np.empty((0, 3))
Iv_Osp1  = np.empty((0, 3))
Iw_Osp1  = np.empty((0, 3))
ANu_Osp1  = np.empty((0, 3))
ANv_Osp1  = np.empty((0, 3))
ANw_Osp1  = np.empty((0, 3))
ANu_n_Osp1  = np.empty((0, 3))
ANv_n_Osp1  = np.empty((0, 3))
ANw_n_Osp1  = np.empty((0, 3))
Cux_Osp1  = np.empty((0, 1))
Cvx_Osp1  = np.empty((0, 1))
Cwx_Osp1  = np.empty((0, 1))
Cuy_Osp1  = np.empty((0, 1))
Cvy_Osp1  = np.empty((0, 1))
Cwy_Osp1  = np.empty((0, 1))
Cuz_Osp1  = np.empty((0, 1))
Cvz_Osp1  = np.empty((0, 1))
Cwz_Osp1  = np.empty((0, 1))

U_Osp2   = np.empty((0, 3))
Dir_Osp2 = np.empty((0, 3))
Iu_Osp2  = np.empty((0, 3))
Iv_Osp2  = np.empty((0, 3))
Iw_Osp2  = np.empty((0, 3))
ANu_Osp2  = np.empty((0, 3))
ANv_Osp2  = np.empty((0, 3))
ANw_Osp2  = np.empty((0, 3))
ANu_n_Osp2  = np.empty((0, 3))
ANv_n_Osp2  = np.empty((0, 3))
ANw_n_Osp2  = np.empty((0, 3))
Cux_Osp2  = np.empty((0, 1))
Cvx_Osp2  = np.empty((0, 1))
Cwx_Osp2  = np.empty((0, 1))
Cuy_Osp2  = np.empty((0, 1))
Cvy_Osp2  = np.empty((0, 1))
Cwy_Osp2  = np.empty((0, 1))
Cuz_Osp2  = np.empty((0, 1))
Cvz_Osp2  = np.empty((0, 1))
Cwz_Osp2  = np.empty((0, 1))

U_Sva   = np.empty((0, 3))
Dir_Sva = np.empty((0, 3))
Iu_Sva  = np.empty((0, 3))
Iv_Sva  = np.empty((0, 3))
Iw_Sva  = np.empty((0, 3))
ANu_Sva  = np.empty((0, 3))
ANv_Sva  = np.empty((0, 3))
ANw_Sva  = np.empty((0, 3))
ANu_n_Sva  = np.empty((0, 3))
ANv_n_Sva  = np.empty((0, 3))
ANw_n_Sva  = np.empty((0, 3))

U_Syn   = np.empty((0, 3))
Dir_Syn = np.empty((0, 3))
Iu_Syn  = np.empty((0, 3))
Iv_Syn  = np.empty((0, 3))
Iw_Syn  = np.empty((0, 3))
ANu_Syn  = np.empty((0, 3))
ANv_Syn  = np.empty((0, 3))
ANw_Syn  = np.empty((0, 3))
ANu_n_Syn  = np.empty((0, 3))
ANv_n_Syn  = np.empty((0, 3))
ANw_n_Syn  = np.empty((0, 3))

for event in range(0,cases['Time_storm'].size):

    date_storm  = datetime.strptime(cases['Time_storm'][event], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)

    date_start  = datetime.strptime(cases['Time_s1'][event], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
    date_end    = datetime.strptime(cases['Time_e1'][event], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None) 
   
    delta       = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
    num_case    = int(np.round((mdates.date2num(date_end)-mdates.date2num(date_start))/delta_1h))+1
    # cor coherence names
    date_begin = round_time (mdates.num2date(mdates.date2num(date_start)-0.5*delta_1h + delta_1h*(num_case-1)   ).replace(tzinfo=None))
    
   
    path      = 'E:/DATA/Results/Osp1 Spectra fitting/'     
    data_name = 'Storm No.' + str(event+1) + '_spectrum fitting Osp1 sensor No.3_'\
                 + str(date_start.year) + '_' + str(date_start.month) + '_' + str(date_start.day)\
                     + '_' + str(date_start.hour) 
    with open(path + data_name + "_smooth_json.txt") as json_file:
        data_Osp1  =  json.load(json_file)                 
    U_Osp1   = np.concatenate([U_Osp1, np.asarray(data_Osp1['U'])])
    Dir_Osp1 = np.concatenate([Dir_Osp1, np.asarray(data_Osp1['Dir'])])
    Iu_Osp1 = np.concatenate([Iu_Osp1, np.asarray(data_Osp1['Iu'])])
    Iv_Osp1 = np.concatenate([Iv_Osp1, np.asarray(data_Osp1['Iv'])])
    Iw_Osp1 = np.concatenate([Iw_Osp1, np.asarray(data_Osp1['Iw'])])
    ANu_Osp1 = np.concatenate([ANu_Osp1, np.asarray(data_Osp1['ANu'])])
    ANv_Osp1 = np.concatenate([ANv_Osp1, np.asarray(data_Osp1['ANv'])])
    ANw_Osp1 = np.concatenate([ANw_Osp1, np.asarray(data_Osp1['ANw'])])
    ANu_n_Osp1 = np.concatenate([ANu_n_Osp1, np.asarray(data_Osp1['ANu_n'])])
    ANv_n_Osp1 = np.concatenate([ANv_n_Osp1, np.asarray(data_Osp1['ANv_n'])])
    ANw_n_Osp1 = np.concatenate([ANw_n_Osp1, np.asarray(data_Osp1['ANw_n'])])


    path      = 'E:/DATA/Results/Osp1 Coherence fitting/'         
    data_name = 'Storm No.' + str(event+1)+ '_coherence fitting Osp1_' \
                 + str(date_begin.year) + '_' + str(date_begin.month) + '_' + str(date_begin.day)\
                     + '_' + str(date_begin.hour) 
    with open(path+ data_name + "_smooth_json.txt") as json_file:
        data_coh_Osp1  =  json.load(json_file)     
    Cux_Osp1 = np.concatenate([Cux_Osp1, np.asarray(data_coh_Osp1['Cux'])])    
    Cvx_Osp1 = np.concatenate([Cvx_Osp1, np.asarray(data_coh_Osp1['Cvx'])])    
    Cwx_Osp1 = np.concatenate([Cwx_Osp1, np.asarray(data_coh_Osp1['Cwx'])])    
    Cuy_Osp1 = np.concatenate([Cuy_Osp1, np.asarray(data_coh_Osp1['Cuy'])])    
    Cvy_Osp1 = np.concatenate([Cvy_Osp1, np.asarray(data_coh_Osp1['Cvy'])])    
    Cwy_Osp1 = np.concatenate([Cwy_Osp1, np.asarray(data_coh_Osp1['Cwy'])])    
    Cuz_Osp1 = np.concatenate([Cuz_Osp1, np.asarray(data_coh_Osp1['Cuz'])])    
    Cvz_Osp1 = np.concatenate([Cvz_Osp1, np.asarray(data_coh_Osp1['Cvz'])])    
    Cwz_Osp1 = np.concatenate([Cwz_Osp1, np.asarray(data_coh_Osp1['Cwz'])])    

    path      = 'E:/DATA/Results/Osp2 Spectra fitting/'             
    data_name = 'Storm No.' + str(event+1) + '_spectrum fitting Osp2 sensor No.3_'\
                 + str(date_start.year) + '_' + str(date_start.month) + '_' + str(date_start.day)\
                     + '_' + str(date_start.hour) 
    with open(path + data_name + "_smooth_json.txt") as json_file:
        data_Osp2  =  json.load(json_file)   
    U_Osp2   = np.concatenate([U_Osp2, np.asarray(data_Osp2['U'])])
    Dir_Osp2 = np.concatenate([Dir_Osp2, np.asarray(data_Osp2['Dir'])])
    Iu_Osp2 = np.concatenate([Iu_Osp2, np.asarray(data_Osp2['Iu'])])
    Iv_Osp2 = np.concatenate([Iv_Osp2, np.asarray(data_Osp2['Iv'])])
    Iw_Osp2 = np.concatenate([Iw_Osp2, np.asarray(data_Osp2['Iw'])])
    ANu_Osp2 = np.concatenate([ANu_Osp2, np.asarray(data_Osp2['ANu'])])
    ANv_Osp2 = np.concatenate([ANv_Osp2, np.asarray(data_Osp2['ANv'])])
    ANw_Osp2 = np.concatenate([ANw_Osp2, np.asarray(data_Osp2['ANw'])])
    ANu_n_Osp2 = np.concatenate([ANu_n_Osp2, np.asarray(data_Osp2['ANu_n'])])
    ANv_n_Osp2 = np.concatenate([ANv_n_Osp2, np.asarray(data_Osp2['ANv_n'])])
    ANw_n_Osp2 = np.concatenate([ANw_n_Osp2, np.asarray(data_Osp2['ANw_n'])])


    path      = 'E:/DATA/Results/Osp2 Coherence fitting/'             
    data_name = 'Storm No.' + str(event+1) + '_coherence fitting Osp2_'\
                 + str(date_begin.year) + '_' + str(date_begin.month) + '_' + str(date_begin.day)\
                     + '_' + str(date_begin.hour)    
    isFile = os.path.isfile(path + data_name + "_smooth_json.txt")        
    
    if isFile==True:                 
        with open(path + data_name + "_smooth_json.txt") as json_file:
            data_coh_Osp2  =  json.load(json_file)     
        Cux_Osp2 = np.concatenate([Cux_Osp2, np.asarray(data_coh_Osp2['Cux'])])    
        Cvx_Osp2 = np.concatenate([Cvx_Osp2, np.asarray(data_coh_Osp2['Cvx'])])    
        Cwx_Osp2 = np.concatenate([Cwx_Osp2, np.asarray(data_coh_Osp2['Cwx'])])    
        Cuy_Osp2 = np.concatenate([Cuy_Osp2, np.asarray(data_coh_Osp2['Cuy'])])    
        Cvy_Osp2 = np.concatenate([Cvy_Osp2, np.asarray(data_coh_Osp2['Cvy'])])    
        Cwy_Osp2 = np.concatenate([Cwy_Osp2, np.asarray(data_coh_Osp2['Cwy'])])    
        Cuz_Osp2 = np.concatenate([Cuz_Osp2, np.asarray(data_coh_Osp2['Cuz'])])    
        Cvz_Osp2 = np.concatenate([Cvz_Osp2, np.asarray(data_coh_Osp2['Cvz'])])    
        Cwz_Osp2 = np.concatenate([Cwz_Osp2, np.asarray(data_coh_Osp2['Cwz'])])  
    else:
        temps    = np.zeros(np.shape(data_coh_Osp1['Cux']))
        Cux_Osp2 = np.concatenate([Cux_Osp2, temps])    
        Cvx_Osp2 = np.concatenate([Cvx_Osp2, temps])    
        Cwx_Osp2 = np.concatenate([Cwx_Osp2, temps])    
        Cuy_Osp2 = np.concatenate([Cuy_Osp2, temps])    
        Cvy_Osp2 = np.concatenate([Cvy_Osp2, temps])    
        Cwy_Osp2 = np.concatenate([Cwy_Osp2, temps])    
        Cuz_Osp2 = np.concatenate([Cuz_Osp2, temps])    
        Cvz_Osp2 = np.concatenate([Cvz_Osp2, temps])    
        Cwz_Osp2 = np.concatenate([Cwz_Osp2, temps])                    

    
    path      = 'E:/DATA/Results/Svar Spectra fitting/'        
    data_name = 'Storm No.' + str(event+1) + '_spectrum fitting Svar sensor No.3_'\
                 + str(date_start.year) + '_' + str(date_start.month) + '_' + str(date_start.day)\
                     + '_' + str(date_start.hour) 
    with open(path+ data_name + "_smooth_json.txt") as json_file:
        data_Sva  =  json.load(json_file) 
    U_Sva   = np.concatenate([U_Sva, np.asarray(data_Sva['U'])])
    Dir_Sva = np.concatenate([Dir_Sva, np.asarray(data_Sva['Dir'])])
    Iu_Sva = np.concatenate([Iu_Sva, np.asarray(data_Sva['Iu'])])
    Iv_Sva = np.concatenate([Iv_Sva, np.asarray(data_Sva['Iv'])])
    Iw_Sva = np.concatenate([Iw_Sva, np.asarray(data_Sva['Iw'])])
    ANu_Sva = np.concatenate([ANu_Sva, np.asarray(data_Sva['ANu'])])
    ANv_Sva = np.concatenate([ANv_Sva, np.asarray(data_Sva['ANv'])])
    ANw_Sva = np.concatenate([ANw_Sva, np.asarray(data_Sva['ANw'])])
    ANu_n_Sva = np.concatenate([ANu_n_Sva, np.asarray(data_Sva['ANu_n'])])
    ANv_n_Sva = np.concatenate([ANv_n_Sva, np.asarray(data_Sva['ANv_n'])])
    ANw_n_Sva = np.concatenate([ANw_n_Sva, np.asarray(data_Sva['ANw_n'])])                     

    path      = 'E:/DATA/Results/Synn Spectra fitting/'        
    data_name = 'Storm No.' + str(event+1) + '_spectrum fitting Synn sensor No.3_'\
                 + str(date_start.year) + '_' + str(date_start.month) + '_' + str(date_start.day)\
                     + '_' + str(date_start.hour) 
    isFile = os.path.isfile(path + data_name + "_smooth_json.txt")        
    
    if isFile==True:                                            
        with open(path + data_name + "_smooth_json.txt") as json_file:
            data_Syn  =  json.load(json_file) 
        U_Syn   = np.concatenate([U_Syn, np.asarray(data_Syn['U'])])
        Dir_Syn = np.concatenate([Dir_Syn, np.asarray(data_Syn['Dir'])])
        Iu_Syn = np.concatenate([Iu_Syn, np.asarray(data_Syn['Iu'])])
        Iv_Syn = np.concatenate([Iv_Syn, np.asarray(data_Syn['Iv'])])
        Iw_Syn = np.concatenate([Iw_Syn, np.asarray(data_Syn['Iw'])])
        ANu_Syn = np.concatenate([ANu_Syn, np.asarray(data_Syn['ANu'])])
        ANv_Syn = np.concatenate([ANv_Syn, np.asarray(data_Syn['ANv'])])
        ANw_Syn = np.concatenate([ANw_Syn, np.asarray(data_Syn['ANw'])])
        ANu_n_Syn = np.concatenate([ANu_n_Syn, np.asarray(data_Syn['ANu_n'])])
        ANv_n_Syn = np.concatenate([ANv_n_Syn, np.asarray(data_Syn['ANv_n'])])
        ANw_n_Syn = np.concatenate([ANw_n_Syn, np.asarray(data_Syn['ANw_n'])]) 
    else:
        temps    = np.zeros(np.shape(data_coh_Osp1['U']))    
        U_Syn   = np.concatenate([U_Syn, temps])
        Dir_Syn = np.concatenate([Dir_Syn, temps])
        Iu_Syn = np.concatenate([Iu_Syn, temps])
        Iv_Syn = np.concatenate([Iv_Syn, temps])
        Iw_Syn = np.concatenate([Iw_Syn, temps])
        ANu_Syn = np.concatenate([ANu_Syn, temps])
        ANv_Syn = np.concatenate([ANv_Syn, temps])
        ANw_Syn = np.concatenate([ANw_Syn, temps])
        ANu_n_Syn = np.concatenate([ANu_n_Syn, temps])
        ANv_n_Syn = np.concatenate([ANv_n_Syn, temps])
        ANw_n_Syn = np.concatenate([ANw_n_Syn, temps])     
    

#%%
result_Osp1 = np.zeros((np.size(Cux_Osp1),17))
 
result_Osp1[:,0] = U_Osp1[:,0]
result_Osp1[:,1] = Dir_Osp1[:,0]
result_Osp1[:,2] = Iu_Osp1[:,0]
result_Osp1[:,3] = Iv_Osp1[:,0]
result_Osp1[:,4] = Iw_Osp1[:,0]    
result_Osp1[:,5] = ANu_Osp1[:,0]    
result_Osp1[:,6] = ANv_Osp1[:,0]    
result_Osp1[:,7] = ANw_Osp1[:,0]    
result_Osp1[:,8] = Cux_Osp1[:,0]    
result_Osp1[:,9] = Cvx_Osp1[:,0]    
result_Osp1[:,10] = Cwx_Osp1[:,0]    
result_Osp1[:,11] = Cuy_Osp1[:,0]    
result_Osp1[:,12] = Cvy_Osp1[:,0]    
result_Osp1[:,13] = Cwy_Osp1[:,0] 
result_Osp1[:,14] = Cuz_Osp1[:,0]    
result_Osp1[:,15] = Cvz_Osp1[:,0]    
result_Osp1[:,16] = Cwz_Osp1[:,0] 

result_Osp1[result_Osp1 == 0]  = np.nan
result_Osp1_sort = np.sort(result_Osp1,0)
result_Osp1_sort[result_Osp1_sort==0] = np.nan

result_Osp1_B = np.zeros((np.size(Cux_Osp1),17))
result_Osp1_B[:,0] = U_Osp1[:,1]
result_Osp1_B[:,1] = Dir_Osp1[:,1]
result_Osp1_B[:,2] = Iu_Osp1[:,1]
result_Osp1_B[:,3] = Iv_Osp1[:,1]
result_Osp1_B[:,4] = Iw_Osp1[:,1]    
result_Osp1_B[:,5] = ANu_Osp1[:,1]    
result_Osp1_B[:,6] = ANv_Osp1[:,1]    
result_Osp1_B[:,7] = ANw_Osp1[:,1]    

result_Osp1_B[result_Osp1_B == 0]  = np.nan
result_Osp1_B_sort = np.sort(result_Osp1_B,0)
result_Osp1_B_sort[result_Osp1_B_sort==0] = np.nan


result_Osp2 = np.zeros((np.size(Cux_Osp1),17))
result_Osp2[:,0] = U_Osp2[:,0]
result_Osp2[:,1] = Dir_Osp2[:,0]
result_Osp2[:,2] = Iu_Osp2[:,0]
result_Osp2[:,3] = Iv_Osp2[:,0]
result_Osp2[:,4] = Iw_Osp2[:,0]    
result_Osp2[:,5] = ANu_Osp2[:,0]    
result_Osp2[:,6] = ANv_Osp2[:,0]    
result_Osp2[:,7] = ANw_Osp2[:,0]    
result_Osp2[:,8] = Cux_Osp2[:,0]    
result_Osp2[:,9] = Cvx_Osp2[:,0]    
result_Osp2[:,10] = Cwx_Osp2[:,0]    
result_Osp2[:,11] = Cuy_Osp2[:,0]    
result_Osp2[:,12] = Cvy_Osp2[:,0]    
result_Osp2[:,13] = Cwy_Osp2[:,0] 
result_Osp2[:,14] = Cuz_Osp2[:,0]    
result_Osp2[:,15] = Cvz_Osp2[:,0]    
result_Osp2[:,16] = Cwz_Osp2[:,0] 

result_Osp2[result_Osp2 == 0]  = np.nan
result_Osp2_sort = np.sort(result_Osp2,0)
result_Osp2_sort[result_Osp2_sort==0] = np.nan

result_Osp2_B = np.zeros((np.size(Cux_Osp1),17))
result_Osp2_B[:,0] = U_Osp2[:,1]
result_Osp2_B[:,1] = Dir_Osp2[:,1]
result_Osp2_B[:,2] = Iu_Osp2[:,1]
result_Osp2_B[:,3] = Iv_Osp2[:,1]
result_Osp2_B[:,4] = Iw_Osp2[:,1]    
result_Osp2_B[:,5] = ANu_Osp2[:,1]    
result_Osp2_B[:,6] = ANv_Osp2[:,1]    
result_Osp2_B[:,7] = ANw_Osp2[:,1]    

result_Osp2_B[result_Osp2_B==0]  = np.nan
result_Osp2_B_sort = np.sort(result_Osp2_B,0)
result_Osp2_B_sort[result_Osp2_B_sort==0] = np.nan

result_Sva = np.zeros((np.size(Cux_Osp1),8))
result_Sva[:,0] = U_Sva[:,0]
result_Sva[:,1] = Dir_Sva[:,0]
result_Sva[:,2] = Iu_Sva[:,0]
result_Sva[:,3] = Iv_Sva[:,0]
result_Sva[:,4] = Iw_Sva[:,0]    
result_Sva[:,5] = ANu_Sva[:,0]    
result_Sva[:,6] = ANv_Sva[:,0]    
result_Sva[:,7] = ANw_Sva[:,0]    

result_Sva[result_Sva==0]  = np.nan
result_Sva_sort = np.sort(result_Sva,0)
result_Sva_sort[result_Sva_sort==0] = np.nan


result_Syn = np.zeros((np.size(Cux_Osp1),8))
result_Syn[:,0] = U_Syn[:,0]
result_Syn[:,1] = Dir_Syn[:,0]
result_Syn[:,2] = Iu_Syn[:,0]
result_Syn[:,3] = Iv_Syn[:,0]
result_Syn[:,4] = Iw_Syn[:,0]    
result_Syn[:,5] = ANu_Syn[:,0]    
result_Syn[:,6] = ANv_Syn[:,0]    
result_Syn[:,7] = ANw_Syn[:,0]    

result_Syn[result_Syn==0]  = np.nan
result_Syn_sort = np.sort(result_Syn,0)
result_Syn_sort[result_Syn_sort==0] = np.nan


#%%
# relationship between Iu and Anu
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(111)
# format the ticks
ax1.plot(result_Osp1[:,4],result_Osp1[:,2],'ko')
plt.legend(loc='best',ncol=1,fontsize=14)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(np.multiply(result_Osp1[:,2],result_Osp1[:,0]) , result_Osp1[:,0],'rs',label='Osp1 A')
ax1.plot(np.multiply(result_Osp1_B[:,2],result_Osp1[:,0]), result_Osp1_B[:,0],'go',label='Osp1 B')
ax1.plot(np.multiply(result_Osp2[:,2],result_Osp1[:,0]) ,  result_Osp2[:,0],'b^',label='Osp2 A')
ax1.plot(np.multiply(result_Osp2_B[:,2],result_Osp1[:,0]) ,result_Osp2_B[:,0],'k*',label='Osp2 B')
ax1.plot(np.multiply(result_Sva[:,2],result_Osp1[:,0]) ,   result_Sva[:,0],'cp',label='Sva A')
ax1.plot(np.multiply(result_Syn[:,2],result_Osp1[:,0]) ,   result_Syn[:,0],'yx',label='Syn A')
plt.legend(loc='best',ncol=1,fontsize=14)
plt.ylabel(r'$\overline{U} $', fontsize=20)
plt.xlabel(r'$I_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

#%% A dependency

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(result_Osp1[:,2],   result_Osp1[:,0],'rs',label='Osp1 A')
ax1.plot(result_Osp1_B[:,2], result_Osp1_B[:,0],'go',label='Osp1 B')
ax1.plot(result_Osp2[:,2],   result_Osp2[:,0],'b^',label='Osp2 A')
ax1.plot(result_Osp2_B[:,2], result_Osp2_B[:,0],'k*',label='Osp2 B')
ax1.plot(result_Sva[:,2],    result_Sva[:,0],'cp',label='Sva A')
ax1.plot(result_Syn[:,2],    result_Syn[:,0],'yx',label='Syn A')
plt.legend(loc='best',ncol=1,fontsize=14)
plt.ylabel(r'$\overline{U} $', fontsize=20)
plt.xlabel(r'$I_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(result_Osp1[:,2],result_Osp1[:,3],'rs',label='Osp1 A')
ax1.plot(result_Osp1_B[:,2],result_Osp1_B[:,3],'go',label='Osp1 B')
ax1.plot(result_Osp2[:,2],result_Osp2[:,3],'b^',label='Osp2 A')
ax1.plot(result_Osp2_B[:,2],result_Osp2_B[:,3],'k*',label='Osp2 B')
ax1.plot(result_Sva[:,2],result_Sva[:,3],'cp',label='Sva A')
ax1.plot(result_Syn[:,2],result_Syn[:,3],'yx',label='Syn A')
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$I_v$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(result_Osp1[:,2],result_Osp1[:,4],'rs',label='Osp1 A')
ax1.plot(result_Osp1_B[:,2],result_Osp1_B[:,4],'go',label='Osp1 B')
ax1.plot(result_Osp2[:,2],result_Osp2[:,4],'b^',label='Osp2 A')
ax1.plot(result_Osp2_B[:,2],result_Osp2_B[:,4],'k*',label='Osp2 B')
ax1.plot(result_Sva[:,2],result_Sva[:,4],'cp',label='Sva A')
ax1.plot(result_Syn[:,2],result_Syn[:,4],'yx',label='Syn A')
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$I_w$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(result_Osp1[:,5],result_Osp1[:,0],'rs',label='Osp1 A')
ax1.plot(result_Osp1_B[:,5],result_Osp1_B[:,0],'go',label='Osp1 B')
ax1.plot(result_Osp2[:,5],result_Osp2[:,0],'b^',label='Osp2 A')
ax1.plot(result_Osp2_B[:,5],result_Osp2_B[:,0],'k*',label='Osp2 B')
ax1.plot(result_Sva[:,5],result_Sva[:,0],'cp',label='Sva A')
ax1.plot(result_Syn[:,5],result_Syn[:,0],'yx',label='Syn A')
plt.ylabel(r'$\overline{U} $', fontsize=20)
plt.xlabel(r'$A_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(result_Osp1[:,5],result_Osp1[:,6],'rs',label='Osp1 A')
ax1.plot(result_Osp1_B[:,5],result_Osp1_B[:,6],'go',label='Osp1 B')
ax1.plot(result_Osp2[:,5],result_Osp2[:,6],'b^',label='Osp2 A')
ax1.plot(result_Osp2_B[:,5],result_Osp2_B[:,6],'k*',label='Osp2 B')
ax1.plot(result_Sva[:,5],result_Sva[:,6],'cp',label='Sva A')
ax1.plot(result_Syn[:,5],result_Syn[:,6],'yx',label='Syn A')
plt.ylabel(r'$A_v$', fontsize=20)
plt.xlabel(r'$A_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(result_Osp1[:,5],result_Osp1[:,7],'rs',label='Osp1 A')
ax1.plot(result_Osp1_B[:,5],result_Osp1_B[:,7],'go',label='Osp1 B')
ax1.plot(result_Osp2[:,5],result_Osp2[:,7],'b^',label='Osp2 A')
ax1.plot(result_Osp2_B[:,5],result_Osp2_B[:,7],'k*',label='Osp2 B')
ax1.plot(result_Sva[:,5],result_Sva[:,7],'cp',label='Sva A')
ax1.plot(result_Syn[:,5],result_Syn[:,7],'yx',label='Syn A')
plt.ylabel(r'$A_w$', fontsize=20)
plt.xlabel(r'$A_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot(result_Osp1[:,2],result_Osp1[:,5],'rs',label='Osp1 A')
ax1.plot(result_Osp1_B[:,2],result_Osp1_B[:,5],'go',label='Osp1 B')
ax1.plot(result_Osp2[:,2],result_Osp2[:,5],'b^',label='Osp2 A')
ax1.plot(result_Osp2_B[:,2],result_Osp2_B[:,5],'k*',label='Osp2 B')
ax1.plot(result_Sva[:,2],result_Sva[:,5],'cp',label='Sva A')
ax1.plot(result_Syn[:,2],result_Syn[:,5],'yx',label='Syn A')
plt.ylabel(r'$A_u$', fontsize=20)
plt.xlabel(r'$I_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(result_Osp1[:,3],result_Osp1[:,6],'rs',label='Osp1 A')
ax1.plot(result_Osp1_B[:,3],result_Osp1_B[:,6],'go',label='Osp1 B')
ax1.plot(result_Osp2[:,3],result_Osp2[:,6],'b^',label='Osp2 A')
ax1.plot(result_Osp2_B[:,3],result_Osp2_B[:,6],'k*',label='Osp2 B')
ax1.plot(result_Sva[:,3],result_Sva[:,6],'cp',label='Sva A')
ax1.plot(result_Syn[:,3],result_Syn[:,6],'yx',label='Syn A')
plt.ylabel(r'$A_v$', fontsize=20)
plt.xlabel(r'$I_v$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(result_Osp1[:,4],result_Osp1[:,7],'rs',label='Osp1 A')
ax1.plot(result_Osp1_B[:,4],result_Osp1_B[:,7],'go',label='Osp1 B')
ax1.plot(result_Osp2[:,4],result_Osp2[:,7],'b^',label='Osp2 A')
ax1.plot(result_Osp2_B[:,4],result_Osp2_B[:,7],'k*',label='Osp2 B')
ax1.plot(result_Sva[:,4],result_Sva[:,7],'cp',label='Sva A')
ax1.plot(result_Syn[:,4],result_Syn[:,7],'yx',label='Syn A')
plt.ylabel(r'$A_w$', fontsize=20)
plt.xlabel(r'$I_w$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

save_tite = 'A parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite) 


#%% Cu dependency

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(result_Osp1[:,0],result_Osp1[:,8],'rs',label='Osp1')
ax1.plot(result_Osp2[:,0],result_Osp2[:,8],'b^',label='Osp2')
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$\overline{U} $', fontsize=20)
plt.ylabel(r'$C_{ux}$', fontsize=20)
ax1.set_ylim(0      , 50)

g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(result_Osp1[:,0],result_Osp1[:,11],'rs',label='Osp1')
ax1.plot(result_Osp2[:,0],result_Osp2[:,11],'b^',label='Osp2')
plt.xlabel(r'$\overline{U} $', fontsize=20)
plt.ylabel(r'$C_{uy}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(result_Osp1[:,0],result_Osp1[:,14],'rs',label='Osp1')
ax1.plot(result_Osp2[:,0],result_Osp2[:,14],'b^',label='Osp2')
plt.xlabel(r'$\overline{U} $', fontsize=20)
plt.ylabel(r'$C_{uz}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(result_Osp1[:,2],result_Osp1[:,8],'rs',label='Osp1')
ax1.plot(result_Osp2[:,2],result_Osp2[:,8],'b^',label='Osp2')
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$C_{ux}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(result_Osp1[:,2],result_Osp1[:,11],'rs',label='Osp1')
ax1.plot(result_Osp2[:,2],result_Osp2[:,11],'b^',label='Osp2')
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$C_{uy}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(result_Osp1[:,2],result_Osp1[:,14],'rs',label='Osp1')
ax1.plot(result_Osp2[:,2],result_Osp2[:,14],'b^',label='Osp2')
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$C_{uz}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot(result_Osp1[:,5],result_Osp1[:,8],'rs',label='Osp1')
ax1.plot(result_Osp2[:,5],result_Osp2[:,8],'b^',label='Osp2')
plt.xlabel(r'$A_u$', fontsize=20)
plt.ylabel(r'$C_{ux}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(result_Osp1[:,5],result_Osp1[:,11],'rs',label='Osp1')
ax1.plot(result_Osp2[:,5],result_Osp2[:,11],'b^',label='Osp2')
plt.xlabel(r'$A_u$', fontsize=20)
plt.ylabel(r'$C_{uy}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(result_Osp1[:,5],result_Osp1[:,14],'rs',label='Osp1')
ax1.plot(result_Osp2[:,5],result_Osp2[:,14],'b^',label='Osp2')
plt.xlabel(r'$A_u$', fontsize=20)
plt.ylabel(r'$C_{uz}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

save_tite = 'Cu parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite) 


#%% Cv dependency

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(result_Osp1[:,0],result_Osp1[:,9],'rs',label='Osp1')
ax1.plot(result_Osp2[:,0],result_Osp2[:,9],'b^',label='Osp2')
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$\overline{U} $', fontsize=20)
plt.ylabel(r'$C_{vx}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(result_Osp1[:,0],result_Osp1[:,12],'rs',label='Osp1')
ax1.plot(result_Osp2[:,0],result_Osp2[:,12],'b^',label='Osp2')
plt.xlabel(r'$\overline{U} $', fontsize=20)
plt.ylabel(r'$C_{vy}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(result_Osp1[:,0],result_Osp1[:,15],'rs',label='Osp1')
ax1.plot(result_Osp2[:,0],result_Osp2[:,15],'b^',label='Osp2')
plt.xlabel(r'$\overline{U} $', fontsize=20)
plt.ylabel(r'$C_{vz}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(result_Osp1[:,3],result_Osp1[:,9],'rs',label='Osp1')
ax1.plot(result_Osp2[:,3],result_Osp2[:,9],'b^',label='Osp2')
plt.xlabel(r'$I_v$', fontsize=20)
plt.ylabel(r'$C_{vx}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(result_Osp1[:,3],result_Osp1[:,12],'rs',label='Osp1')
ax1.plot(result_Osp2[:,3],result_Osp2[:,12],'b^',label='Osp2')
plt.xlabel(r'$I_v$', fontsize=20)
plt.ylabel(r'$C_{vy}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(result_Osp1[:,3],result_Osp1[:,15],'rs',label='Osp1')
ax1.plot(result_Osp2[:,3],result_Osp2[:,15],'b^',label='Osp2')
plt.xlabel(r'$I_v$', fontsize=20)
plt.ylabel(r'$C_{vz}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot(result_Osp1[:,6],result_Osp1[:,9],'rs',label='Osp1')
ax1.plot(result_Osp2[:,6],result_Osp2[:,9],'b^',label='Osp2')
plt.xlabel(r'$A_v$', fontsize=20)
plt.ylabel(r'$C_{vx}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(result_Osp1[:,6],result_Osp1[:,12],'rs',label='Osp1')
ax1.plot(result_Osp2[:,6],result_Osp2[:,12],'b^',label='Osp2')
plt.xlabel(r'$A_v$', fontsize=20)
plt.ylabel(r'$C_{vy}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(result_Osp1[:,6],result_Osp1[:,15],'rs',label='Osp1')
ax1.plot(result_Osp2[:,6],result_Osp2[:,15],'b^',label='Osp2')
plt.xlabel(r'$A_v$', fontsize=20)
plt.ylabel(r'$C_{vz}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

save_tite = 'Cv parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite) 

#%% Cw dependency

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(result_Osp1[:,0],result_Osp1[:,10],'rs',label='Osp1')
ax1.plot(result_Osp2[:,0],result_Osp2[:,10],'b^',label='Osp2')
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$\overline{U} $', fontsize=20)
plt.ylabel(r'$C_{wx}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(result_Osp1[:,0],result_Osp1[:,13],'rs',label='Osp1')
ax1.plot(result_Osp2[:,0],result_Osp2[:,13],'b^',label='Osp2')
plt.xlabel(r'$\overline{U} $', fontsize=20)
plt.ylabel(r'$C_{wy}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(result_Osp1[:,0],result_Osp1[:,16],'rs',label='Osp1')
ax1.plot(result_Osp2[:,0],result_Osp2[:,16],'b^',label='Osp2')
plt.xlabel(r'$\overline{U} $', fontsize=20)
plt.ylabel(r'$C_{wz}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(result_Osp1[:,4],result_Osp1[:,10],'rs',label='Osp1')
ax1.plot(result_Osp2[:,4],result_Osp2[:,10],'b^',label='Osp2')
plt.xlabel(r'$I_w$', fontsize=20)
plt.ylabel(r'$C_{wx}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(result_Osp1[:,4],result_Osp1[:,13],'rs',label='Osp1')
ax1.plot(result_Osp2[:,3],result_Osp2[:,13],'b^',label='Osp2')
plt.xlabel(r'$I_w$', fontsize=20)
plt.ylabel(r'$C_{wy}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(result_Osp1[:,4],result_Osp1[:,16],'rs',label='Osp1')
ax1.plot(result_Osp2[:,4],result_Osp2[:,16],'b^',label='Osp2')
plt.xlabel(r'$I_w$', fontsize=20)
plt.ylabel(r'$C_{wz}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot(result_Osp1[:,7],result_Osp1[:,10],'rs',label='Osp1')
ax1.plot(result_Osp2[:,7],result_Osp2[:,10],'b^',label='Osp2')
plt.xlabel(r'$A_w$', fontsize=20)
plt.ylabel(r'$C_{wx}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(result_Osp1[:,7],result_Osp1[:,13],'rs',label='Osp1')
ax1.plot(result_Osp2[:,7],result_Osp2[:,13],'b^',label='Osp2')
plt.xlabel(r'$A_w$', fontsize=20)
plt.ylabel(r'$C_{wy}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(result_Osp1[:,7],result_Osp1[:,16],'rs',label='Osp1')
ax1.plot(result_Osp2[:,7],result_Osp2[:,16],'b^',label='Osp2')
plt.xlabel(r'$A_w$', fontsize=20)
plt.ylabel(r'$C_{wz}$', fontsize=20)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

save_tite = 'Cw parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite)


#%% C dependency

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(result_Osp1[:,8],result_Osp1[:,9],'rs',label='Osp1')
ax1.plot(result_Osp2[:,8],result_Osp2[:,9],'b^',label='Osp2')
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{ux}$', fontsize=20)
plt.ylabel(r'$C_{vx}$', fontsize=20)
ax1.set_xlim(0      , 50)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(result_Osp1[:,8],result_Osp1[:,10],'rs',label='Osp1')
ax1.plot(result_Osp2[:,8],result_Osp2[:,10],'b^',label='Osp2')
plt.xlabel(r'$C_{ux}$', fontsize=20)
plt.ylabel(r'$C_{wx}$', fontsize=20)
ax1.set_xlim(0      , 50)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(result_Osp1[:,8],result_Osp1[:,11],'rs',label='Osp1')
ax1.plot(result_Osp2[:,8],result_Osp2[:,11],'b^',label='Osp2')
plt.xlabel(r'$C_{ux}$', fontsize=20)
plt.ylabel(r'$C_{uy}$', fontsize=20)
ax1.set_xlim(0      , 50)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(result_Osp1[:,11],result_Osp1[:,12],'rs',label='Osp1')
ax1.plot(result_Osp2[:,11],result_Osp2[:,12],'b^',label='Osp2')
plt.xlabel(r'$C_{uy}$', fontsize=20)
plt.ylabel(r'$C_{vy}$', fontsize=20)
ax1.set_xlim(0      , 50)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(result_Osp1[:,11],result_Osp1[:,13],'rs',label='Osp1')
ax1.plot(result_Osp2[:,11],result_Osp2[:,13],'b^',label='Osp2')
plt.xlabel(r'$C_{uy}$', fontsize=20)
plt.ylabel(r'$C_{wy}$', fontsize=20)
ax1.set_xlim(0      , 50)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(result_Osp1[:,8],result_Osp1[:,14],'rs',label='Osp1')
ax1.plot(result_Osp2[:,8],result_Osp2[:,14],'b^',label='Osp2')
plt.xlabel(r'$C_{ux}$', fontsize=20)
plt.ylabel(r'$C_{uz}$', fontsize=20)
ax1.set_xlim(0      , 50)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot(result_Osp1[:,14],result_Osp1[:,15],'rs',label='Osp1')
ax1.plot(result_Osp2[:,14],result_Osp2[:,15],'b^',label='Osp2')
plt.xlabel(r'$C_{uz}$', fontsize=20)
plt.ylabel(r'$C_{vz}$', fontsize=20)
ax1.set_xlim(0      , 50)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(result_Osp1[:,14],result_Osp1[:,16],'rs',label='Osp1')
ax1.plot(result_Osp2[:,14],result_Osp2[:,16],'b^',label='Osp2')
plt.xlabel(r'$C_{uz}$', fontsize=20)
plt.ylabel(r'$C_{wz}$', fontsize=20)
ax1.set_xlim(0      , 50)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(result_Osp1[:,11],result_Osp1[:,14],'rs',label='Osp1')
ax1.plot(result_Osp2[:,11],result_Osp2[:,14],'b^',label='Osp2')
plt.xlabel(r'$C_{uy}$', fontsize=20)
plt.ylabel(r'$C_{uz}$', fontsize=20)
ax1.set_xlim(0      , 50)
ax1.set_ylim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

save_tite = 'C parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite)
#%% Histogram of A

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(311)
# format the ticks
ax1.plot([6.8,6.8],[0,10],'r-',label='$A_u$=6.8')
ax1.hist([result_Osp1_sort[:,5],result_Osp1_B_sort[:,5],result_Osp2_sort[:,5],result_Osp2_B_sort[:,5],result_Sva_sort[:,5],result_Syn_sort[:,5]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1 A','Osp1 B','Osp2 A','Osp2 B','Sva A','Syn A'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$A_u$', fontsize=20)
plt.ylabel(r'Number of events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()
fig.suptitle('Distribution of the fitted A parameter from 52 strong wind events', fontsize=25)

ax1     = plt.subplot(312)
ax1.plot([9.4,9.4],[0,10],'r-',label=r'$A_v$=9.4')
ax1.hist([result_Osp1_sort[:,6],result_Osp1_B_sort[:,6],result_Osp2_sort[:,6],result_Osp2_B_sort[:,6],result_Sva_sort[:,6],result_Syn_sort[:,6]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1 A','Osp1 B','Osp2 A','Osp2 B','Sva A','Syn A'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$A_v$', fontsize=20)
plt.ylabel(r'Number of events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(313)
# format the ticks
ax1.plot([9.4,9.4],[0,10],'r-',label=r'$A_w$=9.4')
ax1.hist([result_Osp1_sort[:,7],result_Osp1_B_sort[:,7],result_Osp2_sort[:,7],result_Osp2_B_sort[:,7],result_Sva_sort[:,7],result_Syn_sort[:,7]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1 A','Osp1 B','Osp2 A','Osp2 B','Sva A','Syn A'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$A_w$', fontsize=20)
plt.ylabel(r'Number of events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

save_tite = 'Distribution of A parameters.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite) 

#%% Histogram of C

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot([3,3],[0,10],'r-',label='$C_{ux}$=3')
ax1.hist([result_Osp1_sort[:,8],result_Osp2_sort[:,8]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1','Osp2'], range=(0,50))
plt.legend(loc='best',ncol=1,fontsize=14)
ax1.set_xlim(0      , 50)
plt.xlabel(r'$C_{ux}$', fontsize=20)
plt.ylabel(r'Number of events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()
fig.suptitle('Distribution of the fitted C parameter from 52 strong wind events', fontsize=25)

ax1     = plt.subplot(334)
ax1.plot([6,6],[0,10],'r-',label='$C_{vx}$=6')
ax1.hist([result_Osp1_sort[:,9],result_Osp2_sort[:,9]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1','Osp2'], range=(0,50))
plt.legend(loc='best',ncol=1,fontsize=14)
ax1.set_xlim(0      , 50)
plt.xlabel(r'$C_{vx}$', fontsize=20)
plt.ylabel(r'Number of events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot([3,3],[0,10],'r-',label='$C_{wx}$=3')
ax1.hist([result_Osp1_sort[:,10],result_Osp2_sort[:,10]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1','Osp2'], range=(0,50))
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{wx}$', fontsize=20)
ax1.set_xlim(0      , 50)
plt.ylabel(r'Number of events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot([10,10],[0,10],'r-',label='$C_{uy}$=10')
ax1.hist([result_Osp1_sort[:,11],result_Osp2_sort[:,11]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1','Osp2'], range=(0,50))
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{uy}$', fontsize=20)
ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot([6.5,6.5],[0,10],'r-',label='$C_{vy}$=6.5')
ax1.hist([result_Osp1_sort[:,12],result_Osp2_sort[:,12]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1','Osp2'], range=(0,50))
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{vy}$', fontsize=20)
ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot([6.5,6.5],[0,10],'r-',label='$C_{wy}$=6.5')
ax1.hist([result_Osp1_sort[:,13],result_Osp2_sort[:,13]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1','Osp2'], range=(0,50))
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{wy}$', fontsize=20)
ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot([10,10],[0,10],'r-',label='$C_{uz}$=10')
ax1.hist([result_Osp1_sort[:,14],result_Osp2_sort[:,14]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1','Osp2'], range=(0,50))
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{uz}$', fontsize=20)
ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot([6.5,6.5],[0,10],'r-',label='$C_{vz}$=6.5')
ax1.hist([result_Osp1_sort[:,15],result_Osp2_sort[:,15]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1','Osp2'], range=(0,50))
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{vz}$', fontsize=20)
ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot([3,3],[0,10],'r-',label='$C_{wz}$=3')
ax1.hist([result_Osp1_sort[:,13],result_Osp2_sort[:,13]],\
         20,cumulative=False, histtype='bar', alpha=0.4,label=['Osp1','Osp2'], range=(0,50))
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{wy}$', fontsize=20)
ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

save_tite = 'Distribution of C parameters.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite) 

#%%
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(111)
# format the ticks
ax1.hist([result_Osp1[:,5],result_Osp1[:,6],result_Osp1[:,7]],20,cumulative=False, histtype='bar', alpha=0.2,label=['Au','Av','Aw'])
plt.legend(loc='best',ncol=1,fontsize=14)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(111)
# format the ticks
ax1.hist([result_Osp1[:,11],result_Osp1[:,12],result_Osp1[:,13]],20,cumulative=False, histtype='bar', alpha=0.2,label=['Cuy','Cvy','Cwy'])
plt.legend(loc='best',ncol=1,fontsize=14)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()
#%%

dataset_Osp1 = pd.DataFrame({'U_Osp1': U_Osp1[:,0],
                        'Dir_Osp1': Dir_Osp1[:,0],
                        'Iu_Osp1': Iu_Osp1[:,0],
                        'Iv_Osp1': Iv_Osp1[:,0],
                        'Iw_Osp1': Iw_Osp1[:,0],
                        'ANu_Osp1': ANu_Osp1[:,0],
                        'ANv_Osp1': ANv_Osp1[:,0],
                        'ANw_Osp1': ANw_Osp1[:,0],
                        'Cux_Osp1': Cux_Osp1[:,0],
                        'Cvx_Osp1': Cvx_Osp1[:,0],
                        'Cwx_Osp1': Cwx_Osp1[:,0],
                        'Cuy_Osp1': Cuy_Osp1[:,0],
                        'Cvy_Osp1': Cvy_Osp1[:,0],
                        'Cwy_Osp1': Cwy_Osp1[:,0],
                        'Cuz_Osp1': Cuz_Osp1[:,0],
                        'Cvz_Osp1': Cvz_Osp1[:,0],
                        'Cwz_Osp1': Cwz_Osp1[:,0]}                       
                       )
    
dataset_Osp1.replace(0, np.nan, inplace=True)


#%%



result_sort=dataset_Osp1.sort_values(by =['U_Osp1'])
wbl_res   = stats.weibull_min.fit(result_sort['U_Osp1'],floc=0)
logn_res  = stats.lognorm.fit(result_sort['U_Osp1'],floc=0)
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(111)
# format the ticks
ax1.hist(result_sort['U_Osp1'],cumulative=False, density=1,histtype='stepfilled', alpha=0.2)
ax1.plot(result_sort['U_Osp1'],stats.weibull_min.pdf(result_sort['U_Osp1'],*wbl_res), alpha=0.6,label='Weibull')
ax1.plot(result_sort['U_Osp1'],stats.lognorm.pdf(result_sort['U_Osp1'],*logn_res),label='Lognormal')
plt.legend(loc='best',ncol=2,fontsize=14)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


result_sort=dataset.sort_values(by =['ANu_Osp1'])
wbl_res   = stats.weibull_min.fit(result_sort['ANu_Osp1'],floc=0)
logn_res  = stats.lognorm.fit(result_sort['ANu_Osp1'],floc=0)
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(111)
# format the ticks
ax1.hist(result_sort['ANu_Osp1'],cumulative=False, density=1,histtype='stepfilled', alpha=0.2)
ax1.plot(result_sort['ANu_Osp1'],stats.weibull_min.pdf(result_sort['ANu_Osp1'],*wbl_res), alpha=0.6,label='Weibull')
ax1.plot(result_sort['ANu_Osp1'],stats.lognorm.pdf(result_sort['ANu_Osp1'],*logn_res),label='Lognormal')
plt.legend(loc='best',ncol=2,fontsize=14)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


result_sort=dataset.sort_values(by =['Cuy_Osp1'])
wbl_res   = stats.weibull_min.fit(result_sort['Cuy_Osp1'],floc=0)
logn_res  = stats.lognorm.fit(result_sort['Cuy_Osp1'],floc=0)
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(111)
# format the ticks
ax1.hist(result_sort['Cuy_Osp1'],cumulative=False, density=1,histtype='stepfilled', alpha=0.2)
ax1.plot(result_sort['Cuy_Osp1'],stats.weibull_min.pdf(result_sort['Cuy_Osp1'],*wbl_res), alpha=0.6,label='Weibull')
ax1.plot(result_sort['Cuy_Osp1'],stats.lognorm.pdf(result_sort['Cuy_Osp1'],*logn_res),label='Lognormal')
plt.legend(loc='best',ncol=2,fontsize=14)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

#%%
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))

ax1     = plt.subplot(421)
# format the ticks
ax1.plot(result_sort['Dir_Osp1'], result_sort['U_Osp1'],   'rs'  ,label='Osp1 Au',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(422)
# format the ticks
ax1.plot(result_sort['U_Osp1'], result_sort['Iu_Osp1'],   'rs'  ,label='Osp1 Au',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(423)
# format the ticks
ax1.plot(result_sort['Iv_Osp1'], result_sort['Iu_Osp1'],   'rs'  ,label='Osp1 Au',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(424)
# format the ticks
ax1.plot(result_sort['Iw_Osp1'], result_sort['Iu_Osp1'],   'rs'  ,label='Osp1 Au',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(425)
# format the ticks
ax1.plot(result_sort['U_Osp1'], result_sort['ANu_Osp1'],   'rs'  ,label='Osp1 Au',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(426)
# format the ticks
ax1.plot(result_sort['U_Osp1'], result_sort['ANw_Osp1'],   'rs'  ,label='Osp1 Au',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

#%%
# plot the A parameter sensitivity on different sensors (four top sensor at Ospøya)
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))

ax1     = plt.subplot(421)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['U'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['U'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['U'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['U'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.set_ylim(0      , 30 )
plt.legend(loc='best',ncol=2,fontsize=14)
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
fig.suptitle('Storm No.' + str(event+1) + ' spectral fitting from different sensors at Ospøya', fontsize=25)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(422)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Dir'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Dir'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Dir'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Dir'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(423)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iu'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iu'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iu'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iu'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Iu_ref,                           'r-'  ,label='$Iu (N400)$',linewidth=2)
plt.ylabel(r'$I_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(424)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANu_sm[:,0],       'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_sm[:,1],       'go'  ,label='Osp1 B' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_sm_Osp2[:,0],  'b^'  ,label='Osp2 A'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_sm_Osp2[:,1],  'k*'  ,label='Osp2 B'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(425)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iv'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iv'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iv'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iv'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Iv_ref,                           'g-'  ,label='$Iv (N400)$',linewidth=2)
plt.ylabel(r'$I_v$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

            
ax1     = plt.subplot(426)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANv_sm[:,0],       'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_sm[:,1],       'go'  ,label='Osp1 B' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_sm_Osp2[:,0],  'b^'  ,label='Osp2 A'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_sm_Osp2[:,1],  'k*'  ,label='Osp2 B'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Av (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_v$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()          

ax1     = plt.subplot(427)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iw'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iw'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iw'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iw'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Iw_ref,                           'b-'  ,label='$Iw (N400)$',linewidth=2)
plt.ylabel(r'$I_w$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(428)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANw_sm[:,0],       'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_sm[:,1],       'go'  ,label='Osp1 B' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_sm_Osp2[:,0],  'b^'  ,label='Osp2 A'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_sm_Osp2[:,1],  'k*'  ,label='Osp2 B'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Aw_ref,        'b-'  ,label='$Aw (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_w$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()           

save_tite = 'Storm No.' + str(event+1) + '_spectral fitting sensitivity different sensors at Ospøya.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite) 

#%%
# plot the A parameter sensitivity on different sensors (Six top sensor at Ospøya Sva and Syn)
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))

ax1     = plt.subplot(421)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['U'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['U'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['U'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['U'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Sva['U'])[:,0],   'cp'  ,label='Sva A',markeredgecolor='c',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Syn['U'])[:,0],   'yd'  ,label='Syn A',markeredgecolor='y',markersize=8,markerfacecolor='none')
ax1.set_ylim(0      , 30 )
plt.legend(loc='best',ncol=2,fontsize=14)
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
fig.suptitle('Storm No.' + str(event+1) + ' spectral fitting from different top sensors at four masts', fontsize=25)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(422)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Dir'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Dir'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Dir'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Dir'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Sva['Dir'])[:,0],   'cp'  ,label='Sva A',markeredgecolor='c',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Syn['Dir'])[:,0],   'yd'  ,label='Syn A',markeredgecolor='y',markersize=8,markerfacecolor='none')
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(423)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iu'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iu'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iu'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iu'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Sva['Iu'])[:,0],   'cp'  ,label='Sva A',markeredgecolor='c',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Syn['Iu'])[:,0],   'yd'  ,label='Syn A',markeredgecolor='y',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Iu_ref,                           'r-'  ,label='$Iu (N400)$',linewidth=2)
plt.ylabel(r'$I_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(424)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANu_sm[:,0],       'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_sm[:,1],       'go'  ,label='Osp1 B' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_sm_Osp2[:,0],  'b^'  ,label='Osp2 A'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_sm_Osp2[:,1],  'k*'  ,label='Osp2 B'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_sm_Sva[:,0],  'cp'  ,label='Sva A'  ,markeredgecolor='c',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_sm_Syn[:,0],  'yd'  ,label='Syn A'  ,markeredgecolor='y',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(425)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iv'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iv'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iv'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iv'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Sva['Iv'])[:,0],   'cp'  ,label='Sva A',markeredgecolor='c',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Syn['Iv'])[:,0],   'yd'  ,label='Syn A',markeredgecolor='y',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Iv_ref,                           'g-'  ,label='$Iv (N400)$',linewidth=2)
plt.ylabel(r'$I_v$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

            
ax1     = plt.subplot(426)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANv_sm[:,0],       'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_sm[:,1],       'go'  ,label='Osp1 B' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_sm_Osp2[:,0],  'b^'  ,label='Osp2 A'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_sm_Osp2[:,1],  'k*'  ,label='Osp2 B'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_sm_Sva[:,0],  'cp'  ,label='Sva A'  ,markeredgecolor='c',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_sm_Syn[:,0],  'yd'  ,label='Syn A'  ,markeredgecolor='y',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Av (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_v$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()          

ax1     = plt.subplot(427)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iw'])[:,0],   'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm['Iw'])[:,1],   'go'  ,label='Osp1 B',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iw'])[:,0],   'b^'  ,label='Osp2 A',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Osp2['Iw'])[:,1],   'k*'  ,label='Osp2 B',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Sva['Iw'])[:,0],   'cp'  ,label='Sva A',markeredgecolor='c',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), np.asarray(data_sm_Syn['Iw'])[:,0],   'yd'  ,label='Syn A',markeredgecolor='y',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Iw_ref,                           'b-'  ,label='$Iw (N400)$',linewidth=2)
plt.ylabel(r'$I_w$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(428)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANw_sm[:,0],       'rs'  ,label='Osp1 A',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_sm[:,1],       'go'  ,label='Osp1 B' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_sm_Osp2[:,0],  'b^'  ,label='Osp2 A'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_sm_Osp2[:,1],  'k*'  ,label='Osp2 B'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_sm_Sva[:,0],  'cp'  ,label='Sva A'  ,markeredgecolor='c',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_sm_Syn[:,0],  'yd'  ,label='Syn A'  ,markeredgecolor='y',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Aw_ref,        'b-'  ,label='$Aw (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_w$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()           

save_tite = 'Storm No.' + str(event+1) + '_spectral fitting sensitivity different top sensors at four masts.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite) 
#%%
# A parameter sensitivity on spectral smoothing

with open(name_wo + ".txt") as json_file:
    data_wo  =  json.load(json_file)
with open(name_30p + ".txt") as json_file:
    data_30p  =  json.load(json_file)
with open(name_60p + ".txt") as json_file:
    data_60p  =  json.load(json_file)    
with open(name_120p + ".txt") as json_file:
    data_120p  =  json.load(json_file)  
    
delta       = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
date_end    = datetime.strptime(data_wo['date_end']   , "%Y-%m-%d %H:%M")
date_start  = datetime.strptime(data_wo['date_start'] , "%Y-%m-%d %H:%M")
num_case    = int(np.round((mdates.date2num(date_end)-mdates.date2num(date_start))/delta))
    
time        = np.asarray(np.linspace(mdates.date2num(date_start),mdates.date2num(date_end)-delta,num_case))

ANu_wo         = np.asarray(data_wo['ANu'])
ANu_n_wo       = np.asarray(data_wo['ANu_n'])
ANu_30p        = np.asarray(data_30p['ANu_30p'])
ANu_n_30p      = np.asarray(data_30p['ANu_n_30p'])
ANu_60p        = np.asarray(data_60p['ANu_60p'])
ANu_n_60p      = np.asarray(data_60p['ANu_n_60p'])
ANu_120p       = np.asarray(data_120p['ANu_120p'])
ANu_n_120p     = np.asarray(data_120p['ANu_n_120p']) 

ANu_wo[ANu_wo==0] = 'nan'       
ANu_n_wo[ANu_n_wo==0] = 'nan' 
ANu_30p[ANu_30p==0] = 'nan'       
ANu_n_30p[ANu_n_30p==0] = 'nan' 
ANu_60p[ANu_60p==0] = 'nan'       
ANu_n_60p[ANu_n_60p==0] = 'nan' 
ANu_120p[ANu_120p==0] = 'nan'       
ANu_n_120p[ANu_n_120p==0] = 'nan' 

ANv_wo         = np.asarray(data_wo['ANv'])
ANv_n_wo       = np.asarray(data_wo['ANv_n'])
ANv_30p        = np.asarray(data_30p['ANv_30p'])
ANv_n_30p      = np.asarray(data_30p['ANv_n_30p'])
ANv_60p        = np.asarray(data_60p['ANv_60p'])
ANv_n_60p      = np.asarray(data_60p['ANv_n_60p'])
ANv_120p       = np.asarray(data_120p['ANv_120p'])
ANv_n_120p     = np.asarray(data_120p['ANv_n_120p']) 

ANv_wo[ANv_wo==0] = 'nan'       
ANv_n_wo[ANv_n_wo==0] = 'nan' 
ANv_30p[ANv_30p==0] = 'nan'       
ANv_n_30p[ANv_n_30p==0] = 'nan' 
ANv_60p[ANv_60p==0] = 'nan'       
ANv_n_60p[ANv_n_60p==0] = 'nan' 
ANv_120p[ANv_120p==0] = 'nan'       
ANv_n_120p[ANv_n_120p==0] = 'nan' 

ANw_wo         = np.asarray(data_wo['ANw'])
ANw_n_wo       = np.asarray(data_wo['ANw_n'])
ANw_30p        = np.asarray(data_30p['ANw_30p'])
ANw_n_30p      = np.asarray(data_30p['ANw_n_30p'])
ANw_60p        = np.asarray(data_60p['ANw_60p'])
ANw_n_60p      = np.asarray(data_60p['ANw_n_60p'])
ANw_120p       = np.asarray(data_120p['ANw_120p'])
ANw_n_120p     = np.asarray(data_120p['ANw_n_120p']) 

ANw_wo[ANw_wo==0] = 'nan'       
ANw_n_wo[ANw_n_wo==0] = 'nan' 
ANw_30p[ANw_30p==0] = 'nan'       
ANw_n_30p[ANw_n_30p==0] = 'nan' 
ANw_60p[ANw_60p==0] = 'nan'       
ANw_n_60p[ANw_n_60p==0] = 'nan' 
ANw_120p[ANw_120p==0] = 'nan'       
ANw_n_120p[ANw_n_120p==0] = 'nan'

z0       = 0.01
Iu_ref   = (1./np.log(48.8/z0))*np.ones(np.shape(time))
Iv_ref   = 0.84*Iu_ref*np.ones(np.shape(time))  # 3/4. default in N400
Iw_ref   = 0.6*Iu_ref *np.ones(np.shape(time))  # 1/2. default in N400

Au_ref   = 6.8*np.ones(np.shape(time))
Av_ref   = 9.4*np.ones(np.shape(time))
Aw_ref   = 6.8*np.ones(np.shape(time))

#%%
# plot the A parameter sensitivity on raw or normalized spectrum
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))

ax1     = plt.subplot(421)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['U'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.set_ylim(0      , 30 )
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
fig.suptitle('Storm No.' + str(event+1) + ' spectral fitting sensitivity on spectral type', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(422)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['Dir'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(423)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['Iu'])[:,0],   'rs'  ,label='$Measured$',markeredgecolor='r',markersize=8)
ax1.plot(mdates.num2date(time), Iu_ref,                           'r-'  ,label='$Iu (N400)$',linewidth=2)
plt.ylabel(r'$I_u$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(424)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANu_60p[:,0]  ,  'rs'  ,label='raw spectrum'  ,markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_n_60p[:,0],  'k*'  ,label='normalized spectrum'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_u$', fontsize=20)
plt.legend(loc='best',ncol=3,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(425)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['Iv'])[:,0],   'gs'  ,label='$w.o. smooth$',markeredgecolor='g',markersize=8)
ax1.plot(mdates.num2date(time), Iv_ref,                           'g-'  ,label='$Iv (N400)$',linewidth=2)
plt.ylabel(r'$I_v$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

            
ax1     = plt.subplot(426)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANv_60p[:,0]  ,  'rs'  ,label='raw spectrum'  ,markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_n_60p[:,0],  'k*'  ,label='normalized spectrum'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Av (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_v$', fontsize=20)
plt.legend(loc='best',ncol=3,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()          

ax1     = plt.subplot(427)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['Iw'])[:,0],   'bs'  ,label='$w.o. smooth$',markeredgecolor='b',markersize=8)
ax1.plot(mdates.num2date(time), Iw_ref,                           'b-'  ,label='$Iw (N400)$',linewidth=2)
plt.ylabel(r'$I_w$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(428)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANw_60p[:,0]  ,  'rs'  ,label='raw spectrum'  ,markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_n_60p[:,0],  'k*'  ,label='normalized spectrum'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Aw_ref,        'b-'  ,label='$Aw (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_w$', fontsize=20)
plt.legend(loc='best',ncol=3,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()           

save_tite = 'Storm No.' + str(event+1) + '_spectral fitting sensitivity on raw or normalized spectra.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite) 


#%%
# plot the A parameter sensitivity on spectral smoothing
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))

ax1     = plt.subplot(421)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['U'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.set_ylim(0      , 30 )
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
fig.suptitle('Storm No.' + str(event+1) + ' spectral fitting sensitivity on spectra smoothing parameter', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(422)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['Dir'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(423)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['Iu'])[:,0],   'rs'  ,label='$Measured$',markeredgecolor='r',markersize=8)
ax1.plot(mdates.num2date(time), Iu_ref,                           'r-'  ,label='$Iu (N400)$',linewidth=2)
plt.ylabel(r'$I_u$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(424)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANu_wo[:,0],   'rs'  ,label='$w.o. smooth$',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_120p[:,0], 'go'  ,label='$120 points$' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_60p[:,0],  'b^'  ,label='$60 points$'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_30p[:,0],  'k*'  ,label='$30 points$'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_u$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(425)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['Iv'])[:,0],   'gs'  ,label='$w.o. smooth$',markeredgecolor='g',markersize=8)
ax1.plot(mdates.num2date(time), Iv_ref,                           'g-'  ,label='$Iv (N400)$',linewidth=2)
plt.ylabel(r'$I_v$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

            
ax1     = plt.subplot(426)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANv_wo[:,0],   'rs'  ,label='$w.o. smooth$',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_120p[:,0], 'go'  ,label='$120 points$' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_60p[:,0],  'b^'  ,label='$60 points$'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_30p[:,0],  'k*'  ,label='$30 points$'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Av (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_v$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()          

ax1     = plt.subplot(427)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_wo['Iw'])[:,0],   'bs'  ,label='$w.o. smooth$',markeredgecolor='b',markersize=8)
ax1.plot(mdates.num2date(time), Iw_ref,                           'b-'  ,label='$Iw (N400)$',linewidth=2)
plt.ylabel(r'$I_w$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(428)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANw_wo[:,0],   'rs'  ,label='$w.o. smooth$',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_120p[:,0], 'go'  ,label='$120 points$' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_60p[:,0],  'b^'  ,label='$60 points$'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_30p[:,0],  'k*'  ,label='$30 points$'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Aw_ref,        'b-'  ,label='$Aw (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_w$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()           

save_tite = 'Storm No.' + str(event+1) + '_spectral fitting sensitivity on spectra smoothing parameter.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite) 

#%% 
# A parameter sensitivity on length of time series

with open(name_10min + ".txt") as json_file:
    data_10min  =  json.load(json_file)  
with open(name_20min + ".txt") as json_file:
    data_20min  =  json.load(json_file) 
with open(name_1h + ".txt") as json_file:
    data_1h  =  json.load(json_file)  
with open(name_3h + ".txt") as json_file:
    data_3h  =  json.load(json_file) 

delta_10min      = mdates.date2num(datetime(2,1,1,1,10))-mdates.date2num(datetime(2,1,1,1,0))
date_end_10min   = datetime.strptime(data_10min['date_end']   , "%Y-%m-%d %H:%M")
date_start_10min = datetime.strptime(data_10min['date_start'] , "%Y-%m-%d %H:%M")
num_case_10min   = int(np.round((mdates.date2num(date_end_10min)-mdates.date2num(date_start_10min))/delta_10min))   
time_10min       = np.asarray(np.linspace(mdates.date2num(date_start_10min),mdates.date2num(date_end_10min)-delta_10min,num_case_10min))

delta_20min      = mdates.date2num(datetime(2,1,1,1,20))-mdates.date2num(datetime(2,1,1,1,0))
date_end_20min   = datetime.strptime(data_20min['date_end']   , "%Y-%m-%d %H:%M")
date_start_20min = datetime.strptime(data_20min['date_start'] , "%Y-%m-%d %H:%M")
num_case_20min   = int(np.round((mdates.date2num(date_end_20min)-mdates.date2num(date_start_20min))/delta_20min))   
time_20min       = np.asarray(np.linspace(mdates.date2num(date_start_20min),mdates.date2num(date_end_20min)-delta_20min,num_case_20min))

delta_1h      = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
date_end_1h   = datetime.strptime(data_1h['date_end']   , "%Y-%m-%d %H:%M")
date_start_1h = datetime.strptime(data_1h['date_start'] , "%Y-%m-%d %H:%M")
num_case_1h   = int(np.round((mdates.date2num(date_end_1h)-mdates.date2num(date_start_1h))/delta_1h))   
time_1h       = np.asarray(np.linspace(mdates.date2num(date_start_1h),mdates.date2num(date_end_1h)-delta_1h,num_case_1h))

delta_3h      = mdates.date2num(datetime(2,1,1,4,0))-mdates.date2num(datetime(2,1,1,1,0))
date_end_3h   = datetime.strptime(data_3h['date_end']   , "%Y-%m-%d %H:%M")
date_start_3h = datetime.strptime(data_3h['date_start'] , "%Y-%m-%d %H:%M")
num_case_3h   = int(np.round((mdates.date2num(date_end_3h)-mdates.date2num(date_start_3h))/delta_3h))   
time_3h       = np.asarray(np.linspace(mdates.date2num(date_start_3h),mdates.date2num(date_end_3h)-delta_3h,num_case_3h))

ANu_10min         = np.asarray(data_10min['ANu_10min'])
ANu_n_10min       = np.asarray(data_10min['ANu_n_10min'])
ANu_20min        = np.asarray(data_20min['ANu_20min'])
ANu_n_20min      = np.asarray(data_20min['ANu_n_20min'])
ANu_1h        = np.asarray(data_1h['ANu_1h'])
ANu_n_1h      = np.asarray(data_1h['ANu_n_1h'])
ANu_3h       = np.asarray(data_3h['ANu_3h'])
ANu_n_3h     = np.asarray(data_3h['ANu_n_3h']) 

ANu_10min[ANu_10min==0] = 'nan'       
ANu_n_10min[ANu_n_10min==0] = 'nan' 
ANu_20min[ANu_20min==0] = 'nan'       
ANu_n_20min[ANu_n_20min==0] = 'nan' 
ANu_1h[ANu_1h==0] = 'nan'       
ANu_n_1h[ANu_n_1h==0] = 'nan' 
ANu_3h[ANu_3h==0] = 'nan'       
ANu_n_3h[ANu_n_3h==0] = 'nan' 

ANv_10min         = np.asarray(data_10min['ANv_10min'])
ANv_n_10min       = np.asarray(data_10min['ANv_n_10min'])
ANv_20min        = np.asarray(data_20min['ANv_20min'])
ANv_n_20min      = np.asarray(data_20min['ANv_n_20min'])
ANv_1h        = np.asarray(data_1h['ANv_1h'])
ANv_n_1h      = np.asarray(data_1h['ANv_n_1h'])
ANv_3h       = np.asarray(data_3h['ANv_3h'])
ANv_n_3h     = np.asarray(data_3h['ANv_n_3h']) 

ANv_10min[ANv_10min==0] = 'nan'       
ANv_n_10min[ANv_n_10min==0] = 'nan' 
ANv_20min[ANv_20min==0] = 'nan'       
ANv_n_20min[ANv_n_20min==0] = 'nan' 
ANv_1h[ANv_1h==0] = 'nan'       
ANv_n_1h[ANv_n_1h==0] = 'nan' 
ANv_3h[ANv_3h==0] = 'nan'       
ANv_n_3h[ANv_n_3h==0] = 'nan' 

ANw_10min         = np.asarray(data_10min['ANw_10min'])
ANw_n_10min       = np.asarray(data_10min['ANw_n_10min'])
ANw_20min        = np.asarray(data_20min['ANw_20min'])
ANw_n_20min      = np.asarray(data_20min['ANw_n_20min'])
ANw_1h        = np.asarray(data_1h['ANw_1h'])
ANw_n_1h      = np.asarray(data_1h['ANw_n_1h'])
ANw_3h       = np.asarray(data_3h['ANw_3h'])
ANw_n_3h     = np.asarray(data_3h['ANw_n_3h']) 

ANw_10min[ANw_10min==0] = 'nan'       
ANw_n_10min[ANw_n_10min==0] = 'nan' 
ANw_20min[ANw_20min==0] = 'nan'       
ANw_n_20min[ANw_n_20min==0] = 'nan' 
ANw_1h[ANw_1h==0] = 'nan'       
ANw_n_1h[ANw_n_1h==0] = 'nan' 
ANw_3h[ANw_3h==0] = 'nan'       
ANw_n_3h[ANw_n_3h==0] = 'nan'

z0       = 0.01
Iu_ref   = (1./np.log(48.8/z0))*np.ones(np.shape(time_10min))
Iv_ref   = 0.84*Iu_ref*np.ones(np.shape(time_10min))  # 3/4. default in N400
Iw_ref   = 0.6*Iu_ref *np.ones(np.shape(time_10min))  # 1/2. default in N400

Au_ref   = 6.8*np.ones(np.shape(time_10min))
Av_ref   = 9.4*np.ones(np.shape(time_10min))
Aw_ref   = 6.8*np.ones(np.shape(time_10min))
    

#%% 
# plot the A parameter sensitivity on length of time series
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))

ax1     = plt.subplot(421)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time_10min), np.asarray(data_10min['U_10min'])[:,0],   'k*'  ,label='10 min',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_20min), np.asarray(data_20min['U_20min'])[:,0],   'b^'  ,label='20 min',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_1h),    np.asarray(data_1h['U_1h'])[:,0],         'go'  ,label='1 h',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_3h),    np.asarray(data_3h['U_3h'])[:,0],         'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.set_ylim(0      , 30 )
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
plt.legend(loc='best',ncol=4,fontsize=14)
fig.suptitle('Storm No.' + str(event+1) + ' spectral fitting sensitivity on length of analyzed time series', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(422)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time_10min), np.asarray(data_10min['Dir_10min'])[:,0],   'k*'  ,label='10 min',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_20min), np.asarray(data_20min['Dir_20min'])[:,0],   'b^'  ,label='20 min',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_1h),    np.asarray(data_1h['Dir_1h'])[:,0],         'go'  ,label='1 h',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_3h),    np.asarray(data_3h['Dir_3h'])[:,0],         'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.legend(loc='best',ncol=4,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(423)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time_10min), np.asarray(data_10min['Iu_10min'])[:,0],   'k*'  ,label='10 min',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_20min), np.asarray(data_20min['Iu_20min'])[:,0],   'b^'  ,label='20 min',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_1h),    np.asarray(data_1h['Iu_1h'])[:,0],         'go'  ,label='1 h',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_3h),    np.asarray(data_3h['Iu_3h'])[:,0],         'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_10min), Iu_ref,                                    'r-'  ,label='Iu (N400)',linewidth=2)
plt.ylabel(r'$I_u$', fontsize=20)
ax1.set_ylim(0      , 0.2 )
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(424)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time_10min), ANu_10min[:,0],  'k*'  ,label='10 min'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_20min), ANu_20min[:,0],  'b^'  ,label='20 min'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_1h), ANu_1h[:,0],        'go'  ,label='1 h' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_3h), ANu_3h[:,0],        'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_10min), Au_ref,          'r-'  ,label='Au (N400)'  ,linewidth=2)
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_u$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(425)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time_10min), np.asarray(data_10min['Iv_10min'])[:,0],   'k*'  ,label='10 min',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_20min), np.asarray(data_20min['Iv_20min'])[:,0],   'b^'  ,label='20 min',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_1h),    np.asarray(data_1h['Iv_1h'])[:,0],         'go'  ,label='1 h',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_3h),    np.asarray(data_3h['Iv_3h'])[:,0],         'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_10min), Iv_ref,                                    'g-'  ,label='Iv (N400)',linewidth=2)
plt.ylabel(r'$I_v$', fontsize=20)
ax1.set_ylim(0      , 0.2 )
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

            
ax1     = plt.subplot(426)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time_10min), ANv_10min[:,0],  'k*'  ,label='10 min'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_20min), ANv_20min[:,0],  'b^'  ,label='20 min'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_1h), ANv_1h[:,0],        'go'  ,label='1 h' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_3h), ANv_3h[:,0],        'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_10min), Av_ref,          'g-'  ,label='Av (N400)'  ,linewidth=2)
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_v$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()          

ax1     = plt.subplot(427)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time_10min), np.asarray(data_10min['Iw_10min'])[:,0],   'k*'  ,label='10 min',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_20min), np.asarray(data_20min['Iw_20min'])[:,0],   'b^'  ,label='20 min',markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_1h),    np.asarray(data_1h['Iw_1h'])[:,0],         'go'  ,label='1 h',markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_3h),    np.asarray(data_3h['Iw_3h'])[:,0],         'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_10min), Iw_ref,                                    'b-'  ,label='Iw (N400)',linewidth=2)
plt.ylabel(r'$I_w$', fontsize=20)
ax1.set_ylim(0      , 0.2 )
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(428)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time_10min), ANw_10min[:,0],  'k*'  ,label='10 min'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_20min), ANw_20min[:,0],  'b^'  ,label='20 min'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_1h), ANw_1h[:,0],        'go'  ,label='1 h' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_3h), ANw_3h[:,0],        'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time_10min), Aw_ref,          'b-'  ,label='Aw (N400)'  ,linewidth=2)
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_w$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()           

save_tite = 'Storm No.' + str(event+1) + '_spectral fitting sensitivity on length of analyzed time series.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite)     



#%%
# A parameter sensitivity on pwelch segments   
with open(name_1seg + ".txt") as json_file:
    data_1seg  =  json.load(json_file)  
with open(name_3seg + ".txt") as json_file:
    data_3seg  =  json.load(json_file) 
with open(name_6seg + ".txt") as json_file:
    data_6seg  =  json.load(json_file)  
with open(name_12seg + ".txt") as json_file:
    data_12seg  =  json.load(json_file)    

delta       = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
date_end    = datetime.strptime(data_6seg['date_end']   , "%Y-%m-%d %H:%M")
date_start  = datetime.strptime(data_6seg['date_start'] , "%Y-%m-%d %H:%M")
num_case    = int(np.round((mdates.date2num(date_end)-mdates.date2num(date_start))/delta))    
time        = np.asarray(np.linspace(mdates.date2num(date_start),mdates.date2num(date_end)-delta,num_case))

ANu_1seg         = np.asarray(data_1seg['ANu_1seg'])
ANu_n_1seg       = np.asarray(data_1seg['ANu_n_1seg'])
ANu_3seg        = np.asarray(data_3seg['ANu_3seg'])
ANu_n_3seg      = np.asarray(data_3seg['ANu_n_3seg'])
ANu_6seg        = np.asarray(data_6seg['ANu_6seg'])
ANu_n_6seg      = np.asarray(data_6seg['ANu_n_6seg'])
ANu_12seg       = np.asarray(data_12seg['ANu_12seg'])
ANu_n_12seg     = np.asarray(data_12seg['ANu_n_12seg']) 

ANu_1seg[ANu_1seg==0] = 'nan'       
ANu_n_1seg[ANu_n_1seg==0] = 'nan' 
ANu_3seg[ANu_3seg==0] = 'nan'       
ANu_n_3seg[ANu_n_3seg==0] = 'nan' 
ANu_6seg[ANu_6seg==0] = 'nan'       
ANu_n_6seg[ANu_n_6seg==0] = 'nan' 
ANu_12seg[ANu_12seg==0] = 'nan'       
ANu_n_12seg[ANu_n_12seg==0] = 'nan' 

ANv_1seg         = np.asarray(data_1seg['ANv_1seg'])
ANv_n_1seg       = np.asarray(data_1seg['ANv_n_1seg'])
ANv_3seg        = np.asarray(data_3seg['ANv_3seg'])
ANv_n_3seg      = np.asarray(data_3seg['ANv_n_3seg'])
ANv_6seg        = np.asarray(data_6seg['ANv_6seg'])
ANv_n_6seg      = np.asarray(data_6seg['ANv_n_6seg'])
ANv_12seg       = np.asarray(data_12seg['ANv_12seg'])
ANv_n_12seg     = np.asarray(data_12seg['ANv_n_12seg']) 

ANv_1seg[ANv_1seg==0] = 'nan'       
ANv_n_1seg[ANv_n_1seg==0] = 'nan' 
ANv_3seg[ANv_3seg==0] = 'nan'       
ANv_n_3seg[ANv_n_3seg==0] = 'nan' 
ANv_6seg[ANv_6seg==0] = 'nan'       
ANv_n_6seg[ANv_n_6seg==0] = 'nan' 
ANv_12seg[ANv_12seg==0] = 'nan'       
ANv_n_12seg[ANv_n_12seg==0] = 'nan' 

ANw_1seg         = np.asarray(data_1seg['ANw_1seg'])
ANw_n_1seg       = np.asarray(data_1seg['ANw_n_1seg'])
ANw_3seg        = np.asarray(data_3seg['ANw_3seg'])
ANw_n_3seg      = np.asarray(data_3seg['ANw_n_3seg'])
ANw_6seg        = np.asarray(data_6seg['ANw_6seg'])
ANw_n_6seg      = np.asarray(data_6seg['ANw_n_6seg'])
ANw_12seg       = np.asarray(data_12seg['ANw_12seg'])
ANw_n_12seg     = np.asarray(data_12seg['ANw_n_12seg']) 

ANw_1seg[ANw_1seg==0] = 'nan'       
ANw_n_1seg[ANw_n_1seg==0] = 'nan' 
ANw_3seg[ANw_3seg==0] = 'nan'       
ANw_n_3seg[ANw_n_3seg==0] = 'nan' 
ANw_6seg[ANw_6seg==0] = 'nan'       
ANw_n_6seg[ANw_n_6seg==0] = 'nan' 
ANw_12seg[ANw_12seg==0] = 'nan'       
ANw_n_12seg[ANw_n_12seg==0] = 'nan'

z0       = 0.01
Iu_ref   = (1./np.log(48.8/z0))*np.ones(np.shape(time))
Iv_ref   = 0.84*Iu_ref*np.ones(np.shape(time))  # 3/4. default in N400
Iw_ref   = 0.6*Iu_ref *np.ones(np.shape(time))  # 1/2. default in N400

Au_ref   = 6.8*np.ones(np.shape(time))
Av_ref   = 9.4*np.ones(np.shape(time))
Aw_ref   = 6.8*np.ones(np.shape(time))


#%%
# plot the A parameter sensitivity on pwelch segments
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))

ax1     = plt.subplot(421)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_1seg['U'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.set_ylim(0      , 30 )
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
fig.suptitle('Storm No.' + str(event+1) + ' spectral fitting sensitivity on number of pwelch segments', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(422)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_1seg['Dir'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(423)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_1seg['Iu'])[:,0],   'rs'  ,label='$Measured$',markeredgecolor='r',markersize=8)
ax1.plot(mdates.num2date(time), Iu_ref,                           'r-'  ,label='$Iu (N400)$',linewidth=2)
plt.ylabel(r'$I_u$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(424)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANu_12seg[:,0], 'go'  ,label='12 segments' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_6seg[:,0],  'b^'  ,label='6 segments'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_3seg[:,0],  'k*'  ,label='3 segments'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_1seg[:,0],   'rs'  ,label='1 segments',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_u$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(425)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_1seg['Iv'])[:,0],   'gs'  ,label='$w.o. smooth$',markeredgecolor='g',markersize=8)
ax1.plot(mdates.num2date(time), Iv_ref,                           'g-'  ,label='$Iv (N400)$',linewidth=2)
plt.ylabel(r'$I_v$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

            
ax1     = plt.subplot(426)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANv_12seg[:,0], 'go'  ,label='12 segments' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_6seg[:,0],  'b^'  ,label='6 segments'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_3seg[:,0],  'k*'  ,label='3 segments'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_1seg[:,0],   'rs'  ,label='1 segments',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Av (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_v$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()          

ax1     = plt.subplot(427)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_1seg['Iw'])[:,0],   'bs'  ,label='$w.o. smooth$',markeredgecolor='b',markersize=8)
ax1.plot(mdates.num2date(time), Iw_ref,                           'b-'  ,label='$Iw (N400)$',linewidth=2)
plt.ylabel(r'$I_w$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(428)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANw_12seg[:,0], 'go'  ,label='12 segments' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_6seg[:,0],  'b^'  ,label='6 segments'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_3seg[:,0],  'k*'  ,label='3 segments'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_1seg[:,0],   'rs'  ,label='1 segments',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Aw_ref,        'b-'  ,label='$Aw (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_w$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()           

save_tite = 'Storm No.' + str(event+1) + '_spectral fitting sensitivity on number of pwelch segments.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite)     
    
    
#%%
# A parameter sensitivity on moving averaging parameter 

with open(name_5mi + ".txt") as json_file:
    data_5mi  =  json.load(json_file) 
with open(name_10mi + ".txt") as json_file:
    data_10mi  =  json.load(json_file)  
with open(name_20mi + ".txt") as json_file:
    data_20mi  =  json.load(json_file)     
with open(name_60mi + ".txt") as json_file:
    data_60mi  =  json.load(json_file)  
    
delta       = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
date_end    = datetime.strptime(data_5mi['date_end']   , "%Y-%m-%d %H:%M")
date_start  = datetime.strptime(data_5mi['date_start'] , "%Y-%m-%d %H:%M")
num_case    = int(np.round((mdates.date2num(date_end)-mdates.date2num(date_start))/delta))    
time        = np.asarray(np.linspace(mdates.date2num(date_start),mdates.date2num(date_end)-delta,num_case))

ANu_5mi         = np.asarray(data_5mi['ANu_5min'])
ANu_n_5mi       = np.asarray(data_5mi['ANu_n_5min'])
ANu_10mi        = np.asarray(data_10mi['ANu_10min'])
ANu_n_10mi      = np.asarray(data_10mi['ANu_n_10min'])
ANu_20mi        = np.asarray(data_20mi['ANu_20min'])
ANu_n_20mi      = np.asarray(data_20mi['ANu_n_20min'])
ANu_60mi       = np.asarray(data_60mi['ANu_60min'])
ANu_n_60mi     = np.asarray(data_60mi['ANu_n_60min']) 

ANu_5mi[ANu_5mi==0] = 'nan'       
ANu_n_5mi[ANu_n_5mi==0] = 'nan' 
ANu_10mi[ANu_10mi==0] = 'nan'       
ANu_n_10mi[ANu_n_10mi==0] = 'nan' 
ANu_20mi[ANu_20mi==0] = 'nan'       
ANu_n_20mi[ANu_n_20mi==0] = 'nan' 
ANu_60mi[ANu_60mi==0] = 'nan'       
ANu_n_60mi[ANu_n_60mi==0] = 'nan' 

ANv_5mi         = np.asarray(data_5mi['ANv_5min'])
ANv_n_5mi       = np.asarray(data_5mi['ANv_n_5min'])
ANv_10mi        = np.asarray(data_10mi['ANv_10min'])
ANv_n_10mi      = np.asarray(data_10mi['ANv_n_10min'])
ANv_20mi        = np.asarray(data_20mi['ANv_20min'])
ANv_n_20mi      = np.asarray(data_20mi['ANv_n_20min'])
ANv_60mi       = np.asarray(data_60mi['ANv_60min'])
ANv_n_60mi     = np.asarray(data_60mi['ANv_n_60min']) 

ANv_5mi[ANv_5mi==0] = 'nan'       
ANv_n_5mi[ANv_n_5mi==0] = 'nan' 
ANv_10mi[ANv_10mi==0] = 'nan'       
ANv_n_10mi[ANv_n_10mi==0] = 'nan' 
ANv_20mi[ANv_20mi==0] = 'nan'       
ANv_n_20mi[ANv_n_20mi==0] = 'nan' 
ANv_60mi[ANv_60mi==0] = 'nan'       
ANv_n_60mi[ANv_n_60mi==0] = 'nan' 

ANw_5mi         = np.asarray(data_5mi['ANw_5min'])
ANw_n_5mi       = np.asarray(data_5mi['ANw_n_5min'])
ANw_10mi        = np.asarray(data_10mi['ANw_10min'])
ANw_n_10mi      = np.asarray(data_10mi['ANw_n_10min'])
ANw_20mi        = np.asarray(data_20mi['ANw_20min'])
ANw_n_20mi      = np.asarray(data_20mi['ANw_n_20min'])
ANw_60mi       = np.asarray(data_60mi['ANw_60min'])
ANw_n_60mi     = np.asarray(data_60mi['ANw_n_60min']) 

ANw_5mi[ANw_5mi==0] = 'nan'       
ANw_n_5mi[ANw_n_5mi==0] = 'nan' 
ANw_10mi[ANw_10mi==0] = 'nan'       
ANw_n_10mi[ANw_n_10mi==0] = 'nan' 
ANw_20mi[ANw_20mi==0] = 'nan'       
ANw_n_20mi[ANw_n_20mi==0] = 'nan' 
ANw_60mi[ANw_60mi==0] = 'nan'       
ANw_n_60mi[ANw_n_60mi==0] = 'nan'

z0       = 0.01
Iu_ref   = (1./np.log(48.8/z0))*np.ones(np.shape(time))
Iv_ref   = 0.84*Iu_ref*np.ones(np.shape(time))  # 3/4. default in N400
Iw_ref   = 0.6*Iu_ref *np.ones(np.shape(time))  # 1/2. default in N400

Au_ref   = 6.8*np.ones(np.shape(time))
Av_ref   = 9.4*np.ones(np.shape(time))
Aw_ref   = 6.8*np.ones(np.shape(time))


#%%
# plot the A parameter sensitivity on pwelch segments
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))

ax1     = plt.subplot(421)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_5mi['U'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.set_ylim(0      , 30 )
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
fig.suptitle('Storm No.' + str(event+1) + ' spectral fitting sensitivity on moving averaging parameter', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(422)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_5mi['Dir'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(423)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_5mi['Iu'])[:,0],   'rs'  ,label='$Measured$',markeredgecolor='r',markersize=8)
ax1.plot(mdates.num2date(time), Iu_ref,                           'r-'  ,label='$Iu (N400)$',linewidth=2)
plt.ylabel(r'$I_u$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(424)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANu_60mi[:,0], 'go'  ,label='60 mins' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_20mi[:,0],  'b^'  ,label='20 mins'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_10mi[:,0],  'k*'  ,label='10 mins'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_5mi[:,0],   'rs'  ,label='5 mins',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_u$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(425)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_5mi['Iv'])[:,0],   'gs'  ,label='$w.o. smooth$',markeredgecolor='g',markersize=8)
ax1.plot(mdates.num2date(time), Iv_ref,                           'g-'  ,label='$Iv (N400)$',linewidth=2)
plt.ylabel(r'$I_v$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

            
ax1     = plt.subplot(426)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANv_60mi[:,0], 'go'  ,label='60 mins' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_20mi[:,0],  'b^'  ,label='20 mins'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_10mi[:,0],  'k*'  ,label='10 mins'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_5mi[:,0],   'rs'  ,label='5 mins',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Av (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_v$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()          

ax1     = plt.subplot(427)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_5mi['Iw'])[:,0],   'bs'  ,label='$w.o. smooth$',markeredgecolor='b',markersize=8)
ax1.plot(mdates.num2date(time), Iw_ref,                           'b-'  ,label='$Iw (N400)$',linewidth=2)
plt.ylabel(r'$I_w$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(428)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANw_60mi[:,0], 'go'  ,label='60 mins' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_20mi[:,0],  'b^'  ,label='20 mins'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_10mi[:,0],  'k*'  ,label='10 mins'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_5mi[:,0],   'rs'  ,label='5 mins',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Aw_ref,        'b-'  ,label='$Aw (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_w$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()           

save_tite = 'Storm No.' + str(event+1) + '_spectral fitting sensitivity on moving averaging parameter.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite)     


#%% 
# A parameter sensitivity on fitted frequency range    
with open(name_fu + ".txt") as json_file:
    data_fu  =  json.load(json_file) 
with open(name_lp + ".txt") as json_file:
    data_lp  =  json.load(json_file)  
with open(name_hp + ".txt") as json_file:
    data_hp  =  json.load(json_file)     
with open(name_bp + ".txt") as json_file:
    data_bp  =  json.load(json_file)   
    
delta       = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
date_end    = datetime.strptime(data_bp['date_end']   , "%Y-%m-%d %H:%M")
date_start  = datetime.strptime(data_bp['date_start'] , "%Y-%m-%d %H:%M")
num_case    = int(np.round((mdates.date2num(date_end)-mdates.date2num(date_start))/delta))    
time        = np.asarray(np.linspace(mdates.date2num(date_start),mdates.date2num(date_end)-delta,num_case))

ANu_fu         = np.asarray(data_fu['ANu_wo'])
ANu_n_fu       = np.asarray(data_fu['ANu_n_wo'])
ANu_lp        = np.asarray(data_lp['ANu_lp'])
ANu_n_lp      = np.asarray(data_lp['ANu_n_lp'])
ANu_hp        = np.asarray(data_hp['ANu_hp'])
ANu_n_hp      = np.asarray(data_hp['ANu_n_hp'])
ANu_bp       = np.asarray(data_bp['ANu_bp'])
ANu_n_bp     = np.asarray(data_bp['ANu_n_bp']) 

ANu_fu[ANu_fu==0] = 'nan'       
ANu_n_fu[ANu_n_fu==0] = 'nan' 
ANu_lp[ANu_lp==0] = 'nan'       
ANu_n_lp[ANu_n_lp==0] = 'nan' 
ANu_hp[ANu_hp==0] = 'nan'       
ANu_n_hp[ANu_n_hp==0] = 'nan' 
ANu_bp[ANu_bp==0] = 'nan'       
ANu_n_bp[ANu_n_bp==0] = 'nan' 

ANv_fu         = np.asarray(data_fu['ANv_wo'])
ANv_n_fu       = np.asarray(data_fu['ANv_n_wo'])
ANv_lp        = np.asarray(data_lp['ANv_lp'])
ANv_n_lp      = np.asarray(data_lp['ANv_n_lp'])
ANv_hp        = np.asarray(data_hp['ANv_hp'])
ANv_n_hp      = np.asarray(data_hp['ANv_n_hp'])
ANv_bp       = np.asarray(data_bp['ANv_bp'])
ANv_n_bp     = np.asarray(data_bp['ANv_n_bp']) 

ANv_fu[ANv_fu==0] = 'nan'       
ANv_n_fu[ANv_n_fu==0] = 'nan' 
ANv_lp[ANv_lp==0] = 'nan'       
ANv_n_lp[ANv_n_lp==0] = 'nan' 
ANv_hp[ANv_hp==0] = 'nan'       
ANv_n_hp[ANv_n_hp==0] = 'nan' 
ANv_bp[ANv_bp==0] = 'nan'       
ANv_n_bp[ANv_n_bp==0] = 'nan' 

ANw_fu         = np.asarray(data_fu['ANw_wo'])
ANw_n_fu       = np.asarray(data_fu['ANw_n_wo'])
ANw_lp        = np.asarray(data_lp['ANw_lp'])
ANw_n_lp      = np.asarray(data_lp['ANw_n_lp'])
ANw_hp        = np.asarray(data_hp['ANw_hp'])
ANw_n_hp      = np.asarray(data_hp['ANw_n_hp'])
ANw_bp       = np.asarray(data_bp['ANw_bp'])
ANw_n_bp     = np.asarray(data_bp['ANw_n_bp']) 

ANw_fu[ANw_fu==0] = 'nan'       
ANw_n_fu[ANw_n_fu==0] = 'nan' 
ANw_lp[ANw_lp==0] = 'nan'       
ANw_n_lp[ANw_n_lp==0] = 'nan' 
ANw_hp[ANw_hp==0] = 'nan'       
ANw_n_hp[ANw_n_hp==0] = 'nan' 
ANw_bp[ANw_bp==0] = 'nan'       
ANw_n_bp[ANw_n_bp==0] = 'nan'

z0       = 0.01
Iu_ref   = (1./np.log(48.8/z0))*np.ones(np.shape(time))
Iv_ref   = 0.84*Iu_ref*np.ones(np.shape(time))  # 3/4. default in N400
Iw_ref   = 0.6*Iu_ref *np.ones(np.shape(time))  # 1/2. default in N400

Au_ref   = 6.8*np.ones(np.shape(time))
Av_ref   = 9.4*np.ones(np.shape(time))
Aw_ref   = 6.8*np.ones(np.shape(time))


#%%
# plot the A parameter sensitivity on fitted frequency range
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))

ax1     = plt.subplot(421)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_fu['U'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.set_ylim(0      , 30 )
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
fig.suptitle('Storm No.' + str(event+1) + ' spectral fitting sensitivity on frequency range', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(422)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_fu['Dir'])[:,0],   'ks'  ,label='$w.o. smooth$',markeredgecolor='k',markersize=8,markerfacecolor='none')
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(423)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_fu['Iu'])[:,0],   'rs'  ,label='$Measured$',markeredgecolor='r',markersize=8)
ax1.plot(mdates.num2date(time), Iu_ref,                           'r-'  ,label='$Iu (N400)$',linewidth=2)
plt.ylabel(r'$I_u$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(424)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANu_bp[:,0], 'go'  ,label='band pass' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_hp[:,0],  'b^'  ,label='high pass'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_lp[:,0],  'k*'  ,label='low pass'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANu_fu[:,0],   'rs'  ,label='full range',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_u$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(425)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_fu['Iv'])[:,0],   'gs'  ,label='$w.o. smooth$',markeredgecolor='g',markersize=8)
ax1.plot(mdates.num2date(time), Iv_ref,                           'g-'  ,label='$Iv (N400)$',linewidth=2)
plt.ylabel(r'$I_v$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

            
ax1     = plt.subplot(426)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANv_bp[:,0], 'go'  ,label='band pass' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_hp[:,0],  'b^'  ,label='high pass'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_lp[:,0],  'k*'  ,label='low pass'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANv_fu[:,0],   'rs'  ,label='full range',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Av (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_v$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()          

ax1     = plt.subplot(427)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), np.asarray(data_fu['Iw'])[:,0],   'bs'  ,label='$w.o. smooth$',markeredgecolor='b',markersize=8)
ax1.plot(mdates.num2date(time), Iw_ref,                           'b-'  ,label='$Iw (N400)$',linewidth=2)
plt.ylabel(r'$I_w$', fontsize=20)
plt.legend(loc='best',ncol=2,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax1     = plt.subplot(428)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(time), ANw_bp[:,0], 'go'  ,label='band pass' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_hp[:,0],  'b^'  ,label='high pass'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_lp[:,0],  'k*'  ,label='low pass'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), ANw_fu[:,0],   'rs'  ,label='full range',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax1.plot(mdates.num2date(time), Aw_ref,        'b-'  ,label='$Aw (N400)$'  ,linewidth=2)
datemin = np.datetime64(mdates.num2date(time)[0], 'm')
datemax = np.datetime64(mdates.num2date(time)[-1], 'm') + np.timedelta64(1, 'm')
ax1.set_ylim(0      , 50 )
plt.ylabel(r'$A_w$', fontsize=20)
plt.legend(loc='best',ncol=5,fontsize=14)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()           

save_tite = 'Storm No.' + str(event+1) + '_spectral fitting sensitivity on frequency range.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite)         
    
    
#%%
plt.close("all") 
fig, ax = plt.subplots(3, 6, figsize=(30, 12))      

##
# format the ticks
locator = mdates.AutoDateLocator(minticks=2, maxticks=5)
formatter = mdates.ConciseDateFormatter(locator)


Au_ref   = 6.8*np.ones(np.shape(time_10min))
Av_ref   = 9.4*np.ones(np.shape(time_10min))
Aw_ref   = 6.8*np.ones(np.shape(time_10min))
##
ax[0,5].xaxis.set_major_locator(locator)
ax[0,5].xaxis.set_major_formatter(formatter)
ax[0,5].plot(mdates.num2date(time_10min), ANu_10min[:,0],  'k*'  ,label='10 min'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[0,5].plot(mdates.num2date(time_20min), ANu_20min[:,0],  'b^'  ,label='20 min'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[0,5].plot(mdates.num2date(time_1h), ANu_1h[:,0],        'go'  ,label='1 h' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[0,5].plot(mdates.num2date(time_3h), ANu_3h[:,0],        'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[0,5].plot(mdates.num2date(time_10min), Au_ref,          'r-'  ,linewidth=2)
ax[0,5].set_ylim(0      , 50 )
ax[0,5].set_title('Length of time series', fontsize=20)
ax[0,5].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[0,5].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[0,5].tick_params(axis='both', labelsize=16)
ax[0,5].minorticks_on()

ax[1,5].xaxis.set_major_locator(locator)
ax[1,5].xaxis.set_major_formatter(formatter)
ax[1,5].plot(mdates.num2date(time_10min), ANv_10min[:,0],  'k*'  ,label='10 min'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[1,5].plot(mdates.num2date(time_20min), ANv_20min[:,0],  'b^'  ,label='20 min'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[1,5].plot(mdates.num2date(time_1h), ANv_1h[:,0],        'go'  ,label='1 h' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[1,5].plot(mdates.num2date(time_3h), ANv_3h[:,0],        'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[1,5].plot(mdates.num2date(time_10min), Av_ref,          'g-'  ,linewidth=2)
ax[1,5].set_ylim(0      , 50 )
ax[1,5].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[1,5].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[1,5].tick_params(axis='both', labelsize=16)
ax[1,5].minorticks_on()

ax[2,5].xaxis.set_major_locator(locator)
ax[2,5].xaxis.set_major_formatter(formatter)
ax[2,5].plot(mdates.num2date(time_10min), ANw_10min[:,0],  'k*'  ,label='10 min'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[2,5].plot(mdates.num2date(time_20min), ANw_20min[:,0],  'b^'  ,label='20 min'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[2,5].plot(mdates.num2date(time_1h), ANw_1h[:,0],        'go'  ,label='1 h' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[2,5].plot(mdates.num2date(time_3h), ANw_3h[:,0],        'rs'  ,label='3 h',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[2,5].plot(mdates.num2date(time_10min), Aw_ref,          'b-'  ,linewidth=2)
ax[2,5].set_ylim(0      , 50 )
ax[2,5].legend(loc='best',ncol=1,fontsize=14)
ax[2,5].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[2,5].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[2,5].tick_params(axis='both', labelsize=16)
ax[2,5].minorticks_on()

##
Au_ref   = 6.8*np.ones(np.shape(time))
Av_ref   = 9.4*np.ones(np.shape(time))
Aw_ref   = 6.8*np.ones(np.shape(time))
locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
formatter = mdates.ConciseDateFormatter(locator)

ax[0,0].xaxis.set_major_locator(locator)
ax[0,0].xaxis.set_major_formatter(formatter)
ax[0,0].plot(mdates.num2date(time), ANu_60p[:,0]  ,  'rs'  ,label='raw spectrum'  ,markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[0,0].plot(mdates.num2date(time), ANu_n_60p[:,0],  'k*'  ,label='normalized spectrum'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[0,0].plot(mdates.num2date(time), Au_ref,        'r-'  ,linewidth=2)
ax[0,0].set_ylim(0      , 50 )
ax[0,0].set_title('Spectral type', fontsize=20)
ax[0,0].set_ylabel(r'$A_u$', fontsize=20)
ax[0,0].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[0,0].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[0,0].tick_params(axis='both', labelsize=16)
ax[0,0].minorticks_on()

ax[1,0].xaxis.set_major_locator(locator)
ax[1,0].xaxis.set_major_formatter(formatter)
ax[1,0].plot(mdates.num2date(time), ANv_60p[:,0]  ,  'rs'  ,label='raw spectrum'  ,markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[1,0].plot(mdates.num2date(time), ANv_n_60p[:,0],  'k*'  ,label='normalized spectrum'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[1,0].plot(mdates.num2date(time), Av_ref,        'g-'    ,linewidth=2)
ax[1,0].set_ylim(0      , 50 )
ax[1,0].set_ylabel(r'$A_v$', fontsize=20)
ax[1,0].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[1,0].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[1,0].tick_params(axis='both', labelsize=16)
ax[1,0].minorticks_on()

ax[2,0].xaxis.set_major_locator(locator)
ax[2,0].xaxis.set_major_formatter(formatter)
ax[2,0].plot(mdates.num2date(time), ANw_60p[:,0]  ,  'rs'  ,label='raw spectrum'  ,markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[2,0].plot(mdates.num2date(time), ANw_n_60p[:,0],  'k*'  ,label='normalized spectrum'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[2,0].plot(mdates.num2date(time), Aw_ref,        'b-'  ,linewidth=2)
ax[2,0].set_ylim(0      , 50 )
ax[2,0].set_ylabel(r'$A_w$', fontsize=20)
ax[2,0].legend(loc='best',ncol=1,fontsize=14)
ax[2,0].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[2,0].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[2,0].tick_params(axis='both', labelsize=16)
ax[2,0].minorticks_on()

##
ax[0,1].xaxis.set_major_locator(locator)
ax[0,1].xaxis.set_major_formatter(formatter)
ax[0,1].plot(mdates.num2date(time), ANu_wo[:,0],   'rs'  ,label='w.o. smooth',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[0,1].plot(mdates.num2date(time), ANu_120p[:,0], 'go'  ,label='120 points' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[0,1].plot(mdates.num2date(time), ANu_60p[:,0],  'b^'  ,label='60 points'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[0,1].plot(mdates.num2date(time), ANu_30p[:,0],  'k*'  ,label='30 points'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[0,1].plot(mdates.num2date(time), Au_ref,        'r-'  ,linewidth=2)
ax[0,1].set_ylim(0      , 50 )
ax[0,1].set_title('Spectral smoothing parameter', fontsize=20)
ax[0,1].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[0,1].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[0,1].tick_params(axis='both', labelsize=16)
ax[0,1].minorticks_on()

ax[1,1].xaxis.set_major_locator(locator)
ax[1,1].xaxis.set_major_formatter(formatter)
ax[1,1].plot(mdates.num2date(time), ANv_wo[:,0],   'rs'  ,label='$w.o. smooth$',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[1,1].plot(mdates.num2date(time), ANv_120p[:,0], 'go'  ,label='$120 points$' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[1,1].plot(mdates.num2date(time), ANv_60p[:,0],  'b^'  ,label='$60 points$'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[1,1].plot(mdates.num2date(time), ANv_30p[:,0],  'k*'  ,label='$30 points$'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[1,1].plot(mdates.num2date(time), Av_ref,        'g-'  ,linewidth=2)
ax[1,1].set_ylim(0      , 50 )
ax[1,1].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[1,1].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[1,1].tick_params(axis='both', labelsize=16)
ax[1,1].minorticks_on()
   
ax[2,1].xaxis.set_major_locator(locator)
ax[2,1].xaxis.set_major_formatter(formatter)
ax[2,1].plot(mdates.num2date(time), ANw_wo[:,0],   'rs'  ,label='w.o. smooth',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[2,1].plot(mdates.num2date(time), ANw_120p[:,0], 'go'  ,label='120 points' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[2,1].plot(mdates.num2date(time), ANw_60p[:,0],  'b^'  ,label='60 points'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[2,1].plot(mdates.num2date(time), ANw_30p[:,0],  'k*'  ,label='30 points'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[2,1].plot(mdates.num2date(time), Aw_ref,        'b-'  ,linewidth=2)
ax[2,1].set_ylim(0      , 50 )
ax[2,1].legend(loc='best',ncol=1,fontsize=14)
ax[2,1].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[2,1].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[2,1].tick_params(axis='both', labelsize=16)
ax[2,1].minorticks_on()    




##


ax[0,2].xaxis.set_major_locator(locator)
ax[0,2].xaxis.set_major_formatter(formatter)
ax[0,2].plot(mdates.num2date(time), ANu_12seg[:,0], 'go'  ,label='12 segments' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[0,2].plot(mdates.num2date(time), ANu_6seg[:,0],  'b^'  ,label='6 segments'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[0,2].plot(mdates.num2date(time), ANu_3seg[:,0],  'k*'  ,label='3 segments'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[0,2].plot(mdates.num2date(time), ANu_1seg[:,0],   'rs'  ,label='1 segments',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[0,2].plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
ax[0,2].set_ylim(0      , 50 )
ax[0,2].set_title('Pwelch segment numbers', fontsize=20)
ax[0,2].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[0,2].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[0,2].tick_params(axis='both', labelsize=16)
ax[0,2].minorticks_on()

ax[1,2].xaxis.set_major_locator(locator)
ax[1,2].xaxis.set_major_formatter(formatter)
ax[1,2].plot(mdates.num2date(time), ANv_12seg[:,0], 'go'  ,label='12 segments' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[1,2].plot(mdates.num2date(time), ANv_6seg[:,0],  'b^'  ,label='6 segments'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[1,2].plot(mdates.num2date(time), ANv_3seg[:,0],  'k*'  ,label='3 segments'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[1,2].plot(mdates.num2date(time), ANv_1seg[:,0],   'rs'  ,label='1 segments',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[1,2].plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Au (N400)$'  ,linewidth=2)
ax[1,2].set_ylim(0      , 50 )
ax[1,2].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[1,2].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[1,2].tick_params(axis='both', labelsize=16)
ax[1,2].minorticks_on()

ax[2,2].xaxis.set_major_locator(locator)
ax[2,2].xaxis.set_major_formatter(formatter)
ax[2,2].plot(mdates.num2date(time), ANw_12seg[:,0], 'go'  ,label='12 segments' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[2,2].plot(mdates.num2date(time), ANw_6seg[:,0],  'b^'  ,label='6 segments'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[2,2].plot(mdates.num2date(time), ANw_3seg[:,0],  'k*'  ,label='3 segments'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[2,2].plot(mdates.num2date(time), ANw_1seg[:,0],   'rs'  ,label='1 segments',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[2,2].plot(mdates.num2date(time), Aw_ref,        'b-'  ,linewidth=2)
ax[2,2].set_ylim(0      , 50 )
ax[2,2].legend(loc='best',ncol=1,fontsize=14)
ax[2,2].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[2,2].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[2,2].tick_params(axis='both', labelsize=16)
ax[2,2].minorticks_on()


##
ax[0,3].xaxis.set_major_locator(locator)
ax[0,3].xaxis.set_major_formatter(formatter)
ax[0,3].plot(mdates.num2date(time), ANu_60mi[:,0], 'go'  ,label='60 mins' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[0,3].plot(mdates.num2date(time), ANu_20mi[:,0],  'b^'  ,label='20 mins'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[0,3].plot(mdates.num2date(time), ANu_10mi[:,0],  'k*'  ,label='10 mins'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[0,3].plot(mdates.num2date(time), ANu_5mi[:,0],   'rs'  ,label='5 mins',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[0,3].plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
ax[0,3].set_ylim(0      , 50 )
ax[0,3].set_title('Moving average detrend', fontsize=20)
ax[0,3].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[0,3].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[0,3].tick_params(axis='both', labelsize=16)
ax[0,3].minorticks_on()

ax[1,3].xaxis.set_major_locator(locator)
ax[1,3].xaxis.set_major_formatter(formatter)
ax[1,3].plot(mdates.num2date(time), ANv_60mi[:,0], 'go'  ,label='60 mins' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[1,3].plot(mdates.num2date(time), ANv_20mi[:,0],  'b^'  ,label='20 mins'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[1,3].plot(mdates.num2date(time), ANv_10mi[:,0],  'k*'  ,label='10 mins'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[1,3].plot(mdates.num2date(time), ANv_5mi[:,0],   'rs'  ,label='5 mins',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[1,3].plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Au (N400)$'  ,linewidth=2)
ax[1,3].set_ylim(0      , 50 )
ax[1,3].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[1,3].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[1,3].tick_params(axis='both', labelsize=16)
ax[1,3].minorticks_on()

ax[2,3].xaxis.set_major_locator(locator)
ax[2,3].xaxis.set_major_formatter(formatter)
ax[2,3].plot(mdates.num2date(time), ANw_60mi[:,0], 'go'  ,label='60 mins' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[2,3].plot(mdates.num2date(time), ANw_20mi[:,0],  'b^'  ,label='20 mins'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[2,3].plot(mdates.num2date(time), ANw_10mi[:,0],  'k*'  ,label='10 mins'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[2,3].plot(mdates.num2date(time), ANw_5mi[:,0],   'rs'  ,label='5 mins',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[2,3].plot(mdates.num2date(time), Aw_ref,        'b-'  ,linewidth=2)
ax[2,3].legend(loc='best',ncol=1,fontsize=14)
ax[2,3].set_ylim(0      , 50 )
ax[2,3].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[2,3].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[2,3].tick_params(axis='both', labelsize=16)
ax[2,3].minorticks_on()


##
ax[0,4].xaxis.set_major_locator(locator)
ax[0,4].xaxis.set_major_formatter(formatter)
ax[0,4].plot(mdates.num2date(time), ANu_bp[:,0], 'go'  ,label='band pass' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[0,4].plot(mdates.num2date(time), ANu_hp[:,0],  'b^'  ,label='high pass'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[0,4].plot(mdates.num2date(time), ANu_lp[:,0],  'k*'  ,label='low pass'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[0,4].plot(mdates.num2date(time), ANu_fu[:,0],   'rs'  ,label='full range',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[0,4].plot(mdates.num2date(time), Au_ref,        'r-'  ,label='$Au (N400)$'  ,linewidth=2)
ax[0,4].set_title('Fitted frequency range', fontsize=20)
ax[0,4].set_ylim(0      , 50 )
ax[0,4].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[0,4].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[0,4].tick_params(axis='both', labelsize=16)
ax[0,4].minorticks_on()

ax[1,4].xaxis.set_major_locator(locator)
ax[1,4].xaxis.set_major_formatter(formatter)
ax[1,4].plot(mdates.num2date(time), ANv_bp[:,0], 'go'  ,label='band pass' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[1,4].plot(mdates.num2date(time), ANv_hp[:,0],  'b^'  ,label='high pass'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[1,4].plot(mdates.num2date(time), ANv_lp[:,0],  'k*'  ,label='low pass'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[1,4].plot(mdates.num2date(time), ANv_fu[:,0],   'rs'  ,label='full range',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[1,4].plot(mdates.num2date(time), Av_ref,        'g-'  ,label='$Au (N400)$'  ,linewidth=2)
ax[1,4].set_ylim(0      , 50 )
ax[1,4].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[1,4].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[1,4].tick_params(axis='both', labelsize=16)
ax[1,4].minorticks_on()

ax[2,4].xaxis.set_major_locator(locator)
ax[2,4].xaxis.set_major_formatter(formatter)
ax[2,4].plot(mdates.num2date(time), ANw_bp[:,0], 'go'  ,label='band pass' ,markeredgecolor='g',markersize=8,markerfacecolor='none')
ax[2,4].plot(mdates.num2date(time), ANw_hp[:,0],  'b^'  ,label='high pass'  ,markeredgecolor='b',markersize=8,markerfacecolor='none')
ax[2,4].plot(mdates.num2date(time), ANw_lp[:,0],  'k*'  ,label='low pass'  ,markeredgecolor='k',markersize=8,markerfacecolor='none')
ax[2,4].plot(mdates.num2date(time), ANw_fu[:,0],   'rs'  ,label='full range',markeredgecolor='r',markersize=8,markerfacecolor='none')
ax[2,4].plot(mdates.num2date(time), Aw_ref,        'b-'  ,linewidth=2)
ax[2,4].legend(loc='best',ncol=1,fontsize=14)
ax[2,4].set_ylim(0      , 50 )
ax[2,4].grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
ax[2,4].grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax[2,4].tick_params(axis='both', labelsize=16)
ax[2,4].minorticks_on()




save_tite = 'Storm No.' + str(event+1) + '_spectral fitting sensitivity.png'
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(save_tite)     