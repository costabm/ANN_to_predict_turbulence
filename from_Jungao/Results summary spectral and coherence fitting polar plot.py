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
from scipy.stats import gaussian_kde
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
dx_Osp1   = np.empty((0))
dy_Osp1   = np.empty((0))
Time_Osp1 = np.empty((0))


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
dx_Osp2   = np.empty((0))
dy_Osp2   = np.empty((0))
Time_Osp2 = np.empty((0))


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
Time_Sva   = np.empty((0))

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
Time_Syn   = np.empty((0))


U_Osp1_s   = np.empty((0, 3))
Dir_Osp1_s = np.empty((0, 3))
Iu_Osp1_s  = np.empty((0, 3))
Iv_Osp1_s  = np.empty((0, 3))
Iw_Osp1_s  = np.empty((0, 3))
ANu_Osp1_s  = np.empty((0, 3))
ANv_Osp1_s  = np.empty((0, 3))
ANw_Osp1_s  = np.empty((0, 3))
ANu_n_Osp1_s  = np.empty((0, 3))
ANv_n_Osp1_s  = np.empty((0, 3))
ANw_n_Osp1_s  = np.empty((0, 3))
Cux_Osp1_s  = np.empty((0, 1))
Cvx_Osp1_s  = np.empty((0, 1))
Cwx_Osp1_s  = np.empty((0, 1))
Cuy_Osp1_s  = np.empty((0, 1))
Cvy_Osp1_s  = np.empty((0, 1))
Cwy_Osp1_s  = np.empty((0, 1))
Cuz_Osp1_s  = np.empty((0, 1))
Cvz_Osp1_s  = np.empty((0, 1))
Cwz_Osp1_s  = np.empty((0, 1))
dx_Osp1_s   = np.empty((0))
dy_Osp1_s   = np.empty((0))
Time_Osp1_s = np.empty((0))


U_Osp2_s   = np.empty((0, 3))
Dir_Osp2_s = np.empty((0, 3))
Iu_Osp2_s  = np.empty((0, 3))
Iv_Osp2_s  = np.empty((0, 3))
Iw_Osp2_s  = np.empty((0, 3))
ANu_Osp2_s  = np.empty((0, 3))
ANv_Osp2_s  = np.empty((0, 3))
ANw_Osp2_s  = np.empty((0, 3))
ANu_n_Osp2_s  = np.empty((0, 3))
ANv_n_Osp2_s  = np.empty((0, 3))
ANw_n_Osp2_s  = np.empty((0, 3))
Cux_Osp2_s  = np.empty((0, 1))
Cvx_Osp2_s  = np.empty((0, 1))
Cwx_Osp2_s  = np.empty((0, 1))
Cuy_Osp2_s  = np.empty((0, 1))
Cvy_Osp2_s  = np.empty((0, 1))
Cwy_Osp2_s  = np.empty((0, 1))
Cuz_Osp2_s  = np.empty((0, 1))
Cvz_Osp2_s  = np.empty((0, 1))
Cwz_Osp2_s  = np.empty((0, 1))
dx_Osp2_s   = np.empty((0))
dy_Osp2_s   = np.empty((0))
Time_Osp2_s = np.empty((0))


U_Sva_s   = np.empty((0, 3))
Dir_Sva_s = np.empty((0, 3))
Iu_Sva_s  = np.empty((0, 3))
Iv_Sva_s  = np.empty((0, 3))
Iw_Sva_s  = np.empty((0, 3))
ANu_Sva_s  = np.empty((0, 3))
ANv_Sva_s  = np.empty((0, 3))
ANw_Sva_s  = np.empty((0, 3))
ANu_n_Sva_s  = np.empty((0, 3))
ANv_n_Sva_s  = np.empty((0, 3))
ANw_n_Sva_s  = np.empty((0, 3))
Time_Sva_s = np.empty((0))


U_Syn_s   = np.empty((0, 3))
Dir_Syn_s = np.empty((0, 3))
Iu_Syn_s  = np.empty((0, 3))
Iv_Syn_s  = np.empty((0, 3))
Iw_Syn_s  = np.empty((0, 3))
ANu_Syn_s  = np.empty((0, 3))
ANv_Syn_s  = np.empty((0, 3))
ANw_Syn_s  = np.empty((0, 3))
ANu_n_Syn_s  = np.empty((0, 3))
ANv_n_Syn_s  = np.empty((0, 3))
ANw_n_Syn_s  = np.empty((0, 3))
Time_Syn_s = np.empty((0))
#%%

#for event in range(0,cases['Time_storm'].size):
for event in range(0,cases['Time_storm'].size):

    #event       = 0
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

    time_new  = pd.date_range(data_Osp1['date_start'],data_Osp1['date_end'],freq = '1H')  
    Time_Osp1 = np.concatenate( (Time_Osp1,  np.asarray(mdates.date2num(time_new))))
    
    # get the case with maximum velocity within this event
    # first find the cases where stationarinality is acheived
    # then find the maximum wind velocity when spectra and coherence were analyzed 
    tmp_ANu   = np.asarray(data_Osp1['ANu'])[:,0]
    loc_Anu   = np.where(tmp_ANu==0)
    tmp_U     = np.asarray(data_Osp1['U'])[:,0]
    tmp_U[loc_Anu] = np.nan
    
    loc = np.where(tmp_U == np.nanmax(tmp_U))[0]
    if loc.size>1:
        loc= np.asarray([loc[0]])
        
    U_Osp1_s   = np.concatenate((U_Osp1_s, np.asarray(data_Osp1['U'])[loc,:]))
    Dir_Osp1_s = np.concatenate((Dir_Osp1_s, np.asarray(data_Osp1['Dir'])[loc,:]))
    Iu_Osp1_s = np.concatenate((Iu_Osp1_s, np.asarray(data_Osp1['Iu'])[loc,:]))
    Iv_Osp1_s = np.concatenate((Iv_Osp1_s, np.asarray(data_Osp1['Iv'])[loc,:]))
    Iw_Osp1_s = np.concatenate((Iw_Osp1_s, np.asarray(data_Osp1['Iw'])[loc,:]))
    ANu_Osp1_s = np.concatenate((ANu_Osp1_s, np.asarray(data_Osp1['ANu'])[loc,:]))
    ANv_Osp1_s = np.concatenate((ANv_Osp1_s, np.asarray(data_Osp1['ANv'])[loc,:]))
    ANw_Osp1_s = np.concatenate((ANw_Osp1_s, np.asarray(data_Osp1['ANw'])[loc,:]))
    ANu_n_Osp1_s = np.concatenate((ANu_n_Osp1_s, np.asarray(data_Osp1['ANu_n'])[loc,:]))
    ANv_n_Osp1_s = np.concatenate((ANv_n_Osp1_s, np.asarray(data_Osp1['ANv_n'])[loc,:]))
    ANw_n_Osp1_s = np.concatenate((ANw_n_Osp1_s, np.asarray(data_Osp1['ANw_n'])[loc,:]))    
    Time_Osp1_s  = np.concatenate( (Time_Osp1_s,np.asarray(mdates.date2num(time_new[loc]))))
    
    
    

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
    dx_Osp1 = np.concatenate( (dx_Osp1,  np.asarray(data_coh_Osp1['dx'])[:,0]))
    dy_Osp1 = np.concatenate( (dy_Osp1,  np.asarray(data_coh_Osp1['dy'])[:,0]))


    Cux_Osp1_s = np.concatenate([Cux_Osp1_s, np.asarray(data_coh_Osp1['Cux'])[loc,:]])    
    Cvx_Osp1_s = np.concatenate([Cvx_Osp1_s, np.asarray(data_coh_Osp1['Cvx'])[loc,:]])    
    Cwx_Osp1_s = np.concatenate([Cwx_Osp1_s, np.asarray(data_coh_Osp1['Cwx'])[loc,:]])    
    Cuy_Osp1_s = np.concatenate([Cuy_Osp1_s, np.asarray(data_coh_Osp1['Cuy'])[loc,:]])    
    Cvy_Osp1_s = np.concatenate([Cvy_Osp1_s, np.asarray(data_coh_Osp1['Cvy'])[loc,:]])    
    Cwy_Osp1_s = np.concatenate([Cwy_Osp1_s, np.asarray(data_coh_Osp1['Cwy'])[loc,:]])    
    Cuz_Osp1_s = np.concatenate([Cuz_Osp1_s, np.asarray(data_coh_Osp1['Cuz'])[loc,:]])    
    Cvz_Osp1_s = np.concatenate([Cvz_Osp1_s, np.asarray(data_coh_Osp1['Cvz'])[loc,:]])    
    Cwz_Osp1_s = np.concatenate([Cwz_Osp1_s, np.asarray(data_coh_Osp1['Cwz'])[loc,:]])    
    dx_Osp1_s = np.concatenate( (dx_Osp1_s,  np.asarray(data_coh_Osp1['dx'])[loc,0]))
    dy_Osp1_s = np.concatenate( (dy_Osp1_s,  np.asarray(data_coh_Osp1['dy'])[loc,0]))





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

    time_new  = pd.date_range(data_Osp2['date_start'],data_Osp2['date_end'],freq = '1H')  
    Time_Osp2 = np.concatenate( (Time_Osp2,  np.asarray(mdates.date2num(time_new))))

    U_Osp2_s   = np.concatenate((U_Osp2_s, np.asarray(data_Osp2['U'])[loc,:]))
    Dir_Osp2_s = np.concatenate((Dir_Osp2_s, np.asarray(data_Osp2['Dir'])[loc,:]))
    Iu_Osp2_s = np.concatenate((Iu_Osp2_s, np.asarray(data_Osp2['Iu'])[loc,:]))
    Iv_Osp2_s = np.concatenate((Iv_Osp2_s, np.asarray(data_Osp2['Iv'])[loc,:]))
    Iw_Osp2_s = np.concatenate((Iw_Osp2_s, np.asarray(data_Osp2['Iw'])[loc,:]))
    ANu_Osp2_s = np.concatenate((ANu_Osp2_s, np.asarray(data_Osp2['ANu'])[loc,:]))
    ANv_Osp2_s = np.concatenate((ANv_Osp2_s, np.asarray(data_Osp2['ANv'])[loc,:]))
    ANw_Osp2_s = np.concatenate((ANw_Osp2_s, np.asarray(data_Osp2['ANw'])[loc,:]))
    ANu_n_Osp2_s = np.concatenate((ANu_n_Osp2_s, np.asarray(data_Osp2['ANu_n'])[loc,:]))
    ANv_n_Osp2_s = np.concatenate((ANv_n_Osp2_s, np.asarray(data_Osp2['ANv_n'])[loc,:]))
    ANw_n_Osp2_s = np.concatenate((ANw_n_Osp2_s, np.asarray(data_Osp2['ANw_n'])[loc,:]))    
    Time_Osp2_s  = np.concatenate( (Time_Osp2_s,np.asarray(mdates.date2num(time_new[loc]))))


    path      = 'E:/DATA/Results/Osp2 Coherence fitting/'             
    data_name = 'Storm No.' + str(event+1) + '_coherence fitting Osp2_'\
                 + str(date_begin.year) + '_' + str(date_begin.month) + '_' + str(date_begin.day)\
                     + '_' + str(date_begin.hour)    
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
    dx_Osp2 = np.concatenate( (dx_Osp2,  np.asarray(data_coh_Osp2['dx'])[:,0]))
    dy_Osp2 = np.concatenate( (dy_Osp2,  np.asarray(data_coh_Osp2['dy'])[:,0]))   
        
    Cux_Osp2_s = np.concatenate([Cux_Osp2_s, np.asarray(data_coh_Osp2['Cux'])[loc,:]])    
    Cvx_Osp2_s = np.concatenate([Cvx_Osp2_s, np.asarray(data_coh_Osp2['Cvx'])[loc,:]])    
    Cwx_Osp2_s = np.concatenate([Cwx_Osp2_s, np.asarray(data_coh_Osp2['Cwx'])[loc,:]])    
    Cuy_Osp2_s = np.concatenate([Cuy_Osp2_s, np.asarray(data_coh_Osp2['Cuy'])[loc,:]])    
    Cvy_Osp2_s = np.concatenate([Cvy_Osp2_s, np.asarray(data_coh_Osp2['Cvy'])[loc,:]])    
    Cwy_Osp2_s = np.concatenate([Cwy_Osp2_s, np.asarray(data_coh_Osp2['Cwy'])[loc,:]])    
    Cuz_Osp2_s = np.concatenate([Cuz_Osp2_s, np.asarray(data_coh_Osp2['Cuz'])[loc,:]])    
    Cvz_Osp2_s = np.concatenate([Cvz_Osp2_s, np.asarray(data_coh_Osp2['Cvz'])[loc,:]])    
    Cwz_Osp2_s = np.concatenate([Cwz_Osp2_s, np.asarray(data_coh_Osp2['Cwz'])[loc,:]])  
    dx_Osp2_s = np.concatenate( (dx_Osp2_s,  np.asarray(data_coh_Osp2['dx'])[loc,0]))
    dy_Osp2_s = np.concatenate( (dy_Osp2_s,  np.asarray(data_coh_Osp2['dy'])[loc,0]))
    

     
    
        

        
    
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

    time_new  = pd.date_range(data_Sva['date_start'],data_Sva['date_end'],freq = '1H')  
    Time_Sva = np.concatenate( (Time_Sva,  np.asarray(mdates.date2num(time_new))))

    U_Sva_s   = np.concatenate((U_Sva_s, np.asarray(data_Sva['U'])[loc,:]))
    Dir_Sva_s = np.concatenate((Dir_Sva_s, np.asarray(data_Sva['Dir'])[loc,:]))
    Iu_Sva_s = np.concatenate((Iu_Sva_s, np.asarray(data_Sva['Iu'])[loc,:]))
    Iv_Sva_s = np.concatenate((Iv_Sva_s, np.asarray(data_Sva['Iv'])[loc,:]))
    Iw_Sva_s = np.concatenate((Iw_Sva_s, np.asarray(data_Sva['Iw'])[loc,:]))
    ANu_Sva_s = np.concatenate((ANu_Sva_s, np.asarray(data_Sva['ANu'])[loc,:]))
    ANv_Sva_s = np.concatenate((ANv_Sva_s, np.asarray(data_Sva['ANv'])[loc,:]))
    ANw_Sva_s = np.concatenate((ANw_Sva_s, np.asarray(data_Sva['ANw'])[loc,:]))
    ANu_n_Sva_s = np.concatenate((ANu_n_Sva_s, np.asarray(data_Sva['ANu_n'])[loc,:]))
    ANv_n_Sva_s = np.concatenate((ANv_n_Sva_s, np.asarray(data_Sva['ANv_n'])[loc,:]))
    ANw_n_Sva_s = np.concatenate((ANw_n_Sva_s, np.asarray(data_Sva['ANw_n'])[loc,:]))    
    Time_Sva_s  = np.concatenate( (Time_Sva_s,np.asarray(mdates.date2num(time_new[loc]))))
    


    path      = 'E:/DATA/Results/Synn Spectra fitting/'        
    data_name = 'Storm No.' + str(event+1) + '_spectrum fitting Synn sensor No.3_'\
                 + str(date_start.year) + '_' + str(date_start.month) + '_' + str(date_start.day)\
                     + '_' + str(date_start.hour) 
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
    
    time_new  = pd.date_range(data_Syn['date_start'],data_Syn['date_end'],freq = '1H')  
    Time_Syn = np.concatenate( (Time_Syn,  np.asarray(mdates.date2num(time_new))))

    U_Syn_s   = np.concatenate((U_Syn_s, np.asarray(data_Syn['U'])[loc,:]))
    Dir_Syn_s = np.concatenate((Dir_Syn_s, np.asarray(data_Syn['Dir'])[loc,:]))
    Iu_Syn_s = np.concatenate((Iu_Syn_s, np.asarray(data_Syn['Iu'])[loc,:]))
    Iv_Syn_s = np.concatenate((Iv_Syn_s, np.asarray(data_Syn['Iv'])[loc,:]))
    Iw_Syn_s = np.concatenate((Iw_Syn_s, np.asarray(data_Syn['Iw'])[loc,:]))
    ANu_Syn_s = np.concatenate((ANu_Syn_s, np.asarray(data_Syn['ANu'])[loc,:]))
    ANv_Syn_s = np.concatenate((ANv_Syn_s, np.asarray(data_Syn['ANv'])[loc,:]))
    ANw_Syn_s = np.concatenate((ANw_Syn_s, np.asarray(data_Syn['ANw'])[loc,:]))
    ANu_n_Syn_s = np.concatenate((ANu_n_Syn_s, np.asarray(data_Syn['ANu_n'])[loc,:]))
    ANv_n_Syn_s = np.concatenate((ANv_n_Syn_s, np.asarray(data_Syn['ANv_n'])[loc,:]))
    ANw_n_Syn_s = np.concatenate((ANw_n_Syn_s, np.asarray(data_Syn['ANw_n'])[loc,:]))    
    Time_Syn_s  = np.concatenate( (Time_Syn_s,np.asarray(mdates.date2num(time_new[loc]))))        

    

#%%
# collect results for Osp1 sensor A
result_Osp1 = np.zeros((np.size(Cux_Osp1),20))
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
result_Osp1[:,17] = Time_Osp1
result_Osp1[:,18] = dx_Osp1
result_Osp1[:,19] = dy_Osp1
result_Osp1[result_Osp1 == 0]  = np.nan

Osp1   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw','Cux','Cuy', 'Cuz','Cvx','Cvy', 'Cvz','Cwx','Cwy', 'Cwz','dx','dy'])  
Osp1['Time'] = mdates.num2date(Time_Osp1)
Osp1['U']    = result_Osp1[:,0]
Osp1['Dir']  = result_Osp1[:,1]
Osp1['Iu']   = result_Osp1[:,2]
Osp1['Iv']   = result_Osp1[:,3]
Osp1['Iw']   = result_Osp1[:,4]
Osp1['ANu']  = result_Osp1[:,5]
Osp1['ANv']  = result_Osp1[:,6]
Osp1['ANw']  = result_Osp1[:,7]
Osp1['Cux']  = result_Osp1[:,8]
Osp1['Cuy']  = result_Osp1[:,11]
Osp1['Cuz']  = result_Osp1[:,14]
Osp1['Cvx']  = result_Osp1[:,9]
Osp1['Cvy']  = result_Osp1[:,12]
Osp1['Cvz']  = result_Osp1[:,15]
Osp1['Cwx']  = result_Osp1[:,10]
Osp1['Cwy']  = result_Osp1[:,13]
Osp1['Cwz']  = result_Osp1[:,16]
Osp1['dx']   = result_Osp1[:,18]
Osp1['dy']   = result_Osp1[:,19]

#%%
#collect the data (strongest for each event)
result_Osp1_s = np.zeros((np.size(Cux_Osp1_s),20))
result_Osp1_s[:,0] = U_Osp1_s[:,0]
result_Osp1_s[:,1] = Dir_Osp1_s[:,0]
result_Osp1_s[:,2] = Iu_Osp1_s[:,0]
result_Osp1_s[:,3] = Iv_Osp1_s[:,0]
result_Osp1_s[:,4] = Iw_Osp1_s[:,0]    
result_Osp1_s[:,5] = ANu_Osp1_s[:,0]    
result_Osp1_s[:,6] = ANv_Osp1_s[:,0]    
result_Osp1_s[:,7] = ANw_Osp1_s[:,0]    
result_Osp1_s[:,8] = Cux_Osp1_s[:,0]    
result_Osp1_s[:,9] = Cvx_Osp1_s[:,0]    
result_Osp1_s[:,10] = Cwx_Osp1_s[:,0]    
result_Osp1_s[:,11] = Cuy_Osp1_s[:,0]    
result_Osp1_s[:,12] = Cvy_Osp1_s[:,0]    
result_Osp1_s[:,13] = Cwy_Osp1_s[:,0] 
result_Osp1_s[:,14] = Cuz_Osp1_s[:,0]    
result_Osp1_s[:,15] = Cvz_Osp1_s[:,0]    
result_Osp1_s[:,16] = Cwz_Osp1_s[:,0] 
result_Osp1_s[:,17] = Time_Osp1_s
result_Osp1_s[:,18] = dx_Osp1_s
result_Osp1_s[:,19] = dy_Osp1_s
result_Osp1_s[result_Osp1_s == 0]  = np.nan

Osp1_s   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw','Cux','Cuy', 'Cuz','Cvx','Cvy', 'Cvz','Cwx','Cwy', 'Cwz','dx','dy'])  
Osp1_s['Time'] = mdates.num2date(Time_Osp1_s)
Osp1_s['U']    = result_Osp1_s[:,0]
Osp1_s['Dir']  = result_Osp1_s[:,1]
Osp1_s['Iu']   = result_Osp1_s[:,2]
Osp1_s['Iv']   = result_Osp1_s[:,3]
Osp1_s['Iw']   = result_Osp1_s[:,4]
Osp1_s['ANu']  = result_Osp1_s[:,5]
Osp1_s['ANv']  = result_Osp1_s[:,6]
Osp1_s['ANw']  = result_Osp1_s[:,7]
Osp1_s['Cux']  = result_Osp1_s[:,8]
Osp1_s['Cuy']  = result_Osp1_s[:,11]
Osp1_s['Cuz']  = result_Osp1_s[:,14]
Osp1_s['Cvx']  = result_Osp1_s[:,9]
Osp1_s['Cvy']  = result_Osp1_s[:,12]
Osp1_s['Cvz']  = result_Osp1_s[:,15]
Osp1_s['Cwx']  = result_Osp1_s[:,10]
Osp1_s['Cwy']  = result_Osp1_s[:,13]
Osp1_s['Cwz']  = result_Osp1_s[:,16]
Osp1_s['dx']   = result_Osp1_s[:,18]
Osp1_s['dy']   = result_Osp1_s[:,19]

#%% Osp1B
result_Osp1_B = np.zeros((np.size(Cux_Osp1),8))
result_Osp1_B[:,0] = U_Osp1[:,1]
result_Osp1_B[:,1] = Dir_Osp1[:,1]
result_Osp1_B[:,2] = Iu_Osp1[:,1]
result_Osp1_B[:,3] = Iv_Osp1[:,1]
result_Osp1_B[:,4] = Iw_Osp1[:,1]    
result_Osp1_B[:,5] = ANu_Osp1[:,1]    
result_Osp1_B[:,6] = ANv_Osp1[:,1]    
result_Osp1_B[:,7] = ANw_Osp1[:,1]    
result_Osp1_B[result_Osp1_B == 0]  = np.nan

Osp1B   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Osp1B['Time'] = mdates.num2date(Time_Osp1)
Osp1B['U']    = result_Osp1_B[:,0]
Osp1B['Dir']  = result_Osp1_B[:,1]
Osp1B['Iu']   = result_Osp1_B[:,2]
Osp1B['Iv']   = result_Osp1_B[:,3]
Osp1B['Iw']   = result_Osp1_B[:,4]
Osp1B['ANu']  = result_Osp1_B[:,5]
Osp1B['ANv']  = result_Osp1_B[:,6]
Osp1B['ANw']  = result_Osp1_B[:,7]


result_Osp1B_s = np.zeros((np.size(Cux_Osp1_s),8))
result_Osp1B_s[:,0] = U_Osp1_s[:,1]
result_Osp1B_s[:,1] = Dir_Osp1_s[:,1]
result_Osp1B_s[:,2] = Iu_Osp1_s[:,1]
result_Osp1B_s[:,3] = Iv_Osp1_s[:,1]
result_Osp1B_s[:,4] = Iw_Osp1_s[:,1]    
result_Osp1B_s[:,5] = ANu_Osp1_s[:,1]    
result_Osp1B_s[:,6] = ANv_Osp1_s[:,1]    
result_Osp1B_s[:,7] = ANw_Osp1_s[:,1]    
result_Osp1B_s[result_Osp1B_s == 0]  = np.nan

Osp1B_s   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Osp1B_s['Time'] = mdates.num2date(Time_Osp1_s)
Osp1B_s['U']    = result_Osp1B_s[:,0]
Osp1B_s['Dir']  = result_Osp1B_s[:,1]
Osp1B_s['Iu']   = result_Osp1B_s[:,2]
Osp1B_s['Iv']   = result_Osp1B_s[:,3]
Osp1B_s['Iw']   = result_Osp1B_s[:,4]
Osp1B_s['ANu']  = result_Osp1B_s[:,5]
Osp1B_s['ANv']  = result_Osp1B_s[:,6]
Osp1B_s['ANw']  = result_Osp1B_s[:,7]



#%%
# collect results for Osp2 sensor A
result_Osp2 = np.zeros((np.size(Cux_Osp2),20))
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
result_Osp2[:,17] = Time_Osp2
result_Osp2[:,18] = dx_Osp2
result_Osp2[:,19] = dy_Osp2
result_Osp2[result_Osp2 == 0]  = np.nan

Osp2   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw','Cux','Cuy', 'Cuz','Cvx','Cvy', 'Cvz','Cwx','Cwy', 'Cwz','dx','dy'])  
Osp2['Time'] = mdates.num2date(Time_Osp2)
Osp2['U']    = result_Osp2[:,0]
Osp2['Dir']  = result_Osp2[:,1]
Osp2['Iu']   = result_Osp2[:,2]
Osp2['Iv']   = result_Osp2[:,3]
Osp2['Iw']   = result_Osp2[:,4]
Osp2['ANu']  = result_Osp2[:,5]
Osp2['ANv']  = result_Osp2[:,6]
Osp2['ANw']  = result_Osp2[:,7]
Osp2['Cux']  = result_Osp2[:,8]
Osp2['Cuy']  = result_Osp2[:,11]
Osp2['Cuz']  = result_Osp2[:,14]
Osp2['Cvx']  = result_Osp2[:,9]
Osp2['Cvy']  = result_Osp2[:,12]
Osp2['Cvz']  = result_Osp2[:,15]
Osp2['Cwx']  = result_Osp2[:,10]
Osp2['Cwy']  = result_Osp2[:,13]
Osp2['Cwz']  = result_Osp2[:,16]
Osp2['dx']   = result_Osp2[:,18]
Osp2['dy']   = result_Osp2[:,19]

#%%
#collect the data (strongest for each event)
result_Osp2_s = np.zeros((np.size(Cux_Osp2_s),20))
result_Osp2_s[:,0] = U_Osp2_s[:,0]
result_Osp2_s[:,1] = Dir_Osp2_s[:,0]
result_Osp2_s[:,2] = Iu_Osp2_s[:,0]
result_Osp2_s[:,3] = Iv_Osp2_s[:,0]
result_Osp2_s[:,4] = Iw_Osp2_s[:,0]    
result_Osp2_s[:,5] = ANu_Osp2_s[:,0]    
result_Osp2_s[:,6] = ANv_Osp2_s[:,0]    
result_Osp2_s[:,7] = ANw_Osp2_s[:,0]    
result_Osp2_s[:,8] = Cux_Osp2_s[:,0]    
result_Osp2_s[:,9] = Cvx_Osp2_s[:,0]    
result_Osp2_s[:,10] = Cwx_Osp2_s[:,0]    
result_Osp2_s[:,11] = Cuy_Osp2_s[:,0]    
result_Osp2_s[:,12] = Cvy_Osp2_s[:,0]    
result_Osp2_s[:,13] = Cwy_Osp2_s[:,0] 
result_Osp2_s[:,14] = Cuz_Osp2_s[:,0]    
result_Osp2_s[:,15] = Cvz_Osp2_s[:,0]    
result_Osp2_s[:,16] = Cwz_Osp2_s[:,0] 
result_Osp2_s[:,17] = Time_Osp2_s
result_Osp2_s[:,18] = dx_Osp2_s
result_Osp2_s[:,19] = dy_Osp2_s
result_Osp2_s[result_Osp2_s == 0]  = np.nan

Osp2_s   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw','Cux','Cuy', 'Cuz','Cvx','Cvy', 'Cvz','Cwx','Cwy', 'Cwz','dx','dy'])  
Osp2_s['Time'] = mdates.num2date(Time_Osp2_s)
Osp2_s['U']    = result_Osp2_s[:,0]
Osp2_s['Dir']  = result_Osp2_s[:,1]
Osp2_s['Iu']   = result_Osp2_s[:,2]
Osp2_s['Iv']   = result_Osp2_s[:,3]
Osp2_s['Iw']   = result_Osp2_s[:,4]
Osp2_s['ANu']  = result_Osp2_s[:,5]
Osp2_s['ANv']  = result_Osp2_s[:,6]
Osp2_s['ANw']  = result_Osp2_s[:,7]
Osp2_s['Cux']  = result_Osp2_s[:,8]
Osp2_s['Cuy']  = result_Osp2_s[:,11]
Osp2_s['Cuz']  = result_Osp2_s[:,14]
Osp2_s['Cvx']  = result_Osp2_s[:,9]
Osp2_s['Cvy']  = result_Osp2_s[:,12]
Osp2_s['Cvz']  = result_Osp2_s[:,15]
Osp2_s['Cwx']  = result_Osp2_s[:,10]
Osp2_s['Cwy']  = result_Osp2_s[:,13]
Osp2_s['Cwz']  = result_Osp2_s[:,16]
Osp2_s['dx']   = result_Osp2_s[:,18]
Osp2_s['dy']   = result_Osp2_s[:,19]

#%% Osp2 sensor B
result_Osp2_B = np.zeros((np.size(Cux_Osp2),8))
result_Osp2_B[:,0] = U_Osp2[:,1]
result_Osp2_B[:,1] = Dir_Osp2[:,1]
result_Osp2_B[:,2] = Iu_Osp2[:,1]
result_Osp2_B[:,3] = Iv_Osp2[:,1]
result_Osp2_B[:,4] = Iw_Osp2[:,1]    
result_Osp2_B[:,5] = ANu_Osp2[:,1]    
result_Osp2_B[:,6] = ANv_Osp2[:,1]    
result_Osp2_B[:,7] = ANw_Osp2[:,1]    
result_Osp2_B[result_Osp2_B == 0]  = np.nan

Osp2B   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Osp2B['Time'] = mdates.num2date(Time_Osp2)
Osp2B['U']    = result_Osp2_B[:,0]
Osp2B['Dir']  = result_Osp2_B[:,1]
Osp2B['Iu']   = result_Osp2_B[:,2]
Osp2B['Iv']   = result_Osp2_B[:,3]
Osp2B['Iw']   = result_Osp2_B[:,4]
Osp2B['ANu']  = result_Osp2_B[:,5]
Osp2B['ANv']  = result_Osp2_B[:,6]
Osp2B['ANw']  = result_Osp2_B[:,7]


result_Osp2B_s = np.zeros((np.size(Cux_Osp2_s),8))
result_Osp2B_s[:,0] = U_Osp2_s[:,1]
result_Osp2B_s[:,1] = Dir_Osp2_s[:,1]
result_Osp2B_s[:,2] = Iu_Osp2_s[:,1]
result_Osp2B_s[:,3] = Iv_Osp2_s[:,1]
result_Osp2B_s[:,4] = Iw_Osp2_s[:,1]    
result_Osp2B_s[:,5] = ANu_Osp2_s[:,1]    
result_Osp2B_s[:,6] = ANv_Osp2_s[:,1]    
result_Osp2B_s[:,7] = ANw_Osp2_s[:,1]    
result_Osp2B_s[result_Osp2B_s == 0]  = np.nan

Osp2B_s   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Osp2B_s['Time'] = mdates.num2date(Time_Osp2_s)
Osp2B_s['U']    = result_Osp2B_s[:,0]
Osp2B_s['Dir']  = result_Osp2B_s[:,1]
Osp2B_s['Iu']   = result_Osp2B_s[:,2]
Osp2B_s['Iv']   = result_Osp2B_s[:,3]
Osp2B_s['Iw']   = result_Osp2B_s[:,4]
Osp2B_s['ANu']  = result_Osp2B_s[:,5]
Osp2B_s['ANv']  = result_Osp2B_s[:,6]
Osp2B_s['ANw']  = result_Osp2B_s[:,7]

#%% Sva sensor A
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
Sva   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Sva['Time'] = mdates.num2date(Time_Sva)
Sva['U']    = result_Sva[:,0]
Sva['Dir']  = result_Sva[:,1]
Sva['Iu']   = result_Sva[:,2]
Sva['Iv']   = result_Sva[:,3]
Sva['Iw']   = result_Sva[:,4]
Sva['ANu']  = result_Sva[:,5]
Sva['ANv']  = result_Sva[:,6]
Sva['ANw']  = result_Sva[:,7]


result_Sva_s = np.zeros((np.size(Cux_Osp1_s),20))
result_Sva_s[:,0] = U_Sva_s[:,0]
result_Sva_s[:,1] = Dir_Sva_s[:,0]
result_Sva_s[:,2] = Iu_Sva_s[:,0]
result_Sva_s[:,3] = Iv_Sva_s[:,0]
result_Sva_s[:,4] = Iw_Sva_s[:,0]    
result_Sva_s[:,5] = ANu_Sva_s[:,0]    
result_Sva_s[:,6] = ANv_Sva_s[:,0]    
result_Sva_s[:,7] = ANw_Sva_s[:,0]   

Sva_s   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Sva_s['Time'] = mdates.num2date(Time_Sva_s)
Sva_s['U']    = result_Sva_s[:,0]
Sva_s['Dir']  = result_Sva_s[:,1]
Sva_s['Iu']   = result_Sva_s[:,2]
Sva_s['Iv']   = result_Sva_s[:,3]
Sva_s['Iw']   = result_Sva_s[:,4]
Sva_s['ANu']  = result_Sva_s[:,5]
Sva_s['ANv']  = result_Sva_s[:,6]
Sva_s['ANw']  = result_Sva_s[:,7]


#%% Sva sensor B
result_Sva_B = np.zeros((np.size(Cux_Osp1),8))
result_Sva_B[:,0] = U_Sva[:,1]
result_Sva_B[:,1] = Dir_Sva[:,1]
result_Sva_B[:,2] = Iu_Sva[:,1]
result_Sva_B[:,3] = Iv_Sva[:,1]
result_Sva_B[:,4] = Iw_Sva[:,1]    
result_Sva_B[:,5] = ANu_Sva[:,1]    
result_Sva_B[:,6] = ANv_Sva[:,1]    
result_Sva_B[:,7] = ANw_Sva[:,1]    
result_Sva_B[result_Sva_B==0]  = np.nan
Sva_B   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Sva_B['Time'] = mdates.num2date(Time_Sva)
Sva_B['U']    = result_Sva_B[:,0]
Sva_B['Dir']  = result_Sva_B[:,1]
Sva_B['Iu']   = result_Sva_B[:,2]
Sva_B['Iv']   = result_Sva_B[:,3]
Sva_B['Iw']   = result_Sva_B[:,4]
Sva_B['ANu']  = result_Sva_B[:,5]
Sva_B['ANv']  = result_Sva_B[:,6]
Sva_B['ANw']  = result_Sva_B[:,7]


result_SvaB_s = np.zeros((np.size(Cux_Osp1_s),20))
result_SvaB_s[:,0] = U_Sva_s[:,1]
result_SvaB_s[:,1] = Dir_Sva_s[:,1]
result_SvaB_s[:,2] = Iu_Sva_s[:,1]
result_SvaB_s[:,3] = Iv_Sva_s[:,1]
result_SvaB_s[:,4] = Iw_Sva_s[:,1]    
result_SvaB_s[:,5] = ANu_Sva_s[:,1]    
result_SvaB_s[:,6] = ANv_Sva_s[:,1]    
result_SvaB_s[:,7] = ANw_Sva_s[:,1]   

SvaB_s   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
SvaB_s['Time'] = mdates.num2date(Time_Sva_s)
SvaB_s['U']    = result_SvaB_s[:,0]
SvaB_s['Dir']  = result_SvaB_s[:,1]
SvaB_s['Iu']   = result_SvaB_s[:,2]
SvaB_s['Iv']   = result_SvaB_s[:,3]
SvaB_s['Iw']   = result_SvaB_s[:,4]
SvaB_s['ANu']  = result_SvaB_s[:,5]
SvaB_s['ANv']  = result_SvaB_s[:,6]
SvaB_s['ANw']  = result_SvaB_s[:,7]


#%% Syn sensor A
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
Syn   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Syn['Time'] = mdates.num2date(Time_Syn)
Syn['U']    = result_Syn[:,0]
Syn['Dir']  = result_Syn[:,1]
Syn['Iu']   = result_Syn[:,2]
Syn['Iv']   = result_Syn[:,3]
Syn['Iw']   = result_Syn[:,4]
Syn['ANu']  = result_Syn[:,5]
Syn['ANv']  = result_Syn[:,6]
Syn['ANw']  = result_Syn[:,7]


result_Syn_s = np.zeros((np.size(Cux_Osp1_s),20))
result_Syn_s[:,0] = U_Syn_s[:,0]
result_Syn_s[:,1] = Dir_Syn_s[:,0]
result_Syn_s[:,2] = Iu_Syn_s[:,0]
result_Syn_s[:,3] = Iv_Syn_s[:,0]
result_Syn_s[:,4] = Iw_Syn_s[:,0]    
result_Syn_s[:,5] = ANu_Syn_s[:,0]    
result_Syn_s[:,6] = ANv_Syn_s[:,0]    
result_Syn_s[:,7] = ANw_Syn_s[:,0]   

Syn_s   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Syn_s['Time'] = mdates.num2date(Time_Syn_s)
Syn_s['U']    = result_Syn_s[:,0]
Syn_s['Dir']  = result_Syn_s[:,1]
Syn_s['Iu']   = result_Syn_s[:,2]
Syn_s['Iv']   = result_Syn_s[:,3]
Syn_s['Iw']   = result_Syn_s[:,4]
Syn_s['ANu']  = result_Syn_s[:,5]
Syn_s['ANv']  = result_Syn_s[:,6]
Syn_s['ANw']  = result_Syn_s[:,7]


#%% Syn sensor B
result_Syn_B = np.zeros((np.size(Cux_Osp1),8))
result_Syn_B[:,0] = U_Syn[:,1]
result_Syn_B[:,1] = Dir_Syn[:,1]
result_Syn_B[:,2] = Iu_Syn[:,1]
result_Syn_B[:,3] = Iv_Syn[:,1]
result_Syn_B[:,4] = Iw_Syn[:,1]    
result_Syn_B[:,5] = ANu_Syn[:,1]    
result_Syn_B[:,6] = ANv_Syn[:,1]    
result_Syn_B[:,7] = ANw_Syn[:,1]    
result_Syn_B[result_Syn_B==0]  = np.nan
Syn_B   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
Syn_B['Time'] = mdates.num2date(Time_Syn)
Syn_B['U']    = result_Syn_B[:,0]
Syn_B['Dir']  = result_Syn_B[:,1]
Syn_B['Iu']   = result_Syn_B[:,2]
Syn_B['Iv']   = result_Syn_B[:,3]
Syn_B['Iw']   = result_Syn_B[:,4]
Syn_B['ANu']  = result_Syn_B[:,5]
Syn_B['ANv']  = result_Syn_B[:,6]
Syn_B['ANw']  = result_Syn_B[:,7]


result_SynB_s = np.zeros((np.size(Cux_Osp1_s),20))
result_SynB_s[:,0] = U_Syn_s[:,1]
result_SynB_s[:,1] = Dir_Syn_s[:,1]
result_SynB_s[:,2] = Iu_Syn_s[:,1]
result_SynB_s[:,3] = Iv_Syn_s[:,1]
result_SynB_s[:,4] = Iw_Syn_s[:,1]    
result_SynB_s[:,5] = ANu_Syn_s[:,1]    
result_SynB_s[:,6] = ANv_Syn_s[:,1]    
result_SynB_s[:,7] = ANw_Syn_s[:,1]   

SynB_s   = pd.DataFrame(columns=['Time', 'U','Dir', 'Iu','Iv','Iw', 'ANu', 'ANv','ANw'])  
SynB_s['Time'] = mdates.num2date(Time_Syn_s)
SynB_s['U']    = result_SynB_s[:,0]
SynB_s['Dir']  = result_SynB_s[:,1]
SynB_s['Iu']   = result_SynB_s[:,2]
SynB_s['Iv']   = result_SynB_s[:,3]
SynB_s['Iw']   = result_SynB_s[:,4]
SynB_s['ANu']  = result_SynB_s[:,5]
SynB_s['ANv']  = result_SynB_s[:,6]
SynB_s['ANw']  = result_SynB_s[:,7]


#%% collect data with conditions 

# dx=1.5 10deg; dx=2 14.5deg

# get the new data frame with condition of the velocity threshold and distance threshlod for coherences
idx_Osp1 = Osp1.index[(Osp1['U'] > 15) & (Osp1['dx'] > 1.5) & (Osp1['dy'] > 1.5)].tolist()
Osp1_coh_new = Osp1.loc[idx_Osp1]
# get the new data frame with condition of the velocity threshold above 15m/s
idx_Osp1 = Osp1.index[(Osp1['U'] > 15)].tolist()
Osp1_spe_new = Osp1.loc[idx_Osp1]
# get the new data frame with condition of the velocity threshold above 15m/s
idx_Osp1 = Osp1.index[(Osp1['U'] > 20)].tolist()
Osp1_spe20_new = Osp1.loc[idx_Osp1]
# get the new data frame for Osp1B with condition of the velocity threshold above 15m/s
idx_Osp1B = Osp1.index[(Osp1B['U'] > 15)].tolist()
Osp1B_spe_new = Osp1B.loc[idx_Osp1B]
## event by event
# get the new data frame with condition of the velocity threshold and distance threshlod for coherences
idx_Osp1_s = Osp1_s.index[(Osp1_s['U'] > 15) & (Osp1_s['dx'] > 1.5) & (Osp1_s['dy'] > 1.5)].tolist()
Osp1_s_coh_new = Osp1_s.loc[idx_Osp1_s]
# get the new data frame with condition of the velocity threshold above 15m/s
idx_Osp1_s = Osp1_s.index[(Osp1_s['U'] > 15)].tolist()
Osp1_s_spe_new = Osp1_s.loc[idx_Osp1_s]
# get the new data frame with condition of the velocity threshold above 15m/s
idx_Osp1B_s = Osp1_s.index[(Osp1B_s['U'] > 15)].tolist()
Osp1B_s_spe_new = Osp1B_s.loc[idx_Osp1B_s]


# get the new data frame with condition of the velocity threshold and distance threshlod for coherences
idx_Osp2 = Osp2.index[(Osp2['U'] > 15) & (Osp2['dx'] > 1.5) & (Osp2['dy'] > 1.5)].tolist()
Osp2_coh_new = Osp2.loc[idx_Osp2]
# get the new data frame with condition of the velocity threshold above 15m/s
idx_Osp2 = Osp2.index[(Osp2['U'] > 15)].tolist()
Osp2_spe_new = Osp2.loc[idx_Osp2]
# get the new data frame for Osp2B with condition of the velocity threshold above 15m/s
idx_Osp2B = Osp2.index[(Osp2B['U'] > 15)].tolist()
Osp2B_spe_new = Osp2B.loc[idx_Osp2B]
## event by event
# get the new data frame with condition of the velocity threshold and distance threshlod for coherences
idx_Osp2_s = Osp2_s.index[(Osp2_s['U'] > 15) & (Osp2_s['dx'] > 1.5) & (Osp2_s['dy'] > 1.5)].tolist()
Osp2_s_coh_new = Osp2_s.loc[idx_Osp2_s]
# get the new data frame with condition of the velocity threshold above 15m/s
idx_Osp2_s = Osp2_s.index[(Osp2_s['U'] > 15)].tolist()
Osp2_s_spe_new = Osp2_s.loc[idx_Osp2_s]
# get the new data frame with condition of the velocity threshold above 15m/s
idx_Osp2B_s = Osp2_s.index[(Osp2B_s['U'] > 15)].tolist()
Osp2B_s_spe_new = Osp2B_s.loc[idx_Osp2B_s]


# get the new data frame with condition of the velocity threshold above 15m/s
idx_Sva = Sva.index[(Sva['U'] > 15)].tolist()
Sva_spe_new = Sva.loc[idx_Sva]
## event by event
# get the new data frame with condition of the velocity threshold above 15m/s
idx_Sva_s = Sva_s.index[(Sva_s['U'] > 15)].tolist()
Sva_s_spe_new = Sva_s.loc[idx_Sva_s]

# get the new data frame with condition of the velocity threshold above 15m/s
idx_Syn = Syn.index[(Syn['U'] > 15)].tolist()
Syn_spe_new = Syn.loc[idx_Syn]
## event by event
# get the new data frame with condition of the velocity threshold above 15m/s
idx_Syn_s = Syn_s.index[(Syn_s['U'] > 15)].tolist()
Syn_s_spe_new = Syn_s.loc[idx_Syn_s]



#%% rose plots Dir Vs Iu
matplotlib.use('Agg')
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_spe_new['U']
loc_nan = temp.index[(temp > 0)].tolist()

Iu      = Osp1_spe_new['Iu'][loc_nan]
U       = Osp1_spe_new['U'][loc_nan]
Dir     = np.radians(Osp1_spe_new['Dir'][loc_nan])

xy      = np.vstack([np.asarray(Dir),np.asarray(Iu)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Iu)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size,s=z*10, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['ANw'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
z0      = 0.01
Iu_ref  = np.ones(Dir_ref.shape)*(1./np.log(48.8/z0))
ref     = ax.plot(Dir_ref,Iu_ref, 'r-', label='Iu=0.12 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_spe_new['Iu'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'U and $I_u$ from Osp1 sensor A (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'U and Iu from Osp1 sensor A (52 strong wind events).png'
fig.savefig(path + save_tite) 


#%% rose plots Dir Vs Iw
matplotlib.use('Agg')
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_spe_new['U']
loc_nan = temp.index[(temp > 0)].tolist()

Iu      = Osp1_spe_new['Iw'][loc_nan]
U       = Osp1_spe_new['U'][loc_nan]
Dir     = np.radians(Osp1_spe_new['Dir'][loc_nan])

xy      = np.vstack([np.asarray(Dir),np.asarray(Iu)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Iu)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size,s=z*10, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['ANw'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
z0      = 0.01
Iw_ref  = np.ones(Dir_ref.shape)*(1./np.log(48.8/z0))*0.5
ref     = ax.plot(Dir_ref,Iw_ref, 'r-', label='Iw=0.06 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_spe_new['Iw'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'U and $I_w$ from Osp1 sensor A (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'U and Iw from Osp1 sensor A (52 strong wind events).png'
fig.savefig(path + save_tite) 

#%% rose plots Dir Vs Sigma u

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_spe_new['U']
loc_nan = temp.index[(temp > 0)].tolist()

Iu      = Osp1_spe_new['Iu'][loc_nan]*Osp1_spe_new['U'][loc_nan]
U       = Osp1_spe_new['U'][loc_nan]
Dir     = np.radians(Osp1_spe_new['Dir'][loc_nan])

xy      = np.vstack([np.asarray(Dir),np.asarray(Iu)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Iu)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size,s=z*200, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['ANw'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
z0      = 0.01
Iu_ref  = np.ones(Dir_ref.shape)*(1./np.log(48.8/z0))
#ref     = ax.plot(Dir_ref,Iu_ref, 'r-', label='Iu=0.12 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
#ax.set_rmax(Osp1_spe_new['Iu'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'U and $\sigma{_u}$ from Osp1 sensor A (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'U and Sigmau from Osp1 sensor A (52 strong wind events).png'
fig.savefig(path + save_tite) 
#%% rose plots Dir Vs ANu

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_spe_new['ANu']
loc_nan = temp.index[(temp > 0)].tolist()

ANu     = Osp1_spe_new['ANu'][loc_nan]
U       = Osp1_spe_new['U'][loc_nan]
Dir     = np.radians(Osp1_spe_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(ANu)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(ANu)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size,s=z*10000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,ANu,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['ANw'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
ANu_ref = np.ones(Dir_ref.shape)*6.8
ref     = ax.plot(Dir_ref,ANu_ref, 'r-', label='Au=6.8 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_spe_new['ANu'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $A_u$ from Osp1 sensor A (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted A_u from Osp1 sensor A (52 strong wind events).png'
fig.savefig(path + save_tite) 


#%% rose plots Dir Vs ANv

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_spe_new['ANv']
loc_nan = temp.index[(temp > 0)].tolist()

ANv     = Osp1_spe_new['ANv'][loc_nan]
U       = Osp1_spe_new['U'][loc_nan]
Dir     = np.radians(Osp1_spe_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(ANv)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(ANv)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size,s=z*10000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,ANv,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['ANw'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
ANv_ref = np.ones(Dir_ref.shape)*9.4
ref     = ax.plot(Dir_ref,ANv_ref, 'r-', label='Av=9.4 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_spe_new['ANv'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $A_v$ from Osp1 sensor A (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted A_v from Osp1 sensor A (52 strong wind events).png'
fig.savefig(path + save_tite) 


#%% rose plots Dir Vs ANw

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_spe_new['ANw']
loc_nan = temp.index[(temp > 0)].tolist()

ANw     = Osp1_spe_new['ANw'][loc_nan]
U       = Osp1_spe_new['U'][loc_nan]
Dir     = np.radians(Osp1_spe_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(ANw)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(ANw)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size,s=z*10000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,ANw,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['ANw'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
ANw_ref = np.ones(Dir_ref.shape)*9.4
ref     = ax.plot(Dir_ref,ANw_ref, 'r-', label='Aw=9.4 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_spe_new['ANw'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $A_w$ from Osp1 sensor A (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted A_w from Osp1 sensor A (52 strong wind events).png'
fig.savefig(path + save_tite) 


#%% rose plots Dir Vs Cux

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_coh_new['Cux']
loc_nan = temp.index[(temp > 0)].tolist()

Cux     = Osp1_coh_new['Cux'][loc_nan]
U       = Osp1_coh_new['U'][loc_nan]
Dir     = np.radians(Osp1_coh_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(Cux)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Cux)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size, s=z*1000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,Cux,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['Cux'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
Cux_ref = np.ones(Dir_ref.shape)*3
ref     = ax.plot(Dir_ref,Cux_ref, 'r-', label=r'$C_{ux}$=3')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cux'].max()*1.1)
#ax.set_rmax(20)

plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $C_{ux}$ from Osp1 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_ux from Osp1 (52 strong wind events).png'
fig.savefig(path + save_tite) 



#%% rose plots Dir Vs Cuy

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_coh_new['Cuy']
loc_nan = temp.index[(temp > 0)].tolist()

Cuy     = Osp1_coh_new['Cuy'][loc_nan]
U       = Osp1_coh_new['U'][loc_nan]
Dir     = np.radians(Osp1_coh_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(Cuy)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Cuy)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size, s=z*1000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,Cuy,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['Cuy'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
Cuy_ref = np.ones(Dir_ref.shape)*10
ref     = ax.plot(Dir_ref,Cuy_ref, 'r-', label=r'$C_{uy}$=10')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cuy'].max()*1.1)
#ax.set_rmax(20)

plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $C_{uy}$ from Osp1 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_uy from Osp1 (52 strong wind events).png'
fig.savefig(path + save_tite) 


#%% rose plots Dir Vs Cuz

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_coh_new['Cuz']
loc_nan = temp.index[(temp > 0)].tolist()

Cuz     = Osp1_coh_new['Cuz'][loc_nan]
U       = Osp1_coh_new['U'][loc_nan]
Dir     = np.radians(Osp1_coh_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(Cuz)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Cuz)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size, s=z*1000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,Cuz,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['Cuz'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
Cuz_ref = np.ones(Dir_ref.shape)*10
ref     = ax.plot(Dir_ref,Cuz_ref, 'r-', label=r'$C_{uz}$=10')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cuz'].max()*1.1)
#ax.set_rmax(20)

plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $C_{uz}$ from Osp1 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_uz from Osp1 (52 strong wind events).png'
fig.savefig(path + save_tite)


#%% rose plots Dir Vs Cvx

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_coh_new['Cvx']
loc_nan = temp.index[(temp > 0)].tolist()

Cvx     = Osp1_coh_new['Cvx'][loc_nan]
U       = Osp1_coh_new['U'][loc_nan]
Dir     = np.radians(Osp1_coh_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(Cvx)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Cvx)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size, s=z*1000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,Cvx,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['Cvx'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
Cvx_ref = np.ones(Dir_ref.shape)*6
ref     = ax.plot(Dir_ref,Cvx_ref, 'r-', label=r'$C_{vx}$=6')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cvx'].max()*1.1)
ax.set_rmax(6.2)

plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $C_{vx}$ from Osp1 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_vx from Osp1 (52 strong wind events).png'
fig.savefig(path + save_tite)


#%% rose plots Dir Vs Cvy

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_coh_new['Cvy']
loc_nan = temp.index[(temp > 0)].tolist()

Cvy     = Osp1_coh_new['Cvy'][loc_nan]
U       = Osp1_coh_new['U'][loc_nan]
Dir     = np.radians(Osp1_coh_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(Cvy)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Cvy)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size, s=z*1000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,Cvy,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['Cvy'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
Cvy_ref = np.ones(Dir_ref.shape)*6.5
ref     = ax.plot(Dir_ref,Cvy_ref, 'r-', label=r'$C_{vy}$=6.5')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cvy'].max()*1.1)
#ax.set_rmax(20)

plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $C_{vy}$ from Osp1 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_vy from Osp1 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cvz

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_coh_new['Cvz']
loc_nan = temp.index[(temp > 0)].tolist()

Cvz     = Osp1_coh_new['Cvz'][loc_nan]
U       = Osp1_coh_new['U'][loc_nan]
Dir     = np.radians(Osp1_coh_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(Cvz)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Cvz)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size, s=z*1000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,Cvz,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['Cvz'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
Cvz_ref = np.ones(Dir_ref.shape)*6.5
ref     = ax.plot(Dir_ref,Cvz_ref, 'r-', label=r'$C_{vz}$=6.5')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cvz'].max()*1.1)
#ax.set_rmax(20)

plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $C_{vz}$ from Osp1 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_vz from Osp1 (52 strong wind events).png'
fig.savefig(path + save_tite)


#%% rose plots Dir Vs Cwx

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_coh_new['Cwx']
loc_nan = temp.index[(temp > 0)].tolist()

Cwx     = Osp1_coh_new['Cwx'][loc_nan]
U       = Osp1_coh_new['U'][loc_nan]
Dir     = np.radians(Osp1_coh_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(Cwx)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Cwx)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size, s=z*1000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,Cwx,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['Cwx'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
Cwx_ref = np.ones(Dir_ref.shape)*3
ref     = ax.plot(Dir_ref,Cwx_ref, 'r-', label=r'$C_{wx}$=3')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Cwx_ref.max()*1.1)
#ax.set_rmax(20)

plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $C_{wx}$ from Osp1 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_wx from Osp1 (52 strong wind events).png'
fig.savefig(path + save_tite)


#%% rose plots Dir Vs Cwy

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_coh_new['Cwy']
loc_nan = temp.index[(temp > 0)].tolist()

Cwy     = Osp1_coh_new['Cwy'][loc_nan]
U       = Osp1_coh_new['U'][loc_nan]
Dir     = np.radians(Osp1_coh_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(Cwy)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Cwy)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size, s=z*1000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,Cwy,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['Cwy'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
Cwy_ref = np.ones(Dir_ref.shape)*6.5
ref     = ax.plot(Dir_ref,Cwy_ref, 'r-', label=r'$C_{wy}$=6.5')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cwy'].max()*1.1)
#ax.set_rmax(20)

plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $C_{wy}$ from Osp1 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_wy from Osp1 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cwz

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

temp    = Osp1_coh_new['Cwz']
loc_nan = temp.index[(temp > 0)].tolist()

Cwz     = Osp1_coh_new['Cwz'][loc_nan]
U       = Osp1_coh_new['U'][loc_nan]
Dir     = np.radians(Osp1_coh_new['Dir'][loc_nan])

# Calculate the point density
xy      = np.vstack([np.asarray(Dir),np.asarray(Cwz)])
z       = gaussian_kde(xy)(xy)
idx     = U.argsort()
x, y, z = np.asarray(Dir)[idx], np.asarray(Cwz)[idx], z[idx]
size    = np.asarray(U)[idx]

c       =  ax.scatter(x,y,c=size, s=z*1000, alpha=0.75) # the plot has the size of the marker as the density, color the mean wind speed
#c       =  ax.scatter(Dir,Cwz,c=U, alpha=0.75)
#d       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']), Osp1_s_spe_new['Cwz'], c='r', alpha=0.75)

Dir_ref = np.linspace(0,2*np.pi,100)
Cwz_ref = np.ones(Dir_ref.shape)*3
ref     = ax.plot(Dir_ref,Cwz_ref, 'r-', label=r'$C_{wz}$=3')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cwz'].max()*1.1)
#ax.set_rmax(20)

plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)
ax.set_title(r'Fitted $C_{wz}$ from Osp1 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

## plot of the colorbar
#-- obtaining the colormap limits
vmin,vmax = c.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax2, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(r'$\overline{U}_{1h}$ $(m/s)$', fontsize=16)

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_wz from Osp1 (52 strong wind events).png'
fig.savefig(path + save_tite)


#%% SIGMA-U dependency (all six top sensors)

plt.close("all")       
fig     = plt.figure(figsize=(10, 8))
ax1     = plt.subplot(111)
# format the ticks
ax1.plot(Osp1_spe_new['Iu']*Osp1_spe_new['U'],   Osp1_spe_new['U'],   'rs',label='Osp1 A', alpha=0.75)
ax1.plot(Osp1B_spe_new['Iu']*Osp1B_spe_new['U'], Osp1B_spe_new['U'],  'go',label='Osp1 B', alpha=0.75)
ax1.plot(Osp2_spe_new['Iu']*Osp2_spe_new['U'],   Osp2_spe_new['U'],   'b^',label='Osp2 A', alpha=0.75)
ax1.plot(Osp2B_spe_new['Iu']*Osp2B_spe_new['U'],Osp2B_spe_new['U'] ,  'k*',label='Osp2 B', alpha=0.75)
ax1.plot(Sva_spe_new['Iu']*Sva_spe_new['U'],Sva_spe_new['U']       ,  'cp',label='Sva A', alpha=0.75)
ax1.plot(Syn_spe_new['Iu']*Syn_spe_new['U'],Syn_spe_new['U']       ,   'yx',label='Syn A', alpha=0.75)
U_ref   = np.linspace(0,50,100)
z0      = 0.01
Iu_ref  = np.ones(U_ref.shape)*(1./np.log(48.8/z0))
#ax1.plot(U_ref,Iu_ref, 'r-',)# label='N400')
plt.legend(loc='upper left',ncol=2,fontsize=14)
plt.ylabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.xlabel(r'$\sigma{_u}$ $(m/s)$', fontsize=20)
ax1.set_ylim(14,30)
#ax1.set_ylim(0,0.3)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

path      = 'E:/DATA/Plots/'
save_tite = ' SIGMAu-U dependency.png'
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(path + save_tite) 

#%% A dependency (all six top sensors)

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(Osp1_spe_new['U'],   Osp1_spe_new['Iu'],'rs',label='Osp1 A', alpha=0.75)
ax1.plot(Osp1B_spe_new['U'],  Osp1B_spe_new['Iu'],'go',label='Osp1 B', alpha=0.75)
ax1.plot(Osp2_spe_new['U'],   Osp2_spe_new['Iu'],'b^',label='Osp2 A', alpha=0.75)
ax1.plot(Osp2B_spe_new['U'],  Osp2B_spe_new['Iu'],'k*',label='Osp2 B', alpha=0.75)
ax1.plot(Sva_spe_new['U'],    Sva_spe_new['Iu'],'cp',label='Sva A', alpha=0.75)
ax1.plot(Syn_spe_new['U'],    Syn_spe_new['Iu'],'yx',label='Syn A', alpha=0.75)
U_ref   = np.linspace(0,50,100)
z0      = 0.01
Iu_ref  = np.ones(U_ref.shape)*(1./np.log(48.8/z0))
ax1.plot(U_ref,Iu_ref, 'r-',)# label='N400')
plt.legend(loc='upper right',ncol=2,fontsize=14)
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$I_u$', fontsize=20)
ax1.set_xlim(14,30)
ax1.set_ylim(0,0.3)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(Osp1_spe_new['Iu'],    Osp1_spe_new['Iv'],'rs', alpha=0.75)
ax1.plot(Osp1B_spe_new['Iu'],   Osp1B_spe_new['Iv'],'go', alpha=0.75)
ax1.plot(Osp2_spe_new['Iu'],    Osp2_spe_new['Iv'],'b^', alpha=0.75)
ax1.plot(Osp2B_spe_new['Iu'],   Osp2B_spe_new['Iv'],'k*', alpha=0.75)
ax1.plot(Sva_spe_new['Iu'],    Sva_spe_new['Iv'],'cp', alpha=0.75)
ax1.plot(Syn_spe_new['Iu'],    Syn_spe_new['Iv'],'yx', alpha=0.75)
Iu_ref   = np.linspace(0,0.3,100)
Iv_ref   = Iu_ref*0.75
ax1.plot(Iu_ref,Iv_ref, 'r-', label=r'$I_v$=0.75$I_u$')
plt.legend(loc='upper left',ncol=2,fontsize=14)
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$I_v$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(Osp1_spe_new['Iu'],    Osp1_spe_new['Iw'],'rs', alpha=0.75)
ax1.plot(Osp1B_spe_new['Iu'],   Osp1B_spe_new['Iw'],'go',alpha=0.75)
ax1.plot(Osp2_spe_new['Iu'],    Osp2_spe_new['Iw'],'b^', alpha=0.75)
ax1.plot(Osp2B_spe_new['Iu'],   Osp2B_spe_new['Iw'],'k*', alpha=0.75)
ax1.plot(Sva_spe_new['Iu'],    Sva_spe_new['Iw'],'cp', alpha=0.75)
ax1.plot(Syn_spe_new['Iu'],    Syn_spe_new['Iw'],'yx', alpha=0.75)
Iu_ref   = np.linspace(0,0.3,100)
Iw_ref   = Iu_ref*0.5
ax1.plot(Iu_ref,Iw_ref, 'r-', label=r'$I_w$=0.5$I_u$')
plt.legend(loc='upper left',ncol=2,fontsize=14)
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$I_w$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(Osp1_spe_new['U'],   Osp1_spe_new['ANu'],'rs',label='Osp1 A', alpha=0.75)
ax1.plot(Osp1B_spe_new['U'],  Osp1B_spe_new['ANu'],'go',label='Osp1 B', alpha=0.75)
ax1.plot(Osp2_spe_new['U'],   Osp2_spe_new['ANu'],'b^',label='Osp2 A', alpha=0.75)
ax1.plot(Osp2B_spe_new['U'],  Osp2B_spe_new['ANu'],'k*',label='Osp2 B', alpha=0.75)
ax1.plot(Sva_spe_new['U'],    Sva_spe_new['ANu'],'cp',label='Sva A', alpha=0.75)
ax1.plot(Syn_spe_new['U'],    Syn_spe_new['ANu'],'yx',label='Syn A', alpha=0.75)
U_ref   = np.linspace(0,50,100)
z0      = 0.01
Au_ref  = np.ones(U_ref.shape)*6.8
ax1.plot(U_ref,Au_ref, 'r-',)# label='N400')
ax1.set_xlim(14,30)
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$A_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(Osp1_spe_new['ANu'],   Osp1_spe_new['ANv'],'rs',label='Osp1 A', alpha=0.75)
ax1.plot(Osp1B_spe_new['ANu'],  Osp1B_spe_new['ANv'],'go',label='Osp1 B', alpha=0.75)
ax1.plot(Osp2_spe_new['ANu'],   Osp2_spe_new['ANv'],'b^',label='Osp2 A', alpha=0.75)
ax1.plot(Osp2B_spe_new['ANu'],  Osp2B_spe_new['ANv'],'k*',label='Osp2 B', alpha=0.75)
ax1.plot(Sva_spe_new['ANu'],    Sva_spe_new['ANv'],'cp',label='Sva A', alpha=0.75)
ax1.plot(Syn_spe_new['ANu'],    Syn_spe_new['ANv'],'yx',label='Syn A', alpha=0.75)
plt.ylabel(r'$A_v$', fontsize=20)
plt.xlabel(r'$A_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(Osp1_spe_new['ANu'],   Osp1_spe_new['ANw'],'rs',label='Osp1 A', alpha=0.75)
ax1.plot(Osp1B_spe_new['ANu'],  Osp1B_spe_new['ANw'],'go',label='Osp1 B', alpha=0.75)
ax1.plot(Osp2_spe_new['ANu'],   Osp2_spe_new['ANw'],'b^',label='Osp2 A', alpha=0.75)
ax1.plot(Osp2B_spe_new['ANu'],  Osp2B_spe_new['ANw'],'k*',label='Osp2 B', alpha=0.75)
ax1.plot(Sva_spe_new['ANu'],    Sva_spe_new['ANw'],'cp',label='Sva A', alpha=0.75)
ax1.plot(Syn_spe_new['ANu'],    Syn_spe_new['ANw'],'yx',label='Syn A', alpha=0.75)
plt.ylabel(r'$A_w$', fontsize=20)
plt.xlabel(r'$A_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot(Osp1_spe_new['Iu'],   Osp1_spe_new['ANu'],'rs',label='Osp1 A', alpha=0.75)
ax1.plot(Osp1B_spe_new['Iu'],  Osp1B_spe_new['ANu'],'go',label='Osp1 B', alpha=0.75)
ax1.plot(Osp2_spe_new['Iu'],   Osp2_spe_new['ANu'],'b^',label='Osp2 A', alpha=0.75)
ax1.plot(Osp2B_spe_new['Iu'],  Osp2B_spe_new['ANu'],'k*',label='Osp2 B', alpha=0.75)
ax1.plot(Sva_spe_new['Iu'],    Sva_spe_new['ANu'],'cp',label='Sva A', alpha=0.75)
ax1.plot(Syn_spe_new['Iu'],    Syn_spe_new['ANu'],'yx',label='Syn A', alpha=0.75)
plt.ylabel(r'$A_u$', fontsize=20)
plt.xlabel(r'$I_u$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(Osp1_spe_new['Iv'],   Osp1_spe_new['ANv'],'rs',label='Osp1 A', alpha=0.75)
ax1.plot(Osp1B_spe_new['Iv'],  Osp1B_spe_new['ANv'],'go',label='Osp1 B', alpha=0.75)
ax1.plot(Osp2_spe_new['Iv'],   Osp2_spe_new['ANv'],'b^',label='Osp2 A', alpha=0.75)
ax1.plot(Osp2B_spe_new['Iv'],  Osp2B_spe_new['ANv'],'k*',label='Osp2 B', alpha=0.75)
ax1.plot(Sva_spe_new['Iv'],    Sva_spe_new['ANv'],'cp',label='Sva A', alpha=0.75)
ax1.plot(Syn_spe_new['Iv'],    Syn_spe_new['ANv'],'yx',label='Syn A', alpha=0.75)
plt.ylabel(r'$A_v$', fontsize=20)
plt.xlabel(r'$I_v$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(Osp1_spe_new['Iw'],   Osp1_spe_new['ANw'],'rs',label='Osp1 A', alpha=0.75)
ax1.plot(Osp1B_spe_new['Iw'],  Osp1B_spe_new['ANw'],'go',label='Osp1 B', alpha=0.75)
ax1.plot(Osp2_spe_new['Iw'],   Osp2_spe_new['ANw'],'b^',label='Osp2 A', alpha=0.75)
ax1.plot(Osp2B_spe_new['Iw'],  Osp2B_spe_new['ANw'],'k*',label='Osp2 B', alpha=0.75)
ax1.plot(Sva_spe_new['Iw'],    Sva_spe_new['ANw'],'cp',label='Sva A', alpha=0.75)
ax1.plot(Syn_spe_new['Iw'],    Syn_spe_new['ANw'],'yx',label='Syn A', alpha=0.75)
plt.ylabel(r'$A_w$', fontsize=20)
plt.xlabel(r'$I_w$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

path      = 'E:/DATA/Plots/'
save_tite = 'A parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(path + save_tite) 


#%% Cu dependency

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(Osp1_coh_new['U'],Osp1_coh_new['Cux'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['U'],Osp2_coh_new['Cux'],'b^',label='Osp2',alpha=0.75)
plt.legend(loc='best',ncol=1,fontsize=14)
U_ref   = np.linspace(14,30,100)
Cux_ref = np.ones(U_ref.shape)*3
ax1.plot(U_ref,Cux_ref, 'g-',)# label='N400')
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$C_{ux}$', fontsize=20)
ax1.set_xlim(14      , 30)
ax1.set_ylim(0      , Osp1_coh_new['Cux'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(Osp1_coh_new['U'],Osp1_coh_new['Cuy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['U'],Osp2_coh_new['Cuy'],'b^',label='Osp2',alpha=0.75)
U_ref   = np.linspace(14,30,100)
Cuy_ref = np.ones(U_ref.shape)*10
ax1.plot(U_ref,Cuy_ref, 'g-',)# label='N400')
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$C_{uy}$', fontsize=20)
ax1.set_xlim(14      , 30)
ax1.set_ylim(0      , Osp1_coh_new['Cuy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(Osp1_coh_new['U'],Osp1_coh_new['Cuz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['U'],Osp2_coh_new['Cuz'],'b^',label='Osp2',alpha=0.75)
U_ref   = np.linspace(14,30,100)
Cuz_ref = np.ones(U_ref.shape)*10
ax1.plot(U_ref,Cuz_ref, 'g-',)# label='N400')
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$C_{uz}$', fontsize=20)
ax1.set_xlim(14      , 30)
ax1.set_ylim(0      , Osp1_coh_new['Cuz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(Osp1_coh_new['Iu'],Osp1_coh_new['Cux'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Iu'],Osp2_coh_new['Cux'],'b^',label='Osp2',alpha=0.75)
Iu_ref   = np.linspace(0,0.3,100)
Cux_ref = np.ones(Iu_ref.shape)*3
ax1.plot(Iu_ref,Cux_ref, 'g-',)# label='N400')
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$C_{ux}$', fontsize=20)
ax1.set_xlim(0      , 0.30)
ax1.set_ylim(0      , Osp1_coh_new['Cux'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(Osp1_coh_new['Iu'],Osp1_coh_new['Cuy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Iu'],Osp2_coh_new['Cuy'],'b^',label='Osp2',alpha=0.75)
Iu_ref   = np.linspace(0,0.3,100)
Cuy_ref = np.ones(Iu_ref.shape)*10
ax1.plot(Iu_ref,Cuy_ref, 'g-',)# label='N400')
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$C_{uy}$', fontsize=20)
ax1.set_xlim(0      , 0.30)
ax1.set_ylim(0      , Osp1_coh_new['Cuy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(Osp1_coh_new['Iu'],Osp1_coh_new['Cuz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Iu'],Osp2_coh_new['Cuz'],'b^',label='Osp2',alpha=0.75)
Iu_ref   = np.linspace(0,0.3,100)
Cuz_ref = np.ones(Iu_ref.shape)*10
ax1.plot(Iu_ref,Cuz_ref, 'g-',)# label='N400')
plt.xlabel(r'$I_u$', fontsize=20)
plt.ylabel(r'$C_{uz}$', fontsize=20)
ax1.set_xlim(0      , 0.30)
ax1.set_ylim(0      , Osp1_coh_new['Cuz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot(Osp1_coh_new['ANu'],Osp1_coh_new['Cux'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['ANu'],Osp2_coh_new['Cux'],'b^',label='Osp2',alpha=0.75)
Iu_ref   = np.linspace(0,30,100)
Cux_ref = np.ones(Iu_ref.shape)*3
ax1.plot(Iu_ref,Cux_ref, 'g-',)# label='N400')
#ax1.set_xlim(0      , 40)
ax1.set_ylim(0      , Osp1_coh_new['Cux'].max()*1.1)
plt.xlabel(r'$A_u$', fontsize=20)
plt.ylabel(r'$C_{ux}$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(Osp1_coh_new['ANu'],Osp1_coh_new['Cuy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['ANu'],Osp2_coh_new['Cuy'],'b^',label='Osp2',alpha=0.75)
Iu_ref   = np.linspace(0,30,100)
Cuy_ref = np.ones(Iu_ref.shape)*10
ax1.plot(Iu_ref,Cuy_ref, 'g-',)# label='N400')
plt.xlabel(r'$A_u$', fontsize=20)
plt.ylabel(r'$C_{uy}$', fontsize=20)
ax1.set_ylim(0      , Osp1_coh_new['Cuy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(Osp1_coh_new['ANu'],Osp1_coh_new['Cuz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['ANu'],Osp2_coh_new['Cuz'],'b^',label='Osp2',alpha=0.75)
Iu_ref   = np.linspace(0,30,100)
Cuz_ref = np.ones(Iu_ref.shape)*10
ax1.plot(Iu_ref,Cuz_ref, 'g-',)# label='N400')
plt.xlabel(r'$A_u$', fontsize=20)
plt.ylabel(r'$C_{uz}$', fontsize=20)
ax1.set_ylim(0      , Osp1_coh_new['Cuz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

path      = 'E:/DATA/Plots/'
save_tite = 'Cu parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(path + save_tite) 


#%% Cv dependency

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(Osp1_coh_new['U'],Osp1_coh_new['Cvx'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['U'],Osp2_coh_new['Cvx'],'b^',label='Osp2',alpha=0.75)
plt.legend(loc='best',ncol=1,fontsize=14)
U_ref   = np.linspace(14,30,100)
Cvx_ref = np.ones(U_ref.shape)*6
ax1.plot(U_ref,Cvx_ref, 'g-',)# label='N400')
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$C_{vx}$', fontsize=20)
ax1.set_xlim(14      , 30)
ax1.set_ylim(0      , Osp1_coh_new['Cvx'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(Osp1_coh_new['U'],Osp1_coh_new['Cvy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['U'],Osp2_coh_new['Cvy'],'b^',label='Osp2',alpha=0.75)
U_ref   = np.linspace(14,30,100)
Cvy_ref = np.ones(U_ref.shape)*6.5
ax1.plot(U_ref,Cvy_ref, 'g-',)# label='N400')
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$C_{vy}$', fontsize=20)
ax1.set_xlim(14      , 30)
ax1.set_ylim(0      , Osp1_coh_new['Cvy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(Osp1_coh_new['U'],Osp1_coh_new['Cvz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['U'],Osp2_coh_new['Cvz'],'b^',label='Osp2',alpha=0.75)
U_ref   = np.linspace(14,30,100)
Cvz_ref = np.ones(U_ref.shape)*6.5
ax1.plot(U_ref,Cvz_ref, 'g-',)# label='N400')
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$C_{vz}$', fontsize=20)
ax1.set_xlim(14      , 30)
ax1.set_ylim(0      , Osp1_coh_new['Cvz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(Osp1_coh_new['Iv'],Osp1_coh_new['Cvx'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Iv'],Osp2_coh_new['Cvx'],'b^',label='Osp2',alpha=0.75)
Iv_ref   = np.linspace(0,0.3,100)
Cvx_ref = np.ones(Iv_ref.shape)*6
ax1.plot(Iv_ref,Cvx_ref, 'g-',)# label='N400')
plt.xlabel(r'$I_v$', fontsize=20)
plt.ylabel(r'$C_{vx}$', fontsize=20)
ax1.set_xlim(0      , 0.30)
ax1.set_ylim(0      , Osp1_coh_new['Cvx'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(Osp1_coh_new['Iv'],Osp1_coh_new['Cvy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Iv'],Osp2_coh_new['Cvy'],'b^',label='Osp2',alpha=0.75)
Iv_ref   = np.linspace(0,0.3,100)
Cvy_ref = np.ones(Iv_ref.shape)*6.5
ax1.plot(Iv_ref,Cvy_ref, 'g-',)# label='N400')
plt.xlabel(r'$I_v$', fontsize=20)
plt.ylabel(r'$C_{vy}$', fontsize=20)
ax1.set_xlim(0      , 0.30)
ax1.set_ylim(0      , Osp1_coh_new['Cvy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(Osp1_coh_new['Iv'],Osp1_coh_new['Cvz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Iv'],Osp2_coh_new['Cvz'],'b^',label='Osp2',alpha=0.75)
Iv_ref   = np.linspace(0,0.3,100)
Cvz_ref = np.ones(Iv_ref.shape)*6.5
ax1.plot(Iv_ref,Cvz_ref, 'g-',)# label='N400')
plt.xlabel(r'$I_v$', fontsize=20)
plt.ylabel(r'$C_{vz}$', fontsize=20)
ax1.set_xlim(0      , 0.30)
ax1.set_ylim(0      , Osp1_coh_new['Cvz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot(Osp1_coh_new['ANv'],Osp1_coh_new['Cvx'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['ANv'],Osp2_coh_new['Cvx'],'b^',label='Osp2',alpha=0.75)
U_ref   = np.linspace(0,100,100)
Cvx_ref = np.ones(U_ref.shape)*6
ax1.plot(U_ref,Cvx_ref, 'g-',)# label='N400')
#ax1.set_xlim(0      , 40)
ax1.set_ylim(0      , Osp1_coh_new['Cvx'].max()*1.1)
plt.xlabel(r'$A_v$', fontsize=20)
plt.ylabel(r'$C_{vx}$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(Osp1_coh_new['ANv'],Osp1_coh_new['Cvy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['ANv'],Osp2_coh_new['Cvy'],'b^',label='Osp2',alpha=0.75)
U_ref   = np.linspace(0,100,100)
Cvy_ref = np.ones(U_ref.shape)*6.5
ax1.plot(U_ref,Cvy_ref, 'g-',)# label='N400')
plt.xlabel(r'$A_v$', fontsize=20)
plt.ylabel(r'$C_{vy}$', fontsize=20)
ax1.set_ylim(0      , Osp1_coh_new['Cvy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(Osp1_coh_new['ANv'],Osp1_coh_new['Cvz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['ANv'],Osp2_coh_new['Cvz'],'b^',label='Osp2',alpha=0.75)
U_ref   = np.linspace(0,100,100)
Cvz_ref = np.ones(U_ref.shape)*6.5
ax1.plot(U_ref,Cvz_ref, 'g-',)# label='N400')
plt.xlabel(r'$A_v$', fontsize=20)
plt.ylabel(r'$C_{vz}$', fontsize=20)
ax1.set_ylim(0      , Osp1_coh_new['Cvz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

path      = 'E:/DATA/Plots/'
save_tite = 'Cv parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(path + save_tite) 

#%% Cw dependency

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(Osp1_coh_new['U'],Osp1_coh_new['Cwx'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['U'],Osp2_coh_new['Cwx'],'b^',label='Osp2',alpha=0.75)
plt.legend(loc='best',ncol=1,fontsize=14)
U_ref   = np.linspace(14,30,100)
Cwx_ref = np.ones(U_ref.shape)*3
ax1.plot(U_ref,Cwx_ref, 'g-',)# label='N400')
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$C_{wx}$', fontsize=20)
ax1.set_xlim(14      , 30)
ax1.set_ylim(0      , Osp1_coh_new['Cwx'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(Osp1_coh_new['U'],Osp1_coh_new['Cwy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['U'],Osp2_coh_new['Cwy'],'b^',label='Osp2',alpha=0.75)
U_ref   = np.linspace(14,30,100)
Cwy_ref = np.ones(U_ref.shape)*6.5
ax1.plot(U_ref,Cwy_ref, 'g-',)# label='N400')
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$C_{wy}$', fontsize=20)
ax1.set_xlim(14      , 30)
ax1.set_ylim(0      , Osp1_coh_new['Cwy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(Osp1_coh_new['U'],Osp1_coh_new['Cwz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['U'],Osp2_coh_new['Cwz'],'b^',label='Osp2',alpha=0.75)
U_ref   = np.linspace(14,30,100)
Cwz_ref = np.ones(U_ref.shape)*3
ax1.plot(U_ref,Cwz_ref, 'g-',)# label='N400')
plt.xlabel(r'$\overline{U} $ $(m/s)$', fontsize=20)
plt.ylabel(r'$C_{wz}$', fontsize=20)
ax1.set_xlim(14      , 30)
ax1.set_ylim(0      , Osp1_coh_new['Cwz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(Osp1_coh_new['Iw'],Osp1_coh_new['Cwx'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Iw'],Osp2_coh_new['Cwx'],'b^',label='Osp2',alpha=0.75)
Iw_ref   = np.linspace(0,0.3,100)
Cwx_ref = np.ones(Iw_ref.shape)*3
ax1.plot(Iw_ref,Cwx_ref, 'g-',)# label='N400')
plt.xlabel(r'$I_w$', fontsize=20)
plt.ylabel(r'$C_{wx}$', fontsize=20)
ax1.set_xlim(0      , 0.30)
ax1.set_ylim(0      , Osp1_coh_new['Cwx'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(Osp1_coh_new['Iw'],Osp1_coh_new['Cwy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Iw'],Osp2_coh_new['Cwy'],'b^',label='Osp2',alpha=0.75)
Iw_ref   = np.linspace(0,0.3,100)
Cwy_ref = np.ones(Iw_ref.shape)*6.5
ax1.plot(Iw_ref,Cwy_ref, 'g-',)# label='N400')
plt.xlabel(r'$I_w$', fontsize=20)
plt.ylabel(r'$C_{wy}$', fontsize=20)
ax1.set_xlim(0      , 0.30)
ax1.set_ylim(0      , Osp1_coh_new['Cwy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(Osp1_coh_new['Iw'],Osp1_coh_new['Cwz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Iw'],Osp2_coh_new['Cwz'],'b^',label='Osp2',alpha=0.75)
Iw_ref   = np.linspace(0,0.3,100)
Cwz_ref = np.ones(Iw_ref.shape)*3
ax1.plot(Iw_ref,Cwz_ref, 'g-',)# label='N400')
plt.xlabel(r'$I_w$', fontsize=20)
plt.ylabel(r'$C_{wz}$', fontsize=20)
ax1.set_xlim(0      , 0.30)
ax1.set_ylim(0      , Osp1_coh_new['Cwz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot(Osp1_coh_new['ANw'],Osp1_coh_new['Cwx'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['ANw'],Osp2_coh_new['Cwx'],'b^',label='Osp2',alpha=0.75)
Iw_ref   = np.linspace(0,30,100)
Cwx_ref = np.ones(Iw_ref.shape)*3
ax1.plot(Iw_ref,Cwx_ref, 'g-',)# label='N400')
#ax1.set_xlim(0      , 40)
ax1.set_ylim(0      , Osp1_coh_new['Cwx'].max()*1.1)
plt.xlabel(r'$A_w$', fontsize=20)
plt.ylabel(r'$C_{wx}$', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(Osp1_coh_new['ANw'],Osp1_coh_new['Cwy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['ANw'],Osp2_coh_new['Cwy'],'b^',label='Osp2',alpha=0.75)
Iw_ref   = np.linspace(0,30,100)
Cwy_ref = np.ones(Iw_ref.shape)*6.5
ax1.plot(Iw_ref,Cwy_ref, 'g-',)# label='N400')
plt.xlabel(r'$A_w$', fontsize=20)
plt.ylabel(r'$C_{wy}$', fontsize=20)
ax1.set_ylim(0      , Osp1_coh_new['Cwy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(Osp1_coh_new['ANw'],Osp1_coh_new['Cwz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['ANw'],Osp2_coh_new['Cwz'],'b^',label='Osp2',alpha=0.75)
Iw_ref   = np.linspace(0,30,100)
Cwz_ref = np.ones(Iw_ref.shape)*3
ax1.plot(Iw_ref,Cwz_ref, 'g-',)# label='N400')
plt.xlabel(r'$A_w$', fontsize=20)
plt.ylabel(r'$C_{wz}$', fontsize=20)
ax1.set_ylim(0      , Osp1_coh_new['Cwz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

path      = 'E:/DATA/Plots/'
save_tite = 'Cw parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(path + save_tite) 


#%% C dependency

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot(Osp1_coh_new['Cux'],Osp1_coh_new['Cvx'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Cux'],Osp2_coh_new['Cvx'],'b^',label='Osp2',alpha=0.75)
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{ux}$', fontsize=20)
plt.ylabel(r'$C_{vx}$', fontsize=20)
ax1.set_xlim(0      , Osp1_coh_new['Cux'].max()*1.1)
ax1.set_ylim(0      , Osp1_coh_new['Cvx'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(334)
ax1.plot(Osp1_coh_new['Cux'],Osp1_coh_new['Cwx'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Cux'],Osp2_coh_new['Cwx'],'b^',label='Osp2',alpha=0.75)
plt.xlabel(r'$C_{ux}$', fontsize=20)
plt.ylabel(r'$C_{wx}$', fontsize=20)
ax1.set_xlim(0      , Osp1_coh_new['Cux'].max()*1.1)
ax1.set_ylim(0      , Osp1_coh_new['Cwx'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot(Osp1_coh_new['Cux'],Osp1_coh_new['Cuy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Cux'],Osp2_coh_new['Cuy'],'b^',label='Osp2',alpha=0.75)
plt.xlabel(r'$C_{ux}$', fontsize=20)
plt.ylabel(r'$C_{uy}$', fontsize=20)
ax1.set_xlim(0      , Osp1_coh_new['Cux'].max()*1.1)
ax1.set_ylim(0      , Osp1_coh_new['Cuy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot(Osp1_coh_new['Cuy'],Osp1_coh_new['Cvy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Cuy'],Osp2_coh_new['Cvy'],'b^',label='Osp2',alpha=0.75)
plt.xlabel(r'$C_{uy}$', fontsize=20)
plt.ylabel(r'$C_{vy}$', fontsize=20)
ax1.set_xlim(0      , Osp1_coh_new['Cuy'].max()*1.1)
ax1.set_ylim(0      , Osp1_coh_new['Cvy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot(Osp1_coh_new['Cuy'],Osp1_coh_new['Cwy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Cuy'],Osp2_coh_new['Cwy'],'b^',label='Osp2',alpha=0.75)
plt.xlabel(r'$C_{uy}$', fontsize=20)
plt.ylabel(r'$C_{wy}$', fontsize=20)
ax1.set_xlim(0      , Osp1_coh_new['Cuy'].max()*1.1)
ax1.set_ylim(0      , Osp1_coh_new['Cwy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot(Osp1_coh_new['Cvx'],Osp1_coh_new['Cvy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Cvx'],Osp2_coh_new['Cvy'],'b^',label='Osp2',alpha=0.75)
plt.xlabel(r'$C_{vx}$', fontsize=20)
plt.ylabel(r'$C_{vy}$', fontsize=20)
ax1.set_xlim(0      , Osp1_coh_new['Cvx'].max()*1.1)
ax1.set_ylim(0      , Osp1_coh_new['Cvy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

# =============================================================================
# ax1     = plt.subplot(338)
# ax1.plot(Osp1_coh_new['Cux'],Osp1_coh_new['Cuz'],'rs',label='Osp1',alpha=0.75)
# ax1.plot(Osp2_coh_new['Cux'],Osp2_coh_new['Cuz'],'b^',label='Osp2',alpha=0.75)
# plt.xlabel(r'$C_{ux}$', fontsize=20)
# plt.ylabel(r'$C_{uz}$', fontsize=20)
# ax1.set_xlim(0      , Osp1_coh_new['Cux'].max()*1.1)
# ax1.set_ylim(0      , Osp1_coh_new['Cuz'].max()*1.1)
# g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
# g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
# ax1.tick_params(axis='both', labelsize=16)
# plt.minorticks_on()
# plt.show()
# =============================================================================

ax1     = plt.subplot(333)
ax1.plot(Osp1_coh_new['Cuz'],Osp1_coh_new['Cvz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Cuz'],Osp2_coh_new['Cvz'],'b^',label='Osp2',alpha=0.75)
plt.xlabel(r'$C_{uz}$', fontsize=20)
plt.ylabel(r'$C_{vz}$', fontsize=20)
ax1.set_xlim(0      , Osp1_coh_new['Cuz'].max()*1.1)
ax1.set_ylim(0      , Osp1_coh_new['Cvz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot(Osp1_coh_new['Cuz'],Osp1_coh_new['Cwz'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Cuz'],Osp2_coh_new['Cwz'],'b^',label='Osp2',alpha=0.75)
plt.xlabel(r'$C_{uz}$', fontsize=20)
plt.ylabel(r'$C_{wz}$', fontsize=20)
ax1.set_xlim(0      , Osp1_coh_new['Cuz'].max()*1.1)
ax1.set_ylim(0      , Osp1_coh_new['Cwz'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot(Osp1_coh_new['Cwx'],Osp1_coh_new['Cwy'],'rs',label='Osp1',alpha=0.75)
ax1.plot(Osp2_coh_new['Cwx'],Osp2_coh_new['Cwy'],'b^',label='Osp2',alpha=0.75)
plt.xlabel(r'$C_{wx}$', fontsize=20)
plt.ylabel(r'$C_{wy}$', fontsize=20)
ax1.set_xlim(0      , Osp1_coh_new['Cwx'].max()*1.1)
ax1.set_ylim(0      , Osp1_coh_new['Cwy'].max()*1.1)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


# =============================================================================
# ax1     = plt.subplot(339)
# ax1.plot(Osp1_coh_new['Cuy'],Osp1_coh_new['Cuz'],'rs',label='Osp1',alpha=0.75)
# ax1.plot(Osp2_coh_new['Cuy'],Osp2_coh_new['Cuz'],'b^',label='Osp2',alpha=0.75)
# plt.xlabel(r'$C_{uy}$', fontsize=20)
# plt.ylabel(r'$C_{uz}$', fontsize=20)
# ax1.set_xlim(0      , Osp1_coh_new['Cuy'].max()*1.1)
# ax1.set_ylim(0      , Osp1_coh_new['Cuz'].max()*1.1)
# g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
# g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
# ax1.tick_params(axis='both', labelsize=16)
# plt.minorticks_on()
# plt.show()
# =============================================================================

path      = 'E:/DATA/Plots/'
save_tite = 'C parameters dependency.png'
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(path + save_tite)
#%% Histogram of A

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(311)
# format the ticks
ax1.plot([6.8,6.8],[0,75],'r-',label='$A_u$=6.8')
ax1.hist([Osp1_spe_new['ANu'],Osp1B_spe_new['ANu'],Osp2_spe_new['ANu'],Osp2B_spe_new['ANu'],Sva_spe_new['ANu'],Syn_spe_new['ANu']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1 A','Osp1 B','Osp2 A','Osp2 B','Sva A','Syn A'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$A_u$', fontsize=20)
plt.ylabel(r'Number of 1h events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()
fig.suptitle('Distribution of the fitted A parameter from 52 strong wind events', fontsize=25)

ax1     = plt.subplot(312)
ax1.plot([9.4,9.4],[0,80],'r-',label=r'$A_v$=9.4')
ax1.hist([Osp1_spe_new['ANv'],Osp1B_spe_new['ANv'],Osp2_spe_new['ANv'],Osp2B_spe_new['ANv'],Sva_spe_new['ANv'],Syn_spe_new['ANv']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1 A','Osp1 B','Osp2 A','Osp2 B','Sva A','Syn A'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$A_v$', fontsize=20)
plt.ylabel(r'Number of 1h events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(313)
# format the ticks
ax1.plot([9.4,9.4],[0,50],'r-',label=r'$A_w$=9.4')
ax1.hist([Osp1_spe_new['ANw'],Osp1B_spe_new['ANw'],Osp2_spe_new['ANw'],Osp2B_spe_new['ANw'],Sva_spe_new['ANw'],Syn_spe_new['ANw']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1 A','Osp1 B','Osp2 A','Osp2 B','Sva A','Syn A'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$A_w$', fontsize=20)
plt.ylabel(r'Number of 1h events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

path      = 'E:/DATA/Plots/'
save_tite = 'Distribution of A parameters.png'
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(path + save_tite) 

#%% Histogram of C

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(331)
# format the ticks
ax1.plot([3,3],[0,100],'r-',label='$C_{ux}$=3')
ax1.hist([Osp1_coh_new['Cux'],Osp2_coh_new['Cux']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1','Osp2'])
plt.legend(loc='best',ncol=1,fontsize=14)
#ax1.set_xlim(0      , 50)
plt.xlabel(r'$C_{ux}$', fontsize=20)
plt.ylabel(r'Number of 1h events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()
fig.suptitle('Distribution of the fitted C parameter from 52 strong wind events', fontsize=25)

ax1     = plt.subplot(334)
ax1.plot([6,6],[0,70],'r-',label='$C_{vx}$=6')
ax1.hist([Osp1_coh_new['Cvx'],Osp2_coh_new['Cvx']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1','Osp2'])
plt.legend(loc='best',ncol=1,fontsize=14)
#ax1.set_xlim(0      , 50)
plt.xlabel(r'$C_{vx}$', fontsize=20)
plt.ylabel(r'Number of events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(337)
ax1.plot([3,3],[0,170],'r-',label='$C_{wx}$=3')
ax1.hist([Osp1_coh_new['Cwx'],Osp2_coh_new['Cwx']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1','Osp2'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{wx}$', fontsize=20)
#ax1.set_xlim(0      , 50)
plt.ylabel(r'Number of events', fontsize=20)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(332)
ax1.plot([10,10],[0,100],'r-',label='$C_{uy}$=10')
ax1.hist([Osp1_coh_new['Cuy'],Osp2_coh_new['Cuy']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1','Osp2'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{uy}$', fontsize=20)
#ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(335)
ax1.plot([6.5,6.5],[0,70],'r-',label='$C_{vy}$=6.5')
ax1.hist([Osp1_coh_new['Cvy'],Osp2_coh_new['Cvy']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1','Osp2'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{vy}$', fontsize=20)
#ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(338)
ax1.plot([6.5,6.5],[0,40],'r-',label='$C_{wy}$=6.5')
ax1.hist([Osp1_coh_new['Cwy'],Osp2_coh_new['Cwy']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1','Osp2'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{wy}$', fontsize=20)
#ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(333)
ax1.plot([10,10],[0,40],'r-',label='$C_{uz}$=10')
ax1.hist([Osp1_coh_new['Cuz'],Osp2_coh_new['Cuz']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1','Osp2'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{uz}$', fontsize=20)
#ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(336)
ax1.plot([6.5,6.5],[0,40],'r-',label='$C_{vz}$=6.5')
ax1.hist([Osp1_coh_new['Cvz'],Osp2_coh_new['Cvz']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1','Osp2'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{vz}$', fontsize=20)
#ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(339)
ax1.plot([3,3],[0,50],'r-',label='$C_{wz}$=3')
ax1.hist([Osp1_coh_new['Cwz'],Osp2_coh_new['Cwz']],\
         20,cumulative=False, histtype='bar', alpha=0.75,label=['Osp1','Osp2'])
plt.legend(loc='best',ncol=1,fontsize=14)
plt.xlabel(r'$C_{wz}$', fontsize=20)
#ax1.set_xlim(0      , 50)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

path      = 'E:/DATA/Plots/'
save_tite = 'Distribution of C parameters.png'
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(path + save_tite) 


#%% rose plots Dir Vs Iu 6 top sensors
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_spe_new['Dir']),Osp1_spe_new['Iu'],c='r',marker='s', alpha=0.75, label='Osp1 A')
e       =  ax.scatter(np.radians(Osp1B_spe_new['Dir']),Osp1B_spe_new['Iu'],c='g',marker='o',alpha=0.75, label='Osp1 B')
d       =  ax.scatter(np.radians(Osp2_spe_new['Dir']),Osp2_spe_new['Iu'],c='b',marker='^',alpha=0.75, label='Osp2 A')
f       =  ax.scatter(np.radians(Osp2B_spe_new['Dir']),Osp2B_spe_new['Iu'],c='k',marker='*',alpha=0.75, label='Osp2 B')
g       =  ax.scatter(np.radians(Sva_spe_new['Dir']),Sva_spe_new['Iu'],c='c',marker='p',alpha=0.75, label='Sva A')
h       =  ax.scatter(np.radians(Syn_spe_new['Dir']),Syn_spe_new['Iu'],c='y',marker='x',alpha=0.75, label='Syn A')

Dir_ref = np.linspace(0,2*np.pi,100)
z0      = 0.01
Iu_ref  = np.ones(Dir_ref.shape)*(1./np.log(48.8/z0))
ref     = ax.plot(Dir_ref,Iu_ref, 'r-', label='Iu=0.12 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Sva_spe_new['Iu'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'U and $I_u$ from six top sensors (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'U and Iu from six top sensors (52 strong wind events).png'
fig.savefig(path + save_tite)   


#%% rose plots Dir Vs Iw 6 top sensors
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_spe_new['Dir']),Osp1_spe_new['Iw'],c='r',marker='s', alpha=0.75, label='Osp1 A')
e       =  ax.scatter(np.radians(Osp1B_spe_new['Dir']),Osp1B_spe_new['Iw'],c='g',marker='o',alpha=0.75, label='Osp1 B')
d       =  ax.scatter(np.radians(Osp2_spe_new['Dir']),Osp2_spe_new['Iw'],c='b',marker='^',alpha=0.75, label='Osp2 A')
f       =  ax.scatter(np.radians(Osp2B_spe_new['Dir']),Osp2B_spe_new['Iw'],c='k',marker='*',alpha=0.75, label='Osp2 B')
g       =  ax.scatter(np.radians(Sva_spe_new['Dir']),Sva_spe_new['Iw'],c='c',marker='p',alpha=0.75, label='Sva A')
h       =  ax.scatter(np.radians(Syn_spe_new['Dir']),Syn_spe_new['Iw'],c='y',marker='x',alpha=0.75, label='Syn A')

Dir_ref = np.linspace(0,2*np.pi,100)
z0      = 0.01
Iw_ref  = np.ones(Dir_ref.shape)*(1./np.log(48.8/z0))*0.5
ref     = ax.plot(Dir_ref,Iw_ref, 'r-', label='Iw=0.06 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Sva_spe_new['Iw'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'U and $I_w$ from six top sensors (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'U and Iw from six top sensors (52 strong wind events).png'
fig.savefig(path + save_tite)  

#%% rose plots Dir Vs Iu 6 top sensors (1 case per event)

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']),Osp1_s_spe_new['Iu'],c='r',marker='s', alpha=0.75, label='Osp1 A')
e       =  ax.scatter(np.radians(Osp1B_s_spe_new['Dir']),Osp1B_s_spe_new['Iu'],c='g',marker='o',alpha=0.75, label='Osp1 B')
d       =  ax.scatter(np.radians(Osp2_s_spe_new['Dir']),Osp2_s_spe_new['Iu'],c='b',marker='^',alpha=0.75, label='Osp2 A')
f       =  ax.scatter(np.radians(Osp2B_s_spe_new['Dir']),Osp2B_s_spe_new['Iu'],c='k',marker='*',alpha=0.75, label='Osp2 B')
g       =  ax.scatter(np.radians(Sva_s_spe_new['Dir']),Sva_s_spe_new['Iu'],c='c',marker='p',alpha=0.75, label='Sva A')
h       =  ax.scatter(np.radians(Syn_s_spe_new['Dir']),Syn_s_spe_new['Iu'],c='y',marker='x',alpha=0.75, label='Syn A')

Dir_ref = np.linspace(0,2*np.pi,100)
z0      = 0.01
Iu_ref  = np.ones(Dir_ref.shape)*(1./np.log(48.8/z0))
ref     = ax.plot(Dir_ref,Iu_ref, 'r-', label='Iu=0.12 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Sva_spe_new['Iu'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'U and $I_u$ from six top sensors (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'U and Iu from six top sensors (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite) 


#%% rose plots Dir Vs ANu 6 top sensors 
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_spe_new['Dir']),Osp1_spe_new['ANu'],c='r',marker='s', alpha=0.75, label='Osp1 A')
e       =  ax.scatter(np.radians(Osp1B_spe_new['Dir']),Osp1B_spe_new['ANu'],c='g',marker='o',alpha=0.75, label='Osp1 B')
d       =  ax.scatter(np.radians(Osp2_spe_new['Dir']),Osp2_spe_new['ANu'],c='b',marker='^',alpha=0.75, label='Osp2 A')
f       =  ax.scatter(np.radians(Osp2B_spe_new['Dir']),Osp2B_spe_new['ANu'],c='k',marker='*',alpha=0.75, label='Osp2 B')
g       =  ax.scatter(np.radians(Sva_spe_new['Dir']),Sva_spe_new['ANu'],c='c',marker='p',alpha=0.75, label='Sva A')
h       =  ax.scatter(np.radians(Syn_spe_new['Dir']),Syn_spe_new['ANu'],c='y',marker='x',alpha=0.75, label='Syn A')

Dir_ref = np.linspace(0,2*np.pi,100)
ANu_ref = np.ones(Dir_ref.shape)*6.8
ref     = ax.plot(Dir_ref,ANu_ref, 'r-', label='Au=6.8 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Sva_spe_new['ANu'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $A_u$ from six top sensors (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted A_u from six top sensors (52 strong wind events).png'
fig.savefig(path + save_tite) 

#%% rose plots Dir Vs ANu 6 top sensors (1 case per event)

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']),Osp1_s_spe_new['ANu'],c='r',marker='s', alpha=0.75, label='Osp1 A')
e       =  ax.scatter(np.radians(Osp1B_s_spe_new['Dir']),Osp1B_s_spe_new['ANu'],c='g',marker='o',alpha=0.75, label='Osp1 B')
d       =  ax.scatter(np.radians(Osp2_s_spe_new['Dir']),Osp2_s_spe_new['ANu'],c='b',marker='^',alpha=0.75, label='Osp2 A')
f       =  ax.scatter(np.radians(Osp2B_s_spe_new['Dir']),Osp2B_s_spe_new['ANu'],c='k',marker='*',alpha=0.75, label='Osp2 B')
g       =  ax.scatter(np.radians(Sva_s_spe_new['Dir']),Sva_s_spe_new['ANu'],c='c',marker='p',alpha=0.75, label='Sva A')
h       =  ax.scatter(np.radians(Syn_s_spe_new['Dir']),Syn_s_spe_new['ANu'],c='y',marker='x',alpha=0.75, label='Syn A')

Dir_ref = np.linspace(0,2*np.pi,100)
ANu_ref = np.ones(Dir_ref.shape)*6.8
ref     = ax.plot(Dir_ref,ANu_ref, 'r-', label='Au=6.8 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Sva_spe_new['ANu'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $A_u$ from six top sensors (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted A_u from six top sensors (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite) 


#%% rose plots Dir Vs ANv 6 top sensors 
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_spe_new['Dir']),Osp1_spe_new['ANv'],c='r',marker='s', alpha=0.75, label='Osp1 A')
e       =  ax.scatter(np.radians(Osp1B_spe_new['Dir']),Osp1B_spe_new['ANv'],c='g',marker='o',alpha=0.75, label='Osp1 B')
d       =  ax.scatter(np.radians(Osp2_spe_new['Dir']),Osp2_spe_new['ANv'],c='b',marker='^',alpha=0.75, label='Osp2 A')
f       =  ax.scatter(np.radians(Osp2B_spe_new['Dir']),Osp2B_spe_new['ANv'],c='k',marker='*',alpha=0.75, label='Osp2 B')
g       =  ax.scatter(np.radians(Sva_spe_new['Dir']),Sva_spe_new['ANv'],c='c',marker='p',alpha=0.75, label='Sva A')
h       =  ax.scatter(np.radians(Syn_spe_new['Dir']),Syn_spe_new['ANv'],c='y',marker='x',alpha=0.75, label='Syn A')

Dir_ref = np.linspace(0,2*np.pi,100)
ANv_ref = np.ones(Dir_ref.shape)*9.4
ref     = ax.plot(Dir_ref,ANu_ref, 'r-', label='Av=9.4 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_spe_new['ANv'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $A_v$ from six top sensors (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted A_v from six top sensors (52 strong wind events).png'
fig.savefig(path + save_tite)


#%% rose plots Dir Vs ANv 6 top sensors (1 case per event)

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']),Osp1_s_spe_new['ANv'],c='r',marker='s', alpha=0.75, label='Osp1 A')
e       =  ax.scatter(np.radians(Osp1B_s_spe_new['Dir']),Osp1B_s_spe_new['ANv'],c='g',marker='o',alpha=0.75, label='Osp1 B')
d       =  ax.scatter(np.radians(Osp2_s_spe_new['Dir']),Osp2_s_spe_new['ANv'],c='b',marker='^',alpha=0.75, label='Osp2 A')
f       =  ax.scatter(np.radians(Osp2B_s_spe_new['Dir']),Osp2B_s_spe_new['ANv'],c='k',marker='*',alpha=0.75, label='Osp2 B')
g       =  ax.scatter(np.radians(Sva_s_spe_new['Dir']),Sva_s_spe_new['ANv'],c='c',marker='p',alpha=0.75, label='Sva A')
h       =  ax.scatter(np.radians(Syn_s_spe_new['Dir']),Syn_s_spe_new['ANv'],c='y',marker='x',alpha=0.75, label='Syn A')

Dir_ref = np.linspace(0,2*np.pi,100)
ANv_ref = np.ones(Dir_ref.shape)*9.4
ref     = ax.plot(Dir_ref,ANv_ref, 'r-', label='Av=9.4 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Sva_spe_new['ANv'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $A_v$ from six top sensors (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted A_v from six top sensors (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite) 



#%% rose plots Dir Vs ANw 6 top sensors 
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_spe_new['Dir']),Osp1_spe_new['ANw'],c='r',marker='s', alpha=0.75, label='Osp1 A')
e       =  ax.scatter(np.radians(Osp1B_spe_new['Dir']),Osp1B_spe_new['ANw'],c='g',marker='o',alpha=0.75, label='Osp1 B')
d       =  ax.scatter(np.radians(Osp2_spe_new['Dir']),Osp2_spe_new['ANw'],c='b',marker='^',alpha=0.75, label='Osp2 A')
f       =  ax.scatter(np.radians(Osp2B_spe_new['Dir']),Osp2B_spe_new['ANw'],c='k',marker='*',alpha=0.75, label='Osp2 B')
g       =  ax.scatter(np.radians(Sva_spe_new['Dir']),Sva_spe_new['ANw'],c='c',marker='p',alpha=0.75, label='Sva A')
h       =  ax.scatter(np.radians(Syn_spe_new['Dir']),Syn_spe_new['ANw'],c='y',marker='x',alpha=0.75, label='Syn A')

Dir_ref = np.linspace(0,2*np.pi,100)
ANw_ref = np.ones(Dir_ref.shape)*9.4
ref     = ax.plot(Dir_ref,ANw_ref, 'r-', label='Aw=9.4 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_spe_new['ANw'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $A_w$ from six top sensors (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted A_w from six top sensors (52 strong wind events).png'
fig.savefig(path + save_tite)


#%% rose plots Dir Vs ANw 6 top sensors (1 case per event)

plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_spe_new['Dir']),Osp1_s_spe_new['ANw'],c='r',marker='s', alpha=0.75, label='Osp1 A')
e       =  ax.scatter(np.radians(Osp1B_s_spe_new['Dir']),Osp1B_s_spe_new['ANw'],c='g',marker='o',alpha=0.75, label='Osp1 B')
d       =  ax.scatter(np.radians(Osp2_s_spe_new['Dir']),Osp2_s_spe_new['ANw'],c='b',marker='^',alpha=0.75, label='Osp2 A')
f       =  ax.scatter(np.radians(Osp2B_s_spe_new['Dir']),Osp2B_s_spe_new['ANw'],c='k',marker='*',alpha=0.75, label='Osp2 B')
g       =  ax.scatter(np.radians(Sva_s_spe_new['Dir']),Sva_s_spe_new['ANw'],c='c',marker='p',alpha=0.75, label='Sva A')
h       =  ax.scatter(np.radians(Syn_s_spe_new['Dir']),Syn_s_spe_new['ANw'],c='y',marker='x',alpha=0.75, label='Syn A')

Dir_ref = np.linspace(0,2*np.pi,100)
ANw_ref = np.ones(Dir_ref.shape)*9.4
ref     = ax.plot(Dir_ref,ANw_ref, 'r-', label='Aw=9.4 (N400)')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Sva_spe_new['ANw'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $A_w$ from six top sensors (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted A_w from six top sensors (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite) 


#%% rose plots Dir Vs Cux OSP1 and OSP2 (1 case per event)
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_coh_new['Dir']),Osp1_coh_new['Cux'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_coh_new['Dir']),Osp2_coh_new['Cux'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cux_ref = np.ones(Dir_ref.shape)*3
ref     = ax.plot(Dir_ref,Cux_ref, 'r-', label=r'$C_{ux}$=3')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cux'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{ux}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_ux from Osp1 and Osp2 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cux OSP1 and OSP2
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_coh_new['Dir']),Osp1_s_coh_new['Cux'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_s_coh_new['Dir']),Osp2_s_coh_new['Cux'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cux_ref = np.ones(Dir_ref.shape)*3
ref     = ax.plot(Dir_ref,Cux_ref, 'r-', label=r'$C_{ux}$=3')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_s_coh_new['Cux'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{ux}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_ux from Osp1 and Osp2 (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite)


#%% rose plots Dir Vs Cuy OSP1 and OSP2 (1 case per event)
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_coh_new['Dir']),Osp1_coh_new['Cuy'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_coh_new['Dir']),Osp2_coh_new['Cuy'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cuy_ref = np.ones(Dir_ref.shape)*10
ref     = ax.plot(Dir_ref,Cuy_ref, 'r-', label=r'$C_{uy}$=10')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cuy'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{uy}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_uy from Osp1 and Osp2 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cuy OSP1 and OSP2
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_coh_new['Dir']),Osp1_s_coh_new['Cuy'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_s_coh_new['Dir']),Osp2_s_coh_new['Cuy'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cuy_ref = np.ones(Dir_ref.shape)*10
ref     = ax.plot(Dir_ref,Cuy_ref, 'r-', label=r'$C_{uy}$=10')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_s_coh_new['Cuy'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{uy}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_uy from Osp1 and Osp2 (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cuz OSP1 and OSP2 (1 case per event)
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_coh_new['Dir']),Osp1_coh_new['Cuz'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_coh_new['Dir']),Osp2_coh_new['Cuz'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cuz_ref = np.ones(Dir_ref.shape)*10
ref     = ax.plot(Dir_ref,Cuz_ref, 'r-', label=r'$C_{uz}$=10')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cuz'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{uz}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_uz from Osp1 and Osp2 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cuz OSP1 and OSP2
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_coh_new['Dir']),Osp1_s_coh_new['Cuz'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_s_coh_new['Dir']),Osp2_s_coh_new['Cuz'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cuz_ref = np.ones(Dir_ref.shape)*10
ref     = ax.plot(Dir_ref,Cuz_ref, 'r-', label=r'$C_{uz}$=10')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_s_coh_new['Cuz'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{uz}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_uz from Osp1 and Osp2 (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cvx OSP1 and OSP2 (1 case per event)
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_coh_new['Dir']),Osp1_coh_new['Cvx'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_coh_new['Dir']),Osp2_coh_new['Cvx'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cvx_ref = np.ones(Dir_ref.shape)*6
ref     = ax.plot(Dir_ref,Cvx_ref, 'r-', label=r'$C_{vx}$=6')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cvx'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{vx}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_vx from Osp1 and Osp2 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cvx OSP1 and OSP2
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_coh_new['Dir']),Osp1_s_coh_new['Cvx'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_s_coh_new['Dir']),Osp2_s_coh_new['Cvx'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cvx_ref = np.ones(Dir_ref.shape)*6
ref     = ax.plot(Dir_ref,Cvx_ref, 'r-', label=r'$C_{vx}$=6')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_s_coh_new['Cvx'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{vx}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_vx from Osp1 and Osp2 (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cvy OSP1 and OSP2 (1 case per event)
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_coh_new['Dir']),Osp1_coh_new['Cvy'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_coh_new['Dir']),Osp2_coh_new['Cvy'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cvy_ref = np.ones(Dir_ref.shape)*6.5
ref     = ax.plot(Dir_ref,Cvy_ref, 'r-', label=r'$C_{vy}$=6.5')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cvy'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{vy}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_vy from Osp1 and Osp2 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cvy OSP1 and OSP2
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_coh_new['Dir']),Osp1_s_coh_new['Cvy'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_s_coh_new['Dir']),Osp2_s_coh_new['Cvy'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cvy_ref = np.ones(Dir_ref.shape)*6.5
ref     = ax.plot(Dir_ref,Cvy_ref, 'r-', label=r'$C_{vy}$=6.5')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_s_coh_new['Cvy'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{vy}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_vy from Osp1 and Osp2 (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cvz OSP1 and OSP2 (1 case per event)
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_coh_new['Dir']),Osp1_coh_new['Cvz'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_coh_new['Dir']),Osp2_coh_new['Cvz'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cvz_ref = np.ones(Dir_ref.shape)*6.5
ref     = ax.plot(Dir_ref,Cvz_ref, 'r-', label=r'$C_{vz}$=6.5')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cvz'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{vz}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_vz from Osp1 and Osp2 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cvz OSP1 and OSP2
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_coh_new['Dir']),Osp1_s_coh_new['Cvz'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_s_coh_new['Dir']),Osp2_s_coh_new['Cvz'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cvz_ref = np.ones(Dir_ref.shape)*6.5
ref     = ax.plot(Dir_ref,Cvz_ref, 'r-', label=r'$C_{vz}$=6.5')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_s_coh_new['Cvz'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{vz}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_vz from Osp1 and Osp2 (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite)




#%% rose plots Dir Vs Cwx OSP1 and OSP2 (1 case per event)
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_coh_new['Dir']),Osp1_coh_new['Cwx'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_coh_new['Dir']),Osp2_coh_new['Cwx'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cwx_ref = np.ones(Dir_ref.shape)*3
ref     = ax.plot(Dir_ref,Cwx_ref, 'r-', label=r'$C_{wx}$=3')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cwx'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{wx}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_wx from Osp1 and Osp2 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cwx OSP1 and OSP2
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_coh_new['Dir']),Osp1_s_coh_new['Cwx'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_s_coh_new['Dir']),Osp2_s_coh_new['Cwx'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cwx_ref = np.ones(Dir_ref.shape)*3
ref     = ax.plot(Dir_ref,Cwx_ref, 'r-', label=r'$C_{wx}$=3')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_s_coh_new['Cwx'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{wx}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_wx from Osp1 and Osp2 (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cwy OSP1 and OSP2 (1 case per event)
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_coh_new['Dir']),Osp1_coh_new['Cwy'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_coh_new['Dir']),Osp2_coh_new['Cwy'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cwy_ref = np.ones(Dir_ref.shape)*6.5
ref     = ax.plot(Dir_ref,Cwy_ref, 'r-', label=r'$C_{wy}$=6.5')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cwy'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{wy}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_wy from Osp1 and Osp2 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cwy OSP1 and OSP2
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_coh_new['Dir']),Osp1_s_coh_new['Cwy'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_s_coh_new['Dir']),Osp2_s_coh_new['Cwy'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cwy_ref = np.ones(Dir_ref.shape)*6.5
ref     = ax.plot(Dir_ref,Cwy_ref, 'r-', label=r'$C_{wy}$=6.5')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_s_coh_new['Cwy'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{wy}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_wy from Osp1 and Osp2 (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite)




#%% rose plots Dir Vs Cwz OSP1 and OSP2 (1 case per event)
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_coh_new['Dir']),Osp1_coh_new['Cwz'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_coh_new['Dir']),Osp2_coh_new['Cwz'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cwz_ref = np.ones(Dir_ref.shape)*3
ref     = ax.plot(Dir_ref,Cwz_ref, 'r-', label=r'$C_{wz}$=3')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_coh_new['Cwz'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{wz}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_wz from Osp1 and Osp2 (52 strong wind events).png'
fig.savefig(path + save_tite)



#%% rose plots Dir Vs Cwz OSP1 and OSP2
plt.close("all")       
fig     = plt.figure(figsize=(10, 10))
ax      = plt.subplot(111, polar=True)

c       =  ax.scatter(np.radians(Osp1_s_coh_new['Dir']),Osp1_s_coh_new['Cwz'],c='r',marker='s', alpha=0.75, label='Osp1 A')
d       =  ax.scatter(np.radians(Osp2_s_coh_new['Dir']),Osp2_s_coh_new['Cwz'],c='b',marker='^',alpha=0.75, label='Osp2 A')

Dir_ref = np.linspace(0,2*np.pi,100)
Cwz_ref = np.ones(Dir_ref.shape)*3
ref     = ax.plot(Dir_ref,Cwz_ref, 'r-', label=r'$C_{wz}$=3')

ax.set_theta_zero_location('N', offset=360)
ax.set_theta_direction(-1)
ax.set_rmin(0)
ax.set_rmax(Osp1_s_coh_new['Cwz'].max()*1.1)
plt.legend(loc='upper right',ncol=1,fontsize=16)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax.tick_params(axis='both', labelsize=16)

ax.set_title(r'Fitted $C_{wz}$ from Osp1 and Osp2 (52 strong wind events)', fontsize=22)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
plt.minorticks_on()

fig.tight_layout(rect=[0, 0, 0.9, 0.95])

path      = 'E:/DATA/Plots/'
save_tite = 'Fitted C_wz from Osp1 and Osp2 (52 strong wind events, 1 case per event).png'
fig.savefig(path + save_tite)