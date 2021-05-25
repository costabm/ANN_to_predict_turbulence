# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 08:44:42 2020
create raw data files for each storm event

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
import os.path
import matplotlib.dates as mdates
import json
import transformations      as TF

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
def datenum(d):
    return 366 + d.toordinal() + (d - datetime.fromordinal(d.toordinal())).total_seconds()/(24*60*60)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# loop over month
def month_year_iter( start_month, start_year, end_month, end_year ):
    ym_start= 12*start_year + start_month - 1
    ym_end= 12*end_year + end_month 
    loop=[]
    for ym in range( ym_start, ym_end ):
        loop = np.append(loop,divmod( ym, 12 ))
        #y[ym], m[ym] = divmod( ym, 12 )
    loop=loop.reshape((int(loop.size/2),2))    
    return loop

# function to calculate the wind direciton and tranformation matrix from global XYZ to local wind uvw
# local v axis is conti-clockwise 90 deg with respect to u axis
# gloabl X is pointing east and global Y is pointing north
def windXY2dir(X,Y):
    dir_W  =  np.arctan2(X,Y)
    if dir_W<0:
        dir_W=dir_W+2*np.pi
    G_x = [1,0,0]
    G_y = [0,1,0]
    G_z = [0,0,1]
    S_u     = [np.sin(dir_W),        np.cos(dir_W),        0]
    S_v     = [np.sin(dir_W-np.pi/2),np.cos(dir_W-np.pi/2),0]
    S_w     = [0,0,1]
    # transformation matrix from XYZ to UVW     
    TT      = TF.T_xyzXYZ(G_x, G_y, G_z, S_u, S_v, S_w, dim='3x3')        
    return (dir_W,TT)

# function to calculate the wind direction, mean wind speed, local wind uvw components and the time tags
def windXYZ2uvw(data):    
    # calculate the non-mean for each component of the wind components in global XYZ within starting and ending time
    data_temp = np.nanmean(data,0) 
    if ~np.isnan(data.any()):
        [dir_w,tt_w] = windXY2dir(data_temp[0],data_temp[1]) 
        U            = np.matmul(tt_w.T , data_temp)[0]
        uvw          = np.matmul(tt_w.T , data.T).T
        return(dir_w,U,uvw)
    
    
regex_num  = r"[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?" 

z0=0.01   # roughness lengthscale


#%%
cases = pd.read_excel('Storm events.xlsx', sheet_name='Clustered wind storms sort')
delta_1d       = mdates.date2num(datetime(2,1,2,1,0))-mdates.date2num(datetime(2,1,1,1,0))

for i in range(0,cases['Time_storm'].size):
    
    i    = 53 # storm Aina
    time = datetime.strptime(cases['Time_storm'][i], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
    
    time_s  =  mdates.num2date( mdates.date2num(time) - delta_1d*2).replace(tzinfo=None)
    time_e  =  mdates.num2date( mdates.date2num(time) + delta_1d*2).replace(tzinfo=None)

    #Osp1    
    if os.path.isfile('E:/DATA/osp1/' +'Osp1_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')\
        and os.path.isfile('E:/DATA/osp1/' +'Osp1_'+ str(time_e.year) +'_' + str(time_e.month)+ '_time.npz'): 
        # if it is same month, just load data once, otherwise, need to append two months' data 
        if time_s.month==time_e.month:
            A_Time   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')['arr_0'] 
        else:
            A_Time   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')['arr_0']
            temp     = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_e.year) +'_' + str(time_e.month)+ '_time.npz')['arr_0']
            A_Time   = np.append(A_Time,temp,axis=0)        
            
   
        # find the index in the instant data    
        time_S  =  mdates.num2date( mdates.date2num(time) - delta_1d*2).replace(microsecond=0, second=0, minute=00) 
        time_E  =  mdates.num2date( mdates.date2num(time) + delta_1d*2).replace(microsecond=0, second=0, minute=00) 
        # if the starting time is in the time series
        if mdates.date2num(time_S) in A_Time:
            idx_S   = np.where(A_Time == mdates.date2num(time_S))[0][0]
        else:
            idx_S   = np.searchsorted(A_Time[:,0],mdates.date2num(time_S),side='right')
        if mdates.date2num(time_E) in A_Time:            
            idx_E   = np.where(A_Time == mdates.date2num(time_E))[0][0] 
        else:                    
            idx_E   = np.searchsorted(A_Time[:,0],mdates.date2num(time_E),side='right')-1            
        if  idx_S!=idx_E :    
            Time    = A_Time[idx_S:idx_E]
        
            # load raw sensor data
            if time_s.month==time_e.month:
                A   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_s.year) +'_' + str(time_s.month)+ '_A.npz')['arr_0'][idx_S:idx_E,:] 
                B   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_s.year) +'_' + str(time_s.month)+ '_B.npz')['arr_0'][idx_S:idx_E,:] 
                C   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_s.year) +'_' + str(time_s.month)+ '_C.npz')['arr_0'][idx_S:idx_E,:] 
                
            else:
                A   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_s.year) +'_' + str(time_s.month)+ '_A.npz')['arr_0'] 
                temp   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_e.year) +'_' + str(time_e.month)+ '_A.npz')['arr_0'] 
                A   = np.append(A,temp,axis=0)[idx_S:idx_E,:]  
                B   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_s.year) +'_' + str(time_s.month)+ '_B.npz')['arr_0'] 
                temp   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_e.year) +'_' + str(time_e.month)+ '_B.npz')['arr_0'] 
                B   = np.append(B,temp,axis=0)[idx_S:idx_E,:]  
                C   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_s.year) +'_' + str(time_s.month)+ '_C.npz')['arr_0'] 
                temp   = np.load('E:/DATA/osp1/' +'Osp1_'+ str(time_e.year) +'_' + str(time_e.month)+ '_C.npz')['arr_0'] 
                C   = np.append(C,temp,axis=0)[idx_S:idx_E,:]  
        
            # replace the values bigger than 100 to nan   
            flag_A = np.unique(np.where(abs(A)>100)[1])
            A[:,flag_A] = np.nan
            flag_B = np.unique(np.where(abs(B)>100)[1])
            B[:,flag_B] = np.nan
            flag_C = np.unique(np.where(abs(C)>100)[1])
            C[:,flag_C] = np.nan
        
            # save data into a dataframe and then save as a pkl file
            data_s  = pd.DataFrame(columns=['Time_storm', 'A_X', 'A_Y','A_Z', 'B_X', 'B_Y','B_Z','C_X', 'C_Y','C_Z'])    
            data_s['Time_storm']  =    mdates.num2date(Time[:,0], tz=None)
            data_s.loc[data_s.index,['A_X','A_Y','A_Z']] = A
            data_s.loc[data_s.index,['B_X','B_Y','B_Z']] = B
            data_s.loc[data_s.index,['C_X','C_Y','C_Z']] = C
            data_s.to_pickle('E:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
    

    #Osp2    
    if os.path.isfile('E:/DATA/osp2/' +'Osp2_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')\
        and os.path.isfile('E:/DATA/osp2/' +'Osp2_'+ str(time_e.year) +'_' + str(time_e.month)+ '_time.npz'): 
        # if it is same month, just load data once, otherwise, need to append two months' data 
        if time_s.month==time_e.month:
            A_Time   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')['arr_0'] 
        else:
            A_Time   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')['arr_0']
            temp     = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_e.year) +'_' + str(time_e.month)+ '_time.npz')['arr_0']
            A_Time   = np.append(A_Time,temp,axis=0)        
              
        # find the index in the instant data    
        time_S  =  mdates.num2date( mdates.date2num(time) - delta_1d*2).replace(microsecond=0, second=0, minute=00) 
        time_E  =  mdates.num2date( mdates.date2num(time) + delta_1d*2).replace(microsecond=0, second=0, minute=00)     

        if mdates.date2num(time_S) in A_Time:
            idx_S   = np.where(A_Time == mdates.date2num(time_S))[0][0]
        else:
            idx_S   = np.searchsorted(A_Time[:,0],mdates.date2num(time_S),side='right')
        if mdates.date2num(time_E) in A_Time:            
            idx_E   = np.where(A_Time == mdates.date2num(time_E))[0][0] 
        else:                    
            idx_E   = np.searchsorted(A_Time[:,0],mdates.date2num(time_E),side='right')-1            
        if  idx_S!=idx_E :     
            Time    = A_Time[idx_S:idx_E]
        
            # load raw sensor data
            if time_s.month==time_e.month:
                A   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_s.year) +'_' + str(time_s.month)+ '_A.npz')['arr_0'][idx_S:idx_E,:] 
                B   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_s.year) +'_' + str(time_s.month)+ '_B.npz')['arr_0'][idx_S:idx_E,:] 
                C   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_s.year) +'_' + str(time_s.month)+ '_C.npz')['arr_0'][idx_S:idx_E,:] 
                
            else:
                A   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_s.year) +'_' + str(time_s.month)+ '_A.npz')['arr_0'] 
                temp   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_e.year) +'_' + str(time_e.month)+ '_A.npz')['arr_0'] 
                A   = np.append(A,temp,axis=0)[idx_S:idx_E,:]  
                B   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_s.year) +'_' + str(time_s.month)+ '_B.npz')['arr_0'] 
                temp   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_e.year) +'_' + str(time_e.month)+ '_B.npz')['arr_0'] 
                B   = np.append(B,temp,axis=0)[idx_S:idx_E,:]  
                C   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_s.year) +'_' + str(time_s.month)+ '_C.npz')['arr_0'] 
                temp   = np.load('E:/DATA/osp2/' +'Osp2_'+ str(time_e.year) +'_' + str(time_e.month)+ '_C.npz')['arr_0'] 
                C   = np.append(C,temp,axis=0)[idx_S:idx_E,:]  
        
            # replace the values bigger than 100 to nan   
            flag_A = np.unique(np.where(abs(A)>100)[1])
            A[:,flag_A] = np.nan
            flag_B = np.unique(np.where(abs(B)>100)[1])
            B[:,flag_B] = np.nan
            flag_C = np.unique(np.where(abs(C)>100)[1])
            C[:,flag_C] = np.nan
        
            # save data into a dataframe and then save as a pkl file
            data_s  = pd.DataFrame(columns=['Time_storm', 'A_X', 'A_Y','A_Z', 'B_X', 'B_Y','B_Z','C_X', 'C_Y','C_Z'])    
            data_s['Time_storm']  =    mdates.num2date(Time[:,0], tz=None)
            data_s.loc[data_s.index,['A_X','A_Y','A_Z']] = A
            data_s.loc[data_s.index,['B_X','B_Y','B_Z']] = B
            data_s.loc[data_s.index,['C_X','C_Y','C_Z']] = C
            data_s.to_pickle('E:/DATA/osp2/' +'Osp2_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
        
        
    #Svar    
    if os.path.isfile('E:/DATA/svar/' +'Svar_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')\
        and os.path.isfile('E:/DATA/svar/' +'Svar_'+ str(time_e.year) +'_' + str(time_e.month)+ '_time.npz'): 
        # if it is same month, just load data once, otherwise, need to append two months' data 
        if time_s.month==time_e.month:
            A_Time   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')['arr_0'] 
        else:
            A_Time   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')['arr_0']
            temp     = np.load('E:/DATA/svar/' +'Svar_'+ str(time_e.year) +'_' + str(time_e.month)+ '_time.npz')['arr_0']
            A_Time   = np.append(A_Time,temp,axis=0)        
            
        # find the index in the instant data    
        time_S  =  mdates.num2date( mdates.date2num(time) - delta_1d*2).replace(microsecond=0, second=0, minute=00) 
        time_E  =  mdates.num2date( mdates.date2num(time) + delta_1d*2).replace(microsecond=0, second=0, minute=00)     
        if mdates.date2num(time_S) in A_Time:
            idx_S   = np.where(A_Time == mdates.date2num(time_S))[0][0]
        else:
            idx_S   = np.searchsorted(A_Time[:,0],mdates.date2num(time_S),side='right')
        if mdates.date2num(time_E) in A_Time:            
            idx_E   = np.where(A_Time == mdates.date2num(time_E))[0][0] 
        else:                    
            idx_E   = np.searchsorted(A_Time[:,0],mdates.date2num(time_E),side='right')-1            
        if  idx_S!=idx_E :       
            Time    = A_Time[idx_S:idx_E]
        
            # load raw sensor data
            if time_s.month==time_e.month:
                A   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_s.year) +'_' + str(time_s.month)+ '_A.npz')['arr_0'][idx_S:idx_E,:] 
                B   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_s.year) +'_' + str(time_s.month)+ '_B.npz')['arr_0'][idx_S:idx_E,:] 
                C   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_s.year) +'_' + str(time_s.month)+ '_C.npz')['arr_0'][idx_S:idx_E,:] 
                
            else:
                A   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_s.year) +'_' + str(time_s.month)+ '_A.npz')['arr_0'] 
                temp   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_e.year) +'_' + str(time_e.month)+ '_A.npz')['arr_0'] 
                A   = np.append(A,temp,axis=0)[idx_S:idx_E,:]  
                B   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_s.year) +'_' + str(time_s.month)+ '_B.npz')['arr_0'] 
                temp   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_e.year) +'_' + str(time_e.month)+ '_B.npz')['arr_0'] 
                B   = np.append(B,temp,axis=0)[idx_S:idx_E,:]  
                C   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_s.year) +'_' + str(time_s.month)+ '_C.npz')['arr_0'] 
                temp   = np.load('E:/DATA/svar/' +'Svar_'+ str(time_e.year) +'_' + str(time_e.month)+ '_C.npz')['arr_0'] 
                C   = np.append(C,temp,axis=0)[idx_S:idx_E,:]  
        
            # replace the values bigger than 100 to nan   
            flag_A = np.unique(np.where(abs(A)>100)[1])
            A[:,flag_A] = np.nan
            flag_B = np.unique(np.where(abs(B)>100)[1])
            B[:,flag_B] = np.nan
            flag_C = np.unique(np.where(abs(C)>100)[1])
            C[:,flag_C] = np.nan
        
            # save data into a dataframe and then save as a pkl file
            data_s  = pd.DataFrame(columns=['Time_storm', 'A_X', 'A_Y','A_Z', 'B_X', 'B_Y','B_Z','C_X', 'C_Y','C_Z'])    
            data_s['Time_storm']  =    mdates.num2date(Time[:,0], tz=None)
            data_s.loc[data_s.index,['A_X','A_Y','A_Z']] = A
            data_s.loc[data_s.index,['B_X','B_Y','B_Z']] = B
            data_s.loc[data_s.index,['C_X','C_Y','C_Z']] = C
            data_s.to_pickle('E:/DATA/svar/' +'Svar_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
        

    #Synn    
    if os.path.isfile('E:/DATA/synn/' +'Synn_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')\
        and os.path.isfile('E:/DATA/synn/' +'Synn_'+ str(time_e.year) +'_' + str(time_e.month)+ '_time.npz'): 
        # if it is same month, just load data once, otherwise, need to append two months' data 
        if time_s.month==time_e.month:
            A_Time   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')['arr_0'] 
        else:
            A_Time   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_s.year) +'_' + str(time_s.month)+ '_time.npz')['arr_0']
            temp     = np.load('E:/DATA/synn/' +'Synn_'+ str(time_e.year) +'_' + str(time_e.month)+ '_time.npz')['arr_0']
            A_Time   = np.append(A_Time,temp,axis=0)        
    
        # find the index in the instant data    
        time_S  =  mdates.num2date( mdates.date2num(time) - delta_1d*2).replace(microsecond=0, second=0, minute=00) 
        time_E  =  mdates.num2date( mdates.date2num(time) + delta_1d*2).replace(microsecond=0, second=0, minute=00)     
        if mdates.date2num(time_S) in A_Time:
            idx_S   = np.where(A_Time == mdates.date2num(time_S))[0][0]
        else:
            idx_S   = np.searchsorted(A_Time[:,0],mdates.date2num(time_S),side='right')
        if mdates.date2num(time_E) in A_Time:            
            idx_E   = np.where(A_Time == mdates.date2num(time_E))[0][0] 
        else:                    
            idx_E   = np.searchsorted(A_Time[:,0],mdates.date2num(time_E),side='right')-1
        if  idx_S!=idx_E :            
            Time    = A_Time[idx_S:idx_E]
        
            # load raw sensor data
            if time_s.month==time_e.month:
                A   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_s.year) +'_' + str(time_s.month)+ '_A.npz')['arr_0'][idx_S:idx_E,:] 
                B   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_s.year) +'_' + str(time_s.month)+ '_B.npz')['arr_0'][idx_S:idx_E,:] 
                C   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_s.year) +'_' + str(time_s.month)+ '_C.npz')['arr_0'][idx_S:idx_E,:] 
                
            else:
                A   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_s.year) +'_' + str(time_s.month)+ '_A.npz')['arr_0'] 
                temp   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_e.year) +'_' + str(time_e.month)+ '_A.npz')['arr_0'] 
                A   = np.append(A,temp,axis=0)[idx_S:idx_E,:]  
                B   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_s.year) +'_' + str(time_s.month)+ '_B.npz')['arr_0'] 
                temp   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_e.year) +'_' + str(time_e.month)+ '_B.npz')['arr_0'] 
                B   = np.append(B,temp,axis=0)[idx_S:idx_E,:]  
                C   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_s.year) +'_' + str(time_s.month)+ '_C.npz')['arr_0'] 
                temp   = np.load('E:/DATA/synn/' +'Synn_'+ str(time_e.year) +'_' + str(time_e.month)+ '_C.npz')['arr_0'] 
                C   = np.append(C,temp,axis=0)[idx_S:idx_E,:]  
        
            # replace the values bigger than 100 to nan   
            flag_A = np.unique(np.where(abs(A)>100)[1])
            A[:,flag_A] = np.nan
            flag_B = np.unique(np.where(abs(B)>100)[1])
            B[:,flag_B] = np.nan
            flag_C = np.unique(np.where(abs(C)>100)[1])
            C[:,flag_C] = np.nan
        
            # save data into a dataframe and then save as a pkl file
            data_s  = pd.DataFrame(columns=['Time_storm', 'A_X', 'A_Y','A_Z', 'B_X', 'B_Y','B_Z','C_X', 'C_Y','C_Z'])    
            data_s['Time_storm']  =    mdates.num2date(Time[:,0], tz=None)
            data_s.loc[data_s.index,['A_X','A_Y','A_Z']] = A
            data_s.loc[data_s.index,['B_X','B_Y','B_Z']] = B
            data_s.loc[data_s.index,['C_X','C_Y','C_Z']] = C
            data_s.to_pickle('E:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
        
    
    
