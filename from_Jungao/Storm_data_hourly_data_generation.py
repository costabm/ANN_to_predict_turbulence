# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 08:44:42 2020
Storm data visualization and processing

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
import pickle 
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

def wind_dir(X,Y):
    dir_W  =  np.arctan2(X,Y)
    dir_W[dir_W<0] = dir_W[dir_W<0]+2*np.pi   
    return (dir_W)

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
delta_ms         = timedelta(milliseconds=100)


#%%
cases = pd.read_excel('Storm events.xlsx', sheet_name='Clustered wind storms sort')
delta_1d         = mdates.date2num(datetime(2,1,2,1,0))-mdates.date2num(datetime(2,1,1,1,0))
delta_1h         = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
delta_10min      = mdates.date2num(datetime(2,1,1,1,10))-mdates.date2num(datetime(2,1,1,1,0))

for i in range(14,cases['Time_storm'].size):
    
    # i=6, 46, 47 no data for OSP2
    #i    = 15
    time = datetime.strptime(cases['Time_storm'][i], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
    
    date_start  =   mdates.num2date(mdates.date2num(time) - delta_1d*2) 
    date_end    =   mdates.num2date(mdates.date2num(time) + delta_1d*2 - delta_1h) 

# =============================================================================
#     date_start  =   mdates.date2num(time) - delta_1d*2
#     date_end    =   mdates.date2num(time) + delta_1d*2 - delta_1h
#     
# =============================================================================

    #Osp1    
    if os.path.isfile('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl'): 
        Osp1_s = pd.read_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
        Osp2_s = pd.read_pickle('D:/DATA/osp2/' +'Osp2_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
        Svar_s = pd.read_pickle('D:/DATA/svar/' +'Svar_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
        Synn_s = pd.read_pickle('D:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')

        Osp1_s.loc[abs(Osp1_s['A_X'])>100,'A_X'] = np.nan           
        Osp1_s.loc[abs(Osp1_s['A_Y'])>100,'A_Y'] = np.nan
        Osp1_s.loc[abs(Osp1_s['A_Z'])>100,'A_Z'] = np.nan
        Osp1_s.loc[abs(Osp1_s['B_X'])>100,'B_X'] = np.nan
        Osp1_s.loc[abs(Osp1_s['B_Y'])>100,'B_Y'] = np.nan
        Osp1_s.loc[abs(Osp1_s['B_Z'])>100,'B_Z'] = np.nan
        Osp1_s.loc[abs(Osp1_s['C_X'])>100,'C_X'] = np.nan
        Osp1_s.loc[abs(Osp1_s['C_Y'])>100,'C_Y'] = np.nan
        Osp1_s.loc[abs(Osp1_s['C_Z'])>100,'C_Z'] = np.nan 

        Osp2_s.loc[abs(Osp2_s['A_X'])>100,'A_X'] = np.nan           
        Osp2_s.loc[abs(Osp2_s['A_Y'])>100,'A_Y'] = np.nan
        Osp2_s.loc[abs(Osp2_s['A_Z'])>100,'A_Z'] = np.nan
        Osp2_s.loc[abs(Osp2_s['B_X'])>100,'B_X'] = np.nan
        Osp2_s.loc[abs(Osp2_s['B_Y'])>100,'B_Y'] = np.nan
        Osp2_s.loc[abs(Osp2_s['B_Z'])>100,'B_Z'] = np.nan
        Osp2_s.loc[abs(Osp2_s['C_X'])>100,'C_X'] = np.nan
        Osp2_s.loc[abs(Osp2_s['C_Y'])>100,'C_Y'] = np.nan
        Osp2_s.loc[abs(Osp2_s['C_Z'])>100,'C_Z'] = np.nan 
        
        Svar_s.loc[abs(Svar_s['A_X'])>100,'A_X'] = np.nan           
        Svar_s.loc[abs(Svar_s['A_Y'])>100,'A_Y'] = np.nan
        Svar_s.loc[abs(Svar_s['A_Z'])>100,'A_Z'] = np.nan
        Svar_s.loc[abs(Svar_s['B_X'])>100,'B_X'] = np.nan
        Svar_s.loc[abs(Svar_s['B_Y'])>100,'B_Y'] = np.nan
        Svar_s.loc[abs(Svar_s['B_Z'])>100,'B_Z'] = np.nan
        Svar_s.loc[abs(Svar_s['C_X'])>100,'C_X'] = np.nan
        Svar_s.loc[abs(Svar_s['C_Y'])>100,'C_Y'] = np.nan
        Svar_s.loc[abs(Svar_s['C_Z'])>100,'C_Z'] = np.nan

        Synn_s.loc[abs(Synn_s['A_X'])>100,'A_X'] = np.nan           
        Synn_s.loc[abs(Synn_s['A_Y'])>100,'A_Y'] = np.nan
        Synn_s.loc[abs(Synn_s['A_Z'])>100,'A_Z'] = np.nan
        Synn_s.loc[abs(Synn_s['B_X'])>100,'B_X'] = np.nan
        Synn_s.loc[abs(Synn_s['B_Y'])>100,'B_Y'] = np.nan
        Synn_s.loc[abs(Synn_s['B_Z'])>100,'B_Z'] = np.nan
        Synn_s.loc[abs(Synn_s['C_X'])>100,'C_X'] = np.nan
        Synn_s.loc[abs(Synn_s['C_Y'])>100,'C_Y'] = np.nan
        Synn_s.loc[abs(Synn_s['C_Z'])>100,'C_Z'] = np.nan          

    if mdates.num2date(Osp1_s.Time[0]).minute<30:
        date_start = datetime(year=mdates.num2date(Osp1_s.Time[0]).year, month= mdates.num2date(Osp1_s.Time[0]).month, day= mdates.num2date(Osp1_s.Time[0]).day,
                              hour=mdates.num2date(Osp1_s.Time[0]).hour, minute= 30, second=0, microsecond=0 ) 
    else:
        date_start = datetime(year=mdates.num2date(Osp1_s.Time[0]).year, month= mdates.num2date(Osp1_s.Time[0]).month, day= mdates.num2date(Osp1_s.Time[0]).day,
                              hour=(mdates.num2date(Osp1_s.Time[0]).hour+1)%24, minute= 30, second=0, microsecond=0 ) 

    if mdates.num2date(Osp1_s.Time.iloc[-1]).minute<30:
        date_end = datetime(year=mdates.num2date(Osp1_s.Time.iloc[-1]).year, month= mdates.num2date(Osp1_s.Time.iloc[-1]).month, day= mdates.num2date(Osp1_s.Time.iloc[-1]).day,
                              hour=(mdates.num2date(Osp1_s.Time[0]).hour+1)%24, minute= 30, second=0, microsecond=0 ) 
    else:
        date_end = datetime(year=mdates.num2date(Osp1_s.Time.iloc[-1]).year, month= mdates.num2date(Osp1_s.Time.iloc[-1]).month, day= mdates.num2date(Osp1_s.Time.iloc[-1]).day,
                              hour=(mdates.num2date(Osp1_s.Time[0]).hour+1)%24, minute= 30, second=0, microsecond=0 ) 


#%% Osp1
    # here we use 1 hour interval
    num_1h           = int(np.floor((mdates.date2num(date_end)-mdates.date2num(date_start))/delta_1h))

    # create the startng and ending time array  
    time_s = np.array([date_start + timedelta(hours=i) for i in range(0,num_1h)])
    time_e = np.array([date_start + timedelta(hours=i) for i in range(1,num_1h+1)])
      
    # find the cloest index of the starting and ending time of each hour   
    idxs=Osp1_s.loc[:,'Time'].searchsorted (mdates.date2num(time_s),side='right') 
    idxe=Osp1_s.loc[:,'Time'].searchsorted (mdates.date2num(time_e),side='left') 

    time_ref   = []    
    A_dir      = []
    A_Dir      = []    
    A_U        = []
    A_uvw      = []    
    B_dir      = []
    B_Dir      = []        
    B_U        = []
    B_uvw      = []
    C_dir      = []
    C_Dir      = []        
    C_U        = []
    C_uvw      = []
    time_ss    = []     
    
    for j in range(0,num_1h):        
        # if idxs=idxe, then no data avaliable in this hour
        if idxs[j] !=  idxe[j]:
            date_ref  =  mdates.num2date((mdates.date2num(time_s[j]) + mdates.date2num(time_e[j]))/2)
    
            [Dir_w_A,U_A,uvw_A] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_X','A_Y','A_Z']],dtype=np.float64)) 
            dir_w_A             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_Y']],dtype=np.float64)[:,0])
            Dir_w_A=(np.degrees(Dir_w_A)+180) %360
            dir_w_A=(np.degrees(dir_w_A)+180) %360
            A_Dir.append(Dir_w_A)
            A_dir.extend(dir_w_A)
            A_U.append(U_A)
            A_uvw.extend(uvw_A)
            time_ref.append(date_ref)
            time_ss.extend(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['Time']],dtype=np.float64))
            
            [Dir_w_B,U_B,uvw_B] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_X','B_Y','B_Z']],dtype=np.float64))
            dir_w_B             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_Y']],dtype=np.float64)[:,0])            
            Dir_w_B=(np.degrees(Dir_w_B)+180) %360
            dir_w_B=(np.degrees(dir_w_B)+180) %360            
            B_Dir.append(Dir_w_B)
            B_dir.extend(dir_w_B)            
            B_U.append(U_B)
            B_uvw.extend(uvw_B)            

            [Dir_w_C,U_C,uvw_C] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_X','C_Y','C_Z']],dtype=np.float64))
            dir_w_C             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_Y']],dtype=np.float64)[:,0])     
            Dir_w_C=(np.degrees(Dir_w_C)+180) %360
            dir_w_C=(np.degrees(dir_w_C)+180) %360            
            C_Dir.append(Dir_w_C)
            C_dir.extend(dir_w_C)            
            C_U.append(U_C)
            C_uvw.extend(uvw_C)            
            
    A_uvw = np.array(A_uvw)
    B_uvw = np.array(B_uvw)
    C_uvw = np.array(C_uvw)
    A_dir = np.array(A_dir) 
    B_dir = np.array(B_dir) 
    C_dir = np.array(C_dir) 
    time_ss = np.array(time_ss)  
    
    # save instaneous wind data into a dataframe and then save as a pkl file
    Osp1_1h_ins  = pd.DataFrame(columns=['Time', 'A_dir','A_u', 'A_v','A_w','B_dir', 'B_u', 'B_v','B_w','C_dir','C_u', 'C_v','C_w'])    
    Osp1_1h_ins['Time']  =    time_ss[:,0]
    Osp1_1h_ins.loc[Osp1_1h_ins.index,['A_dir']] = A_dir
    Osp1_1h_ins.loc[Osp1_1h_ins.index,['A_u','A_v','A_w']] = A_uvw
    Osp1_1h_ins.loc[Osp1_1h_ins.index,['B_dir']] = B_dir    
    Osp1_1h_ins.loc[Osp1_1h_ins.index,['B_u','B_v','B_w']] = B_uvw
    Osp1_1h_ins.loc[Osp1_1h_ins.index,['C_dir']] = C_dir    
    Osp1_1h_ins.loc[Osp1_1h_ins.index,['C_u','C_v','C_w']] = C_uvw
    
    Osp1_1h_ins.to_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_1h' +'_ins.pkl')
    
    # save mean wind data into a dataframe and then save as a pkl file
    Osp1_1h_mean  = pd.DataFrame(columns=['Time', 'A_Dir','A_U','B_Dir', 'B_U', 'C_Dir','C_U'])    
    Osp1_1h_mean['Time']  =    time_ref
    Osp1_1h_mean.loc[Osp1_1h_mean.index,['A_Dir']] = A_Dir
    Osp1_1h_mean.loc[Osp1_1h_mean.index,['A_U']] = A_U
    Osp1_1h_mean.loc[Osp1_1h_mean.index,['B_Dir']] = B_Dir    
    Osp1_1h_mean.loc[Osp1_1h_mean.index,['B_U']] = B_U
    Osp1_1h_mean.loc[Osp1_1h_mean.index,['C_Dir']] = C_Dir    
    Osp1_1h_mean.loc[Osp1_1h_mean.index,['C_U']] = C_U
    
    Osp1_1h_mean.to_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_1h' +'_mean.pkl')


# =============================================================================
#     Osp1_1h  = {'A_U':  A_U,   'B_U':  B_U,   'C_U':  C_U,   'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir,\
#                 'A_uvw':A_uvw, 'B_uvw':B_uvw, 'C_uvw':C_uvw, 'A_dir':A_dir, 'B_dir':B_dir, 'C_dir':C_dir,\
#                 'time':time_ss, 'Time':time_ref    }  
#     
#     with open('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_1h' +'.pkl', 'wb') as f:    
#         pickle.dump(Osp1_1h, f, pickle.HIGHEST_PROTOCOL)
# =============================================================================
      
# =============================================================================
#     with open('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_1h' +'.pkl', 'rb') as f:    
#         tt=pickle.load( f)
# =============================================================================
#%% Osp1
# here we use 20min interval
    num_20min        = int(np.floor((mdates.date2num(date_end)-mdates.date2num(date_start))/delta_10min/2))        
    # create the startng and ending time array  
    time_s = np.array([date_start + timedelta(minutes=i*20) for i in range(0,num_20min)])
    time_e = np.array([date_start + timedelta(minutes=i*20) for i in range(1,num_20min+1)])
      
    # find the cloest index of the starting and ending time of each hour   
    idxs=Osp1_s.loc[:,'Time'].searchsorted (mdates.date2num(time_s),side='right') 
    idxe=Osp1_s.loc[:,'Time'].searchsorted (mdates.date2num(time_e),side='left') 
        
    time_ref   = []    
    A_dir      = []
    A_Dir      = []    
    A_U        = []
    A_uvw      = []    
    B_dir      = []
    B_Dir      = []        
    B_U        = []
    B_uvw      = []
    C_dir      = []
    C_Dir      = []        
    C_U        = []
    C_uvw      = []
    time_ss    = []     
    
    for j in range(0,num_20min):        
        # if idxs=idxe, then no data avaliable in this hour
        if idxs[j] !=  idxe[j]:
            date_ref  =  mdates.num2date((mdates.date2num(time_s[j]) + mdates.date2num(time_e[j]))/2)
    
            [Dir_w_A,U_A,uvw_A] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_X','A_Y','A_Z']],dtype=np.float64)) 
            dir_w_A             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_Y']],dtype=np.float64)[:,0])
            Dir_w_A=(np.degrees(Dir_w_A)+180) % 360
            dir_w_A=(np.degrees(dir_w_A)+180) % 360
            A_Dir.append(Dir_w_A)
            A_dir.extend(dir_w_A)
            A_U.append(U_A)
            A_uvw.extend(uvw_A)
            time_ref.append(date_ref)
            time_ss.extend(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['Time']],dtype=np.float64))
            
            [Dir_w_B,U_B,uvw_B] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_X','B_Y','B_Z']],dtype=np.float64))
            dir_w_B             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_Y']],dtype=np.float64)[:,0])            
            Dir_w_B=(np.degrees(Dir_w_B)+180) % 360
            dir_w_B=(np.degrees(dir_w_B)+180) % 360            
            B_Dir.append(Dir_w_B)
            B_dir.extend(dir_w_B)            
            B_U.append(U_B)
            B_uvw.extend(uvw_B)            

            [Dir_w_C,U_C,uvw_C] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_X','C_Y','C_Z']],dtype=np.float64))
            dir_w_C             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_Y']],dtype=np.float64)[:,0])     
            Dir_w_C=(np.degrees(Dir_w_C)+180) % 360
            dir_w_C=(np.degrees(dir_w_C)+180) % 360            
            C_Dir.append(Dir_w_C)
            C_dir.extend(dir_w_C)            
            C_U.append(U_C)
            C_uvw.extend(uvw_C)            
            
    A_uvw = np.array(A_uvw)
    B_uvw = np.array(B_uvw)
    C_uvw = np.array(C_uvw)
    A_dir = np.array(A_dir) 
    B_dir = np.array(B_dir) 
    C_dir = np.array(C_dir) 
    time_ss = np.array(time_ss)  
    
    # save instaneous wind data into a dataframe and then save as a pkl file
    Osp1_20min_ins  = pd.DataFrame(columns=['Time', 'A_dir','A_u', 'A_v','A_w','B_dir', 'B_u', 'B_v','B_w','C_dir','C_u', 'C_v','C_w'])    
    Osp1_20min_ins['Time']  =    time_ss[:,0]
    Osp1_20min_ins.loc[Osp1_20min_ins.index,['A_dir']] = A_dir
    Osp1_20min_ins.loc[Osp1_20min_ins.index,['A_u','A_v','A_w']] = A_uvw
    Osp1_20min_ins.loc[Osp1_20min_ins.index,['B_dir']] = B_dir    
    Osp1_20min_ins.loc[Osp1_20min_ins.index,['B_u','B_v','B_w']] = B_uvw
    Osp1_20min_ins.loc[Osp1_20min_ins.index,['C_dir']] = C_dir    
    Osp1_20min_ins.loc[Osp1_20min_ins.index,['C_u','C_v','C_w']] = C_uvw
    
    Osp1_20min_ins.to_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_20min' +'_ins.pkl')
    
    # save mean wind data into a dataframe and then save as a pkl file
    Osp1_20min_mean  = pd.DataFrame(columns=['Time', 'A_Dir','A_U','B_Dir', 'B_U', 'C_Dir','C_U'])    
    Osp1_20min_mean['Time']  =    time_ref
    Osp1_20min_mean.loc[Osp1_20min_mean.index,['A_Dir']] = A_Dir
    Osp1_20min_mean.loc[Osp1_20min_mean.index,['A_U']] = A_U
    Osp1_20min_mean.loc[Osp1_20min_mean.index,['B_Dir']] = B_Dir    
    Osp1_20min_mean.loc[Osp1_20min_mean.index,['B_U']] = B_U
    Osp1_20min_mean.loc[Osp1_20min_mean.index,['C_Dir']] = C_Dir    
    Osp1_20min_mean.loc[Osp1_20min_mean.index,['C_U']] = C_U
    
    Osp1_20min_mean.to_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_20min' +'_mean.pkl')
    
    
    
    
    
# =============================================================================
#     Osp1_20min  = {'A_U':  A_U,   'B_U':  B_U,   'C_U':  C_U,   'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir,\
#                 'A_uvw':A_uvw, 'B_uvw':B_uvw, 'C_uvw':C_uvw, 'A_dir':A_dir, 'B_dir':B_dir, 'C_dir':C_dir,\
#                 'time':time_ss, 'Time':time_ref    }            
#     with open('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_20min' +'.pkl', 'wb') as f:    
#         pickle.dump(Osp1_20min, f, pickle.HIGHEST_PROTOCOL)  
# =============================================================================
        
#%% Osp1
    # here we use 10 mins interval
    num_10min        = int(np.floor((mdates.date2num(date_end)-mdates.date2num(date_start))/delta_10min))
    # create the startng and ending time array  
    time_s = np.array([date_start + timedelta(minutes=i*10) for i in range(0,num_10min)])
    time_e = np.array([date_start + timedelta(minutes=i*10) for i in range(1,num_10min+1)])
      
    # find the cloest index of the starting and ending time of each hour   
    idxs=Osp1_s.loc[:,'Time'].searchsorted (mdates.date2num(time_s),side='right') 
    idxe=Osp1_s.loc[:,'Time'].searchsorted (mdates.date2num(time_e),side='left') 
        
    time_ref   = []    
    A_dir      = []
    A_Dir      = []    
    A_U        = []
    A_uvw      = []    
    B_dir      = []
    B_Dir      = []        
    B_U        = []
    B_uvw      = []
    C_dir      = []
    C_Dir      = []        
    C_U        = []
    C_uvw      = []
    time_ss    = []     
    
    for j in range(0,num_10min):        
        # if idxs=idxe, then no data avaliable in this hour
        if idxs[j] !=  idxe[j]:
            date_ref  =  mdates.num2date((mdates.date2num(time_s[j]) + mdates.date2num(time_e[j]))/2)
    
            [Dir_w_A,U_A,uvw_A] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_X','A_Y','A_Z']],dtype=np.float64)) 
            dir_w_A             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_Y']],dtype=np.float64)[:,0])
            Dir_w_A=(np.degrees(Dir_w_A)+180) %360
            dir_w_A=(np.degrees(dir_w_A)+180) %360
            A_Dir.append(Dir_w_A)
            A_dir.extend(dir_w_A)
            A_U.append(U_A)
            A_uvw.extend(uvw_A)
            time_ref.append(date_ref)
            time_ss.extend(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['Time']],dtype=np.float64))
            
            [Dir_w_B,U_B,uvw_B] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_X','B_Y','B_Z']],dtype=np.float64))
            dir_w_B             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_Y']],dtype=np.float64)[:,0])            
            Dir_w_B=(np.degrees(Dir_w_B)+180) %360
            dir_w_B=(np.degrees(dir_w_B)+180) %360            
            B_Dir.append(Dir_w_B)
            B_dir.extend(dir_w_B)            
            B_U.append(U_B)
            B_uvw.extend(uvw_B)            

            [Dir_w_C,U_C,uvw_C] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_X','C_Y','C_Z']],dtype=np.float64))
            dir_w_C             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_Y']],dtype=np.float64)[:,0])     
            Dir_w_C=(np.degrees(Dir_w_C)+180) %360
            dir_w_C=(np.degrees(dir_w_C)+180) %360            
            C_Dir.append(Dir_w_C)
            C_dir.extend(dir_w_C)            
            C_U.append(U_C)
            C_uvw.extend(uvw_C)            
            
    A_uvw = np.array(A_uvw)
    B_uvw = np.array(B_uvw)
    C_uvw = np.array(C_uvw)
    A_dir = np.array(A_dir) 
    B_dir = np.array(B_dir) 
    C_dir = np.array(C_dir) 
    time_ss = np.array(time_ss)  
    
    
    # save instaneous wind data into a dataframe and then save as a pkl file
    Osp1_10min_ins  = pd.DataFrame(columns=['Time', 'A_dir','A_u', 'A_v','A_w','B_dir', 'B_u', 'B_v','B_w','C_dir','C_u', 'C_v','C_w'])    
    Osp1_10min_ins['Time']  =    time_ss[:,0]
    Osp1_10min_ins.loc[Osp1_10min_ins.index,['A_dir']] = A_dir
    Osp1_10min_ins.loc[Osp1_10min_ins.index,['A_u','A_v','A_w']] = A_uvw
    Osp1_10min_ins.loc[Osp1_10min_ins.index,['B_dir']] = B_dir    
    Osp1_10min_ins.loc[Osp1_10min_ins.index,['B_u','B_v','B_w']] = B_uvw
    Osp1_10min_ins.loc[Osp1_10min_ins.index,['C_dir']] = C_dir    
    Osp1_10min_ins.loc[Osp1_10min_ins.index,['C_u','C_v','C_w']] = C_uvw
    
    Osp1_10min_ins.to_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_10min' +'_ins.pkl')
    
    # save mean wind data into a dataframe and then save as a pkl file
    Osp1_10min_mean  = pd.DataFrame(columns=['Time', 'A_Dir','A_U','B_Dir', 'B_U', 'C_Dir','C_U'])    
    Osp1_10min_mean['Time']  =    time_ref
    Osp1_10min_mean.loc[Osp1_10min_mean.index,['A_Dir']] = A_Dir
    Osp1_10min_mean.loc[Osp1_10min_mean.index,['A_U']] = A_U
    Osp1_10min_mean.loc[Osp1_10min_mean.index,['B_Dir']] = B_Dir    
    Osp1_10min_mean.loc[Osp1_10min_mean.index,['B_U']] = B_U
    Osp1_10min_mean.loc[Osp1_10min_mean.index,['C_Dir']] = C_Dir    
    Osp1_10min_mean.loc[Osp1_10min_mean.index,['C_U']] = C_U
    
    Osp1_10min_mean.to_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_10min' +'_mean.pkl')
       
    
    
    
# =============================================================================
#     Osp1_10min  = {'A_U':  A_U,   'B_U':  B_U,   'C_U':  C_U,   'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir,\
#                 'A_uvw':A_uvw, 'B_uvw':B_uvw, 'C_uvw':C_uvw, 'A_dir':A_dir, 'B_dir':B_dir, 'C_dir':C_dir,\
#                 'time':time_ss, 'Time':time_ref    }            
#     with open('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_10min' +'.pkl', 'wb') as f:    
#         pickle.dump(Osp1_10min, f, pickle.HIGHEST_PROTOCOL)    
# =============================================================================

#%% Osp1
    # here we use 3 hour interval
    num_3h           = int(np.floor((mdates.date2num(date_end)-mdates.date2num(date_start))/delta_1h/3))

    # create the startng and ending time array  
    time_s = np.array([date_start + timedelta(hours=i*3) for i in range(0,num_3h)])
    time_e = np.array([date_start + timedelta(hours=i*3) for i in range(1,num_3h+1)])
      
    # find the cloest index of the starting and ending time of each hour   
    idxs=Osp1_s.loc[:,'Time'].searchsorted (mdates.date2num(time_s),side='right') 
    idxe=Osp1_s.loc[:,'Time'].searchsorted (mdates.date2num(time_e),side='left') 

    time_ref   = []    
    A_dir      = []
    A_Dir      = []    
    A_U        = []
    A_uvw      = []    
    B_dir      = []
    B_Dir      = []        
    B_U        = []
    B_uvw      = []
    C_dir      = []
    C_Dir      = []        
    C_U        = []
    C_uvw      = []
    time_ss    = []     
    
    for j in range(0,num_3h):        
        # if idxs=idxe, then no data avaliable in this hour
        if idxs[j] !=  idxe[j]:
            date_ref  =  mdates.num2date((mdates.date2num(time_s[j]) + mdates.date2num(time_e[j]))/2)
    
            [Dir_w_A,U_A,uvw_A] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_X','A_Y','A_Z']],dtype=np.float64)) 
            dir_w_A             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['A_Y']],dtype=np.float64)[:,0])
            Dir_w_A=(np.degrees(Dir_w_A)+180) %360
            dir_w_A=(np.degrees(dir_w_A)+180) %360
            A_Dir.append(Dir_w_A)
            A_dir.extend(dir_w_A)
            A_U.append(U_A)
            A_uvw.extend(uvw_A)
            time_ref.append(date_ref)
            time_ss.extend(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['Time']],dtype=np.float64))
            
            [Dir_w_B,U_B,uvw_B] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_X','B_Y','B_Z']],dtype=np.float64))
            dir_w_B             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['B_Y']],dtype=np.float64)[:,0])            
            Dir_w_B=(np.degrees(Dir_w_B)+180) %360
            dir_w_B=(np.degrees(dir_w_B)+180) %360            
            B_Dir.append(Dir_w_B)
            B_dir.extend(dir_w_B)            
            B_U.append(U_B)
            B_uvw.extend(uvw_B)            

            [Dir_w_C,U_C,uvw_C] = windXYZ2uvw(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_X','C_Y','C_Z']],dtype=np.float64))
            dir_w_C             = wind_dir(np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_X']],dtype=np.float64)[:,0],np.asarray(Osp1_s.loc[idxs[j]:idxe[j],['C_Y']],dtype=np.float64)[:,0])     
            Dir_w_C=(np.degrees(Dir_w_C)+180) %360
            dir_w_C=(np.degrees(dir_w_C)+180) %360            
            C_Dir.append(Dir_w_C)
            C_dir.extend(dir_w_C)            
            C_U.append(U_C)
            C_uvw.extend(uvw_C)            
            
    A_uvw = np.array(A_uvw)
    B_uvw = np.array(B_uvw)
    C_uvw = np.array(C_uvw)
    A_dir = np.array(A_dir) 
    B_dir = np.array(B_dir) 
    C_dir = np.array(C_dir) 
    time_ss = np.array(time_ss)  

    
    # save instaneous wind data into a dataframe and then save as a pkl file
    Osp1_3h_ins  = pd.DataFrame(columns=['Time', 'A_dir','A_u', 'A_v','A_w','B_dir', 'B_u', 'B_v','B_w','C_dir','C_u', 'C_v','C_w'])    
    Osp1_3h_ins['Time']  =    time_ss[:,0]
    Osp1_3h_ins.loc[Osp1_3h_ins.index,['A_dir']] = A_dir
    Osp1_3h_ins.loc[Osp1_3h_ins.index,['A_u','A_v','A_w']] = A_uvw
    Osp1_3h_ins.loc[Osp1_3h_ins.index,['B_dir']] = B_dir    
    Osp1_3h_ins.loc[Osp1_3h_ins.index,['B_u','B_v','B_w']] = B_uvw
    Osp1_3h_ins.loc[Osp1_3h_ins.index,['C_dir']] = C_dir    
    Osp1_3h_ins.loc[Osp1_3h_ins.index,['C_u','C_v','C_w']] = C_uvw
    
    Osp1_3h_ins.to_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_3h' +'_ins.pkl')
    
    # save mean wind data into a dataframe and then save as a pkl file
    Osp1_3h_mean  = pd.DataFrame(columns=['Time', 'A_Dir','A_U','B_Dir', 'B_U', 'C_Dir','C_U'])    
    Osp1_3h_mean['Time']  =    time_ref
    Osp1_3h_mean.loc[Osp1_3h_mean.index,['A_Dir']] = A_Dir
    Osp1_3h_mean.loc[Osp1_3h_mean.index,['A_U']] = A_U
    Osp1_3h_mean.loc[Osp1_3h_mean.index,['B_Dir']] = B_Dir    
    Osp1_3h_mean.loc[Osp1_3h_mean.index,['B_U']] = B_U
    Osp1_3h_mean.loc[Osp1_3h_mean.index,['C_Dir']] = C_Dir    
    Osp1_3h_mean.loc[Osp1_3h_mean.index,['C_U']] = C_U
    
    Osp1_3h_mean.to_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_3h' +'_mean.pkl')
   


# =============================================================================
#     Osp1_3h  = {'A_U':  A_U,   'B_U':  B_U,   'C_U':  C_U,   'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir,\
#                 'A_uvw':A_uvw, 'B_uvw':B_uvw, 'C_uvw':C_uvw, 'A_dir':A_dir, 'B_dir':B_dir, 'C_dir':C_dir,\
#                 'time':time_ss, 'Time':time_ref    } 
#     with open('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_3h' +'.pkl', 'wb') as f:    
#         pickle.dump(Osp1_3h, f, pickle.HIGHEST_PROTOCOL)    
# 
# =============================================================================

#%% Osp2
    # here we use 1 hour interval
    num_1h           = int(np.floor((mdates.date2num(date_end)-mdates.date2num(date_start))/delta_1h))

    # create the startng and ending time array  
    time_s = np.array([date_start + timedelta(hours=i) for i in range(0,num_1h)])
    time_e = np.array([date_start + timedelta(hours=i) for i in range(1,num_1h+1)])
      
    # find the cloest index of the starting and ending time of each hour   
    idxs=Osp2_s.loc[:,'Time'].searchsorted (mdates.date2num(time_s),side='right') 
    idxe=Osp2_s.loc[:,'Time'].searchsorted (mdates.date2num(time_e),side='left') 

    time_ref   = []    
    A_dir      = []
    A_Dir      = []    
    A_U        = []
    A_uvw      = []    
    B_dir      = []
    B_Dir      = []        
    B_U        = []
    B_uvw      = []
    C_dir      = []
    C_Dir      = []        
    C_U        = []
    C_uvw      = []
    time_ss    = []     
    
    for j in range(0,num_1h):        
        # if idxs=idxe, then no data avaliable in this hour
        if idxs[j] !=  idxe[j]:
            date_ref  =  mdates.num2date((mdates.date2num(time_s[j]) + mdates.date2num(time_e[j]))/2)
    
            [Dir_w_A,U_A,uvw_A] = windXYZ2uvw(np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['A_X','A_Y','A_Z']],dtype=np.float64)) 
            dir_w_A             = wind_dir(np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['A_X']],dtype=np.float64)[:,0],np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['A_Y']],dtype=np.float64)[:,0])
            Dir_w_A=(np.degrees(Dir_w_A)+180) %360
            dir_w_A=(np.degrees(dir_w_A)+180) %360
            A_Dir.append(Dir_w_A)
            A_dir.extend(dir_w_A)
            A_U.append(U_A)
            A_uvw.extend(uvw_A)
            time_ref.append(date_ref)
            time_ss.extend(np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['Time']],dtype=np.float64))
            
            [Dir_w_B,U_B,uvw_B] = windXYZ2uvw(np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['B_X','B_Y','B_Z']],dtype=np.float64))
            dir_w_B             = wind_dir(np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['B_X']],dtype=np.float64)[:,0],np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['B_Y']],dtype=np.float64)[:,0])            
            Dir_w_B=(np.degrees(Dir_w_B)+180) %360
            dir_w_B=(np.degrees(dir_w_B)+180) %360            
            B_Dir.append(Dir_w_B)
            B_dir.extend(dir_w_B)            
            B_U.append(U_B)
            B_uvw.extend(uvw_B)            

            [Dir_w_C,U_C,uvw_C] = windXYZ2uvw(np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['C_X','C_Y','C_Z']],dtype=np.float64))
            dir_w_C             = wind_dir(np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['C_X']],dtype=np.float64)[:,0],np.asarray(Osp2_s.loc[idxs[j]:idxe[j],['C_Y']],dtype=np.float64)[:,0])     
            Dir_w_C=(np.degrees(Dir_w_C)+180) %360
            dir_w_C=(np.degrees(dir_w_C)+180) %360            
            C_Dir.append(Dir_w_C)
            C_dir.extend(dir_w_C)            
            C_U.append(U_C)
            C_uvw.extend(uvw_C)            
            
    A_uvw = np.array(A_uvw)
    B_uvw = np.array(B_uvw)
    C_uvw = np.array(C_uvw)
    A_dir = np.array(A_dir) 
    B_dir = np.array(B_dir) 
    C_dir = np.array(C_dir) 
    time_ss = np.array(time_ss)  
    
    # save instaneous wind data into a dataframe and then save as a pkl file
    Osp2_1h_ins  = pd.DataFrame(columns=['Time', 'A_dir','A_u', 'A_v','A_w','B_dir', 'B_u', 'B_v','B_w','C_dir','C_u', 'C_v','C_w'])    
    Osp2_1h_ins['Time']  =    time_ss[:,0]
    Osp2_1h_ins.loc[Osp2_1h_ins.index,['A_dir']] = A_dir
    Osp2_1h_ins.loc[Osp2_1h_ins.index,['A_u','A_v','A_w']] = A_uvw
    Osp2_1h_ins.loc[Osp2_1h_ins.index,['B_dir']] = B_dir    
    Osp2_1h_ins.loc[Osp2_1h_ins.index,['B_u','B_v','B_w']] = B_uvw
    Osp2_1h_ins.loc[Osp2_1h_ins.index,['C_dir']] = C_dir    
    Osp2_1h_ins.loc[Osp2_1h_ins.index,['C_u','C_v','C_w']] = C_uvw
    
    Osp2_1h_ins.to_pickle('D:/DATA/osp2/' +'Osp2_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp2_1h' +'_ins.pkl')
    
    # save mean wind data into a dataframe and then save as a pkl file
    Osp2_1h_mean  = pd.DataFrame(columns=['Time', 'A_Dir','A_U','B_Dir', 'B_U', 'C_Dir','C_U'])    
    Osp2_1h_mean['Time']  =    time_ref
    Osp2_1h_mean.loc[Osp2_1h_mean.index,['A_Dir']] = A_Dir
    Osp2_1h_mean.loc[Osp2_1h_mean.index,['A_U']] = A_U
    Osp2_1h_mean.loc[Osp2_1h_mean.index,['B_Dir']] = B_Dir    
    Osp2_1h_mean.loc[Osp2_1h_mean.index,['B_U']] = B_U
    Osp2_1h_mean.loc[Osp2_1h_mean.index,['C_Dir']] = C_Dir    
    Osp2_1h_mean.loc[Osp2_1h_mean.index,['C_U']] = C_U
    
    Osp2_1h_mean.to_pickle('D:/DATA/osp2/' +'Osp2_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp2_1h' +'_mean.pkl')
       
    
    

# =============================================================================
#     Osp2_1h  = {'A_U':  A_U,   'B_U':  B_U,   'C_U':  C_U,   'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir,\
#                 'A_uvw':A_uvw, 'B_uvw':B_uvw, 'C_uvw':C_uvw, 'A_dir':A_dir, 'B_dir':B_dir, 'C_dir':C_dir,\
#                 'time':time_ss, 'Time':time_ref    }  
#     
#     with open('D:/DATA/osp2/' +'Osp2_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp2_1h' +'.pkl', 'wb') as f:    
#         pickle.dump(Osp2_1h, f, pickle.HIGHEST_PROTOCOL)         
# =============================================================================
   
#%% Svar
    # here we use 1 hour interval
    num_1h           = int(np.floor((mdates.date2num(date_end)-mdates.date2num(date_start))/delta_1h))

    # create the startng and ending time array  
    time_s = np.array([date_start + timedelta(hours=i) for i in range(0,num_1h)])
    time_e = np.array([date_start + timedelta(hours=i) for i in range(1,num_1h+1)])
      
    # find the cloest index of the starting and ending time of each hour   
    idxs=Svar_s.loc[:,'Time'].searchsorted (mdates.date2num(time_s),side='right') 
    idxe=Svar_s.loc[:,'Time'].searchsorted (mdates.date2num(time_e),side='left') 

    time_ref   = []    
    A_dir      = []
    A_Dir      = []    
    A_U        = []
    A_uvw      = []    
    B_dir      = []
    B_Dir      = []        
    B_U        = []
    B_uvw      = []
    C_dir      = []
    C_Dir      = []        
    C_U        = []
    C_uvw      = []
    time_ss    = []     
    
    for j in range(0,num_1h):        
        # if idxs=idxe, then no data avaliable in this hour
        if idxs[j] !=  idxe[j]:
            date_ref  =  mdates.num2date((mdates.date2num(time_s[j]) + mdates.date2num(time_e[j]))/2)
    
            [Dir_w_A,U_A,uvw_A] = windXYZ2uvw(np.asarray(Svar_s.loc[idxs[j]:idxe[j],['A_X','A_Y','A_Z']],dtype=np.float64)) 
            dir_w_A             = wind_dir(np.asarray(Svar_s.loc[idxs[j]:idxe[j],['A_X']],dtype=np.float64)[:,0],np.asarray(Svar_s.loc[idxs[j]:idxe[j],['A_Y']],dtype=np.float64)[:,0])
            Dir_w_A=(np.degrees(Dir_w_A)+180) %360
            dir_w_A=(np.degrees(dir_w_A)+180) %360
            A_Dir.append(Dir_w_A)
            A_dir.extend(dir_w_A)
            A_U.append(U_A)
            A_uvw.extend(uvw_A)
            time_ref.append(date_ref)
            time_ss.extend(np.asarray(Svar_s.loc[idxs[j]:idxe[j],['Time']],dtype=np.float64))
            
            [Dir_w_B,U_B,uvw_B] = windXYZ2uvw(np.asarray(Svar_s.loc[idxs[j]:idxe[j],['B_X','B_Y','B_Z']],dtype=np.float64))
            dir_w_B             = wind_dir(np.asarray(Svar_s.loc[idxs[j]:idxe[j],['B_X']],dtype=np.float64)[:,0],np.asarray(Svar_s.loc[idxs[j]:idxe[j],['B_Y']],dtype=np.float64)[:,0])            
            Dir_w_B=(np.degrees(Dir_w_B)+180) %360
            dir_w_B=(np.degrees(dir_w_B)+180) %360            
            B_Dir.append(Dir_w_B)
            B_dir.extend(dir_w_B)            
            B_U.append(U_B)
            B_uvw.extend(uvw_B)            

            [Dir_w_C,U_C,uvw_C] = windXYZ2uvw(np.asarray(Svar_s.loc[idxs[j]:idxe[j],['C_X','C_Y','C_Z']],dtype=np.float64))
            dir_w_C             = wind_dir(np.asarray(Svar_s.loc[idxs[j]:idxe[j],['C_X']],dtype=np.float64)[:,0],np.asarray(Svar_s.loc[idxs[j]:idxe[j],['C_Y']],dtype=np.float64)[:,0])     
            Dir_w_C=(np.degrees(Dir_w_C)+180) %360
            dir_w_C=(np.degrees(dir_w_C)+180) %360            
            C_Dir.append(Dir_w_C)
            C_dir.extend(dir_w_C)            
            C_U.append(U_C)
            C_uvw.extend(uvw_C)            
            
    A_uvw = np.array(A_uvw)
    B_uvw = np.array(B_uvw)
    C_uvw = np.array(C_uvw)
    A_dir = np.array(A_dir) 
    B_dir = np.array(B_dir) 
    C_dir = np.array(C_dir) 
    time_ss = np.array(time_ss)  


    # save instaneous wind data into a dataframe and then save as a pkl file
    Svar_1h_ins  = pd.DataFrame(columns=['Time', 'A_dir','A_u', 'A_v','A_w','B_dir', 'B_u', 'B_v','B_w','C_dir','C_u', 'C_v','C_w'])    
    Svar_1h_ins['Time']  =    time_ss[:,0]
    Svar_1h_ins.loc[Svar_1h_ins.index,['A_dir']] = A_dir
    Svar_1h_ins.loc[Svar_1h_ins.index,['A_u','A_v','A_w']] = A_uvw
    Svar_1h_ins.loc[Svar_1h_ins.index,['B_dir']] = B_dir    
    Svar_1h_ins.loc[Svar_1h_ins.index,['B_u','B_v','B_w']] = B_uvw
    Svar_1h_ins.loc[Svar_1h_ins.index,['C_dir']] = C_dir    
    Svar_1h_ins.loc[Svar_1h_ins.index,['C_u','C_v','C_w']] = C_uvw
    
    Svar_1h_ins.to_pickle('D:/DATA/svar/' +'Svar_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Svar_1h' +'_ins.pkl')
    
    # save mean wind data into a dataframe and then save as a pkl file
    Svar_1h_mean  = pd.DataFrame(columns=['Time', 'A_Dir','A_U','B_Dir', 'B_U', 'C_Dir','C_U'])    
    Svar_1h_mean['Time']  =    time_ref
    Svar_1h_mean.loc[Svar_1h_mean.index,['A_Dir']] = A_Dir
    Svar_1h_mean.loc[Svar_1h_mean.index,['A_U']] = A_U
    Svar_1h_mean.loc[Svar_1h_mean.index,['B_Dir']] = B_Dir    
    Svar_1h_mean.loc[Svar_1h_mean.index,['B_U']] = B_U
    Svar_1h_mean.loc[Svar_1h_mean.index,['C_Dir']] = C_Dir    
    Svar_1h_mean.loc[Svar_1h_mean.index,['C_U']] = C_U
    
    Svar_1h_mean.to_pickle('D:/DATA/svar/' +'Svar_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Svar_1h' +'_mean.pkl')
   



# =============================================================================
#     Svar_1h  = {'A_U':  A_U,   'B_U':  B_U,   'C_U':  C_U,   'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir,\
#                 'A_uvw':A_uvw, 'B_uvw':B_uvw, 'C_uvw':C_uvw, 'A_dir':A_dir, 'B_dir':B_dir, 'C_dir':C_dir,\
#                 'time':time_ss, 'Time':time_ref    }  
#     
#     with open('D:/DATA/svar/' +'Svar_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Svar_1h' +'.pkl', 'wb') as f:    
#         pickle.dump(Svar_1h, f, pickle.HIGHEST_PROTOCOL) 
# 
# =============================================================================
          

#%% Synn
    # here we use 1 hour interval
    num_1h           = int(np.floor((mdates.date2num(date_end)-mdates.date2num(date_start))/delta_1h))

    # create the startng and ending time array  
    time_s = np.array([date_start + timedelta(hours=i) for i in range(0,num_1h)])
    time_e = np.array([date_start + timedelta(hours=i) for i in range(1,num_1h+1)])
      
    # find the cloest index of the starting and ending time of each hour   
    idxs=Synn_s.loc[:,'Time'].searchsorted (mdates.date2num(time_s),side='right') 
    idxe=Synn_s.loc[:,'Time'].searchsorted (mdates.date2num(time_e),side='left') 

    time_ref   = []    
    A_dir      = []
    A_Dir      = []    
    A_U        = []
    A_uvw      = []    
    B_dir      = []
    B_Dir      = []        
    B_U        = []
    B_uvw      = []
    C_dir      = []
    C_Dir      = []        
    C_U        = []
    C_uvw      = []
    time_ss    = []     
    
    for j in range(0,num_1h):        
        # if idxs=idxe, then no data avaliable in this hour
        if idxs[j] !=  idxe[j]:
            date_ref  =  mdates.num2date((mdates.date2num(time_s[j]) + mdates.date2num(time_e[j]))/2)
    
            [Dir_w_A,U_A,uvw_A] = windXYZ2uvw(np.asarray(Synn_s.loc[idxs[j]:idxe[j],['A_X','A_Y','A_Z']],dtype=np.float64)) 
            dir_w_A             = wind_dir(np.asarray(Synn_s.loc[idxs[j]:idxe[j],['A_X']],dtype=np.float64)[:,0],np.asarray(Synn_s.loc[idxs[j]:idxe[j],['A_Y']],dtype=np.float64)[:,0])
            Dir_w_A=(np.degrees(Dir_w_A)+180) %360
            dir_w_A=(np.degrees(dir_w_A)+180) %360
            A_Dir.append(Dir_w_A)
            A_dir.extend(dir_w_A)
            A_U.append(U_A)
            A_uvw.extend(uvw_A)
            time_ref.append(date_ref)
            time_ss.extend(np.asarray(Synn_s.loc[idxs[j]:idxe[j],['Time']],dtype=np.float64))
            
            [Dir_w_B,U_B,uvw_B] = windXYZ2uvw(np.asarray(Synn_s.loc[idxs[j]:idxe[j],['B_X','B_Y','B_Z']],dtype=np.float64))
            dir_w_B             = wind_dir(np.asarray(Synn_s.loc[idxs[j]:idxe[j],['B_X']],dtype=np.float64)[:,0],np.asarray(Synn_s.loc[idxs[j]:idxe[j],['B_Y']],dtype=np.float64)[:,0])            
            Dir_w_B=(np.degrees(Dir_w_B)+180) %360
            dir_w_B=(np.degrees(dir_w_B)+180) %360            
            B_Dir.append(Dir_w_B)
            B_dir.extend(dir_w_B)            
            B_U.append(U_B)
            B_uvw.extend(uvw_B)            

            [Dir_w_C,U_C,uvw_C] = windXYZ2uvw(np.asarray(Synn_s.loc[idxs[j]:idxe[j],['C_X','C_Y','C_Z']],dtype=np.float64))
            dir_w_C             = wind_dir(np.asarray(Synn_s.loc[idxs[j]:idxe[j],['C_X']],dtype=np.float64)[:,0],np.asarray(Synn_s.loc[idxs[j]:idxe[j],['C_Y']],dtype=np.float64)[:,0])     
            Dir_w_C=(np.degrees(Dir_w_C)+180) %360
            dir_w_C=(np.degrees(dir_w_C)+180) %360            
            C_Dir.append(Dir_w_C)
            C_dir.extend(dir_w_C)            
            C_U.append(U_C)
            C_uvw.extend(uvw_C)            
            
    A_uvw = np.array(A_uvw)
    B_uvw = np.array(B_uvw)
    C_uvw = np.array(C_uvw)
    A_dir = np.array(A_dir) 
    B_dir = np.array(B_dir) 
    C_dir = np.array(C_dir) 
    time_ss = np.array(time_ss)  
    
    
    # save instaneous wind data into a dataframe and then save as a pkl file
    Synn_1h_ins  = pd.DataFrame(columns=['Time', 'A_dir','A_u', 'A_v','A_w','B_dir', 'B_u', 'B_v','B_w','C_dir','C_u', 'C_v','C_w'])    
    Synn_1h_ins['Time']  =    time_ss[:,0]
    Synn_1h_ins.loc[Synn_1h_ins.index,['A_dir']] = A_dir
    Synn_1h_ins.loc[Synn_1h_ins.index,['A_u','A_v','A_w']] = A_uvw
    Synn_1h_ins.loc[Synn_1h_ins.index,['B_dir']] = B_dir    
    Synn_1h_ins.loc[Synn_1h_ins.index,['B_u','B_v','B_w']] = B_uvw
    Synn_1h_ins.loc[Synn_1h_ins.index,['C_dir']] = C_dir    
    Synn_1h_ins.loc[Synn_1h_ins.index,['C_u','C_v','C_w']] = C_uvw
    
    Synn_1h_ins.to_pickle('D:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Synn_1h' +'_ins.pkl')
    
    # save mean wind data into a dataframe and then save as a pkl file
    Synn_1h_mean  = pd.DataFrame(columns=['Time', 'A_Dir','A_U','B_Dir', 'B_U', 'C_Dir','C_U'])    
    Synn_1h_mean['Time']  =    time_ref
    Synn_1h_mean.loc[Synn_1h_mean.index,['A_Dir']] = A_Dir
    Synn_1h_mean.loc[Synn_1h_mean.index,['A_U']] = A_U
    Synn_1h_mean.loc[Synn_1h_mean.index,['B_Dir']] = B_Dir    
    Synn_1h_mean.loc[Synn_1h_mean.index,['B_U']] = B_U
    Synn_1h_mean.loc[Synn_1h_mean.index,['C_Dir']] = C_Dir    
    Synn_1h_mean.loc[Synn_1h_mean.index,['C_U']] = C_U
    
    Synn_1h_mean.to_pickle('D:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Synn_1h' +'_mean.pkl')
       

# =============================================================================
#     Synn_1h  = {'A_U':  A_U,   'B_U':  B_U,   'C_U':  C_U,   'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir,\
#                 'A_uvw':A_uvw, 'B_uvw':B_uvw, 'C_uvw':C_uvw, 'A_dir':A_dir, 'B_dir':B_dir, 'C_dir':C_dir,\
#                 'time':time_ss, 'Time':time_ref    }  
#     
#     with open('D:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Synn_1h' +'.pkl', 'wb') as f:    
#         pickle.dump(Synn_1h, f, pickle.HIGHEST_PROTOCOL)  
# =============================================================================
        
 #%%   
# =============================================================================
#     with open('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_1h' +'.pkl', 'rb') as f:    
#         Osp1_1h=pickle.load( f)
#     with open('D:/DATA/osp2/' +'Osp2_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp2_1h' +'.pkl', 'rb') as f:    
#         Osp2_1h=pickle.load( f)
#     with open('D:/DATA/svar/' +'Svar_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Svar_1h' +'.pkl', 'rb') as f:    
#         Svar_1h=pickle.load( f)        
#     with open('D:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Synn_1h' +'.pkl', 'rb') as f:    
#         Synn_1h=pickle.load( f)
#     
#     Osp1_1h['A_Dir']= np.asarray(Osp1_1h['A_Dir'])
#     
#     if  round(max(Osp1_1h['A_dir']))-round(min(Osp1_1h['A_dir']))==360:
#         Osp1_1h['A_dir'][Osp1_1h['A_dir']<20] =  Osp1_1h['A_dir'][Osp1_1h['A_dir']<20]+360 
#         Osp1_1h['A_Dir'][Osp1_1h['A_Dir']<20] =  Osp1_1h['A_Dir'][Osp1_1h['A_Dir']<20]+360 
# =============================================================================
    
# =============================================================================
#     dt          = 0.1
#     Nwin  = round(60*60./dt);     
#     # plot the wind direction, mean wind speed and along wind turbulence intensity
#     plt.close("all")       
#     fig     = plt.figure(figsize=(20, 12))
#     ax1     = plt.subplot(311)
#     # format the ticks
#     locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
#     formatter = mdates.ConciseDateFormatter(locator)
#     ax1.xaxis.set_major_locator(locator)
#     ax1.xaxis.set_major_formatter(formatter)
#     ax1.plot(Osp1_1h['time'], Osp1_1h['A_dir'], 'k-',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
#     #movingdir_Osp1   =  np.asarray(pd.DataFrame(Osp1_1h['A_dir']).rolling(Nwin,min_periods=1).mean())
#     #ax1.plot(Osp1_1h['time'], movingdir_Osp1, 'r-',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
#     ax1.plot(Osp1_1h['Time'], Osp1_1h['A_Dir'], 'r-',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=8)   
#     datemin = np.datetime64(Osp1_1h['time'][0][0].replace(tzinfo=None), 'h')
#     datemax = np.datetime64(Osp1_1h['time'][-1][0].replace(tzinfo=None), 'm') + np.timedelta64(1, 'h')
#     ax1.set_xlim(datemin, datemax)
#     plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
#     fig.suptitle(str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_time_history', fontsize=25)
#     plt.rc('xtick', direction='in', color='k')
#     plt.rc('ytick', direction='in', color='k')
#     plt.legend(loc='best',ncol=1,fontsize=16)
#     g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
#     g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
#     ax1.tick_params(axis='both', labelsize=16)
#     plt.minorticks_on()
#     plt.show()
#     
#     
#     ax2     = plt.subplot(312)
#     # format the ticks
#     locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
#     formatter = mdates.ConciseDateFormatter(locator)
#     ax2.xaxis.set_major_locator(locator)
#     ax2.xaxis.set_major_formatter(formatter)
#     ax2.plot(Osp1_1h['time'], Osp1_1h['A_uvw'][:,0], 'b-',label='$\overline{U} + u $ $(Osp1\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
#     #movingU_Osp1   =  np.asarray(pd.DataFrame(Osp1_1h['A_uvw'][:,0]).rolling(Nwin,min_periods=1).mean())
#     #ax2.plot(Osp1_1h['time'], movingU_Osp1, 'r-',label='$\overline{U} $ $(Osp1\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
#     ax2.plot(Osp1_1h['Time'], Osp1_1h['A_U'], 'r-',label='$\overline{U} $ $(Osp1\_A)$',markeredgecolor='k',markersize=8)    
#     datemin = np.datetime64(Osp1_s.loc[0,'Time'].replace(tzinfo=None), 'h')
#     datemax = np.datetime64(Osp1_s.loc[Osp1_s.index[-1],'Time'].replace(tzinfo=None), 'm') + np.timedelta64(1, 'h')
#     ax2.set_xlim(datemin, datemax)
#     plt.ylabel(r'$\overline{U} + u$ (m s$^{-1})$', fontsize=20)
#     ax2.set_title('', fontsize=25)
#     plt.rc('xtick', direction='in', color='k')
#     plt.rc('ytick', direction='in', color='k')
#     plt.legend(loc='best',ncol=1,fontsize=16)
#     g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
#     g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
#     ax2.tick_params(axis='both', labelsize=16)
#     plt.minorticks_on()
#     plt.show()
#     
#     
#     ax3     = plt.subplot(313)
#     # format the ticks
#     locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
#     formatter = mdates.ConciseDateFormatter(locator)
#     ax3.xaxis.set_major_locator(locator)
#     ax3.xaxis.set_major_formatter(formatter)
#     ax3.plot(Osp1_1h['Time'], Osp1_1h['A_U'], 'ro-',label='$\overline{U} $ $(Osp1\_A)$',markeredgecolor='r',markersize=8,alpha=0.5)    
#     ax3.plot(Osp2_1h['Time'], Osp2_1h['A_U'], 'ko-',label='$\overline{U} $ $(Osp2\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)    
#     ax3.plot(Svar_1h['Time'], Svar_1h['A_U'], 'go-',label='$\overline{U} $ $(Svar\_A)$',markeredgecolor='g',markersize=8,alpha=0.5)    
#     ax3.plot(Synn_1h['Time'], Synn_1h['A_U'], 'bo-',label='$\overline{U} $ $(Synn\_A)$',markeredgecolor='b',markersize=8,alpha=0.5)    
# 
#     datemin = np.datetime64(Osp1_s.loc[0,'Time'].replace(tzinfo=None), 'h')
#     datemax = np.datetime64(Osp1_s.loc[Osp1_s.index[-1],'Time'].replace(tzinfo=None), 'm') + np.timedelta64(1, 'h')
#     ax3.set_xlim(datemin, datemax)
#     plt.ylabel(r'$\overline{U}$ (m s$^{-1})$', fontsize=20)
#     ax3.set_title('', fontsize=25)
#     plt.rc('xtick', direction='in', color='k')
#     plt.rc('ytick', direction='in', color='k')
#     plt.legend(loc='best',ncol=1,fontsize=16)
#     g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
#     g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
#     ax3.tick_params(axis='both', labelsize=16)
#     plt.minorticks_on()
#     plt.show()    
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     save_tite = str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_time_history.png'
#     fig.savefig(save_tite)     
# =============================================================================
