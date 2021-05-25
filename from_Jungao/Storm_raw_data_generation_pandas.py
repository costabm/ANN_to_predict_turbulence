# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 08:44:42 2020
create raw data files for each storm event
based on the new pandas dataframe data

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
import gc

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
delta_ms   = timedelta(milliseconds=100)

for i in range(20,cases['Time_storm'].size):
    
    #i=46
    
    time = datetime.strptime(cases['Time_storm'][i], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
    
    time_s  =  mdates.num2date( mdates.date2num(time) - delta_1d*2).replace(tzinfo=None)
    time_e  =  mdates.num2date( mdates.date2num(time) + delta_1d*2).replace(tzinfo=None)

    year_s = str(time_s.year)
    if time_s.month<10:
        mon_s = '0' + str(time_s.month)
    else:
        mon_s = str(time_s.month)
        
    year_e = str(time_e.year)
    if time_e.month<10:
        mon_e = '0' + str(time_e.month)
    else:
        mon_e = str(time_e.month)        
        
    #Osp1    
    if os.path.isfile('D:/DATA/osp1/' +'Osp1_'+ year_s +'_' + mon_s+ '.pkl')\
        and os.path.isfile('D:/DATA/osp1/' +'Osp1_'+ year_e +'_' + mon_e+ '.pkl'): 
        # if it is same month, just load data once, otherwise, need to append two months' data         
        # load raw sensor data
        if mon_s==mon_e:            
            data = pd.read_pickle('D:/DATA/osp1/' +'Osp1_'+ year_s +'_' + mon_s+ '.pkl')                       
        else:
            data = pd.read_pickle('D:/DATA/osp1/' +'Osp1_'+ year_s +'_' + mon_s+ '.pkl')
            temp = pd.read_pickle('D:/DATA/osp1/' +'Osp1_'+ year_e +'_' + mon_e+ '.pkl')
            data = data.append(temp,ignore_index=True)          
  
        # find the index in the instant data    
        time_S  =   mdates.date2num(time) - delta_1d*2
        time_E  =   mdates.date2num(time) + delta_1d*2

        date_start = mdates.date2num( datetime(int(year_s),  int(mon_s),  1,0,0)           )
        if mon_e=='12':   
            date_end   = mdates.date2num( datetime(int(year_e)+1,1,1,0,0) - delta_ms  )    
        else:      
            date_end   = mdates.date2num( datetime(int(year_e),int(mon_e)+1,1,0,0) - delta_ms  )

        time_new = pd.date_range(mdates.num2date(date_start),mdates.num2date(date_end),freq = '100L')
        time_new = pd.Series(mdates.date2num(time_new))

        if len( time_new[time_new==time_S].index.values.astype(int))>0:
            idx_S = time_new[time_new==time_S].index.values.astype(int)[0]
            idx_E = time_new[time_new==time_E].index.values.astype(int)[0]
           
            data_s      = data.loc[idx_S:idx_E,:].reset_index(drop=True)           
            data_s.Time = time_new.loc[idx_S:idx_E].reset_index(drop=True)
            data_s.loc[abs(data_s['A_X'])>100,'A_X'] = np.nan           
            data_s.loc[abs(data_s['A_Y'])>100,'A_Y'] = np.nan
            data_s.loc[abs(data_s['A_Z'])>100,'A_Z'] = np.nan
            data_s.loc[abs(data_s['B_X'])>100,'B_X'] = np.nan
            data_s.loc[abs(data_s['B_Y'])>100,'B_Y'] = np.nan
            data_s.loc[abs(data_s['B_Z'])>100,'B_Z'] = np.nan
            data_s.loc[abs(data_s['C_X'])>100,'C_X'] = np.nan
            data_s.loc[abs(data_s['C_Y'])>100,'C_Y'] = np.nan
            data_s.loc[abs(data_s['C_Z'])>100,'C_Z'] = np.nan             
            data_s.to_pickle('D:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
            print ('storm for osp1 ' + str(time) + 'is finished')
            
            del data,data_s,time_new
            gc.collect() 
    #Osp2    
    if os.path.isfile('D:/DATA/osp2/' +'Osp2_'+ year_s +'_' + mon_s+ '.pkl')\
        and os.path.isfile('D:/DATA/osp2/' +'Osp2_'+ year_e +'_' + mon_e+ '.pkl'): 
        # if it is same month, just load data once, otherwise, need to append two months' data         
        # load raw sensor data
        if mon_s==mon_e:            
            data = pd.read_pickle('D:/DATA/osp2/' +'Osp2_'+ year_s +'_' + mon_s+ '.pkl')                       
        else:
            data = pd.read_pickle('D:/DATA/osp2/' +'Osp2_'+ year_s +'_' + mon_s+ '.pkl')
            temp = pd.read_pickle('D:/DATA/osp2/' +'Osp2_'+ year_e +'_' + mon_e+ '.pkl')
            data = data.append(temp,ignore_index=True)          
  
        # find the index in the instant data    
        time_S  =   mdates.date2num(time) - delta_1d*2
        time_E  =   mdates.date2num(time) + delta_1d*2
        
        date_start = mdates.date2num( datetime(int(year_s),  int(mon_s),  1,0,0)           )
        if mon_e=='12':   
            date_end   = mdates.date2num( datetime(int(year_e)+1,1,1,0,0) - delta_ms  )    
        else:      
            date_end   = mdates.date2num( datetime(int(year_e),int(mon_e)+1,1,0,0) - delta_ms  )

        time_new = pd.date_range(mdates.num2date(date_start),mdates.num2date(date_end),freq = '100L')
        time_new = pd.Series(mdates.date2num(time_new))

        if len( time_new[time_new==time_S].index.values.astype(int))>0:
            idx_S = time_new[time_new==time_S].index.values.astype(int)[0]
            idx_E = time_new[time_new==time_E].index.values.astype(int)[0]
           
            data_s      = data.loc[idx_S:idx_E,:].reset_index(drop=True)           
            data_s.Time = time_new.loc[idx_S:idx_E].reset_index(drop=True)
            data_s.loc[abs(data_s['A_X'])>100,'A_X'] = np.nan           
            data_s.loc[abs(data_s['A_Y'])>100,'A_Y'] = np.nan
            data_s.loc[abs(data_s['A_Z'])>100,'A_Z'] = np.nan
            data_s.loc[abs(data_s['B_X'])>100,'B_X'] = np.nan
            data_s.loc[abs(data_s['B_Y'])>100,'B_Y'] = np.nan
            data_s.loc[abs(data_s['B_Z'])>100,'B_Z'] = np.nan
            data_s.loc[abs(data_s['C_X'])>100,'C_X'] = np.nan
            data_s.loc[abs(data_s['C_Y'])>100,'C_Y'] = np.nan
            data_s.loc[abs(data_s['C_Z'])>100,'C_Z'] = np.nan             
            data_s.to_pickle('D:/DATA/osp2/' +'Osp2_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
            print ('storm for osp2 ' + str(time) + 'is finished')
            del data,data_s,time_new
            gc.collect()    
            
    #Svar    
    if os.path.isfile('D:/DATA/svar/' +'Svar_'+ year_s +'_' + mon_s+ '.pkl')\
        and os.path.isfile('D:/DATA/svar/' +'Svar_'+ year_e +'_' + mon_e+ '.pkl'): 
        # if it is same month, just load data once, otherwise, need to append two months' data         
        # load raw sensor data
        if mon_s==mon_e:            
            data = pd.read_pickle('D:/DATA/svar/' +'Svar_'+ year_s +'_' + mon_s+ '.pkl')                       
        else:
            data = pd.read_pickle('D:/DATA/svar/' +'Svar_'+ year_s +'_' + mon_s+ '.pkl')
            temp = pd.read_pickle('D:/DATA/svar/' +'Svar_'+ year_e +'_' + mon_e+ '.pkl')
            data = data.append(temp,ignore_index=True)          
  
        # find the index in the instant data    
        time_S  =   mdates.date2num(time) - delta_1d*2
        time_E  =   mdates.date2num(time) + delta_1d*2

        date_start = mdates.date2num( datetime(int(year_s),  int(mon_s),  1,0,0)           )
        if mon_e=='12':   
            date_end   = mdates.date2num( datetime(int(year_e)+1,1,1,0,0) - delta_ms  )    
        else:      
            date_end   = mdates.date2num( datetime(int(year_e),int(mon_e)+1,1,0,0) - delta_ms  )

        time_new = pd.date_range(mdates.num2date(date_start),mdates.num2date(date_end),freq = '100L')
        time_new = pd.Series(mdates.date2num(time_new))

        if len( time_new[time_new==time_S].index.values.astype(int))>0:
            idx_S = time_new[time_new==time_S].index.values.astype(int)[0]
            idx_E = time_new[time_new==time_E].index.values.astype(int)[0]
           
            data_s      = data.loc[idx_S:idx_E,:].reset_index(drop=True)           
            data_s.Time = time_new.loc[idx_S:idx_E].reset_index(drop=True)
            data_s.loc[abs(data_s['A_X'])>100,'A_X'] = np.nan           
            data_s.loc[abs(data_s['A_Y'])>100,'A_Y'] = np.nan
            data_s.loc[abs(data_s['A_Z'])>100,'A_Z'] = np.nan
            data_s.loc[abs(data_s['B_X'])>100,'B_X'] = np.nan
            data_s.loc[abs(data_s['B_Y'])>100,'B_Y'] = np.nan
            data_s.loc[abs(data_s['B_Z'])>100,'B_Z'] = np.nan
            data_s.loc[abs(data_s['C_X'])>100,'C_X'] = np.nan
            data_s.loc[abs(data_s['C_Y'])>100,'C_Y'] = np.nan
            data_s.loc[abs(data_s['C_Z'])>100,'C_Z'] = np.nan             
            data_s.to_pickle('D:/DATA/svar/' +'Svar_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
            print ('storm for svar ' + str(time) + 'is finished')
            del data,data_s,time_new
            gc.collect() 

    #Synn    
    if os.path.isfile('D:/DATA/synn/' +'Synn_'+ year_s +'_' + mon_s+ '.pkl')\
        and os.path.isfile('D:/DATA/synn/' +'Synn_'+ year_e +'_' + mon_e+ '.pkl'): 
        # if it is same month, just load data once, otherwise, need to append two months' data         
        # load raw sensor data
        if mon_s==mon_e:            
            data = pd.read_pickle('D:/DATA/synn/' +'Synn_'+ year_s +'_' + mon_s+ '.pkl')                       
        else:
            data = pd.read_pickle('D:/DATA/synn/' +'Synn_'+ year_s +'_' + mon_s+ '.pkl')
            temp = pd.read_pickle('D:/DATA/synn/' +'Synn_'+ year_e +'_' + mon_e+ '.pkl')
            data = data.append(temp,ignore_index=True)          
  
        # find the index in the instant data    
        time_S  =   mdates.date2num(time) - delta_1d*2
        time_E  =   mdates.date2num(time) + delta_1d*2

        date_start = mdates.date2num( datetime(int(year_s),  int(mon_s),  1,0,0)           )
        if mon_e=='12':   
            date_end   = mdates.date2num( datetime(int(year_e)+1,1,1,0,0) - delta_ms  )    
        else:      
            date_end   = mdates.date2num( datetime(int(year_e),int(mon_e)+1,1,0,0) - delta_ms  )

        time_new = pd.date_range(mdates.num2date(date_start),mdates.num2date(date_end),freq = '100L')
        time_new = pd.Series(mdates.date2num(time_new))

        if len( time_new[time_new==time_S].index.values.astype(int))>0:
            idx_S = time_new[time_new==time_S].index.values.astype(int)[0]
            idx_E = time_new[time_new==time_E].index.values.astype(int)[0]
           
            data_s      = data.loc[idx_S:idx_E,:].reset_index(drop=True)           
            data_s.Time = time_new.loc[idx_S:idx_E].reset_index(drop=True)
            data_s.loc[abs(data_s['A_X'])>100,'A_X'] = np.nan           
            data_s.loc[abs(data_s['A_Y'])>100,'A_Y'] = np.nan
            data_s.loc[abs(data_s['A_Z'])>100,'A_Z'] = np.nan
            data_s.loc[abs(data_s['B_X'])>100,'B_X'] = np.nan
            data_s.loc[abs(data_s['B_Y'])>100,'B_Y'] = np.nan
            data_s.loc[abs(data_s['B_Z'])>100,'B_Z'] = np.nan
            data_s.loc[abs(data_s['C_X'])>100,'C_X'] = np.nan
            data_s.loc[abs(data_s['C_Y'])>100,'C_Y'] = np.nan
            data_s.loc[abs(data_s['C_Z'])>100,'C_Z'] = np.nan            
            data_s.to_pickle('D:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl')
            print ('storm for synn ' + str(time) + 'is finished')
            del data,data_s,time_new
            gc.collect()     
        
