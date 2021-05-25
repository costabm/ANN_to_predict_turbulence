# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:19:13 2020

Create the U, u, v, w from raw sonic data


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

import matplotlib.dates as mdates
import json
import transformations      as TF

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

def wind_dir(X,Y):
    dir_W  =  np.arctan2(X,Y)
    dir_W[dir_W<0] = dir_W[dir_W<0]+2*np.pi   
    return (dir_W)

regex_num  = r"[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?" 

# gound coordinate vector
G_x = [1,0,0]
G_y = [0,1,0]
G_z = [0,0,1]

cases = pd.read_excel('files_osp1.xls', sheet_name='files_osp1')
# get the ending time tag 
time_e = cases['Last timestep'][cases.index[-1]] 
temp = np.asarray(re.findall(regex_num,time_e), dtype='float64')    
year_e =     int(temp[0])
mon_e  =     int(abs(temp[1]))
day_e  =     int(abs(temp[2]))
hour_e =     int(temp[3])
minu_e =     int(temp[4])
sec_e  =     int(np.floor(temp[5]))
msec_e =     int(np.round((temp[5]-np.floor(temp[5]))*1000))
time_e = mdates.date2num( datetime(year_e,mon_e,day_e,hour_e,minu_e,sec_e,msec_e))   

# get all the time tags
time_case   = np.zeros((cases['Last timestep'].size+1,1)) 
for i,time in enumerate(cases['First timestep']):
    temp = np.asarray(re.findall(regex_num,time), dtype='float64')    
    year =     int(temp[0])
    mon  =     int(abs(temp[1]))
    day  =     int(abs(temp[2]))
    hour =     int(temp[3])
    minu =     int(temp[4])
    sec  =     int(np.floor(temp[5]))
    msec =     int(np.round((temp[5]-np.floor(temp[5]))*1000))

    time_case[i,0] = mdates.date2num( datetime(year,mon,day,hour,minu,sec,msec) )    
time_case[i+1,0] = time_e

# delta_mon is one month time
delta_ms         = timedelta(milliseconds=100)
delta_10min      = mdates.date2num(datetime(2,1,1,1,10))-mdates.date2num(datetime(2,1,1,1,0))
delta_1h         = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
delta_mon        = mdates.date2num(datetime(2,2,1,1,0))-mdates.date2num(datetime(2,1,1,1,0))

date_s_ref = datetime(2015,12,4,0,0,0)
date_e_ref = mdates.num2date(time_case[-1,0]+delta_mon).replace(tzinfo=None)

num_10min        = int(np.floor((mdates.date2num(date_e_ref)-mdates.date2num(date_s_ref))/delta_10min))
num_1h           = int(np.floor((mdates.date2num(date_e_ref)-mdates.date2num(date_s_ref))/delta_1h))
num_mon          = int(np.floor((mdates.date2num(date_e_ref)-mdates.date2num(date_s_ref))/delta_mon))

monthloop  = month_year_iter(date_s_ref.month,date_s_ref.year,date_e_ref.month,date_e_ref.year)


for i in range(0,monthloop.shape[0]-1):
    date_start = mdates.date2num( datetime(int(monthloop[i,0]),  int(monthloop[i,1])+1,  1,0,0)           )
    date_end   = mdates.date2num( datetime(int(monthloop[i+1,0]),int(monthloop[i+1,1])+1,1,0,0) - delta_ms)

    if i==0:
        date_start = mdates.date2num( date_s_ref)
    if i==monthloop.shape[0]-1:   
        date_end   = mdates.num2date(time_case[-1,0]).replace(tzinfo=None)
    
    #create the string of 'date_search '
    t_s = mdates.num2date(date_start)
    year = str(t_s.year)
    mon = str(t_s.month)   

    # load the sensor data in global XYZ coordinates        
    A  =  np.load('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_A.npz')['arr_0']   
    B  =  np.load('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_B.npz')['arr_0']    
    C  =  np.load('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_C.npz')['arr_0']    
    time  =  np.load('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_time.npz')['arr_0']  
    time   = pd.Series(time[:,0])

    # replace the values bigger than 100 to nan   
    flag_A = np.unique(np.where(abs(A)>100)[1])
    A[:,flag_A] = np.nan
    flag_B = np.unique(np.where(abs(B)>100)[1])
    B[:,flag_B] = np.nan
    flag_C = np.unique(np.where(abs(C)>100)[1])
    C[:,flag_C] = np.nan
    
    time_ref = []
    A_dir      = []
    A_U        = []
    A_uvw      = []    
    B_dir      = []
    B_U        = []
    B_uvw      = []
    C_dir      = []
    C_U        = []
    C_uvw      = []    
    num_1h     = int(np.floor((date_end-date_start)/delta_1h))

    # create the startng and ending time array  
    time_s = np.arange(date_start,date_end-delta_1h,delta_1h)
    time_e = np.arange(date_start+delta_1h,date_end,delta_1h)
      
    # find the cloest index of the starting and ending time of each hour   
    idxs=time.searchsorted (time_s,side='right') 
    idxe=time.searchsorted (time_e,side='left') 


    for j in range(0,num_1h):        
        # if idxs=idxe, then no data avaliable in this hour
        if idxs[j] !=  idxe[j]:
            date_ref  =  date_start + delta_1h*j + delta_1h/2            
    
    
            [dir_w_A,U_A,uvw_A] = windXYZ2uvw(A[idxs[j]:idxe[j],:])   
            A_dir.append(dir_w_A)
            A_U.append(U_A)
            A_uvw.extend(uvw_A)
            time_ref.append(date_ref)
            
            [dir_w_B,U_B,uvw_B] = windXYZ2uvw(B[idxs[j]:idxe[j],:])   
            B_dir.append(dir_w_B)
            B_U.append(U_B)
            B_uvw.extend(uvw_B)            

            [dir_w_C,U_C,uvw_C] = windXYZ2uvw(C[idxs[j]:idxe[j],:])   
            C_dir.append(dir_w_C)
            C_U.append(U_C)
            C_uvw.extend(uvw_C)            
            
    A_uvw = np.array(A_uvw)
    B_uvw = np.array(B_uvw)
    C_uvw = np.array(C_uvw)   
    
    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_A_time_1h.npz', time_ref)
    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_A_dir_1h.npz', A_dir)
    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_A_U_1h.npz', A_U)
    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_A_uvw_1h.npz', A_uvw)
    
    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_B_dir_1h.npz', B_dir)
    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_B_U_1h.npz', B_U)
    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_B_uvw_1h.npz', B_uvw)

    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_C_dir_1h.npz', C_dir)
    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_C_U_1h.npz', C_U)
    np.savez_compressed('D:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_C_uvw_1h.npz', C_uvw)

    