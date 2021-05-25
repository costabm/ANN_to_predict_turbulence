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
def windXYZ2uvw(data,time,date_s,date_e):    
    # obtain the index of the starting and ending time_tag
    temp_s = time - date_s
    temp_e = time - date_e    
    ids    = np.where(abs(temp_s)==min(abs(temp_s[temp_s>=0])))[0][0]
    ide    = np.where(abs(temp_e)==min(abs(temp_e[temp_e<=0])))[0][0]
    
    # calculate the non-mean for each component of the wind components in global XYZ within starting and ending time
    data_temp = np.nanmean(data[ids:ide,:],0)   
    if ~np.isnan(data_temp.any()):
        [dir_w,tt_w] = windXY2dir(data_temp[0],data_temp[1]) 
        U            = np.matmul(tt_w.T , data_temp)[0]
        uvw          = np.matmul(tt_w.T , data[ids:ide,:].T).T
        time_ref     = (date_s+date_e)/2
        return(time_ref,dir_w,U,uvw)


regex_num  = r"[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?" 

# gound coordinate vector
G_x = [1,0,0]
G_y = [0,1,0]
G_z = [0,0,1]

cases = pd.read_excel('files_svar.xls', sheet_name='files_svar')
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

date_s_ref = datetime(2015,3,18,0,0,0)
date_e_ref = mdates.num2date(time_case[-1,0]+delta_mon).replace(tzinfo=None)

num_10min        = int(np.floor((mdates.date2num(date_e_ref)-mdates.date2num(date_s_ref))/delta_10min))
num_1h           = int(np.floor((mdates.date2num(date_e_ref)-mdates.date2num(date_s_ref))/delta_1h))
num_mon          = int(np.floor((mdates.date2num(date_e_ref)-mdates.date2num(date_s_ref))/delta_mon))

monthloop  = month_year_iter(date_s_ref.month,date_s_ref.year,date_e_ref.month,date_e_ref.year)


for i in range(10,monthloop.shape[0]-1):
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
    A  =  np.load('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_A.npz')['arr_0']   
    B  =  np.load('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_B.npz')['arr_0']    
    C  =  np.load('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_C.npz')['arr_0']    
    time  =  np.load('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_time.npz')['arr_0']  

    # replace the values bigger than 100 to nan   
    flag_A = np.unique(np.where(abs(A)>100)[1])
    A[:,flag_A] = np.nan
    flag_B = np.unique(np.where(abs(B)>100)[1])
    B[:,flag_B] = np.nan
    flag_C = np.unique(np.where(abs(C)>100)[1])
    C[:,flag_C] = np.nan
    
    A_time_ref = []
    A_dir      = []
    A_U        = []
    A_uvw      = []    
    B_time_ref = []
    B_dir      = []
    B_U        = []
    B_uvw      = []
    C_time_ref = []
    C_dir      = []
    C_U        = []
    C_uvw      = []    
    num_1h     = int(np.floor((date_end-date_start)/delta_1h))
    for j in range(0,num_1h):
        date_ref  =  date_start + delta_1h*j + delta_1h/2
        
        date_s = date_start + delta_1h*j
        date_e = date_start + delta_1h*j + delta_1h        

        [time_ref_A,dir_w_A,U_A,uvw_A] = windXYZ2uvw(A,time,date_s,date_e)        
        A_time_ref = np.append(A_time_ref,time_ref_A)
        A_dir      = np.append(A_dir,dir_w_A)
        A_U        = np.append(A_U,U_A)
        A_uvw      = np.append(A_uvw,uvw_A)
        
        [time_ref_B,dir_w_B,U_B,uvw_B] = windXYZ2uvw(B,time,date_s,date_e)        
        B_time_ref = np.append(B_time_ref,time_ref_B)
        B_dir      = np.append(B_dir,dir_w_B)
        B_U        = np.append(B_U,U_B)
        B_uvw      = np.append(B_uvw,uvw_B)

        [time_ref_C,dir_w_C,U_C,uvw_C] = windXYZ2uvw(C,time,date_s,date_e)        
        C_time_ref = np.append(C_time_ref,time_ref_C)
        C_dir      = np.append(C_dir,dir_w_C)
        C_U        = np.append(C_U,U_C)
        C_uvw      = np.append(C_uvw,uvw_C)    
    
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_A_time_1h.npz', A_time_ref)
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_A_dir_1h.npz', A_dir)
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_A_U_1h.npz', A_U)
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_A_uvw_1h.npz', A_uvw)
    
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_B_time_1h.npz', B_time_ref)
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_B_dir_1h.npz', B_dir)
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_B_U_1h.npz', B_U)
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_B_uvw_1h.npz', B_uvw)

    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_C_time_1h.npz', C_time_ref)
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_C_dir_1h.npz', C_dir)
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_C_U_1h.npz', C_U)
    np.savez_compressed('D:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_C_uvw_1h.npz', C_uvw)

    