# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 08:44:42 2020
select the wind stroms based on four masts and reference station
Criteria:
    mast: top sensor (50m, 1 h average) velocity above 19m/s using 'signal.find_peaks'
    ref light house: 10mins velocity above 19m/s

# old matplotlib 3.1.3 count num from 0001-01-01 00:00:00

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

from sklearn import neighbors, datasets
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


# round time to 1 minute
def round_time(dt=None, round_to=60):
   if dt == None: 
       dt = datetime.now()
   seconds = (dt - dt.min).seconds
   rounding = (seconds+round_to/2) // round_to * round_to
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)
    
    
regex_num  = r"[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?" 

z0=0.01   # roughness lengthscale



#%%load OSP1    
cases = pd.read_excel('files_osp1.xls', sheet_name='files_osp1')
height= pd.read_excel('files_osp1.xls', sheet_name='Height') 
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


# delta_mon is one month time
delta_ms         = timedelta(milliseconds=100)
delta_10min      = mdates.date2num(datetime(2,1,1,1,10))-mdates.date2num(datetime(2,1,1,1,0))
delta_1h         = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
delta_mon        = mdates.date2num(datetime(2,2,1,1,0))-mdates.date2num(datetime(2,1,1,1,0))

date_s_ref = datetime(2015,12,4,0,0,0)
date_e_ref = mdates.num2date(time_e+delta_mon).replace(tzinfo=None)

monthloop  = month_year_iter(date_s_ref.month,date_s_ref.year,date_e_ref.month,date_e_ref.year)

# load all the 1 hour mean values 
A_U    = []
A_Dir  = []
Time   = []
B_U    = []
B_Dir  = []
C_U    = []
C_Dir  = []
for i in range(0,monthloop.shape[0]-1):
    date_start = mdates.date2num( datetime(int(monthloop[i,0]),  int(monthloop[i,1])+1,  1,0,0)           )

    if i==0:
        date_start = mdates.date2num( date_s_ref)
    
    #create the string of 'date_search '
    t_s = mdates.num2date(date_start)
    year = str(t_s.year)
    mon = str(t_s.month)     
    U    =    np.load('E:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_A_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_A_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees
    time =    np.load('E:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_A_time_1h.npz')['arr_0']     
    A_U   = np.append(A_U,U)
    A_Dir = np.append(A_Dir,Dir)
    Time  = np.append(Time,time)
    
    U    =    np.load('E:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_B_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_B_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees    
    B_U   = np.append(B_U,U)
    B_Dir = np.append(B_Dir,Dir)

    U    =    np.load('E:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_C_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/osp1/' +'Osp1_'+ str(year) +'_' + str(mon)+ '_C_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees    
    C_U   = np.append(C_U,U)
    C_Dir = np.append(C_Dir,Dir)    
A_Dir  = (A_Dir+180)%360
B_Dir  = (B_Dir+180)%360
C_Dir  = (C_Dir+180)%360


# =============================================================================
# A_U  = A_U*np.log(10/z0)/np.log(height['A'][0]/z0)
# B_U  = B_U*np.log(10/z0)/np.log(height['B'][0]/z0)
# C_U  = C_U*np.log(10/z0)/np.log(height['C'][0]/z0)
# =============================================================================


Osp1  = {'Time':Time, 'A_U':A_U, 'B_U':B_U, 'C_U':C_U, 'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir}

#%%Load osp2
cases = pd.read_excel('files_osp2.xls', sheet_name='files_osp2')
height= pd.read_excel('files_osp2.xls', sheet_name='Height') 

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


# delta_mon is one month time
delta_ms         = timedelta(milliseconds=100)
delta_10min      = mdates.date2num(datetime(2,1,1,1,10))-mdates.date2num(datetime(2,1,1,1,0))
delta_1h         = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
delta_mon        = mdates.date2num(datetime(2,2,1,1,0))-mdates.date2num(datetime(2,1,1,1,0))

date_s_ref = datetime(2015,12,19,0,0,0)
date_e_ref = mdates.num2date(time_e+delta_mon).replace(tzinfo=None)

monthloop  = month_year_iter(date_s_ref.month,date_s_ref.year,date_e_ref.month,date_e_ref.year)

# load all the 1 hour mean values 
A_U    = []
A_Dir  = []
Time   = []
B_U    = []
B_Dir  = []
C_U    = []
C_Dir  = []
for i in range(0,monthloop.shape[0]-1):
    date_start = mdates.date2num( datetime(int(monthloop[i,0]),  int(monthloop[i,1])+1,  1,0,0)           )

    if i==0:
        date_start = mdates.date2num( date_s_ref)
    
    #create the string of 'date_search '
    t_s = mdates.num2date(date_start)
    year = str(t_s.year)
    mon = str(t_s.month)     
    U    =    np.load('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_A_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_A_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees
    time =    np.load('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_A_time_1h.npz')['arr_0']     
    A_U   = np.append(A_U,U)
    A_Dir = np.append(A_Dir,Dir)
    Time  = np.append(Time,time)
    
    U    =    np.load('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_B_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_B_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees    
    B_U   = np.append(B_U,U)
    B_Dir = np.append(B_Dir,Dir)

    U    =    np.load('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_C_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_C_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees    
    C_U   = np.append(C_U,U)
    C_Dir = np.append(C_Dir,Dir)    
A_Dir  = (A_Dir+180)%360
B_Dir  = (B_Dir+180)%360
C_Dir  = (C_Dir+180)%360


# =============================================================================
# A_U  = A_U*np.log(10/z0)/np.log(height['A'][0]/z0)
# B_U  = B_U*np.log(10/z0)/np.log(height['B'][0]/z0)
# C_U  = C_U*np.log(10/z0)/np.log(height['C'][0]/z0)
# =============================================================================

Osp2  = {'Time':Time, 'A_U':A_U, 'B_U':B_U, 'C_U':C_U, 'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir}

#%%Load Svar
cases = pd.read_excel('files_svar.xls', sheet_name='files_svar')
height= pd.read_excel('files_svar.xls', sheet_name='Height') 

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


# delta_mon is one month time
delta_ms         = timedelta(milliseconds=100)
delta_10min      = mdates.date2num(datetime(2,1,1,1,10))-mdates.date2num(datetime(2,1,1,1,0))
delta_1h         = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
delta_mon        = mdates.date2num(datetime(2,2,1,1,0))-mdates.date2num(datetime(2,1,1,1,0))

date_s_ref = datetime(2015,3,18,0,0,0)
date_e_ref = mdates.num2date(time_e+delta_mon).replace(tzinfo=None)

monthloop  = month_year_iter(date_s_ref.month,date_s_ref.year,date_e_ref.month,date_e_ref.year)

# load all the 1 hour mean values 
A_U    = []
A_Dir  = []
Time   = []
B_U    = []
B_Dir  = []
C_U    = []
C_Dir  = []
for i in range(0,monthloop.shape[0]-1):
    date_start = mdates.date2num( datetime(int(monthloop[i,0]),  int(monthloop[i,1])+1,  1,0,0)           )

    if i==0:
        date_start = mdates.date2num( date_s_ref)
    
    #create the string of 'date_search '
    t_s = mdates.num2date(date_start)
    year = str(t_s.year)
    mon = str(t_s.month)     
    U    =    np.load('E:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_A_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_A_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees
    time =    np.load('E:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_A_time_1h.npz')['arr_0']     
    A_U   = np.append(A_U,U)
    A_Dir = np.append(A_Dir,Dir)
    Time  = np.append(Time,time)
    
    U    =    np.load('E:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_B_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_B_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees    
    B_U   = np.append(B_U,U)
    B_Dir = np.append(B_Dir,Dir)

    U    =    np.load('E:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_C_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/svar/' +'Svar_'+ str(year) +'_' + str(mon)+ '_C_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees    
    C_U   = np.append(C_U,U)
    C_Dir = np.append(C_Dir,Dir)    
A_Dir  = (A_Dir+180)%360
B_Dir  = (B_Dir+180)%360
C_Dir  = (C_Dir+180)%360


# =============================================================================
# A_U  = A_U*np.log(10/z0)/np.log(height['A'][0]/z0)
# B_U  = B_U*np.log(10/z0)/np.log(height['B'][0]/z0)
# C_U  = C_U*np.log(10/z0)/np.log(height['C'][0]/z0)
# =============================================================================

Svar  = {'Time':Time, 'A_U':A_U, 'B_U':B_U, 'C_U':C_U, 'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir}
#%%Load Synn
cases = pd.read_excel('files_synn.xls', sheet_name='files_synn')
height= pd.read_excel('files_synn.xls', sheet_name='Height') 

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


# delta_mon is one month time
delta_ms         = timedelta(milliseconds=100)
delta_10min      = mdates.date2num(datetime(2,1,1,1,10))-mdates.date2num(datetime(2,1,1,1,0))
delta_1h         = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
delta_mon        = mdates.date2num(datetime(2,2,1,1,0))-mdates.date2num(datetime(2,1,1,1,0))

date_s_ref = datetime(2015,2,23,0,0,0)
date_e_ref = mdates.num2date(time_e+delta_mon).replace(tzinfo=None)

monthloop  = month_year_iter(date_s_ref.month,date_s_ref.year,date_e_ref.month,date_e_ref.year)

# load all the 1 hour mean values 
A_U    = []
A_Dir  = []
Time   = []
B_U    = []
B_Dir  = []
C_U    = []
C_Dir  = []
for i in range(0,monthloop.shape[0]-1):
    date_start = mdates.date2num( datetime(int(monthloop[i,0]),  int(monthloop[i,1])+1,  1,0,0)           )

    if i==0:
        date_start = mdates.date2num( date_s_ref)
    
    #create the string of 'date_search '
    t_s = mdates.num2date(date_start)
    year = str(t_s.year)
    mon = str(t_s.month)     
    U    =    np.load('E:/DATA/synn/' +'Synn_'+ str(year) +'_' + str(mon)+ '_A_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/synn/' +'Synn_'+ str(year) +'_' + str(mon)+ '_A_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees
    time =    np.load('E:/DATA/synn/' +'Synn_'+ str(year) +'_' + str(mon)+ '_A_time_1h.npz')['arr_0']     
    A_U   = np.append(A_U,U)
    A_Dir = np.append(A_Dir,Dir)
    Time  = np.append(Time,time)
    
    U    =    np.load('E:/DATA/synn/' +'Synn_'+ str(year) +'_' + str(mon)+ '_B_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/synn/' +'Synn_'+ str(year) +'_' + str(mon)+ '_B_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees    
    B_U   = np.append(B_U,U)
    B_Dir = np.append(B_Dir,Dir)

    U    =    np.load('E:/DATA/synn/' +'Synn_'+ str(year) +'_' + str(mon)+ '_C_U_1h.npz')['arr_0'] 
    Dir  =    np.load('E:/DATA/synn/' +'Synn_'+ str(year) +'_' + str(mon)+ '_C_dir_1h.npz')['arr_0']/np.pi*180  # change radians to degrees    
    C_U   = np.append(C_U,U)
    C_Dir = np.append(C_Dir,Dir)    
A_Dir  = (A_Dir+180)%360
B_Dir  = (B_Dir+180)%360
C_Dir  = (C_Dir+180)%360


# =============================================================================
# A_U  = A_U*np.log(10/z0)/np.log(height['A'][0]/z0)
# B_U  = B_U*np.log(10/z0)/np.log(height['B'][0]/z0)
# C_U  = C_U*np.log(10/z0)/np.log(height['C'][0]/z0)
# =============================================================================

Synn  = {'Time':Time, 'A_U':A_U, 'B_U':B_U, 'C_U':C_U, 'A_Dir':A_Dir, 'B_Dir':B_Dir, 'C_Dir':C_Dir}
#%%Load ref station data

data = pd.read_excel('SLÅTTERØY FYR.xlsx',sheet_name='48330')





#%%
# find the peak above height and with distance of at least 48 hours
Osp1_A_p,_  = signal.find_peaks(Osp1['A_U'],height=19,distance=2*24)
Osp2_A_p,_  = signal.find_peaks(Osp2['A_U'],height=19,distance=2*24)
Svar_A_p,_  = signal.find_peaks(Svar['A_U'],height=19,distance=2*24)
Synn_A_p,_  = signal.find_peaks(Synn['A_U'],height=19,distance=2*24)
Ref_p,_     = signal.find_peaks(data['Ws'],height=19,distance=2*24)


# new matplotlib count num 1970-01-01T00:00 while the old count from 0001-01-01 00:00:00
dateo      = datetime(1968, 12, 31, 0, 0, 0)
delta_date = datenum(dateo)

#get the time tag of each selected storm
Time_Osp1_A_storm  = mdates.num2date(Osp1['Time'][Osp1_A_p])
Time_Osp2_A_storm  = mdates.num2date(Osp2['Time'][Osp2_A_p])
Time_Svar_A_storm  = mdates.num2date(Svar['Time'][Svar_A_p])
Time_Synn_A_storm  = mdates.num2date(Synn['Time'][Synn_A_p])


Time_Ref_storm     = mdates.num2date(mdates.date2num(pd.to_datetime(data['Date'][Ref_p])))
Time_Ref           = mdates.num2date(mdates.date2num(pd.to_datetime(data['Date'])))

# summarzie the all the time tags of storms
Time_storm = mdates.date2num(Time_Osp1_A_storm + Time_Osp2_A_storm + Time_Svar_A_storm + Time_Synn_A_storm + Time_Ref_storm)
Time_storm_sorted = mdates.date2num(mdates.num2date((np.sort(pd.Series(Time_storm).unique())),tz=None))


data_s  = pd.DataFrame(columns=['Time_storm', 'Osp1A_U', 'Osp1A_Dir','Osp1B_U', 'Osp1B_Dir','Osp1C_U', 'Osp1C_Dir',\
                                              'Osp2A_U', 'Osp2A_Dir','Osp2B_U', 'Osp2B_Dir','Osp2C_U', 'Osp2C_Dir',\
                                              'SvarA_U', 'SvarA_Dir','SvarB_U', 'SvarB_Dir','SvarC_U', 'SvarC_Dir',\
                                              'SynnA_U', 'SynnA_Dir','SynnB_U', 'SynnB_Dir','SynnC_U', 'SynnC_Dir',\
                                              'Ref_U', 'Ref_Dir'])
data_s['Time_storm']  = mdates.num2date(Time_storm_sorted)
    
# collect all the data in the selected storm events
# find index in Osp1 correspnds to common storm
idx_osp1 = np.where(np.in1d(Osp1['Time'], Time_storm_sorted))[0]
idx      = np.where(np.in1d(Time_storm_sorted, Osp1['Time'][idx_osp1]))[0]
data_s.loc[idx,'Osp1A_U']   = Osp1['A_U'][idx_osp1]
data_s.loc[idx,'Osp1A_Dir'] = Osp1['A_Dir'][idx_osp1]
data_s.loc[idx,'Osp1B_U']   = Osp1['B_U'][idx_osp1]
data_s.loc[idx,'Osp1B_Dir'] = Osp1['B_Dir'][idx_osp1]
data_s.loc[idx,'Osp1C_U']   = Osp1['C_U'][idx_osp1]
data_s.loc[idx,'Osp1C_Dir'] = Osp1['C_Dir'][idx_osp1]

idx_osp2 = np.where(np.in1d(Osp2['Time'], Time_storm_sorted))[0]
idx      = np.where(np.in1d(Time_storm_sorted, Osp2['Time'][idx_osp2]))[0]
data_s.loc[idx,'Osp2A_U']   = Osp2['A_U'][idx_osp2]
data_s.loc[idx,'Osp2A_Dir'] = Osp2['A_Dir'][idx_osp2]
data_s.loc[idx,'Osp2B_U']   = Osp2['B_U'][idx_osp2]
data_s.loc[idx,'Osp2B_Dir'] = Osp2['B_Dir'][idx_osp2]
data_s.loc[idx,'Osp2C_U']   = Osp2['C_U'][idx_osp2]
data_s.loc[idx,'Osp2C_Dir'] = Osp2['C_Dir'][idx_osp2]

idx_svar = np.where(np.in1d(Svar['Time'], Time_storm_sorted))[0]
idx      = np.where(np.in1d(Time_storm_sorted, Svar['Time'][idx_svar]))[0]
data_s.loc[idx,'SvarA_U']   = Svar['A_U'][idx_svar]
data_s.loc[idx,'SvarA_Dir'] = Svar['A_Dir'][idx_svar]
data_s.loc[idx,'SvarB_U']   = Svar['B_U'][idx_svar]
data_s.loc[idx,'SvarB_Dir'] = Svar['B_Dir'][idx_svar]
data_s.loc[idx,'SvarC_U']   = Svar['C_U'][idx_svar]
data_s.loc[idx,'SvarC_Dir'] = Svar['C_Dir'][idx_svar]

idx_synn = np.where(np.in1d(Synn['Time'], Time_storm_sorted))[0]
idx      = np.where(np.in1d(Time_storm_sorted, Synn['Time'][idx_synn]))[0]
data_s.loc[idx,'SynnA_U']   = Synn['A_U'][idx_synn]
data_s.loc[idx,'SynnA_Dir'] = Synn['A_Dir'][idx_synn]
data_s.loc[idx,'SynnB_U']   = Synn['B_U'][idx_synn]
data_s.loc[idx,'SynnB_Dir'] = Synn['B_Dir'][idx_synn]
data_s.loc[idx,'SynnC_U']   = Synn['C_U'][idx_synn]
data_s.loc[idx,'SynnC_Dir'] = Synn['C_Dir'][idx_synn]

idx_ref = np.where(np.in1d(mdates.date2num(pd.to_datetime(data['Date'])), Time_storm_sorted))[0]
idx     = np.where(np.in1d(Time_storm_sorted, mdates.date2num(pd.to_datetime(data['Date'][idx_ref]))))[0]
data_s.loc[idx,'Ref_U']      = data.loc[idx_ref,'Ws'].values
data_s.loc[idx,'Ref_Dir']    = data.loc[idx_ref,'Wd'].values


#%%
# find the date clusters using KernelDensity dunction defining bandwidth
temp = np.asarray(Time_storm).reshape(-1, 1)
kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=1).fit(temp)
s = np.linspace(temp[0]-10,temp[-1]+10,100000)
e = kde.score_samples(s.reshape(-1,1))
plt.plot(s, e)

mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

mdates.num2date(temp[temp<s[mi[0]]])

print(mdates.num2date(temp[(temp < s[mi[0]])] ))
print(' ')
for i in range(mi.size-1) :
    print(mdates.num2date(temp[(temp >= s[mi[i]]) * (temp <= s[mi[i+1]])]))
    print(' ')

# get the averaged time tag for each storm event (rounded to integer hour)     
time_ss = [] 
time_ss_ref = []     
for i in range(mi.size-1) :
    time_ss.append(mdates.num2date(np.mean(temp[(temp >= s[mi[i]]) * (temp <= s[mi[i+1]])])).replace(microsecond=0, second=0, minute=30) )
    time_ss_ref.append(mdates.num2date(np.mean(temp[(temp >= s[mi[i]]) * (temp <= s[mi[i+1]])])).replace(microsecond=0, second=0, minute=00) )

# get the averaged time tag for each storm event (rounded to integer hour)         
data_ss  = pd.DataFrame(columns=['Time_storm', 'Osp1A_U', 'Osp1A_Dir','Osp1B_U', 'Osp1B_Dir','Osp1C_U', 'Osp1C_Dir',\
                                              'Osp2A_U', 'Osp2A_Dir','Osp2B_U', 'Osp2B_Dir','Osp2C_U', 'Osp2C_Dir',\
                                              'SvarA_U', 'SvarA_Dir','SvarB_U', 'SvarB_Dir','SvarC_U', 'SvarC_Dir',\
                                              'SynnA_U', 'SynnA_Dir','SynnB_U', 'SynnB_Dir','SynnC_U', 'SynnC_Dir',\
                                              'Ref_U', 'Ref_Dir'])
data_ss['Time_storm']  = time_ss

# create a pandas series to find the index for each mast
Time_ss    = mdates.date2num(time_ss)
idx_osp1   = np.where(np.in1d(Osp1['Time'], Time_ss))[0]
idx        = np.where(np.in1d(Time_ss, Osp1['Time'][idx_osp1]))[0]
data_ss.loc[idx,'Osp1A_U']   = Osp1['A_U'][idx_osp1]
data_ss.loc[idx,'Osp1A_Dir'] = Osp1['A_Dir'][idx_osp1]
data_ss.loc[idx,'Osp1B_U']   = Osp1['B_U'][idx_osp1]
data_ss.loc[idx,'Osp1B_Dir'] = Osp1['B_Dir'][idx_osp1]
data_ss.loc[idx,'Osp1C_U']   = Osp1['C_U'][idx_osp1]
data_ss.loc[idx,'Osp1C_Dir'] = Osp1['C_Dir'][idx_osp1]

idx_osp2   = np.where(np.in1d(Osp2['Time'], Time_ss))[0]
idx        = np.where(np.in1d(Time_ss, Osp2['Time'][idx_osp2]))[0]
data_ss.loc[idx,'Osp2A_U']   = Osp2['A_U'][idx_osp2]
data_ss.loc[idx,'Osp2A_Dir'] = Osp2['A_Dir'][idx_osp2]
data_ss.loc[idx,'Osp2B_U']   = Osp2['B_U'][idx_osp2]
data_ss.loc[idx,'Osp2B_Dir'] = Osp2['B_Dir'][idx_osp2]
data_ss.loc[idx,'Osp2C_U']   = Osp2['C_U'][idx_osp2]
data_ss.loc[idx,'Osp2C_Dir'] = Osp2['C_Dir'][idx_osp2]

idx_svar   = np.where(np.in1d(Svar['Time'], Time_ss))[0]
idx        = np.where(np.in1d(Time_ss, Svar['Time'][idx_svar]))[0]
data_ss.loc[idx,'SvarA_U']   = Svar['A_U'][idx_svar]
data_ss.loc[idx,'SvarA_Dir'] = Svar['A_Dir'][idx_svar]
data_ss.loc[idx,'SvarB_U']   = Svar['B_U'][idx_svar]
data_ss.loc[idx,'SvarB_Dir'] = Svar['B_Dir'][idx_svar]
data_ss.loc[idx,'SvarC_U']   = Svar['C_U'][idx_svar]
data_ss.loc[idx,'SvarC_Dir'] = Svar['C_Dir'][idx_svar]

idx_synn   = np.where(np.in1d(Synn['Time'], Time_ss))[0]
idx        = np.where(np.in1d(Time_ss, Synn['Time'][idx_synn]))[0]
data_ss.loc[idx,'SynnA_U']   = Synn['A_U'][idx_synn]
data_ss.loc[idx,'SynnA_Dir'] = Synn['A_Dir'][idx_synn]
data_ss.loc[idx,'SynnB_U']   = Synn['B_U'][idx_synn]
data_ss.loc[idx,'SynnB_Dir'] = Synn['B_Dir'][idx_synn]
data_ss.loc[idx,'SynnC_U']   = Synn['C_U'][idx_synn]
data_ss.loc[idx,'SynnC_Dir'] = Synn['C_Dir'][idx_synn]

Time_ss    = mdates.date2num(time_ss_ref)
idx_ref    = np.where(np.in1d(mdates.date2num(pd.to_datetime(data['Date'])), Time_ss))[0]
idx        = np.where(np.in1d(Time_ss, mdates.date2num(pd.to_datetime(data['Date'][idx_ref]))))[0]
data_ss.loc[idx,'Ref_U']      = data.loc[idx_ref,'Ws'].values
data_ss.loc[idx,'Ref_Dir']    = data.loc[idx_ref,'Wd'].values


#%%

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(221)
ax1.scatter(data_ss['Ref_Dir'], data_ss['Osp1A_Dir'], marker='o',alpha=0.5)
ax1.plot([0,360],[0,360],'k-')
ax1.set_xlim(0, 360)
ax1.set_ylim(0, 360)
plt.xlabel(r'$Dir\_{Ref}$ $( ^o)$', fontsize=20)
plt.ylabel(r'$Dir\_{Osp1\_A}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
ax1.set_title('', fontsize=25)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
fig.suptitle('Wind direction correlation between reference station and top sensors at Bjørnafjorden', fontsize=25)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(222)
ax1.scatter(data_ss['Ref_Dir'], data_ss['Osp2A_Dir'], marker='o',alpha=0.5)
ax1.plot([0,360],[0,360],'k-')
ax1.set_xlim(0, 360)
ax1.set_ylim(0, 360)
plt.xlabel(r'$Dir\_{Ref}$ $( ^o)$', fontsize=20)
plt.ylabel(r'$Dir\_{Osp2\_A}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(223)
ax1.scatter(data_ss['Ref_Dir'], data_ss['SvarA_Dir'], marker='o',alpha=0.5)
ax1.plot([0,360],[0,360],'k-')
ax1.set_xlim(0, 360)
ax1.set_ylim(0, 360)
plt.xlabel(r'$Dir\_{Ref}$ $( ^o)$', fontsize=20)
plt.ylabel(r'$Dir\_{Svar\_A}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax1     = plt.subplot(224)
ax1.scatter(data_ss['Ref_Dir'], data_ss['SynnA_Dir'], marker='o',alpha=0.5)
ax1.plot([0,360],[0,360],'k-')
ax1.set_xlim(0, 360)
ax1.set_ylim(0, 360)
plt.xlabel(r'$Dir\_{Ref}$ $( ^o)$', fontsize=20)
plt.ylabel(r'$Dir\_{Synn\_A}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

fig.tight_layout(rect=[0, 0, 1, 0.95])
save_tite = 'Wind direction correlation between reference station and top sensors at Bjørnafjorden.png'
fig.savefig(save_tite) 

#%% plot the comparison at OSP1
    

time_s = datetime(2019,1,1,0,0,0)
time_e = datetime(2019,1,1,14,0,0)   
time_s = mdates.num2date(Time[0])
time_e = mdates.num2date(Time[-1])
    
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(211)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(Osp1['Time']), Osp1['A_Dir'], 'k-',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=8)
ax1.plot(Time_Osp1_A_storm, Osp1['A_Dir'][Osp1_A_p],  'r*',label='Storm',markeredgecolor='r',markersize=12)
datemin = np.datetime64(time_s, 'm')
datemax = np.datetime64(time_e, 'm') + np.timedelta64(1, 'm')
ax1.set_xlim(datemin, datemax)
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
plt.legend(loc='best',ncol=1,fontsize=16)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax2     = plt.subplot(212)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)
ax2.plot(mdates.num2date(Osp1['Time']), Osp1['A_U'], 'k-',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=8)
ax2.plot(Time_Osp1_A_storm, Osp1['A_U'][Osp1_A_p],  'r*',label='Storm',markeredgecolor='r',markersize=12)

datemin = np.datetime64(time_s, 'm')
datemax = np.datetime64(time_e, 'm') + np.timedelta64(1, 'm')
ax2.set_xlim(datemin, datemax)
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
ax2.set_title('', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
plt.legend(loc='best',ncol=1,fontsize=16)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax2.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


#%% plot OSP1 with Ref station     

time_s = datetime(2019,1,1,0,0,0)
time_e = datetime(2019,1,1,14,0,0)   
time_s = mdates.num2date(Time[0])
time_e = mdates.num2date(Time[-1])
    
plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(211)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
#ax1.plot(mdates.num2date(Osp1['Time']), Osp1['A_Dir'], 'k-',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=8)
ax1.plot(Time_Osp1_A_storm, Osp1['A_Dir'][Osp1_A_p],  'r*',label='Storm (Osp1)',markeredgecolor='r',markersize=12)
#ax1.plot(Time_Ref, data['Wd'],  'b-',label='$Dir_{wi}$ $(Ref)$',markeredgecolor='g',markersize=12,alpha=0.6)
ax1.plot(Time_Ref_storm, data['Wd'][Ref_p],  'g*',label='Storm (Ref)',markeredgecolor='g',markersize=12,alpha=0.6)
datemin = np.datetime64(time_s, 'm')
datemax = np.datetime64(time_e, 'm') + np.timedelta64(1, 'm')
ax1.set_xlim(time_s, time_e)
ax1.set_ylim(0, 360)
ax1.set_title('Comparison of mean wind direction and velocity between Osp1$\_A$ and Slåtterøy Fyr', fontsize=25)
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
plt.legend(loc='best',ncol=2,fontsize=16)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax2     = plt.subplot(212)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)
ax2.plot(mdates.num2date(Osp1['Time']), Osp1['A_U'], 'k-',label=r'$\overline{U} $ $(Osp1\_A)$',markeredgecolor='k',markersize=8)
ax2.plot(Time_Osp1_A_storm, Osp1['A_U'][Osp1_A_p],  'r*',label='Storm (Osp1)',markeredgecolor='r',markersize=12)
ax2.plot(Time_Ref, data['Ws'],  'b-',label=r'$\overline{U} $ $(Ref)$',markeredgecolor='g',markersize=12,alpha=0.5)
ax2.plot(Time_Ref_storm, data['Ws'][Ref_p],  'g*',label='Storm (Ref)',markeredgecolor='g',markersize=12,alpha=0.6)
datemin = np.datetime64(time_s, 'm')
datemax = np.datetime64(time_e, 'm') + np.timedelta64(1, 'm')
ax2.set_xlim(datemin, datemax)
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
ax2.set_title('', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
plt.legend(loc='best',ncol=2,fontsize=16)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax2.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


#%% plot the top sensors of four masts with ref station 
  
time_s = mdates.num2date(min(Osp1['Time'][0],Osp2['Time'][0],Svar['Time'][0],Synn['Time'][0]))
time_e = mdates.num2date(Time[-1])

plt.close("all")       
fig     = plt.figure(figsize=(20, 12))
ax1     = plt.subplot(511)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.plot(mdates.num2date(Osp1['Time']), Osp1['A_U'], 'k-',label=r'$\overline{U} $ $(Osp1\_A)$',markeredgecolor='k',markersize=8)
ax1.plot(Time_Osp1_A_storm, Osp1['A_U'][Osp1_A_p],  'r*',label='Storm (Osp1)',markeredgecolor='r',markersize=12)
datemin = np.datetime64(time_s, 'm')
datemax = np.datetime64(time_e, 'm') + np.timedelta64(1, 'm')
ax1.set_xlim(datemin, datemax)
ax1.set_title('Comparison of 1-h mean wind velocities among top sensors and Slåtterøy Fyr', fontsize=25)
plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
plt.legend(loc='lower left',ncol=1,fontsize=16)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax1.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax2     = plt.subplot(512)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)
ax2.plot(mdates.num2date(Osp2['Time']), Osp2['A_U'], 'k-',label=r'$\overline{U} $ $(Osp2\_A)$',markeredgecolor='k',markersize=8)
ax2.plot(Time_Osp2_A_storm, Osp2['A_U'][Osp2_A_p],  'r*',label='Storm (Osp2)',markeredgecolor='r',markersize=12)
datemin = np.datetime64(time_s, 'm')
datemax = np.datetime64(time_e, 'm') + np.timedelta64(1, 'm')
ax2.set_xlim(time_s, time_e)
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
ax2.set_title('', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
plt.legend(loc='lower left',ncol=1,fontsize=16)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax2.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax3     = plt.subplot(513)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax3.xaxis.set_major_locator(locator)
ax3.xaxis.set_major_formatter(formatter)
ax3.plot(mdates.num2date(Svar['Time']), Svar['A_U'], 'k-',label=r'$\overline{U} $ $(Svar\_A)$',markeredgecolor='k',markersize=8)
ax3.plot(Time_Svar_A_storm, Svar['A_U'][Svar_A_p],  'r*',label='Storm (Svar)',markeredgecolor='r',markersize=12)
datemin = np.datetime64(time_s, 'm')
datemax = np.datetime64(time_e, 'm') + np.timedelta64(1, 'm')
ax3.set_xlim(time_s, time_e)
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
ax3.set_title('', fontsize=25)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
plt.legend(loc='lower left',ncol=1,fontsize=16)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax3.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()

ax4     = plt.subplot(514)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax4.xaxis.set_major_locator(locator)
ax4.xaxis.set_major_formatter(formatter)
ax4.plot(mdates.num2date(Synn['Time']), Synn['A_U'], 'k-',label=r'$\overline{U} $ $(Synn\_A)$',markeredgecolor='k',markersize=8)
ax4.plot(Time_Synn_A_storm, Synn['A_U'][Synn_A_p],  'r*',label='Storm (Synn)',markeredgecolor='r',markersize=12)
datemin = np.datetime64(time_s, 'm')
datemax = np.datetime64(time_e, 'm') + np.timedelta64(1, 'm')
ax4.set_xlim(time_s, time_e)
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
ax4.set_title('', fontsize=25)
ax4.set_ylim(0, 30)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
plt.legend(loc='lower left',ncol=1,fontsize=16)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax4.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


ax5     = plt.subplot(515)
# format the ticks
locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
formatter = mdates.ConciseDateFormatter(locator)
ax5.xaxis.set_major_locator(locator)
ax5.xaxis.set_major_formatter(formatter)
ax5.plot(Time_Ref, data['Ws'],  'b-',label=r'$\overline{U} $ $(Ref)$',markeredgecolor='g',markersize=12,alpha=0.5)
ax5.plot(Time_Ref_storm, data['Ws'][Ref_p],  'g*',label='Storm (Ref)',markeredgecolor='g',markersize=12,alpha=0.6)
datemin = np.datetime64(time_s, 'm')
datemax = np.datetime64(time_e, 'm') + np.timedelta64(1, 'm')
ax5.set_xlim(time_s, time_e)
plt.ylabel(r'$\overline{U} $ (m s$^{-1})$', fontsize=20)
ax5.set_title('', fontsize=25)
ax5.set_ylim(0, 30)
plt.rc('xtick', direction='in', color='k')
plt.rc('ytick', direction='in', color='k')
plt.legend(loc='lower left',ncol=1,fontsize=16)
g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
ax5.tick_params(axis='both', labelsize=16)
plt.minorticks_on()
plt.show()


