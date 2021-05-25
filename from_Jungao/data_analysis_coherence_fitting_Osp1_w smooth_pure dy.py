
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  18 13:56:06 2019
Read the processed storm events
and calculate the coherence parameters 

@author: JUNWAN
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

# =============================================================================
# boom_direction_mag_from_mast_deg    = '358.0 deg (154003), 178.0 deg (154004), 358.0 deg (154005)'
# instrument_north_direction_mag_deg  = '178.0 deg (154003), 358.0 deg (154004), 178.0 deg (154005)'
# instrument_north_direction_true_deg = '178.57 deg (154003), 358.57 deg (154004), 178.57 deg (154005)'
# =============================================================================


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
    
# round time to 1 minute
def round_time(dt=None, round_to=60):
   if dt == None: 
       dt = datetime.now()
   seconds = (dt - dt.min).seconds
   rounding = (seconds+round_to/2) // round_to * round_to
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)
  
    
regex_num  = r"[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?" 

z0=0.01   # roughness lengthscale
delta_ms         = timedelta(milliseconds=100)
delta_2d         = timedelta(days=2)
delta_30min      = timedelta(minutes=30)



#%%
cases = pd.read_excel('Storm events.xlsx', sheet_name='Clustered wind storms sort (t)')
delta_1d         = mdates.date2num(datetime(2,1,2,1,0))-mdates.date2num(datetime(2,1,1,1,0))
delta_1h         = mdates.date2num(datetime(2,1,1,2,0))-mdates.date2num(datetime(2,1,1,1,0))
delta_10min      = mdates.date2num(datetime(2,1,1,1,10))-mdates.date2num(datetime(2,1,1,1,0))

for event in range(0,cases['Time_storm'].size):

    #event = 0
    time  = datetime.strptime(cases['Time_storm'][event], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
    
    # starting and ending time of analyzed data window (based on storm mean wind speed time history)
    time_s1 = datetime.strptime(cases['Time_s1'][event], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
    time_e1 = datetime.strptime(cases['Time_e1'][event], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)  

    # Osp1 height info
    info_osp1        = pd.read_excel('E:/DATA/files_osp1.xls', sheet_name='Height')  
    
    if os.path.isfile('E:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl'):
        with open('E:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_1h' +'_ins.pkl', 'rb') as f:    
            Osp1_1h_ins  = pickle.load( f )   
        with open('E:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_1h' +'_mean.pkl', 'rb') as f:    
            Osp1_1h_mean = pickle.load( f )  

        # number of hourly events   
        num_case    = int(np.round((mdates.date2num(time_e1)-mdates.date2num(time_s1))/delta_1h))+1
        
       
        # define the size of the fitted spectral parameters 
        Cux    = np.zeros((num_case,1))  
        Cuy    = np.zeros((num_case,1)) 
        Cuz    = np.zeros((num_case,1)) 
        Cvx    = np.zeros((num_case,1))  
        Cvy    = np.zeros((num_case,1)) 
        Cvz    = np.zeros((num_case,1)) 
        Cwx    = np.zeros((num_case,1))  
        Cwy    = np.zeros((num_case,1)) 
        Cwz    = np.zeros((num_case,1)) 
        
        U      = np.zeros((num_case,3)) 
        Dir    = np.zeros((num_case,3)) 
        Iu     = np.zeros((num_case,3)) 
        Iv     = np.zeros((num_case,3)) 
        Iw     = np.zeros((num_case,3))         
        ddx    = np.zeros((num_case,3)) 
        ddy    = np.zeros((num_case,3))        
    #%%
    
        for flag_case in range(0,num_case):
            #%
            #flag_case=7
            # make sure the starting and ending date is half a hour before and after the time_tag (to be consistent with mean values)
            date_begin = round_time (mdates.num2date(mdates.date2num(time_s1)-0.5*delta_1h + delta_1h*flag_case   ).replace(tzinfo=None))
            date_stop  = round_time (mdates.num2date(mdates.date2num(date_begin) + delta_1h                       ).replace(tzinfo=None))
            # integer hour tag 
            date       = mdates.num2date(mdates.date2num(time_s1)+delta_1h*flag_case   ).replace(tzinfo=None).replace(microsecond=0)
            # get the index of the starting and ending hour index in time 
            
            # the original date number is based on matplotlib 3.1.3, so we have to create new time series here
            # the time_new is created based on the time tag of the mean wind storm data
            date_new_start =  Osp1_1h_mean.iloc[0,0]  - delta_30min          
            date_new_end   =  Osp1_1h_mean.iloc[-1,0] + delta_30min  -  delta_ms 
            time_new = pd.date_range(date_new_start,date_new_end,freq = '100L')        
            time_new = pd.Series(mdates.date2num(time_new))
            
            # corresponding time tag of each hourly event 
            num_s     = time_new.index[time_new == mdates.date2num(date_begin)].tolist()
            num_e     = time_new.index[time_new == mdates.date2num(date_stop)].tolist()
            Num       = Osp1_1h_mean.index[Osp1_1h_mean['Time'].dt.tz_localize(None) == date].tolist()        
            
            
            # get the target time series
            tt    = time_new.iloc[num_s[0]:num_e[0]].reset_index(drop=True)
            
            # check the ratio of NANs in data, if it's more than 10%, we do not work on it. otherwise, we interpolate
            nan_ratio = Osp1_1h_ins.loc[num_s[0]:num_e[0],['A_u','B_u','C_u']].isnull().sum().sum()/Osp1_1h_ins.loc[num_s[0]:num_e[0],['A_u','B_u','C_u']].size.sum()
            if  nan_ratio < 0.05:
                
                # get the raw data (interpolate the data for NAN using linear method)
                
                dd    = np.asarray(Osp1_1h_ins.loc[num_s[0]:num_e[0]-1,['A_dir','B_dir','C_dir']].interpolate(axis=0,limit_direction='both'))
                UU    = np.asarray(Osp1_1h_ins.loc[num_s[0]:num_e[0]-1,['A_u','B_u','C_u']].interpolate(axis=0,limit_direction='both'))        
                VV    = np.asarray(Osp1_1h_ins.loc[num_s[0]:num_e[0]-1,['A_v','B_v','C_v']].interpolate(axis=0,limit_direction='both'))  
                WW    = np.asarray(Osp1_1h_ins.loc[num_s[0]:num_e[0]-1,['A_w','B_w','C_w']].interpolate(axis=0,limit_direction='both'))  
                
                # get the mean value (10 MINS AVERAGING)
                dt          = 0.1
                Nwin        = round(10*60./dt)
                DD          =  np.asarray(pd.DataFrame(dd).rolling(Nwin,min_periods=1).mean())        
                movingU     =  np.asarray(pd.DataFrame(UU).rolling(Nwin,min_periods=1).mean())        
                movingV     =  np.asarray(pd.DataFrame(VV).rolling(Nwin,min_periods=1).mean())        
                movingW     =  np.asarray(pd.DataFrame(WW).rolling(Nwin,min_periods=1).mean())        
                
                # get the turbulent components
                dd[dd<100]  =  dd[dd<100]+360;
                uu          =  UU - movingU
                vv          =  VV - movingV
                ww          =  WW - movingW
                
                # statistical data
                meanU    = np.mean(UU,axis=0).reshape((1,3))
                meanDir  = np.mean(DD,axis=0).reshape((1,3))   
                stdu_mea = np.std(uu,axis=0).reshape((1,3))
                stdv_mea = np.std(vv,axis=0).reshape((1,3))
                stdw_mea = np.std(ww,axis=0).reshape((1,3))
                Iu_mea   = np.divide(stdu_mea,meanU)
                Iv_mea   = np.divide(stdv_mea,meanU)
                Iw_mea   = np.divide(stdw_mea,meanU)
                
                U[flag_case,:]   = meanU 
                Dir[flag_case,:] = meanDir 
                Iu[flag_case,:]  = Iu_mea 
                Iv[flag_case,:]  = Iv_mea 
                Iw[flag_case,:]  = Iw_mea
                
    
                # reference data 
                z0       = 0.01;
                Lux_ref  = 100*np.power((np.asarray(info_osp1.iloc[0,:])/10),0.3);
                Lvx_ref  = 1/4* Lux_ref;
                Lwx_ref  = 1/12*Lux_ref;
                Lux_ref  = Lux_ref.reshape(1,(np.size(Lux_ref)))
                Lvx_ref  = Lvx_ref.reshape(1,(np.size(Lux_ref)))
                Lwx_ref  = Lwx_ref.reshape(1,(np.size(Lux_ref)))
                Iu_ref   = (1./np.log(np.asarray(info_osp1.iloc[0,:])/z0)).reshape((1,np.size(Lux_ref)))
                Iv_ref   = 3/4.*Iu_ref;
                Iw_ref   = 1/2.*Iu_ref;    
                stdu_ref = np.multiply(meanU,Iu_ref)
                stdv_ref = np.multiply(meanU,Iv_ref)
                stdw_ref = np.multiply(meanU,Iw_ref)

                # range of the moving mean velocity
                delta_U  = (np.max(movingU,0)-np.min(movingU,0)).reshape((1,3))
                delta_Dir  = (np.max(DD,0)-np.min(DD,0)).reshape((1,3))

                dist  = 8   # distance between two top sensors are 8m
                yaw_s = -2   # for ospøya, the boom is -2 degress from true north or true south
                yaw_w = np.mean(meanDir[0,0:2])-yaw_s
                if yaw_w>180:
                    yaw_w=np.abs(yaw_w-270)
                else:
                    yaw_w=np.abs(yaw_w-90)
                # dx the along-wind speration distance; dy is the lateral seperation distance ; dz is the vertical seperation distance    
                # dx ad dy are for based on sensor 1 and 2, and dz is based on sensor 1 and 3
                global dx,dy,dz,U_coh  
                dx = dist*np.sin(np.radians(yaw_w))
                dy = dist*np.cos(np.radians(yaw_w))
                dz = np.asarray(info_osp1.iloc[0,:])[0]-np.asarray(info_osp1.iloc[0,:])[2]
                U_coh = np.mean(meanU[0,0:2])
                ddx[flag_case,0]  = dx
                ddy[flag_case,0]  = dy    

            
                #%% this is to check the stationarity range of moving mean should be smaller than +-15% of the static mean      
                if delta_U[0,0]<0.3*meanU[0,0] and delta_Dir[0,0]<20 and dx<=1.5:    # this is to check the stationarity range of moving mean should be smaller than +-10% of the static mean      
                # spectral analysis of the wind time series
                    Nblock   = 6*3; # Nblock = number of overlapping segment. Friday 07/06/2019, I have used Nblock = 2. here I use 6 blocks to have segments of 10 min
                    N        = np.shape(uu)[0];
                    Nfft     = np.round(N/Nblock); 
                    fs       = 1/dt; # sampling frequency
                    
                    f, Su1 = signal.csd(np.transpose(uu[:,0]), np.transpose(uu[:,0]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Su2 = signal.csd(np.transpose(uu[:,1]), np.transpose(uu[:,1]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Su3 = signal.csd(np.transpose(uu[:,2]), np.transpose(uu[:,2]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Su12= signal.csd(np.transpose(uu[:,0]), np.transpose(uu[:,1]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Su13= signal.csd(np.transpose(uu[:,0]), np.transpose(uu[:,2]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                
                    f, Sv1 = signal.csd(np.transpose(vv[:,0]), np.transpose(vv[:,0]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Sv2 = signal.csd(np.transpose(vv[:,1]), np.transpose(vv[:,1]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Sv3 = signal.csd(np.transpose(vv[:,2]), np.transpose(vv[:,2]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Sv12= signal.csd(np.transpose(vv[:,0]), np.transpose(vv[:,1]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Sv13= signal.csd(np.transpose(vv[:,0]), np.transpose(vv[:,2]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                
                    f, Sw1 = signal.csd(np.transpose(ww[:,0]), np.transpose(ww[:,0]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Sw2 = signal.csd(np.transpose(ww[:,1]), np.transpose(ww[:,1]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Sw3 = signal.csd(np.transpose(ww[:,2]), np.transpose(ww[:,2]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Sw12= signal.csd(np.transpose(ww[:,0]), np.transpose(ww[:,1]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                    f, Sw13= signal.csd(np.transpose(ww[:,0]), np.transpose(ww[:,2]),fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                
                    Coh_u  = np.real(np.divide(Su12,np.power(np.multiply(Su1,Su2),0.5)))
                    Coh_v  = np.real(np.divide(Sv12,np.power(np.multiply(Sv1,Sv2),0.5)))
                    Coh_w  = np.real(np.divide(Sw12,np.power(np.multiply(Sw1,Sw2),0.5)))
                
                    Coh_uz  = np.real(np.divide(Su13,np.power(np.multiply(Su1,Su3),0.5)))
                    Coh_vz  = np.real(np.divide(Sv13,np.power(np.multiply(Sv1,Sv3),0.5)))
                    Coh_wz  = np.real(np.divide(Sw13,np.power(np.multiply(Sw1,Sw3),0.5)))
                
                
                    
                
                  
# =============================================================================
#                     def Coh_lateral(f,Cx,Cy):
#                         return np.exp(-1*np.divide(f,U_coh)*np.power(np.power(Cx*dx,2)+np.power(Cy*dy,2),0.5))    
#                     
# =============================================================================
                    # based on Etienne's comments
                    def Coh_lateral(f,Cx,Cy):
                        return np.exp(-1*np.divide(f,U_coh)*np.power(np.power(Cx*dx,2)+np.power(Cy*dy,2),0.5))*np.cos(dx*2*np.pi*f/U_coh) 

                    def Coh_lateral_dy(f,Cy):
                        return np.exp(-1*np.divide(f,U_coh)*Cy*dy)
                    
                    def Coh_vertical(f,Cz):
                        return np.exp(-1*np.divide(f,U_coh)*dz*Cz)      
            
                    # bin average the spectra in logspace
                    num_log = 60
                    f_s_start = np.log10(f[1])
                    f_s_end   = np.log10(f[-1])    
                    f_s  = np.logspace(f_s_start,f_s_end,num_log+1).reshape(num_log+1,1)
                    Coh_u_s = np.zeros((np.size(f_s)-1,1))
                    Coh_v_s = np.zeros((np.size(f_s)-1,1))
                    Coh_w_s = np.zeros((np.size(f_s)-1,1))   
                    Coh_uz_s = np.zeros((np.size(f_s)-1,1))
                    Coh_vz_s = np.zeros((np.size(f_s)-1,1))
                    Coh_wz_s = np.zeros((np.size(f_s)-1,1))     
                
                    Coh_u_s, edges, _ = binned_statistic(f,Coh_u, statistic='mean', bins=f_s[:,0])
                    Coh_v_s, edges, _ = binned_statistic(f,Coh_v, statistic='mean', bins=f_s[:,0])
                    Coh_w_s, edges, _ = binned_statistic(f,Coh_w, statistic='mean', bins=f_s[:,0])
                    Coh_uz_s, edges, _ = binned_statistic(f,Coh_uz, statistic='mean', bins=f_s[:,0])
                    Coh_vz_s, edges, _ = binned_statistic(f,Coh_vz, statistic='mean', bins=f_s[:,0])
                    Coh_wz_s, edges, _ = binned_statistic(f,Coh_wz, statistic='mean', bins=f_s[:,0])    
                    f_s  = edges[:-1]+np.diff(edges)/2
                    nan_index = np.isnan(Coh_u_s)
                    f_s       = f_s[~nan_index]
                    Coh_u_s      = Coh_u_s[~nan_index].reshape((np.size(f_s),1))
                    Coh_v_s      = Coh_v_s[~nan_index].reshape((np.size(f_s),1))
                    Coh_w_s      = Coh_w_s[~nan_index].reshape((np.size(f_s),1))
                    Coh_uz_s      = Coh_uz_s[~nan_index].reshape((np.size(f_s),1))
                    Coh_vz_s      = Coh_vz_s[~nan_index].reshape((np.size(f_s),1))
                    Coh_wz_s      = Coh_wz_s[~nan_index].reshape((np.size(f_s),1))    
                    f_s          = f_s.reshape((np.size(f_s),1))    
                
                    # fit the coherence 0.01Hz<f<0.5Hz
                    fu_s = 0.01
                    fu_e = 1
                
                    indfu_e  =  np.where(abs(f_s-fu_e)==min(abs(f_s-fu_e)))[0]
                    indfu_s  =  np.where(abs(f_s-fu_s)==min(abs(f_s-fu_s)))[0] 
                    

                    Cuy[flag_case,0],  pcov = curve_fit(Coh_lateral_dy,f_s[indfu_s[0]:indfu_e[0],0], Coh_u_s[indfu_s[0]:indfu_e[0],0],p0=(6),maxfev=100000)
                    Cvy[flag_case,0],  pcov = curve_fit(Coh_lateral_dy,f_s[indfu_s[0]:indfu_e[0],0], Coh_v_s[indfu_s[0]:indfu_e[0],0],p0=(6),maxfev=100000)
                    Cwy[flag_case,0],  pcov = curve_fit(Coh_lateral_dy,f_s[indfu_s[0]:indfu_e[0],0], Coh_w_s[indfu_s[0]:indfu_e[0],0],p0=(6),maxfev=100000)
                    
          
                    Cuz[flag_case,0],  pcov = curve_fit(Coh_vertical,f_s[indfu_s[0]:indfu_e[0],0], Coh_uz_s[indfu_s[0]:indfu_e[0],0],p0=(6),maxfev=100000)
                    Cvz[flag_case,0],  pcov = curve_fit(Coh_vertical,f_s[indfu_s[0]:indfu_e[0],0], Coh_vz_s[indfu_s[0]:indfu_e[0],0],p0=(6),maxfev=100000)
                    Cwz[flag_case,0],  pcov = curve_fit(Coh_vertical,f_s[indfu_s[0]:indfu_e[0],0], Coh_wz_s[indfu_s[0]:indfu_e[0],0],p0=(6),maxfev=100000)
                    
                #%%
                    f_hor      = 1/56.06*0.9
                    f_ver      = 1/6.89*0.9
                    matplotlib.use('Agg')
                    
                    plt.close("all")       
                    fig     = plt.figure(figsize=(20, 12))
                    ax1     = plt.subplot(321)
                    ax1.plot([fu_s, fu_s],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                    ax1.plot([fu_e, fu_e],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)            
                    ax1.semilogx(f_s,Coh_u_s,  'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_lateral(f,Cux[flag_case,0],Cuy[flag_case,0]),  'r-', label='$Fitted$ (Cux='+ str(np.round(Cux[flag_case,0],2))+';Cuy='+str(np.round(Cuy[flag_case,0],2))+')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_lateral(f,3,10),  'b--', label='$Ref$ (Cux='+ str(3)+';Cuy='+str(10)+')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6) 
                    ax1.fill_between(f,-10,10,where= f>f_hor,facecolor='grey', alpha=0.5)   
                    ax1.set_xlim(0.005, 2)
                    ax1.set_ylim(np.min(Coh_u_s)*1.1, 1.1)
                    plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                    plt.ylabel(r'$Coh_u$ $horizontal$', fontsize=20)
                    fig.suptitle('Storm No.' + str(event+1)+ ' Dir=' + str(np.round(meanDir[0,0],1)) + '$^o$'+'  Osp1 at height ' + str(np.round(np.asarray(info_osp1.iloc[0,:])[0],1)) + 'm ' \
                                  + 'from '  + date_begin.strftime("%Y-%m-%d %H:%M") + ' to ' + date_stop.strftime("%Y-%m-%d %H:%M") +'\n'+ 'dx=' +str(np.round(dx,2))+'m; '+'dy=' \
                                      +str(np.round(dy,2))+'m; ' + 'dz=' +str(np.round(dz,2))+'m', fontsize=25)
                    plt.rc('xtick', direction='in', color='k')
                    plt.rc('ytick', direction='in', color='k')
                    plt.legend(loc='lower left',ncol=1,fontsize=14)
                    g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                    g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                    ax1.tick_params(axis='both', labelsize=16)
                    plt.minorticks_on()
                    #plt.show()
                
                    ax1     = plt.subplot(323)
                    ax1.plot([fu_s, fu_s],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                    ax1.plot([fu_e, fu_e],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)             
                    ax1.semilogx(f_s,Coh_v_s,  'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_lateral(f,Cvx[flag_case,0],Cvy[flag_case,0]),  'r-', label='$Fitted$ (Cvx='+ str(np.round(Cvx[flag_case,0],2))+';Cvy='+str(np.round(Cvy[flag_case,0],2))+')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_lateral(f,6,6.5),  'b--', label='$Ref$ (Cvx='+ str(6)+';Cvy='+str(6.5)+')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.fill_between(f,-10,10,where= f>f_hor,facecolor='grey', alpha=0.5)   
                    ax1.set_xlim(0.005, 2)
                    ax1.set_ylim(np.min(Coh_v_s)*1.1, 1.1)
                    plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                    plt.ylabel(r'$Coh_v$ $horizontal$', fontsize=20)
                    plt.rc('xtick', direction='in', color='k')
                    plt.rc('ytick', direction='in', color='k')
                    plt.legend(loc='lower left',ncol=1,fontsize=14)
                    g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                    g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                    ax1.tick_params(axis='both', labelsize=16)
                    plt.minorticks_on()
                    #plt.show()   
                
                
                    ax1     = plt.subplot(325)
                    ax1.plot([fu_s, fu_s],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                    ax1.plot([fu_e, fu_e],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)            
                    ax1.semilogx(f_s,Coh_w_s,  'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_lateral(f,Cwx[flag_case,0],Cwy[flag_case,0]),  'r-', label='$Fitted$ (Cwx='+ str(np.round(Cwx[flag_case,0],2))+';Cwy='+str(np.round(Cwy[flag_case,0],2))+')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_lateral(f,3,6.5),  'b--', label='$Ref$ (Cwx='+ str(3)+';Cwy='+str(6.5)+')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.fill_between(f,-10,10,where= f>f_hor,facecolor='grey', alpha=0.5)   
                    ax1.set_xlim(0.005, 2)
                    ax1.set_ylim(np.min(Coh_w_s)*1.1, 1.1)
                    plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                    plt.ylabel(r'$Coh_w$ $horizontal$', fontsize=20)
                    plt.rc('xtick', direction='in', color='k')
                    plt.rc('ytick', direction='in', color='k')
                    plt.legend(loc='lower left',ncol=1,fontsize=14)
                    g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                    g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                    ax1.tick_params(axis='both', labelsize=16)
                    plt.minorticks_on()
                    #plt.show() 
                
                
                    ax1     = plt.subplot(322)
                    ax1.plot([fu_s, fu_s],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                    ax1.plot([fu_e, fu_e],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)            
                    ax1.semilogx(f_s,Coh_uz_s,  'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_vertical(f,Cuz[flag_case,0]),  'r-', label='$Fitted$ (Cuz='+ str(np.round(Cuz[flag_case,0],2)) +')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_vertical(f,10),  'b--', label='$Ref$ (Cuz='+ str(10) +')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.fill_between(f,-10,10,where= f>f_ver,facecolor='grey', alpha=0.5)   
                    ax1.set_xlim(0.005, 2)
                    ax1.set_ylim(np.min(Coh_uz_s)*1.1, 1.1)
                    plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                    plt.ylabel(r'$Coh_u$ $vertical$', fontsize=20)
                    plt.rc('xtick', direction='in', color='k')
                    plt.rc('ytick', direction='in', color='k')
                    plt.legend(loc='lower left',ncol=1,fontsize=14)
                    g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                    g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                    ax1.tick_params(axis='both', labelsize=16)
                    plt.minorticks_on()
                    #plt.show()  
                    
                    ax1     = plt.subplot(324)
                    ax1.plot([fu_s, fu_s],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                    ax1.plot([fu_e, fu_e],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)            
                    ax1.semilogx(f_s,Coh_vz_s,  'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_vertical(f,Cvz[flag_case,0]),  'r-', label='$Fitted$ (Cvz='+ str(np.round(Cvz[flag_case,0],2)) +')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_vertical(f,6.5),  'b--', label='$Ref$ (Cvz='+ str(6.5) +')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.fill_between(f,-10,10,where= f>f_ver,facecolor='grey', alpha=0.5)   
                    ax1.set_xlim(0.005, 2)
                    ax1.set_ylim(np.min(Coh_vz_s)*1.1, 1.1)
                    plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                    plt.ylabel(r'$Coh_v$ $vertical$', fontsize=20)
                    plt.rc('xtick', direction='in', color='k')
                    plt.rc('ytick', direction='in', color='k')
                    plt.legend(loc='lower left',ncol=1,fontsize=14)
                    g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                    g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                    ax1.tick_params(axis='both', labelsize=16)
                    plt.minorticks_on()
                    #plt.show()       
                    
                    ax1     = plt.subplot(326)
                    ax1.plot([fu_s, fu_s],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                    ax1.plot([fu_e, fu_e],[-10, 10],'k-', linewidth=2, markeredgecolor='k',markersize=8)            
                    ax1.semilogx(f_s,Coh_wz_s,  'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_vertical(f,Cwz[flag_case,0]),  'r-', label='$Fitted$ (Cwz='+ str(np.round(Cwz[flag_case,0],2)) +')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.semilogx(f,Coh_vertical(f,3),  'b--', label='$Ref$ (Cwz='+ str(3) +')' \
                                 ,linewidth=1, markeredgecolor='k',markersize=6)
                    ax1.fill_between(f,-10,10,where= f>f_ver,facecolor='grey', alpha=0.5)   
                    ax1.set_xlim(0.005, 2)
                    ax1.set_ylim(np.min(Coh_wz_s)*1.1, 1.1)
                    plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                    plt.ylabel(r'$Coh_w$ $vertical$', fontsize=20)
                    plt.rc('xtick', direction='in', color='k')
                    plt.rc('ytick', direction='in', color='k')
                    plt.legend(loc='lower left',ncol=1,fontsize=14)
                    g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                    g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                    ax1.tick_params(axis='both', labelsize=16)
                    plt.minorticks_on()
                    #plt.show()    
                    plt.close(fig)
                
                    fig.tight_layout(rect=[0, 0, 1, 0.92])
                    
                    path      = 'E:/DATA/Plots/Osp1 Coherence fitting/'
                    save_tite = 'Storm No.' + str(event+1) + '_coherence fitting Osp1_'\
                                 + str(date_begin.year) + '_' + str(date_begin.month) + '_' + str(date_begin.day)\
                                     + '_' + str(date_begin.hour) + 'smooth pure_dy.png'
                    fig.savefig(path + save_tite)    
                            
                 
                        
            #%%            
        result={}
        result['date_start'] = time_s1.strftime("%Y-%m-%d %H:%M")
        result['date_end']   = time_e1.strftime("%Y-%m-%d %H:%M")
        result['U']          = U
        result['Dir']        = Dir  
        result['Iu']         = Iu
        result['Iv']         = Iv
        result['Iw']         = Iw
        result['dx']         = ddx
        result['dy']         = ddy        
        result['Cux']       = Cux   
        result['Cuy']       = Cuy  
        result['Cuz']       = Cuz 
        result['Cvx']       = Cvx   
        result['Cvy']       = Cvy  
        result['Cvz']       = Cvz 
        result['Cwx']       = Cwx   
        result['Cwy']       = Cwy  
        result['Cwz']       = Cwz 
        
        save_tite = 'Storm No.' + str(event+1) + '_coherence fitting Osp1_'\
                     + str(date_begin.year) + '_' + str(date_begin.month) + '_' + str(date_begin.day)\
                         + '_' + str(date_begin.hour) 
        path      = 'E:/DATA/Results/Osp1 Coherence fitting/'
        with open(path + save_tite + 'pure_dy_smooth_json.txt', "w") as outfile:
            json.dump(result, outfile, cls=NumpyEncoder)
                
                
            
            
