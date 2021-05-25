# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 08:44:42 2020
Storm data spectral fitting

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

    #%%
    #event    = 0
    time = datetime.strptime(cases['Time_storm'][event], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
    
    # starting and ending time of analyzed data window (based on storm mean wind speed time history)
    time_s1 = datetime.strptime(cases['Time_s1'][event], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
    time_e1 = datetime.strptime(cases['Time_e1'][event], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)  

    # Synn 
    info_synn        = pd.read_excel('D:/DATA/files_synn.xls', sheet_name='Height')  
    
    if os.path.isfile('D:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl'):
        with open('D:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Synn_1h' +'_ins.pkl', 'rb') as f:    
            Synn_1h_ins  = pickle.load( f )   
        with open('D:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Synn_1h' +'_mean.pkl', 'rb') as f:    
            Synn_1h_mean = pickle.load( f )  



        cases_synn= pd.read_excel('files_synn.xls', sheet_name='files_synn')
     
        # number of hourly events   
        num_case    = int(np.round((mdates.date2num(time_e1)-mdates.date2num(time_s1))/delta_1h))+1
        
        U      = np.zeros((num_case,3)) 
        Dir    = np.zeros((num_case,3)) 
        Iu     = np.zeros((num_case,3)) 
        Iv     = np.zeros((num_case,3)) 
        Iw     = np.zeros((num_case,3)) 
        
        # define the size of the fitted spectral parameters 
        ANu    = np.zeros((num_case,3))  
        ANu_n  = np.zeros((num_case,3)) 
        ANu2   = np.zeros((num_case,3)) 
        ANu_n2 = np.zeros((num_case,3)) 
        BNu2   = np.zeros((num_case,3)) 
        BNu_n2 = np.zeros((num_case,3)) 
        AHu    = np.zeros((num_case,3)) 
        BHu    = np.zeros((num_case,3)) 
        AHu_n  = np.zeros((num_case,3)) 
        BHu_n  = np.zeros((num_case,3)) 
        
        ANv    = np.zeros((num_case,3)) 
        ANv_n  = np.zeros((num_case,3)) 
        ANv2   = np.zeros((num_case,3)) 
        ANv_n2 = np.zeros((num_case,3)) 
        BNv2   = np.zeros((num_case,3)) 
        BNv_n2 = np.zeros((num_case,3)) 
        
        
        ANw    = np.zeros((num_case,3)) 
        ANw_n  = np.zeros((num_case,3)) 
        ANw2   = np.zeros((num_case,3)) 
        ANw_n2 = np.zeros((num_case,3)) 
        BNw2   = np.zeros((num_case,3)) 
        BNw_n2 = np.zeros((num_case,3)) 
        ABw    = np.zeros((num_case,3)) 
        BBw    = np.zeros((num_case,3)) 
        ABw_n  = np.zeros((num_case,3)) 
        BBw_n  = np.zeros((num_case,3))     
        
        for flag_case in range(0,num_case):
    
            #%%
            #flag_case=7
            # make sure the starting and ending date is half a hour before and after the time_tag (to be consistent with mean values)
            date_begin = round_time (mdates.num2date(mdates.date2num(time_s1)-0.5*delta_1h + delta_1h*flag_case   ).replace(tzinfo=None))
            date_stop  = round_time (mdates.num2date(mdates.date2num(date_begin) + delta_1h                       ).replace(tzinfo=None))
            # integer hour tag 
            date       = mdates.num2date(mdates.date2num(time_s1)+delta_1h*flag_case   ).replace(tzinfo=None).replace(microsecond=0)
            # get the index of the starting and ending hour index in time 
            
            # the original date number is based on matplotlib 3.1.3, so we have to create new time series here
            # the time_new is created based on the time tag of the mean wind storm data
            date_new_start =  Synn_1h_mean.iloc[0,0]  - delta_30min          
            date_new_end   =  Synn_1h_mean.iloc[-1,0] + delta_30min  -  delta_ms 
            time_new = pd.date_range(date_new_start,date_new_end,freq = '100L')        
            time_new = pd.Series(mdates.date2num(time_new))
            
            # corresponding time tag of each hourly event 
            num_s     = time_new.index[time_new == mdates.date2num(date_begin)].tolist()
            num_e     = time_new.index[time_new == mdates.date2num(date_stop)].tolist()
            Num       = Synn_1h_mean.index[Synn_1h_mean['Time'].dt.tz_localize(None) == date].tolist()        
            
            
            # get the target time series
            tt    = time_new.iloc[num_s[0]:num_e[0]].reset_index(drop=True)
            
            # check the ratio of NANs in data, if it's more than 10%, we do not work on it. otherwise, we interpolate
            nan_ratio = Synn_1h_ins.loc[num_s[0]:num_e[0],['A_u','B_u','C_u']].isnull().sum().sum()/Synn_1h_ins.loc[num_s[0]:num_e[0],['A_u','B_u','C_u']].size.sum()
            if  nan_ratio < 0.05:
                
                # get the raw data (interpolate the data for NAN using linear method)
                
                dd    = np.asarray(Synn_1h_ins.loc[num_s[0]:num_e[0]-1,['A_dir','B_dir','C_dir']].interpolate(axis=0,limit_direction='both'))
                UU    = np.asarray(Synn_1h_ins.loc[num_s[0]:num_e[0]-1,['A_u','B_u','C_u']].interpolate(axis=0,limit_direction='both'))        
                VV    = np.asarray(Synn_1h_ins.loc[num_s[0]:num_e[0]-1,['A_v','B_v','C_v']].interpolate(axis=0,limit_direction='both'))  
                WW    = np.asarray(Synn_1h_ins.loc[num_s[0]:num_e[0]-1,['A_w','B_w','C_w']].interpolate(axis=0,limit_direction='both'))  
                
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
                Lux_ref  = 100*np.power((np.asarray(info_synn.iloc[0,:])/10),0.3);
                Lvx_ref  = 1/4* Lux_ref;
                Lwx_ref  = 1/12*Lux_ref;
                Lux_ref  = Lux_ref.reshape(1,(np.size(Lux_ref)))
                Lvx_ref  = Lvx_ref.reshape(1,(np.size(Lux_ref)))
                Lwx_ref  = Lwx_ref.reshape(1,(np.size(Lux_ref)))
                Iu_ref   = (1./np.log(np.asarray(info_synn.iloc[0,:])/z0)).reshape((1,np.size(Lux_ref)))
                Iv_ref   = 3/4.*Iu_ref;
                Iw_ref   = 1/2.*Iu_ref;    
                stdu_ref = np.multiply(meanU,Iu_ref)
                stdv_ref = np.multiply(meanU,Iv_ref)
                stdw_ref = np.multiply(meanU,Iw_ref)
    
    
                # spectral analysis of the wind time series
                Nblock   = 6; # Nblock = number of overlapping segment. Friday 07/06/2019, I have used Nblock = 2. here I use 6 blocks to have segments of 30 min
                N        = np.shape(uu)[0];
                Nfft     = np.round(N/Nblock); 
                fs       = 1/dt; # sampling frequency
                
                f, Su = signal.welch(np.transpose(uu), fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                f, Sv = signal.welch(np.transpose(vv), fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                f, Sw = signal.welch(np.transpose(ww), fs, window='hanning',  nperseg=Nfft, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
                
                f  =  f.reshape((np.size(f),1))
                Su =  np.transpose(Su)
                Sv =  np.transpose(Sv)
                Sw =  np.transpose(Sw)
    
    
                # bin average the spectra in logspace
                num_log = 60
                f_s_start = np.log10(1/600)
                f_s_end   = np.log10(5)
                
                f_s  = np.logspace(f_s_start,f_s_end,num_log+1).reshape(num_log+1,1)
                Su_s = np.zeros((np.size(f_s)-1,3))
                Sv_s = np.zeros((np.size(f_s)-1,3))
                Sw_s = np.zeros((np.size(f_s)-1,3))   
                
                for i in range(0, 3):
                    Su_s[:,i], edges, _ = binned_statistic(f[:,0],Su[:,i], statistic='mean', bins=f_s[:,0])
                    Sv_s[:,i], edges, _ = binned_statistic(f[:,0],Sv[:,i], statistic='mean', bins=f_s[:,0])
                    Sw_s[:,i], edges, _ = binned_statistic(f[:,0],Sw[:,i], statistic='mean', bins=f_s[:,0])
                f_s  = edges[:-1]+np.diff(edges)/2
                
                nan_index = np.isnan(Su_s)
                f_s       = f_s [~nan_index[:,0]]
                Su_s      = Su_s[~nan_index].reshape((np.size(f_s),3))
                Sv_s      = Sv_s[~nan_index].reshape((np.size(f_s),3))
                Sw_s      = Sw_s[~nan_index].reshape((np.size(f_s),3))
                f_s       = f_s.reshape((np.size(f_s),1))
                
    # =============================================================================
    #             plt.close("all")       
    #             fig     = plt.figure(figsize=(20, 12))
    #             ax2     = plt.subplot(111)
    #             ax2.loglog(f[:,0],Su[:,0],  'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
    #             ax2.loglog(f_s,Su_s[:,0],  '-r', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
    # =============================================================================
            
                # fit the wind spectra u<0.1Hz,v<0.1Hz,w<0.5Hz
                fu_s = 0.01
                fu_e = 1
                fv_s = 0.01
                fv_e = 1
                fw_s = 0.1
                fw_e = 1
                
                indfu_e  =  np.where(abs(f_s-fu_e)==min(abs(f_s-fu_e)))[0]
                indfv_e  =  np.where(abs(f_s-fv_e)==min(abs(f_s-fv_e)))[0]
                indfw_e  =  np.where(abs(f_s-fw_e)==min(abs(f_s-fw_e)))[0]
                indfu_s  =  np.where(abs(f_s-fu_s)==min(abs(f_s-fu_s)))[0]
                indfv_s  =  np.where(abs(f_s-fv_s)==min(abs(f_s-fv_s)))[0]
                indfw_s  =  np.where(abs(f_s-fw_s)==min(abs(f_s-fw_s)))[0]
                
                
                fre_u_n      =np.dot(f_s,Lux_ref/meanU)
                fre_v_n      =np.dot(f_s,Lvx_ref/meanU)
                fre_w_n      =np.dot(f_s,Lwx_ref/meanU)
                
                Sf_u_n   = np.multiply(np.dot(f_s,1/np.power(stdu_mea,2)),Su_s)
                Sf_v_n   = np.multiply(np.dot(f_s,1/np.power(stdv_mea,2)),Sv_s)
                Sf_w_n   = np.multiply(np.dot(f_s,1/np.power(stdw_mea,2)),Sw_s)
                
                # range of the moving mean velocity
                delta_U  = (np.max(movingU,0)-np.min(movingU,0)).reshape((1,3))
                delta_Dir  = (np.max(DD,0)-np.min(DD,0)).reshape((1,3))
    
    # =============================================================================
    #             plt.close("all")       
    #             fig     = plt.figure(figsize=(20, 12))
    #             ax2     = plt.subplot(311)
    #             locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
    #             formatter = mdates.ConciseDateFormatter(locator)
    #             ax2.xaxis.set_major_locator(locator)
    #             ax2.xaxis.set_major_formatter(formatter)
    #             ax2.plot(mdates.num2date(tt), UU[:,0], 'b-',label='$\overline{U} + u $ $(Synn\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
    #             ax2.plot(mdates.num2date(tt), movingU[:,0], 'r-',label='$\overline{U} $ $(Synn\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
    #             ax2     = plt.subplot(312)
    #             locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
    #             formatter = mdates.ConciseDateFormatter(locator)
    #             ax2.xaxis.set_major_locator(locator)
    #             ax2.xaxis.set_major_formatter(formatter)
    #             ax2.plot(mdates.num2date(tt), UU[:,1], 'b-',label='$\overline{U} + u $ $(Synn\_B)$',markeredgecolor='k',markersize=8,alpha=0.5)
    #             ax2.plot(mdates.num2date(tt), movingU[:,1], 'r-',label='$\overline{U} $ $(Synn\_B)$',markeredgecolor='k',markersize=8,alpha=0.5)
    #             ax2     = plt.subplot(313)
    #             locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
    #             formatter = mdates.ConciseDateFormatter(locator)
    #             ax2.xaxis.set_major_locator(locator)
    #             ax2.xaxis.set_major_formatter(formatter)
    #             ax2.plot(mdates.num2date(tt), UU[:,2], 'b-',label='$\overline{U} + u $ $(Synn\_C)$',markeredgecolor='k',markersize=8,alpha=0.5)
    #             ax2.plot(mdates.num2date(tt), movingU[:,2], 'r-',label='$\overline{U} $ $(Synn\_C)$',markeredgecolor='k',markersize=8,alpha=0.5)            
    # =============================================================================
                #%
                # i here represents the sensor number for each station
                for i in range(0, 3):
                    if delta_U[0,i]<0.3*meanU[0,i] and delta_Dir[0,i]<20:    # this is to check the stationarity range of moving mean should be smaller than +-10% of the static mean      
                        
                        
                        #%
                        # define the wind spectra
                        global tempu,tempu1,tempv,tempv1,tempw,tempw1
                        tempu      = Lux_ref[0,i]*np.power(stdu_mea[0,i],2)/meanU[0,i];     # Lu*stdu^2/U
                        tempu1     = Lux_ref[0,i]/meanU[0,i];                               # L/U       
                        tempu_ref  = Lux_ref[0,i]*np.power(stdu_ref[0,i],2)/meanU[0,i];     # L*std^2/U
                        tempu1_ref = Lux_ref[0,i]/meanU[0,i];                               # L/U        
                        tempv      = Lvx_ref[0,i]*np.power(stdv_mea[0,i],2)/meanU[0,i];     # Lv*stdv^2/U
                        tempv1     = Lvx_ref[0,i]/meanU[0,i];                               # L/U       
                        tempv_ref  = Lvx_ref[0,i]*np.power(stdv_ref[0,i],2)/meanU[0,i];     # L*std^2/U
                        tempv1_ref = Lvx_ref[0,i]/meanU[0,i];                               # L/U  
                        tempw      = Lwx_ref[0,i]*np.power(stdw_mea[0,i],2)/meanU[0,i];     # Lw*stdw^2/U
                        tempw1     = Lwx_ref[0,i]/meanU[0,i];                               # L/U       
                        tempw_ref  = Lwx_ref[0,i]*np.power(stdw_ref[0,i],2)/meanU[0,i];     # L*std^2/U
                        tempw1_ref = Lwx_ref[0,i]/meanU[0,i];                               # L/U
            
            
                        def N400u_ref(f, A):
                            return np.divide(A*tempu_ref,np.power(1+1.5*A*tempu1_ref*f,5/3))
                        def N400v_ref(f, A):
                            return np.divide(A*tempv_ref,np.power(1+1.5*A*tempv1_ref*f,5/3))
                        def N400w_ref(f, A):
                            return np.divide(A*tempw_ref,np.power(1+1.5*A*tempw1_ref*f,5/3))                                            
                        def N400u_1par(f, A):
                            return np.divide(A*tempu,np.power(1+1.5*A*tempu1*f,5/3))            
                        def N400_n_1par(f_hat, A):
                            return np.divide(A*f_hat,np.power(1+1.5*A*f_hat,5/3))            
                        def N400u_2par(f, A, B):
                            return np.divide(A*tempu,np.power(1+1.5*B*tempu1*f,5/3))            
                        def N400_n_2par(f_hat, A, B):
                            return np.divide(A*f_hat,np.power(1+1.5*B*f_hat,5/3))           
                        def Harris_u(f, A, B):
                            return np.divide(A*tempu,np.power(1+B*np.power(f*tempu1,2),5/6))            
                        def Harris_u_n(f_hat, A, B):
                            return np.divide(A*f_hat,np.power(1+B*np.power(f_hat,2),5/6))
                        def N400v_1par(f, A):
                            return np.divide(A*tempv,np.power(1+1.5*A*tempv1*f,5/3))            
                        def N400v_2par(f, A, B):
                            return np.divide(A*tempv,np.power(1+1.5*B*tempv1*f,5/3))            
                        def N400w_1par(f, A):
                            return np.divide(A*tempw,np.power(1+1.5*A*tempw1*f,5/3))            
                        def N400w_2par(f, A, B):
                            return np.divide(A*tempw,np.power(1+1.5*B*tempw1*f,5/3))                        
                        
                        def Froya_u(f, Uref, z):
                            n     = 0.468
                            f_hat = 172*np.power((z/10),2/3)*np.power((Uref/10),-0.75)*f
                            Su    = 320*np.divide(np.power((Uref/10),2)*np.power((z/10),0.45),np.power(1+np.power(f_hat,n),5/(3*n)))
                            return Su
            
                        def Busch_Panofsky_w(f,A,B):
                            return np.divide(A*tempw,1+B*np.power(tempw1*f,5/3))
                        
                        def Busch_Panofsky_w_n(f_hat,A,B):
                            return np.divide(A*f_hat,1+B*np.power(f_hat,5/3))
                        
                        def Sfn_from_Sf(f,Sf):
                            var= np.trapz(Sf,f,axis=0)
                            Sfn= np.multiply(np.dot(f, 1/var).reshape(np.size(f),1),Sf)
                            return Sfn
                        
                        #%
                        # collect the fitted parameters
                        
                        ANu[flag_case,i],  pcov   = curve_fit(N400u_1par,   f_s[indfu_s[0]:indfu_e[0],0],       Su_s[indfu_s[0]:indfu_e[0],i],      p0=6,maxfev=10000)
                        ANu_n[flag_case,i], pcovn = curve_fit(N400_n_1par, fre_u_n[indfu_s[0]:indfu_e[0],i], Sf_u_n [indfu_s[0]:indfu_e[0],i], p0=6,maxfev=10000)           
                        [ANu2[flag_case,i],BNu2[flag_case,i]], pcov   = curve_fit(N400u_2par,   f_s[indfu_s[0]:indfu_e[0],0],       Su_s[indfu_s[0]:indfu_e[0],i],      p0=(6,6),maxfev=10000)
                        [ANu_n2[flag_case,i],BNu_n2[flag_case,i]], pcovn = curve_fit(N400_n_2par, fre_u_n[indfu_s[0]:indfu_e[0],i], Sf_u_n [indfu_s[0]:indfu_e[0],i], p0=(6,6),maxfev=10000)            
                        [AHu[flag_case,i],BHu[flag_case,i]], pcovn   = curve_fit(Harris_u,   f_s[indfu_s[0]:indfu_e[0],0],       Su_s[indfu_s[0]:indfu_e[0],i],      p0=(4,70),maxfev=10000)
                        [AHu_n[flag_case,i],BHu_n[flag_case,i]], pcovn = curve_fit(Harris_u_n, fre_u_n[indfu_s[0]:indfu_e[0],i], Sf_u_n [indfu_s[0]:indfu_e[0],i], p0=(4,70),maxfev=10000)
            
                        ANv[flag_case,i], pcov   = curve_fit(N400v_1par,   f_s[indfv_s[0]:indfv_e[0],0],       Sv_s[indfv_s[0]:indfv_e[0],i],      p0=6,maxfev=10000)
                        ANv_n[flag_case,i], pcovn = curve_fit(N400_n_1par, fre_v_n[indfv_s[0]:indfv_e[0],i], Sf_v_n [indfv_s[0]:indfv_e[0],i], p0=6,maxfev=10000)            
                        [ANv2[flag_case,i],BNv2[flag_case,i]], pcov   = curve_fit(N400v_2par,   f_s[indfv_s[0]:indfv_e[0],0],       Sv_s[indfv_s[0]:indfv_e[0],i],      p0=(6,6),maxfev=10000)
                        [ANv_n2[flag_case,i],BNv_n2[flag_case,i]], pcovn = curve_fit(N400_n_2par, fre_v_n[indfv_s[0]:indfv_e[0],i], Sf_v_n [indfv_s[0]:indfv_e[0],i], p0=(6,6),maxfev=10000)            
            
                        ANw[flag_case,i], pcov   = curve_fit(N400w_1par,          f_s[indfw_s[0]:indfw_e[0],0],       Sw_s[indfw_s[0]:indfw_e[0],i],      p0=6,maxfev=10000)
                        ANw_n[flag_case,i], pcovn = curve_fit(N400_n_1par,        fre_w_n[indfw_s[0]:indfw_e[0],i], Sf_w_n [indfw_s[0]:indfw_e[0],i], p0=6,maxfev=10000)            
                        [ANw2[flag_case,i],BNw2[flag_case,i]], pcov   = curve_fit(N400w_2par,          f_s[indfw_s[0]:indfw_e[0],0],       Sw_s[indfw_s[0]:indfw_e[0],i],      p0=(6,6),maxfev=10000)
                        [ANw_n2[flag_case,i],BNw_n2[flag_case,i]], pcovn = curve_fit(N400_n_2par,        fre_w_n[indfw_s[0]:indfw_e[0],i], Sf_w_n [indfw_s[0]:indfw_e[0],i], p0=(6,6),maxfev=10000)           
                        [ABw[flag_case,i],BBw[flag_case,i]], pcov     = curve_fit(Busch_Panofsky_w,   f_s[indfw_s[0]:indfw_e[0],0],       Sw_s[indfw_s[0]:indfw_e[0],i],      p0=(6,6),maxfev=10000)
                        [ABw_n[flag_case,i],BBw_n[flag_case,i]], pcovn   = curve_fit(Busch_Panofsky_w_n, fre_w_n[indfw_s[0]:indfw_e[0],i], Sf_w_n [indfw_s[0]:indfw_e[0],i], p0=(6,6),maxfev=10000)            
                    
                        matplotlib.use('Agg')
                    
                        #% plot the spectra in the raw and normalized format
                        plt.close("all")
                        fig     = plt.figure(figsize=(20, 12))
                        ax2     = plt.subplot(111)
        
                        ax2.loglog(fre_u_n[:,i],N400_n_1par(fre_u_n[:,i], 6.8),                 'b--',label='$N400$ $ ref 6.8$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_u_n[:,i],N400_n_1par(fre_u_n[:,i], 9.4),                 'r--',label='$N400$ $ ref 9.4$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_u_n[:,i],N400_n_1par(fre_u_n[:,i], 20),                 'k--',label='$N400$ $ ref 20$',linewidth=2, markeredgecolor='k',markersize=8)
        
                        ax2.set_xlim(0.005, 15)               
                        plt.xlabel(r'$f*L_{ux}/\overline{U}$', fontsize=20)
                        plt.ylabel(r'$f*S_{fu}$ $/\sigma_{u\_measure}^2$', fontsize=20)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='upper right',ncol=2,fontsize=16)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax2.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        #%
                        # calculate the reference wind speed at 10 m height
                        f_hor      = 1/56.06*0.9
                        f_ver      = 1/6.89*0.9
                        
                        Uref       =  meanU[0,i]*(1+np.log(np.asarray(info_synn.iloc[0,:])[i]/10)/np.log(10/z0))
                        Su_n_froya =  Sfn_from_Sf(f_s,Froya_u(f_s[:,0], Uref, np.asarray(info_synn.iloc[0,:])[i]).reshape(np.size(f_s),1))
                        plt.close("all")
                        fig     = plt.figure(figsize=(20, 12))
                        ax1     = plt.subplot(321)
                        ax1.plot([fu_s, fu_s],[np.min(Su[:,i]), np.max(Su[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot([fu_e, fu_e],[np.min(Su[:,i]), np.max(Su[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f_s[:,0],Su_s[:,i],                                               'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                        ax1.semilogx(f[:,0],N400u_1par(f[:,0], ANu[flag_case,i]),                      'r-', label='$Fitted$ $ N400$ $( 1 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],N400u_2par(f[:,0], ANu2[flag_case,i],BNu2[flag_case,i]),   'g-', label='$Fitted$ $ N400$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],Harris_u(f[:,0], AHu[flag_case,i], BHu[flag_case,i]),     'y-', label='$Fitted$ $ Harris$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],N400u_ref(f[:,0], 6.8),                                   'b--',label='$N400$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],Froya_u(f[:,0], Uref, np.asarray(info_synn.iloc[0,:])[i]),  'c--',label='$Frøya$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)           
                        ax1.fill_between(f[:,0],0,1000,where= f[:,0]>f_hor,facecolor='grey', alpha=0.5)
                        ax1.set_xlim(0.005, 2)
                        ax1.set_ylim(0, 1.2*np.max(Su_s[2:-1,i]))
                        plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                        plt.ylabel(r'$S_{fu}$ ($m^2 s^{-1})$', fontsize=20)
                        fig.suptitle('Storm No.' + str(event+1)+ ' Dir=' + str(np.round(meanDir[0,i],1)) + '$^o$'+'  Synn at height ' + str(np.round(np.asarray(info_synn.iloc[0,:])[i],1)) + 'm ' \
                                      + 'from '  + date_begin.strftime("%Y-%m-%d %H:%M") + ' to ' + date_stop.strftime("%Y-%m-%d %H:%M"), fontsize=25)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='upper right',ncol=2,fontsize=16)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax1.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        ax2     = plt.subplot(322)
                        ax2.plot([fre_u_n[indfu_s[0],i], fre_u_n[indfu_s[0],i]],[0, 1000],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.plot([fre_u_n[indfu_e[0],i], fre_u_n[indfu_e[0],i]],[0, 1000],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_u_n[:,i],Sf_u_n[:,i],                                    'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                        ax2.loglog(fre_u_n[:,i],N400_n_1par(fre_u_n[:,i], ANu_n[flag_case,i]),            'r-', label='$Fitted$ $ N400$ $( 1 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_u_n[:,i],N400_n_2par(fre_u_n[:,i], ANu_n2[flag_case,i],BNu_n2[flag_case,i]), 'g-', label='$Fitted$ $ N400$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_u_n[:,i],Harris_u_n(fre_u_n[:,i], AHu_n[flag_case,i], BHu_n[flag_case,i]),   'y-', label='$Fitted$ $ Harris$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_u_n[:,i],N400_n_1par(fre_u_n[:,i], 6.8),                 'b--',label='$N400$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_u_n[:,i],Su_n_froya,                                     'c--',label='$Frøya$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.fill_between(fre_u_n[:,i],0,1000,where= fre_u_n[:,i]>f_hor*Lux_ref[0,i]/meanU[0,i],facecolor='grey', alpha=0.5)
                        ax2.set_ylim(0.6*Sf_u_n[-1,i],  1.3*np.max(Sf_u_n[:,i]))
                        ax2.set_xlim(0.05, 15)               
                        plt.xlabel(r'$f*L_{ux}/\overline{U}$', fontsize=20)
                        plt.ylabel(r'$f*S_{fu}$ $/\sigma_{u\_measure}^2$', fontsize=20)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='lower left',ncol=2,fontsize=16)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax2.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        
                        ax1     = plt.subplot(323)
                        ax1.plot([fv_s, fv_s],[np.min(Sv[:,i]), np.max(Sv[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot([fv_e, fv_e],[np.min(Sv[:,i]), np.max(Sv[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f_s[:,0],Sv_s[:,i],                                               'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                        ax1.semilogx(f[:,0],N400v_1par(f[:,0], ANv[flag_case,i]),                      'r-', label='$Fitted$ $ N400$ $( 1 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],N400v_2par(f[:,0], ANv2[flag_case,i],BNv2[flag_case,i]),   'g-', label='$Fitted$ $ N400$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],N400v_ref(f[:,0], 9.4),                                   'b--',label='$N400$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.fill_between(f[:,0],0,1000,where= f[:,0]>f_hor,facecolor='grey', alpha=0.5)          
                        ax1.set_xlim(0.005, 2)
                        ax1.set_ylim(0, 1.2*np.max(Sv_s[:,i]))
                        plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                        plt.ylabel(r'$S_{fv}$ ($m^2 s^{-1})$', fontsize=20)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='upper right',ncol=2,fontsize=16)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax1.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                         
                        ax2     = plt.subplot(324)
                        ax2.plot([fre_v_n[indfv_s[0],i], fre_v_n[indfv_s[0],i]],[0, 1000],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.plot([fre_v_n[indfv_e[0],i], fre_v_n[indfv_e[0],i]],[0, 1000],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_v_n[:,i],Sf_v_n[:,i],                                    'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                        ax2.loglog(fre_v_n[:,i],N400_n_1par(fre_v_n[:,i], ANv_n[flag_case,i]),            'r-', label='$Fitted$ $ N400$ $( 1 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_v_n[:,i],N400_n_2par(fre_v_n[:,i], ANv_n2[flag_case,i],BNv_n2[flag_case,i]), 'g-', label='$Fitted$ $ N400$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_v_n[:,i],N400_n_1par(fre_v_n[:,i], 6.8),                 'b--',label='$N400$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.fill_between(fre_v_n[:,i],0,1000,where= fre_v_n[:,i]>f_hor*Lvx_ref[0,i]/meanU[0,i],facecolor='grey', alpha=0.5)
                        ax2.set_ylim(0.6*Sf_v_n[-1,i], 1.3*np.max(Sf_v_n[:,i]))  
                        ax2.set_xlim(0.01, 5)                        
                        plt.xlabel(r'$f*L_{vx}/\overline{U}$', fontsize=20)
                        plt.ylabel(r'$f*S_{fv}$ $/\sigma_{v\_measure}^2$', fontsize=20)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='lower left',ncol=2,fontsize=16)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax2.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
            
                        ax1     = plt.subplot(325)
                        ax1.plot([fw_s, fw_s],[np.min(Sw[:,i]), np.max(Sw[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot([fw_e, fw_e],[np.min(Sw[:,i]), np.max(Sw[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f_s[:,0],Sw_s[:,i],                                                  'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                        ax1.plot(f[:,0],N400w_1par(f[:,0], ANw[flag_case,i]),                      'r-', label='$Fitted$ $ N400$ $( 1 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot(f[:,0],N400w_2par(f[:,0], ANw2[flag_case,i],BNw2[flag_case,i]),   'g-', label='$Fitted$ $ N400$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot(f[:,0],Busch_Panofsky_w(f[:,0], ABw[flag_case,i], BBw[flag_case,i]),     'y-', label='$Fitted$ $ Busch Panofsky$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot(f[:,0],N400w_ref(f[:,0], 6.8),                                   'b--',label='$N400$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.fill_between(f[:,0],0,1000,where= f[:,0]>f_ver,facecolor='grey', alpha=0.5)           
                        ax1.set_xlim(0.01, 2)
                        ax1.set_ylim(0, 1.2*np.max(Sw_s[4:-1,i]))
                        plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                        plt.ylabel(r'$S_{fw}$ ($m^2 s^{-1})$', fontsize=20)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='upper right',ncol=2,fontsize=16)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax1.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        
                        ax2     = plt.subplot(326)
                        ax2.plot([fre_w_n[indfw_s[0],i], fre_w_n[indfw_s[0],i]],[0, 1000],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.plot([fre_w_n[indfw_e[0],i], fre_w_n[indfw_e[0],i]],[0, 1000],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_w_n[:,i],Sf_w_n[:,i],                                    'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                        ax2.loglog(fre_w_n[:,i],N400_n_1par(fre_w_n[:,i], ANw_n[flag_case,i]),            'r-', label='$Fitted$ $ N400$ $( 1 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_w_n[:,i],N400_n_2par(fre_w_n[:,i], ANw_n2[flag_case,i],BNw_n2[flag_case,i]), 'g-', label='$Fitted$ $ N400$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_w_n[:,i],Busch_Panofsky_w_n(fre_w_n[:,i], ABw_n[flag_case,i], BBw_n[flag_case,i]),   'y-', label='$Fitted$ $ Busch Panofsky$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.loglog(fre_w_n[:,i],N400_n_1par(fre_w_n[:,i], 6.8),                 'b--',label='$N400$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax2.fill_between(fre_w_n[:,i],0,1000,where= fre_w_n[:,i]>f_ver*Lwx_ref[0,i]/meanU[0,i],facecolor='grey', alpha=0.5)
                        ax2.set_ylim(0.6*Sf_w_n[-1,i], 1.3*np.max(Sf_w_n[:,i]))   
                        ax2.set_xlim(0.01, 2)                        
                        plt.xlabel(r'$f*L_{wx}/\overline{U}$', fontsize=20)
                        plt.ylabel(r'$f*S_{fw}$ $/\sigma_{w\_measure}^2$', fontsize=20)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='lower left',ncol=2,fontsize=16)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax2.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        plt.close(fig)
                        
                        path      = 'D:/DATA/Plots/Synn Spectra fitting/'
                        save_tite = 'Storm No.' + str(event+1) + '_spectrum fitting Synn sensor No.' + str(i+1) + '_'\
                                     + str(date_begin.year) + '_' + str(date_begin.month) + '_' + str(date_begin.day)\
                                         + '_' + str(date_begin.hour) + 'smooth.png'
                        fig.savefig(path + save_tite)    
                        
                        
                        
                        
                        
                        
            
                        #% plot the time histories and fitted raw spectra
                        f_hor      = 1/56.06*0.9
                        f_ver      = 1/6.89*0.9  
                        Uref       =  meanU[0,i]*(1+np.log(np.asarray(info_synn.iloc[0,:])[i]/10)/np.log(10/z0))
                        Su_n_froya =  Sfn_from_Sf(f_s,Froya_u(f_s[:,0], Uref, np.asarray(info_synn.iloc[0,:])[i]).reshape(np.size(f_s),1))
                        plt.close("all")       
                        fig     = plt.figure(figsize=(20, 12))
                        ax1     = plt.subplot(421)
                        # format the ticks
                        locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
                        formatter = mdates.ConciseDateFormatter(locator)
                        ax1.xaxis.set_major_locator(locator)
                        ax1.xaxis.set_major_formatter(formatter)
                        ax1.plot(mdates.num2date(tt), dd[:,i], 'k-',label='$Dir_{wi}$',markeredgecolor='k',markersize=8, alpha =0.7)
                        ax1.plot(mdates.num2date(tt), DD[:,i], 'r-',label='$Mean$ $ Dir_{wi}$',markeredgecolor='k',markersize=8)
                        datemin = np.datetime64(mdates.num2date(tt)[0].replace(tzinfo=None), 'm')
                        datemax = np.datetime64(mdates.num2date(tt)[-1].replace(tzinfo=None), 'm') + np.timedelta64(1, 'm')
                        ax1.set_xlim(datemin, datemax)
                        ax1.set_xticklabels([])
                        plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
                        fig.suptitle('Storm No.' + str(event+1) + ' Synn at height ' + str(np.round(np.asarray(info_synn.iloc[0,:])[i],1)) + 'm '\
                                     + 'from '  + date_begin.strftime("%Y-%m-%d %H:%M") + ' to ' + date_stop.strftime("%Y-%m-%d %H:%M"), fontsize=25)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax1.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        
                        ax2     = plt.subplot(422)
                        # format the ticks
                        locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
                        formatter = mdates.ConciseDateFormatter(locator)
                        ax2.xaxis.set_major_locator(locator)
                        ax2.xaxis.set_major_formatter(formatter)
                        ax2.plot([mdates.num2date(tt)[0].replace(tzinfo=None), mdates.num2date(tt)[-1].replace(tzinfo=None)], [1.15*meanU[0,i],1.15*meanU[0,i]], 'b--',markeredgecolor='k',markersize=8, linewidth=2)
                        ax2.plot([mdates.num2date(tt)[0].replace(tzinfo=None), mdates.num2date(tt)[-1].replace(tzinfo=None)], [0.85*meanU[0,i],0.85*meanU[0,i]], 'b--',markeredgecolor='k',markersize=8, linewidth=2)
                        ax2.plot(mdates.num2date(tt), UU[:,i]+uu[:,i], 'k-',label='$\overline{u} + u^{\prime}$',markeredgecolor='k',markersize=8, alpha =0.7)
                        ax2.plot(mdates.num2date(tt), movingU[:,i]   , 'r-',label='$\overline{u}$',markeredgecolor='k',markersize=8)
                        datemin = np.datetime64(mdates.num2date(tt)[0].replace(tzinfo=None), 'm')
                        datemax = np.datetime64(mdates.num2date(tt)[-1].replace(tzinfo=None), 'm') + np.timedelta64(1, 'm')
                        ax2.set_xlim(datemin, datemax)
                        plt.ylabel(r'$\overline{U} + u$ (m s$^{-1})$', fontsize=20)
                        ax2.set_title('', fontsize=25)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax2.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        
                        ax2     = plt.subplot(423)
                        # format the ticks
                        locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
                        formatter = mdates.ConciseDateFormatter(locator)
                        ax2.xaxis.set_major_locator(locator)
                        ax2.xaxis.set_major_formatter(formatter)
                        ax2.plot(mdates.num2date(tt), uu[:,i]        , 'r-',label='$\overline{u}$',markeredgecolor='k',markersize=8)
                        datemin = np.datetime64(mdates.num2date(tt)[0].replace(tzinfo=None), 'm')
                        datemax = np.datetime64(mdates.num2date(tt)[-1].replace(tzinfo=None), 'm') + np.timedelta64(1, 'm')
                        ax2.set_xlim(datemin, datemax)
                        ax2.set_xticklabels([])
                        plt.ylabel(r'$u$ (m s$^{-1})$', fontsize=20)
                        ax2.set_title('', fontsize=25)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax2.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                                    
                        ax1     = plt.subplot(424)
                        # format the ticks
                        ax1.plot([fu_s, fu_s],[np.min(Su[:,i]), np.max(Su[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot([fu_e, fu_e],[np.min(Su[:,i]), np.max(Su[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f_s[:,0],Su_s[:,i],                                               'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                        ax1.semilogx(f[:,0],N400u_1par(f[:,0], ANu[flag_case,i]),                      'r-', label='$Fitted$ $ N400$ $( 1 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],N400u_2par(f[:,0], ANu2[flag_case,i],BNu2[flag_case,i]),   'g-', label='$Fitted$ $ N400$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],Harris_u(f[:,0], AHu[flag_case,i], BHu[flag_case,i]),     'y-', label='$Fitted$ $ Harris$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],N400u_ref(f[:,0], 6.8),                                   'b--',label='$N400$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],Froya_u(f[:,0], Uref, np.asarray(info_synn.iloc[0,:])[i]),  'c--',label='$Frøya$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)           
                        ax1.fill_between(f[:,0],0,1000,where= f[:,0]>f_hor,facecolor='grey', alpha=0.5)
                        ax1.set_xlim(0.005, 2)
                        ax1.set_ylim(0, 1.2*np.max(Su_s[2:-1,i]))
                        plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                        plt.ylabel(r'$S_{fu}$ ($m^2 s^{-1})$', fontsize=20)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='upper right',ncol=3,fontsize=14,framealpha =0.6)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax1.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()           
                    
                        ax2     = plt.subplot(425)
                        # format the ticks
                        locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
                        formatter = mdates.ConciseDateFormatter(locator)
                        ax2.xaxis.set_major_locator(locator)
                        ax2.xaxis.set_major_formatter(formatter)
                        ax2.plot(mdates.num2date(tt), vv[:,i]        , 'b-',label='$\overline{v}$',markeredgecolor='k',markersize=8)
                        datemin = np.datetime64(mdates.num2date(tt)[0].replace(tzinfo=None), 'm')
                        datemax = np.datetime64(mdates.num2date(tt)[-1].replace(tzinfo=None), 'm') + np.timedelta64(1, 'm')
                        ax2.set_xlim(datemin, datemax)
                        ax2.set_xticklabels([])
                        plt.ylabel(r'$v$ (m s$^{-1})$', fontsize=20)
                        ax2.set_title('', fontsize=25)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax2.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        
                        ax1     = plt.subplot(426)
                        # format the ticks
                        ax1.plot([fv_s, fv_s],[np.min(Sv[:,i]), np.max(Sv[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot([fv_e, fv_e],[np.min(Sv[:,i]), np.max(Sv[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f_s[:,0],Sv_s[:,i],                                               'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                        ax1.semilogx(f[:,0],N400v_1par(f[:,0], ANv[flag_case,i]),                      'r-', label='$Fitted$ $ N400$ $( 1 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],N400v_2par(f[:,0], ANv2[flag_case,i],BNv2[flag_case,i]),   'g-', label='$Fitted$ $ N400$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f[:,0],N400v_ref(f[:,0], 9.4),                                   'b--',label='$N400$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.fill_between(f[:,0],0,1000,where= f[:,0]>f_hor,facecolor='grey', alpha=0.5)          
                        ax1.set_xlim(0.005, 2)
                        ax1.set_ylim(0, 1.2*np.max(Sv_s[:,i]))
                        plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                        plt.ylabel(r'$S_{fv}$ ($m^2 s^{-1})$', fontsize=20)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='upper right',ncol=2,fontsize=16)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax1.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        
                        ax2     = plt.subplot(427)
                        # format the ticks
                        locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
                        formatter = mdates.ConciseDateFormatter(locator)
                        ax2.xaxis.set_major_locator(locator)
                        ax2.xaxis.set_major_formatter(formatter)
                        ax2.plot(mdates.num2date(tt), ww[:,i]        , 'g-',label='$\overline{w}$',markeredgecolor='k',markersize=8)
                        datemin = np.datetime64(mdates.num2date(tt)[0].replace(tzinfo=None), 'm')
                        datemax = np.datetime64(mdates.num2date(tt)[-1].replace(tzinfo=None), 'm') + np.timedelta64(1, 'm')
                        ax2.set_xlim(datemin, datemax)
                        plt.ylabel(r'$w$ (m s$^{-1})$', fontsize=20)
                        ax2.set_title('', fontsize=25)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax2.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        
                        ax1     = plt.subplot(428)
                        # format the ticks
                        ax1.plot([fw_s, fw_s],[np.min(Sw[:,i]), np.max(Sw[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot([fw_e, fw_e],[np.min(Sw[:,i]), np.max(Sw[:,i])],'k-', linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.semilogx(f_s[:,0],Sw_s[:,i],                                                  'ks', label='$Measured$',linewidth=1, markeredgecolor='k',markersize=6)
                        ax1.plot(f[:,0],N400w_1par(f[:,0], ANw[flag_case,i]),                      'r-', label='$Fitted$ $ N400$ $( 1 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot(f[:,0],N400w_2par(f[:,0], ANw2[flag_case,i],BNw2[flag_case,i]),   'g-', label='$Fitted$ $ N400$ $( 2 para)$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot(f[:,0],Busch_Panofsky_w(f[:,0], ABw[flag_case,i], BBw[flag_case,i]),     'y-', label='$Fitted$ $ Busch Panofsky$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.plot(f[:,0],N400w_ref(f[:,0], 6.8),                                   'b--',label='$N400$ $ ref$',linewidth=2, markeredgecolor='k',markersize=8)
                        ax1.fill_between(f[:,0],0,1000,where= f[:,0]>f_ver,facecolor='grey', alpha=0.5)           
                        ax1.set_xlim(0.01, 2)
                        ax1.set_ylim(0, 1.2*np.max(Sw_s[4:-1,i]))
                        plt.xlabel(r'$f$ ($ Hz)$', fontsize=20)
                        plt.ylabel(r'$S_{fw}$ ($m^2 s^{-1})$', fontsize=20)
                        plt.rc('xtick', direction='in', color='k')
                        plt.rc('ytick', direction='in', color='k')
                        plt.legend(loc='upper right',ncol=2,fontsize=16)
                        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
                        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
                        ax1.tick_params(axis='both', labelsize=16)
                        plt.minorticks_on()
                        #plt.show()
                        plt.close(fig)
                        
                        path      = 'D:/DATA/Plots/Synn Spectra fitting/'
                        save_tite = 'Storm No.' + str(event+1) + '_time series and spectra Synn sensor No.' + str(i+1) + '_'\
                                     + str(date_begin.year) + '_' + str(date_begin.month) + '_' + str(date_begin.day)\
                                         + '_' + str(date_begin.hour) + '.png'
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.savefig(path + save_tite)   
    
        #%% summarize the results for each event           
        result={}
        result['date_start'] = time_s1.strftime("%Y-%m-%d %H:%M")
        result['date_end']   = time_e1.strftime("%Y-%m-%d %H:%M")
        result['U']          = U
        result['Dir']        = Dir
        result['Iu']         = Iu
        result['Iv']         = Iv   
        result['Iw']         = Iw   
        
        result['ANu']        = ANu   
        result['ANu_n']      = ANu_n  
        result['ANu2']       = ANu2 
        result['ANu_n2']     = ANu_n2 
        result['BNu2']       = BNu2 
        result['BNu_n2']     = BNu_n2 
        result['AHu']        = AHu 
        result['BHu']        = BHu 
        result['AHu_n']      = AHu_n  
        result['BHu_n']      = BHu_n
          
        result['ANv']        = ANv 
        result['ANv_n']      = ANv_n 
        result['ANv2']       = ANv2  
        result['ANv_n2']     = ANv_n2  
        result['BNv2']       = BNv2  
        result['BNv_n2']     = BNv_n2 
        
        result['ANw']        = ANw   
        result['ANw_n']      = ANw_n  
        result['ANw2']       = ANw2 
        result['ANw_n2']     = ANw_n2 
        result['BNw2']       = BNw2 
        result['BNw_n2']     = BNw_n2 
        result['ABw']        = ABw 
        result['BBw']        = BBw 
        result['ABw_n']      = ABw_n  
        result['BBw_n']      = BBw_n
        
        path      = 'D:/DATA/Results/Synn Spectra fitting/'
        
        save_tite = 'Storm No.' + str(event+1) + '_spectrum fitting Synn sensor No.' + str(3) + '_'\
                     + str(time_s1.year) + '_' + str(time_s1.month) + '_' + str(time_s1.day)\
                         + '_' + str(time_s1.hour) 
                         
        with open(path + save_tite + '_smooth_json.txt', "w") as outfile:
            json.dump(result, outfile, cls=NumpyEncoder)                            
