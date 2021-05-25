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

data = pd.read_excel('SLÅTTERØY FYR.xlsx',sheet_name='48330')


for i in range(0,cases['Time_storm'].size):
    
    #i    = 28
    time = datetime.strptime(cases['Time_storm'][i], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
    Time_Ref           = mdates.num2date(mdates.date2num(pd.to_datetime(data['Date'])))
    
 #%%   
    if os.path.isfile('E:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_raw_data.pkl'): 

        with open('E:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_1h_mean' +'.pkl', 'rb') as f:    
            Osp1_1h=pickle.load( f)
        with open('E:/DATA/osp2/' +'Osp2_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp2_1h_mean' +'.pkl', 'rb') as f:    
            Osp2_1h=pickle.load( f)
        with open('E:/DATA/svar/' +'Svar_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Svar_1h_mean' +'.pkl', 'rb') as f:    
            Svar_1h=pickle.load( f)        
        with open('E:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Synn_1h_mean' +'.pkl', 'rb') as f:    
            Synn_1h=pickle.load( f)
        
        with open('E:/DATA/osp1/' +'Osp1_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp1_1h_ins' +'.pkl', 'rb') as f:    
            Osp1_ins=pickle.load( f)
        with open('E:/DATA/osp2/' +'Osp2_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Osp2_1h_ins' +'.pkl', 'rb') as f:    
            Osp2_ins=pickle.load( f)
        with open('E:/DATA/svar/' +'Svar_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Svar_1h_ins' +'.pkl', 'rb') as f:    
            Svar_ins=pickle.load( f)        
        with open('E:/DATA/synn/' +'Synn_'+ str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_' + 'Synn_1h_ins' +'.pkl', 'rb') as f:    
            Synn_ins=pickle.load( f)            
        
       
        # plot the wind direction, mean wind speed and along wind turbulence intensity
        plt.close("all")       
        fig     = plt.figure(figsize=(20, 12))
        ax1     = plt.subplot(311)
        # format the ticks
        locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        #ax1.plot(Osp1_1h['time'], Osp1_1h['A_dir'], 'k-',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
        #movingdir_Osp1   =  np.asarray(pd.DataFrame(Osp1_1h['A_dir']).rolling(Nwin,min_periods=1).mean())
        #ax1.plot(Osp1_1h['time'], movingdir_Osp1, 'r-',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
        Osp1_ins['A_dir'][Osp1_ins['A_dir']<50] = Osp1_ins['A_dir'][Osp1_ins['A_dir']<50]+360
        Osp1_1h['A_Dir'][Osp1_1h['A_Dir']<50] = Osp1_1h['A_Dir'][Osp1_1h['A_Dir']<50]+360
        ax1.plot(Osp1_ins['Time'], Osp1_ins['A_dir'], 'ko',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=1)   
        ax1.plot(Osp1_1h['Time'], Osp1_1h['A_Dir'], 'r-',label='$Dir_{wi}$ $(Osp1\_A)$',markeredgecolor='k',markersize=8)   
        datemin = np.datetime64(Osp1_1h['Time'].iloc[0].replace(tzinfo=None), 'h')
        datemax = np.datetime64(Osp1_1h['Time'].iloc[-1].replace(tzinfo=None), 'm') + np.timedelta64(1, 'h')
        ax1.set_xlim(datemin, datemax)
        plt.ylabel(r'$Dir_{wi}$ $( ^o)$', fontsize=20)
        fig.suptitle(str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_time_history', fontsize=25)
        plt.rc('xtick', direction='in', color='k')
        plt.rc('ytick', direction='in', color='k')
        plt.legend(loc='best',ncol=1,fontsize=16)
        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
        ax1.tick_params(axis='both', labelsize=16)
        plt.minorticks_on()
        plt.show()
        
        
        ax2     = plt.subplot(312)
        # format the ticks
        locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax2.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_formatter(formatter)
        #ax2.plot(Osp1_1h['time'], Osp1_1h['A_uvw'][:,0], 'b-',label='$\overline{U} + u $ $(Osp1\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
        #movingU_Osp1   =  np.asarray(pd.DataFrame(Osp1_1h['A_uvw'][:,0]).rolling(Nwin,min_periods=1).mean())
        #ax2.plot(Osp1_1h['time'], movingU_Osp1, 'r-',label='$\overline{U} $ $(Osp1\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)
        ax2.plot(Osp1_ins['Time'], Osp1_ins['A_u'], 'ko',label='$\overline{U}+u $ $(Osp1\_A)$',markeredgecolor='k',markersize=1)    
        ax2.plot(Osp1_1h['Time'], Osp1_1h['A_U'], 'r-',label='$\overline{U} $ $(Osp1\_A)$',markeredgecolor='k',markersize=8)    
        datemin = np.datetime64(Osp1_1h['Time'].iloc[0].replace(tzinfo=None), 'h')
        datemax = np.datetime64(Osp1_1h['Time'].iloc[-1].replace(tzinfo=None), 'm') + np.timedelta64(1, 'h')
        ax2.set_xlim(datemin, datemax)
        ax2.set_ylim(0, 40)
       
        plt.ylabel(r'$\overline{U} + u$ (m s$^{-1})$', fontsize=20)
        ax2.set_title('', fontsize=25)
        plt.rc('xtick', direction='in', color='k')
        plt.rc('ytick', direction='in', color='k')
        plt.legend(loc='best',ncol=1,fontsize=16)
        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
        ax2.tick_params(axis='both', labelsize=16)
        plt.minorticks_on()
        plt.show()
        
        
        ax3     = plt.subplot(313)
        # format the ticks
        locator = mdates.AutoDateLocator(minticks=7, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax3.xaxis.set_major_locator(locator)
        ax3.xaxis.set_major_formatter(formatter)
        ax3.plot(Osp1_1h['Time'], Osp1_1h['A_U'], 'ro-',label='$\overline{U} $ $(Osp1\_A)$',markeredgecolor='r',markersize=8,alpha=0.5)    
        ax3.plot(Osp2_1h['Time'], Osp2_1h['A_U'], 'ko-',label='$\overline{U} $ $(Osp2\_A)$',markeredgecolor='k',markersize=8,alpha=0.5)    
        ax3.plot(Svar_1h['Time'], Svar_1h['A_U'], 'go-',label='$\overline{U} $ $(Svar\_A)$',markeredgecolor='g',markersize=8,alpha=0.5)    
        ax3.plot(Synn_1h['Time'], Synn_1h['A_U'], 'bo-',label='$\overline{U} $ $(Synn\_A)$',markeredgecolor='b',markersize=8,alpha=0.5)    
        ax3.plot(Time_Ref, data['Ws'],  'co-',label=r'$\overline{U} $ $(Ref)$',markeredgecolor='c',markersize=3,alpha=0.5)
    
        datemin = np.datetime64(Osp1_1h['Time'].iloc[0].replace(tzinfo=None), 'h')
        datemax = np.datetime64(Osp1_1h['Time'].iloc[-1].replace(tzinfo=None), 'm') + np.timedelta64(1, 'h')
        ax3.set_xlim(datemin, datemax)
        plt.ylabel(r'$\overline{U}$ (m s$^{-1})$', fontsize=20)
        ax3.set_title('', fontsize=25)
        plt.rc('xtick', direction='in', color='k')
        plt.rc('ytick', direction='in', color='k')
        plt.legend(loc='best',ncol=1,fontsize=16)
        g1 = plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.35)
        g2 = plt.grid(b=True, which='minor', color='k', linestyle='-', linewidth=0.1)
        ax3.tick_params(axis='both', labelsize=16)
        plt.minorticks_on()
        plt.show()    
        fig.tight_layout(rect=[0, 0, 1, 1])
        save_tite = 'new' + str(time.year) +'_' + str(time.month)+'_' + str(time.day)+ '_storm_time_history new.png'
        fig.savefig(save_tite)     