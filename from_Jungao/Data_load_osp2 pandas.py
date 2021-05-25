# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:30:05 2020

@author: junwan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:19:13 2020

Load the wind measurements at Bjørnafjorden
Extra cases with big file size
Need to sperate the files into smaller sizes 


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
import os
import gc
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

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

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



regex_time = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"     
regex_num  = r"[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?" 


cases = pd.read_excel('files_osp2.xls', sheet_name='files_osp2')
case_nr       = np.shape(cases)[0]
Ori   = pd.read_excel('files_osp2.xls', sheet_name='Orientation')

# get the starting time tag 
time_s = cases['First timestep'][0]
temp   = np.asarray(re.findall(regex_num,time_s), dtype='float64')    
year_s =     int(temp[0])
mon_s  =     int(abs(temp[1]))
day_s  =     int(abs(temp[2]))
hour_s =     int(temp[3])
minu_s =     int(temp[4])
sec_s  =     int(np.floor(temp[5]))
msec_s =     int(np.round((temp[5]-np.floor(temp[5]))*1000000))
time_s = mdates.date2num( datetime(year_s,mon_s,day_s,hour_s,minu_s,sec_s,msec_s))   

# get the ending time tag 
time_e = cases['Last timestep'][cases.index[-1]] 
temp = np.asarray(re.findall(regex_num,time_e), dtype='float64')    
year_e =     int(temp[0])
mon_e  =     int(abs(temp[1]))
day_e  =     int(abs(temp[2]))
hour_e =     int(temp[3])
minu_e =     int(temp[4])
sec_e  =     int(np.floor(temp[5]))
msec_e =     int(np.round((temp[5]-np.floor(temp[5]))*1000000))
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
    msec =     int(np.round((temp[5]-np.floor(temp[5]))*1000000))

    time_case[i,0] = mdates.date2num( datetime(year,mon,day,hour,minu,sec,msec) )    
time_case[i+1,0] = time_e


# total months
no_mon  =  (12-mon_s+1) + mon_e + (year_e-year_s-1)*12

# delta is one month time, to get the month with full data (start and end)
delta      = mdates.date2num(datetime(2,2,1,1,0))-mdates.date2num(datetime(2,1,1,1,0))
date_s_ref = mdates.num2date(time_case[0,0])
date_e_ref = mdates.num2date(time_case[-1,0]+delta)

#delta_ms is 100 milliseconds
delta_ms   = timedelta(milliseconds=100)

monthloop  = month_year_iter(date_s_ref.month,date_s_ref.year,date_e_ref.month,date_e_ref.year)

#%%
# i=44, 2019-08 some data loss at end of month
for i in range(0,monthloop.shape[0]-1):
    
    date_start = mdates.date2num( datetime(int(monthloop[i,0]),  int(monthloop[i,1])+1,  1,0,0)           )
    date_end   = mdates.date2num( datetime(int(monthloop[i+1,0]),int(monthloop[i+1,1])+1,1,0,0) - delta_ms)

    # to find the staring the ending date of the files 
    temp= time_case-date_start
    if i==0:
        ids =  np.where(abs(temp)==min(abs(temp)))
    else:
        ids = np.where(abs(temp)==min(abs(temp[temp<0])))
    
    temp= time_case-date_end
    if i==monthloop.shape[0]-2:
        ide = np.where(abs(temp)==min(abs(temp)))
    else:
        ide = np.where(abs(temp)==min(abs(temp[temp>0])))


    
    #collect all the data from relevant month    
    appended_data = []
    for j in range(ids[0][0],ide[0][0]):     
        file_name     = cases['Filename'][j]
        data = pd.read_csv('E:/DATA/osp2/' + file_name, skiprows=[0,2,3])
        appended_data.append(data)
    appended_data = pd.concat(appended_data,axis=0,ignore_index=True)
     

    #create the string of 'date_search '
    t_s = mdates.num2date(date_start)
    year = str(t_s.year)
    if t_s.month<10:
        mon = '0' + str(t_s.month)
    else:
        mon = str(t_s.month)

    # search for all the dataframe index with the current month
    date_search =   year +'-' +mon
    data_new  = appended_data[appended_data['TIMESTAMP'].str.contains(date_search)]

    del appended_data,data 
    gc.collect()
    
    A     = np.asarray([data_new['Sonic_A_U_Axis_Velocity'],data_new['Sonic_A_V_Axis_Velocity'],data_new['Sonic_A_W_Axis_Velocity']])
    B     = np.asarray([data_new['Sonic_B_U_Axis_Velocity'],data_new['Sonic_B_V_Axis_Velocity'],data_new['Sonic_B_W_Axis_Velocity']])
    C     = np.asarray([data_new['Sonic_C_U_Axis_Velocity'],data_new['Sonic_C_V_Axis_Velocity'],data_new['Sonic_C_W_Axis_Velocity']])
    temp_A= np.asarray(data_new['Sonic_A_Sonic_Temperature'])
    time  = mdates.date2num(pd.to_datetime(data_new['TIMESTAMP']))

    # find the places where its "NAN" and replace these with np.nan      
    A[A=='NAN']=np.nan
    A      = np.asarray(A, dtype='float64')
    B[B=='NAN']=np.nan
    B      = np.asarray(B, dtype='float64')
    C[C=='NAN']=np.nan
    C      = np.asarray(C, dtype='float64')
    
    
    # delete big variable to free memory
    del data_new 
    gc.collect()
    
    #time =    datetime.strptime(temp_t, '%Y-%m-%d %H:%M:%S.%f')
   
    # rotate the sensor original UVW to ground XYZ 
    # gound coordinate vector
    G_x = [1,0,0]
    G_y = [0,1,0]
    G_z = [0,0,1]
    
    # SENSOR ORIENTATION IS GIVEN FROM THE DATA SERVER BY KVT
    # sensor coordinate vecotr in global XYZ using the angle U vecotr points
    S_ang_A = np.radians(Ori['A'][0])    
    S_u     = [np.sin(S_ang_A),        np.cos(S_ang_A),        0]
    S_v     = [np.sin(S_ang_A-np.pi/2),np.cos(S_ang_A-np.pi/2),0]
    S_w     = [0,0,1]
    # transformation matrix from UVW to XYZ  (V_S = T*V_G)     
    TT_A    = TF.T_xyzXYZ(S_u, S_v, S_w, G_x, G_y, G_z, dim='3x3')    
    A_new   = np.transpose(np.matmul(TT_A.T,A) )
    
    S_ang_B = np.radians(Ori['B'][0])    
    S_u     = [np.sin(S_ang_B),        np.cos(S_ang_B),        0]
    S_v     = [np.sin(S_ang_B-np.pi/2),np.cos(S_ang_B-np.pi/2),0]
    S_w     = [0,0,1]
    # transformation matrix from UVW to XYZ  (V_S = T*V_G)     
    TT_B    = TF.T_xyzXYZ(S_u, S_v, S_w, G_x, G_y, G_z, dim='3x3')    
    B_new   = np.transpose(np.matmul(TT_B.T,B ))
    
    S_ang_C = np.radians(Ori['C'][0])    
    S_u     = [np.sin(S_ang_C),        np.cos(S_ang_C),        0]
    S_v     = [np.sin(S_ang_C-np.pi/2),np.cos(S_ang_C-np.pi/2),0]
    S_w     = [0,0,1]
    # transformation matrix from UVW to XYZ  (V_S = T*V_G)   
    TT_C    = TF.T_xyzXYZ(S_u, S_v, S_w, G_x, G_y, G_z, dim='3x3')    
    C_new   = np.transpose(np.matmul(TT_C.T,C) )     

    # create the full time series

    time_new = pd.date_range(mdates.num2date(date_start),mdates.num2date(date_end),freq = '100L')        
    time_new = pd.Series(mdates.date2num(time_new))
    
    # find the index of the recorded time tag in the full time series
    idx=time_new.searchsorted (time,side='left') 
    
    # assign all the pre-processed date to a dataframe                    
    data  = pd.DataFrame(columns=['Time', 'A_temp','A_X', 'A_Y','A_Z', 'B_X', 'B_Y','B_Z','C_X', 'C_Y','C_Z'])    
    data  = data.assign(Time=time_new)
    data.loc[idx,'A_temp']  =  temp_A
    data.loc[idx,['A_X','A_Y','A_Z'] ] =  A_new
    data.loc[idx,['B_X','B_Y','B_Z'] ] =  B_new    
    data.loc[idx,['C_X','C_Y','C_Z'] ] =  C_new     

    data.to_pickle('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '.pkl')

    print(str(year) +'_' + str(mon) +' is finished')    
    

    np.savez_compressed('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_A.npz', A_new)
    np.savez_compressed('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_B.npz', B_new)
    np.savez_compressed('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_C.npz', C_new)
    np.savez_compressed('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_time.npz', time)
    np.savez_compressed('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_temp_A.npz', temp_A)
    del A,B,C,A_new,B_new,C_new,time,temp_A,data 
    gc.collect()      
# =============================================================================
#     result={}
#     result['time']     = time  
#     result['temp_A']   = temp_A  
#     result['A']        = A_new  
#     result['B']        = B_new 
#     result['C']        = C_new
#     del A,B,C,A_new,B_new,C_new,time,temp_A  
#     gc.collect()    
#     with open('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_A.json', "w") as outfile:
#         json.dump(result['A'], outfile, cls=NumpyEncoder)    
#         
#     with open('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_B.json', "w") as outfile:
#         json.dump(result['B'], outfile, cls=NumpyEncoder)    
#     
#     with open('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_C.json', "w") as outfile:
#         json.dump(result['C'], outfile, cls=NumpyEncoder)    
#     
#     with open('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_time.json', "w") as outfile:
#         json.dump(result['time'], outfile, cls=NumpyEncoder)    
# 
#     with open('E:/DATA/osp2/' +'Osp2_'+ str(year) +'_' + str(mon)+ '_temp_A.json', "w") as outfile:
#         json.dump(result['temp_A'], outfile, cls=NumpyEncoder)    
# 
#     del result   
#     gc.collect()
# =============================================================================


# this is to find the exact date of starting and ending timestamp
# =============================================================================
#     # find the index of the staring and ending time stamp
#     idxs  = appended_data[appended_data['TIMESTAMP'].str.contains(date_start_s)].index[0]-1
#     idxe  = appended_data[appended_data['TIMESTAMP'].str.contains(date_end_s)].  index[0]
# =============================================================================
# =============================================================================
#     # for 2019-08, it stopped at 2019-08-29 03:00:03.1
#     idxe  = appended_data[appended_data['TIMESTAMP'].str.contains('2019-08-29 03:00:03.1')].  index[0] 
# =============================================================================
# =============================================================================
#     #create the string of 'date_start '
#     t_s = mdates.num2date(date_start)
#     year = str(t_s.year)
#     if t_s.month<10:
#         mon = '0' + str(t_s.month)
#     else:
#         mon = str(t_s.month)
#     if t_s.day<10:
#         day = '0' + str(t_s.day)
#     else:
#         day = str(t_s.day)
#     if t_s.hour<10:
#         hour = '0' + str(t_s.hour)
#     else:
#         hour = str(t_s.hour)
#     if t_s.minute<10:
#         minu = '0' + str(t_s.minute)
#     else:
#         minu = str(t_s.minute)
#     if t_s.second<10:
#         sec = '0' + str(t_s.second)
#     else:
#         sec = str(t_s.second)
#     msec = str(int(t_s.microsecond/100000)+1) # move to the next one to have the unique tag    
#     date_start_s=   year +'-' +mon +'-' +day + ' ' + hour + ':' + minu + ':' + sec + '.' + msec
# 
#     #create the string of 'date_start '        
#     t_e = mdates.num2date(date_end)
#     year = str(t_e.year)
#     if t_e.month<10:
#         mon = '0' + str(t_e.month)
#     else:
#         mon = str(t_e.month)
#     if t_e.day<10:
#         day = '0' + str(t_e.day)
#     else:
#         day = str(t_e.day)
#     if t_e.hour<10:
#         hour = '0' + str(t_e.hour)
#     else:
#         hour = str(t_e.hour)
#     if t_e.minute<10:
#         minu = '0' + str(t_e.minute)
#     else:
#         minu = str(t_e.minute)
#     if t_e.second<10:
#         sec = '0' + str(t_e.second)
#     else:
#         sec = str(t_e.second)
#     msec = str(int(t_e.microsecond/100000))   
#     date_end_s=   year +'-' +mon +'-' +day + ' ' + hour + ':' + minu + ':' + sec + '.' + msec
# =============================================================================
# =============================================================================
#     A     = np.asarray([appended_data['Sonic_A_U_Axis_Velocity'][idxs:idxe],appended_data['Sonic_A_V_Axis_Velocity'][idxs:idxe],appended_data['Sonic_A_W_Axis_Velocity'][idxs:idxe]])
#     B     = np.asarray([appended_data['Sonic_B_U_Axis_Velocity'][idxs:idxe],appended_data['Sonic_B_V_Axis_Velocity'][idxs:idxe],appended_data['Sonic_B_W_Axis_Velocity'][idxs:idxe]])
#     C     = np.asarray([appended_data['Sonic_C_U_Axis_Velocity'][idxs:idxe],appended_data['Sonic_C_V_Axis_Velocity'][idxs:idxe],appended_data['Sonic_C_W_Axis_Velocity'][idxs:idxe]])
#     temp_A= np.asarray(appended_data['Sonic_A_Sonic_Temperature'][idxs:idxe])
#     time_s= np.asarray(appended_data['TIMESTAMP'][idxs:idxe])
# =============================================================================
