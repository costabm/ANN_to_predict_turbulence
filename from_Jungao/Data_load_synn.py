# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:19:13 2020

Load the wind measurements at Bjørnafjorden


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

cases = pd.read_excel('files_synn.xls', sheet_name='files_synn')

case_nr       = np.shape(cases)[0]

for event in range(233,case_nr):

    file_name     = cases['Filename'][event]
    
    regex_time = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.][\d]" 
    regex_time = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}" 
    
    regex_num  = r"[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?" 
    
    with open('D:/DATA/synn/' + file_name,'r') as fid:
        raw     = fid.readlines()
        num     = len(raw)-4
        A       = np.zeros((num,3))
        B       = np.zeros((num,3))
        C       = np.zeros((num,3))
        temp_A  = np.zeros((num,1))
        time    = np.zeros((num,1))
        
        for i in range (0,num):
            
            line = raw[i+4].replace("NAN",'-1000')
            
            date = np.asarray(re.findall(regex_time,line))
            temp = np.asarray(re.findall(regex_num,line), dtype='float64')         
        
            year =     int(temp[0])
            mon  =     int(abs(temp[1]))
            day  =     int(abs(temp[2]))
            hour =     int(temp[3])
            minu =     int(temp[4])
            sec  =     int(np.floor(temp[5]))
            msec =     int(np.round((temp[5]-np.floor(temp[5]))*1000))
        
            time[i,0] = mdates.date2num( datetime(year,mon,day,hour,minu,sec,msec) )
            
            A[i,0] =   temp[7]
            A[i,1] =   temp[8]
            A[i,2] =   temp[9]
            temp_A[i] =   temp[10]
            
            B[i,0] =   temp[11]
            B[i,1] =   temp[12]
            B[i,2] =   temp[13]   
            
            C[i,0] =   temp[14]
            C[i,1] =   temp[15]
            C[i,2] =   temp[16]  
            
            
        result={}
        result['time']     = time  
        result['temp_A']   = temp_A  
        result['A']        = A  
        result['B']        = B 
        result['C']        = C  

        with open('D:/DATA/synn/' + file_name + '_json.txt', "w") as outfile:
            json.dump(result, outfile, cls=NumpyEncoder)           
    
