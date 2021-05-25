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

regex_time = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"     
regex_num  = r"[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?" 


cases = pd.read_excel('files_osp1.xls', sheet_name='files_osp1')
case_nr       = np.shape(cases)[0]

event = 1
file_name     = cases['Filename'][event]
line_num_ref  =  file_len('D:/DATA/osp1/' + file_name)

event = 188
file_name     = cases['Filename'][event]
data_size     = os.path.getsize('D:/DATA/osp1/' + file_name)
line_num      =  file_len('D:/DATA/osp1/' + file_name)

num_files     = int(np.ceil(line_num/line_num_ref))

with open('D:/DATA/osp1/' + file_name,'r') as fid:
    for block in range(0,num_files):        

        if block<num_files-1:            
            start   = 3+line_num_ref*block
            end     = 3+line_num_ref*(block+1)
        else:
            start   = 3+line_num_ref*block
            end     = line_num-1
        
        strat   = 3
        end     = 10
        
        
        num     = end-start
        A       = [num,3]
        B       = [num,3]
        C       = [num,3]
        temp_A  = [num,1]
        time    = [num,1]  
          
        for j,dataline in enumerate(fid):
            if block<num_files-1 and end>=j>start:      # skip the header      
                line = dataline.replace("NAN",'-1000')
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
            else:
                break
                 
        
        result={}
        result['time']     = time  
        result['temp_A']   = temp_A  
        result['A']        = A  
        result['B']        = B 
        result['C']        = C
    
        with open('D:/DATA/osp1/' + file_name + '_'+ block +'_json.txt', "w") as outfile:
            json.dump(result, outfile, cls=NumpyEncoder)                           
                    
                
                

