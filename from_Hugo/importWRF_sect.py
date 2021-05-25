# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:20:30 2020

@author: hugmor
"""
import numpy as np
import sys, os, netCDF4
from scipy.io import savemat
import csv
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import pickle, datetime

#%% Parameters
inpoint = [5.354315,60.136282]                            # Synnøytangen
inpoint = [5.372,60.1274]                                 # North Bjørnafjorden
inpoint = [5.3698,60.0823]                                # South Bjørnafjorden
inpoint = [5.5264,59.8925]                                # Langenuen
inpoint = [5.371,60.105]                                # mid Bjørnafjorden

frompoint = [5.3698,60.0823]                              # South Bjørnafjorden
topoint = [5.372,60.1274]                                 # North Bjørnafjorden
#file_name = 'synnoyWRF'

from_year = 2008
file_name = 'BjørnafjordenWRF'
select_ptorsect = 'sect'

#%% functions --------------------------------------------------------------------------------------------------------------------

def  nearest_grid_point(in_grid_x,in_grid_y,inpoint):
# %
# % Method for finding nearest grid coordinates to a point in long lat from a long lat grid
# %
# % input 
# % in_grid_x - n x m matrix with lon values (lon_rho)
# % in_grid_y - n x m matrix with lat values (lat_rho)
# % in_point - 2 x 1 array with lon values at (1,1) and lat value at (2,1)
# %
# % How to get in_point at the right format for the function
# % in_point=E';
# % in_point(:,2)=N';
# % in_point has then the size (nx2) where n is the number of points one want to find the grid coordinates of
# % inpoint=in_point(1,:)' has the size 2x1 and is the first point one want to find the grid coordinates of
# % 
# % output
# % row - nearest x-grid point to in_point
# % col - nearest y-grid point to in_point
# % shortest - the distance in meter from in_point to the nearest in_grid y_grid point 
# %
# % in_grid=lon_rho;
# % in_grid(:,:,2)=lat_rho;
# % 
# %
# %
# %compute Euclidean distances:
    distan=np.sqrt((in_grid_x-inpoint[0])**2+(in_grid_y-inpoint[1])**2)
# %
# % Find the shortest distance and the index of this element
    shortest=np.min(np.min(distan));
    row, col = np.where(distan==shortest);
# %
    print('The nearest grid point is:{} '.format(row[0]), 'x {}'.format(col[0]),  ' which is {}'.format(shortest*1000),
          'm from the given location')
#%
    return row, col, shortest



#%% Load wind --------------------------------------------------------------------------------------------------------------------
 
#folder_loc = 'O:\Landsdekkende\Ferjefri E39\Fjordkryssinger\\12_Bjørnafjorden\Fag\\01 - Miljødata\Fag\Vind\Kjeller vindteknikk\WRF-2000-2017'
folder_loc = r'O:\Utbygging\Fagress\BFA40 Konstruksjoner\10 Faggrupper\01 Metocean\Data\Vinddata\3d\Vind'

os.chdir(folder_loc)
files = os.listdir(folder_loc)
allfiles = []
timefiles = []

#os.remove('southWRF')
# filter our all the nc files
ncfiles = []
for i, file in enumerate(files):
    if (file[-4:] == ".nc4"):
        ncfiles.append(file)
print(ncfiles)

fp = ncfiles[0]
nc = netCDF4.Dataset(folder_loc)
latitude=nc['latitude'][:].data
longitude=nc['longitude'][:].data

ncfiles=ncfiles[-1:]

total_length =  nc['time'][:].data.size
date_ = pd.to_datetime(
    [(datetime.datetime.min + 
      datetime.timedelta(hours=np.min(nc['time'][kk].data))) for kk in range(total_length)])

left_cut=np.where(date_.year>=from_year)[0][0]

# span of latitudes
print('min latitude: ', np.min(nc['latitude'][:].data), 'max latitude: ', np.max(nc['latitude'][:].data))

# span of time
print('Start date: ', str(date_[left_cut]), 'End date: ', str(date_[-1]))
print('Start date: ', np.min(nc['time'][0].data), 'End date: ', np.max(nc['time'][-1].data))
print('Start date: ', (datetime.datetime.min + 
      datetime.timedelta(hours=np.min(nc['time'][0].data))).strftime("%d/%m/%Y %H:%M:%S"), 'End date: ',
       (datetime.datetime.min + datetime.timedelta(hours=np.min(nc['time'][-1].data))).strftime("%d/%m/%Y %H:%M:%S"))

#%%

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minutes >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+datetime.timedelta(hours=t.minute//30))

def import_data_pt(inpoint,left_cut,file_name):
    row, col, shortest = nearest_grid_point(longitude,latitude,inpoint);
    # extract data
    #[row,col] = inpoint
    print([row[0],col[0]])
    print([nc['longitude'][row[0],col[0]].data,nc['latitude'][row[0],col[0]].data])
    ws = nc['ws'][row[0],col[0],left_cut:].data
    print('ws: {}'.format(ws.shape))
    #wd = (270-nc['wd'][row[0],col[0],left_cut:].data)%360
    wd = (nc['wd'][row[0],col[0],left_cut:].data)%360
    print('wd: {}'.format(wd.shape))
        
    t = nc['time'][left_cut:].data
    jt = nc['jdate'][left_cut:].data
    with open(file_name+'pt',"wb") as f:
        pickle.dump([ws,wd,t,jt],f)
        
    return ws, wd, t, jt

def import_data(frompoint, topoint,left_cut,file_name):
    row_f, col_f, shortest_f = nearest_grid_point(longitude,latitude,frompoint)
    row_t, col_t, shortest_t = nearest_grid_point(longitude,latitude,topoint)
    Dx = row_t-row_f
    Dy = col_t-col_f
    step_xy = np.maximum(np.abs(Dx),np.abs(Dy))
    if step_xy==np.abs(Dx):
        coords=[[x,int((Dy/Dx)*x+col_f-(Dy/Dx)*row_f)] for 
                x in np.arange(row_f,row_t+1*np.sign(Dx),1*np.sign(Dx))]
    if step_xy==np.abs(Dy):
        coords=[[int((Dx/Dy)*y+row_f-(Dx/Dy)*col_f),y] for
                y in np.arange(col_f,col_t+1*np.sign(Dy),1*np.sign(Dy))]
    u, vdir = [], []
    t,jt = [], []
    # extract data
    for kk in range(step_xy[0]+1):
        print(kk)
        ws,wd=[],[]
        [row,col] = coords[kk]
        print([row,col])
        print([nc['longitude'][row,col].data,nc['latitude'][row,col].data])
        ws = nc['ws'][row,col,left_cut:].data
        print('ws: {} {:.2f}'.format(ws.shape,np.max(ws)))
        #wd = (270-nc['wd'][row,col,left_cut:].data)%360
        wd = (nc['wd'][row,col,left_cut:].data)%360
        print('wd: {}'.format(wd.shape))
        u.append(ws)
        #vdir = np.concatenate((vdir,wd))
        vdir.append(wd)
    t = nc['time'][left_cut:].data
    jt = nc['jdate'][left_cut:].data
    # t = np.concatenate((t,date_yr))
    # jt = np.concatenate((jt,jdate_yr))
    
    # plots map and section location
    plt.figure(figsize=(15,10))
    plt.pcolor(nc['landmask'][:].data.T)
    plt.plot([row_f+1.5,row_t+1.5],[col_f+1.5,col_t+1.5],'r-d')
    plt.grid()
    
    # determine date from variable "time" - hours since 0001-01-01
    date_ = pd.to_datetime(
    [(datetime.datetime.min + 
      datetime.timedelta(hours=np.min(t[kk]))) for kk in range(len(t))])
    
    # determine date from "jdate"
    jdate_=[]
    for datenum in jt:
        jdate_.append(hour_rounder((datetime.datetime.fromordinal(int(datenum)) + datetime.timedelta(days=datenum%1) - 
               datetime.timedelta(days = 366))))
    
    with open(file_name,"wb") as f:
        pickle.dump([np.array(u),np.array(vdir),date_,jdate_],f)
        
    return u, vdir, t, jt



# if file_name in files:
#     with open(file_name, 'rb') as f:
#         xs,zxs,tx,jtx = pickle.load(f)
# else:
#     xs,zxs,tx,jtx = import_data(frompoint, topoint,left_cut,file_name)

if select_ptorsect=='sect':
    xs,zxs,tx,jtx = import_data(frompoint, topoint,left_cut,file_name)    
else:
    ws,wd,tx,jtx = import_data_pt(inpoint,left_cut,file_name)  


#tx = tx.astype(int)
#jtx,jty = jtx.astype(int), jty.astype(int)



txs=[]
for datenum in jtx:
    txs.append(hour_rounder((datetime.datetime.fromordinal(int(datenum)) + datetime.timedelta(days=datenum%1) - 
               datetime.timedelta(days = 366))))
txs=np.array(txs)    


#%% save to csv and mat files
#sys.exit("Program interrupted")
#matfile

if select_ptorsect=='sect':
    savemat(file_name+'.mat', {'xs':xs,'zxs':zxs,'tx':tx,'jtx':jtx})
else:
    savemat(file_name+'.mat', {'ws':ws,'wd':wd,'tx':tx,'jtx':jtx})
# X = np.vstack((['Speed','Dir','Date'],np.array([xs,zxs,jtx]).T))
# np.savetxt(file_name+'.csv',X, delimiter=",")

#%%

if select_ptorsect=='sect':
    lines = np.array([xs,zxs,jtx,tx]).T
    
    header = ['Speed','Dir','JDate','Date']
    
    with open(file_name+'.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header) # write the header
        # write the actual content line by line
        for l in lines:
            writer.writerow(l)
        # or we can write in a whole
        # writer.writerows(lines)
    
#%
else:
    date_ = pd.to_datetime(
        [(datetime.datetime.min + 
          datetime.timedelta(hours=np.min(tx[kk]))) for kk in range(len(tx))])
    
    
    plt.figure(figsize=(15,10))
    #plt.plot_date(date_,xs[5])
    plt.plot_date(date_,ws)
    
    plt.figure(figsize=(15,10))
    plt.plot_date(date_,wd)
    
    #%
    fig,big_axes = plt.subplots(2,1,figsize = (10, 10))
    for big_ax in big_axes:
            big_ax.tick_params(labelcolor=(1.,1.,1.,0.0),top=False,bottom=False,
                               left=False,right=False,axis='both', which='both')
            big_ax._frameon=False
    
    ax1 = fig.add_subplot(2,1,1, projection='windrose')
    ax1.bar(wd,ws, normed=True, nsector=12, 
                    bins = np.arange(0,30,5), 
           cmap = plt.cm.Blues, edgecolor = 'black', linewidth = .1)
    ax1.set_legend(bbox_to_anchor=(1.2,-0.1), fontsize=16)
    ax1.set_facecolor('whitesmoke')
    table = ax1._info['table']
    bins = ax1._info['bins']
    freq_table = pd.DataFrame(table).T
    
    freq_table.index=['{:d}'.format(int(col+15)%360) for col in ax1._info['dir']]
    freq_table.columns=[ '[{:s}:{:s}]'.format(str(row),str(bins[cc+1])) 
                      for cc,row in enumerate(bins[:-1])]
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.axis('tight')
    ax2.axis('off')
    pd.plotting.table(ax2, np.round(freq_table, 2), loc='center')
    plt.title(file_name)