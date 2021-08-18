"""
This study uses Machine Learning to predict turbulence at a new location with a given topography, where the learning data
consists only of turbulence data and topography data at other nearby locations.
Created: June, 2021
Contact: bercos@vegvesen.no
"""

import json
import copy
import datetime
import os
import time
from collections import Counter
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from scipy import stats
from scipy.optimize import curve_fit
import bisect
import torch
import pandas as pd
from torch import Tensor
from torch.nn import Linear, MSELoss, L1Loss, CrossEntropyLoss, SmoothL1Loss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
from read_0p1sec_data import create_processed_data_files, compile_all_processed_data_into_1_file
from find_storms import create_storm_data_files, compile_storm_data_files, find_storm_timestamps, organized_dataframes_of_storms, merge_two_all_stats_files
import optuna
from elevation_profile_generator import elevation_profile_generator, plot_elevation_profile, get_point2_from_point1_dir_and_dist
from sklearn.metrics import r2_score


print('Osp2_A is now being attempted, instead of Osp1_A')
nice_str_dict = {'osp1_A': 'Ospøya 1', 'osp2_A': 'Ospøya 2', 'osp2_B': 'Ospøya 2', 'synn_A': 'Synnøytangen', 'svar_A': 'Svarvhelleholmen', 'land_A': 'Landrøypynten', 'neso_A': 'Nesøya'}

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def convert_angle_to_0_2pi_interval(angle, input_and_output_in_degrees=True):
    if input_and_output_in_degrees:
        return angle % 360
    else:
        return angle % (2*np.pi)


def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def density_scatter(x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots(figsize=(8,6), dpi=400, subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
    data , x_e, y_e = np.histogram2d(np.sin(x)*y, np.cos(x)*y, bins = bins, density = True)
    # z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    x_i = 0.5 * (x_e[1:] + x_e[:-1])
    y_i = 0.5 * (y_e[1:] + y_e[:-1])
    z = interpn((x_i, y_i), data, np.vstack([np.sin(x)*y, np.cos(x)*y]).T, method="splinef2d", bounds_error=False)
    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    ax.scatter( x, y, c=z, s=1, alpha=0.3, **kwargs )
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    return ax, np.min(z), np.max(z)


def example_of_elevation_profile_at_given_point_dir_dist(point_1=[-34625., 6700051.], direction_deg=160, step_distance=False, total_distance=False,
                                                   list_of_distances=[i*(5.+5.*i) for i in range(45)], plot=True):
    point_2 = get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=direction_deg, distance=list_of_distances[-1] if list_of_distances else total_distance)
    dists, heights = elevation_profile_generator(point_1=point_1, point_2=point_2, step_distance=step_distance, list_of_distances=list_of_distances)
    if plot:
        plot_elevation_profile(point_1=point_1, point_2=point_2, step_distance=step_distance, list_of_distances=list_of_distances)
    return dists, heights

# example_of_elevation_profile_at_given_point_dir_dist(point_1=[-34625., 6700051.], direction_deg=160, step_distance=False, total_distance=False,
#                                                    list_of_distances=[i*(5.+5.*i) for i in range(45)], plot=True)


def slopes_in_deg_from_dists_and_heights(dists, heights):
    delta_dists = dists[1:] - dists[:-1]
    delta_heights = heights[1:] - heights[:-1]
    return np.rad2deg(np.arctan(delta_heights/delta_dists))


def plot_topography_per_anem(list_of_degs = list(range(360)), list_of_distances=[i*(5.+5.*i) for i in range(45)], plot_topography=True, plot_slopes=False):
    from orography import synn_EN_33, svar_EN_33, osp1_EN_33, osp2_EN_33, land_EN_33, neso_EN_33
    anem_EN_33 = {'synn':synn_EN_33, 'svar':svar_EN_33, 'osp1':osp1_EN_33, 'osp2':osp2_EN_33, 'land':land_EN_33, 'neso':neso_EN_33}
    for anem, anem_coor in anem_EN_33.items():
        dists_all_dirs = []
        heights_all_dirs = []
        slopes_all_dirs = []
        degs = []
        for d in list_of_degs:
            dists, heights = example_of_elevation_profile_at_given_point_dir_dist(point_1=anem_coor, direction_deg=d, step_distance=False, total_distance=False, list_of_distances=list_of_distances, plot=False)
            dists_all_dirs.append(dists)
            heights_all_dirs.append(heights)
            if plot_slopes:
               slopes_all_dirs.append(slopes_in_deg_from_dists_and_heights(dists, heights))
            degs.append(d)
        degs = np.array(degs)
        dists_all_dirs = np.array(dists_all_dirs)
        heights_all_dirs = np.array(heights_all_dirs)
        slopes_all_dirs = np.array(slopes_all_dirs)
        if plot_topography:
            cmap = copy.copy(plt.get_cmap('magma_r'))
            heights_all_dirs = np.ma.masked_where(heights_all_dirs == 0, heights_all_dirs)  # set mask where height is 0, to be converted to another color
            fig, (ax, cax) = plt.subplots(nrows=2, figsize=(5.5,2.3+0.5), dpi=400, gridspec_kw={"height_ratios": [1, 0.05]})
            im = ax.pcolormesh(degs, dists_all_dirs[0], heights_all_dirs.T, cmap=cmap, shading='auto', vmin = 0., vmax = 800.)
            ax.set_title(nice_str_dict[anem+'_A']+': '+'Upstream topography;')
            ax.patch.set_color('skyblue')
            ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
            ax.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
            ax.set_yticks([0,2000,4000,6000,8000,10000])
            ax.set_yticklabels([0,2,4,6,8,10])
            ax.set_xlabel('Wind from direction [\N{DEGREE SIGN}]')
            ax.set_ylabel('Upstream distance [km]')
            # cax.set_xlabel('test')
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label('Height above sea level [m]')
            plt.tight_layout(pad=0.05)
            plt.savefig(os.path.join(os.getcwd(), 'plots', f'Topography_per_anem-{anem}.png'))
            plt.show()
        if plot_slopes:
            cmap = copy.copy(plt.get_cmap('seismic'))
            fig, (ax, cax) = plt.subplots(nrows=2, figsize=(5.5, 2.3 + 0.5), dpi=400, gridspec_kw={"height_ratios": [1, 0.05]})
            im = ax.pcolormesh(degs, 6[0], slopes_all_dirs.T, cmap=cmap, shading='auto') #, vmin = -30., vmax = 30.)
            ax.set_title(nice_str_dict[anem + '_A'] + ': ' + 'Upstream slopes;')
            ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
            ax.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
            ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
            ax.set_yticklabels([0, 2, 4, 6, 8, 10])
            ax.set_xlabel('Wind from direction [\N{DEGREE SIGN}]')
            ax.set_ylabel('Upstream distance [km]')
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label('Slope [\N{DEGREE SIGN}]')
            plt.tight_layout(pad=0.05)
            plt.savefig(os.path.join(os.getcwd(), 'plots', f'Slopes_per_anem-{anem}.png'))
            plt.show()
    return None

# plot_topography_per_anem(list_of_degs = list(range(360)), list_of_distances=[i*(5.+5.*i) for i in range(45)], plot_topography=True, plot_slopes=False)


def get_heights_from_X_dirs_and_dists(point_1, array_of_dirs, cone_angles, dists):
    heights = []
    for a in cone_angles:
        X_dir_anem_yawed = convert_angle_to_0_2pi_interval(array_of_dirs + a, input_and_output_in_degrees=True)
        points_2 = np.array([get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=d, distance=dists[-1]) for d in X_dir_anem_yawed])
        heights.append(np.array([elevation_profile_generator(point_1=point_1, point_2=p2, step_distance=False, list_of_distances=dists)[1] for p2 in points_2]))
    return np.array(heights)

windward_dists = [i*(5. + 5.*i) for i in range(45)]
leeward_dists =  [i*(10.+10.*i) for i in range(10)]
side_dists =     [i*(10.+10.*i) for i in range(10)]


def get_all_10_min_data_at_z_48m(U_min = 0, windward_dists=windward_dists, leeward_dists=leeward_dists, side_dists=side_dists):
    print('Collecting all 10-min wind data... (takes 10-60 minutes)')
    min10_df_all_means, min10_df_all_dirs, min10_df_all_Iu, min10_df_all_Iv, min10_df_all_Iw, min10_df_all_avail = merge_two_all_stats_files()
    assert min10_df_all_means['ts'].equals(min10_df_all_dirs['ts'])
    assert min10_df_all_dirs['ts'].equals(min10_df_all_Iu['ts'])
    assert min10_df_all_Iu['ts'].equals(min10_df_all_Iv['ts'])
    assert min10_df_all_Iv['ts'].equals(min10_df_all_Iw['ts'])
    assert min10_df_all_Iw['ts'].equals(min10_df_all_avail['ts'])
    ts_df = min10_df_all_means['ts']
    columns_to_drop = ['ts', 'osp1_C', 'osp2_C', 'svar_B','svar_C','synn_B','synn_C','land_B','land_C','neso_B','neso_C']  # Discarding all data that are not at Z=48m
    X_means_df = min10_df_all_means.drop(columns=columns_to_drop)
    X_dirs_df =  min10_df_all_dirs.drop(columns=columns_to_drop)
    X_Iu_df = min10_df_all_Iu.drop(columns=columns_to_drop)
    idxs_where_cond = (U_min <= X_means_df)  # Discarding all data with U below U_min
    X_means_df = X_means_df[idxs_where_cond]  # convert to nan each entry associated with U < U_min
    X_dirs_df  = X_dirs_df[idxs_where_cond]  # convert to nan each entry associated with U < U_min
    X_Iu_df    = X_Iu_df[idxs_where_cond]  # convert to nan each entry associated with U < U_min
    idxs_where_drop_1 = X_means_df.isna().all(axis=1).to_numpy()  # idxs to drop, where all columns are nan
    idxs_where_drop_2 = X_dirs_df.isna().all(axis=1).to_numpy()  # idxs to drop, where all columns are nan
    idxs_where_drop_3 = X_Iu_df.isna().all(axis=1).to_numpy()  # idxs to drop, where all columns are nan
    idxs_where_drop_all = np.where(np.logical_or.reduce((idxs_where_drop_1, idxs_where_drop_2, idxs_where_drop_3)))[0]
    ts_df = ts_df.drop(idxs_where_drop_all)  # droping rows where all columns are nan
    X_means_df = X_means_df.drop(idxs_where_drop_all)  # droping rows where all columns are nan
    X_dirs_df = X_dirs_df.drop(idxs_where_drop_all)  # droping rows where all columns are nan
    X_Iu_df = X_Iu_df.drop(idxs_where_drop_all)  # droping rows where all columns are nan
    X_std_u_df = X_Iu_df.multiply(X_means_df)
    all_anem_list = ['osp1_A', 'osp1_B', 'osp2_A', 'osp2_B', 'svar_A', 'synn_A', 'land_A', 'neso_A']
    # Organizing the data into an input matrix (with shape shape (n_samples, n_features)):
    X_data = []
    y_data = []
    ts_data = []
    other_data = []
    data_len_of_each_anem = []
    mast_UTM_33 = {'synn':[-34515., 6705758.], 'osp1':[-39375., 6703464.], 'osp2':[-39350., 6703204.], 'svar':[-34625., 6700051.], 'land':[-35446., 6688200.], 'neso':[-30532., 6682896.]}
    for mast_anem in all_anem_list:
        # We allow different data lengths for each anemometer, but a row with at least one nan is removed from a given anemometer df of 'means' 'dirs' and 'stds'
        X_mean_dir_std_anem = pd.DataFrame({'ts': ts_df,'means': X_means_df[mast_anem], 'dirs': X_dirs_df[mast_anem], 'stds': X_std_u_df[mast_anem]}).dropna(axis=0, how='any')
        ts_anem = X_mean_dir_std_anem['ts']
        X_mean_anem =  np.array(X_mean_dir_std_anem['means'])
        X_dir_anem =   np.array(X_mean_dir_std_anem['dirs'])
        X_std_u_anem = np.array(X_mean_dir_std_anem['stds'])
        point_1 = mast_UTM_33[mast_anem[:4]]
        windward_cone_angles = [-14, -9, -5, -2, 0, 2, 5, 9, 14]  # deg. the a angles within the "+-15 deg cone of influence" of the windward terrain on the wind properties
        leeward_cone_angles = [165,170,175,180,185,190,195]  # deg. the a angles within the "+-15 deg cone of influence" of the windward terrain on the wind properties
        side_cone_angles = [30, 45, 60, 75, 90, 105, 120, 135, 150]
        side_cone_angles = side_cone_angles + list(np.array(side_cone_angles) * -1)
        windward_heights = get_heights_from_X_dirs_and_dists(point_1, X_dir_anem, windward_cone_angles, windward_dists)
        leeward_heights =  get_heights_from_X_dirs_and_dists(point_1, X_dir_anem,  leeward_cone_angles, leeward_dists)
        side_heights =     get_heights_from_X_dirs_and_dists(point_1, X_dir_anem,     side_cone_angles, side_dists)
        windward_middle_index = int(len(windward_cone_angles)/2)
        X_data_1_anem = np.concatenate((X_mean_anem[:,None], X_dir_anem[:,None], windward_heights[windward_middle_index], np.mean(windward_heights,axis=0)[:,1:], np.std(windward_heights,axis=0)[:,1:]
                                                                                ,np.mean(leeward_heights,axis=0),  np.std(leeward_heights,axis=0)[:,1:]
                                                                                ,np.mean(side_heights,axis=0)    , np.std(side_heights,axis=0)[:,1:]
                                                                                ,), axis=1)  # [1:,:] because point_1 has always std = 0 (same point 1 for all cone_angles)
        other_data_1_anem = [windward_heights.tolist()] #, leeward_heights.tolist(), side_heights.tolist()]
        # todo: replace "X_dir_anem[:,None]" with two inputs: "np.sin(np.deg2rad....X_dir_anem[:,None])", "np.cos(X_dir_anem[:,None])"
        # todo: remember rad2deg afterwards
        X_data.append(X_data_1_anem)
        y_data.append(X_std_u_anem)
        ts_data.append(ts_anem)
        other_data.append(other_data_1_anem)  # final shape (list of lists of 3D arrays: [n_anem][windward/leeward/side][n_cone_angles, n_samples, n_terrain_points]
        data_len_of_each_anem.append(len(X_mean_anem))
        print(f'{mast_anem}: All ML-input-data is collected')
    X_data = np.concatenate(tuple(i for i in X_data), axis=0)  # n_steps + 2 <=> number of terrain heights (e.g. 500) + wind speed (1) + wind dir (1)
    y_data = np.concatenate(y_data)
    start_idxs_of_each_anem = [0] + np.cumsum(data_len_of_each_anem)[:-1].tolist()
    return  X_data, y_data, all_anem_list, start_idxs_of_each_anem, ts_data, other_data


def get_df_of_merged_temperatures(list_of_file_paths, label_to_read='Lufttemperatur', smooth_str='3600s'):
    list_of_dfs = [pd.read_csv(file, delimiter=';', skipfooter=1, engine='python') for file in list_of_file_paths]
    n_files = len(list_of_dfs)
    for i in range(n_files):  # converting the timestamps to datetime objects
        list_of_dfs[i]['Tid(norsk normaltid)'] = pd.to_datetime(list_of_dfs[i]['Tid(norsk normaltid)'], format='%d.%m.%Y %H:%M')
        list_of_dfs[i] = list_of_dfs[i].rename(columns={'Tid(norsk normaltid)':'ts_all_1h', label_to_read:'temperature_'+str(i+1)}).set_index('ts_all_1h').drop(['Navn','Stasjon'], axis=1)
        list_of_dfs[i]['temperature_'+str(i+1)] = pd.to_numeric(list_of_dfs[i]['temperature_'+str(i+1)].str.replace(',','.'))
    ts_min = min([min(list_of_dfs[i].index) for i in range(n_files)])
    ts_max = max([max(list_of_dfs[i].index) for i in range(n_files)])
    ts_all_1h = pd.date_range(ts_min, ts_max, freq='h')
    list_of_all_dfs = [pd.DataFrame({'ts_all_1h':ts_all_1h}).set_index('ts_all_1h')] + list_of_dfs
    final_df = pd.DataFrame().join(list_of_all_dfs, how="outer")
    final_df['temperature_final'] = final_df.mean(axis=1, skipna=True).rolling(smooth_str).mean()
    return final_df


def get_max_num_consec_NaN(df):
    return max(df.isnull().astype(int).groupby(df.notnull().astype(int).cumsum()).cumsum())


def predict_turbulence_with_ML():
    U_min = 2
    # ##################################################################
    # # # Getting the data for first time
    # X_data_nonnorm, y_data_nonnorm, all_anem_list, start_idxs_of_each_anem, ts_data, other_data = get_all_10_min_data_at_z_48m(U_min=U_min) ##  takes a few minutes...
    # # # # ##################################################################
    # # # # # Saving data
    # data_path = os.path.join(os.getcwd(), 'processed_data_for_ML', f'X_y_ML_ready_data_Umin_{U_min}_masts_6_new_dirs_w_ts')
    # np.savez_compressed(data_path, X=X_data_nonnorm, y=y_data_nonnorm, m=all_anem_list, i=start_idxs_of_each_anem, t=np.array(ts_data, dtype=object)) # o=other_data)
    # with open(data_path+'_other_data.txt', 'w') as outfile:
    #     json.dump(other_data, outfile)
    ##################################################################
    y_data_type = 'Iu'  # 'U', 'std', 'Iu'
    only_1_elevation_profile = True
    add_roughness_input = True
    input_weather_data = True
    input_wind_data = True
    do_sector_avg = True
    dir_sector_amp = 1  # amplitude in degrees of each directional sector, for calculating mean properties

    # for only_1_elevation_profile, add_roughness_input in zip([True,False], [True,False]):
    for input_weather_data, input_wind_data in [(True, True)]:  # , (False, False)]:
        # for U_min in [0,2,5]:
        for do_sector_avg in [False]:  #, True]:
            # for dir_sector_amp in [1,3]:
            for dir_sector_amp in [1]:
                # Loading data already saved
                data_path = os.path.join(os.getcwd(), 'processed_data_for_ML', f'X_y_ML_ready_data_Umin_{U_min}_masts_6_new_dirs_w_ts')
                loaded_data = np.load(data_path + '.npz')
                X_data_nonnorm_orig = loaded_data['X']
                X_data_nonnorm = copy.deepcopy(X_data_nonnorm_orig)
                X_dirs = copy.deepcopy(X_data_nonnorm[:,1])
                y_data_nonnorm = loaded_data['y']
                all_anem_list = loaded_data['m']
                n_anems = len(all_anem_list)
                loaded_data.allow_pickle = True
                X_ts = loaded_data['t']
                # X_other = loaded_data['o']
                start_idxs_of_each_anem = loaded_data['i']
                dir_sectors = np.arange(0,360,dir_sector_amp)
                X_dir_sectors = np.searchsorted(dir_sectors, X_dirs, side='right') - 1  # groups all the measured directions into sectors

                if y_data_type == 'U':
                    y_data_nonnorm = copy.deepcopy(X_data_nonnorm_orig[:,0])
                    X_data_nonnorm = copy.deepcopy(np.delete(X_data_nonnorm, 0, axis=1))
                elif y_data_type == 'Iu':
                    y_data_nonnorm = copy.deepcopy(y_data_nonnorm / X_data_nonnorm[:,0])

                if only_1_elevation_profile:
                    X_data_nonnorm = copy.deepcopy(X_data_nonnorm[:,:2+45])

                # if not input_wind_data: todo: TRASH
                #     X_data_nonnorm = copy.deepcopy(X_data_nonnorm[:,2:])

                if add_roughness_input:
                    X_roughness = X_data_nonnorm_orig[:,2:2+45].astype(bool).astype(float)  # water level will become 0, everything else will become 1
                    X_roughness = X_roughness[:,3:]  # the first 2 points are always 1 so they become NaN when normalized!
                    X_data_nonnorm = copy.deepcopy(np.hstack((X_data_nonnorm, X_roughness)))

                if input_weather_data:
                    # Getting other weather data from https://seklima.met.no/ (use Microsoft Edge!)
                    weather_data_path = os.path.join(os.getcwd(), 'weather_data')
                    list_weather_data_files = os.listdir(weather_data_path)
                    list_air_temp_files = [os.path.join(weather_data_path, file) for file in list_weather_data_files if 'air' in file]
                    list_sea_temp_files = [os.path.join(weather_data_path, file) for file in list_weather_data_files if 'sea' in file]
                    df_air_temp = get_df_of_merged_temperatures(list_air_temp_files, label_to_read='Lufttemperatur', smooth_str='1h')
                    df_sea_temp = get_df_of_merged_temperatures(list_sea_temp_files, label_to_read='Sjøtemperatur' , smooth_str='24h')
                    print('Max. consecutive NaN air temperatures (1h each) before interpolation:', get_max_num_consec_NaN(df_air_temp['temperature_final']))
                    print('Max. consecutive NaN sea temperatures (1h each) before interpolation:', get_max_num_consec_NaN(df_sea_temp['temperature_final']))
                    # Closing the NaN gaps with interpolation to 'Nearest'
                    df_air_temp = df_air_temp.interpolate(method='linear', axis=0) # .filter(['temperature_final'])
                    df_sea_temp = df_sea_temp.interpolate(method='linear', axis=0) # .filter(['temperature_final'])
                    assert 0 == get_max_num_consec_NaN(df_air_temp['temperature_final']) == get_max_num_consec_NaN(df_sea_temp['temperature_final'])
                    # Intersecting the weather data (every 1h) with the anemometer data (every 10min)
                    X_ts_rounded = [pd.to_datetime(X_ts[i]).round('60min') for i in range(n_anems)]
                    air_temp_idxs = [np.intersect1d(X_ts_rounded[i], df_air_temp.index, return_indices=True)[2] for i in range(n_anems)]  # idxs where first unique X_ts_rounded[i] matches df_air_temp.index
                    sea_temp_idxs = [np.intersect1d(X_ts_rounded[i], df_sea_temp.index, return_indices=True)[2] for i in range(n_anems)]  # idxs where first unique X_ts_rounded[i] matches df_sea_temp.index
                    X_ts_rounded_repeats = [dict(Counter(X_ts_rounded[i])) for i in range(n_anems)]  # since intersect1d only gives first index of unique value, we need to repeat (give e.g. same air temp) to several X_ts_rounded with same value
                    X_air_temp = []
                    X_sea_temp = []
                    for a in range(n_anems):
                        X_air_temp_1_anem = []
                        X_sea_temp_1_anem = []
                        for idx_air, idx_sea, rep in zip(air_temp_idxs[a], sea_temp_idxs[a], X_ts_rounded_repeats[a].values()):
                            X_air_temp_1_anem.extend([df_air_temp['temperature_final'].iloc[idx_air]] * rep)
                            X_sea_temp_1_anem.extend([df_sea_temp['temperature_final'].iloc[idx_sea]] * rep)
                        X_air_temp.append(X_air_temp_1_anem)
                        X_sea_temp.append(X_sea_temp_1_anem)
                        # For confirmation of the code above, run e.g.: print(X_ts_rounded[4][500:520]), print(X_air_temp[4][500:520]),
                        # print(df_air_temp[np.logical_and(X_ts_rounded[4].iloc[500] <= df_air_temp.index, df_air_temp.index < X_ts_rounded[4].iloc[520])])
                    X_air_temp_flat = np.array([item for sublist in X_air_temp for item in sublist])  # flattening the list of lists X_air_temp
                    X_sea_temp_flat = np.array([item for sublist in X_sea_temp for item in sublist])  # flattening the list of lists X_sea_temp
                    X_temp_diff = X_air_temp_flat - X_sea_temp_flat  # reinforcing the idea that the difference between air and sea temperature is the important parameter
                    # Finally appending the weather data to the X_data_nonnorm
                    X_data_nonnorm = np.hstack((X_air_temp_flat[:,None], X_sea_temp_flat[:,None], X_temp_diff[:,None], X_data_nonnorm))

                n_samples = X_data_nonnorm.shape[0]
                n_features = X_data_nonnorm.shape[1]
                start_idxs_of_each_anem_2 = np.array(start_idxs_of_each_anem.tolist() + [n_samples])  # this one includes the final index as well


                # Plotting correlation coefficient per feature
                plt.figure(figsize=(20,5))
                y_plot_corrcoef = [np.corrcoef(X_data_nonnorm[:,i], y_data_nonnorm)[0][1] for i in range(X_data_nonnorm.shape[1])]
                x_plot_corrcoef = [i for i in range(X_data_nonnorm.shape[1])]
                plt.plot(x_plot_corrcoef, y_plot_corrcoef, marker='o')
                plt.show()

                #################################################################
                # Statistical analyses of the distribution of turbulence. Converting y_data (1 dimension) into y2_data (2 dimensions -> 2 Weibull params)
                ##################################################################################################################
                # Obtaining the 4 Exponentiated-Weibull parameters, valid for a 1 deg sector of 1 anemometer, and copying these values for all data that falls in the same 1 deg sector and anem
                PDF_used = 'mean'  # 'mean', 'rayleigh', 'weibull', 'expweibull'
                n_outputs = 1
                y_PDF_data_nonnorm = np.empty((len(y_data_nonnorm), n_outputs))  # 3 parameters in the exponentiated weibull distribution
                y_PDF_data_nonnorm[:] = np.nan
                y_PDF_data_nonnorm_360dirs = []
                for anem_idx in range(len(all_anem_list)):
                    y_PDF_data_nonnorm_per_anem = []
                    anem_slice = slice(start_idxs_of_each_anem_2[anem_idx], start_idxs_of_each_anem_2[anem_idx + 1])
                    for s in range(len(dir_sectors)):
                        idxs_360dir = np.where(X_dir_sectors[anem_slice] == s)[0]
                        if len(idxs_360dir) > 10 :  # minimum number of datapoints, otherwise -> np.nan
                            if PDF_used == 'mean':
                                param_mean = np.mean(y_data_nonnorm[anem_slice][idxs_360dir])  # params = a(alpha), c(shape==k), loc, scale(lambda)
                                y_PDF_data_nonnorm[anem_slice][idxs_360dir] = np.array([param_mean])  # parameter '2' not included (floc, which is set to 0)
                                y_PDF_data_nonnorm_per_anem.append(np.array([param_mean]))
                            if PDF_used == 'rayleigh':
                                _, param_scale = stats.rayleigh.fit(y_data_nonnorm[anem_slice][idxs_360dir], floc = 0)  # params = a(alpha), c(shape==k), loc, scale(lambda)
                                y_PDF_data_nonnorm[anem_slice][idxs_360dir] = np.array([param_scale])  # parameter '2' not included (floc, which is set to 0)
                                y_PDF_data_nonnorm_per_anem.append(np.array([param_scale]))
                            if PDF_used == 'weibull':
                                _      , param_c, param_loc, param_scale = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir], f0 = 1)  # params = a(alpha), c(shape==k), loc, scale(lambda)
                                y_PDF_data_nonnorm[anem_slice][idxs_360dir] = np.array([param_c, param_loc, param_scale])  # parameter '2' not included (floc, which is set to 0)
                                y_PDF_data_nonnorm_per_anem.append(np.array([param_c, param_loc, param_scale]))
                            elif PDF_used == 'expweibull':
                                param_a, param_c, param_loc, param_scale = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir])  # params = a(alpha), c(shape==k), loc, scale(lambda)
                                y_PDF_data_nonnorm[anem_slice][idxs_360dir] = np.array([param_a, param_c, param_loc, param_scale])  # parameter '2' not included (floc, which is set to 0)
                                y_PDF_data_nonnorm_per_anem.append(np.array([param_a, param_c, param_loc, param_scale]))
                        else:
                            print(f'Not enough data for sector {s} ({dir_sectors[s]})')
                            y_PDF_data_nonnorm_per_anem.append(np.ones(n_outputs)*np.nan)
                    y_PDF_data_nonnorm_360dirs.append(y_PDF_data_nonnorm_per_anem)
                    print(f'{all_anem_list[anem_idx]}: PDF parameters obtained')


                y_PDF_data_nonnorm_zscores = (y_PDF_data_nonnorm - np.nanmean(y_PDF_data_nonnorm, axis=0) ) / np.nanstd(y_PDF_data_nonnorm, axis=0)
                bools_outliers = np.logical_or(np.any(-3.9 > y_PDF_data_nonnorm_zscores, axis=1),
                                               np.any( 3.9 < y_PDF_data_nonnorm_zscores, axis=1))
                print(f'Maximum and minimum zscores in y_PDF_data is {max(y_PDF_data_nonnorm_zscores)[0]} and {min(y_PDF_data_nonnorm_zscores)[0]}')


                y_data_nonnorm_zscores = (y_data_nonnorm - np.nanmean(y_data_nonnorm, axis=0) ) / np.nanstd(y_data_nonnorm, axis=0)
                bools_outliers = np.logical_or(-3.9 > y_data_nonnorm_zscores,
                                                50 < y_data_nonnorm_zscores)
                print(f'Maximum and minimum zscores in y_data is {max(y_data_nonnorm_zscores)} and {min(y_data_nonnorm_zscores)}')



                # ##################################################################
                # # Converting outliers to nan
                # y_PDF_data_nonnorm[np.where(bools_outliers)[0]] = np.nan
                # y_data_nonnorm[np.where(bools_outliers)[0]] = np.nan
                if y_data_type == 'Iu':
                    y_data_nonnorm[y_data_nonnorm>1] = np.nan
                # ##################################################################

                ##################################################################
                # Converting X_dirs into cos(X_dirs) and sin(X_dirts) for the ANN to understand directions better (note that 359 deg is almost the same as 1 deg)
                dirs_idx = 4 if input_weather_data else 1
                if input_weather_data:
                    X_data_nonnorm[:,2] = X_data_nonnorm[:,3]  # passing U to the idx 2
                    X_data_nonnorm[:,3] = np.cos(np.deg2rad(X_data_nonnorm[:,4]))
                    X_data_nonnorm[:,4] = np.sin(np.deg2rad(X_data_nonnorm[:,4]))
                    print('Tair-Tsea is being overwritten with cos(dir). Dir is being overwritten with sin(dir)')
                    print('Final shape of X data: Tair, Tsea, U, cos(dir), sin(dir), Z1 Z2..., R1 R2...')
                else:
                    print('ATTENTION: Cos(dir) and Sin(dir) is NOT IMPLEMENTED when input_weather_data is False')

                ##################################################################
                # Organizing all the data in a dataframe
                all_data = pd.concat([pd.DataFrame(X_data_nonnorm), pd.DataFrame(y_data_nonnorm), pd.DataFrame(y_PDF_data_nonnorm)], axis=1)
                aux_ones = np.zeros(n_samples)  # auxiliary variable with zeros
                aux_ones[start_idxs_of_each_anem] = 1  # ... and then with ones where the data changes to a new anemometer
                idxs_each_anem = (np.cumsum(aux_ones) - 1).astype(int)  # array with index of each anemometer, for each data point
                all_data['anem'] = idxs_each_anem

                # Removing NaN from the data
                all_data = all_data.dropna(axis=0, how='any').reset_index(drop=True)  # first n_features columns are X_data, then 1 column for y_data, then 3 or 4 columns to y_PDF_data, then 1 column for anem idxs

                # Updating all the data
                X_data_nonnorm =     all_data.iloc[:,:n_features].to_numpy()
                y_data_nonnorm =     all_data.iloc[:, n_features].to_numpy()
                y_PDF_data_nonnorm = all_data.iloc[:,n_features+1:-1].to_numpy()
                start_idxs_of_each_anem = np.intersect1d(all_data['anem'], np.arange(len(all_anem_list)).tolist(), return_indices=True)[1]
                # X_dir_sectors = np.searchsorted(dir_sectors, X_data_nonnorm[:,1], side='right') - 1  # groups all the measured directions into sectors
                n_samples = X_data_nonnorm.shape[0]
                start_idxs_of_each_anem_2 = np.array(start_idxs_of_each_anem.tolist() + [n_samples])  # this one includes the final index as well


                # X_dirs = copy.deepcopy(X_data_nonnorm[:,dirs_idx])
                X_dirs = np.rad2deg([convert_angle_to_0_2pi_interval(i) for i in np.arctan2(X_data_nonnorm[:,4], X_data_nonnorm[:,3])])

                X_dir_sectors = np.searchsorted(dir_sectors, X_dirs, side='right') - 1  # groups all the measured directions into sectors

                ##################################################################
                # Normalizing data
                X_maxs = np.max(X_data_nonnorm, axis=0)
                X_mins = np.min(X_data_nonnorm, axis=0)
                y_max =  np.max(y_data_nonnorm)
                y_min =  np.min(y_data_nonnorm)
                y_PDF_maxs = np.nanmax(y_PDF_data_nonnorm, axis=0)
                y_PDF_mins = np.nanmin(y_PDF_data_nonnorm, axis=0)
                X_data =     (X_data_nonnorm - X_mins) / (X_maxs - X_mins)
                y_data =     (y_data_nonnorm -  y_min) / ( y_max - y_min)
                y_PDF_data = (y_PDF_data_nonnorm - y_PDF_mins) / (y_PDF_maxs - y_PDF_mins)
                ##################################################################

                # ################################################################## @@@@  TERRIBLE IDEA! THEN E.G. TURBULENCE FROM SEA IS VERY DIFFERENT FROM SVAR AND OSP
                # # Further removing the mean from each anemometer data!!!!!!!!!!!!
                # mean_all_y_data = np.mean(y_data)
                # for anem_idx in range(len(all_anem_list)):
                #     anem_slice = slice(start_idxs_of_each_anem_2[anem_idx], start_idxs_of_each_anem_2[anem_idx + 1])
                #     y_data[anem_slice] = y_data[anem_slice] - np.mean(y_data[anem_slice]) + mean_all_y_data
                # ################################################################## @@@@

                # Generating synthetic data, using only the weibull parameters for each 1-deg sector and anemometer
                y_data_synth =  np.empty((len(y_data_nonnorm)))  # 4 parameters in the exponentiated weibull distribution
                y_data_synth[:] = np.nan
                for anem_idx in range(len(all_anem_list)):
                    anem_slice = slice(start_idxs_of_each_anem_2[anem_idx], start_idxs_of_each_anem_2[anem_idx + 1])
                    for s in range(len(dir_sectors)):
                        idxs_360dir = np.where(X_dir_sectors[anem_slice] == s)[0]
                        if PDF_used == 'rayleigh':
                            param_scale = y_PDF_data_nonnorm_360dirs[anem_idx][s]
                            y_data_synth[anem_slice][idxs_360dir] = stats.rayleigh.rvs(0, param_scale, size=len(idxs_360dir))
                        elif PDF_used == 'weibull':
                            param_c, param_loc, param_scale = y_PDF_data_nonnorm_360dirs[anem_idx][s]
                            y_data_synth[anem_slice][idxs_360dir] = stats.exponweib.rvs(      1, param_c, param_loc, param_scale, size=len(idxs_360dir))
                        elif PDF_used == 'expweibull':
                            param_a, param_c, param_loc, param_scale = y_PDF_data_nonnorm_360dirs[anem_idx][s]
                            y_data_synth[anem_slice][idxs_360dir] = stats.exponweib.rvs(param_a, param_c, param_loc, param_scale, size=len(idxs_360dir))

                # Plotting synthetic data by anemometer
                def plot_y_PDF_param_data():
                    for anem_idx in range(len(all_anem_list)):
                        anem_slice = slice(start_idxs_of_each_anem_2[anem_idx],start_idxs_of_each_anem_2[anem_idx+1])
                        plt.figure()
                        plt.title(f'Measured. {all_anem_list[anem_idx]}')
                        plt.scatter(X_dirs[anem_slice], y_data_nonnorm[anem_slice], s=0.01, alpha = 0.2, c='black', label='Measured')
                        plt.plot(dir_sectors, y_PDF_data_nonnorm_360dirs[anem_idx], alpha=0.5, c='red', label='Mean')
                        plt.legend(loc=1)
                        plt.ylim([0,1])
                        plt.savefig(os.path.join(os.getcwd(), 'plots', y_data_type + f'_data_and_fitted_Ray_{all_anem_list[anem_idx]}_Umin_5.png'))
                        plt.show()


                def plot_measured_VS_synth_data():
                    for anem_idx in range(len(all_anem_list)):
                        anem_slice = slice(start_idxs_of_each_anem_2[anem_idx],start_idxs_of_each_anem_2[anem_idx+1])
                        x = np.deg2rad(X_dirs[anem_slice])
                        y1 = y_data_nonnorm[anem_slice]
                        y2 = y_data_synth[anem_slice]
                        data1, x1_e, y1_e = np.histogram2d(np.sin(x)*y1, np.cos(x)*y1, bins=30, density=True)
                        data2, x2_e, y2_e = np.histogram2d(np.sin(x)*y2, np.cos(x)*y2, bins=[x1_e, y1_e], density=True)
                        x1_i = 0.5 * (x1_e[1:] + x1_e[:-1])
                        y1_i = 0.5 * (y1_e[1:] + y1_e[:-1])
                        x2_i = 0.5 * (x2_e[1:] + x2_e[:-1])
                        y2_i = 0.5 * (y2_e[1:] + y2_e[:-1])
                        z1 = interpn((x1_i, y1_i), data1, np.vstack([np.sin(x)*y1, np.cos(x)*y1]).T, method="splinef2d", bounds_error=False)
                        z2 = interpn((x2_i, y2_i), data2, np.vstack([np.sin(x)*y2, np.cos(x)*y2]).T, method="splinef2d", bounds_error=False)
                        z1[np.where(np.isnan(z1))] = 0.0  # To be sure to plot all data
                        z2[np.where(np.isnan(z2))] = 0.0  # To be sure to plot all data
                        idx1 = z1.argsort()   # Sort the points by density, so that the densest points are plotted last
                        idx2 = z2.argsort()   # Sort the points by density, so that the densest points are plotted last
                        x1, y1, z1 = x[idx1], y1[idx1], z1[idx1]
                        x2, y2, z2 = x[idx2], y2[idx2], z2[idx2]
                        # 1st plot - Measured
                        fig, ax = plt.subplots(figsize=(8, 6), dpi=300, subplot_kw={'projection': 'polar'})
                        ax.set_theta_zero_location("N")
                        ax.set_theta_direction(-1)
                        ax.scatter(x1, y1, c=z1, s=1, alpha=0.3)
                        norm = Normalize(vmin = np.min([z1,z2]), vmax = np.max([z1,z2]))
                        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
                        cbar.ax.set_ylabel('Density')
                        plt.title(f'Measured. {all_anem_list[anem_idx]}')
                        plt.ylim([0, 2])
                        # ax.text(np.deg2rad(20), 4.4, '$\sigma(u)\/[m/s]$')
                        plt.tight_layout()
                        plt.savefig(os.path.join(os.getcwd(), 'plots', f'measured_std_u_{all_anem_list[anem_idx]}.png'))
                        plt.show()
                        # 2nd plot - Synthetic
                        fig, ax = plt.subplots(figsize=(8, 6), dpi=300, subplot_kw={'projection': 'polar'})
                        ax.set_theta_zero_location("N")
                        ax.set_theta_direction(-1)
                        ax.scatter(x2, y2, c=z2, s=1, alpha=0.3)
                        norm = Normalize(vmin = np.min([z1,z2]), vmax = np.max([z1,z2]))
                        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
                        cbar.ax.set_ylabel('Density')
                        plt.title(f'Synthetic (Weibull). {all_anem_list[anem_idx]}')
                        plt.ylim([0, 2])
                        # ax.text(np.deg2rad(20), 4.4, '$\sigma(u)\/[m/s]$')
                        plt.tight_layout()
                        plt.savefig(os.path.join(os.getcwd(), 'plots', f'synth_std_u_{all_anem_list[anem_idx]}.png'))
                        plt.show()

                ########################################
                # MACHINE LEARNING
                ########################################
                # Neural network
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'cuda' or 'cpu'. 'cuda' doesn't seem to be working...
                def train_and_test_NN(X_train, y_train, X_test, y_test, hp, print_loss_per_epoch=True, print_results=True, R2_of='values'):
                    """
                    Args:
                        R2_of:
                        X_train:
                        y_train:
                        X_test:
                        y_test:
                        hp:  hyperparameters. e.g. {'lr':1E-1, 'batch_size':92, 'weight_decay':1E-4, 'momentum':0.9, 'n_epochs':10, 'n_hid_layers':2}
                        print_loss_per_epoch:
                        print_results:
                    Returns:
                    """
                    learn_rate = hp['lr']
                    batch_size = hp['batch_size']
                    weight_decay = hp['weight_decay']
                    momentum = hp['momentum']
                    n_epochs = hp['n_epochs']
                    n_hid_layers = hp['n_hid_layers']
                    my_activation_func = hp['activation']  # ReLU, ELU, LeakyReLU, etc.
                    criterion = hp['loss']  # define the loss function
                    n_samples_train = X_train.shape[0]
                    # Building a neural network dynamically
                    torch.manual_seed(0)  # make the following random numbers reproducible
                    n_features = X_train.shape[1]  # number of independent variables in the polynomial
                    n_outputs = y_train.shape[1]  # number of independent variables in the polynomial
                    n_hid_layer_neurons = n_features  # Fancy for: More monomials, more neurons...
                    my_nn = torch.nn.Sequential()
                    my_nn.add_module(name='0', module=torch.nn.Linear(n_features, n_hid_layer_neurons))  # Second layer
                    my_nn.add_module(name='0A', module=my_activation_func())  # Activation function
                    for i in range(n_hid_layers):  # Hidden layers
                        n_neurons_last_layer = (list(my_nn.modules())[-2]).out_features
                        my_nn.add_module(name=str(i + 1), module=torch.nn.Linear(n_neurons_last_layer, max(round(2/3 * n_neurons_last_layer),5)))
                        my_nn.add_module(name=f'{i + 1}A', module=my_activation_func())
                    n_neurons_last_layer = (list(my_nn.modules())[-2]).out_features
                    my_nn.add_module(name=str(n_hid_layers + 1), module=torch.nn.Linear(n_neurons_last_layer, n_outputs))  # Output layer
                    optimizer = SGD(my_nn.parameters(), lr=learn_rate, weight_decay=weight_decay, momentum=momentum)  # define the optimizer
                    # torch.seed()  # make random numbers again random
                    my_nn.to(device)  # To GPU if available
                    # Training
                    # writer = SummaryWriter(f'runs/my_math_learning_tensorboard')  # For later using TensorBoard, for visualization
                    assert (n_samples_train/batch_size).is_integer(), "Change batch size so that n_iterations is integer"
                    n_iterations = int(n_samples_train/batch_size)
                    for epoch in range(n_epochs):
                        epoch_loss = 0
                        idxs_shuffled = torch.randperm(n_samples_train)
                        for b in range(n_iterations):
                            batch_idxs = idxs_shuffled[b*batch_size:b*batch_size+batch_size]
                            y_pred = my_nn(Variable(X_train[batch_idxs]))
                            loss = criterion(y_pred, Variable(y_train[batch_idxs].view(batch_size,n_outputs), requires_grad=False))
                            epoch_loss = loss.item()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        if print_loss_per_epoch:
                            print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
                        # writer.add_scalar('Training Loss', epoch_loss, global_step=epoch)  # writing to TensorBoard
                    # Testing
                    n_samples_test = X_test.shape[0]
                    with torch.no_grad():
                        y_pred = my_nn(Variable(X_test))
                    if R2_of == 'values':
                        y_test_total_mean = torch.mean(y_test.view(n_samples_test, n_outputs), axis=0)
                        SS_res_test = torch.sum((y_test.view(n_samples_test,n_outputs) - y_pred.view(n_samples_test,n_outputs))**2)
                        SS_tot_test = torch.sum((y_test.view(n_samples_test,n_outputs) - y_test_total_mean)**2)
                        R2_test = 1 - SS_res_test / SS_tot_test
                        idxs_to_print = np.random.randint(0, len(y_pred), 10)  # a few random values to be printed
                    elif R2_of == 'means':
                        X_test_cos_dir = X_test[:, 3] * (X_maxs[3] - X_mins[3]) + X_mins[3]
                        X_test_sin_dir = X_test[:, 4] * (X_maxs[4] - X_mins[4]) + X_mins[4]
                        X_test_dirs = convert_angle_to_0_2pi_interval(torch.atan2(X_test_sin_dir, X_test_cos_dir), input_and_output_in_degrees=False)
                        X_dir_sectors = np.searchsorted(dir_sectors, torch.rad2deg(X_test_dirs).cpu().numpy(), side='right') - 1  # groups all the measured directions into sectors
                        y_test_mean = np.array([np.mean(y_test.cpu().numpy()[np.where(X_dir_sectors == d)[0]]) for d in dir_sectors])
                        y_pred_mean = np.array([np.mean(y_pred.cpu().numpy()[np.where(X_dir_sectors == d)[0]]) for d in dir_sectors])
                        # Removing NaN from the data after organizing it into a dataframe
                        all_mean_data = pd.DataFrame({'dir_sectors':dir_sectors, 'y_test_mean':y_test_mean, 'y_pred_mean':y_pred_mean}).dropna(axis=0, how='any').reset_index(drop=True)
                        R2_test = r2_score(all_mean_data['y_test_mean'], all_mean_data['y_pred_mean'])  # r2_score would give error if there was a NaN.
                    if print_results:
                        print(f'R2 (of {R2_of}!) on test dataset: ----> {R2_test} <---- . Learning rate: {learn_rate}')
                        print(f"Prediction: {y_pred[idxs_to_print]}")
                        print(f"Reference:   {y_test[idxs_to_print]}")
                        print(f'Batch size: {batch_size}')
                    return y_pred, R2_test
                ##################################################################################################################


                ##################################################################################################################
                # TRAINING FROM CERTAIN ANEMOMETERS AND TESTING AT 1 GIVEN ANEMOMETER


                def get_idxs_from_anems(anem_list, n_samples, all_anem_list):
                    """
                    Args:
                        anem_list:
                        n_samples: number of all data points. e.g. X_data.shape[0]
                        all_anem_list: list of all anemometers available in the data
                    Returns: All indexes of the datapoints relative to the anemometers given in anem_list
                    """
                    list_idxs = []
                    for a in anem_list:
                        anem_start_idx = start_idxs_of_each_anem_2[np.where(all_anem_list == a)[0]][0]
                        anem_end_idx =   start_idxs_of_each_anem_2[np.where(all_anem_list == a)[0]+1][0]
                        list_idxs.extend(np.where((anem_start_idx <= np.arange(n_samples)) & (np.arange(n_samples) < anem_end_idx))[0])
                    return np.array(list_idxs)


                def get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, anem_to_test, all_anem_list, X_data, y_PDF_data, batch_size_desired=15000, batch_size_lims=[5000,30000]):
                    """
                    Args:
                        anem_to_train:
                        anem_to_test:
                        all_anem_list:
                        X_data:
                        y_PDF_data:
                        batch_size_desired:
                        batch_size_lims:
                    Returns: Input (X) and output (y) data, for training and testing, from given lists of anemometers to be used in the training and testing. Batch size
                    """
                    # Getting X and y training data
                    n_samples = X_data.shape[0]
                    train_idxs = get_idxs_from_anems(anem_to_train, n_samples, all_anem_list)
                    test_idxs  = get_idxs_from_anems( anem_to_test, n_samples, all_anem_list)
                    X_train = Tensor(X_data[train_idxs]).to(device)
                    X_test =  Tensor(X_data[ test_idxs]).to(device)
                    y_train = Tensor(y_PDF_data[train_idxs]).to(device)
                    y_test =  Tensor(y_PDF_data[ test_idxs]).to(device)
                    # Getting batch size (which can change the X and y data, by trimming a few data points!)
                    n_samples_train = X_train.shape[0]
                    batch_size_possibilities = np.array(sympy.divisors(n_samples_train))  # [1, 2, 4, 23, 46, 92, 4051, 8102, 16204, 93173, 186346, 372692]
                    batch_size = min(batch_size_possibilities, key=lambda x: abs(x - batch_size_desired))
                    batch_cond = batch_size_lims[0] < batch_size < batch_size_lims[1]
                    time_start = datetime.datetime.now()
                    while not batch_cond:
                        time_elapsed = datetime.datetime.now() - time_start
                        if time_elapsed.seconds > 5:
                            raise TimeoutError
                        # Removing 1 data point to assess new possibilities for the batch size (integer divisor of data size)
                        X_train = X_train[:-1,:]
                        y_train = y_train[:-1,:]
                        X_test = X_test[:-1,:]
                        y_test = y_test[:-1,:]
                        n_samples_train = X_train.shape[0]
                        batch_size_possibilities = np.array(sympy.divisors(n_samples_train))  # [1, 2, 4, 23, 46, 92, 4051, 8102, 16204, 93173, 186346, 372692]
                        batch_size = min(batch_size_possibilities, key=lambda x: abs(x - batch_size_desired))
                        batch_cond = batch_size_lims[0] < batch_size < batch_size_lims[1]
                    return X_train, y_train, X_test, y_test, batch_size


                # n_hid_layers = 2
                # n_epochs = 25
                # momentum = 0.9
                # activation_fun = torch.nn.modules.activation.ELU
                activation_fun_dict = {'ReLU': torch.nn.modules.activation.ReLU,
                                       'ELU': torch.nn.modules.activation.ELU,
                                       'LeakyReLU': torch.nn.modules.activation.LeakyReLU}
                loss_fun_dict = {'SmoothL1Loss': SmoothL1Loss(),
                                 'MSELoss': MSELoss(),
                                 'L1Loss': L1Loss(),
                                 'LogCoshLoss': LogCoshLoss()}

                def find_optimal_hp_for_each_of_my_cases(my_NN_cases, X_data, y_data, n_trials, print_loss_per_epoch=False, print_results=False, optimize_R2_of='values'):
                    hp_opt_results = []
                    for my_NN_case in my_NN_cases:
                        anem_to_train = my_NN_case['anem_to_train']
                        anem_to_test = my_NN_case['anem_to_test']
                        X_train, y_train, X_test, y_test, batch_size = get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, anem_to_test, all_anem_list, X_data, y_data)
                        # Beautiful MAGIC happening
                        def hp_opt_objective(trial):
                            weight_decay = trial.suggest_float("weight_decay", 1E-7, 1E-1, log=True)
                            lr =           trial.suggest_float("lr",          0.001, 0.8, log=True)
                            momentum = trial.suggest_float("momentum", 0., 0.95)
                            n_hid_layers = trial.suggest_int('n_hid_layers', 2, 6)
                            n_epochs = trial.suggest_int('n_epochs', 5, 50)
                            activation_fun_name = trial.suggest_categorical('activation', list(activation_fun_dict))
                            activation_fun = activation_fun_dict[activation_fun_name]
                            loss_fun_name = trial.suggest_categorical('loss', list(loss_fun_dict))
                            loss_fun = loss_fun_dict[loss_fun_name]
                            hp = {'lr': lr,
                                  'batch_size': batch_size,
                                  'weight_decay': weight_decay,
                                  'momentum': momentum,
                                  'n_epochs': n_epochs,
                                  'n_hid_layers': n_hid_layers,
                                  'activation': activation_fun,
                                  'loss': loss_fun}
                            _, R2 = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=print_loss_per_epoch, print_results=print_results, R2_of=optimize_R2_of)
                            return R2
                        while True:
                            try:  # because now and then the ANN produces an error (e.g. weights explode during learning)
                                study = optuna.create_study(direction='maximize')
                                study.optimize(hp_opt_objective, n_trials=n_trials)
                            except ValueError:
                                continue
                            break
                        hp_opt_result = {'anem_to_test': anem_to_test, 'anems_to_train': anem_to_train, 'best_params':study.best_params, 'best_value': study.best_value}
                        hp_opt_results.append(hp_opt_result)
                    return hp_opt_results

                my_NN_cases = [{'anem_to_train': ['osp1_A', 'osp2_A', 'synn_A', 'svar_A', 'neso_A'],
                                'anem_to_test': ['land_A']},
                               {'anem_to_train': ['osp1_A', 'osp2_A', 'synn_A', 'svar_A', 'land_A'],
                                'anem_to_test': ['neso_A']},
                               {'anem_to_train': ['osp2_A', 'synn_A', 'svar_A', 'land_A', 'neso_A'],
                                'anem_to_test': ['osp1_A']},
                               {'anem_to_train': ['synn_A', 'svar_A', 'land_A', 'neso_A'],
                                'anem_to_test': ['osp2_A']},
                               {'anem_to_train': ['osp1_A', 'osp2_A', 'svar_A', 'land_A', 'neso_A'],
                                'anem_to_test': ['synn_A']},
                               {'anem_to_train': ['osp1_A', 'osp2_A', 'synn_A', 'land_A', 'neso_A'],
                                'anem_to_test': ['svar_A']}
                              ]




                # # Testing trivial case. Normalization is important!
                # test_X_data = np.array([np.arange(1000000), np.arange(1000000), np.arange(1000000)]).T/1000000   * 10 - 9
                # test_y_data = (np.arange(1000000).T/1000000)[:,None]                                             * 10 - 9
                # find_optimal_hp_for_each_of_my_cases(my_NN_cases, X_data=test_X_data, y_data=test_y_data, n_trials=1, print_loss_per_epoch=True, print_results=True)
                #


                # X_data_backup = copy.deepcopy(X_data)
                # # X_data = copy.deepcopy(X_data_backup)
                # # X_data = np.delete(X_data, [0,1], axis=1) # NOT WORKING FOR THE BEAUTIFUL PLOTS THAT WILL REQUIRE THESE VALUES
                # # X_data = np.delete(X_data, np.arange(2,91), axis=1) # NOT WORKING FOR THE BEAUTIFUL PLOTS THAT WILL REQUIRE THESE VALUES
                # X_data = np.delete(X_data, 1, axis=1) # NOT WORKING FOR THE BEAUTIFUL PLOTS THAT WILL REQUIRE THESE VALUES
                # X_data = np.random.uniform(0,1,size=X_data_backup.shape)

                n_trials = 200

                if do_sector_avg:
                    # y_PDF_data
                    hp_opt_results_PDF  = find_optimal_hp_for_each_of_my_cases(my_NN_cases, X_data=X_data, y_data=y_PDF_data, n_trials=n_trials)
                else:
                    # y_data
                    hp_opt_results = find_optimal_hp_for_each_of_my_cases(my_NN_cases, X_data=X_data, y_data=y_data[:, None], n_trials=n_trials, optimize_R2_of='means')

                with open('hp_opt_results.txt', 'w') as file:
                    file.write(json.dumps(str(hp_opt_results)))  # use `json.loads` to do the reverse

                for case_idx in range(len(my_NN_cases)):
                    my_NN_case = my_NN_cases[case_idx]
                    anem_to_train = my_NN_case['anem_to_train']
                    anem_to_test = my_NN_case['anem_to_test']
                    if do_sector_avg:
                        hp_PDF = hp_opt_results_PDF[case_idx]['best_params']
                        hp['activation'] = activation_fun_dict[hp['activation']]
                        X_train, y_train, X_test, y_test, batch_size = get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, anem_to_test, all_anem_list, X_data, y_PDF_data, batch_size_desired=15000, batch_size_lims=[5000,30000])
                        hp_PDF['batch_size'] = batch_size
                        if not input_wind_data:
                            features_to_del = [3, 4] if input_weather_data else [0, 1]
                            X_train_2 = Tensor(np.delete(X_train.cpu().numpy(), features_to_del, axis=1)).to(device)
                            X_test_2  = Tensor(np.delete( X_test.cpu().numpy(), features_to_del, axis=1)).to(device)
                            y_PDF_pred, R2 = train_and_test_NN(X_train_2, y_train, X_test_2, y_test, hp=hp_PDF, print_loss_per_epoch=True, print_results=True)
                        else:
                            y_PDF_pred, R2 = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp_PDF, print_loss_per_epoch=True, print_results=True)
                        n_features = X_data_nonnorm.shape[1]
                    else:
                        hp = hp_opt_results[case_idx]['best_params']
                        if type(hp['activation']) == str:
                            hp['activation'] = activation_fun_dict[hp['activation']]
                            hp['loss'] = loss_fun_dict[hp['loss']]
                        X_train, y_train, X_test, y_test, batch_size = get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, anem_to_test, all_anem_list, X_data, y_data[:,None], batch_size_desired=15000, batch_size_lims=[5000,30000])
                        hp['batch_size'] = batch_size
                        y_pred, R2 = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=True, print_results=True)

                    # PLOT PREDICTIONS
                    # # Old:
                    # X_test_dirs_nonnorm = X_test[:, dirs_idx].cpu().numpy() * (X_maxs[dirs_idx] - X_mins[dirs_idx]) + X_mins[dirs_idx]
                    # New:
                    X_test_cos_dir = X_test[:, 3].cpu().numpy() * (X_maxs[3] - X_mins[3]) + X_mins[3]
                    X_test_sin_dir = X_test[:, 4].cpu().numpy() * (X_maxs[4] - X_mins[4]) + X_mins[4]
                    X_test_dirs_nonnorm = np.rad2deg([convert_angle_to_0_2pi_interval(i, input_and_output_in_degrees=False) for i in np.arctan2(X_test_sin_dir, X_test_cos_dir)])
                    y_test_nonnorm = y_test.cpu().numpy() * (y_max - y_min) + y_min
                    anem_idx = np.where(all_anem_list == my_NN_cases[case_idx]['anem_to_test'][0])[0][0]
                    anem_slice = slice(start_idxs_of_each_anem_2[anem_idx], start_idxs_of_each_anem_2[anem_idx + 1])

                    plt.figure(figsize=(5.5,2.3), dpi=400)
                    # plt.title(nice_str_dict[my_NN_cases[case_idx]['anem_to_test'][0]] + '.  $R^2='+ str(np.round(R2.item(), 2))+'$')
                    plt.title('10-min values of $I_u$ ($R^2=' + str(np.round(R2.item(), 2)) + '$);')
                    ########## ATTENTION: I SHOULDN'T BE USING y_data, but instead y_test since one data point was removed to find a divisor for the batch size.
                    # CERTAIN TRASH: plt.scatter(X_dirs[anem_slice], y_data[anem_slice] * (y_max-y_min) + y_min, s=0.01, alpha=0.2, c='black', label='Measured')
                    # PERHAPS TRASH: plt.scatter(X_dirs[anem_slice], y_PDF_data[anem_slice] * (y_PDF_maxs-y_PDF_mins) + y_PDF_mins, s=0.01, alpha=0.2, c='blue', label='Measured Mean')
                    plt.scatter(X_test_dirs_nonnorm, y_test_nonnorm, s=0.01, alpha=0.2, c='black', label='Measured')
                    if do_sector_avg:
                        plt.scatter(X_test_dirs_nonnorm, y_PDF_pred.cpu().numpy() * (y_PDF_maxs - y_PDF_mins) + y_PDF_mins, s=0.01, alpha=0.2, c='orange', label='Predicted Mean')
                    else:
                        y_pred_nonnorm = y_pred.cpu().numpy() * (y_max - y_min) + y_min
                        plt.scatter(X_test_dirs_nonnorm, y_pred_nonnorm, s=0.01, alpha=0.2, c='orange', label='Predicted')
                    plt.legend(markerscale=30., loc=1)
                    plt.ylabel('$I_u$')
                    # plt.xlabel('Wind from direction [\N{DEGREE SIGN}]')
                    plt.xlim([0, 360])
                    plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
                    ax = plt.gca()
                    ax.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
                    plt.ylim([0, 0.7])
                    plt.tight_layout(pad=0.05)
                    plt.savefig(os.path.join(os.getcwd(), 'plots', f'E_{case_idx}_{y_data_type}_{anem_to_test[0]}_Umin_{U_min}_Rough-{str(add_roughness_input)[0]}_Weather-{str(input_weather_data)[0]}_1Profile-{str(only_1_elevation_profile)[0]}_Sector-{dir_sector_amp}_Avg-{str(do_sector_avg)[0]}.png'))
                    # plt.show()

                    # PLOT MEANS OF PREDICTIONS
                    if not do_sector_avg:
                        X_dir_sectors = np.searchsorted(dir_sectors, X_test_dirs_nonnorm, side='right') - 1  # groups all the measured directions into sectors
                        y_test_mean = np.array([np.mean(y_test_nonnorm[np.where(X_dir_sectors == d)[0]]) for d in dir_sectors])
                        y_pred_mean = np.array([np.mean(y_pred_nonnorm[np.where(X_dir_sectors == d)[0]]) for d in dir_sectors])
                        # Removing NaN from the data after organizing it into a dataframe
                        # all_mean_data = pd.concat([pd.DataFrame(dir_sectors), pd.DataFrame(y_test_mean), pd.DataFrame(y_pred_mean)], axis=1).dropna(axis=0, how='any').reset_index(drop=True)
                        all_mean_data = pd.DataFrame({'dir_sectors':dir_sectors, 'y_test_mean':y_test_mean, 'y_pred_mean':y_pred_mean}).dropna(axis=0, how='any').reset_index(drop=True)
                        R2_of_means = r2_score(all_mean_data['y_test_mean'], all_mean_data['y_pred_mean'])  # r2_score would give error if there was a NaN.
                        plt.figure(figsize=(5.5,2.3), dpi=400)
                        # plt.title(nice_str_dict[my_NN_cases[case_idx]['anem_to_test'][0]] + '.  $R^2='+ str(np.round(R2_of_means, 2))+'$')
                        plt.title('1-deg-wide means of $I_u$ ($R^2=' +str(np.round(R2_of_means, 2)) + '$).')
                        # plt.scatter(X_test_dirs_nonnorm, y_test_nonnorm, s=0.01, alpha=0.2, c='black') #, label='Measured')
                        plt.scatter(dir_sectors, y_test_mean, s=3, alpha=0.8, c='black', label='Measured means')
                        plt.scatter(dir_sectors, y_pred_mean, s=3, alpha=0.8, c='darkorange', label='Predicted means')
                        plt.legend(markerscale=2.5, loc=1)
                        plt.ylabel('$\overline{I_u}$')
                        # plt.xlabel('Wind from direction [\N{DEGREE SIGN}]')
                        plt.xlim([0, 360])
                        plt.xticks([0,45,90,135,180,225,270,315,360])
                        ax = plt.gca()
                        ax.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
                        plt.ylim([0, 0.7])
                        plt.tight_layout(pad=0.05)
                        plt.savefig(os.path.join(os.getcwd(), 'plots', f'E_{case_idx}_mean_{y_data_type}_{anem_to_test[0]}_Umin_{U_min}_Rough-{str(add_roughness_input)[0]}_Weather-{str(input_weather_data)[0]}_1Profile-{str(only_1_elevation_profile)[0]}_Sector-{dir_sector_amp}_Avg-{str(do_sector_avg)[0]}.png'))
                        # plt.show()
    return None
































































#
#
# # TRASH BELLOW @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# # REMOVING MEANS AND DIRECTIONS FROM THE INPUT DATA
# # X_data = np.delete(X_data, [0,1], axis=1)
#
#
#
#
# # # Getting values to predict and predicted values
# # hp = {'lr':1.7E-1, 'batch_size':batch_size, 'weight_decay':1.85E-3, 'momentum':0.9, 'n_epochs':35,
# #       'n_hid_layers':3, 'activation':torch.nn.ELU, 'loss':MSELoss()}
# # y_pred, _ = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=True)
#
#
#
# # De-normalizing
# dir_test = X_test[:,1].cpu().numpy() * X_maxs[1]
# y_test_nonnorm  = np.squeeze(y_test.cpu().numpy()) * y_PDF_maxs
# y_pred_nonnorm = np.squeeze(y_pred.cpu().numpy()) * y_PDF_maxs
#
# # From PREDICTED Weibull to PREDICTED data
# y_pred_samples = stats.exponweib.rvs(1, y_pred_nonnorm[:,0], 0, y_pred_nonnorm[:,1], size=len(y_pred_nonnorm))
#
#
#
#
#
#
#
#
#
#
# # Plotting PREDICTED data by anemometer
# # density_scatter(np.deg2rad(X_data_nonnorm[test_idxs,1]), y_pred_samples)
# # plt.title(f'Synthetic data from predicted Weibull params. \n {anem_to_test}')
# # plt.ylim([0, 4])
# # ax.text(np.deg2rad(18), 4.4, '$\sigma(u)\/[m/s]$')
# # plt.tight_layout()
# # plt.savefig(os.path.join(os.getcwd(), 'plots', f'predicted_synth_std_u_{anem_to_test}.png'))
# # plt.show()
#
#
# x = np.deg2rad(X_data_nonnorm[test_idxs,1])
# y1 = y_data_nonnorm[test_idxs]
# y2 = y_pred_samples
#
# data1, x1_e, y1_e = np.histogram2d(np.sin(x) * y1, np.cos(x) * y1, bins=30, density=True)
# data2, x2_e, y2_e = np.histogram2d(np.sin(x) * y2, np.cos(x) * y2, bins=[x1_e, y1_e], density=True)
# x1_i = 0.5 * (x1_e[1:] + x1_e[:-1])
# y1_i = 0.5 * (y1_e[1:] + y1_e[:-1])
# x2_i = 0.5 * (x2_e[1:] + x2_e[:-1])
# y2_i = 0.5 * (y2_e[1:] + y2_e[:-1])
# z1 = interpn((x1_i, y1_i), data1, np.vstack([np.sin(x) * y1, np.cos(x) * y1]).T, method="splinef2d", bounds_error=False)
# z2 = interpn((x2_i, y2_i), data2, np.vstack([np.sin(x) * y2, np.cos(x) * y2]).T, method="splinef2d", bounds_error=False)
# z1[np.where(np.isnan(z1))] = 0.0  # To be sure to plot all data
# z2[np.where(np.isnan(z2))] = 0.0  # To be sure to plot all data
# idx1 = z1.argsort()  # Sort the points by density, so that the densest points are plotted last
# idx2 = z2.argsort()  # Sort the points by density, so that the densest points are plotted last
# x1, y1, z1 = x[idx1], y1[idx1], z1[idx1]
# x2, y2, z2 = x[idx2], y2[idx2], z2[idx2]
# # 1st plot
# fig, ax = plt.subplots(figsize=(8, 6), dpi=300, subplot_kw={'projection': 'polar'})
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.scatter(x1, y1, c=z1, s=1, alpha=0.3)
# norm = Normalize(vmin=np.min([z1, z2]), vmax=np.max([z1, z2]))
# cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
# cbar.ax.set_ylabel('Density')
# plt.title(f'Measured. {anem_to_test}')
# plt.ylim([0, 4])
# ax.text(np.deg2rad(20), 4.4, '$\sigma(u)\/[m/s]$')
# plt.tight_layout()
# plt.savefig(os.path.join(os.getcwd(), 'plots', f'__measured_std_u_{anem_to_test}.png'))
# plt.show()
# # 2nd plot
# fig, ax = plt.subplots(figsize=(8, 6), dpi=300, subplot_kw={'projection': 'polar'})
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.scatter(x2, y2, c=z2, s=1, alpha=0.3)
# norm = Normalize(vmin=np.min([z1, z2]), vmax=np.max([z1, z2]))
# cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
# cbar.ax.set_ylabel('Density')
# plt.title(f'Synthetic from predicted Weibull params. {anem_to_test}')
# plt.ylim([0, 4])
# ax.text(np.deg2rad(20), 4.4, '$\sigma(u)\/[m/s]$')
# plt.tight_layout()
# plt.savefig(os.path.join(os.getcwd(), 'plots', f'PREDICTED_synth_std_u_{anem_to_test}.png'))
# plt.show()
#
#
#
#
# # plt.plot(y_test_nonnorm[4500:5000,1], label = 'Test')
# # plt.plot(y_pred_nonnorm[4500:5000,1], label = 'Pred')
# # plt.legend()
# # plt.show()
#
#
#
# ##################################################################################################################
# # WEIBULL PARAMS - TRAINING FROM 18 ALTERNATE-10-DEG-WIDE-WIND-SECTORS AND TESTING THE REMAINING 18 SECTORS, AT EACH ANEMOMETER
# ##################################################################################################################
# # todo: copied from the std_u section. Needs to be adapted to Weibull params
#
# # Remove the direction, to be extra certain that the NN doesn't "cheat"
# # X_data = np.delete(X_data, 1, axis=1) # NOT WORKING FOR THE BEAUTIFUL PLOTS THAT WILL REQUIRE THESE VALUES
#
# # Separating training and testing data
# # train_angle_domain = [[x, x+22.5] for x in np.arange(0, 360, 45)]  # in degrees
# # test_angle_domain  = [[x+22.5, x+45] for x in np.arange(0, 360, 45)]  # in degrees
# train_angle_domain = [[x, x+20] for x in np.arange(0, 360, 60)]  # in degrees
# test_angle_domain  = [[x+35, x+45] for x in np.arange(0, 360, 60)]  # in degrees
# train_bools = np.logical_or.reduce([(a[0]<X_data_nonnorm[:,1]) & (X_data_nonnorm[:,1]<a[1]) for a in train_angle_domain])  # https://stackoverflow.com/questions/20528328/numpy-logical-or-for-more-than-two-arguments
# test_bools =  np.logical_or.reduce([(a[0]<X_data_nonnorm[:,1]) & (X_data_nonnorm[:,1]<a[1]) for a in test_angle_domain])
# X_train = Tensor(X_data[train_bools]).to(device)
# y_train = Tensor(y_PDF_data[train_bools]).to(device)
# X_test =  Tensor(X_data[test_bools]).to(device)
# y_test =  Tensor(y_PDF_data[test_bools]).to(device)
#
# n_samples_train = X_train.shape[0]
# batch_size_possibilities = sympy.divisors(n_samples_train)  # [1, 2, 4, 23, 46, 92, 4051, 8102, 16204, 93173, 186346, 372692]
#
#
# # Getting values to predict and predicted values
# hp = {'lr':1E-1, 'batch_size':6329, 'weight_decay':1E-4, 'momentum':0.9, 'n_epochs':25, 'n_hid_layers':1, 'activation':torch.nn.ReLU, 'loss':MSELoss()}
# y_pred = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=True)
#
# # Choosing only the results of a given anemometer (e.g. svar -> Svarvahelleholmen)
# anem_train = np.searchsorted(start_idxs_of_each_anem, np.where(train_bools)[0], side='right') - 1  # to which anemometer, (indexed from all_anem_list), does each test sample belong to
# anem_test =  np.searchsorted(start_idxs_of_each_anem, np.where(test_bools )[0], side='right') - 1
# train_idxs_svar = np.where(anem_train == np.where(all_anem_list == 'svar_A')[0])[0]
# test_idxs_svar  = np.where(anem_test  == np.where(all_anem_list == 'svar_A')[0])[0]
# X_train_svar = X_train[train_idxs_svar].cpu().numpy()
# y_train_svar = np.squeeze(y_train[train_idxs_svar].cpu().numpy())  # simply converting to numpy and removing the empty dimension of the shape (n_train_samples,1)
# X_test_svar = X_test[test_idxs_svar].cpu().numpy()
# y_test_svar = np.squeeze(y_test[test_idxs_svar].cpu().numpy())  # simply converting to numpy and removing the empty dimension of the shape (n_test_samples,1)
# y_pred_svar = np.squeeze(y_pred[test_idxs_svar].cpu().numpy())  # simply converting to numpy and removing the empty dimension of the shape (n_test_samples,1)
#
# # De-normalizing
# dir_train_svar = X_train_svar[:,1] * X_maxs[1]
# dir_test_svar = X_test_svar[:,1] * X_maxs[1]
# std_u_train_svar = y_train_svar * y_max
# std_u_test_svar  = y_test_svar * y_max
# std_u_pred_svar = y_pred_svar * y_max
#
# # Organizing the results into sectors
# train_sector_bools = [(a[0]<dir_train_svar) & (dir_train_svar<a[1]) for a in train_angle_domain]
# test_sector_bools  = [(a[0]<dir_test_svar)  & (dir_test_svar<a[1])  for a in test_angle_domain]
# train_sector_idxs = [np.where(train_sector_bools[i])[0] for i in range(len(train_sector_bools))]
# test_sector_idxs  = [np.where(test_sector_bools[i] )[0] for i in range(len(test_sector_bools ))]
# dir_means_train_per_sector_svar =   np.array([np.mean(dir_train_svar[l]) for l in train_sector_idxs])
# dir_means_test_per_sector_svar  =   np.array([np.mean(dir_test_svar[l] ) for l in test_sector_idxs ])
# std_u_means_train_per_sector_svar = np.array([np.mean(std_u_train_svar[l]) for l in train_sector_idxs])
# std_u_means_test_per_sector_svar  = np.array([np.mean(std_u_test_svar[l]) for l in test_sector_idxs])
# std_u_means_pred_per_sector_svar = np.array([np.mean(std_u_pred_svar[l]) for l in test_sector_idxs])
# std_u_std_train_per_sector_svar = np.array([np.std(std_u_train_svar[l]) for l in train_sector_idxs])
# std_u_std_test_per_sector_svar  = np.array([np.std(std_u_test_svar[l] ) for l in test_sector_idxs ])
# std_u_std_pred_per_sector_svar =   np.array([np.std( std_u_pred_svar[l]) for l in test_sector_idxs])
#
# # Plotting beautiful plots
# fig = plt.figure(figsize=(8,6), dpi=400)
# ax = fig.add_subplot(projection='polar')
# plt.title('Anemometer at Svarvahelleholmen (Z = 48 m)\n')
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.scatter( np.deg2rad(dir_train_svar), std_u_train_svar, s=1, alpha=0.6, c='lightgreen', label='Training data')
# ax.scatter( np.deg2rad(dir_test_svar) , std_u_test_svar , s=1, alpha=0.6, c='skyblue', label='Testing data')
# ax.errorbar(np.deg2rad(dir_means_train_per_sector_svar), std_u_means_train_per_sector_svar, std_u_std_train_per_sector_svar, c='forestgreen', elinewidth=3, alpha=0.9, fmt='.', label='$\sigma(train)$')
# ax.errorbar(np.deg2rad(dir_means_test_per_sector_svar) , std_u_means_test_per_sector_svar , std_u_std_test_per_sector_svar , c='dodgerblue', elinewidth=4, alpha=0.8, fmt='o', label='$\sigma(test)$')
# ax.errorbar(np.deg2rad(dir_means_test_per_sector_svar) , std_u_means_pred_per_sector_svar , std_u_std_pred_per_sector_svar , c='orange', elinewidth=2, alpha=0.9, fmt='.', label='Prediction', zorder=5)
# handles, labels = ax.get_legend_handles_labels()
# plt.ylim((None,4))
# ax.text(np.deg2rad(18), 4.4, '$\sigma(u)\/[m/s]$')
# plt.savefig(os.path.join(os.getcwd(), 'plots', 'std_u_Svar.png'))
# plt.show()
# fig = plt.figure(figsize=(2, 1.6), dpi=400)
# plt.axis('off')
# plt.legend(handles, labels)
# plt.savefig(os.path.join(os.getcwd(), 'plots', 'std_u_Svar_legend.png'))
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # TRASH:
# # OLD VERSION USING curve_fit and my own weibull function. It didn't work properly
# # n_bins = 100
# # hist = np.histogram(y_data_nonnorm[idxs_360dir], bins=n_bins, density=True)
# # hist_x = [(a+b)/2 for a,b in zip(hist[1][:-1], hist[1][1:])]
# # hist_y = hist[0]
# # try:
# #     popt, pcov = curve_fit(weibull_PDF, hist_x, hist_y)  # param. optimal values, param estimated covariance
# # except:
# #     print(f'exception at anem {anem_idx} for dir {d}')
# #     popt, pcov = curve_fit(weibull_PDF, hist_x, hist_y, bounds=(1E-5, 10))
# # plt.plot(weibull_x, weibull_PDF(weibull_x, *popt), label='my_func')
# # plt.plot(weibull_x, weibull_PDF(weibull_x, *popt2), label='weibfit')
#
#
# # PLOTTING HISTOGRAMS AND PDF FITS
# anem_idx = 3
# d = 45
# anem_slice = slice(start_idxs_of_each_anem_2[anem_idx], start_idxs_of_each_anem_2[anem_idx + 1])
# idxs_360dir = np.where(X_dir_sectors[anem_slice] == d)[0]
# if len(idxs_360dir) > 10 :  # minimum number of datapoints, otherwise -> np.nan
#     params_weib    = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir], floc=0, f0 = 1)  # params = a(alpha), c(shape==k), loc, scale(lambda)
#     params_expweib = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir], floc=0)
#     params_expweib2 = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir])
#     params_rayleigh = stats.rayleigh.fit(y_data_nonnorm[anem_slice][idxs_360dir], floc=0)
#
#     plt.figure()
#     plt.title(f'Wind from {dir_sectors[d]}$\u00B0C$ to {dir_sectors[d+1]}$\u00B0C$. Anemometer: "{all_anem_list[anem_idx]}"')
#     weibull_x = np.linspace(0, 5, 100)
#     plt.plot(weibull_x, stats.exponweib.pdf(weibull_x, *params_weib   ), label='Weibull fit (2 parameters)', lw=3, alpha=0.8)
#     plt.plot(weibull_x, stats.exponweib.pdf(weibull_x, *params_expweib), label='Exp. Weibull fit (3 parameters)', lw=3, alpha=0.8)
#     plt.plot(weibull_x, stats.exponweib.pdf(weibull_x, *params_expweib2), label='Exp. Weibull fit 2 (4 parameters)', lw=3, alpha=0.8)
#     plt.hist(y_data_nonnorm[anem_slice][idxs_360dir], bins=50, density=True, label='Normalized histogram of $\sigma(u)$', alpha=0.5)
#     plt.xlim([0,5])
#     plt.legend()
#     plt.show()
#
#
#
#
# # Testing visually manually that the two new masts are correctly oriented
# import json
# with open(r'C:\Users\bercos\PycharmProjects\Metocean\processed_data\00-10-00_stats_2015-11-01_00-00-00_2015-12-01_00-00-00', "r") as json_file:
#     pro_data_1 = json.load(json_file)
# with open(r'C:\Users\bercos\PycharmProjects\Metocean\processed_data_2\00-10-00_stats_2015-11-01_00-00-00_2015-12-01_00-00-00', "r") as json_file:
#     pro_data_2 = json.load(json_file)
#
# from find_storms import organized_dataframes_of_storms
# min10_df_all_means, min10_df_all_dirs, min10_df_all_Iu, min10_df_all_Iv, min10_df_all_Iw, min10_df_all_avail = organized_dataframes_of_storms(foldername='processed_data_2',
#                                                                                                                                               compiled_fname='00-10-00_stats_2015-11-01_00-00-00_2015-12-01_00-00-00',
#                                                                                                                                               mast_list=['land','neso'])
#
#
#
#
# # TRASH PLOTS
# # plt.figure(figsize=(20,5))
# # plt.plot(df_air_temp['temperature_1'].dropna(),     alpha=0.2, lw=0.5)
# # plt.plot(df_air_temp['temperature_2'].dropna(),     alpha=0.2, lw=0.5)
# # # plt.plot(df_air_temp['temperature_3'].dropna(),     alpha=0.2, lw=0.5)
# # plt.plot(df_air_temp['temperature_final'].dropna(), alpha=0.5, lw=0.5, label='smooth')
# # plt.legend()
# # plt.ylim([0,20])
# # plt.show()


