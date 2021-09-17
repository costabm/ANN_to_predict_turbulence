"""
This study uses Machine Learning to predict turbulence at a new location with a given topography, where the learning data
consists only of turbulence data and topography data at other nearby locations.
THIS VERSION USES ALL 6 ANEMOMETERS TO ESTIMATE TURBULENCE AT DIFFERENT POINTS ALONG THE BJØRNAFJORD
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
import matplotlib
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
from orography import synn_EN_33, svar_EN_33, osp1_EN_33, osp2_EN_33, land_EN_33, neso_EN_33
from create_minigrid_data_from_raw_WRF_500_data import bridge_WRF_nodes_coor_func, rad, deg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'cuda' or 'cpu'. 'cuda' doesn't seem to be working...

########################################################################################################################################################################################################
# PREDICTING TURBULENCE ALONG THE BJØRNAFJORD:
########################################################################################################################################################################################################
# Getting bridge nodes
n_bridge_nodes = 11
coors_bridge = bridge_WRF_nodes_coor_func(n_bridge_WRF_nodes = n_bridge_nodes, unit='deg')
# to convert from Lat/Lon to UTM-33, use the website: https://www.kartverket.no/en/on-land/posisjon/transformere-koordinater-enkeltvis
assert n_bridge_nodes == 11, "n_bridge_nodes needs to be 11. Otherwise, you need to manually create the new array of bridge node coordinates, using the website above"
coors_bridge = np.array([[-34449.260, 6699999.046],
                         [-34244.818, 6700380.872],
                         [-34057.265, 6700792.767],
                         [-33888.469, 6701230.609],
                         [-33740.109, 6701690.024],
                         [-33613.662, 6702166.417],
                         [-33510.378, 6702655.026],
                         [-33431.282, 6703150.969],
                         [-33377.153, 6703649.290],
                         [-33348.522, 6704145.006],
                         [-33345.665, 6704633.167]])
bj_pts_EN_33 = {}
bj_pts_nice_str = {}
for i in range(n_bridge_nodes):
    bj_pts_EN_33['bj'+f'{i+1:02}']=coors_bridge[i].tolist()
    bj_pts_nice_str['bj'+f'{i+1:02}']='Bjørnafjord P'+str(i+1)


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


def example_of_elevation_profile_at_given_point_dir_dist(point_1, direction_deg=160, step_distance=False, total_distance=False,
                                                         list_of_distances=[i*(5.+5.*i) for i in range(45)], plot=True):
    point_2 = get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=direction_deg, distance=list_of_distances[-1] if list_of_distances else total_distance)
    dists, heights = elevation_profile_generator(point_1=point_1, point_2=point_2, step_distance=step_distance, list_of_distances=list_of_distances)
    if plot:
        plot_elevation_profile(point_1=point_1, point_2=point_2, step_distance=step_distance, list_of_distances=list_of_distances)
    return dists, heights

example_of_elevation_profile_at_given_point_dir_dist(point_1=coors_bridge[5], direction_deg=160, step_distance=False, total_distance=False,
                                                   list_of_distances=[i*(5.+5.*i) for i in range(45)], plot=True)


def slopes_in_deg_from_dists_and_heights(dists, heights):
    delta_dists = dists[1:] - dists[:-1]
    delta_heights = heights[1:] - heights[:-1]
    return np.rad2deg(np.arctan(delta_heights/delta_dists))


def plot_topography_per_point(list_of_degs = list(range(360)), list_of_distances=[i*(5.+5.*i) for i in range(45)]):
    for pt, pt_coor in bj_pts_EN_33.items():
        dists_all_dirs = []
        heights_all_dirs = []
        degs = []
        for d in list_of_degs:
            dists, heights = example_of_elevation_profile_at_given_point_dir_dist(point_1=pt_coor, direction_deg=d, step_distance=False, total_distance=False, list_of_distances=list_of_distances, plot=False)
            dists_all_dirs.append(dists)
            heights_all_dirs.append(heights)
            degs.append(d)
        degs = np.array(degs)
        dists_all_dirs = np.array(dists_all_dirs)
        heights_all_dirs = np.array(heights_all_dirs)
        cmap = copy.copy(plt.get_cmap('magma_r'))
        heights_all_dirs = np.ma.masked_where(heights_all_dirs == 0, heights_all_dirs)  # set mask where height is 0, to be converted to another color
        fig, (ax, cax) = plt.subplots(nrows=2, figsize=(0.9*5.5,0.9*(2.3+0.5)), dpi=400/0.9, gridspec_kw={"height_ratios": [1, 0.05]})
        im = ax.pcolormesh(degs, dists_all_dirs[0], heights_all_dirs.T, cmap=cmap, shading='auto', vmin = 0., vmax = 800.)
        ax.set_title('Upstream topography at ' + bj_pts_nice_str[pt])
        ax.patch.set_color('skyblue')
        ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
        ax.set_yticks([0,2000,4000,6000,8000,10000])
        ax.set_yticklabels([0,2,4,6,8,10])
        ax.set_xlabel('Wind from direction [\N{DEGREE SIGN}]')
        ax.set_ylabel('Upstream dist. [km]')
        # cax.set_xlabel('test')
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label('Height above sea level [m]')
        plt.tight_layout(h_pad=0.05, w_pad=0.2)
        plt.subplots_adjust(hspace=0.9)
        # plt.savefig(os.path.join(os.getcwd(), 'plots', f'Topography_per_point-{pt}.png'))
        plt.show()
    return None

plot_topography_per_point(list_of_degs = list(range(360)), list_of_distances=[i*(5.+5.*i) for i in range(45)])


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


# Getting the transition zones as described in NS-EN 1991-1-4:2005/NA:2009, NA.4.3.2(2)
def get_one_roughness_transition_zone(dists, heights):
    """
    When using the Eurocode to predict turbulence, two zones with different terrain roughnessess can be chosen. The transition zone is the upstream distance that divides these two categories.
    My own classification algorithm is smple: Try each distance and choose the one with less misclassifications (ground vs sea) (ascending vs descending roughness with the wind)
    """
    roughs = heights.astype(bool).astype(float)
    assert len(dists) == len(roughs)
    assert np.unique(roughs) in np.array([0,1])  # roughs need to be made of 0's and 1's only
    n_points = len(dists)
    roughs_inv = np.abs(roughs - 1)  # inverted vector: 0's become 1's and 1's become 0's.
    n_wrong = n_points  # start with all wrong
    for i in range(n_points):
        asc_n_wrong_near = np.sum(roughs_inv[:i])
        asc_n_wrong_far  = np.sum(roughs[i:])
        des_n_wrong_near = np.sum(roughs[:i])
        des_n_wrong_far  = np.sum(roughs_inv[i:])
        if (asc_n_wrong_near + asc_n_wrong_far) < n_wrong:  # Descending roughness with the wind (Ascending roughness with upstream dist: 1 to 0)
            n_wrong = asc_n_wrong_near + asc_n_wrong_far
            transition_idx = i
            direction = 'ascending'
        if (des_n_wrong_near + des_n_wrong_far) < n_wrong:  # Ascending roughness with the wind (Descending roughness with upstream dist: 1 to 0)
            n_wrong = des_n_wrong_near + des_n_wrong_far
            transition_idx = i
            direction = 'descending'
    if transition_idx == 0:
        transition_dist = 1  # 1 meter, instead of 0 meters, to prevent errors in dividing by 0
    else:
        transition_dist = (dists[transition_idx-1] + dists[transition_idx]) / 2
    return transition_idx, transition_dist, direction


def get_all_roughness_transition_zones(step_distance=False, total_distance=False, list_of_distances=[i*(5.+5.*i) for i in range(45)]):
    transition_zones = {}
    for anem, anem_coor in anem_EN_33.items():
        transition_zones[anem] = []
        for d in list(range(360)):
            dists, heights = example_of_elevation_profile_at_given_point_dir_dist(point_1=anem_coor, direction_deg=d, step_distance=step_distance, total_distance=total_distance, list_of_distances=list_of_distances, plot=False)
            transition_zones[anem].append(get_one_roughness_transition_zone(dists, heights))
    return transition_zones


def plot_roughness_transitions_per_anem(list_of_degs = list(range(360)), step_distance=False, total_distance=False, list_of_distances=[i*(5.+5.*i) for i in range(45)]):
    transition_zones = get_all_roughness_transition_zones()
    from orography import synn_EN_33, svar_EN_33, osp1_EN_33, osp2_EN_33, land_EN_33, neso_EN_33
    anem_EN_33 = {'synn':synn_EN_33, 'svar':svar_EN_33, 'osp1':osp1_EN_33, 'osp2':osp2_EN_33, 'land':land_EN_33, 'neso':neso_EN_33}
    for anem, anem_coor in anem_EN_33.items():
        dists_all_dirs = []
        roughs_all_dirs = []
        degs = []
        for d in list_of_degs:
            dists, heights = example_of_elevation_profile_at_given_point_dir_dist(point_1=anem_coor, direction_deg=d, step_distance=step_distance, total_distance=total_distance, list_of_distances=list_of_distances, plot=False)
            roughs = heights.astype(bool).astype(float)
            dists_all_dirs.append(dists)
            roughs_all_dirs.append(roughs)
            degs.append(d)
        degs = np.array(degs)
        dists_all_dirs = np.array(dists_all_dirs)
        roughs_all_dirs = np.array(roughs_all_dirs)
        fig, ax = plt.subplots(figsize=(5.5,2.3+0.5), dpi=400)
        ax.pcolormesh(degs, dists_all_dirs[0], roughs_all_dirs.T, cmap=matplotlib.colors.ListedColormap(['skyblue', 'navajowhite']), shading='auto') #, vmin = 0., vmax = 1.)
        xB = np.array([item[1] for item in transition_zones[anem]])
        asc_desc = np.array([item[2] for item in transition_zones[anem]])
        asc_idxs = np.where(asc_desc=='ascending')
        des_idxs = np.where(asc_desc=='descending')
        ax.scatter(degs[asc_idxs], xB[asc_idxs], s=1, alpha=0.4, color='red', label='ascending')
        ax.scatter(degs[des_idxs], xB[des_idxs], s=1, alpha=0.4, color='green', label='descending')
        ax.set_title(nice_str_dict[anem+'_A']+': '+'Upstream topography;')
        ax.patch.set_color('skyblue')
        ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
        ax.set_yticks([0,2000,4000,6000,8000,10000])
        ax.set_yticklabels([0,2,4,6,8,10])
        ax.set_xlabel('Wind from direction [\N{DEGREE SIGN}]')
        ax.set_ylabel('Upstream distance [km]')
        # cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        # cbar.set_label('Height above sea level [m]')
        plt.legend(loc=1)
        plt.tight_layout(pad=0.05)
        # plt.savefig(os.path.join(os.getcwd(), 'plots', f'Rough_transitions_per_anem-{anem}.png'))
        plt.show()
    return None


def get_all_Iu_with_eurocode():
    transition_zones = get_all_roughness_transition_zones()
    z = 48
    kI = 1.0
    c0 = 1.0
    z0_sea = 0.003
    z0_gro = 0.3
    z0_II = 0.05
    kr_sea = 0.19 * (z0_sea/z0_II)**0.07
    kr_gro = 0.19 * (z0_gro/z0_II)**0.07
    cr_sea = kr_sea * np.log(z/z0_sea)
    cr_gro = kr_gro * np.log(z/z0_gro)
    c_season = 1.0
    c_alt = 1.0  # 1.0 because H < H0, where H0 is 900m for Rogaland
    c_prob = 1.0
    Iu = {}
    for pt in bj_pts_EN_33.keys():
        Iu[pt] = []
        xB_all_dirs = [item[1] for item in transition_zones[pt]]
        asc_desc = [item[2] for item in transition_zones[pt]]
        for d in range(360):
            if 360 - 45 / 2 < d or d < 0 + 45 / 2:  # N
                c_dir = 0.9
            elif 45 - 45 / 2 < d < 45 + 45 / 2:  # NØ
                c_dir = 0.6
            elif 90 - 45 / 2 < d < 90 + 45 / 2:  # Ø
                c_dir = 0.8
            elif 135 - 45 / 2 < d < 135 + 45 / 2:  # Ø
                c_dir = 0.9
            else:
                c_dir = 1.0
            if 'osp' in pt:
                vb0 = 28  # Kommune: Austevoll 28m/s (Osp1 and Osp2)
            else:
                vb0 = 26  # Kommune: Tysnes 26m/s (Svar, Land, Neso); Os (Old kommune) 26 m/s (Synn)
            vb = c_dir * c_season * c_alt * c_prob * vb0
            vm_sea = cr_sea * c0 * vb
            vm_gro = cr_gro * c0 * vb
            Iu_sea = kI / (c0 * np.log(z / z0_sea))
            Iu_gro = kI / (c0 * np.log(z / z0_gro))
            xB = xB_all_dirs[d] / 1000  # in kilometers
            if asc_desc[d]=='ascending':
                n = 3
                IuA = Iu_sea
                IuB = Iu_gro
                vmA = vm_sea
                vmB = vm_gro
                cS = 10 ** (-0.04 * n * np.log10(xB/10))
                denominator = min(vmB * cS , vmA)
            elif asc_desc[d]=='descending':
                n = -3
                IuA = Iu_gro
                IuB = Iu_sea
                vmA = vm_gro
                vmB = vm_sea
                cS = 2 - 10 ** (-0.04 * abs(n) * np.log10(xB/10))
                denominator = max(vmB * cS , vmA)
            numerator = IuA * vmA * (1-xB/10) + IuB * vmB * xB/10
            Iu[pt].append(numerator / denominator)
    return Iu


Iu_EN = get_all_Iu_with_eurocode()

def generate_new_data(U_min):
    ##################################################################
    # # Getting the data for first time
    X_data_nonnorm, y_data_nonnorm, all_anem_list, start_idxs_of_each_anem, ts_data, other_data = get_all_10_min_data_at_z_48m(U_min=U_min) ##  takes a few minutes...
    # # # ##################################################################
    # # # # Saving data
    data_path = os.path.join(os.getcwd(), 'processed_data_for_ML', f'X_y_ML_ready_data_Umin_{U_min}_masts_6_new_dirs_w_ts')
    np.savez_compressed(data_path, X=X_data_nonnorm, y=y_data_nonnorm, m=all_anem_list, i=start_idxs_of_each_anem, t=np.array(ts_data, dtype=object)) # o=other_data)
    with open(data_path+'_other_data.txt', 'w') as outfile:
        json.dump(other_data, outfile)
    #################################################################
    return None


def predict_mean_turbulence_with_ML(n_trials):
    # Do the same as before, but using only topographic data, and mean Iu! The total number of samples will be greatly reduced, to only 360*6!
    result_name_tag = 'A'  # add this to the plot names and hp_opt text file
    U_min = 5
    dir_sector_amp = 1

    # Loading data already saved
    data_path = os.path.join(os.getcwd(), 'processed_data_for_ML', f'X_y_ML_ready_data_Umin_{U_min}_masts_6_new_dirs_w_ts')
    loaded_data = np.load(data_path + '.npz')
    loaded_data.allow_pickle = True
    X_data_nonnorm =          loaded_data['X']
    y_data_nonnorm =          loaded_data['y']
    all_anem_list =           loaded_data['m']
    start_idxs_of_each_anem = loaded_data['i']
    X_ts =                    loaded_data['t']
    print(f'Number of features {X_data_nonnorm.shape[1]}: U(1)+Dir(1)+WindwardHeightsMiddleCone(45)+MeanWindWard(44)+STDWindWard(44)+MeanLeeWard(10)+STDLeeWard(9)+MeanSideWard(10)+STDSideWard(9)')
    n_samples = len(y_data_nonnorm)
    dir_sectors = np.arange(0, 360, dir_sector_amp)
    n_sectors = len(dir_sectors)
    idxs_of_each_anem = np.append(start_idxs_of_each_anem, n_samples)

    def get_sect_mean_data():  # get Sectoral mean data
        df_sect_means = {}
        df_mins_maxs = pd.DataFrame()
        for anem_idx, anem in enumerate(all_anem_list):
            anem_slice = slice(idxs_of_each_anem[anem_idx], idxs_of_each_anem[anem_idx+1])
            X_U = X_data_nonnorm[anem_slice, 0]
            X_dirs = X_data_nonnorm[anem_slice, 1]
            X_sectors = np.searchsorted(dir_sectors, X_dirs, side='right') - 1  # groups all the measured directions into sectors
            Z_vectors = X_data_nonnorm[anem_slice,2:2+45]
            R_vectors = np.array(np.array(Z_vectors, dtype=bool), dtype=float)
            y_std_u =  y_data_nonnorm[anem_slice]
            y_Iu = y_std_u / X_U
            df_Sectors = pd.DataFrame({'X_sectors':X_sectors})
            df_U = pd.DataFrame({'U':X_U})
            df_Z = pd.DataFrame(Z_vectors).add_prefix('Z')
            df_R = pd.DataFrame(R_vectors).add_prefix('R')
            df_std_u = pd.DataFrame({'std_u':y_std_u})
            df_Iu = pd.DataFrame({'Iu':y_Iu})
            df_data_1_anem = pd.concat([df_Sectors, df_U, df_Z, df_R, df_std_u, df_Iu], axis=1)
            df_n_samples_per_sector = df_data_1_anem.groupby('X_sectors').size().reset_index(name='n_samples')  # sectors with 0 samples will vanish here!
            sectors_with_data = df_n_samples_per_sector['X_sectors'].to_numpy()
            sectors_with_no_data = [x for x in dir_sectors if x not in sectors_with_data]
            df_n_samples_per_sector = df_n_samples_per_sector.append(pd.DataFrame({'X_sectors':sectors_with_no_data, 'n_samples':np.zeros(len(sectors_with_no_data))})).sort_values(by=['X_sectors']).reset_index(drop=True)  # Manually inserting the sectors without data that vanished previously
            df_sect_means_1_anem = pd.concat([df_n_samples_per_sector, df_data_1_anem.groupby('X_sectors').mean()], axis=1)
            # Changing mean values to nan where number of samples is less than a threshold:
            n_samples_threshold = 3
            columns_to_convert_to_nan = [c for c in df_sect_means_1_anem.columns if c not in ['X_sectors','n_samples']]
            df_sect_means_1_anem.loc[df_sect_means_1_anem['n_samples'] < n_samples_threshold, columns_to_convert_to_nan] = np.nan
            df_mins_maxs_1_anem = df_sect_means_1_anem.agg([min, max])
            df_sect_means[anem] = df_sect_means_1_anem
            df_mins_maxs = df_mins_maxs.append(df_mins_maxs_1_anem)
        df_mins_maxs = df_mins_maxs.agg([min, max])  # Needed to normalize data by mins and maxs of all anems!
        return df_sect_means, df_mins_maxs

    df_sect_means, df_mins_maxs = get_sect_mean_data()
    df_mins_maxs.to_csv('df_mins_maxs.csv')

    def get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, anem_to_test, df_sect_means, df_mins_maxs, inputs_initials=['Z','R'], output='Iu', batch_size_desired=360, batch_size_lims=[30, 30000], remove_nan_data=True):
        """
        Returns: Input (X) and output (y) data, for training and testing, from given lists of anemometers to be used in the training and testing. Batch size
        """
        df_mins_maxs = copy.deepcopy(df_mins_maxs)  # mins and maxs from all anems (both training and testing)
        df_sect_means = copy.deepcopy(df_sect_means)
        if remove_nan_data:
            for anem in anem_to_train + anem_to_test:
                df_sect_means[anem] = df_sect_means[anem].dropna(axis=0, how='any').reset_index(drop=True)
        # Transforming ['Z','R','Iu'] into ['Z0','Z1','Z2',...,'R0','R1','R2',...,'Iu']:
        inputs_all = []  # names of all keys of the df_sect_means to be used in the X and y data
        for col_name in list(df_mins_maxs.columns):
            if any([col_name.startswith(inputs_initials[j]) for j in range(len(inputs_initials))]):
                inputs_all.append(col_name)
        # Organizing into Training, Testing, Input (X) and Output (y) data. Also normalizing the data into the [0,1] interval.
        X_all, y_all = {}, {}
        for anem in anem_to_train + anem_to_test:
            X_mins = np.array(df_mins_maxs.loc['min'][inputs_all])
            X_maxs = np.array(df_mins_maxs.loc['max'][inputs_all])
            y_min =  np.array(df_mins_maxs.loc['min'][output])
            y_max =  np.array(df_mins_maxs.loc['max'][output])
            X_all[anem] = np.true_divide(np.array(df_sect_means[anem][inputs_all]) - X_mins, (X_maxs - X_mins), where=((X_maxs - X_mins)!=0))  # if maxs-mins==0 (dumb non-varying  input) do nothing
            y_all[anem] = np.true_divide(np.array(df_sect_means[anem][output]    ) - y_min , (y_max  -  y_min), where=((y_max   - y_min)!=0))  # if maxs-mins==0 (dumb non-varying output) do nothing
        sectors_train = pd.concat([df_sect_means[anem]['X_sectors'] for anem in anem_to_train]).to_numpy(int)
        sectors_test  = pd.concat([df_sect_means[anem]['X_sectors'] for anem in  anem_to_test]).to_numpy(int)
        U_test        = pd.concat([df_sect_means[anem][        'U'] for anem in  anem_to_test]).to_numpy(float)
        X_train = pd.concat([pd.DataFrame(X_all[anem]) for anem in anem_to_train]).to_numpy()
        X_test  = pd.concat([pd.DataFrame(X_all[anem]) for anem in  anem_to_test]).to_numpy()
        y_train = pd.concat([pd.DataFrame(y_all[anem]) for anem in anem_to_train]).to_numpy()
        y_test  = pd.concat([pd.DataFrame(y_all[anem]) for anem in  anem_to_test]).to_numpy()
        ### OLD TRASH
        # X_train = np.array([X_all[anem] for anem in anem_to_train])  # shape:(anems, sectors, features)
        # X_test =  np.array([X_all[anem] for anem in anem_to_test])   # shape:(anems, sectors, features)
        # y_train = np.array([y_all[anem] for anem in anem_to_train])  # shape:(anems, sectors)
        # y_test =  np.array([y_all[anem] for anem in anem_to_test])   # shape:(anems, sectors)
        # # Flattening the two anems and sectors dimensions into one.
        # X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])  # shape:(anems * sectors, features)
        # X_test  =  X_test.reshape( X_test.shape[0] *  X_test.shape[1],  X_test.shape[2])  # shape:(anems * sectors, features)
        # y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1])                    # shape:(anems * sectors)
        # y_test  =  y_test.reshape( y_test.shape[0] *  y_test.shape[1])                    # shape:(anems * sectors)
        ### OLD TRASH
        # Converting to Tensor (GPU-accelerated)
        X_train = Tensor(X_train).to(device)
        X_test  = Tensor(X_test ).to(device)
        y_train = Tensor(y_train).to(device).view(y_train.shape[0], 1)
        y_test  = Tensor( y_test).to(device).view( y_test.shape[0], 1)
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
            X_train = X_train[:-1, :]
            y_train = y_train[:-1, :]
            X_test = X_test[:-1, :]
            y_test = y_test[:-1, :]
            sectors_train = sectors_train[:-1]
            sectors_test = sectors_test[:-1]
            U_test = U_test[:-1]
            n_samples_train = X_train.shape[0]
            batch_size_possibilities = np.array(sympy.divisors(n_samples_train))  # [1, 2, 4, 23, 46, 92, 4051, 8102, 16204, 93173, 186346, 372692]
            batch_size = min(batch_size_possibilities, key=lambda x: abs(x - batch_size_desired))
            batch_cond = batch_size_lims[0] < batch_size < batch_size_lims[1]
        return X_train, y_train, X_test, y_test, batch_size, sectors_train, sectors_test, U_test

    # Neural network
    def train_and_test_NN(X_train, y_train, X_test, y_test, hp, print_loss_per_epoch=True, print_results=True):
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
            my_nn.add_module(name=str(i + 1), module=torch.nn.Linear(n_neurons_last_layer, max(round(2 / 3 * n_neurons_last_layer), 5)))
            my_nn.add_module(name=f'{i + 1}A', module=my_activation_func())
        n_neurons_last_layer = (list(my_nn.modules())[-2]).out_features
        my_nn.add_module(name=str(n_hid_layers + 1), module=torch.nn.Linear(n_neurons_last_layer, n_outputs))  # Output layer
        optimizer = SGD(my_nn.parameters(), lr=learn_rate, weight_decay=weight_decay, momentum=momentum)  # define the optimizer
        # torch.seed()  # make random numbers again random
        my_nn.to(device)  # To GPU if available
        # Training
        # writer = SummaryWriter(f'runs/my_math_learning_tensorboard')  # For later using TensorBoard, for visualization
        assert (n_samples_train / batch_size).is_integer(), "Change batch size so that n_iterations is integer"
        n_iterations = int(n_samples_train / batch_size)
        for epoch in range(n_epochs):
            epoch_loss = 0
            idxs_shuffled = torch.randperm(n_samples_train)
            for b in range(n_iterations):
                batch_idxs = idxs_shuffled[b * batch_size:b * batch_size + batch_size]
                y_pred = my_nn(Variable(X_train[batch_idxs]))
                loss = criterion(y_pred, Variable(y_train[batch_idxs].view(batch_size, n_outputs), requires_grad=False))
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
            y_test_total_mean = torch.mean(y_test.view(n_samples_test, n_outputs), axis=0)
            SS_res_test = torch.sum((y_test.view(n_samples_test, n_outputs) - y_pred.view(n_samples_test, n_outputs)) ** 2)
            SS_tot_test = torch.sum((y_test.view(n_samples_test, n_outputs) - y_test_total_mean) ** 2)
            R2_test = 1 - SS_res_test / SS_tot_test
            idxs_to_print = np.random.randint(0, len(y_pred), 10)  # a few random values to be printed
        if print_results:
            print(f'R2 on test dataset: ----> {R2_test} <---- . Learning rate: {learn_rate}')
            print(f"Prediction: {y_pred[idxs_to_print]}")
            print(f"Reference:   {y_test[idxs_to_print]}")
            print(f'Batch size: {batch_size}')
        return y_pred, R2_test

    ##################################################################################################################

    activation_fun_dict = {'ReLU': torch.nn.modules.activation.ReLU,
                           'ELU': torch.nn.modules.activation.ELU,
                           'LeakyReLU': torch.nn.modules.activation.LeakyReLU}
    loss_fun_dict = {'SmoothL1Loss': SmoothL1Loss(),
                     'MSELoss': MSELoss(),
                     'L1Loss': L1Loss(),
                     'LogCoshLoss': LogCoshLoss()}

    def find_optimal_hp_for_each_of_my_cases(my_NN_cases, df_sect_means, df_mins_maxs, n_trials, print_loss_per_epoch=False, print_results=False):
        hp_opt_results = []
        for my_NN_case in my_NN_cases:
            anem_to_train = my_NN_case['anem_to_train']
            anem_to_test = my_NN_case['anem_to_test']
            X_train, y_train, X_test, y_test, batch_size, _, _, _ = get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, anem_to_test, df_sect_means, df_mins_maxs)
            print('X_train shape is:' + str(X_train.shape))
            print('batch_size shape is:' + str(batch_size))
            # Beautiful MAGIC happening
            def hp_opt_objective(trial):
                weight_decay = trial.suggest_float("weight_decay", 1E-7, 1E-1, log=True)
                lr = trial.suggest_float("lr", 0.001, 0.8, log=True)
                momentum = trial.suggest_float("momentum", 0., 0.95)
                n_hid_layers = trial.suggest_int('n_hid_layers', 2, 6)
                n_epochs = trial.suggest_int('n_epochs', 10, 3000)
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
                _, R2 = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=print_loss_per_epoch, print_results=print_results)
                return R2
            while True:
                try:  # because now and then the ANN produces an error (e.g. weights explode during learning)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(hp_opt_objective, n_trials=n_trials)
                except ValueError:
                    continue
                break
            hp_opt_result = {'anem_to_test': anem_to_test, 'anems_to_train': anem_to_train, 'best_params': study.best_params, 'best_value': study.best_value}
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

    these_hp_opt = find_optimal_hp_for_each_of_my_cases(my_NN_cases, df_sect_means, df_mins_maxs, n_trials=n_trials)

    # Saving the results into a txt file, only if "these" results are better than the ones already stored in txt
    for case_idx in range(len(my_NN_cases)):
        my_NN_case = my_NN_cases[case_idx]
        anem_to_test = my_NN_case['anem_to_test']
        try:
            with open(f'hp_opt.txt', 'r') as prev_file:
                prev_hp_opt_results = eval(json.load(prev_file))
            tested_results = np.array([[tested_idx, *i['anem_to_test']] for tested_idx, i in enumerate(prev_hp_opt_results)])
            tested_results_idx = np.where(tested_results[:, 1] == anem_to_test)[0]
            if len(tested_results_idx):
                if these_hp_opt[case_idx]['best_value'] < prev_hp_opt_results[tested_results_idx[0]]['best_value']:
                    these_hp_opt[case_idx] = prev_hp_opt_results[tested_results_idx[0]]
        except FileNotFoundError:
            print(anem_to_test)
            print('No file with name: ' + f'hp_opt.txt !!')
    hp_opt = copy.deepcopy(these_hp_opt)
    with open(f'hp_opt.txt', 'w') as file:
        file.write(json.dumps(str(hp_opt)))

    for case_idx in range(len(my_NN_cases)):
        my_NN_case = my_NN_cases[case_idx]
        anem_to_train = my_NN_case['anem_to_train']
        anem_to_test = my_NN_case['anem_to_test']

        hp = copy.deepcopy(hp_opt[case_idx]['best_params'])
        if type(hp['activation']) == str:
            hp['activation'] = activation_fun_dict[hp['activation']]
        if type(hp['loss']) == str:
            hp['loss'] = loss_fun_dict[hp['loss']]
        X_train, y_train, X_test, y_test, batch_size, sectors_train, sectors_test, U_test = get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, anem_to_test, df_sect_means, df_mins_maxs)
        hp['batch_size'] = batch_size
        y_pred, R2 = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=True, print_results=True)

        y_test_nonnorm = np.ndarray.flatten(y_test.cpu().numpy()) * (df_mins_maxs['Iu'].loc['max'] - df_mins_maxs['Iu'].loc['min']) + df_mins_maxs['Iu'].loc['min']
        y_pred_nonnorm = np.ndarray.flatten(y_pred.cpu().numpy()) * (df_mins_maxs['Iu'].loc['max'] - df_mins_maxs['Iu'].loc['min']) + df_mins_maxs['Iu'].loc['min']

        # PLOT MEANS OF PREDICTIONS
        R2_ANN = str(np.round(R2.cpu().numpy(), 3))
        Iu_EN_anem = np.array(Iu_EN[anem_to_test[0][:-2]])
        Iu_EN_at_sectors_w_data = Iu_EN_anem[sectors_test]
        R2_with_EN = np.round(r2_score(y_test_nonnorm, Iu_EN_at_sectors_w_data), 3)
        fig, ax1 = plt.subplots(figsize=(5.5*0.93, 2.5*0.93), dpi=400/0.93)
        # plt.title(nice_str_dict[my_NN_cases[case_idx]['anem_to_test'][0]] + '.  $R^2='+ str(np.round(R2_of_means, 2))+'$')
        ax1.set_title(f"Sectoral averages of $I_u$ at {nice_str_dict[my_NN_cases[case_idx]['anem_to_test'][0]]}")
        # plt.scatter(X_test_dirs_nonnorm, y_test_nonnorm, s=0.01, alpha=0.2, c='black') #, label='Measured')
        ax1.scatter(sectors_test, y_test_nonnorm, s=12, alpha=0.6, c='black', zorder=0.98, edgecolors='none', marker='s', label='Measurements')
        ax1.scatter(sectors_test, y_pred_nonnorm, s=12, alpha=0.6, c='darkorange', zorder=1.0, edgecolors='none', marker='o', label='ANN predictions')
        # plt.scatter(np.arange(360), Iu_EN[anem_to_test[0][:-2]], s=4, alpha=0.7, c='deepskyblue', label='NS-EN 1991-1-4')
        ax1.scatter(sectors_test, Iu_EN_at_sectors_w_data, s=12, alpha=0.6, c='green', zorder=0.99, edgecolors='none', marker='^', label='NS-EN 1991-1-4')
        ax2 = ax1.twinx()
        ax2.scatter(sectors_test, U_test, s=3, alpha=0.3, color='deepskyblue', zorder=0.97, edgecolors='none', marker='D', label='$\overline{U}$')
        # ax2.plot(sectors_test, U_test, alpha=0.3, color='deepskyblue', zorder=0.97, label='$\overline{U}$')
        ax1.legend(markerscale=2., loc=2, handletextpad=0.1)
        ax2.legend(markerscale=2., loc=1, handletextpad=0.1)
        ax1.set_ylabel('$\overline{I_u}$')
        ax2.set_ylabel('$\overline{U}\/\/\/[m/s]$')
        ax1.set_xlabel('Wind from direction [\N{DEGREE SIGN}]')
        ax1.set_xlim([0, 360])
        ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax1.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
        ax1.set_ylim([0, 0.63])
        ax2.set_ylim([0, 20])
        fig.tight_layout(pad=0.05)
        plt.savefig(os.path.join(os.getcwd(), 'plots', f'{result_name_tag}_Iu_Case_{case_idx}_{anem_to_test[0]}_Umin_{U_min}_Sector-{dir_sector_amp}_ANNR2_{R2_ANN}_ENR2_{R2_with_EN}.png'))
        # plt.show()
    return None

# predict_mean_turbulence_with_ML(n_trials=1)


