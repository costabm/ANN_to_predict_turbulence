"""
This study uses Machine Learning to predict turbulence at a new location with a given topography, where the learning data
consists only of turbulence data and topography data at other nearby locations.
THIS VERSION USES ALL 6 ANEMOMETERS TO ESTIMATE TURBULENCE AT DIFFERENT POINTS ALONG THE BJØRNAFJORD
Created: September, 2021
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

# Anemometer coordinates
anem_nice_str = {'osp1_A': 'Ospøya 1', 'osp2_A': 'Ospøya 2', 'osp2_B': 'Ospøya 2', 'synn_A': 'Synnøytangen', 'svar_A': 'Svarvhelleholmen', 'land_A': 'Landrøypynten', 'neso_A': 'Nesøya'}
anem_EN_33 = {'synn': synn_EN_33, 'svar': svar_EN_33, 'osp1': osp1_EN_33, 'osp2': osp2_EN_33, 'land': land_EN_33, 'neso': neso_EN_33}
# Getting bridge node coordinates
n_bridge_nodes = 11
latlons_bridge = bridge_WRF_nodes_coor_func(n_bridge_WRF_nodes = n_bridge_nodes, unit='deg')
# to convert from Lat/Lon to UTM-33, use the website: https://www.kartverket.no/en/on-land/posisjon/transformere-koordinater-enkeltvis
assert n_bridge_nodes == 11, "n_bridge_nodes needs to be 11. Otherwise, you need to manually create the new array of bridge node coordinates, using the website above"
bj_coors = np.array([[-34449.260, 6699999.046],
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
    bj_pts_EN_33['bj'+f'{i+1:02}']=bj_coors[i].tolist()
    bj_pts_nice_str['bj'+f'{i+1:02}']='Bjørnafjord P'+str(i+1)
# Merging dictionaries (anems + bj_pts) into one dict all_pts
all_pts_EN_33 = dict(anem_EN_33)
all_pts_EN_33.update(bj_pts_EN_33)
all_pts_nice_str = dict(anem_nice_str)
all_pts_nice_str.update(bj_pts_nice_str)

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


def example_of_elevation_profile_at_given_point_dir_dist(point_1, direction_deg=160, step_distance=False, total_distance=False,
                                                         list_of_distances=[i*(5.+5.*i) for i in range(45)], plot=True):
    point_2 = get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=direction_deg, distance=list_of_distances[-1] if list_of_distances else total_distance)
    dists, heights = elevation_profile_generator(point_1=point_1, point_2=point_2, step_distance=step_distance, list_of_distances=list_of_distances)
    if plot:
        plot_elevation_profile(point_1=point_1, point_2=point_2, step_distance=step_distance, list_of_distances=list_of_distances)
    return dists, heights
# example_of_elevation_profile_at_given_point_dir_dist(point_1=bj_coors[5], direction_deg=160, step_distance=False, total_distance=False,
#                                                    list_of_distances=[i*(5.+5.*i) for i in range(45)], plot=True)


def plot_topography_per_point(list_of_degs = list(range(360)), list_of_distances=[i*(5.+5.*i) for i in range(45)]):
    for pt, pt_coor in all_pts_EN_33.items():
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
# plot_topography_per_point(list_of_degs = list(range(360)), list_of_distances=[i*(5.+5.*i) for i in range(45)])


def get_heights_from_X_dirs_and_dists(point_1, array_of_dirs, cone_angles, dists):
    heights = []
    for a in cone_angles:
        X_dir_anem_yawed = convert_angle_to_0_2pi_interval(array_of_dirs + a, input_and_output_in_degrees=True)
        points_2 = np.array([get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=d, distance=dists[-1]) for d in X_dir_anem_yawed])
        heights.append(np.array([elevation_profile_generator(point_1=point_1, point_2=p2, step_distance=False, list_of_distances=dists)[1] for p2 in points_2]))
    return np.array(heights)


def get_all_10_min_data_at_z_48m(U_min = 0):
    print('Collecting all 10-min wind data... (takes 10-60 minutes)')
    # windward_dists = [i * ( 5. +  5. * i) for i in range(45)]  # adopted in the main results of the COTech paper
    # windward_dists = [i * (41.666666666666664 + 41.666666666666664 * i) for i in range(15+1)]
    windward_dists = [i * (14.245014245014245 + 14.245014245014245 * i) for i in range(25 + 1)]
    # windward_dists = [i * ( 10.75268817204301 +  10.75268817204301 * i) for i in range(30+1)]
    # windward_dists = [i * ( 4.830917874396135 +  4.830917874396135 * i) for i in range(45+1)]
    # windward_dists = [i * (  2.73224043715847 +   2.73224043715847 * i) for i in range(60+1)]

    leeward_dists =  [i * (10. + 10. * i) for i in range(10)]
    side_dists =     [i * (10. + 10. * i) for i in range(10)]
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


def get_one_roughness_transition_zone(dists, heights):
    """
    Getting the transition zones as described in NS-EN 1991-1-4:2005/NA:2009, NA.4.3.2(2)
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
    for pt, pt_coor in all_pts_EN_33.items():
        transition_zones[pt] = []
        for d in list(range(360)):
            dists, heights = example_of_elevation_profile_at_given_point_dir_dist(point_1=pt_coor, direction_deg=d, step_distance=step_distance, total_distance=total_distance, list_of_distances=list_of_distances, plot=False)
            transition_zones[pt].append(get_one_roughness_transition_zone(dists, heights))
    return transition_zones


def plot_roughness_transitions_per_anem(list_of_degs = list(range(360)), step_distance=False, total_distance=False, list_of_distances=[i*(5.+5.*i) for i in range(45)]):
    transition_zones = get_all_roughness_transition_zones()
    for pt, pt_coor in all_pts_EN_33.items():
        dists_all_dirs = []
        roughs_all_dirs = []
        degs = []
        for d in list_of_degs:
            dists, heights = example_of_elevation_profile_at_given_point_dir_dist(point_1=pt_coor, direction_deg=d, step_distance=step_distance, total_distance=total_distance, list_of_distances=list_of_distances, plot=False)
            roughs = heights.astype(bool).astype(float)
            dists_all_dirs.append(dists)
            roughs_all_dirs.append(roughs)
            degs.append(d)
        degs = np.array(degs)
        dists_all_dirs = np.array(dists_all_dirs)
        roughs_all_dirs = np.array(roughs_all_dirs)
        fig, ax = plt.subplots(figsize=(5.5,2.3+0.5), dpi=400)
        ax.pcolormesh(degs, dists_all_dirs[0], roughs_all_dirs.T, cmap=matplotlib.colors.ListedColormap(['skyblue', 'navajowhite']), shading='auto') #, vmin = 0., vmax = 1.)
        xB = np.array([item[1] for item in transition_zones[pt]])
        asc_desc = np.array([item[2] for item in transition_zones[pt]])
        asc_idxs = np.where(asc_desc=='ascending')
        des_idxs = np.where(asc_desc=='descending')
        ax.scatter(degs[asc_idxs], xB[asc_idxs], s=1, alpha=0.4, color='red', label='ascending')
        ax.scatter(degs[des_idxs], xB[des_idxs], s=1, alpha=0.4, color='green', label='descending')
        ax.set_title(bj_pts_nice_str[pt]+': '+'Upstream topography;')
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
    for pt in all_pts_EN_33.keys():
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
    data_path = os.path.join(os.getcwd(), 'processed_data_for_ML', f'X_y_ML_ready_data_Umin_{U_min}_masts_6_new_dirs_w_ts_26_dists')
    np.savez_compressed(data_path, X=X_data_nonnorm, y=y_data_nonnorm, m=all_anem_list, i=start_idxs_of_each_anem, t=np.array(ts_data, dtype=object)) # o=other_data)
    with open(data_path+'_other_data.txt', 'w') as outfile:
        json.dump(other_data, outfile)
    #################################################################
    return None

# generate_new_data(U_min=5)

# # Cases
# my_cases = [{'anem_to_cross_val': ['osp1_A', 'osp2_A', 'svar_A', 'land_A', 'neso_A'], # a final anemometer is left out to assess the final model performance, without performing any optimzation
#              'anem_to_test':      ['synn_A']},                                        # leave-one-out-cross-validation of train+testing+hp_optimization
#             {'anem_to_cross_val': ['osp1_A', 'osp2_A', 'synn_A', 'land_A', 'neso_A'],
#              'anem_to_test':      ['svar_A']},
#             {'anem_to_cross_val': ['osp2_A', 'synn_A', 'svar_A', 'land_A', 'neso_A'],
#              'anem_to_test':      ['osp1_A']},
#             {'anem_to_cross_val': ['osp1_A', 'synn_A', 'svar_A', 'land_A', 'neso_A'],
#              'anem_to_test':      ['osp2_A']},
#             {'anem_to_cross_val': ['osp1_A', 'osp2_A', 'synn_A', 'svar_A', 'neso_A'],
#              'anem_to_test':      ['land_A']},
#             {'anem_to_cross_val': ['osp1_A', 'osp2_A', 'synn_A', 'svar_A', 'land_A'],
#              'anem_to_test':      ['neso_A']},
#             ]
my_cases = [{'anem_to_cross_val': ['synn_A', 'osp1_A', 'osp2_A', 'svar_A', 'land_A', 'neso_A'], # a final anemometer is left out to assess the final model performance, without performing any optimzation
             'anem_to_test':      ['bj01']}]

# my_cases = [{'anem_to_cross_val': ['synn_A', 'osp1_A', 'osp2_A', 'svar_A', 'land_A', 'neso_A'], # a final anemometer is left out to assess the final model performance, without performing any optimzation
#              'anem_to_test':      list(bj_pts_EN_33.keys())}]


def predict_mean_turbulence_with_ML_at_BJ(my_cases, n_hp_trials, name_prefix, n_dists=45, make_plots=True):
    """
        n_dists: e.g. 16, 31, 45, 61. Number of entries of each of the Z and R vectors
    """
    U_min = 5
    dists_dict = {'16':[i * (41.666666666666664 + 41.666666666666664 * i) for i in range(15+1)],
                  '26':[i * (14.245014245014245 + 14.245014245014245 * i) for i in range(25+1)],
                  '31': [i * (10.75268817204301 + 10.75268817204301 * i) for i in range(30 + 1)],
                  '45':[i * (                5. +                 5. * i) for i in range(45)],  # adopted in the main results of the COTech paper
                  '61':[i * (  2.73224043715847 +   2.73224043715847 * i) for i in range(60+1)]}
    dists_vec = dists_dict[str(n_dists)]

    dir_sector_amp = 1
    # Loading data already saved
    data_path = os.path.join(os.getcwd(), 'processed_data_for_ML', f'X_y_ML_ready_data_Umin_{U_min}_masts_6_new_dirs_w_ts_{n_dists}_dists')
    loaded_data = np.load(data_path + '.npz')
    loaded_data.allow_pickle = True
    X_data_nonnorm =          loaded_data['X']
    y_data_nonnorm =          loaded_data['y']
    all_anem_list =           loaded_data['m']
    all_pts_list = list(all_anem_list) + list(bj_pts_EN_33.keys())
    start_idxs_of_each_anem = loaded_data['i']
    X_ts =                    loaded_data['t']
    print(f'Number of features {X_data_nonnorm.shape[1]}: U(1)+Dir(1)+WindwardHeightsMiddleCone(45)+MeanWindWard(44)+STDWindWard(44)+MeanLeeWard(10)+STDLeeWard(9)+MeanSideWard(10)+STDSideWard(9)')
    n_samples = len(y_data_nonnorm)
    dir_sectors = np.arange(0, 360, dir_sector_amp)
    n_sectors = len(dir_sectors)
    idxs_of_each_anem = np.append(start_idxs_of_each_anem, n_samples)

    def split_list_into_strictly_ascending_sublists(lst):
        return np.split(lst, np.where(np.diff(lst) < 0)[0] + 1)

    def get_sect_mean_data():  # get Sectoral mean data
        df_sect_means = {}
        df_mins_maxs = pd.DataFrame()
        for pt_idx, pt in enumerate(all_pts_list):
            if 'bj' not in pt:
                anem_slice = slice(idxs_of_each_anem[pt_idx], idxs_of_each_anem[pt_idx+1])
                X_U = X_data_nonnorm[anem_slice, 0]
                X_dirs = X_data_nonnorm[anem_slice, 1]
                X_sectors = np.searchsorted(dir_sectors, X_dirs, side='right') - 1  # groups all the measured directions into sectors
                Z_vectors = X_data_nonnorm[anem_slice,2:2+n_dists]
                R_vectors = np.array(np.array(Z_vectors, dtype=bool), dtype=float)
                y_std_u =  y_data_nonnorm[anem_slice]
                y_Iu = y_std_u / X_U
                df_Sectors = pd.DataFrame({'X_sectors': X_sectors})
                df_U = pd.DataFrame({'U': X_U})
                df_Z = pd.DataFrame(Z_vectors).add_prefix('Z')
                df_R = pd.DataFrame(R_vectors).add_prefix('R')
                df_std_u = pd.DataFrame({'std_u': y_std_u})
                df_Iu = pd.DataFrame({'Iu': y_Iu})
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
                df_sect_means[pt] = df_sect_means_1_anem
                df_mins_maxs = df_mins_maxs.append(df_mins_maxs_1_anem)
            elif 'bj' in pt:
                X_U = np.zeros(len(dir_sectors)) * np.nan
                X_sectors = dir_sectors
                Z_vectors = get_heights_from_X_dirs_and_dists(all_pts_EN_33[pt], dir_sectors, [0], dists_vec)[0]
                R_vectors = np.array(np.array(Z_vectors, dtype=bool), dtype=float)
                y_std_u = np.zeros(len(dir_sectors)) * np.nan
                y_Iu = Iu_EN[pt]
                df_Sectors = pd.DataFrame({'X_sectors':X_sectors})
                df_U = pd.DataFrame({'U':X_U})
                df_Z = pd.DataFrame(Z_vectors).add_prefix('Z')
                df_R = pd.DataFrame(R_vectors).add_prefix('R')
                df_std_u = pd.DataFrame({'std_u':y_std_u})
                df_Iu = pd.DataFrame({'Iu':y_Iu})
                df_sect_means_1_anem = pd.concat([df_Sectors, df_U, df_Z, df_R, df_std_u, df_Iu], axis=1)
                df_mins_maxs_1_anem = df_sect_means_1_anem.agg([min, max])
                df_sect_means[pt] = df_sect_means_1_anem
                df_mins_maxs = df_mins_maxs.append(df_mins_maxs_1_anem)
        df_mins_maxs = df_mins_maxs.agg([min, max])  # Needed to normalize data by mins and maxs of all anems!
        return df_sect_means, df_mins_maxs
    df_sect_means, df_mins_maxs = get_sect_mean_data()
    df_mins_maxs.to_csv('df_mins_maxs.csv')

    def get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, anem_to_test, df_sect_means, df_mins_maxs, inputs_initials=['Z','R'], output='Iu', batch_size_desired='full', batch_size_lims=[30, 30000], remove_nan_data=True):
        """
        Returns: Input (X) and output (y) data, for training and testing, from given lists of anemometers to be used in the training and testing. Batch size
        """
        if isinstance(anem_to_test, str):
            anem_to_test = [anem_to_test]  # if it is str e.g. 'osp1_A' then convert to ['osp1_A']
        df_mins_maxs = copy.deepcopy(df_mins_maxs)  # mins and maxs from all anems (both training and testing)
        df_sect_means = copy.deepcopy(df_sect_means)
        if remove_nan_data:
            for anem in anem_to_train + anem_to_test:
                if 'bj' not in anem:  # bj points have nans that we don't want to remove
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
        # Converting to Tensor (GPU-accelerated)
        X_train = Tensor(X_train).to(device)
        X_test  = Tensor(X_test ).to(device)
        y_train = Tensor(y_train).to(device).view(y_train.shape[0], 1)
        y_test  = Tensor( y_test).to(device).view( y_test.shape[0], 1)
        # Getting batch size (which can change the X and y data, by trimming a few data points!)
        n_samples_train = X_train.shape[0]
        if batch_size_desired == 'full':
            batch_size = n_samples_train
        else:
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
        n_first_hid_layer_neurons = max(round(2 / 3 * n_features), 2)
        my_nn = torch.nn.Sequential()
        if n_hid_layers == 0:
            my_nn.add_module(name='1', module=torch.nn.Linear(n_features, n_outputs))
            my_nn.add_module(name='1A', module=torch.nn.modules.activation.ReLU())  # Activation function
        else:
            my_nn.add_module(name='1', module=torch.nn.Linear(n_features, n_first_hid_layer_neurons))  # Second layer
            my_nn.add_module(name='1A', module=my_activation_func())  # Activation function
            for i in range(1, n_hid_layers):  # Hidden layers
                n_neurons_last_layer = (list(my_nn.modules())[-2]).out_features
                my_nn.add_module(name=str(i+1), module=torch.nn.Linear(n_neurons_last_layer, max(round(2 / 3 * n_neurons_last_layer), 2)))
                my_nn.add_module(name=f'{i+1}A', module=my_activation_func())
            n_neurons_last_layer = (list(my_nn.modules())[-2]).out_features
            my_nn.add_module(name=str(n_hid_layers + 1), module=torch.nn.Linear(n_neurons_last_layer, n_outputs))  # Output layer
            my_nn.add_module(name=str(n_hid_layers + 1)+'A', module=torch.nn.modules.activation.ReLU())
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
            # ACCURACY NEEDS TO BE DONE ON NON-NORMALIZED DATA WHEREAS R2 IS EQUAL REGARDLESS OF NORMALIZATION
            # accuracy = 100 - 100 * torch.mean(torch.abs(y_pred.view(n_samples_test, n_outputs) - y_test.view(n_samples_test, n_outputs)) / y_test.view(n_samples_test, n_outputs))
            idxs_to_print = np.random.randint(0, len(y_pred), 10)  # a few random values to be printed
        if print_results:
            print(f'R2 on test dataset: ----> {R2_test} <---- . Learning rate: {learn_rate}')
            print(f"Prediction: {y_pred[idxs_to_print]}")
            print(f"Reference:   {y_test[idxs_to_print]}")
            print(f'Batch size: {batch_size}')
        return y_pred, R2_test.cpu().numpy()

    ##################################################################################################################

    # activation_fun_dict = {'ReLU': torch.nn.modules.activation.ReLU,
    #                        'ELU': torch.nn.modules.activation.ELU,
    #                        'LeakyReLU': torch.nn.modules.activation.LeakyReLU}
    activation_fun_dict = {'ELU': torch.nn.modules.activation.ELU}
    # loss_fun_dict = {'MSELoss': MSELoss(),  # 'SmoothL1Loss': SmoothL1Loss(),
    #                  'L1Loss': L1Loss(),
    #                  'LogCoshLoss': LogCoshLoss()}
    loss_fun_dict = {'L1Loss': L1Loss()}


    def optimize_hp_for_all_cross_val_cases(my_cases, df_sect_means, df_mins_maxs, n_hp_trials, print_loss_per_epoch=False, print_results=False):
        hp_opt_results = []
        for my_case in my_cases:
            anem_to_cross_val = my_case['anem_to_cross_val']
            def hp_opt_objective(trial):
                weight_decay = trial.suggest_float("weight_decay",1E-3, 1, log=True )  # 1E-5, 1, log=True)
                lr = trial.suggest_float("lr", 0.001, 1, log=True)
                momentum = trial.suggest_float("momentum", 0., 0.95)
                n_hid_layers = trial.suggest_int('n_hid_layers',2,2)  # , 0, 4)
                n_epochs = trial.suggest_int('n_epochs',1000,1000)  #, 20, 2000)
                activation_fun_name = trial.suggest_categorical('activation', list(activation_fun_dict))
                activation_fun = activation_fun_dict[activation_fun_name]
                loss_fun_name = trial.suggest_categorical('loss', list(loss_fun_dict))
                loss_fun = loss_fun_dict[loss_fun_name]
                hp = {'lr': lr,
                      'weight_decay': weight_decay,
                      'momentum': momentum,
                      'n_epochs': n_epochs,
                      'n_hid_layers': n_hid_layers,
                      'activation': activation_fun,
                      'loss': loss_fun}
                R2_of_all_cross_val = []
                for anem_to_test in anem_to_cross_val:
                    anem_to_train = [a for a in anem_to_cross_val if a != anem_to_test]
                    X_train, y_train, X_test, y_test, batch_size, _, _, _ = get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, [anem_to_test], df_sect_means, df_mins_maxs)
                    hp['batch_size'] = batch_size
                    _, R2 = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=print_loss_per_epoch, print_results=print_results)
                    R2_of_all_cross_val.append(float(R2))
                trial.set_user_attr('R2_each_split', R2_of_all_cross_val)
                # print(f'R2 of each split (during cross-validation): {R2_of_all_cross_val}')
                return np.mean(R2_of_all_cross_val)
            while True:
                try:  # because sometimes the ANN produces an error (e.g. weights explode during learning)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(hp_opt_objective, n_trials=n_hp_trials)
                except ValueError:
                    continue
                break
            hp_opt_result = {'anem_to_cross_val': anem_to_cross_val, 'anem_to_test': my_case['anem_to_test'], 'best_params': study.best_params,
                             'R2_each_split': study.best_trial.user_attrs['R2_each_split'], 'best_validation_value': study.best_value}
            hp_opt_results.append(hp_opt_result)
        return hp_opt_results

    these_hp_opt = optimize_hp_for_all_cross_val_cases(my_cases, df_sect_means, df_mins_maxs, n_hp_trials=n_hp_trials)

    # Finally, running the ANN for all Training Cases (no anem is left-out for cross-validation) and testing the ANN for the final test:
    for case_idx in range(len(my_cases)):
        my_case = my_cases[case_idx]
        anem_to_train = my_case['anem_to_cross_val']
        anem_to_test = my_case['anem_to_test']
        # Saving the results into a txt file, but first checking if there are already better results stored in txt, and if so, bring them here
        try:
            with open(f'{name_prefix}_hp_opt_cross_val.txt', 'r') as prev_file:
                prev_hp_opt_results = eval(json.load(prev_file))
            tested_results = np.array([[tested_idx, *i['anem_to_test']] for tested_idx, i in enumerate(prev_hp_opt_results)])
            tested_results_idx = np.where(tested_results[:, 1] == anem_to_test)[0]
            if len(tested_results_idx):
                if these_hp_opt[case_idx]['best_validation_value'] < prev_hp_opt_results[tested_results_idx[0]]['best_validation_value']:
                    these_hp_opt[case_idx] = prev_hp_opt_results[tested_results_idx[0]]
        except FileNotFoundError:
            print(anem_to_test)
            print('No file with name: ' + f'{name_prefix}_hp_opt_cross_val.txt !!')
        # Testing the final testing data:
        hp = copy.deepcopy(these_hp_opt[case_idx]['best_params'])
        if type(hp['activation']) == str:
            hp['activation'] = activation_fun_dict[hp['activation']]
        if type(hp['loss']) == str:
            hp['loss'] = loss_fun_dict[hp['loss']]
        X_train, y_train, X_test, y_test, batch_size, sectors_train, sectors_test, U_test = get_X_y_train_and_test_and_batch_size_from_anems(anem_to_train, anem_to_test, df_sect_means, df_mins_maxs)
        idxs_sectors_test = np.array([i for i, item in enumerate(dir_sectors) if item in sectors_test])
        hp['batch_size'] = batch_size
        y_pred, R2 = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=False, print_results=False)
        y_test_nonnorm = np.ndarray.flatten(y_test.cpu().numpy()) * (df_mins_maxs['Iu'].loc['max'] - df_mins_maxs['Iu'].loc['min']) + df_mins_maxs['Iu'].loc['min']
        y_pred_nonnorm = np.ndarray.flatten(y_pred.cpu().numpy()) * (df_mins_maxs['Iu'].loc['max'] - df_mins_maxs['Iu'].loc['min']) + df_mins_maxs['Iu'].loc['min']
        if 'bj' in anem_to_test:
            Iu_EN_anem = np.array(Iu_EN[anem_to_test[0]])
        else:
            Iu_EN_anem = np.array(Iu_EN[anem_to_test[0]])[idxs_sectors_test]
        accuracy_ANN = 100 - 100 * np.mean(np.abs(y_pred_nonnorm - y_test_nonnorm) / y_test_nonnorm)
        accuracy_EC  = 100 - 100 * np.mean(np.abs(    Iu_EN_anem - y_test_nonnorm) / y_test_nonnorm)
        print(f'Testing: {anem_to_test[0]}... R2: {np.round(R2,4)}. Accuracy: {np.round(accuracy_ANN,4)}')
        these_hp_opt[case_idx]['final_ANN_R2_test_value'] = float(R2)
        these_hp_opt[case_idx]['final_ANN_accuracy'] = float(accuracy_ANN)
        these_hp_opt[case_idx]['final_EC_R2_test_value'] = float(r2_score(y_test_nonnorm, Iu_EN_anem))
        these_hp_opt[case_idx]['final_EC_accuracy'] =  float(accuracy_EC)

        # PLOT MEANS OF PREDICTIONS
        if make_plots:  # these plots are suitable for the predictions at the middle of Bjørnafjord, where there are no measurements
            unique_pts = []  # lets not plot the same point twice, even though it might be included twice in the testing
            for idx_pt_to_test, pt_to_test in enumerate(anem_to_test):
                if pt_to_test not in unique_pts:
                    unique_pts.append(pt_to_test)
                    sectors_test_by_pt = split_list_into_strictly_ascending_sublists(sectors_test)
                    sectors_test_1_pt = sectors_test_by_pt[idx_pt_to_test]
                    lens_by_pt = [len(sectors_test_by_pt[i]) for i in range(0,idx_pt_to_test+1)]  # n_sectors for each point. e.g: [360,360,317,302,360,...]
                    start_idxs_of_each_pt = np.cumsum([0] + lens_by_pt)
                    pt_slice = slice(start_idxs_of_each_pt[idx_pt_to_test], start_idxs_of_each_pt[idx_pt_to_test+1])  # only meant when sectors_test includes several tested anems??
                    R2_ANN = str(np.round(R2, 3))
                    fig, ax1 = plt.subplots(figsize=(5.5*0.93, 2.5*0.93), dpi=400/0.93)
                    ax1.set_title(f"Sectoral averages of $I_u$ at {all_pts_nice_str[my_cases[case_idx]['anem_to_test'][idx_pt_to_test]]}")
                    ax1.scatter(sectors_test[pt_slice], y_pred_nonnorm[pt_slice], s=12, alpha=0.6, c='darkorange', zorder=1.0, edgecolors='none', marker='o', label='ANN predictions')
                    if 'bj' in pt_to_test:
                        ax1.scatter(sectors_test[pt_slice], Iu_EN_anem, s=12, alpha=0.6, c='green', zorder=0.99, edgecolors='none', marker='^', label='NS-EN 1991-1-4')
                        ax1.legend(markerscale=2., loc=1, handletextpad=0.1)
                    else:
                        ax1.scatter(sectors_test[pt_slice], Iu_EN_anem, s=12, alpha=0.6, c='green', zorder=0.99, edgecolors='none', marker='^', label='NS-EN 1991-1-4')
                        measur_anem = np.array(y_test_nonnorm[pt_slice])
                        ax1.scatter(sectors_test[pt_slice], measur_anem, s=12, alpha=0.6, c='black', zorder=0.98, edgecolors='none', marker='s', label='Measurements')
                        ax2 = ax1.twinx()
                        ax2.scatter(sectors_test, U_test, s=3, alpha=0.3, color='deepskyblue', zorder=0.97, edgecolors='none', marker='D', label='$\overline{U}$')
                        ax1.legend(markerscale=2., loc=2, handletextpad=0.1)
                        ax2.legend(markerscale=2., loc=1, handletextpad=0.1)
                        ax2.set_ylabel('$\overline{U}\/\/\/[m/s]$')
                        ax2.set_ylim([0, 20])
                    ax1.set_ylabel('$\overline{I_u}$')
                    ax1.set_xlabel('Wind from direction [\N{DEGREE SIGN}]')
                    ax1.set_xlim([0, 360])
                    ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
                    ax1.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
                    ax1.set_ylim([0, 0.63])
                    fig.tight_layout(pad=0.05)
                    plt.savefig(os.path.join(os.getcwd(), 'plots', f'{name_prefix}_Bjorna_Iu_Case_{case_idx}_{anem_to_test[0]}_Umin_{U_min}_Sector-{dir_sector_amp}_ANNR2_{R2_ANN}.png'))
                    plt.show()

    with open(f'{name_prefix}_hp_opt_cross_val.txt', 'w') as file:
        file.write(json.dumps(str(these_hp_opt)))

    return None

for n_dists in [45]:  # n_dists is the number of upstream points in the Z and R vectors used in the input. 45 is the default (but 16, 31, and 60 points were also attempted, with similar results)
    for i in range(0,20):
        predict_mean_turbulence_with_ML_at_BJ(my_cases, n_hp_trials=50, name_prefix=f'test_bj_11_points_'+str(i))


def get_results_from_txt_file(n_final_tests_per_anem):
    results = {}  # shape: ('n_final_tests', n_anems_tested, 'features'), where '' represents dictionary
    for name_prefix in range(n_final_tests_per_anem):
        with open(f'dists_26_{name_prefix}_hp_opt_cross_val.txt', 'r') as file:
            results[str(name_prefix)] = eval(json.load(file))
    return results


def plot_R2_and_accuracies(n_final_tests_per_anem=5, bw=0.75, markersize=5.5, font_scale=1.18):
    import seaborn as sns
    import matplotlib.patches as mpatches

    results = get_results_from_txt_file(n_final_tests_per_anem)

    df_results_R2 = pd.DataFrame()
    df_results_acc = pd.DataFrame()
    for t in range(n_final_tests_per_anem):
        for a in range(6):
            anem_to_test = results[str(t)][a][    'anem_to_test'][0]
            df_results_R2 = df_results_R2.append({'anem'        :anem_nice_str[anem_to_test],
                                                  'ANN'      :results[str(t)][a]['final_ANN_R2_test_value'],
                                                  'NS-EN 1991-1-4'       :results[str(t)][a]['final_EC_R2_test_value'],
                                                  }, ignore_index=True)
            df_results_acc=df_results_acc.append({'anem': anem_nice_str[anem_to_test],
                                                  'ANN': results[str(t)][a]['final_ANN_accuracy'],
                                                  'NS-EN 1991-1-4': results[str(t)][a]['final_EC_accuracy']
                                                  }, ignore_index=True)
    # df_results_R2_melted =  pd.melt( df_results_R2, id_vars="anem", var_name="Method", value_name="values").dropna()
    # df_results_acc_melted = pd.melt(df_results_acc, id_vars="anem", var_name="Method", value_name="values").dropna()

    # Finding the R2 result among the ANN models of each anemometer that is closest to the mean (expected value of) R2.
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    df_results_R2_mean = df_results_R2.groupby('anem').mean()
    anem_nice_str_no_anem_B = {i:a for i,a in anem_nice_str.items() if '_B' not in i}
    R2_of_avg_models = [find_nearest(df_results_R2[df_results_R2['anem']==anem]['ANN'], df_results_R2_mean[df_results_R2_mean.index==anem]['ANN'].array[0]) for anem in anem_nice_str_no_anem_B.values()]
    R2_of_avg_models = {i:R2_of_avg_models[j] for j,i in enumerate(anem_nice_str_no_anem_B.values())}
    # Adding column of bools, stating which one of the ANN models is the one closest to the mean (the one to be plotted later one as a fair expectation of performance)
    df_results_R2['closest_to_mean'] = False
    for idx, row in df_results_R2.iterrows():
        anem = row['anem']
        if row['ANN'] == R2_of_avg_models[anem]:
            df_results_R2.loc[idx, 'closest_to_mean'] = True
    df_results_acc['closest_to_mean'] = df_results_R2['closest_to_mean']

    # Patches for the fake plt.legend()
    ANN_countour_patch = mpatches.Patch(color='navajowhite', edgecolor=None, alpha=1., label='KDE of ANN predictions')
    ANN_each_patch = plt.plot([],[], color='orange', alpha=0.9, marker="o", markersize=markersize, ls="", label='All ANN models')
    ANN_plot_patch = plt.plot([], [], color='maroon', alpha=0.9, marker="o", markersize=markersize, ls="", label='Plotted models (Fig. 4)')
    EC_patch = plt.plot([], [], color='green', alpha=0.9, lw=3.0, label='NS-EN 1991-1-4')
    plt.close()

    # # R2 Plot
    # fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(7,5), dpi=400)
    # sns.set(style="whitegrid")
    # sns.barplot(x="anem", y='NS-EN 1991-1-4', data=df_results_R2, ci=0, capsize=0.75, errwidth=3, zorder=0.99, alpha=0., facecolor=(1, 1, 1, 0.5), errcolor='green', saturation=1.0)
    # sns.violinplot(x="anem", y="ANN", data=df_results_R2, inner=None, color='navajowhite', saturation=1.0, bw=bw, scale='width', linewidth=0.1)
    # plt.setp(ax.collections, alpha=.6)  # setting the transparency
    # # # sns.boxplot(x='anem', y='values', hue='Method', data=df_results_R2_melted)
    # # sns.catplot(x='anem', y='values', hue='Method', data=df_results_R2_melted, kind='violin', inner=None, alpha=0.5)
    # sns.swarmplot(x="anem", y="ANN", hue='closest_to_mean', data=df_results_R2, alpha=0.9, size=markersize, palette=['orange','maroon'])
    # plt.xticks(rotation=30)
    # plt.xlabel('')
    # plt.ylabel('$R^2$')
    # plt.ylim([-0.7, 1.0])
    # # plt.legend(handles=[*ANN_each_patch, *ANN_plot_patch, ANN_countour_patch, *EC_patch], loc=3, ncol=2)
    # ax.get_legend().remove()
    # sns.despine(left=True)
    # plt.tight_layout()
    # plt.show()


    # HORIZONTAL TEST
    # R2 Plot
    for _ in range(2):
        fig, axs = plt.subplots(sharex=True, sharey=True, figsize=(6,5), dpi=400)
        plt.xlim([-0.7, 1.0])
        sns.set(style="whitegrid", font_scale=font_scale)
        sns.barplot(y="anem", x='NS-EN 1991-1-4', data=df_results_R2, ci=0, capsize=0.75, errwidth=3, zorder=0.99, alpha=0., facecolor=(1, 1, 1, 0.5), errcolor='green', saturation=1.0)
        sns.violinplot(y="anem", x="ANN", data=df_results_R2, inner=None, color='navajowhite', saturation=1.0, bw=bw, scale='width', linewidth=0.2)
        plt.yticks(rotation=0)
        sns.despine(left=True)
        plt.tight_layout()
        fig.set_size_inches((6*2.6,5*2.6))  # increasing figsize before swarmplot and then shrinking it, leads to a nice overlay of points.
        sns.swarmplot(y="anem", x="ANN", hue='closest_to_mean', data=df_results_R2, alpha=0.7, edgecolor='black', linewidth=0.3, size=markersize, palette=['orange','maroon'])
        fig.set_size_inches((6, 5))
        axs.get_legend().remove()
        plt.ylabel('')
        plt.xlabel('$R^2$')
        plt.savefig(os.path.join(os.getcwd(), 'plots', f'violin_R2.png'))
        plt.show()
    fig, axs = plt.subplots(sharex=True, sharey=True, figsize=(5,5), dpi=400)
    sns.set(style="whitegrid", font_scale=font_scale)
    sns.barplot(y="anem", x='NS-EN 1991-1-4', data=df_results_acc, ci=0, capsize=0.75, errwidth=3, zorder=0.99, alpha=0., facecolor=(1, 1, 1, 0.5), errcolor='green', saturation=1.0)
    sns.violinplot(y="anem", x="ANN", data=df_results_acc, inner=None, color='navajowhite', saturation=1.0, bw=bw, scale='width', linewidth=0.2)
    plt.xlim([70,100])
    sns.despine(left=True)
    fig.set_size_inches((5*2.6, 5*2.6))
    sns.swarmplot(y="anem", x="ANN", hue='closest_to_mean', data=df_results_acc, alpha=0.7, edgecolor='black', linewidth=0.3, size=markersize, palette=['orange','maroon'])
    fig.set_size_inches((5, 5))
    axs.get_legend().remove()
    plt.yticks([0, 1, 2, 3, 4, 5], ["", "", "", "", "", ""])
    plt.ylabel('')
    plt.xlabel('Accuracy [%]')
    plt.legend(handles=[*ANN_each_patch, *ANN_plot_patch, ANN_countour_patch, *EC_patch], bbox_to_anchor=(0.6, 0.99))
    # plt.legend(handles=[*ANN_each_patch, *ANN_plot_patch, *EC_patch], bbox_to_anchor=(0.6, 0.99))
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plots', f'violin_Acc.png'))
    plt.show()

    print('Note: To know which ANN correspond to the "closest to the mean", open the dataframe df_results_R2 and notice the column "closest_to_mean"')

    # # Accuracy Plot
    # fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(7,5), dpi=400)
    # sns.set(style="whitegrid")
    # sns.barplot(x="anem", y='NS-EN 1991-1-4', data=df_results_acc, ci=0, capsize=0.75, errwidth=3, zorder=0.99, alpha=0., facecolor=(1, 1, 1, 0.5), errcolor='green', saturation=1.0)
    # sns.violinplot(x="anem", y="ANN", data=df_results_acc, inner=None, color='navajowhite', saturation=1.0, bw=bw, scale='width', linewidth=0.1)
    # plt.setp(ax.collections, alpha=.6)  # setting the transparency
    # # # sns.boxplot(x='anem', y='values', hue='Method', data=df_results_acc_melted)
    # # sns.catplot(x='anem', y='values', hue='Method', data=df_results_acc_melted, kind='violin', inner=None, alpha=0.5)
    # sns.swarmplot(x="anem", y="ANN", hue='closest_to_mean', data=df_results_acc, alpha=0.9, size=markersize, palette=['orange','maroon'])
    # plt.xticks(rotation=30)
    # plt.xlabel('')
    # plt.ylabel('Accuracy [%]')
    # plt.ylim([70,100])
    # plt.legend(handles=[*ANN_each_patch, *ANN_plot_patch, ANN_countour_patch, *EC_patch], loc=1, ncol=2)
    # sns.despine(left=True)
    # plt.tight_layout()
    # plt.show()

    pass


plot_R2_and_accuracies()


def plot_histograms_of_hyperparameters(n_final_tests_per_anem=20):
    from matplotlib.ticker import MaxNLocator
    import matplotlib.style
    import matplotlib as mpl
    mpl.style.use('default')
    font = {'family': 'DejaVu Sans',
            'size': 24}
    matplotlib.rc('font', **font)


    results = get_results_from_txt_file(n_final_tests_per_anem)

    # Converting all results into a dataframe where each row is an "optimal" set of hyperparameters, respective to one optimized ANN
    list_of_params = list(results['0'][0]['best_params'].keys())
    df_results_hp = pd.DataFrame(columns=list_of_params)
    for t in range(n_final_tests_per_anem):
        for a in range(6):
            df_results_hp = df_results_hp.append(pd.Series(dtype=float), ignore_index=True)  # add empty row
            for p in results[str(t)][a]['best_params'].keys():
                df_results_hp.iloc[-1][p] = results[str(t)][a]['best_params'][p]
    df_results_hp['loss'] = df_results_hp['loss'].astype(str).str[:-4]

    def plot_loghist(x, bins, color, alpha):
        hist, bins = np.histogram(x, bins=bins)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.hist(x, bins=logbins, color=color, alpha=alpha)
        plt.xscale('log')


    def plot_common_details():
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.box(False)
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        if bool(plt.gca().xaxis.get_label().get_text()):
            fig_name = plt.gca().xaxis.get_label().get_text()
        else:
            fig_name = plt.gca().xaxis.get_ticklabels()[0].get_text()
        plt.savefig(os.path.join(os.getcwd(), 'plots', f'{fig_name}.png'))
        plt.show()

    figsize_1 = (8, 4)
    figsize_2 = (6, 4)
    color='royalblue'
    alpha=0.6

    plt.figure(figsize=figsize_1)
    plot_loghist(df_results_hp['lr'], bins=20, color=color, alpha=alpha)
    plt.xlabel('Learning rate')
    plt.ylabel('Count')
    plot_common_details()

    plt.figure(figsize=figsize_1)
    plot_loghist(df_results_hp['weight_decay'], bins=20, color=color, alpha=alpha)
    plt.xlabel('Weight decay')
    plot_common_details()

    plt.figure(figsize=figsize_1)
    plt.hist(df_results_hp['momentum'], bins=20, color=color, alpha=alpha)
    plt.xlabel('Momentum')
    plot_common_details()

    plt.figure(figsize=figsize_2)
    plt.hist(df_results_hp['n_epochs'], bins=20, color=color, alpha=alpha)
    plt.ylabel('Count')
    plt.xlabel('Num. of epochs')
    plot_common_details()

    plt.figure(figsize=figsize_2)
    df_results_hp['n_hid_layers'].value_counts(sort=False).reindex([0,1,2,3,4], fill_value=0).plot.bar(rot=0, color=color, alpha=alpha)
    # plt.hist(df_results_hp['n_hid_layers'], bins=15, color=color, alpha=alpha)
    plt.xlabel('Num. of hid. layers')
    # plt.xlim([-0,4])
    plot_common_details()

    plt.figure(figsize=figsize_2)
    df_results_hp['activation'].value_counts(sort=True).plot.bar(rot=0, color=color, alpha=alpha)
    plot_common_details()

    # bar_order = ['L1', 'SmoothL1','MSE','LogCosh']
    bar_order = ['MSE', 'LogCosh','L1']
    plt.figure(figsize=figsize_2)
    df_results_hp['loss'].value_counts(sort=True).plot.bar(rot=0, color=color, alpha=alpha)
    # df_results_hp['loss'].value_counts(sort=True).loc[bar_order].plot.bar(rot=0, color=color, alpha=alpha)
    # label = plt.gca().axes.xaxis.get_majorticklabels()[-2]
    # dx = 0.1
    # offset = matplotlib.transforms.ScaledTranslation(dx, 0, plt.gcf().dpi_scale_trans)
    # label.set_transform(label.get_transform() + offset)  # offseting the last label to the right to fit the plot
    # label_2 = plt.gca().axes.xaxis.get_majorticklabels()[-1]
    # dx = 0.08
    # offset = matplotlib.transforms.ScaledTranslation(dx, 0, plt.gcf().dpi_scale_trans)
    # label_2.set_transform(label_2.get_transform() + offset)  # offseting the last label to the right to fit the plot

    plot_common_details()


plot_histograms_of_hyperparameters()




