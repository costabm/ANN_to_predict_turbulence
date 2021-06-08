import datetime
import numpy as np
import pandas as pd
from read_0p1sec_data import create_processed_data_files, compile_all_processed_data_into_1_file
from find_storms import create_storm_data_files, compile_storm_data_files, find_storm_timestamps, organized_dataframes_of_storms
import matplotlib.pyplot as plt

from elevation_profile_generator import elevation_profile_generator, plot_elevation_profile, get_point2_from_point1_dir_and_dist


# create_processed_data_files(date_start=datetime.datetime.strptime('2018-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f'), n_months=12*6, window='00:10:00', save_in_folder='processed_data')
# compile_all_processed_data_into_1_file(data_str='01-00-00_stats', save_str='01-00-00_all_stats', save_json=True, foldername='processed_data')
# create_storm_data_files(window='00:10:00', input_fname='01-00-00_all_stats')
# compile_storm_data_files(save_str='00-10-00_all_storms')

# # Getting post-processed and organized storm data
# storm_df_all_means, storm_df_all_dirs, storm_df_all_Iu, storm_df_all_Iv, storm_df_all_Iw, storm_df_all_avail = organized_dataframes_of_storms(foldername='processed_storm_data', compiled_fname='00-10-00_all_storms')
#
# # Plotting U, Iu and sigma_U along the fjord from measurements nearby (at different heights!):
# delta_dir = 30
# dir_separators = np.arange(0,360.001, delta_dir)
# for dir_min, dir_max in zip(dir_separators[:-1], dir_separators[1:]):
#     print(dir_min, dir_max)
#     idxs_where_storm = (dir_min < storm_df_all_dirs.mean(axis=1)) & (storm_df_all_dirs.mean(axis=1) < dir_max)
#     storm_means = storm_df_all_means[idxs_where_storm]
#     storm_Iu = storm_df_all_Iu[idxs_where_storm]
#     x_pos = [0, 2400, 2600, 5000]  # todo: improve this..
#     x_labels = ['synn_C', 'osp1_C', 'osp2_C', 'svar_C']
#     plt.figure(figsize=(5,10))
#     plt.title(f'Dir between {dir_min} and {dir_max}')
#     plt.errorbar(x_pos, y=storm_Iu.mean()[x_labels], yerr=storm_Iu.std()[x_labels], color='blue', label='Iu')
#     plt.errorbar(x_pos, y=storm_means.mean()[x_labels] / 20, yerr=storm_means.std()[x_labels] / 20, color='orange', label='U / 20')
#     plt.errorbar(x_pos, y=(storm_Iu[x_labels] * storm_means[x_labels]).mean(), yerr=(storm_Iu[x_labels] * storm_means[x_labels]).std()[x_labels] / 20, color='green', label='sigmaU')
#     plt.ylim(bottom=0)
#     plt.legend()
#     plt.show()
#
#
# # Obtaining wind profile at each mast:
#
#
#
#
# # Checking variability in the std_u, for different U and different dir
# U_min = 12
# U_max = 15
# delta_dir = 20
# dir_separators = np.arange(0, 360.001, delta_dir)
# mast_list = ['synn', 'osp1', 'osp2', 'svar']
# anem_list = ['A', 'B', 'C']
# for mast in mast_list:
#     plt.figure(figsize=(3.7, 3.7), dpi=800)
#     ax = plt.subplot(111, projection='polar')
#     ax.set_title(mast)
#     ax.set_theta_zero_location("N")
#     ax.set_theta_direction(-1)
#     for anem in anem_list:
#         mast_anem = mast + '_' + anem
#         dir_centre = []
#         Iu_mean = []
#         Iu_std = []
#         u_std_mean = []
#         u_std_std = []
#         for dir_min, dir_max in zip(dir_separators[:-1], dir_separators[1:]):
#             dir_centre.append((dir_max + dir_min)/2)
#             idxs_where_cond = (dir_min <= storm_df_all_dirs[mast_anem]) & (storm_df_all_dirs[mast_anem] <= dir_max) & (U_min <= storm_df_all_means[mast_anem]) & (storm_df_all_means[mast_anem] <= U_max)
#             Iu_mean.append(storm_df_all_Iu[mast_anem][idxs_where_cond].mean())
#             Iu_std.append(storm_df_all_Iu[mast_anem][idxs_where_cond].std())
#             u_std = (storm_df_all_Iu[mast_anem][idxs_where_cond]).multiply(storm_df_all_means[mast_anem][idxs_where_cond])
#             u_std_mean.append(u_std.mean())
#             u_std_std.append(u_std.std())
#         # ax.errorbar(np.deg2rad(dir_centre), Iu_mean, yerr=Iu_std)
#         ax.errorbar(np.deg2rad(dir_centre), u_std_mean, yerr=u_std_std, label=anem)
#     plt.legend()
#     plt.show()




# Wind-aligned terrain profiles
point_1 = [-34625., 6700051.]
step = 10.  # meters
point_2 = get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=160, distance=5000)
dists, heights = elevation_profile_generator(point_1=point_1, point_2=point_2, step_distance=step)
plot_elevation_profile(point_1=point_1, point_2=point_2, step_distance=step)
# Getting all 10-min data
min10_df_all_means, min10_df_all_dirs, min10_df_all_Iu, min10_df_all_Iv, min10_df_all_Iw, min10_df_all_avail = organized_dataframes_of_storms(foldername='processed_data', compiled_fname='00-10-00_all_stats')

# Discarding all data that are not at Z=48m
U_min = 5
columns_to_drop = ['ts', 'osp1_C', 'osp2_C', 'svar_B','svar_C','synn_B','synn_C']
X_means_df = min10_df_all_means.drop(columns=columns_to_drop)
X_dirs_df =  min10_df_all_dirs.drop(columns=columns_to_drop)
X_Iu_df = min10_df_all_Iu.drop(columns=columns_to_drop)
# Discarding all data with U below U_min
idxs_where_cond = (U_min <= X_means_df)  # todo: include the dir intervals here when removing some separating some data from the training into testing
X_means_df = X_means_df[idxs_where_cond].dropna(axis=0, how='all')
X_dirs_df  = X_dirs_df[idxs_where_cond].dropna(axis=0, how='all')
X_Iu_df    = X_Iu_df[idxs_where_cond].dropna(axis=0, how='all')
X_std_u_df = X_Iu_df.multiply(X_means_df)
# Organizing the data into an input matrix (with shape shape (n_samples, n_features)):
n_df_rows_to_use = 80000
X_train = []
y_train = []
mast_UTM_33 = {'synn':[-34515., 6705758.], 'osp1':[-39375., 6703464.], 'osp2':[-39350., 6703204.], 'svar':[-34625., 6700051.]}
for mast_anem in ['osp1_A', 'osp1_B', 'osp2_A' 'osp2_B', 'svar_A', 'synn_A']:
    X_mean_anem = X_means_df[mast_anem].dropna()
    X_dir_anem = X_dirs_df[mast_anem].dropna()
    X_std_u_anem = X_std_u_df[mast_anem].dropna()
    point_1 = mast_UTM_33[mast_anem[:4]]
    points_2 = np.array([get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=d, distance=5000) for d in X_dir_anem])
    heights = [elevation_profile_generator(point_1=point_1, point_2=p2, step_distance=step)[1] for p2 in points_2[227:229]]
    X_train_1_sample =
    X_train.append(X_train_1_sample)
    X_means_df[:n_df_rows_to_use]


# masts locations in UTM 33:




