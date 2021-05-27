"""
Created: May, 2021
Contact: bercos@vegvesen.no
-----------------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import json
from read_0p1sec_data import compile_all_processed_data_into_1_file, from_V_NWZ_dict_to_V_Lw
from find_storms import find_storm_timestamps

ts_storm = find_storm_timestamps()

with open(os.path.join(os.getcwd(), 'processed_storm_data', '00-10-00_all_storms'), "r") as json_file:
    storm_dict = json.load(json_file)



    storm_df_all_means = pd.DataFrame(columns=['ts'])
    storm_df_all_dirs = pd.DataFrame(columns=['ts'])
    storm_df_all_stds = pd.DataFrame(columns=['ts'])
    storm_df_all_availabilities = pd.DataFrame(columns=['ts'])
    for mast in ['osp1', 'osp2', 'svar', 'synn']:
        for anem in ['A', 'B', 'C']:
            # Means
            storm_df_means = pd.DataFrame()
            storm_df_means['ts'] = [time for time in storm_dict[mast][anem]['ts']]
            U_Lw, betas_c = from_V_NWZ_dict_to_V_Lw(V_NWZ_dict=storm_dict[mast][anem]['means'], also_return_betas_c=True)
            storm_df_means[mast+'_'+anem] = U_Lw[0]
            storm_df_all_means = pd.merge(storm_df_all_means, storm_df_means, how="outer", on=["ts"])
            # Dirs
            storm_df_dirs = pd.DataFrame()
            storm_df_dirs['ts'] = storm_df_means['ts']
            storm_df_dirs[mast+'_'+anem] = betas_c
            storm_df_all_dirs = pd.merge(storm_df_all_dirs, storm_df_dirs, how="outer", on=["ts"])
            # STDs / OR INSTEAD COVS???
            storm_df_stds = pd.DataFrame()
            storm_df_stds['ts'] = storm_df_means['ts']
            storm_df_stds[mast+'_'+anem] = U_Lw[0]
            storm_df_all_stds = pd.merge(storm_df_all_stds, storm_df_stds, how="outer", on=["ts"])


#
#
# # TRASH
# storm_df, storm_dict = strong_wind_events_per_anem_all_merged(fname='00-10-00_all_storms')
# all_fnames = os.listdir(os.path.join(os.getcwd(), 'processed_data'))
# storm_str = 'storm'
# storm_pro_data = []
# for f in all_fnames:
#     if storm_str in f:
#         with open(os.path.join(os.getcwd(), 'processed_data', f), "r") as json_file:
#             storm_pro_data.append(json.load(json_file))


# # HOW TO TRANSFORM STANDARD DEVIATIONS FROM ONE SYSTEM TO ANOTHER (COVARIANCE MATRIX REQUIRED)
# from read_0p1sec_data import R_z
# import numpy as np
# V_1 = np.array([[8,12,14,20], [11,9,10,10], [-1, 0, 2,1]])
# T_21 = R_z(np.pi/4).T
# V_2 = T_21 @ V_1
# V_1_stds = np.std(V_1, axis=1)
# V_1_cov = np.cov(V_1, bias=True)
# V_2_stds = np.std(V_2, axis=1)
# V_2_stds_2nd_method = np.sqrt(np.diag(T_21 @ V_1_cov @ T_21.T))


