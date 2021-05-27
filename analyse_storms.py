"""
Created: May, 2021
Contact: bercos@vegvesen.no
-----------------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import json
from read_0p1sec_data import compile_all_processed_data_into_1_file
from find_storms import find_storm_timestamps

ts_storm = find_storm_timestamps()

with open(os.path.join(os.getcwd(), 'processed_data', '00-10-00_all_storms'), "r") as json_file:
    storm_dict = json.load(json_file)



    storm_df_all_means = pd.DataFrame(columns=['ts'])
    storm_df_all_stds = pd.DataFrame(columns=['ts'])
    storm_df_all_availabilities = pd.DataFrame(columns=['ts'])
    for mast in ['osp1', 'osp2', 'svar', 'synn']:
        for anem in ['A', 'B', 'C']:
            storm_df = pd.DataFrame()
            storm_df['ts'] = [time for time_list in storm_dict[mast][anem]['ts'] for time in time_list]
            storm_df[mast+'_'+anem] = [U for U_list in storm_dict[mast][anem]['means'] for U in U_list]
            storm_df_all_means = pd.merge(storm_df_all_means, storm_df, how="outer", on=["ts"])


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



