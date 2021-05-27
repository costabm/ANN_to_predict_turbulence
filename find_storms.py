"""
Created: May, 2021
Contact: bercos@vegvesen.no
-----------------------------------------------------------------------------------------
First, run all functions in this file.
-----------------------------------------------------------------------------------------
To get the timestamps of each storm, run:
find_storm_timestamps()
-----------------------------------------------------------------------------------------
To create processed storm data (e.g. 10 min statistics), run:
create_storm_data_files(window='00:10:00')
-----------------------------------------------------------------------------------------
To compile all storm data files into one file, run:
compile_storm_data_files()
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from read_0p1sec_data import process_data_fun, compile_all_processed_data_into_1_file, from_V_NWZ_dict_to_V_Lw
from operator import itemgetter
from itertools import groupby


def consecutive_lists_within_list(list_with_consecutive_numbers):
    """
    Organizes a list of numbers into lists of consecutive numbers
    Example:
        consecutive_lists_within_list([1,2,3,6,7,10,20,21])
        returns: [[1, 2, 3], [6, 7], [10], [20, 21]]
    """
    consec_list = []
    for _, g in groupby(enumerate(list_with_consecutive_numbers), lambda x:x[0]-x[1]):
        consec_list.append(list(map(itemgetter(1), g)))
    return consec_list


def strong_wind_events_per_anem(ts, U, U_cond_1=13.9, U_cond_2=17.2):
    """
    Given a mean wind (U) array and its timestamps (ts), finds the consecutive timespans where the wind is always above U_cond_1 and reaches a maximum of at least U_cond_2.
    Each anemometer is looked at independently, through ts and U.
    :return: dict.
    """
    assert len(ts) == len(U)
    idxs_cond_1 = np.where(U_cond_1 <= U)[0]
    idxs_cond_1_consec_lists = consecutive_lists_within_list(idxs_cond_1)
    n_cond_1_events = len(idxs_cond_1_consec_lists)
    U_cond_1_consec_lists =  [U[idxs_cond_1_consec_lists[i]] for i in range(n_cond_1_events)]
    strong_wind_idxs = []
    for idxs, U_cond_1_event in zip(idxs_cond_1_consec_lists, U_cond_1_consec_lists):
        if any(U_cond_1_event >= U_cond_2):
            strong_wind_idxs.append(idxs)
    strong_wind_ts = [ts[i] for i in strong_wind_idxs]
    strong_wind_U  =  [U[i] for i in strong_wind_idxs]
    return {'ts': strong_wind_ts, 'U': strong_wind_U}


def strong_wind_events_per_anem_all_merged(input_fname='01-00-00_all_stats'):
    """
    All the individual strong wind events of each anemometer are merged together. This way, the timestamps where at least 1 anemometer is experiencing strong wind can be obtained.
    Only the mean wind
    Args:
        input_fname: The name of the file containing all the info necessary. e.g. '01-00-00_all_stats' or '00-10-00_all_stats'
    Returns: (df, dict) with all timestamps and wind speeds where at least 1 anem. feels strong wind, BUT, concomitant non-strong wind speeds (concomitant w/ strong winds elsewhere) are still missing!
    """
    with open(os.path.join(os.getcwd(), 'processed_data', input_fname), "r") as json_file:
        pro_data = json.load(json_file)
    # Finding strong wind events (also denoted here as storms) indepentently for each anemometer
    storm_df_all = pd.DataFrame(columns=['ts'])
    storm_dict_all = {}  # all anemometers from all masts
    for mast in ['osp1', 'osp2', 'svar', 'synn']:
        storm_dict_all[mast] = {}
        for anem in ['A', 'B', 'C']:
            storm_df = pd.DataFrame()
            ts = pd.to_datetime(pro_data[mast][anem]['ts'])
            U_Lw = from_V_NWZ_dict_to_V_Lw(V_NWZ_dict=pro_data[mast][anem]['means'])
            storm_dict_all[mast][anem] = strong_wind_events_per_anem(ts=ts , U=U_Lw[0])
            storm_df['ts'] = [time for time_list in storm_dict_all[mast][anem]['ts'] for time in time_list]
            storm_df[mast+'_'+anem] = [U for U_list in storm_dict_all[mast][anem]['U'] for U in U_list]
            # print(f'N storms in {mast}-{anem}: ' + str(len(storm_dict_all[mast][anem]['ts'])))
            storm_df_all = pd.merge(storm_df_all, storm_df, how="outer", on=["ts"])
    return storm_df_all.sort_values(by='ts', ignore_index=True), storm_dict_all


def listing_consecutive_time_stamps(list_of_timestamps, interval='01:00:00'):
    """
    Args:
        list_of_timestamps: One list of all timestamps
        interval: interval to define "consecutive"
    Returns: An organized list of lists of consecutive timestamps (e.g. a list of storms, where each storm is a small list of the consecutive timestamps of that storm)
    """
    big_list = []  # list of all small lists
    small_list = []  # one list of consecutive timestamps (e.g. timestamps of one storm)
    sorted_list_of_timestamps = sorted(list_of_timestamps)
    for t_prev, t in zip(sorted_list_of_timestamps[:-1], sorted_list_of_timestamps[1:]):
        small_list.append(t_prev)
        if t_prev + pd.to_timedelta(interval) != t:
            big_list.append(small_list)
            small_list = []
    # Last storm:
    small_list.append(t)
    big_list.append(small_list)
    return big_list


def find_storm_timestamps():
    df_storms_concomit_wind_missing, dict_storms_concomit_wind_missing = strong_wind_events_per_anem_all_merged(input_fname='01-00-00_all_stats')
    ts_storms_all_in_one_list = df_storms_concomit_wind_missing['ts']
    ts_storms_organized = listing_consecutive_time_stamps(ts_storms_all_in_one_list, interval='01:00:00')
    return ts_storms_organized


def create_storm_data_files(window='00:10:00'):
    """
    This ensures that we also create wind data that does not necessarily qualify as strong wind, but that it occurs at the same time as (concomitant to) strong winds elsewhere.
    """
    ts_storms = find_storm_timestamps()
    for i, t_list in enumerate(ts_storms):
        t_start = t_list[0].strftime(format='%Y-%m-%d %H:%M:%S.%f')[:-5]
        t_end  = t_list[-1].strftime(format='%Y-%m-%d %H:%M:%S.%f')[:-5]
        process_data_fun(window=window, masts_to_read=['synn', 'osp1', 'osp2', 'svar'], date_from_read=t_start, date_to_read=t_end, raw_data_folder='D:\PhD\Metocean_raw_data',
                         include_fitted_spectral_quantities=False, check_data_has_same_lens=True, save_json=True, save_in_folder='processed_storm_data', save_fname_suffix='_storm_' + str(i + 1))
        print(f'Storm data is now processed, from {t_start} to {t_end}')


def compile_storm_data_files():
    compile_all_processed_data_into_1_file(data_str='storm_', save_str='00-10-00_all_storms', save_json=True, foldername='processed_storm_data')


def create_excel_with_all_storms():
    pass


def plot_storm_ws_per_anem(dict_storms_concomit_wind_missing):
    for mast in ['osp1', 'osp2', 'svar', 'synn']:
        for anem in ['A', 'B', 'C']:
            ts_storms = dict_storms_concomit_wind_missing[mast][anem]['ts']
            U_storms =  dict_storms_concomit_wind_missing[mast][anem]['U']
            plt.figure(figsize=(20,4), dpi=300)
            for ts, U in zip(ts_storms, U_storms):
                plt.plot(ts, U, alpha=0.9, lw=0.2, color='blue')
                plt.axvspan(xmin=ts[0], xmax=ts[-1], color='orange')
                # print(np.max(U), ts[0], ts[-1])
            # plt.plot(pd.to_datetime(pro_data['osp1']['A']['ts']), U_Lw[1], label='v')
            # plt.plot(pd.to_datetime(pro_data['osp1']['A']['ts']), U_Lw[2], label='w')
            plt.xlim(pd.DatetimeIndex(['2015-02-01 00:00:00', '2020-05-01 00:00:00']))
            plt.show()



# TRASH
# FINDING PROBLEM WITH SYNNØYTANGEN WIND SPEEDS (TOO HIGH!!)
# from read_0p1sec_data import read_0p1sec_data_fun
# synn_data = read_0p1sec_data_fun(masts_to_read=['synn'], date_from_read='2019-03-27 01:20:00', date_to_read='2019-03-27 01:30:00', raw_data_folder='D:\PhD\Metocean_raw_data')
# svar_data = read_0p1sec_data_fun(masts_to_read=['svar'], date_from_read='2015-05-26 11:00:00', date_to_read='2015-05-26 12:00:00', raw_data_folder='D:\PhD\Metocean_raw_data')
# # Time period full of bad data that is hard to find!
# synn_data = read_0p1sec_data_fun(masts_to_read=['synn'], date_from_read='2019-01-03 12:19:30', date_to_read='2019-01-03 12:20:30', raw_data_folder='D:\PhD\Metocean_raw_data')
# # synn_data['synn'][synn_data['synn']['Sonic_A_U_Axis_Velocity'] > 2]
# synn_data['synn']['Sonic_A_U_Axis_Velocity'].isna().sum()
# test_rolling_mean = synn_data['synn'].drop('TIMESTAMP', axis=1).rolling(window=50, min_periods=10, center=True, closed='right').mean()
# test_rolling_std  = synn_data['synn'].drop('TIMESTAMP', axis=1).rolling(window=50, min_periods=10, center=True, closed='right').std()
# test_rolling_zscore = (synn_data['synn'].drop('TIMESTAMP', axis=1) - test_rolling_mean) / test_rolling_std



