import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import json
from read_0p1sec_data import beta_cardinal_given_speed_towards_N_and_W, T_uvw_NWZ_fun, next_possible_time, prev_or_current_possible_time
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


def strong_wind_events_1_anemometer(ts, U, U_cond_1=13.9, U_cond_2=17.2):
    """
    Given a mean wind (U) array and its timestamps (ts), finds the consecutive timespans where the wind is always above U_cond_1 and reaches a maximum of at least U_cond_2
    :return:
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


def strong_wind_events_func():
    pass

fname = '01-00-00_all_stats'
with open(os.path.join(os.getcwd(), 'processed_data', fname), "r") as json_file:
    pro_data = json.load(json_file)

# Finding strong wind events (also denoted here as storms) indepentently for each anemometer
storm_dict_all = {}  # all anemometers from all masts
for mast in ['osp1', 'osp2', 'svar', 'synn']:
    storm_dict_all[mast] = {}
    for anem in ['A', 'B', 'C']:
        ts = pd.to_datetime(pro_data[mast][anem]['ts'])
        V_NWZ_N =  np.array(pro_data[mast][anem]['means']['to_North'])
        V_NWZ_W =  np.array(pro_data[mast][anem]['means']['to_West'])
        V_NWZ_Z =  np.array(pro_data[mast][anem]['means']['to_Zenith'])
        V_NWZ = np.array([V_NWZ_N, V_NWZ_W, V_NWZ_Z])
        betas_c = beta_cardinal_given_speed_towards_N_and_W(V_NWZ[0], V_NWZ[1])
        T_uvw_NWZ = np.array([T_uvw_NWZ_fun(b) for b in betas_c])
        V_Lw = np.einsum('nij,jn->in', T_uvw_NWZ, V_NWZ, optimize=True)  # Lw - Local wind coordinate system (U+u, v, w)
        storm_dict_all[mast][anem] = strong_wind_events_1_anemometer(ts=ts , U=V_Lw[0])
        print(f'N storms in {mast}-{anem}: ' + str(len(storm_dict_all[mast][anem]['ts'])))
        # storm_idxs, storm_ts, storm_U = storm_dict['idxs'], storm_dict['ts'], storm_dict['U']
# Finding the general strong wind events, which satisfy the strond wind conditions at least at one anemometer:
df_test = pd.DataFrame(storm_dict_all['osp2']['A'])




ts_storms = storm_dict_all['synn']['A']['ts']
U_storms = storm_dict_all['synn']['A']['U']
n_storms = len(U_storms)

ts_flat = [item for sublist in ts_storms for item in sublist]
U_flat =  [item for sublist in U_storms for item in sublist]


plt.figure(figsize=(20,4), dpi=300)
for ts, U in zip(ts_storms, U_storms):
    plt.plot(ts, U, alpha=0.9, lw=0.2, color='blue')
    plt.axvspan(xmin=ts[0], xmax=ts[-1], color='orange')
# plt.plot(pd.to_datetime(pro_data['osp1']['A']['ts']), V_Lw[1], label='v')
# plt.plot(pd.to_datetime(pro_data['osp1']['A']['ts']), V_Lw[2], label='w')
# plt.xlim(event[0], event[-1])
plt.show()



# FINDING PROBLEM WITH SYNNØYTANGEN WIND SPEEDS (TOO HIGH!!)
[(np.where(U_storms[i] > 1000)[0],i) for i in range(n_storms)]
from read_0p1sec_data import read_0p1sec_data_fun
synn_data = read_0p1sec_data_fun(masts_to_read=['synn'], date_from_read='2019-01-03 12:00:00.0', date_to_read='2019-01-03 13:00:00.0', raw_data_folder='D:\PhD\Metocean_raw_data')

# synn_data['synn'][synn_data['synn']['Sonic_A_U_Axis_Velocity'] > 2]

synn_data['synn']['Sonic_A_U_Axis_Velocity'].isna().sum()
test_rolling_mean = synn_data['synn']['Sonic_A_U_Axis_Velocity'].rolling(window=10).mean()
test_rolling_mean.isna().sum()

pd.merge  # todo: merge all anemometers (check outer vs inner merge)

# Thought process:
# Storm 1 "s1":
# Find first point "s1_thresh_init" with wind speed > 17.2 m/s (gale).
# Find point "s1_max" with maximum wind speed between "s1_thresh_init" and a ~2 day window after it.
# Walk both backward and forward from "s1_max" until wind speed falls bellow 13.9 m/s (near gale), finding both points "s1_start" and "s2_end" that define the storm.
# Do the same for all anemometers. Label all timestamps as either strong wind or not. Add (do 'or') on the boolean labels to find all strong winds episodes that occur in at least 1 anem.
# Storm 2 "s2":
# Start looking after "s2_end".







# test 3