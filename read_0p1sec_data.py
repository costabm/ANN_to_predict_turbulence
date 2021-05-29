"""
Created: May, 2021
Contact: bercos@vegvesen.no
-----------------------------------------------------------------------------------------
First, run all functions in this file.
-----------------------------------------------------------------------------------------
To read the raw 10 Hz data in a specific data interval, run e.g.:
read_0p1sec_data_fun(masts_to_read=['synn','osp1','osp2','svar'], date_from_read='2017-12-21 00:15:00.0', date_to_read='2017-12-21 05:15:00.0', raw_data_folder='D:\PhD\Metocean_raw_data')
-----------------------------------------------------------------------------------------
To process the 10 Hz data into e.g. 1h or 10min data (window='01:00:00' or window='00:10:00', respectively), storing it as json files (one file per month), in the consistent coordinate system
"to_North", "to_West", "to_Zenith", run e.g.:
create_processed_data_files(date_start=datetime.datetime.strptime('2015-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f'), n_months=12*6, window='01:00:00', save_in_folder='processed_data')
-----------------------------------------------------------------------------------------
After having all the json files of the processed data, to compile them all into one file, run e.g.:
compile_all_processed_data_into_1_file(data_str='01-00-00_stats', save_str='01-00-00_all_stats', save_json=True, foldername='processed_data')
"""

import numpy as np
import pandas as pd
import os
import datetime
import more_itertools
import json
from dateutil.relativedelta import relativedelta
import scipy.stats


def empty_dataframe():
    return pd.DataFrame(columns=['TIMESTAMP', 'Sonic_A_U_Axis_Velocity', 'Sonic_A_V_Axis_Velocity', 'Sonic_A_W_Axis_Velocity',
                                 'Sonic_B_U_Axis_Velocity', 'Sonic_B_V_Axis_Velocity', 'Sonic_B_W_Axis_Velocity',
                                 'Sonic_C_U_Axis_Velocity', 'Sonic_C_V_Axis_Velocity', 'Sonic_C_W_Axis_Velocity'])


def remove_outliers_using_zscore(df, window_to_test_zscore=100, min_periods=30, prob_interval_to_keep = 0.9999, warn_non_normal_data=True, warn_path='C:\example_folder'):
    """
    Calculates the probability (using the "Z score") of each wind measurement, by using the moving mean and moving STD of the measurements (locally Gaussian) in a window centred at that observation.
    Highly unlikely values (with probability of ocurring == 1 - prob_interval_to_keep) are converted to 'nan'
    Args:
        df: dataframe of wind raw data, already given for a specific mast
        window_to_test_zscore: number of elements to use in the rolling window. e.g. 100 (= 10 sec). Window should be more than double min_periods to avoid many unnecessary 'nan'
        min_periods: Minimum size of valid (!=nan) samples inside the window. Should be at least 30 for the zscore to be considered meaningful
        prob_interval_to_keep: e.g. 0.9999 -> Only values in the interval with 99.99% chance of ocurring are considered. Otherwise, convert to 'nan'
        warn_non_normal_data: True -> print the mean and std of the zscores observed whenever they are very different from those expected
        warn_path: Pass a path to be printed, for easily finding where this dataframe was obtained from
    Returns: dataframe where outliers are replaced by 'nan'
    """
    zscore_min, zscore_max = scipy.stats.norm.interval(prob_interval_to_keep)
    df_no_time = df.drop('TIMESTAMP', axis=1)
    df_rolling = df_no_time.rolling(window=window_to_test_zscore, min_periods=min_periods, center=True)  # min_periods =
    rolling_mean = df_rolling.mean()
    rolling_std = df_rolling.std()
    rolling_zscore = (df_no_time - rolling_mean) / rolling_std
    df_new = df_no_time[(zscore_min < rolling_zscore) & (rolling_zscore < zscore_max)]
    df_new.insert(0, 'TIMESTAMP', df['TIMESTAMP'])
    if warn_non_normal_data:
        zscore_mean = rolling_zscore.mean()
        zscore_std  = rolling_zscore.std()
        if any(zscore_mean > 0.1) or any(zscore_mean < -0.1) or any(zscore_std < 0.8) or any(zscore_std > 1.2):
            print(f"Warning: See the file {warn_path}. Data between {df['TIMESTAMP'].iloc[0]} and {df['TIMESTAMP'].iloc[-1]} should not be assumed normally-distributed, because:")
            print('Mean of zscores (should be 0):')
            print(zscore_mean)
            print('STD of zscores (should be 1):')
            print(zscore_std)
    return df_new


def read_0p1sec_data_fun(masts_to_read=['synn','osp1','osp2','svar'], date_from_read='2017-12-21 00:00:00.0', date_to_read='2017-12-21 05:00:00.0', raw_data_folder='D:\PhD\Metocean_raw_data',
                         remove_outliers=True):
    """
    :param masts_to_read: Choose from: 'osp1', 'osp2', 'svar', 'synn'
    :param date_from_read: date/time to read from
    :param date_to_read: date/time to read to
    :param raw_data_folder: Due to lack of permissions, the O: folder cannot be used directly. An external SDD was used to store the raw data
    :param remove_outliers: To remove or not, the outliers. A Z score analysis is performed on 10 sec rolling window, and extreme values removed.
    :return: nested dictionary with all relevant data that satisfies the input requirements
    """
    # Reading the general excel files (that describe all raw files with info about each first and last timestamps), given: masts_to_read
    n_masts_to_read = len(masts_to_read)
    time_stamp_files = []
    for mast in masts_to_read:
        for file_str in os.listdir(raw_data_folder):
            if 'files_' + mast in file_str:
                time_stamp_files.append(pd.read_excel(os.path.join(raw_data_folder, file_str)))
    # Finding which raw data files need to be read, given: masts_to_read, date_from_read and date_to_read
    files_to_read = []
    for i in range(n_masts_to_read):
        files_to_read.append(time_stamp_files[i].loc[(date_from_read <= time_stamp_files[i]['Last timestep']) & (time_stamp_files[i]['First timestep'] <= date_to_read)]['Filename'])
    # Reading and finding the data that satisfies the masts and dates given
    df_final = {}
    for mast in range(n_masts_to_read):
        df_temp = []  # temporary list of raw files of one mast, to be concatenated
        if len(files_to_read[mast]):
            for file in range(len(files_to_read[mast])):
                file_path = os.path.join(raw_data_folder, masts_to_read[mast], files_to_read[mast].iloc[file])
                try:
                    df = pd.read_csv(file_path, delimiter=',', skiprows=[0, 2, 3], na_values=['NAN','1000'],
                                     usecols=['TIMESTAMP','Sonic_A_U_Axis_Velocity','Sonic_A_V_Axis_Velocity','Sonic_A_W_Axis_Velocity',
                                                          'Sonic_B_U_Axis_Velocity','Sonic_B_V_Axis_Velocity','Sonic_B_W_Axis_Velocity',
                                                          'Sonic_C_U_Axis_Velocity','Sonic_C_V_Axis_Velocity','Sonic_C_W_Axis_Velocity'],
                                     dtype={'TIMESTAMP':str, 'Sonic_A_U_Axis_Velocity': np.float32, 'Sonic_A_V_Axis_Velocity': np.float32, 'Sonic_A_W_Axis_Velocity': np.float32,
                                                             'Sonic_B_U_Axis_Velocity': np.float32, 'Sonic_B_V_Axis_Velocity': np.float32, 'Sonic_B_W_Axis_Velocity': np.float32,
                                                             'Sonic_C_U_Axis_Velocity': np.float32, 'Sonic_C_V_Axis_Velocity': np.float32, 'Sonic_C_W_Axis_Velocity': np.float32})
                    # date_from_read_omit_p0 = date_from_read[:-2] if date_from_read[-2:] == '.0' else date_from_read
                    # date_to_read_omit_p0   =   date_to_read[:-2] if   date_to_read[-2:] == '.0' else   date_to_read
                    df = df[(df['TIMESTAMP'] >= date_from_read) & (df['TIMESTAMP'] <= date_to_read)]  # droping all rows outside requested datetime interval
                    if remove_outliers:
                        df = remove_outliers_using_zscore(df, warn_non_normal_data=True, warn_path=file_path)
                    df_temp.append(df)
                except ValueError:
                    print(f'ValueError encountered when reading csv file. Discarding the following dataframe: {file_path}. Using empty dataframe instead.')
                    default_view_n_columns = pd.get_option("display.max_columns")
                    pd.set_option('display.max_columns', None)
                    print(pd.read_csv(file_path, delimiter=',', skiprows=[0, 2, 3], na_values='NAN'))
                    pd.set_option('display.max_columns',default_view_n_columns)
                    df_empty = empty_dataframe()
                    df_temp.append(df_empty)
        else:
            df_empty = empty_dataframe()
            df_temp.append(df_empty)
        df_final[masts_to_read[mast]] = pd.concat(df_temp).drop_duplicates(subset='TIMESTAMP').sort_values(by=['TIMESTAMP'])
    return df_final


def R_z(alpha):
    """Rotation matrix around axis z"""
    return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                     [np.sin(alpha),  np.cos(alpha), 0],
                     [            0,              0, 1]])


def T_NWZ_Gill_fun(mast, anem):
    """
    Transformation matrix from the local fixed axes of the Gill anemometer "U" "V" "W" (note that U is not necessarily aligned with the mean wind) to the "N" (North) "W" ( West) "Z" (Zenith) axes.
    Note also that N and W mean wind velocities that go in that direction, i.e. TO North and TO West (not from)
    :param mast: str with mast name. 'synn', 'osp1', 'osp2', 'svar'
    :param anem: str with anemometer letter. 'A', 'B', 'C'
    """
    with open("mast_angles.txt", 'r') as f:  # angles between true North and the original unadjusted "U" axis
        mast_angles_deg = eval(f.read())
    mast_angles_rad = dict((key, np.deg2rad(value)) for key, value in mast_angles_deg.items())
    return np.transpose(R_z(mast_angles_rad[mast+'_'+anem]))


def T_uvw_NWZ_fun(mean_beta_cardinal):
    """
    Transformation matrix from NWZ to the Local mean wind wind (u,v,w), given a mean beta cardinal angle
    """
    return np.transpose(R_z(np.pi - mean_beta_cardinal))


def beta_cardinal_given_speed_towards_N_and_W(N,W):
    """
    Calculates the cardinal direction [0,2*pi[, given a vector with given values in the "N" North and "W" West axes.
    0 is North, rotating clockwise.
    :param N: North-axis. Use velocity TO North (not FROM North!)
    :param W: West-axis. Use velocity TO WEST (not FROM West!)
    :return: angle where the wind comes FROM, in radians
    ---------------------------------------------------------------
    Can by tested by:
    print(np.rad2deg(beta_cardinal_given_speed_towards_N_and_W(N=1, W=0)))
    print(np.rad2deg(beta_cardinal_given_speed_towards_N_and_W(N=0, W=-1)))
    print(np.rad2deg(beta_cardinal_given_speed_towards_N_and_W(N=-1, W=0)))
    print(np.rad2deg(beta_cardinal_given_speed_towards_N_and_W(N=0, W=1)))
    print(np.rad2deg(beta_cardinal_given_speed_towards_N_and_W(N=1, W=0.001)))
    Which gives:
    180.0 (wind goes to North, so comes from South)
    270.0
    0.0
    90.0
    179.9427042395855
    """
    angle = np.arctan2(W , -N)
    where_negative = angle < 0  # where the angle is negative...
    adjust_2pi = where_negative * 2*np.pi  # ... adding 2*np.pi is required
    return angle + adjust_2pi


def from_V_NWZ_dict_to_V_Lw(V_NWZ_dict, also_return_betas_c=False):
    """
    Args:
        V_NWZ_dict: e.g. pro_data[mast][anem]['means'] or storm_dict[mast][anem]['means']
        also_return_betas_c: Whether to include betas_c in the return or not
    """
    V_NWZ_N = np.array(V_NWZ_dict['to_North'])
    V_NWZ_W = np.array(V_NWZ_dict['to_West'])
    V_NWZ_Z = np.array(V_NWZ_dict['to_Zenith'])
    V_NWZ = np.array([V_NWZ_N, V_NWZ_W, V_NWZ_Z])
    betas_c = beta_cardinal_given_speed_towards_N_and_W(V_NWZ[0], V_NWZ[1])
    T_uvw_NWZ = np.array([T_uvw_NWZ_fun(b) for b in betas_c])
    V_Lw = np.einsum('tij,jt->it', T_uvw_NWZ, V_NWZ, optimize=True)  # Lw - Local wind coordinate system (U+u, v, w)
    if also_return_betas_c:
        return V_Lw, betas_c
    else:
        return V_Lw


def check_if_ts_are_consecutive(list_of_timestamps, interval='01:00:00'):
    """
    :param list_of_timestamps: list
    :param interval: str. Use '01:00:00' for 1h. Use '00:10:00' for 10min
    :return: bool. True only if all timestamps are consecutive with the given constant interval
    """
    datetimes = [datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in list_of_timestamps]
    datetime_interval = datetime.datetime.strptime(interval, '%H:%M:%S') - datetime.datetime.strptime('00:00:00', '%H:%M:%S')
    bools = []
    for prev, now in zip(datetimes[:-1], datetimes[1:]):
        bools.append(now - prev == datetime_interval)
    idxs_of_Falses = np.where(~np.array(bools))[0]
    return all(bools), idxs_of_Falses


def check_max_n_of_consecutive_NaN(df):
    """
    :param df: one of the raw dataframes from MetOcean. e.g. read_0p1sec_data_fun()['osp1']
    :return: The maximum number of consecutive rows where at least one nan value is found, using o, f
    """
    inds = np.where(pd.isna(df).any(1))[0]
    return max([len(list(group)) for group in more_itertools.consecutive_groups(inds)])


def save_processed_data(processed_data, foldername='processed_data', fname='data_10min'):
    with open(os.path.join(os.getcwd(), foldername, fname), "w") as fp:
        json.dump(processed_data, fp)


def next_possible_time(dt, possible_stamps):
    """
    From a list of possible precise timestamps, returns the first stamp that is immediatelly larger (but not equal!)
    """
    for s in possible_stamps:
        if s > dt.time():
            return datetime.datetime.combine(dt.date(), s)  # current stamp


def prev_or_current_possible_time(dt, possible_stamps):
    """
    From a list of possible precise timestamps, returns the closest stamp that is immediatelly smaller or equal
    """
    for s_prev, s in zip(possible_stamps[:-1], possible_stamps[1:]):
        if s > dt.time():  # if s is larger...
            return datetime.datetime.combine(dt.date(), s_prev)  # ...then return the previous one!


def get_list_of_precise_timestamps(precision='01:00:00'):
    """
    Generates a list of possible precise timestamps, from '00:00:00' to '23:59:59', with an interval = precision
    """
    dt_precision = datetime.datetime.strptime(precision, '%H:%M:%S') - datetime.datetime.strptime('00:00:00', '%H:%M:%S')
    assert datetime.timedelta(seconds=3600*24) % dt_precision == datetime.timedelta(0), "precision needs to be an integer factor of a 24 h period"
    n_possible_stamps = int((datetime.datetime.strptime('23:59:59', '%H:%M:%S') - datetime.datetime.strptime('00:00:00', '%H:%M:%S')) / dt_precision)
    return [(datetime.datetime.strptime('00:00:00', '%H:%M:%S') + dt_precision * i).time() for i in range(n_possible_stamps+1)]


def get_list_of_precise_datetimes_between_2_dts(dt1, dt2, precision='01:00:00'):
    possible_timestamps = get_list_of_precise_timestamps(precision=precision)
    dt1_next = next_possible_time(dt1, possible_timestamps)
    dt2_prev = prev_or_current_possible_time(dt2, possible_timestamps)
    dt_precision = datetime.datetime.strptime(precision, '%H:%M:%S') - datetime.datetime.strptime('00:00:00', '%H:%M:%S')
    n_possible_stamps = int((dt2_prev - dt1_next) / dt_precision)
    return [dt1_next + dt_precision*i for i in range(n_possible_stamps+1)]


def check_processed_data_has_same_len(data_processed):
    """
    :param data_processed: obtained from process_data_fun()
    :return: True if all 10min or 1h data have the same number of entries, for all masts and anemometers
    """
    # data_processed_copy = {i:data_processed[i] for i in data_processed if i in 'c'}
    list_of_lens = [[len(data_processed[v1][v2]['ts']) for v2 in data_processed[v1]] for v1 in data_processed]
    if len(set(np.array(list_of_lens).flatten())) == 1:
        return True
    else:
        print([[len(data_processed[m][a]['means']) for a in data_processed[m]] for m in data_processed])
        return False


def fitted_spectral_quantities_to_raw_data():
    # To be implemented
    pass


def process_data_fun(window='01:00:00', masts_to_read=['synn', 'osp1', 'osp2', 'svar'], date_from_read='2017-12-21 00:00:00.0', date_to_read='2017-12-21 05:00:00.0',
                     raw_data_folder='D:\PhD\Metocean_raw_data', include_fitted_spectral_quantities=False, check_data_has_same_lens=True, save_json=False, save_in_folder='processed_data',
                     save_fname_suffix=''):
    """
    :param window: str. Use '01:00:00' for 1h statistics. Use '00:10:00' for 10-min statistics. Other windows are possible as long as they are an integer factor of a 24 h period.
    :param masts_to_read: list. Choose from: 'osp1', 'osp2', 'svar', 'synn'
    :param date_from_read: str. date/time to read from
    :param date_to_read: str. date/time to read to
    :param include_fitted_spectral_quantities: bool
    :param check_data_has_same_lens:  bool
    :param save_json: bool
    :param raw_data_folder: str. Due to lack of permissions, the O: folder cannot be used directly. An external SDD was used to store the raw data
    :param save_in_folder: string to append to the file name when saving the data if save_json=True
    :param save_fname_suffix: e.g. '_storm_1' for storm 1
    :return: nested dictionary with all 1-hour (or 10-min, or another interval) mean, covariance, and timestamp data, at the desired masts and time interval
    """
    data = read_0p1sec_data_fun(masts_to_read=masts_to_read, date_from_read=date_from_read, date_to_read=date_to_read, raw_data_folder=raw_data_folder)
    data_processed = {}
    for mast in masts_to_read:
        data_processed[mast] = {}
        for anem in ['A', 'B', 'C']:
            # print(f'Max time with consecutive NaNs: {check_max_n_of_consecutive_NaN(data[mast])/10} seconds')  # Makes it 10% slower
            ts = data[mast]['TIMESTAMP']
            dt1 = datetime.datetime.strptime(date_from_read, '%Y-%m-%d %H:%M:%S.%f')
            dt2 = datetime.datetime.strptime(date_to_read,   '%Y-%m-%d %H:%M:%S.%f')
            all_precise_timestamps = get_list_of_precise_datetimes_between_2_dts(dt1, dt2, precision=window)
            all_precise_timestamps_str = [all_precise_timestamps[i].strftime('%Y-%m-%d %H:%M:%S') for i in range(len(all_precise_timestamps))]
            # The next line is MUCH FASTER than using np.where(np.isin(ts_with_start,all_precise_timestamps_str))[0]:
            _measured_precise_timestamps, measured_precise_time_idxs, _ = np.intersect1d(ts, all_precise_timestamps_str, assume_unique=True, return_indices=True)
            # Because we need all the data from idx=0 up to the first measured_precise_time_idx, we add 0 to this list (actually we use -1 instead of 0 because prev_time_idx+1 will be used)
            measured_precise_time_idxs_with_start = [-1] + measured_precise_time_idxs.tolist()  #
            # Converting wind raw data in arbitrary axes to Lw - Local Wind axes
            U_Gill = data[mast][f'Sonic_{anem}_U_Axis_Velocity'].values
            V_Gill = data[mast][f'Sonic_{anem}_V_Axis_Velocity'].values
            W_Gill = data[mast][f'Sonic_{anem}_W_Axis_Velocity'].values
            V_KVT = np.array([U_Gill, V_Gill, W_Gill])
            V_NWZ = T_NWZ_Gill_fun(mast, anem) @ V_KVT  # North-West-Zenith coordinate system
            timestamps = []
            means = []
            cov = []
            availability = []
            for prev_time_idx, time_idx in zip(measured_precise_time_idxs_with_start[:-1], measured_precise_time_idxs_with_start[1:]):
                assert prev_time_idx <= time_idx, "Somehow there are still duplicate rows?"
                V_NWZ_chunk = V_NWZ[:, prev_time_idx+1:time_idx+1]  # note that time_idx should be included in the calc, but not prev_time_idx
                if not np.isnan(V_NWZ_chunk).all():  # if there is at least some data
                    V_NWZ_chunk_df = pd.DataFrame({'to_North': V_NWZ_chunk[0], 'to_West': V_NWZ_chunk[1], 'to_Zenith': V_NWZ_chunk[2]})
                    timestamps.append(ts.iloc[time_idx])
                    means.append(V_NWZ_chunk_df.mean().to_numpy(dtype='float32').tolist()) # To calculate the covariance matrix with nan values in the mix, pd.cov is necessary.
                    cov.append(V_NWZ_chunk_df.cov().to_numpy(dtype='float32').tolist())
                    availability.append((1-V_NWZ_chunk_df.isna().mean()).to_numpy(dtype='float32').tolist())
                    if include_fitted_spectral_quantities:
                        fitted_spectral_quantities_to_raw_data()  # then Lw coordinates should be stored perhaps, instead of NWZ...?
                        raise NotImplementedError
            data_processed[mast][anem] = {'ts':timestamps, 'means':means, 'covar':cov, 'availability':availability}  #, 'missedstamps':missed_timestamps, 'description':description}
    if check_data_has_same_lens:
        if not check_processed_data_has_same_len(data_processed):
            print("The 10min/1h data has different size for different masts/anemometers!")
    if save_json:
        fname = (window + '_stats_' + date_from_read[:-2] + '_' + date_to_read[:-2] + save_fname_suffix).replace(" ", "_").replace(":", "-")
        save_processed_data(data_processed, foldername=save_in_folder, fname=fname)
    return data_processed


def create_processed_data_files(date_start=datetime.datetime.strptime('2015-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f'), n_months=12 * 6, window='01:00:00', save_in_folder='processed_data'):
    """
    Generates many processed data files, one month long each, given a start date and n_monts.
    Args:
        date_start: choose any date before the measurements start
        window: time window of data processing. e.g. '01:00:00' or '00:10:00' for 1h or 10min statistics
        n_months: num of months
        save_in_folder: folder name
    """
    dates_list = [date_start + relativedelta(months=1*m) for m in range(n_months+1)]
    for dt1, dt2 in zip(dates_list[:-1], dates_list[1:]):
        dt1_str = dt1.strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]
        dt2_str = dt2.strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]
        process_data_fun(window=window, masts_to_read=['synn', 'osp1', 'osp2', 'svar'], date_from_read=dt1_str, date_to_read=dt2_str, save_json=True, save_in_folder=save_in_folder)
        print(f'Data is now processed, from {dt1_str} to {dt2_str}')


def create_empty_nested_dictionary():
    empty_dict = {}
    for m in ['osp1', 'osp2', 'svar', 'synn']:
        empty_dict[m] = {}
        for a in ['A', 'B', 'C']:
            empty_dict[m][a] = {}
            for x in ['ts', 'means', 'covar', 'availability']:
                if x == 'ts' or x== 'covar':
                    empty_dict[m][a][x] = []
                else:
                    empty_dict[m][a][x] = {}
                    for d in ['to_North', 'to_West', 'to_Zenith']:
                        empty_dict[m][a][x][d] = []
    return empty_dict


def compile_all_processed_data_into_1_file(data_str='01-00-00_stats', save_str='01-00-00_all_stats', save_json=True, foldername='processed_data'):
    """
    Merging all json files that include data_str in their name, into one single json file
    """
    all_fnames = os.listdir(os.path.join(os.getcwd(), foldername))
    all_1h_files = []
    for f in all_fnames:
        if data_str in f:
            with open(os.path.join(os.getcwd(), foldername, f), "r") as json_file:
                all_1h_files.append(json.load(json_file))
    pro_file = create_empty_nested_dictionary()
    for f in all_1h_files:
        for m in ['osp1', 'osp2', 'svar', 'synn']:
            for a in ['A','B','C']:
                for x in ['ts', 'means', 'covar', 'availability']:
                    if f[m][a][x]:  # if there is data
                        if x == 'ts' or x== 'covar':
                            pro_file[m][a][x] += f[m][a][x]
                        else:
                            for idx, d in enumerate(['to_North', 'to_West', 'to_Zenith']):
                                pro_file[m][a][x][d] += np.array(f[m][a][x]).T.tolist()[idx]
    if save_json:
        save_processed_data(processed_data=pro_file, foldername=foldername, fname=save_str)
    return pro_file


