import numpy as np
import pandas as pd
from find_storms import find_storm_timestamps
from read_0p1sec_data import process_data_fun, raw_data_to_NWZ_fun
import matplotlib.pyplot as plt

storm_timestamps = find_storm_timestamps()
st_window_str = '00:10:00'
st_window_sec = 600  # seconds. Moving Window
mast_list = ['synn', 'osp1', 'osp2', 'svar']

storm_MSE = {}
for storm_id in range(17, len(storm_timestamps)):
    storm_MSE[str(storm_id)] = {}
    time_str_format = '%Y-%m-%d %H:%M:%S.%f'

    # print(storm_timestamps[storm_id])
    date_from_read = storm_timestamps[storm_id][ 0].strftime(time_str_format)[:-5]
    date_to_read   = storm_timestamps[storm_id][-1].strftime(time_str_format)[:-5]

    # Statationary (st) and Non-stationary (nst) Data:
    st_data = process_data_fun(window=st_window_str, masts_to_read=mast_list, date_from_read=date_from_read, date_to_read=date_to_read,
                                raw_data_folder='D:\PhD\Metocean_raw_data', include_fitted_spectral_quantities=False, check_data_has_same_lens=True, save_json=False)
    raw_data = raw_data_to_NWZ_fun(masts_to_read=mast_list, date_from_read=date_from_read, date_to_read=date_to_read, raw_data_folder='D:\PhD\Metocean_raw_data')

    for mast in mast_list:
        anem = 'A'
        fs = 10  # Hertz. Sampling frequency in the data
        st_window_sec = 600
        st_ts = pd.to_datetime(st_data[mast][anem]['ts'])
        if len(st_ts):
            st_ts_w_start = st_ts.insert(loc=0, item=st_ts[0] - pd.to_timedelta(st_window_str))
            st_V = np.array(st_data[mast]['A']['means']).T
            nst_ts = pd.to_datetime(raw_data[mast][anem]['ts'])
            nst_V = pd.DataFrame(raw_data[mast][anem]['V_NWZ'].T).rolling(st_window_sec * fs, center=True, min_periods=int(st_window_sec/2)).mean().to_numpy().T
            raw_V = raw_data[mast][anem]['V_NWZ']
            # Organizing data in one DataFrame:
            st_df =  pd.DataFrame({ 'ts':st_ts, 'block_num':np.arange(len(st_ts)), 'st_V_N':st_V[0], 'st_V_W':st_V[1], 'st_V_Z':st_V[2]})
            nst_df = pd.DataFrame({'ts':nst_ts, 'nst_V_N':nst_V[0], 'nst_V_W':nst_V[1], 'nst_V_Z':nst_V[2]})
            df_all = pd.merge(st_df, nst_df, how='outer', on='ts', sort=True)  # merging st and nst data.
            df_all[['st_V_N','st_V_W','st_V_Z','block_num']] = df_all[['st_V_N','st_V_W','st_V_Z','block_num']].fillna(method='backfill')
            df_all = df_all.dropna()
            df_all['block_num'] = df_all['block_num'].astype(int)
            # Finding out large non-stationarities:
            df_MSE = pd.DataFrame({'block_num': df_all['block_num'],
                                   'V_N':(df_all['nst_V_N']-df_all['st_V_N'])**2,
                                   'V_W':(df_all['nst_V_W']-df_all['st_V_W'])**2,
                                   'V_Z':(df_all['nst_V_Z']-df_all['st_V_Z'])**2}) # df of Mean Square Errors)
            df_MSE = df_MSE.groupby('block_num').sum()
            df_MSE['ts'] = st_ts
            storm_MSE[str(storm_id)][mast+'_'+anem] = df_MSE
            # Confirming the merging is well done (exact same graph obtained, but with each horizontal bar connected to the next one):
            if any((df_MSE['V_N'] + df_MSE['V_W']) > 100000):
                NWZ_idx = 0
                plt.figure(dpi=400)
                plt.title(f'Storm: {storm_id}. Mast: {mast}')
                for i, (st_ts1, st_ts2) in enumerate(zip(st_ts_w_start[:-1], st_ts_w_start[1:])):
                    plt.plot([st_ts1, st_ts2], [st_V[NWZ_idx,i], st_V[NWZ_idx,i]], alpha=0.7, c='blue', lw=1)
                plt.plot(nst_ts, nst_V[NWZ_idx], alpha=0.7, c='orange', lw=1)
                plt.plot(nst_ts, raw_V[NWZ_idx], alpha=0.3, c='green', lw=0.2)
                # plt.xlim(nst_ts.iloc[0*6000], nst_ts.iloc[10*6000])
                plt.show()
                # plt.figure(dpi=400)
                # plt.title('Confirmation (after merging dataframes)')
                # plt.plot(df_all['ts'], df_all.iloc[:,NWZ_idx+2], alpha=0.7, c='blue', lw=1)
                # plt.plot(df_all['ts'], df_all.iloc[:,NWZ_idx+2+3], alpha=0.7, c='orange', lw=1)
                # plt.plot(nst_ts, raw_V[NWZ_idx], alpha=0.3, c='green', lw=0.2)
                # plt.xlim(nst_ts.iloc[0*6000], nst_ts.iloc[10*6000])
                # plt.show()
            print(f'Storm {storm_id} assessed')




