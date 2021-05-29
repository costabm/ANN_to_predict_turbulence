import datetime
from read_0p1sec_data import create_processed_data_files, compile_all_processed_data_into_1_file
from find_storms import create_storm_data_files, compile_storm_data_files

# create_processed_data_files(date_start=datetime.datetime.strptime('2015-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f'), n_months=12*6, window='01:00:00', save_in_folder='processed_data')
# compile_all_processed_data_into_1_file(data_str='01-00-00_stats', save_str='01-00-00_all_stats', save_json=True, foldername='processed_data')

create_storm_data_files(window='00:10:00', input_fname='01-00-00_all_stats')

compile_storm_data_files()


