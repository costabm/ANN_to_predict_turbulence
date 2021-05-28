"""
Created: May, 2021
Contact: bercos@vegvesen.no
-----------------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import json
from find_storms import organized_dataframes_of_storms, find_storm_timestamps


storm_df_all_means, storm_df_all_dirs, storm_df_all_Iu, storm_df_all_Iv, storm_df_all_Iw, storm_df_all_avail = organized_dataframes_of_storms()

ts_storms = find_storm_timestamps()













