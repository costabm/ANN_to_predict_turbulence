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
import matplotlib.pyplot as plt

storm_df_all_means, storm_df_all_dirs, storm_df_all_Iu, storm_df_all_Iv, storm_df_all_Iw, storm_df_all_avail = organized_dataframes_of_storms()

ts_storms = find_storm_timestamps()

df_to_analyse = pd.DataFrame()
df_to_analyse['dirs'] = storm_df_all_dirs['synn_A']
df_to_analyse['mean'] = storm_df_all_means['synn_A']
df_to_analyse = df_to_analyse.dropna()
plt.hist2d(df_to_analyse['dirs'], df_to_analyse['mean'], bins=100)
plt.show()


df_to_analyse = pd.DataFrame()
df_to_analyse['dirs'] = storm_df_all_dirs['synn_A']
df_to_analyse['Iu'] = storm_df_all_Iu['synn_A']
df_to_analyse = df_to_analyse.dropna()
plt.hist2d(df_to_analyse['dirs'], df_to_analyse['Iu'], bins=100)
plt.show()







