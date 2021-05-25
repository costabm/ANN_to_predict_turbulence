import os
import numpy as np
import netCDF4  # necessary to open the raw data. https://unidata.github.io/netcdf4-python/
import datetime
import matplotlib.pyplot as plt
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

# Getting the already pre-processed and interpolated WRF data at the exact bridge nodes desired
dataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated\WRF_at_bridge_nodes.nc'), 'r')
lats = dataset['latitudes'][:].data
lons = dataset['longitudes'][:].data
ws = dataset['ws'][:].data
wd = dataset['wd'][:].data
time = dataset['time'][:].data
time_in_datetime = [datetime.datetime.min + datetime.timedelta(hours=int(i)) for i in time]
n_time_points = dataset.variables['time'].shape[-1]
n_nodes = len(lats)
dataset.close()

# Storm analysis - Considering only timepoints where ws is larger than threshold in AT LEAST one point
storm_threshold = 15  # m/s
storm_idxs = []
for i in range(n_nodes):
    storm_idxs.extend(np.where(ws[i] > storm_threshold)[0].tolist())
storm_idxs = np.sort(np.unique(storm_idxs))


fig = plt.figure(figsize=(10,30), dpi=300)
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [(0,'#ffffff'), (1e-20,'#440053'), (0.2,'#404388'), (0.4,'#2a788e'),
                                                                    (0.6,'#21a784'), (0.8,'#78d151'), (1,'#fde624'),], N=256)
for i in range(n_nodes):
    ax = fig.add_subplot(11, 1, i+1, projection='scatter_density')
    density = ax.scatter_density(wd[i,storm_idxs], ws[i,storm_idxs], cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
plt.tight_layout()
plt.savefig(r'plots\11_scatter_plots_storm.png')




