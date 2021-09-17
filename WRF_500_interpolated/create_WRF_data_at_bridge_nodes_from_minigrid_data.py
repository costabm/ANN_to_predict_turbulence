import os
import numpy as np
from scipy.interpolate import griddata
import netCDF4  # necessary to open the raw data. https://unidata.github.io/netcdf4-python/
from create_minigrid_data_from_raw_WRF_500_data import bridge_WRF_nodes_coor_func

# Getting the already pre-processed data (mini-grid of the relevant WRF 500m datapoints that are near the bridge)
dataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated', 'WRF_500m_minigrid.nc'), 'r')
lats_grid = dataset['latitudes'][:].data
lons_grid = dataset['longitudes'][:].data
ws_grid = dataset['ws'][:].data
wd_grid = dataset['wd'][:].data
time = dataset['time'][:].data
n_time_points = dataset.variables['time'].shape[-1]
dataset.close()

# Getting bridge nodes
lats_bridge, lons_bridge = bridge_WRF_nodes_coor_func().transpose()
n_bridge_nodes = len(lats_bridge)

# Interpolating wind speeds and directions onto the bridge nodes
print('Interpolation might take 5-10 min to run...')
ws_interp = np.array([griddata(points=(lats_grid,lons_grid), values=ws_grid[:,t], xi=(lats_bridge, lons_bridge), method='linear') for t in range(n_time_points)]).transpose()
wd_interp = np.array([griddata(points=(lats_grid,lons_grid), values=wd_grid[:,t], xi=(lats_bridge, lons_bridge), method='linear') for t in range(n_time_points)]).transpose()

# Saving the newly obtained WRF dataset at the bridge nodes
bridgedataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated', r'WRF_at_bridge_nodes.nc'), 'w', format='NETCDF4')
bridgedataset.createDimension('n_nodes', n_bridge_nodes)
bridgedataset.createDimension('n_time_points', n_time_points)
bridgedataset_lats = bridgedataset.createVariable('latitudes', 'f4', ('n_nodes',))  # f4: 32-bit signed floating point
bridgedataset_lons = bridgedataset.createVariable('longitudes', 'f4', ('n_nodes',))  # f4: 32-bit signed floating point
bridgedataset_ws = bridgedataset.createVariable('ws', 'f4', ('n_nodes', 'n_time_points',))  # f4: 32-bit signed floating point
bridgedataset_wd = bridgedataset.createVariable('wd', 'f4', ('n_nodes', 'n_time_points',))  # f4: 32-bit signed floating point
bridgedataset_time = bridgedataset.createVariable('time', 'i4', ('n_time_points',))  # i4: 32-bit signed integer
bridgedataset_lats[:] = lats_bridge
bridgedataset_lons[:] = lons_bridge
bridgedataset_ws[:] = ws_interp
bridgedataset_wd[:] = wd_interp
bridgedataset_time[:] = time
bridgedataset['time'].description = """Number of hours since 01/01/0001 00:00:00 (use datetime.datetime.min + datetime.timedelta(hours=bridgedataset['time'])"""
bridgedataset.close()

