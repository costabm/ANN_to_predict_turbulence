import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator, RectBivariateSpline, griddata
from orography import get_all_geotiffs_merged
import matplotlib.pyplot as plt


lon_mosaic, lat_mosaic, imgs_mosaic = get_all_geotiffs_merged()

point_1 = [-2501, 6750201]  # [longitude, latitude]
point_2 = [ -501, 6750201]  # [longitude, latitude]
step = 30  # meters

def get_arr_of_along_profile_distances(point_1, point_2, step):
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    delta_lon = point_2[0] - point_1[0]
    delta_lat = point_2[1] - point_1[1]
    total_distance = np.sqrt(delta_lon**2 + delta_lat**2)  # SRSS
    n_steps_float = total_distance / step
    new_dists = np.arange(    0, total_distance, delta_lon/n_steps_float)
    new_lons = np.arange(point_1[0], point_2[0], delta_lon/n_steps_float)
    new_lats = np.arange(point_1[1], point_2[1], delta_lat/n_steps_float)
    # Due to the sad fact that RectBivariateSpline only accepts strictly ascending points...:
    lat_mosaic_flipped = np.flip(lat_mosaic, axis=0)
    imgs_mosaic_flipped = np.flip(imgs_mosaic, axis=0)
    coor_fun = RectBivariateSpline(x=lon_mosaic[0,:], y=lat_mosaic_flipped[:,0], z=imgs_mosaic_flipped, kx=1, ky=1)
    coor = coor_fun(new_lons, new_lats, grid=False)


plt.plot(new_dists, coor)
plt.show()