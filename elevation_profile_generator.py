import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator, RectBivariateSpline, griddata
from orography import get_all_geotiffs_merged
import matplotlib.pyplot as plt


lon_mosaic, lat_mosaic, imgs_mosaic = get_all_geotiffs_merged()


def get_arr_of_along_profile_distances(point_1, point_2, step):
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    tolerance_trick = 1  # meter. Force lat or lon to always change at least 1 meter to avoid error in our method in np.arange
    if point_2[0] - point_1[0] == 0:
        point_2 = [point_2[0] + tolerance_trick, point_2[1]]
    if point_2[1] - point_1[1] == 0:
        point_2 = [point_2[0], point_2[1] + tolerance_trick]
    delta_lon = point_2[0] - point_1[0]
    delta_lat = point_2[1] - point_1[1]
    total_distance = np.sqrt(delta_lon**2 + delta_lat**2)  # SRSS
    n_steps_float = total_distance / step
    new_dists = np.arange(    0, total_distance, delta_lon/n_steps_float)
    new_lons = np.arange(point_1[0], point_2[0], delta_lon/n_steps_float)
    new_lats = np.arange(point_1[1], point_2[1], delta_lat/n_steps_float)
    # Flipping arrays, due to the sad fact that RectBivariateSpline only accepts strictly ascending points...
    lat_mosaic_flipped = np.flip(lat_mosaic, axis=0)
    imgs_mosaic_flipped = np.flip(imgs_mosaic, axis=0)
    heights_fun = RectBivariateSpline(x=lon_mosaic[0,:], y=lat_mosaic_flipped[:,0], z=imgs_mosaic_flipped, kx=1, ky=1)
    heights = heights_fun(new_lons, new_lats, grid=False)
    return new_dists, heights

# todo: this is all messed up, results make no sense!
point_1 = [-100260., 6750201.]  # [longitude, latitude]
point_2 = [ -501., 6750211.]  # [longitude, latitude]
step = 100.  # meters

new_dists, heights = get_arr_of_along_profile_distances(point_1, point_2, step)
plt.plot(new_dists, heights)
plt.show()

plt.figure(dpi=800)
plt.imshow(imgs_mosaic)
plt.show()