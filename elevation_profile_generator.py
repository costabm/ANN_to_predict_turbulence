import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator, RectBivariateSpline, griddata
from orography import get_all_geotiffs_merged
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

lon_mosaic, lat_mosaic, imgs_mosaic = get_all_geotiffs_merged()

print('Building a K-D tree (takes approx. 1 minute)...')
my_tree = cKDTree(np.array([lon_mosaic.ravel(), lat_mosaic.ravel()]).T)
print('...done!')


def get_point2_from_point1_dir_and_dist(point_1=[-34625., 6700051.], direction_deg=180, distance=5000):
    lon_1, lat_1 = point_1
    lon_2 = lon_1 + np.sin(np.deg2rad(direction_deg)) * distance
    lat_2 = lat_1 + np.cos(np.deg2rad(direction_deg)) * distance
    return np.array([lon_2, lat_2])


def elevation_profile_generator(point_1, point_2, step_distance=10):
    """
    Args:
        point_1: e.g. [-35260., 6700201.]
        point_2: e.g. [-34501., 6690211.]
        step_distance: Should be larger or equal to the database grid resolution (in this case dtm10 -> 10meters)
    Returns: horizontal distances to point 1 and heights of each interpolated point between point 1 and 2
    """
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    tolerance_trick = 1  # meter. Force lat or lon to always change at least 1 meter to avoid error in our approach with np.arange
    if point_2[0] - point_1[0] == 0:
        point_2 = [point_2[0] + tolerance_trick, point_2[1]]
    if point_2[1] - point_1[1] == 0:
        point_2 = [point_2[0], point_2[1] + tolerance_trick]
    delta_lon = point_2[0] - point_1[0]
    delta_lat = point_2[1] - point_1[1]
    total_distance = np.sqrt(delta_lon**2 + delta_lat**2)  # SRSS
    n_steps_float = total_distance / step_distance
    new_dists = np.arange(    0, total_distance, total_distance/n_steps_float)
    new_lons = np.arange(point_1[0], point_2[0], delta_lon/n_steps_float)
    new_lats = np.arange(point_1[1], point_2[1], delta_lat/n_steps_float)
    # Sometimes (rarely) either new_lons or new_lats manages to squeeze 1 more point in the arange. To avoid this do:
    new_lons = new_lons[:min(len(new_lons), len(new_lats))]
    new_lats = new_lats[:min(len(new_lons), len(new_lats))]
    print('idea: implement non uniform discretization! More resolution nearby, and less resolution far away')
    # # Flipping arrays, due to the sad fact that RectBivariateSpline only accepts strictly ascending points... NOT WORKING WELL
    # lat_mosaic_flipped = np.flip(lat_mosaic, axis=0)
    # imgs_mosaic_flipped = np.flip(imgs_mosaic, axis=0)
    # # Using Splines (only strictly ascending axis accepted):
    # heights_fun = RectBivariateSpline(x=lon_mosaic[0,:], y=lat_mosaic_flipped[:,0], z=imgs_mosaic_flipped, kx=1, ky=1)
    # heights = heights_fun(new_lons, new_lats, grid=False)
    # # Using a Regular Grid Interpolator (only strictly ascending axis accepted)  NOT WORKING WELL:
    # heights_fun = RegularGridInterpolator(points=(lon_mosaic[0, :], lat_mosaic_flipped[:, 0]), values=imgs_mosaic_flipped, method='nearest')
    # heights = heights_fun(np.array([new_lons, new_lats]).T)
    # Using KDTree to find nearest neighbor:
    new_lon_lat_idxs = my_tree.query(np.array([new_lons,new_lats]).T)[1]
    heights = imgs_mosaic.ravel()[new_lon_lat_idxs]
    return new_dists, heights


def plot_elevation_profile(point_1=[-35260., 6700201.], point_2=[-34501., 6690211.], step_distance=10):
    new_dists, heights = elevation_profile_generator(point_1, point_2, step_distance)
    plt.figure()
    plt.title('Terrain Profile')
    plt.plot(new_dists, heights)
    plt.xlabel('Distance [m]')
    plt.ylabel('Height [m]')
    plt.show()

    plt.figure(dpi=600)
    plt.title('Topography and selected points')
    bbox = ((lon_mosaic.min(),   lon_mosaic.max(),
             lat_mosaic.min(),  lat_mosaic.max()))
    plt.xlim(bbox[0], bbox[1])
    plt.ylim(bbox[2], bbox[3])
    plt.imshow(imgs_mosaic, extent=bbox, zorder=0)
    plt.scatter(point_1[0], point_1[1], s=2, c='red')
    plt.scatter(point_2[0], point_2[1], s=2, c='red')
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.show()

# plot_elevation_profile(point_1=[-35260., 6700201.], point_2=[-34501., 6690211.], step_distance=10)