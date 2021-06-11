import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator, RectBivariateSpline, griddata
from orography import get_all_geotiffs_merged
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

lon_mosaic, lat_mosaic, imgs_mosaic = get_all_geotiffs_merged()

print('Building a K-D tree (takes approx. 1 minute)...')
my_tree = cKDTree(np.array([lon_mosaic.ravel(), lat_mosaic.ravel()]).T)


def get_point2_from_point1_dir_and_dist(point_1=[-34625., 6700051.], direction_deg=180, distance=5000):
    lon_1, lat_1 = point_1
    lon_2 = lon_1 + np.sin(np.deg2rad(direction_deg)) * distance
    lat_2 = lat_1 + np.cos(np.deg2rad(direction_deg)) * distance
    return np.array([lon_2, lat_2])


def elevation_profile_generator(point_1, point_2, step_distance=10, list_of_distances=False):
    """
    Args:
        point_1: e.g. [-35260., 6700201.]
        point_2: e.g. [-34501., 6690211.]
        step_distance: Should be larger or equal to the database grid resolution (in this case dtm10 -> 10meters)
        list_of_distances: e.g. False -> uses linear step distance. e.g. [i*(5+5*i) for i in range(45)] -> has non-even steps
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
    if list_of_distances:
        new_dists = np.array(list_of_distances)
    else:
        n_steps = int(np.round(total_distance / step_distance))
        new_dists = np.linspace(0, total_distance, n_steps)
    new_lons = point_1[0] + new_dists / total_distance * delta_lon
    new_lats = point_1[1] + new_dists / total_distance * delta_lat
    # OLD VERSION DOWN
    # new_lons = np.linspace(point_1[0], point_2[0], n_steps)
    # new_lats = np.linspace(point_1[1], point_2[1], n_steps)
    new_lon_lat_idxs = my_tree.query(np.array([new_lons,new_lats]).T)[1]
    heights = imgs_mosaic.ravel()[new_lon_lat_idxs]
    return new_dists, heights


def plot_elevation_profile(point_1, point_2, step_distance, list_of_distances):
    new_dists, heights = elevation_profile_generator(point_1, point_2, step_distance=step_distance, list_of_distances=list_of_distances)
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