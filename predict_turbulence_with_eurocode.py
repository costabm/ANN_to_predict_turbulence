import numpy as np
from predict_turbulence_with_ML import plot_topography_per_anem, example_of_elevation_profile_at_given_point_dir_dist
from orography import synn_EN_33, svar_EN_33, osp1_EN_33, osp2_EN_33, land_EN_33, neso_EN_33
import matplotlib.pyplot as plt
from sklearn import svm


anem_EN_33 = {'synn': synn_EN_33, 'svar': svar_EN_33, 'osp1': osp1_EN_33, 'osp2': osp2_EN_33, 'land': land_EN_33, 'neso': neso_EN_33}
for anem, anem_coor in anem_EN_33.items():
    dists_all_dirs = []
    heights_all_dirs = []
    slopes_all_dirs = []
    degs = []
    for d in list(range(360)):
        dists, heights = example_of_elevation_profile_at_given_point_dir_dist(point_1=anem_coor, direction_deg=d, step_distance=100, total_distance=10000, list_of_distances=False, plot=False)
        roughs = heights.astype(bool).astype(float)


# My own classification algorithm. Simple: Minimize number of misclassifications
def get_transition_zone(dists, roughs):
    assert len(dists) == len(roughs)
    n_points = len(dists)
    roughs_inv = np.abs(roughs - 1)  # inverted vector: 0's become 1's and 1's become 0's.
    # Descending roughness upstream: 1 to 0 (Ascending roughness with the wind)
    n_wrong = n_points  # start with all wrong
    for i in range(n_points):
        des_n_wrong_near = np.sum(roughs_inv[:i])
        des_n_wrong_far  = np.sum(roughs[i:])
        asc_n_wrong_near = np.sum(roughs[:i])
        asc_n_wrong_far  = np.sum(roughs_inv[i:])
        if (des_n_wrong_near + des_n_wrong_far) < n_wrong:
            n_wrong = des_n_wrong_near + des_n_wrong_far
            transition_idx = i
            direction = 'descending'
        if (asc_n_wrong_near + asc_n_wrong_far) < n_wrong:
            n_wrong = asc_n_wrong_near + asc_n_wrong_far
            transition_idx = i
            direction = 'ascending'
    transition_dist = (dists[transition_idx-1] + dists[transition_idx]) / 2
    return transition_idx, transition_dist, direction

get_transition_zone(dists, roughs)




#
#
# X = np.vstack((dists, np.zeros(len(dists)))).T
# Y = roughs
# X_sea = X[np.where(Y==0)]
# X_gro = X[np.where(Y==1)]
#
# #
# # np.random.seed(10)
# # pos = np.hstack((np.random.randn(20, 1) + 3, np.zeros((20, 1))))
# # neg = np.hstack((np.random.randn(20, 1) - 3, np.zeros((20, 1))))
# # X = np.r_[pos, neg]
# # Y = [0] * 20 + [1] * 20
#
# clf = svm.SVC(C= 0.0001, kernel='linear')
# clf.fit(X, Y)
# w = clf.coef_[0]
# x_0 = -clf.intercept_[0]/w[0]
# margin = w[0]
#
# plt.figure()
# x_min, x_max = np.floor(X.min()), np.ceil(X.max())
# y_min, y_max = -3, 3
# yy = np.linspace(y_min, y_max)
# XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
# Z = clf.predict(np.c_[XX.ravel(), np.zeros(XX.size)]).reshape(XX.shape)
# plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
# plt.plot(x_0*np.ones(shape=yy.shape), yy, 'k-')
# plt.plot(x_0*np.ones(shape=yy.shape) - margin, yy, 'k--')
# plt.plot(x_0*np.ones(shape=yy.shape) + margin, yy, 'k--')
# plt.scatter(X_sea, np.zeros(shape=X_sea.shape), s=80, marker='o')
# plt.scatter(X_gro, np.zeros(shape=X_gro.shape), s=80, marker='^')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.show()
#
#
#
#
#
#
#
#
#
#
# plot_topography_per_anem(list_of_degs = list(range(360)), list_of_distances=[i*(5.+5.*i) for i in range(45)], plot_topography=False, plot_slopes=True)
# plot_topography_per_anem(list_of_degs = list(range(360)), list_of_distances=np.arange(0,10000,300).tolist(), plot_topography=False, plot_slopes=True)
#
# B = 0
# L0 = 400
# a = 3
# z = 50
# H = 33.8
# Lh = 300
# delta_Szmax = 2 * H / Lh
# c0 = 1 + delta_Szmax * ((B/L0) / ((B/L0)+0.4)) * (1) * np.exp(-(a*z/Lh))
# print(c0)
#
