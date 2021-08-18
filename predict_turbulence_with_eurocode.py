import numpy as np
from predict_turbulence_with_ML import plot_topography_per_anem, example_of_elevation_profile_at_given_point_dir_dist, nice_str_dict
from orography import synn_EN_33, svar_EN_33, osp1_EN_33, osp2_EN_33, land_EN_33, neso_EN_33
import matplotlib.pyplot as plt
import matplotlib










#
# from sklearn import svm
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
