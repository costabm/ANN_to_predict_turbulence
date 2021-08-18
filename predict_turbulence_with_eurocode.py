import numpy as np
from predict_turbulence_with_ML import plot_topography_per_anem, example_of_elevation_profile_at_given_point_dir_dist, nice_str_dict
from orography import synn_EN_33, svar_EN_33, osp1_EN_33, osp2_EN_33, land_EN_33, neso_EN_33
import matplotlib.pyplot as plt
import matplotlib


anem_EN_33 = {'synn': synn_EN_33, 'svar': svar_EN_33, 'osp1': osp1_EN_33, 'osp2': osp2_EN_33, 'land': land_EN_33, 'neso': neso_EN_33}


# Getting the transition zones as described in NS-EN 1991-1-4:2005/NA:2009, NA.4.3.2(2)
def get_one_roughness_transition_zone(dists, heights):
    """
    When using the Eurocode to predict turbulence, two zones with different terrain roughnessess can be chosen. The transition zone is the upstream distance that divides these two categories.
    My own classification algorithm is smple: Try each distance and choose the one with less misclassifications (ground vs sea) (ascending vs descending roughness with the wind)
    """
    roughs = heights.astype(bool).astype(float)
    assert len(dists) == len(roughs)
    assert np.unique(roughs) in np.array([0,1])  # roughs need to be made of 0's and 1's only
    n_points = len(dists)
    roughs_inv = np.abs(roughs - 1)  # inverted vector: 0's become 1's and 1's become 0's.
    n_wrong = n_points  # start with all wrong
    for i in range(n_points):
        asc_n_wrong_near = np.sum(roughs_inv[:i])
        asc_n_wrong_far  = np.sum(roughs[i:])
        des_n_wrong_near = np.sum(roughs[:i])
        des_n_wrong_far  = np.sum(roughs_inv[i:])
        if (asc_n_wrong_near + asc_n_wrong_far) < n_wrong:  # Descending roughness with the wind (Ascending roughness with upstream dist: 1 to 0)
            n_wrong = asc_n_wrong_near + asc_n_wrong_far
            transition_idx = i
            direction = 'ascending'
        if (des_n_wrong_near + des_n_wrong_far) < n_wrong:  # Ascending roughness with the wind (Descending roughness with upstream dist: 1 to 0)
            n_wrong = des_n_wrong_near + des_n_wrong_far
            transition_idx = i
            direction = 'descending'
    if transition_idx == 0:
        transition_dist = 1  # 1 meter, instead of 0 meters, to prevent errors in dividing by 0
    else:
        transition_dist = (dists[transition_idx-1] + dists[transition_idx]) / 2
    return transition_idx, transition_dist, direction


def get_all_roughness_transition_zones(step_distance=False, total_distance=False, list_of_distances=[i*(5.+5.*i) for i in range(45)]):
    transition_zones = {}
    for anem, anem_coor in anem_EN_33.items():
        transition_zones[anem] = []
        for d in list(range(360)):
            dists, heights = example_of_elevation_profile_at_given_point_dir_dist(point_1=anem_coor, direction_deg=d, step_distance=step_distance, total_distance=total_distance, list_of_distances=list_of_distances, plot=False)
            transition_zones[anem].append(get_one_roughness_transition_zone(dists, heights))
    return transition_zones


def plot_roughness_transitions_per_anem(list_of_degs = list(range(360)), step_distance=False, total_distance=False, list_of_distances=[i*(5.+5.*i) for i in range(45)]):
    transition_zones = get_all_roughness_transition_zones()
    from orography import synn_EN_33, svar_EN_33, osp1_EN_33, osp2_EN_33, land_EN_33, neso_EN_33
    anem_EN_33 = {'synn':synn_EN_33, 'svar':svar_EN_33, 'osp1':osp1_EN_33, 'osp2':osp2_EN_33, 'land':land_EN_33, 'neso':neso_EN_33}
    for anem, anem_coor in anem_EN_33.items():
        dists_all_dirs = []
        roughs_all_dirs = []
        degs = []
        for d in list_of_degs:
            dists, heights = example_of_elevation_profile_at_given_point_dir_dist(point_1=anem_coor, direction_deg=d, step_distance=step_distance, total_distance=total_distance, list_of_distances=list_of_distances, plot=False)
            roughs = heights.astype(bool).astype(float)
            dists_all_dirs.append(dists)
            roughs_all_dirs.append(roughs)
            degs.append(d)
        degs = np.array(degs)
        dists_all_dirs = np.array(dists_all_dirs)
        roughs_all_dirs = np.array(roughs_all_dirs)
        fig, ax = plt.subplots(figsize=(5.5,2.3+0.5), dpi=400)
        ax.pcolormesh(degs, dists_all_dirs[0], roughs_all_dirs.T, cmap=matplotlib.colors.ListedColormap(['skyblue', 'navajowhite']), shading='auto') #, vmin = 0., vmax = 1.)
        xB = np.array([item[1] for item in transition_zones[anem]])
        asc_desc = np.array([item[2] for item in transition_zones[anem]])
        asc_idxs = np.where(asc_desc=='ascending')
        des_idxs = np.where(asc_desc=='descending')
        ax.scatter(degs[asc_idxs], xB[asc_idxs], s=1, alpha=0.4, color='red', label='ascending')
        ax.scatter(degs[des_idxs], xB[des_idxs], s=1, alpha=0.4, color='green', label='descending')
        ax.set_title(nice_str_dict[anem+'_A']+': '+'Upstream topography;')
        ax.patch.set_color('skyblue')
        ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
        ax.set_yticks([0,2000,4000,6000,8000,10000])
        ax.set_yticklabels([0,2,4,6,8,10])
        ax.set_xlabel('Wind from direction [\N{DEGREE SIGN}]')
        ax.set_ylabel('Upstream distance [km]')
        # cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        # cbar.set_label('Height above sea level [m]')
        plt.legend(loc=1)
        plt.tight_layout(pad=0.05)
        # plt.savefig(os.path.join(os.getcwd(), 'plots', f'Rough_transitions_per_anem-{anem}.png'))
        plt.show()
    return None


def get_all_Iu_with_eurocode():
    transition_zones = get_all_roughness_transition_zones()
    z = 48
    kI = 1.0
    c0 = 1.0
    z0_sea = 0.003
    z0_gro = 0.3
    z0_II = 0.05
    kr_sea = 0.19 * (z0_sea/z0_II)**0.07
    kr_gro = 0.19 * (z0_gro/z0_II)**0.07
    cr_sea = kr_sea * np.log(z/z0_sea)
    cr_gro = kr_gro * np.log(z/z0_gro)
    c_season = 1.0
    c_alt = 1.0  # 1.0 because H < H0, where H0 is 900m for Rogaland
    c_prob = 1.0
    Iu = {}
    for anem in anem_EN_33.keys():
        Iu[anem] = []
        xB_all_dirs = [item[1] for item in transition_zones[anem]]
        asc_desc = [item[2] for item in transition_zones[anem]]
        for d in range(360):
            if 360 - 45 / 2 < d or d < 0 + 45 / 2:  # N
                c_dir = 0.9
            elif 45 - 45 / 2 < d < 45 + 45 / 2:  # NØ
                c_dir = 0.6
            elif 90 - 45 / 2 < d < 90 + 45 / 2:  # Ø
                c_dir = 0.8
            elif 135 - 45 / 2 < d < 135 + 45 / 2:  # Ø
                c_dir = 0.9
            else:
                c_dir = 1.0
            if 'osp' in anem:
                vb0 = 28  # Kommune: Austevoll 28m/s (Osp1 and Osp2)
            else:
                vb0 = 26  # Kommune: Tysnes 26m/s (Svar, Land, Neso); Os (Old kommune) 26 m/s (Synn)
            vb = c_dir * c_season * c_alt * c_prob * vb0
            vm_sea = cr_sea * c0 * vb
            vm_gro = cr_gro * c0 * vb
            Iu_sea = kI / (c0 * np.log(z / z0_sea))
            Iu_gro = kI / (c0 * np.log(z / z0_gro))
            xB = xB_all_dirs[d] / 1000  # in kilometers
            if asc_desc[d]=='ascending':
                n = 3
                IuA = Iu_sea
                IuB = Iu_gro
                vmA = vm_sea
                vmB = vm_gro
                cS = 10 ** (-0.04 * n * np.log10(xB/10))
                denominator = min(vmB * cS , vmA)
            elif asc_desc[d]=='descending':
                n = -3
                IuA = Iu_gro
                IuB = Iu_sea
                vmA = vm_gro
                vmB = vm_sea
                cS = 2 - 10 ** (-0.04 * abs(n) * np.log10(xB/10))
                denominator = max(vmB * cS , vmA)
            numerator = IuA * vmA * (1-xB/10) + IuB * vmB * xB/10
            Iu[anem].append(numerator / denominator)
    return Iu


Iu_EN = get_all_Iu_with_eurocode()

for anem in anem_EN_33.keys():
    plt.figure(figsize=(5.5, 2.3), dpi=400)
    plt.title(anem)
    plt.scatter(np.arange(360), Iu_EN[anem], s=3, alpha=0.8, c='dodgerblue', label='$I_u$ (NS-EN 1991-1-4)')
    plt.xlim([0, 360])
    plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax = plt.gca()
    ax.set_xticklabels(['0(N)', '45', '90(E)', '135', '180(S)', '225', '270(W)', '315', '360'])
    plt.ylim([0, 0.7])
    plt.legend(markerscale=2.5, loc=1)
    plt.tight_layout(pad=0.05)
    plt.show()








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
