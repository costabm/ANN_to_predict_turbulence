# # HOW TO TRANSFORM STANDARD DEVIATIONS FROM ONE SYSTEM TO ANOTHER (COVARIANCE MATRIX REQUIRED)
# from read_0p1sec_data import R_z
# import numpy as np
# V_1 = np.array([[8,12,14,20], [11,9,10,10], [-1, 0, 2,1]])
# T_21 = R_z(np.pi/4).T
# V_2 = T_21 @ V_1
# V_1_stds = np.std(V_1, axis=1)
# V_1_cov = np.cov(V_1, bias=True)
# V_2_stds = np.std(V_2, axis=1)
# V_2_stds_2nd_meth
#
# od = np.sqrt(np.diag(T_21 @ V_1_cov @ T_21.T))