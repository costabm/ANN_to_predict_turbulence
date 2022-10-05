"""
Converting the K12 concept node coordinates, provided by the AMC group, from Orcaflex coordinates to UTM 32 ones.

According to "Appendix M - SBJ-32-C5-AMC-26-RE-113_1 Mooring system", Chapter 2.2:
All coordinates in the mooring system is given in the same conventions as Orcaflex and the main
analysis model unless other specified. The Orcaflex coordinate are defined as a right-handed
Cartesian coordinate system (UTM zone 32):
    x-axis pointing north with zero in UTM 6,666,000 mN
    y-axis pointing west, with zero in UTM 298,000 mE
    z-axis pointing upwards, with zero in the mean waterline
    directions are defined in propagation direction counter-clockwise from x-axis, e.g. 0 deg means waves from
        south and 90 deg means waves from east.

To confirm and visualize some obtained coordinates use: http://rcn.montana.edu/Resources/Converter.aspx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the following Excel file with the data coordinates:
# O:\Landsdekkende\Ferjefri E39\Fjordkryssinger\12_Bjørnafjorden\Flytebru\Fase5\AMC\Dok\Milepæl 10\Datafiler\
# \Datafiler - Globalanalyser.zip\Globalanalyser\K12.zip\K12\07\K12_07_designers_format.xlsx
df = pd.read_excel('other/K12_node_coords/K12_07_designers_format.xlsx', sheet_name='bridge')  # file with coordinates

# Get K12 --GIRDER-- node coordinates in UTM32 (Easting, Northing, height above mean water level)
K12_Gs = df[['NodeID', 'X', 'Y', 'Z']].drop_duplicates(subset='NodeID')  # drop repeated nodes.
K12_Gs = K12_Gs[['X', 'Y', 'Z']].to_numpy()  # K12 coordinates in the global coordinate system used by AMC
zero_Gs_UTM32 = np.array([298000, 6666000, 0])  # UTM32 (Easting, Northing, height) of the Gs zero ref. (see docstring)
K12_Gs_mod = K12_Gs @ np.diag([1, -1, 1])  # Make the Y-axis (pointing West) negative to conform with "Easting"
K12_Gs_mod = np.array([K12_Gs_mod[:, 1], K12_Gs_mod[:, 0], K12_Gs_mod[:, 2]]).T  # Changed position of X with Y.
K12_UTM32 = K12_Gs_mod + zero_Gs_UTM32

# Get K12 --TOWER-- centerline node coordinates in UTM32 (Easting, Northing, height above mean water level)
K12_Gs_tower_base = df[df['Tag'] == 'A2'].drop_duplicates(subset='NodeID')  # Identified as 'A2' axis in the 'Tag' column.
K12_Gs_tower_base = K12_Gs_tower_base[['X', 'Y', 'Z']]
K12_Gs_tower_base['Z'] = 0  # Z starts at 0 ("Appendix A - SBJ-32-C5-AMC-90-RE-101_1 Drawings binder")
K12_Gs_tower_base = K12_Gs_tower_base.to_numpy()
K12_Gs_tower_base_mod = K12_Gs_tower_base @ np.diag([1, -1, 1])  # Make the Y-axis (pointing West) negative
K12_Gs_tower_base_mod = np.array([K12_Gs_tower_base_mod[:,1], K12_Gs_tower_base_mod[:,0], K12_Gs_tower_base_mod[:,2]]).T
K12_UTM32_tower_base = np.squeeze(K12_Gs_tower_base_mod + zero_Gs_UTM32)
K12_UTM32_tower_top = np.array([K12_UTM32_tower_base[0], K12_UTM32_tower_base[1], 220])
K12_UTM32_tower = np.row_stack([K12_UTM32_tower_base, K12_UTM32_tower_top])


def K12_girder_node_coords(n_girder_nodes=3):
    x = np.linspace(0, len(K12_UTM32)-1, n_girder_nodes)
    return np.array([np.interp(x, np.arange(len(K12_UTM32)), K12_UTM32[:, i]) for i in range(3)]).T


def K12_tower_node_coords(n_tower_nodes=3):
    x = np.linspace(0, len(K12_UTM32_tower)-1, n_tower_nodes)
    return np.array([np.interp(x, np.arange(len(K12_UTM32_tower)), K12_UTM32_tower[:, i]) for i in range(3)]).T


# Plotting bridge girder
n_girder_nodes = 50
n_tower_nodes = 10
K12_UTM32_interp_girder = K12_girder_node_coords(n_girder_nodes=n_girder_nodes)
K12_UTM32_interp_tower = K12_tower_node_coords(n_tower_nodes=n_tower_nodes)

fig = plt.figure(figsize=(20, 20), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.scatter(K12_UTM32[:, 0], K12_UTM32[:, 1], K12_UTM32[:, 2], s=0.1, c='blue')
ax.scatter(K12_UTM32_interp_girder[:, 0], K12_UTM32_interp_girder[:, 1], K12_UTM32_interp_girder[:, 2], s=10, c='orange')
ax.scatter(K12_UTM32_interp_tower[:, 0], K12_UTM32_interp_tower[:, 1], K12_UTM32_interp_tower[:, 2], s=10, c='green')
ax.set_box_aspect([1,1,0.1])
ax.set_xlim3d([2.99174018e+05 - 3000, 2.99174018e+05 + 3000])
ax.set_ylim3d([6.66854498e+06 - 3000, 6.66854498e+06 + 3000])
plt.show()
