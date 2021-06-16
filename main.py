import datetime
import os
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from scipy import stats
from scipy.optimize import curve_fit
import bisect
import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, L1Loss, CrossEntropyLoss, SmoothL1Loss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
from read_0p1sec_data import create_processed_data_files, compile_all_processed_data_into_1_file
from find_storms import create_storm_data_files, compile_storm_data_files, find_storm_timestamps, organized_dataframes_of_storms

from elevation_profile_generator import elevation_profile_generator, plot_elevation_profile, get_point2_from_point1_dir_and_dist

def density_scatter(x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots(figsize=(8,6), dpi=400, subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
    data , x_e, y_e = np.histogram2d(np.sin(x)*y, np.cos(x)*y, bins = bins, density = True)
    # z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    x_i = 0.5 * (x_e[1:] + x_e[:-1])
    y_i = 0.5 * (y_e[1:] + y_e[:-1])
    z = interpn((x_i, y_i), data, np.vstack([np.sin(x)*y, np.cos(x)*y]).T, method="splinef2d", bounds_error=False)
    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    ax.scatter( x, y, c=z, s=1, alpha=0.3, **kwargs )
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    return ax, np.min(z), np.max(z)


# create_processed_data_files(date_start=datetime.datetime.strptime('2018-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f'), n_months=12*6, window='00:10:00', save_in_folder='processed_data')
# compile_all_processed_data_into_1_file(data_str='01-00-00_stats', save_str='01-00-00_all_stats', save_json=True, foldername='processed_data')
# create_storm_data_files(window='00:10:00', input_fname='01-00-00_all_stats')
# compile_storm_data_files(save_str='00-10-00_all_storms')

# # Getting post-processed and organized storm data
# storm_df_all_means, storm_df_all_dirs, storm_df_all_Iu, storm_df_all_Iv, storm_df_all_Iw, storm_df_all_avail = organized_dataframes_of_storms(foldername='processed_storm_data', compiled_fname='00-10-00_all_storms')
#
# # Plotting U, Iu and sigma_U along the fjord from measurements nearby (at different heights!):
# delta_dir = 30
# dir_separators = np.arange(0,360.001, delta_dir)
# for dir_min, dir_max in zip(dir_separators[:-1], dir_separators[1:]):
#     print(dir_min, dir_max)
#     idxs_where_storm = (dir_min < storm_df_all_dirs.mean(axis=1)) & (storm_df_all_dirs.mean(axis=1) < dir_max)
#     storm_means = storm_df_all_means[idxs_where_storm]
#     storm_Iu = storm_df_all_Iu[idxs_where_storm]
#     x_pos = [0, 2400, 2600, 5000]  # todo: improve this..
#     x_labels = ['synn_C', 'osp1_C', 'osp2_C', 'svar_C']
#     plt.figure(figsize=(5,10))
#     plt.title(f'Dir between {dir_min} and {dir_max}')
#     plt.errorbar(x_pos, y=storm_Iu.mean()[x_labels], yerr=storm_Iu.std()[x_labels], color='blue', label='Iu')
#     plt.errorbar(x_pos, y=storm_means.mean()[x_labels] / 20, yerr=storm_means.std()[x_labels] / 20, color='orange', label='U / 20')
#     plt.errorbar(x_pos, y=(storm_Iu[x_labels] * storm_means[x_labels]).mean(), yerr=(storm_Iu[x_labels] * storm_means[x_labels]).std()[x_labels] / 20, color='green', label='sigmaU')
#     plt.ylim(bottom=0)
#     plt.legend()
#     plt.show()
#
#
# # Obtaining wind profile at each mast:
#
#
#
#
# # Checking variability in the std_u, for different U and different dir
# U_min = 12
# U_max = 15
# delta_dir = 20
# dir_separators = np.arange(0, 360.001, delta_dir)
# mast_list = ['synn', 'osp1', 'osp2', 'svar']
# anem_list = ['A', 'B', 'C']
# for mast in mast_list:
#     plt.figure(figsize=(3.7, 3.7), dpi=800)
#     ax = plt.subplot(111, projection='polar')
#     ax.set_title(mast)
#     ax.set_theta_zero_location("N")
#     ax.set_theta_direction(-1)
#     for anem in anem_list:
#         mast_anem = mast + '_' + anem
#         dir_centre = []
#         Iu_mean = []
#         Iu_std = []
#         u_std_mean = []
#         u_std_std = []
#         for dir_min, dir_max in zip(dir_separators[:-1], dir_separators[1:]):
#             dir_centre.append((dir_max + dir_min)/2)
#             idxs_where_cond = (dir_min <= storm_df_all_dirs[mast_anem]) & (storm_df_all_dirs[mast_anem] <= dir_max) & (U_min <= storm_df_all_means[mast_anem]) & (storm_df_all_means[mast_anem] <= U_max)
#             Iu_mean.append(storm_df_all_Iu[mast_anem][idxs_where_cond].mean())
#             Iu_std.append(storm_df_all_Iu[mast_anem][idxs_where_cond].std())
#             u_std = (storm_df_all_Iu[mast_anem][idxs_where_cond]).multiply(storm_df_all_means[mast_anem][idxs_where_cond])
#             u_std_mean.append(u_std.mean())
#             u_std_std.append(u_std.std())
#         # ax.errorbar(np.deg2rad(dir_centre), Iu_mean, yerr=Iu_std)
#         ax.errorbar(np.deg2rad(dir_centre), u_std_mean, yerr=u_std_std, label=anem)
#     plt.legend()
#     plt.show()


def test_elevation_profile_at_given_point_dir_dist(point_1=[-34625., 6700051.], direction_deg=160, step_distance=False, total_distance=False,
                                                   list_of_distances=[i*(5.+5.*i) for i in range(45)], plot=True):
    point_2 = get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=direction_deg, distance=list_of_distances[-1] if list_of_distances else total_distance)
    dists, heights = elevation_profile_generator(point_1=point_1, point_2=point_2, step_distance=step_distance, list_of_distances=list_of_distances)
    if plot:
        plot_elevation_profile(point_1=point_1, point_2=point_2, step_distance=step_distance, list_of_distances=list_of_distances)
    return dists, heights
test_elevation_profile_at_given_point_dir_dist(point_1=[-34625., 6700051.], direction_deg=180,step_distance=False,total_distance=False, list_of_distances=[i*(5.+5.*i) for i in range(45)],plot=True)
# test_elevation_profile_at_given_point_dir_dist(point_1=[-34625., 6700051.], direction_deg=180, step_distance=10., total_distance=9900., list_of_distances=False, plot=True)


def convert_angle_to_0_2pi_interval(angle, input_and_output_in_degrees=True):
    if input_and_output_in_degrees:
        return angle % 360
    else:
        return angle % (2*np.pi)
    # if input_and_output_in_degrees:
    #     angle = np.deg2rad(angle)
    # new_angle = np.arctan2(np.sin(angle), np.cos(angle))
    # if new_angle < 0:
    #     new_angle = abs(new_angle) + 2 * (np.pi - abs(new_angle))
    # assert 0 <= new_angle <= 2*np.pi
    # if input_and_output_in_degrees:
    #     new_angle = np.rad2deg(new_angle)
    # return new_angle


def get_all_10_min_data_at_z_48m(U_min = 5, terrain_profile_dists=[i*(5.+5.*i) for i in range(45)]):
    print('Collecting all 10-min wind data... (takes 10-15 minutes)')
    min10_df_all_means, min10_df_all_dirs, min10_df_all_Iu, min10_df_all_Iv, min10_df_all_Iw, min10_df_all_avail = organized_dataframes_of_storms(foldername='processed_data', compiled_fname='00-10-00_all_stats')
    columns_to_drop = ['ts', 'osp1_C', 'osp2_C', 'svar_B','svar_C','synn_B','synn_C']  # Discarding all data that are not at Z=48m
    X_means_df = min10_df_all_means.drop(columns=columns_to_drop)
    X_dirs_df =  min10_df_all_dirs.drop(columns=columns_to_drop)
    X_Iu_df = min10_df_all_Iu.drop(columns=columns_to_drop)
    idxs_where_cond = (U_min <= X_means_df)  # Discarding all data with U below U_min
    X_means_df = X_means_df[idxs_where_cond].dropna(axis=0, how='all')
    X_dirs_df  = X_dirs_df[idxs_where_cond].dropna(axis=0, how='all')
    X_Iu_df    = X_Iu_df[idxs_where_cond].dropna(axis=0, how='all')
    X_std_u_df = X_Iu_df.multiply(X_means_df)
    mast_anem_list = ['osp1_A', 'osp1_B', 'osp2_A', 'osp2_B', 'svar_A', 'synn_A']
    # Organizing the data into an input matrix (with shape shape (n_samples, n_features)):
    X_data = []
    y_data = []
    data_len_of_each_anem = []
    mast_UTM_33 = {'synn':[-34515., 6705758.], 'osp1':[-39375., 6703464.], 'osp2':[-39350., 6703204.], 'svar':[-34625., 6700051.]}
    for mast_anem in mast_anem_list:
        # We allow different data lengths for each anemometer, but a row with at least one nan is removed from a given anemometer df of 'means' 'dirs' and 'stds'
        X_mean_dir_std_anem = pd.DataFrame({'means': X_means_df[mast_anem], 'dirs': X_dirs_df[mast_anem], 'stds': X_std_u_df[mast_anem]}).dropna(axis=0, how='any')
        X_mean_anem =  np.array(X_mean_dir_std_anem['means'])
        X_dir_anem =   np.array(X_mean_dir_std_anem['dirs'])
        X_std_u_anem = np.array(X_mean_dir_std_anem['stds'])
        point_1 = mast_UTM_33[mast_anem[:4]]
        windward_cone_angles = [-14, -9, -5, -2, 0, 2, 5, 9, 14]  # deg. the a angles within the "+-15 deg cone of influence" of the windward terrain on the wind properties
        heights = []
        for a in windward_cone_angles:
            X_dir_anem_yawed = convert_angle_to_0_2pi_interval(X_dir_anem + a, input_and_output_in_degrees=True)
            points_2 = np.array([get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=d, distance=terrain_profile_dists[-1]) for d in X_dir_anem_yawed])
            heights.append(np.array([elevation_profile_generator(point_1=point_1, point_2=p2, step_distance=False, list_of_distances=terrain_profile_dists)[1] for p2 in points_2]))
        heights = np.array(heights)
        X_data_1_anem = np.concatenate((X_mean_anem[:,None], X_dir_anem[:,None], np.mean(heights,axis=0), np.std(heights,axis=0)[:,1:]), axis=1)  # [1:,:] because point_1 has always std = 0 (same point 1 for all windward_cone_angles)
        X_data.append(X_data_1_anem)
        y_data.append(X_std_u_anem)
        data_len_of_each_anem.append(len(X_mean_anem))
        print(f'{mast_anem}: All ML-input-data is collected')
    X_data = np.concatenate(tuple(i for i in X_data), axis=0)  # n_steps + 2 <=> number of terrain heights (e.g. 500) + wind speed (1) + wind dir (1)
    y_data = np.concatenate(y_data)
    start_idxs_of_each_anem = [0] + np.cumsum(data_len_of_each_anem)[:-1].tolist()
    return  X_data, y_data, mast_anem_list, start_idxs_of_each_anem

########################################
# MACHINE LEARNING
########################################

# Neural network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'cuda' or 'cpu'. 'cuda' doesn't seem to be working...
def train_and_test_NN(X_train, y_train, X_test, y_test, hp, print_loss_per_epoch=True):
    """
    Args:
        X_train:
        y_train:
        X_test:
        y_test:
        hp:  hyperparameters. e.g. {'lr':1E-1, 'batch_size':92, 'weight_decay':1E-4, 'momentum':0.9, 'n_epochs':10, 'n_hid_layers':2}
        print_loss_per_epoch:
    Returns:
    """
    learn_rate = hp['lr']
    batch_size = hp['batch_size']
    weight_decay = hp['weight_decay']
    momentum = hp['momentum']
    n_epochs = hp['n_epochs']
    n_hid_layers = hp['n_hid_layers']
    my_activation_func = hp['activation']  # ReLU, ELU, LeakyReLU, etc.
    criterion = hp['loss']  # define the loss function
    n_samples_train = X_train.shape[0]
    # Building a neural network dynamically
    torch.manual_seed(0)  # make the following random numbers reproducible
    n_features = X_train.shape[1]  # number of independent variables in the polynomial
    n_outputs = y_train.shape[1]  # number of independent variables in the polynomial
    n_hid_layer_neurons = n_features  # Fancy for: More monomials, more neurons...
    my_nn = torch.nn.Sequential()
    my_nn.add_module(name='0', module=torch.nn.Linear(n_features, n_hid_layer_neurons))  # Second layer
    my_nn.add_module(name='0A', module=my_activation_func())  # Activation function
    for i in range(n_hid_layers):  # Hidden layers
        my_nn.add_module(name=str(i + 1), module=torch.nn.Linear(n_hid_layer_neurons, n_hid_layer_neurons))
        my_nn.add_module(name=f'{i + 1}A', module=my_activation_func())
    my_nn.add_module(name=str(n_hid_layers + 1), module=torch.nn.Linear(n_hid_layer_neurons, n_outputs))  # Output layer

    optimizer = SGD(my_nn.parameters(), lr=learn_rate, weight_decay=weight_decay, momentum=momentum)  # define the optimizer
    torch.seed()  # make random numbers again random
    my_nn.to(device)  # To GPU if available
    # Training
    # writer = SummaryWriter(f'runs/my_math_learning_tensorboard')  # For later using TensorBoard, for visualization
    assert (n_samples_train/batch_size).is_integer(), "Change batch size so that n_iterations is integer"
    n_iterations = int(n_samples_train/batch_size)
    for epoch in range(n_epochs):
        epoch_loss = 0
        idxs_shuffled = torch.randperm(n_samples_train)
        for b in range(n_iterations):
            batch_idxs = idxs_shuffled[b*batch_size:b*batch_size+batch_size]
            y_pred = my_nn(Variable(X_train[batch_idxs]))
            loss = criterion(y_pred, Variable(y_train[batch_idxs].view(batch_size,n_outputs), requires_grad=False))
            epoch_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if print_loss_per_epoch:
            print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
        # writer.add_scalar('Training Loss', epoch_loss, global_step=epoch)  # writing to TensorBoard
    # Testing
    n_samples_test = X_test.shape[0]
    with torch.no_grad():
        y_pred = my_nn(Variable(X_test))
        SS_res_test = torch.sum((y_test.view(n_samples_test,n_outputs) - y_pred.view(n_samples_test,n_outputs))**2)
        y_pred_mean = torch.mean(y_pred, axis=0)
        SS_tot_test = torch.sum((y_test.view(n_samples_test,n_outputs) - y_pred_mean)**2)
        R2_test = 1 - SS_res_test / SS_tot_test
    idxs_to_print = np.random.randint(0, len(y_pred), 10)  # a few random values to be printed
    print(f'R2 on test dataset ----> {R2_test} <---- . Learning rate: {learn_rate}')
    print(f"Prediction: {y_pred[idxs_to_print]}")
    print(f"Reference:   {y_test[idxs_to_print]}")
    print(f'Batch size: {batch_size}')
    return y_pred

# ##################################################################
# # Getting the data for first time
# X_data_nonnorm, y_data_nonnorm, mast_anem_list, start_idxs_of_each_anem = get_all_10_min_data_at_z_48m(U_min=0) ##  takes a few minutes...
# ##################################################################
# # Saving data
# data_path = os.path.join(os.getcwd(), 'processed_data', 'X_y_ML_ready_data_Umin_0_9elev')
# np.savez_compressed(data_path, X=X_data_nonnorm, y=y_data_nonnorm, m=mast_anem_list, i=start_idxs_of_each_anem)
##################################################################
# Loading data already saved
data_path = os.path.join(os.getcwd(), 'processed_data', 'X_y_ML_ready_data_Umin_0_9elev')
loaded_data = np.load(data_path + '.npz')
X_data_nonnorm = loaded_data['X']
y_data_nonnorm = loaded_data['y']
mast_anem_list = loaded_data['m']
start_idxs_of_each_anem = loaded_data['i']
X_360dirs = np.searchsorted(np.arange(0,360,1), X_data_nonnorm[:,1], side='right') - 1
n_samples = X_data_nonnorm.shape[0]
start_idxs_of_each_anem_2 = np.array(start_idxs_of_each_anem.tolist() + [n_samples])  # this one includes the final index as well
#################################################################
# Normalizing data
X_maxs = np.max(X_data_nonnorm, axis=0)
y_max = np.max(y_data_nonnorm)
X_data = X_data_nonnorm/X_maxs
y_data = y_data_nonnorm/y_max
##################################################################

##################################################################################################################
# Statistical analyses of the distribution of turbulence. Converting y_data (1 dimension) into y2_data (2 dimensions -> 2 Weibull params)
##################################################################################################################
# # Plotting data by anemometer
# for anem_idx in range(6):
#     anem_slice = slice(start_idxs_of_each_anem_2[anem_idx],start_idxs_of_each_anem_2[anem_idx+1])
#     density_scatter(np.deg2rad(X_data_nonnorm[anem_slice,1]), y_data_nonnorm[anem_slice])
#     plt.title(f'{mast_anem_list[anem_idx]}')
#     plt.ylim([0, 5])
#     plt.savefig(os.path.join(os.getcwd(), 'plots', f'measured_std_u_{mast_anem_list[anem_idx]}.png'))
#     plt.show()

# Obtaining the 4 Exponentiated-Weibull parameters, valid for a 1 deg sector of 1 anemometer, and copying these values for all data that falls in the same 1 deg sector and anem
PDF_used = 'weibull'  # 'weibull' or 'expweibull'
n_outputs = 2 if PDF_used == 'weibull' else 3
y_PDF_data_nonnorm = np.empty((len(y_data_nonnorm), n_outputs))  # 3 parameters in the exponentiated weibull distribution
y_PDF_data_nonnorm[:] = np.nan
y_PDF_data_nonnorm_360dirs = []
for anem_idx in range(6):
    y_PDF_data_nonnorm_per_anem = []
    anem_slice = slice(start_idxs_of_each_anem_2[anem_idx], start_idxs_of_each_anem_2[anem_idx + 1])
    for d in range(360):
        idxs_360dir = np.where(X_360dirs[anem_slice] == d)[0]
        if len(idxs_360dir) > 5 :  # minimum number of datapoints, otherwise -> np.nan
            if PDF_used == 'weibull':
                _      , param_c, _, param_scale = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir], floc = 0, f0 = 1)  # params = a(alpha), c(shape==k), loc, scale(lambda)
                y_PDF_data_nonnorm[anem_slice][idxs_360dir] = np.array([param_c, param_scale])  # parameter '2' not included (floc, which is set to 0)
                y_PDF_data_nonnorm_per_anem.append(np.array([param_c, param_scale]))
            elif PDF_used == 'expweibull':
                param_a, param_c, _, param_scale = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir], floc = 0)  # params = a(alpha), c(shape==k), loc, scale(lambda)
                y_PDF_data_nonnorm[anem_slice][idxs_360dir] = np.array([param_a, param_c,param_scale])  # parameter '2' not included (floc, which is set to 0)
                y_PDF_data_nonnorm_per_anem.append(np.array([param_a, param_c,param_scale]))
    y_PDF_data_nonnorm_360dirs.append(y_PDF_data_nonnorm_per_anem)
    print(f'{mast_anem_list[anem_idx]}: Weibull parameters obtained')
y_PDF_data_nonnorm_360dirs = np.array(y_PDF_data_nonnorm_360dirs)
y_PDF_maxs = np.max(y_PDF_data_nonnorm, axis=0)
y_PDF_data = y_PDF_data_nonnorm / y_PDF_maxs

# Generating synthetic data, using only the weibull parameters for each 1-deg sector and anemometer
y_data_synth =  np.empty((len(y_data_nonnorm)))  # 4 parameters in the exponentiated weibull distribution
y_data_synth[:] = np.nan
for anem_idx in range(6):
    anem_slice = slice(start_idxs_of_each_anem_2[anem_idx], start_idxs_of_each_anem_2[anem_idx + 1])
    for d in range(360):
        idxs_360dir = np.where(X_360dirs[anem_slice] == d)[0]
        if PDF_used == 'weibull':
            param_c, param_scale = y_PDF_data_nonnorm_360dirs[anem_idx][d]
            y_data_synth[anem_slice][idxs_360dir] = stats.exponweib.rvs(      1, param_c, 0, param_scale, size=len(idxs_360dir))
        elif PDF_used == 'expweibull':
            param_a, param_c, param_scale = y_PDF_data_nonnorm_360dirs[anem_idx][d]
            y_data_synth[anem_slice][idxs_360dir] = stats.exponweib.rvs(param_a, param_c, 0, param_scale, size=len(idxs_360dir))

# Plotting synthetic data by anemometer
for anem_idx in range(6):
    anem_slice = slice(start_idxs_of_each_anem_2[anem_idx],start_idxs_of_each_anem_2[anem_idx+1])
    x = np.deg2rad(X_data_nonnorm[anem_slice, 1])
    y1 = y_data_nonnorm[anem_slice]
    y2 = y_data_synth[anem_slice]

    data1, x1_e, y1_e = np.histogram2d(np.sin(x)*y1, np.cos(x)*y1, bins=30, density=True)
    data2, x2_e, y2_e = np.histogram2d(np.sin(x)*y2, np.cos(x)*y2, bins=[x1_e, y1_e], density=True)
    x1_i = 0.5 * (x1_e[1:] + x1_e[:-1])
    y1_i = 0.5 * (y1_e[1:] + y1_e[:-1])
    x2_i = 0.5 * (x2_e[1:] + x2_e[:-1])
    y2_i = 0.5 * (y2_e[1:] + y2_e[:-1])
    z1 = interpn((x1_i, y1_i), data1, np.vstack([np.sin(x)*y1, np.cos(x)*y1]).T, method="splinef2d", bounds_error=False)
    z2 = interpn((x2_i, y2_i), data2, np.vstack([np.sin(x)*y2, np.cos(x)*y2]).T, method="splinef2d", bounds_error=False)
    z1[np.where(np.isnan(z1))] = 0.0  # To be sure to plot all data
    z2[np.where(np.isnan(z2))] = 0.0  # To be sure to plot all data
    idx1 = z1.argsort()   # Sort the points by density, so that the densest points are plotted last
    idx2 = z2.argsort()   # Sort the points by density, so that the densest points are plotted last
    x1, y1, z1 = x[idx1], y1[idx1], z1[idx1]
    x2, y2, z2 = x[idx2], y2[idx2], z2[idx2]
    # 1st plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300, subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.scatter(x1, y1, c=z1, s=1, alpha=0.3)
    norm = Normalize(vmin = np.min([z1,z2]), vmax = np.max([z1,z2]))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    plt.title(f'Measured. {mast_anem_list[anem_idx]}')
    plt.ylim([0, 4])
    ax.text(np.deg2rad(20), 4.4, '$\sigma(u)\/[m/s]$')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plots', f'measured_std_u_{mast_anem_list[anem_idx]}.png'))
    plt.show()
    # 2nd plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300, subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.scatter(x2, y2, c=z2, s=1, alpha=0.3)
    norm = Normalize(vmin = np.min([z1,z2]), vmax = np.max([z1,z2]))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    plt.title(f'Synthetic (Weibull). {mast_anem_list[anem_idx]}')
    plt.ylim([0, 4])
    ax.text(np.deg2rad(20), 4.4, '$\sigma(u)\/[m/s]$')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plots', f'synth_std_u_{mast_anem_list[anem_idx]}.png'))
    plt.show()


##################################################################################################################
# STD_u - TRAINING FROM 18 ALTERNATE-10-DEG-WIDE-WIND-SECTORS AND TESTING THE REMAINING 18 SECTORS, AT EACH ANEMOMETER
##################################################################################################################

# Remove the direction, to be extra certain that the NN doesn't "cheat"
# X_data = np.delete(X_data, 1, axis=1) # NOT WORKING FOR THE BEAUTIFUL PLOTS THAT WILL REQUIRE THESE VALUES

# Separating training and testing data
train_angle_domain = [[x, x+22.5] for x in np.arange(0, 360, 45)]  # in degrees
test_angle_domain  = [[x+22.5, x+45] for x in np.arange(0, 360, 45)]  # in degrees
train_bools = np.logical_or.reduce([(a[0]<X_data_nonnorm[:,1]) & (X_data_nonnorm[:,1]<a[1]) for a in train_angle_domain])  # https://stackoverflow.com/questions/20528328/numpy-logical-or-for-more-than-two-arguments
test_bools =  np.logical_or.reduce([(a[0]<X_data_nonnorm[:,1]) & (X_data_nonnorm[:,1]<a[1]) for a in test_angle_domain])
X_train = Tensor(X_data[train_bools]).to(device)
y_train = Tensor(y_data[train_bools]).to(device)
X_test =  Tensor(X_data[test_bools]).to(device)
y_test =  Tensor(y_data[test_bools]).to(device)

n_samples_train = X_train.shape[0]
batch_size_possibilities = sympy.divisors(n_samples_train)  # [1, 2, 4, 23, 46, 92, 4051, 8102, 16204, 93173, 186346, 372692]


# Getting values to predict and predicted values
hp = {'lr':1E-1, 'batch_size':6329, 'weight_decay':1E-4, 'momentum':0.9, 'n_epochs':25, 'n_hid_layers':1, 'activation':torch.nn.ReLU, 'loss':MSELoss()}
y_pred = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=True)

# Choosing only the results of a given anemometer (e.g. svar -> Svarvahelleholmen)
anem_train = np.searchsorted(start_idxs_of_each_anem, np.where(train_bools)[0], side='right') - 1  # to which anemometer, (indexed from mast_anem_list), does each test sample belong to
anem_test =  np.searchsorted(start_idxs_of_each_anem, np.where(test_bools )[0], side='right') - 1
train_idxs_svar = np.where(anem_train == np.where(mast_anem_list == 'svar_A')[0])[0]
test_idxs_svar  = np.where(anem_test  == np.where(mast_anem_list == 'svar_A')[0])[0]
X_train_svar = X_train[train_idxs_svar].cpu().numpy()
y_train_svar = np.squeeze(y_train[train_idxs_svar].cpu().numpy())  # simply converting to numpy and removing the empty dimension of the shape (n_train_samples,1)
X_test_svar = X_test[test_idxs_svar].cpu().numpy()
y_test_svar = np.squeeze(y_test[test_idxs_svar].cpu().numpy())  # simply converting to numpy and removing the empty dimension of the shape (n_test_samples,1)
y_pred_svar = np.squeeze(y_pred[test_idxs_svar].cpu().numpy())  # simply converting to numpy and removing the empty dimension of the shape (n_test_samples,1)

# De-normalizing
dir_train_svar = X_train_svar[:,1] * X_maxs[1]
dir_test_svar = X_test_svar[:,1] * X_maxs[1]
std_u_train_svar = y_train_svar * y_max
std_u_test_svar  = y_test_svar * y_max
std_u_pred_svar = y_pred_svar * y_max

# Organizing the results into sectors
train_sector_bools = [(a[0]<dir_train_svar) & (dir_train_svar<a[1]) for a in train_angle_domain]
test_sector_bools  = [(a[0]<dir_test_svar)  & (dir_test_svar<a[1])  for a in test_angle_domain]
train_sector_idxs = [np.where(train_sector_bools[i])[0] for i in range(len(train_sector_bools))]
test_sector_idxs  = [np.where(test_sector_bools[i] )[0] for i in range(len(test_sector_bools ))]
dir_means_train_per_sector_svar =   np.array([np.mean(dir_train_svar[l]) for l in train_sector_idxs])
dir_means_test_per_sector_svar  =   np.array([np.mean(dir_test_svar[l] ) for l in test_sector_idxs ])
std_u_means_train_per_sector_svar = np.array([np.mean(std_u_train_svar[l]) for l in train_sector_idxs])
std_u_means_test_per_sector_svar  = np.array([np.mean(std_u_test_svar[l]) for l in test_sector_idxs])
std_u_means_pred_per_sector_svar = np.array([np.mean(std_u_pred_svar[l]) for l in test_sector_idxs])
std_u_std_train_per_sector_svar = np.array([np.std(std_u_train_svar[l]) for l in train_sector_idxs])
std_u_std_test_per_sector_svar  = np.array([np.std(std_u_test_svar[l] ) for l in test_sector_idxs ])
std_u_std_pred_per_sector_svar =   np.array([np.std( std_u_pred_svar[l]) for l in test_sector_idxs])

# Plotting beautiful plots
fig = plt.figure(figsize=(8,6), dpi=400)
ax = fig.add_subplot(projection='polar')
plt.title('Anemometer at Svarvahelleholmen (Z = 48 m)\n')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.scatter( np.deg2rad(dir_train_svar), std_u_train_svar, s=1, alpha=0.6, c='lightgreen', label='Training data')
ax.scatter( np.deg2rad(dir_test_svar) , std_u_test_svar , s=1, alpha=0.6, c='skyblue', label='Testing data')
ax.errorbar(np.deg2rad(dir_means_train_per_sector_svar), std_u_means_train_per_sector_svar, std_u_std_train_per_sector_svar, c='forestgreen', elinewidth=3, alpha=0.9, fmt='.', label='$\sigma(train)$')
ax.errorbar(np.deg2rad(dir_means_test_per_sector_svar) , std_u_means_test_per_sector_svar , std_u_std_test_per_sector_svar , c='dodgerblue', elinewidth=4, alpha=0.8, fmt='o', label='$\sigma(test)$')
ax.errorbar(np.deg2rad(dir_means_test_per_sector_svar) , std_u_means_pred_per_sector_svar , std_u_std_pred_per_sector_svar , c='orange', elinewidth=2, alpha=0.9, fmt='.', label='Prediction', zorder=5)
handles, labels = ax.get_legend_handles_labels()
plt.ylim((None,4))
ax.text(np.deg2rad(18), 4.4, '$\sigma(u)\/[m/s]$')
plt.savefig(os.path.join(os.getcwd(), 'plots', 'std_u_Svar.png'))
plt.show()
fig = plt.figure(figsize=(2, 1.6), dpi=400)
plt.axis('off')
plt.legend(handles, labels)
plt.savefig(os.path.join(os.getcwd(), 'plots', 'std_u_Svar_legend.png'))
plt.show()



##################################################################################################################
# STD_u - TRAINING FROM 5 ANEMOMETERS AND TESTING REMAINING 1 ANEMOMETER AT SYNNOYTANGEN
##################################################################################################################

# Remove the direction, to be extra certain that the NN doesn't "cheat"
# X_data = np.delete(X_data, 1, axis=1) # NOT WORKING FOR THE BEAUTIFUL PLOTS THAT WILL REQUIRE THESE VALUES

anem_to_test = 'svar_A'
anem_start_idx = start_idxs_of_each_anem_2[np.where(mast_anem_list == anem_to_test)[0]][0]
anem_end_idx = start_idxs_of_each_anem_2[np.where(mast_anem_list == anem_to_test)[0]+1][0]
test_idxs = np.where((anem_start_idx <= np.arange(n_samples)) & (np.arange(n_samples) < anem_end_idx))[0]
train_idxs = np.array(list(set(np.arange(n_samples)) - set(test_idxs)))
X_train = Tensor(X_data[train_idxs]).to(device)
y_train = Tensor(y_data[train_idxs]).to(device)
X_test =  Tensor(X_data[test_idxs]).to(device)
y_test =  Tensor(y_data[test_idxs]).to(device)


n_samples_train = X_train.shape[0]
batch_size_possibilities = np.array(sympy.divisors(n_samples_train))  # [1, 2, 4, 23, 46, 92, 4051, 8102, 16204, 93173, 186346, 372692]
batch_size_desired = 4000
batch_size = min(batch_size_possibilities, key=lambda x:abs(x-batch_size_desired))
assert batch_size > 1000, "Warning ! Batch size too small can be extremely slow"

# Getting values to predict and predicted values
hp = {'lr':1E-1, 'batch_size':batch_size, 'weight_decay':0, 'momentum':0, 'n_epochs':35,
      'n_hid_layers':1, 'activation':torch.nn.LeakyReLU, 'loss':MSELoss()}
y_pred = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=True)

# De-normalizing
dir_test = X_test[:,1].cpu().numpy() * X_maxs[1]
std_u_test  = np.squeeze(y_test.cpu().numpy()) * y_max
std_u_pred = np.squeeze(y_pred.cpu().numpy()) * y_max

# Plotting beautiful plots
# fig = plt.figure(figsize=(8,6), dpi=400)
# ax = fig.add_subplot(projection='polar')
# plt.title(f'Anemometer "{anem_to_test}" (Z = 48 m)\n')
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.scatter( np.deg2rad(dir_test) , std_u_test , s=0.1, alpha=0.3, c='skyblue', marker='.', label='Testing data')
# ax.scatter( np.deg2rad(dir_test) , std_u_pred , s=0.1, alpha=0.3, c='yellow', marker='x', label='Predictions')
# # ax.errorbar(np.deg2rad(dir_test), std_u_means_train_per_sector_svar, std_u_std_train_per_sector_svar, c='forestgreen', elinewidth=3, alpha=0.9, fmt='.', label='$\sigma(train)$')
# # ax.errorbar(np.deg2rad(dir_means_test_per_sector_svar) , std_u_means_test_per_sector_svar , std_u_std_test_per_sector_svar , c='dodgerblue', elinewidth=4, alpha=0.8, fmt='o', label='$\sigma(test)$')
# # ax.errorbar(np.deg2rad(dir_means_test_per_sector_svar) , std_u_means_pred_per_sector_svar , std_u_std_pred_per_sector_svar , c='orange', elinewidth=2, alpha=0.9, fmt='.', label='Prediction', zorder=5)
# handles, labels = ax.get_legend_handles_labels()
# plt.ylim((None,4))
# ax.text(np.deg2rad(18), 4.4, '$\sigma(u)\/[m/s]$')
# plt.savefig(os.path.join(os.getcwd(), 'plots', 'std_u_Svar.png'))
# plt.show()
# fig = plt.figure(figsize=(2, 1.6), dpi=400)
# plt.axis('off')
# plt.legend(handles, labels)
# plt.savefig(os.path.join(os.getcwd(), 'plots', 'std_u_Svar_legend.png'))
# plt.show()



density_scatter( x=np.deg2rad(dir_test) , y=std_u_test, ax = None, sort = True, bins = 50)
plt.ylim([None, 4])
plt.show()

density_scatter( x=np.deg2rad(dir_test) , y=std_u_pred, ax = None, sort = True, bins = 50)
plt.ylim([None, 4])
plt.show()



##################################################################################################################
# WEIBULL PARAMS - TRAINING FROM CERTAIN ANEMOMETERS AND TESTING AT 1 GIVEN ANEMOMETER
##################################################################################################################

# Remove the direction, to be extra certain that the NN doesn't "cheat"
# X_data = np.delete(X_data, 1, axis=1) # NOT WORKING FOR THE BEAUTIFUL PLOTS THAT WILL REQUIRE THESE VALUES


anem_to_test = 'osp1_A'
anem_no_train = ['osp1_A', 'osp1_B', 'osp2_A', 'osp2_B']  # anemometer to exclude from the training (besides the anem_to_test!) Because e.g. osp1_A and osp1_B have exact same terrain profiles
anem_start_idx = start_idxs_of_each_anem_2[np.where(mast_anem_list == anem_to_test)[0]][0]
anem_end_idx = start_idxs_of_each_anem_2[np.where(mast_anem_list == anem_to_test)[0]+1][0]
no_train_idxs = []
for a in anem_no_train:
    anem_no_train_start_idx = start_idxs_of_each_anem_2[np.where(mast_anem_list == a)[0]][0]
    anem_no_train_end_idx =   start_idxs_of_each_anem_2[np.where(mast_anem_list == a)[0]+1][0]
    no_train_idxs.extend(np.where((anem_no_train_start_idx <= np.arange(n_samples)) & (np.arange(n_samples) < anem_no_train_end_idx))[0])
no_train_idxs = np.array(no_train_idxs)
test_idxs = np.where((anem_start_idx <= np.arange(n_samples)) & (np.arange(n_samples) < anem_end_idx))[0]
train_idxs = np.array(list(set(np.arange(n_samples)) - set(test_idxs) - set(no_train_idxs)))
X_train = Tensor(X_data[train_idxs]).to(device)
y_train = Tensor(y_PDF_data[train_idxs]).to(device)
X_test =  Tensor(X_data[test_idxs]).to(device)
y_test =  Tensor(y_PDF_data[test_idxs]).to(device)

n_samples_train = X_train.shape[0]
batch_size_possibilities = np.array(sympy.divisors(n_samples_train))  # [1, 2, 4, 23, 46, 92, 4051, 8102, 16204, 93173, 186346, 372692]
batch_size_desired = 5000
batch_size = min(batch_size_possibilities, key=lambda x:abs(x-batch_size_desired))
batch_cond = 1000 < batch_size < 10000
if not batch_cond:
    while not batch_cond:
        X_train = X_train [:-1,:]
        y_train = y_train [:-1,:]
        X_test  = X_test  [:-1,:]
        y_test  = y_test  [:-1,:]
        n_samples_train = X_train.shape[0]
        batch_size_possibilities = np.array(sympy.divisors(n_samples_train))  # [1, 2, 4, 23, 46, 92, 4051, 8102, 16204, 93173, 186346, 372692]
        batch_size_desired = 5000
        batch_size = min(batch_size_possibilities, key=lambda x: abs(x - batch_size_desired))
        batch_cond = 1000 < batch_size < 30000
assert batch_size > 1000, "Warning ! Batch size too small can be extremely slow"
# Getting values to predict and predicted values
hp = {'lr':1E-1, 'batch_size':batch_size, 'weight_decay':2E-3, 'momentum':0.9, 'n_epochs':50,
      'n_hid_layers':3, 'activation':torch.nn.ELU, 'loss':MSELoss()}
y_pred = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=True)

# De-normalizing
dir_test = X_test[:,1].cpu().numpy() * X_maxs[1]
y_test_nonnorm  = np.squeeze(y_test.cpu().numpy()) * y_PDF_maxs
y_pred_nonnorm = np.squeeze(y_pred.cpu().numpy()) * y_PDF_maxs

# From PREDICTED Weibull to PREDICTED data
y_pred_samples = stats.exponweib.rvs(1, y_pred_nonnorm[:,0], 0, y_pred_nonnorm[:,1], size=len(y_pred_nonnorm))


# Plotting PREDICTED data by anemometer
# density_scatter(np.deg2rad(X_data_nonnorm[test_idxs,1]), y_pred_samples)
# plt.title(f'Synthetic data from predicted Weibull params. \n {anem_to_test}')
# plt.ylim([0, 4])
# ax.text(np.deg2rad(18), 4.4, '$\sigma(u)\/[m/s]$')
# plt.tight_layout()
# plt.savefig(os.path.join(os.getcwd(), 'plots', f'predicted_synth_std_u_{anem_to_test}.png'))
# plt.show()


x = np.deg2rad(X_data_nonnorm[test_idxs,1])
y1 = y_data_nonnorm[test_idxs]
y2 = y_pred_samples

data1, x1_e, y1_e = np.histogram2d(np.sin(x) * y1, np.cos(x) * y1, bins=30, density=True)
data2, x2_e, y2_e = np.histogram2d(np.sin(x) * y2, np.cos(x) * y2, bins=[x1_e, y1_e], density=True)
x1_i = 0.5 * (x1_e[1:] + x1_e[:-1])
y1_i = 0.5 * (y1_e[1:] + y1_e[:-1])
x2_i = 0.5 * (x2_e[1:] + x2_e[:-1])
y2_i = 0.5 * (y2_e[1:] + y2_e[:-1])
z1 = interpn((x1_i, y1_i), data1, np.vstack([np.sin(x) * y1, np.cos(x) * y1]).T, method="splinef2d", bounds_error=False)
z2 = interpn((x2_i, y2_i), data2, np.vstack([np.sin(x) * y2, np.cos(x) * y2]).T, method="splinef2d", bounds_error=False)
z1[np.where(np.isnan(z1))] = 0.0  # To be sure to plot all data
z2[np.where(np.isnan(z2))] = 0.0  # To be sure to plot all data
idx1 = z1.argsort()  # Sort the points by density, so that the densest points are plotted last
idx2 = z2.argsort()  # Sort the points by density, so that the densest points are plotted last
x1, y1, z1 = x[idx1], y1[idx1], z1[idx1]
x2, y2, z2 = x[idx2], y2[idx2], z2[idx2]
# 1st plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=300, subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.scatter(x1, y1, c=z1, s=1, alpha=0.3)
norm = Normalize(vmin=np.min([z1, z2]), vmax=np.max([z1, z2]))
cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
cbar.ax.set_ylabel('Density')
plt.title(f'Measured. {anem_to_test}')
plt.ylim([0, 4])
ax.text(np.deg2rad(20), 4.4, '$\sigma(u)\/[m/s]$')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'plots', f'__measured_std_u_{anem_to_test}.png'))
plt.show()
# 2nd plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=300, subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.scatter(x2, y2, c=z2, s=1, alpha=0.3)
norm = Normalize(vmin=np.min([z1, z2]), vmax=np.max([z1, z2]))
cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
cbar.ax.set_ylabel('Density')
plt.title(f'Synthetic from predicted Weibull params. {anem_to_test}')
plt.ylim([0, 4])
ax.text(np.deg2rad(20), 4.4, '$\sigma(u)\/[m/s]$')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'plots', f'PREDICTED_synth_std_u_{anem_to_test}.png'))
plt.show()




# plt.plot(y_test_nonnorm[4500:5000,1], label = 'Test')
# plt.plot(y_pred_nonnorm[4500:5000,1], label = 'Pred')
# plt.legend()
# plt.show()



##################################################################################################################
# WEIBULL PARAMS - TRAINING FROM 18 ALTERNATE-10-DEG-WIDE-WIND-SECTORS AND TESTING THE REMAINING 18 SECTORS, AT EACH ANEMOMETER
##################################################################################################################
# todo: copied from the std_u section. Needs to be adapted to Weibull params

# Remove the direction, to be extra certain that the NN doesn't "cheat"
# X_data = np.delete(X_data, 1, axis=1) # NOT WORKING FOR THE BEAUTIFUL PLOTS THAT WILL REQUIRE THESE VALUES

# Separating training and testing data
# train_angle_domain = [[x, x+22.5] for x in np.arange(0, 360, 45)]  # in degrees
# test_angle_domain  = [[x+22.5, x+45] for x in np.arange(0, 360, 45)]  # in degrees
train_angle_domain = [[x, x+20] for x in np.arange(0, 360, 60)]  # in degrees
test_angle_domain  = [[x+35, x+45] for x in np.arange(0, 360, 60)]  # in degrees
train_bools = np.logical_or.reduce([(a[0]<X_data_nonnorm[:,1]) & (X_data_nonnorm[:,1]<a[1]) for a in train_angle_domain])  # https://stackoverflow.com/questions/20528328/numpy-logical-or-for-more-than-two-arguments
test_bools =  np.logical_or.reduce([(a[0]<X_data_nonnorm[:,1]) & (X_data_nonnorm[:,1]<a[1]) for a in test_angle_domain])
X_train = Tensor(X_data[train_bools]).to(device)
y_train = Tensor(y_PDF_data[train_bools]).to(device)
X_test =  Tensor(X_data[test_bools]).to(device)
y_test =  Tensor(y_PDF_data[test_bools]).to(device)

n_samples_train = X_train.shape[0]
batch_size_possibilities = sympy.divisors(n_samples_train)  # [1, 2, 4, 23, 46, 92, 4051, 8102, 16204, 93173, 186346, 372692]


# Getting values to predict and predicted values
hp = {'lr':1E-1, 'batch_size':6329, 'weight_decay':1E-4, 'momentum':0.9, 'n_epochs':25, 'n_hid_layers':1, 'activation':torch.nn.ReLU, 'loss':MSELoss()}
y_pred = train_and_test_NN(X_train, y_train, X_test, y_test, hp=hp, print_loss_per_epoch=True)

# Choosing only the results of a given anemometer (e.g. svar -> Svarvahelleholmen)
anem_train = np.searchsorted(start_idxs_of_each_anem, np.where(train_bools)[0], side='right') - 1  # to which anemometer, (indexed from mast_anem_list), does each test sample belong to
anem_test =  np.searchsorted(start_idxs_of_each_anem, np.where(test_bools )[0], side='right') - 1
train_idxs_svar = np.where(anem_train == np.where(mast_anem_list == 'svar_A')[0])[0]
test_idxs_svar  = np.where(anem_test  == np.where(mast_anem_list == 'svar_A')[0])[0]
X_train_svar = X_train[train_idxs_svar].cpu().numpy()
y_train_svar = np.squeeze(y_train[train_idxs_svar].cpu().numpy())  # simply converting to numpy and removing the empty dimension of the shape (n_train_samples,1)
X_test_svar = X_test[test_idxs_svar].cpu().numpy()
y_test_svar = np.squeeze(y_test[test_idxs_svar].cpu().numpy())  # simply converting to numpy and removing the empty dimension of the shape (n_test_samples,1)
y_pred_svar = np.squeeze(y_pred[test_idxs_svar].cpu().numpy())  # simply converting to numpy and removing the empty dimension of the shape (n_test_samples,1)

# De-normalizing
dir_train_svar = X_train_svar[:,1] * X_maxs[1]
dir_test_svar = X_test_svar[:,1] * X_maxs[1]
std_u_train_svar = y_train_svar * y_max
std_u_test_svar  = y_test_svar * y_max
std_u_pred_svar = y_pred_svar * y_max

# Organizing the results into sectors
train_sector_bools = [(a[0]<dir_train_svar) & (dir_train_svar<a[1]) for a in train_angle_domain]
test_sector_bools  = [(a[0]<dir_test_svar)  & (dir_test_svar<a[1])  for a in test_angle_domain]
train_sector_idxs = [np.where(train_sector_bools[i])[0] for i in range(len(train_sector_bools))]
test_sector_idxs  = [np.where(test_sector_bools[i] )[0] for i in range(len(test_sector_bools ))]
dir_means_train_per_sector_svar =   np.array([np.mean(dir_train_svar[l]) for l in train_sector_idxs])
dir_means_test_per_sector_svar  =   np.array([np.mean(dir_test_svar[l] ) for l in test_sector_idxs ])
std_u_means_train_per_sector_svar = np.array([np.mean(std_u_train_svar[l]) for l in train_sector_idxs])
std_u_means_test_per_sector_svar  = np.array([np.mean(std_u_test_svar[l]) for l in test_sector_idxs])
std_u_means_pred_per_sector_svar = np.array([np.mean(std_u_pred_svar[l]) for l in test_sector_idxs])
std_u_std_train_per_sector_svar = np.array([np.std(std_u_train_svar[l]) for l in train_sector_idxs])
std_u_std_test_per_sector_svar  = np.array([np.std(std_u_test_svar[l] ) for l in test_sector_idxs ])
std_u_std_pred_per_sector_svar =   np.array([np.std( std_u_pred_svar[l]) for l in test_sector_idxs])

# Plotting beautiful plots
fig = plt.figure(figsize=(8,6), dpi=400)
ax = fig.add_subplot(projection='polar')
plt.title('Anemometer at Svarvahelleholmen (Z = 48 m)\n')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.scatter( np.deg2rad(dir_train_svar), std_u_train_svar, s=1, alpha=0.6, c='lightgreen', label='Training data')
ax.scatter( np.deg2rad(dir_test_svar) , std_u_test_svar , s=1, alpha=0.6, c='skyblue', label='Testing data')
ax.errorbar(np.deg2rad(dir_means_train_per_sector_svar), std_u_means_train_per_sector_svar, std_u_std_train_per_sector_svar, c='forestgreen', elinewidth=3, alpha=0.9, fmt='.', label='$\sigma(train)$')
ax.errorbar(np.deg2rad(dir_means_test_per_sector_svar) , std_u_means_test_per_sector_svar , std_u_std_test_per_sector_svar , c='dodgerblue', elinewidth=4, alpha=0.8, fmt='o', label='$\sigma(test)$')
ax.errorbar(np.deg2rad(dir_means_test_per_sector_svar) , std_u_means_pred_per_sector_svar , std_u_std_pred_per_sector_svar , c='orange', elinewidth=2, alpha=0.9, fmt='.', label='Prediction', zorder=5)
handles, labels = ax.get_legend_handles_labels()
plt.ylim((None,4))
ax.text(np.deg2rad(18), 4.4, '$\sigma(u)\/[m/s]$')
plt.savefig(os.path.join(os.getcwd(), 'plots', 'std_u_Svar.png'))
plt.show()
fig = plt.figure(figsize=(2, 1.6), dpi=400)
plt.axis('off')
plt.legend(handles, labels)
plt.savefig(os.path.join(os.getcwd(), 'plots', 'std_u_Svar_legend.png'))
plt.show()















# TRASH:
# OLD VERSION USING curve_fit and my own weibull function. It didn't work properly
# n_bins = 100
# hist = np.histogram(y_data_nonnorm[idxs_360dir], bins=n_bins, density=True)
# hist_x = [(a+b)/2 for a,b in zip(hist[1][:-1], hist[1][1:])]
# hist_y = hist[0]
# try:
#     popt, pcov = curve_fit(weibull_PDF, hist_x, hist_y)  # param. optimal values, param estimated covariance
# except:
#     print(f'exception at anem {anem_idx} for dir {d}')
#     popt, pcov = curve_fit(weibull_PDF, hist_x, hist_y, bounds=(1E-5, 10))
# plt.plot(weibull_x, weibull_PDF(weibull_x, *popt), label='my_func')
# plt.plot(weibull_x, weibull_PDF(weibull_x, *popt2), label='weibfit')


# PLOTTING HISTOGRAMS AND PDF FITS
for anem_idx in [5]:
    anem_slice = slice(start_idxs_of_each_anem_2[anem_idx], start_idxs_of_each_anem_2[anem_idx + 1])
    for d in [180]:
        idxs_360dir = np.where(X_360dirs[anem_slice] == d)[0]
        if len(idxs_360dir) > 5 :  # minimum number of datapoints, otherwise -> np.nan
            params_weib    = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir], floc = 0, f0 = 1)  # params = a(alpha), c(shape==k), loc, scale(lambda)
            params_expweib = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir], floc=0)
            params_expweib2 = stats.exponweib.fit(y_data_nonnorm[anem_slice][idxs_360dir])

plt.figure()
plt.title(f'Wind from {d-1}$\u00B0C$ to {d}$\u00B0C$. Anemometer: "{mast_anem_list[anem_idx]}"')
weibull_x = np.linspace(0, 5, 100)
plt.plot(weibull_x, stats.exponweib.pdf(weibull_x, *params_weib   ), label='Weibull fit (2 parameters)', lw=3, alpha=0.8)
plt.plot(weibull_x, stats.exponweib.pdf(weibull_x, *params_expweib), label='Exp. Weibull fit (3 parameters)', lw=3, alpha=0.8)
plt.plot(weibull_x, stats.exponweib.pdf(weibull_x, *params_expweib2), label='Exp. Weibull fit 2 (4 parameters)', lw=3, alpha=0.8)
plt.hist(y_data_nonnorm[anem_slice][idxs_360dir], bins=50, density=True, label='Normalized histogram of $\sigma(u)$', alpha=0.5)
plt.legend()
plt.show()
