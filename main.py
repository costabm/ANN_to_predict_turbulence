import datetime
import sympy
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
from read_0p1sec_data import create_processed_data_files, compile_all_processed_data_into_1_file
from find_storms import create_storm_data_files, compile_storm_data_files, find_storm_timestamps, organized_dataframes_of_storms
import matplotlib.pyplot as plt

from elevation_profile_generator import elevation_profile_generator, plot_elevation_profile, get_point2_from_point1_dir_and_dist


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


def test_elevation_profile_at_given_point_dir_dist(point_1=[-34625., 6700051.], step_distance=10., direction_deg=160, total_distance=5000., plot=True):
    point_2 = get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=direction_deg, distance=total_distance)
    dists, heights = elevation_profile_generator(point_1=point_1, point_2=point_2, step_distance=step_distance)
    if plot:
        plot_elevation_profile(point_1=point_1, point_2=point_2, step_distance=step_distance)
    return dists, heights


def get_all_10_min_data_at_z_48m(U_min = 5, step_distance=10., total_distance=5000.):
    print('Collecting all 10-min wind data... (takes approx. 3 minutes)')
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
    # Organizing the data into an input matrix (with shape shape (n_samples, n_features)):
    X_data = []
    y_data = []
    mast_UTM_33 = {'synn':[-34515., 6705758.], 'osp1':[-39375., 6703464.], 'osp2':[-39350., 6703204.], 'svar':[-34625., 6700051.]}
    for mast_anem in ['osp1_A', 'osp1_B', 'osp2_A', 'osp2_B', 'svar_A', 'synn_A']:
        # We allow different data lengths for each anemometer, but a row with at least one nan is removed from a given anemometer df of 'means' 'dirs' and 'stds'
        X_mean_dir_std_anem = pd.DataFrame({'means': X_means_df[mast_anem], 'dirs': X_dirs_df[mast_anem], 'stds': X_std_u_df[mast_anem]}).dropna(axis=0, how='any')
        X_mean_anem =   np.array(X_mean_dir_std_anem['means'])
        X_dir_anem =    np.array(X_mean_dir_std_anem['dirs'])
        X_std_u_anem =  np.array(X_mean_dir_std_anem['stds'])
        point_1 = mast_UTM_33[mast_anem[:4]]
        points_2 = np.array([get_point2_from_point1_dir_and_dist(point_1=point_1, direction_deg=d, distance=total_distance) for d in X_dir_anem])
        heights = np.array([elevation_profile_generator(point_1=point_1, point_2=p2, step_distance=step_distance)[1] for p2 in points_2])
        X_data_1_sample = np.concatenate((X_mean_anem[:,None], X_dir_anem[:,None], heights), axis=1)
        X_data.append(X_data_1_sample)
        y_data.append(X_std_u_anem)
        print(f'{mast_anem}: All ML-input-data is collected')
    X_data = np.concatenate(tuple(i for i in X_data), axis=0)  # n_steps + 2 <=> number of terrain heights (e.g. 500) + wind speed (1) + wind dir (1)
    y_data = np.concatenate(y_data)
    return  X_data, y_data

########################################
# MACHINE LEARNING
########################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'cuda' or 'cpu'. 'cuda' doesn't seem to be working...

##################################################################
# Getting the data                                              ##
X_data_nonnorm, y_data_nonnorm = get_all_10_min_data_at_z_48m() ##  takes a few minutes...
##################################################################

# Normalizing data
X_maxs = np.max(X_data_nonnorm, axis=0)
y_max = np.max(y_data_nonnorm)
X_data = X_data_nonnorm/X_maxs
y_data = y_data_nonnorm/y_max


# Remove the direction, to be extra certain that the NN doesn't "cheat"
X_data = np.delete(X_data, 1, axis=1) # !!!!


# Separating training and testing data
train_angle_domain = [[x, x+5] for x in np.arange(0, 360, 10)]  # in degrees
train_bools = np.logical_or.reduce([(f[0]<X_data_nonnorm[:,1]) & (X_data_nonnorm[:,1]<f[1]) for f in train_angle_domain])  # https://stackoverflow.com/questions/20528328/numpy-logical-or-for-more-than-two-arguments
test_bools = np.array([not i for i in train_bools])
X_train = Tensor(X_data[train_bools]).to(device)
y_train = Tensor(y_data[train_bools]).to(device)
X_test =  Tensor(X_data[test_bools]).to(device)
y_test =  Tensor(y_data[test_bools]).to(device)

# Hyperparameters
n_samples_train = X_train.shape[0]
n_hidden_layers = 2
learn_rate = 5E-4
weight_decay = 1E-4
momentum = 0.9
n_epochs = 50
batch_size_desired = 100  # desired because it might not be an integer divisor of n_samples_train, therefore the nearest divisor is found and used
batch_size = min(sympy.divisors(n_samples_train), key=lambda x:abs(x-batch_size_desired))  # Finds the integer divisor of n_samples_train that is closer to the batch_size_desired
print(f'Batch size is {batch_size}')

# Building a neural network dynamically
torch.manual_seed(0)  # make the following random numbers reproducible
n_features = X_train.shape[1]  # number of independent variables in the polynomial
n_hidden_layer_neurons = n_features  # Fancy for: More monomials, more neurons...
my_nn = torch.nn.Sequential()
my_activation_func = torch.nn.ELU  # Relu, ELU, LeakyReLU, etc.
my_nn.add_module(name='0', module=torch.nn.Linear(n_features, n_hidden_layer_neurons))  # Second layer
my_nn.add_module(name='0A', module=my_activation_func())  # Activation function
for i in range(n_hidden_layers):  # Hidden layers
    my_nn.add_module(name=str(i + 1), module=torch.nn.Linear(n_hidden_layer_neurons, n_hidden_layer_neurons))
    my_nn.add_module(name=f'{i + 1}A', module=my_activation_func())
my_nn.add_module(name=str(n_hidden_layers + 1), module=torch.nn.Linear(n_hidden_layer_neurons, 1))  # Output layer
criterion = MSELoss()  # define the loss function
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
        loss = criterion(y_pred, Variable(y_train[batch_idxs].view(batch_size,1), requires_grad=False))
        epoch_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
    # writer.add_scalar('Training Loss', epoch_loss, global_step=epoch)  # writing to TensorBoard

# Testing
n_samples_test = 10000
test_idxs_all_shuffled = torch.randperm(X_test.shape[0])
test_idxs = test_idxs_all_shuffled[:n_samples_test]

with torch.no_grad():
    y_test_pred = my_nn(Variable(X_test[test_idxs]))
    # if normalize_y is False:
    #     y_test_pred = normalize_y_func(y_test_pred, denormalize=True)
    SS_res_test = torch.sum((y_test[test_idxs].view(n_samples_test,1) - y_test_pred) ** 2)
    t_test_pred_mean = torch.mean(y_test_pred)
    SS_tot_test = torch.sum((y_test[test_idxs] - t_test_pred_mean) ** 2)
    R2_test = 1 - SS_res_test / SS_tot_test
print('SS of residuals on test dataset: ' + str(SS_res_test))
print('R2 on test dataset: ' + str(R2_test) + ' <----------------------')
print(f"Prediction: {y_test_pred[-8:].flatten()}")
print(f"Expected:   {y_test[test_idxs][-8:].flatten()}")

