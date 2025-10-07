import os
import torch
import time
import copy
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from run_robot.arg_parser import argument_parser
from plot_parser import plt_parser
from plants.robots import RobotsSystem, RobotsDataset
from plot_functions import plot_trajectories, plot_trajectories_mean_and_std, plot_traj_vs_time, plot_traj_mean_and_std
from controllers.PB_controller import PerfBoostController
from loss_functions_BF import RobotsLoss
from scipy.io import savemat




BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----- Overwriting plot options -----
plt_set = plt_parser()
plt_set.document = 'ppt'
plt_set.plot_dim_x = 12.0 # (inch). Dimension of the x-axis
plt_set.plot_dim_y = 5.0 # (inch). Dimension of the y-axis

# ----- Overwriting arguments -----
args = argument_parser()
# System
args.n_agents = 1
# Dataset
args.num_rollouts = 8
args.num_rollouts_valid = 2
args.num_rollouts_test = 5
args.horizon = 250 # Number of samples for a simulation in S
args.epochs = 6899 # Real number of epochs is +1
args.std_init_plant = 0.2 # IC standard deviation (to generate the dataset)
args.std_noise_plant = 0.005 # Noise standard deviation (to generate the dataset)
args.log_epoch = args.epochs//10 if args.epochs//10 > 0 else 1
args.batch_size = args.num_rollouts # full batch
# DNN
args.nn_type = "REN"
args.non_linearity = "tanh"
args.dim_internal = 4 # dimension of the REN internal state
args.dim_nl = 4 # dimension of the input for the nonlinear element of the REN
args.cont_init_std = 0.1 # Weights initialization std
args.lr = 1e-3  # Learning rate
t_ext = args.horizon

args.alpha_barrier_x = 1000000
args.alpha_barrier_u = 0

# Obstacle avoidance
args.alpha_obst = 1e1 #Defoult 5e3
args.delta_obst = 1.1 # Percentage of the safe distance within the run_robot and the obstacle

#saving results
args.saving = True
# Loading previous results
args.load = False

# Initial state and reference HERE
plant_input_init = None     # all zero
plant_state_init = None     # Initial state of the plant without w_0

# ------------ 2. Datasets creation ------------
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, std_noise=args.std_noise_plant) #Generating the sample for different ICs extracted from a Gaussian distribution
# divide to train and test
train_data, valid_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_valid_samples= args.num_rollouts_valid,  num_test_samples= args.num_rollouts_test)
# data for plots
plot_data = torch.zeros(1, t_ext, train_data.shape[-1])
plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
# batch the data
seed_for_batching = 2022
generator = torch.Generator() # Fixing a generator for the train_dataloader
generator.manual_seed(seed_for_batching)
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator) # Training on minibatch based on batch_size
# DataLoader for plot (the batch is all the training data)
train_dataloader_epoch = DataLoader(train_data, batch_size=train_data.shape[0], shuffle=False)

# ------------ 3. Plant (with prestabilizing controller) ------------
sys = RobotsSystem(n_agents=args.n_agents,
                   xbar=dataset.xbar,
                   x_init=plant_state_init,
                   u_init=plant_input_init,
                   linear_plant=args.linearize_plant,
                   k=args.spring_const,
                   x_max=[float('inf'), float('inf'), 0.5, 0.5],
                   x_min=[-float('inf'), -float('inf'), -0.5, -0.5],
                   u_max=[float('inf'), float('inf')],  # [0.5, 0.5],
                   u_min=[-float('inf'), -float('inf')]  # [-0.5, -0.5]
                   )

# ------------ 4. Controller ------------
ctl = PerfBoostController(noiseless_forward=sys.noiseless_forward,
                          input_init=sys.x_init,
                          output_init=sys.u_init,
                          nn_type=args.nn_type,
                          non_linearity=args.non_linearity,
                          dim_internal=args.dim_internal,
                          dim_nl=args.dim_nl,
                          initialization_std=args.cont_init_std
                          )
# Count the number of parameters
total_n_params = sum(p.numel() for p in ctl.parameters() if p.requires_grad)
print("[INFO] Number of parameters: %i" % total_n_params)

# plot the noise-free closed-loop trajectories before training the controller with the initial parameters of the performance-boosting controller
x_log_nf, _, u_log_nf = sys.rollout(ctl, plot_data)
plot_trajectories(x_log_nf[0, :, :], T=t_ext, x_max = sys.x_max[:2], x_min = sys.x_min[:2], plt_opt = plt_set, str_title = 'Noise-free')
plot_traj_vs_time(args.horizon, x_log_nf[0, :args.horizon, :], u_log_nf[0, :args.horizon, :], x_max = sys.x_max, x_min = sys.x_min, u_max = sys.u_max, u_min = sys.u_min, plt_opt = plt_set, singl = True, str_title = 'Noise-free')

# ------------ 5. Loss ------------
args.Q = 1*torch.eye(sys.state_dim)
args.R = 0.1*torch.eye(sys.in_dim)

loss_fn = RobotsLoss(
    Q = args.Q , R = args.R,
    alpha_barrier_u = args.alpha_barrier_u, alpha_barrier_x = args.alpha_barrier_x,
    sys = sys,
    alpha_obst=args.alpha_obst, delta_obst=args.delta_obst  # TODO see how to menage (also for a potential reference)
)
# ------------ 6. Optimizer ------------
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)

# ------------ 7. Training ------------
print('------------ Begin training ------------')
best_valid_loss = 1e6
# Initialization of the list for plotting
LOSS_TOT_TR = np.zeros(args.epochs + 1)
LOSS_PER_TR = np.zeros(args.epochs + 1)
LOSS_CONSTR_TR = np.zeros(args.epochs + 1)
LOSS_OBS_TR = np.zeros(args.epochs + 1)
LOSS_TOT_V = np.zeros(args.epochs + 1)
LOSS_PER_V = np.zeros(args.epochs + 1)
LOSS_CONSTR_V = np.zeros(args.epochs + 1)
LOSS_OBS_V = np.zeros(args.epochs + 1)

best_params = ctl.state_dict()

f_best = False  # Flag for plotting strings
t = time.time()
epoch_best = 0

if args.load:
    filename_load ='PB_current_parameters_with_None_epochs_and_initial_std_0.1'
    data = sio.loadmat(filename_load + '.mat')
    epoch = data.get('epoch')
    LOSS_TOT_TR = data.get('LOSS_TOT_TR')
    LOSS_PER_TR = data.get('LOSS_PER_TR')
    LOSS_CONSTR_TR = data.get('LOSS_CONSTR_TR')
    LOSS_OBS_TR = data.get('LOSS_OBS_TR')
    LOSS_TOT_V = data.get('LOSS_TOT_V')
    LOSS_PER_V = data.get('LOSS_PER_V')
    LOSS_CONSTR_V = data.get('LOSS_CONSTR_V')
    LOSS_OBS_V = data.get('LOSS_OBS_V')
    checkpoint = torch.load(filename_load + '.pth')
    ctl.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



# Performance boosting Step
for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()
        # simulate over horizon steps
        x_log, _, u_log = sys.rollout(
            controller=ctl, data=train_data_batch, train=True,
        )
        # loss of this rollout
        loss_tot_roll, loss_per_roll, loss_constr_roll, loss_obs_roll = loss_fn.forward(x_log, u_log)
        # take a step
        loss_tot_roll.backward()
        optimizer.step()

    LOSS_TOT_TR[epoch] = loss_tot_roll.detach()
    LOSS_PER_TR[epoch] = loss_per_roll.detach()
    LOSS_CONSTR_TR[epoch] = loss_constr_roll.detach()
    LOSS_OBS_TR[epoch] = loss_obs_roll.detach()


    if args.return_best:
        # roll out the current controller on the valid data
        with torch.no_grad():
            x_log_valid, _, u_log_valid = sys.rollout(
                controller=ctl, data=valid_data, train=False,
            )
            # loss of the valid data
            loss_tot_valid, loss_per_valid, loss_constr_valid, loss_obs_valid = loss_fn.forward(x_log_valid, u_log_valid)

            if args.alpha_barrier_x > 0 or args.alpha_barrier_u > 0:
                LOSS_TOT_V[epoch] = loss_tot_valid.detach()
                LOSS_PER_V[epoch] = loss_per_valid.detach()
                LOSS_CONSTR_V[epoch] = loss_constr_valid.detach()
                LOSS_OBS_V[epoch] = loss_obs_valid.detach()

        # compare with the best valid loss
        if loss_tot_valid.item() < best_valid_loss:
            best_valid_loss = loss_tot_valid.item()
            best_params = copy.deepcopy(ctl.state_dict())
            f_best = True
            epoch_best = epoch

    # print info
    if epoch % args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.4f' % (epoch, loss_tot_roll.item())
        msg += ' ---||--- validation loss: %.4f' % loss_tot_valid.item()
        duration = time.time() - t
        msg += ' ---||--- time for  %i epochs: %.0f s' % (args.log_epoch, duration)
        msg += ' best in epoch %i' % (epoch_best)
        if f_best:
            msg += ' (!! best so far !!)'
            f_best = False
        print(msg)

        t = time.time()

    if (epoch % 500) == 0:
        print('Saving Parameters')
        file_name = 'PB_current_parameters_with_' + str(epoch) + '_epochs_and_initial_std_' + str(
            args.cont_init_std)
        saving_dict = {
            'epoch': epoch,
            'LOSS_TOT_TR': LOSS_TOT_TR,
            'LOSS_PER_TR': LOSS_PER_TR,
            'LOSS_CONSTR_TR': LOSS_CONSTR_TR,
            'LOSS_OBS_TR': LOSS_OBS_TR,
            'LOSS_TOT_V': LOSS_TOT_V,
            'LOSS_PER_V': LOSS_PER_V,
            'LOSS_CONSTR_V': LOSS_CONSTR_V,
            'LOSS_OBS_V': LOSS_OBS_V,
        }

        file_name_mat = file_name + '.mat'
        savemat(file_name_mat, saving_dict)
        checkpoint = {
            'model_state_dict': copy.deepcopy(ctl.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        file_name_pth = file_name + '.pth'
        torch.save(checkpoint, file_name_pth)



# ------------ 8. Plotting Results ------------
ctl.load_state_dict(best_params)
# Plot noise-free trajectories vs time after training
x_log_nf, _, u_log_nf = sys.rollout(ctl, plot_data)  # Noise-free plot
plot_traj_vs_time(t_ext, x_log_nf[0, :, :], u_log_nf[0, :, :],
                  x_max=sys.x_max, x_min=sys.x_min, u_max=sys.u_max, u_min=sys.u_min,
                  plt_opt=plt_set,
                  epoch_num=epoch_best, best=True, singl=True, str_title='Noise-free')
# Plot noise-free position's trajectories after training with an obstacle
if args.alpha_obst is not None:
    plot_trajectories(
        x_log_nf[0, :, :], T=t_ext, x_max = sys.x_max[:2], x_min = sys.x_min[:2],
        plt_opt = plt_set, str_title = 'Noise-free',
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_radius=loss_fn.obstacle_radius
    )

# Recovering training data
with torch.no_grad():
    x_log_tr, _, u_log_tr = sys.rollout(
        controller=ctl, data=train_data, train=False,
    )

plot_traj_mean_and_std(t_ext, x_log_tr, u_log_tr, x_max=sys.x_max, x_min=sys.x_min, u_max=sys.u_max, u_min=sys.u_min, plt_opt=plt_set,
                       epoch_num=epoch_best, best=True, singl=True, str_title='Training')

if args.alpha_obst is not None:
    plot_trajectories_mean_and_std( # Plot with std as an ellipse
        x_log_tr, T=t_ext, x_max = sys.x_max[:2], x_min = sys.x_min[:2],
        plt_opt = plt_set, str_title = 'Training',
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_radius=loss_fn.obstacle_radius, ell = 1
    )
    plot_trajectories_mean_and_std( # Plot with std as an errorbar
        x_log_tr, T=t_ext, x_max = sys.x_max[:2], x_min = sys.x_min[:2],
        plt_opt = plt_set, str_title = 'Training',
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_radius=loss_fn.obstacle_radius, ell = 0
    )

plt.plot(LOSS_TOT_TR)
plt.show()
plt.plot(LOSS_PER_TR)
plt.show()
plt.plot(LOSS_CONSTR_TR)
plt.show()
plt.plot(LOSS_OBS_TR)
plt.show()

# Recovering validation data
with torch.no_grad():
    x_log_valid, _, u_log_valid = sys.rollout(
        controller=ctl, data=valid_data, train=False,
    )
plot_traj_mean_and_std(t_ext, x_log_valid, u_log_valid, x_max=sys.x_max, x_min=sys.x_min, u_max=sys.u_max, u_min=sys.u_min, plt_opt=plt_set,
                       epoch_num=epoch_best, best=True, singl=True, str_title='Validation')

if args.alpha_obst is not None:
    plot_trajectories_mean_and_std( # Plot with std as an ellipse
        x_log_valid, T=t_ext, x_max = sys.x_max[:2], x_min = sys.x_min[:2],
        plt_opt = plt_set, str_title = 'Validation',
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_radius=loss_fn.obstacle_radius, ell = 1
    )
    plot_trajectories_mean_and_std( # Plot with std as an errorbar
        x_log_valid, T=t_ext, x_max = sys.x_max[:2], x_min = sys.x_min[:2],
        plt_opt = plt_set, str_title = 'Validation',
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_radius=loss_fn.obstacle_radius, ell = 0
    )

plt.plot(LOSS_TOT_V)
plt.show()
plt.plot(LOSS_PER_V)
plt.show()
plt.plot(LOSS_CONSTR_V)
plt.show()
plt.plot(LOSS_OBS_V)
plt.show()

# Recovering testing data
with torch.no_grad():
    x_log_test, _, u_log_test = sys.rollout(
        controller=ctl, data=test_data, train=False,
    )
plot_traj_mean_and_std(t_ext, x_log_test, u_log_test, x_max=sys.x_max, x_min=sys.x_min, u_max=sys.u_max, u_min=sys.u_min, plt_opt=plt_set,
                        epoch_num=epoch_best, best=True, singl=True, str_title='Testing')

if args.alpha_obst is not None:
    plot_trajectories_mean_and_std( # Plot with std as an ellipse
        x_log_test, T=t_ext, x_max = sys.x_max[:2], x_min = sys.x_min[:2],
        plt_opt = plt_set, str_title = 'Testing',
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_radius=loss_fn.obstacle_radius, ell = 1
    )
    plot_trajectories_mean_and_std( # Plot with std as an errorbar
        x_log_test, T=t_ext, x_max = sys.x_max[:2], x_min = sys.x_min[:2],
        plt_opt = plt_set, str_title = 'Testing',
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_radius=loss_fn.obstacle_radius, ell = 0
    )

print('------------ Checking Constraints for Training data ------------')
sys.checking_constraints(x_log=x_log_tr, u_log=u_log_tr)
print('------------ Checking Constraints for Validation data ------------')
sys.checking_constraints(x_log=x_log_valid, u_log=u_log_valid)
print('------------ Checking Constraints for Testing data ------------')
sys.checking_constraints(x_log=x_log_test, u_log=u_log_test)

input("Press Enter to save...")


# ------------ 9. Saving Results ------------
if args.saving:

    if args.std_noise_plant == 0:
        file_name = 'PB_BF'
    else:
        file_name = 'PB_BF_noise'

    file_name += '_delta_obs' + str(args.delta_obst) + '_n_e' + str(args.epochs) + '_stdTheta' + str(args.cont_init_std)
    saving_dict = {
    'args_num_rollouts': args.num_rollouts,
    'args_horizon': args.horizon,
    'args_epochs': args.epochs,
    'args_std_init_plant': args.std_init_plant,
    'args_batch_size': args.batch_size,
    'args_nn_type': args.nn_type,
    'args_non_linearity': args.non_linearity,
    'args_dim_internal': args.dim_internal,
    'args_dim_nl': args.dim_nl,
    'args_cont_init_std': args.cont_init_std,
    'args_lr': args.lr,
    'args_alpha_barrier_x': args.alpha_barrier_x,
    'args_alpha_barrier_u': args.alpha_barrier_u,
    'x_nf': x_log_nf, #Noise-free state evolution
    'u_nf': u_log_nf, #Noise-free input evolution
    'x_tr': x_log_tr,  # State evolution with training data
    'u_tr': u_log_tr,  # Input evolution with training data
    'x_valid': x_log_valid,  # State evolution with validation data
    'u_valid': u_log_valid,  # Input evolution with validation data
    'x_test': x_log_test,  # State evolution with testing data
    'u_test': u_log_test,  # Input evolution with testing data
    'LOSS_TOT_TR': LOSS_TOT_TR,  # Total training loss
    'LOSS_PER_TR': LOSS_PER_TR,  # Performance training loss
    'LOSS_CONSTR_TR': LOSS_CONSTR_TR,  # Constraint training loss
    'LOSS_OBS_TR': LOSS_OBS_TR,  # Obstacle training loss
    'LOSS_TOT_V': LOSS_TOT_V,  # Total validation loss
    'LOSS_PER_V': LOSS_PER_V,  # Performance validation loss
    'LOSS_CONSTR_V': LOSS_CONSTR_V,  # Constraint validation loss
    'LOSS_OBS_V': LOSS_OBS_V,  # Obstacle validation loss
    'sys_n_agents': sys.n_agents,
    'sys_x_max': sys.x_max,
    'sys_x_min': sys.x_min,
    'sys_u_max': sys.u_max,
    'sys_u_min': sys.u_min,
    'minibatch_seed': seed_for_batching,
    'std_parameters': args.cont_init_std,
    'alpha_obs': args.alpha_obst,
    'delta_obs': args.delta_obst
    }
    if args.alpha_barrier_x>0 or args.alpha_barrier_u>0:
        alpha_barrier_x_string = "{:e}".format(args.alpha_barrier_x)
        mantis_alpha_barrier_x, esp_alpha_barrier_x = alpha_barrier_x_string.split('e')
        alpha_barrier_x_string = mantis_alpha_barrier_x.rstrip('0') + 'e' + str(int(esp_alpha_barrier_x))
        alpha_barrier_u_string = "{:e}".format(args.alpha_barrier_u)
        mantis_alpha_barrier_u, esp_alpha_barrier_u = alpha_barrier_u_string.split('e')
        alpha_barrier_u_string = mantis_alpha_barrier_u.rstrip('0') + 'e' + str(int(esp_alpha_barrier_u))
        file_name += '_BF' + '_alpha_x' + str(alpha_barrier_x_string) + '_alpha_u' + str(alpha_barrier_u_string)
    else:
        file_name += '_only_perf'

    file_name += '.mat'
    savemat(file_name, saving_dict)
