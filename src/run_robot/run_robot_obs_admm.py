import os
import torch
from torch.optim import lr_scheduler

import cvxpy as cp
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


from torch.utils.data import DataLoader, TensorDataset
from run_robot.arg_parser import argument_parser
from plot_parser import plt_parser
from plants.robots import RobotsSystem, RobotsDataset
from plot_functions import plot_trajectories, plot_trajectories_mean_and_std, plot_traj_vs_time, plot_traj_mean_and_std, plt_loss_vs_epochs, plt_loss_mean_and_std , plt_quantity_vs_it
from controllers.PB_controller import PerfBoostController
from loss_functions_admm import RobotsLoss
from admm.admm_fun import AdmmFun
from plants.save_data import saving_data
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
args.num_rollouts_valid = 1 #int(0.1*args.num_rollouts)
args.num_rollouts_test = 5 #int(0.1*args.num_rollouts)
args.horizon = 250 # Number of samples for a simulation in S
args.epochs = 5
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
args.lr = 1e-3  # Learning rate REN
t_ext = args.horizon
# ADMM
# args.admm_penalty_update = False # Hyperparameter update flag
# args.rho = 0.5
args.admm_penalty_update = True # Hyperparameter update flag
args.rho_init = 0.5
args.mu = 10
args.tau_incr = 2
args.tau_decr = 2
if args.admm_penalty_update:
    rho = args.rho_init
else:
    rho = args.rho

args.eps_abs = 1e-5 # Absolute tolerance for the stopping criteria
args.eps_rel = 1e-4
# Scheduler
args.scheduler_it = 50 #50 #Number of iterations after which the scheduler updates the learning rate
args.scheduler_gamma = 0.5 #0.5 #Multiplicative factor for the learning rate every args.scheduler_it iterations
args.learning_rate_bound = 1e-6

# Obstacle avoidance
args.alpha_obst = 1e1
args.delta_obst = 1.1 # Percentage of the safe distance within the run_robot and the obstacle

# Relative tolerance for the stopping criteria
args.max_iter = 500000
#saving results
args.saving = True
# Loading previous results
args.load = False

# Initial state and reference
plant_input_init = None     # all zero
plant_state_init = None     # Initial state of the plant without w_0

# ------------ 2. Datasets creation ------------
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, std_noise=args.std_noise_plant) #Generating the sample for different ICs extracted from a Gaussian distribution
# divide to train and test
train_data, valid_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_valid_samples= args.num_rollouts_valid,  num_test_samples= args.num_rollouts_test)
# data for plots
plot_data = torch.zeros(1, t_ext, train_data.shape[-1])
plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)


# ------------ 3. Plant (with prestabilizing controller) ------------
sys = RobotsSystem(n_agents=args.n_agents,
                   xbar=dataset.xbar,
                   x_init=plant_state_init,
                   u_init=plant_input_init,
                   linear_plant=args.linearize_plant,
                   k=args.spring_const,
                   x_max = [float('inf'), float('inf'), 0.5, 0.5],
                   x_min = [-float('inf'), -float('inf'), -0.5, -0.5],
                   u_max = [float('inf'), float('inf')],
                   u_min = [-float('inf'), -float('inf')]
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

# ------------ 5. Creation of ADMM dataset with projected variables and Lagrangian multipliers ------------
# Allocation of ADMM function HERE
if args.admm_penalty_update:
    admm_f = AdmmFun(args.num_rollouts*(sys.state_dim+sys.in_dim)*args.horizon,
                      args.num_rollouts*(sys.state_dim+sys.in_dim)*args.horizon + total_n_params,
                      args.eps_abs,
                      args.eps_rel,
                      mu = args.mu,
                      tau_incr = args.tau_incr,
                      tau_decr = args.tau_decr)
else:
    admm_f = AdmmFun(args.num_rollouts*(sys.state_dim+sys.in_dim)*args.horizon,
                      args.num_rollouts*(sys.state_dim+sys.in_dim)*args.horizon + total_n_params,
                      args.eps_abs,
                      args.eps_rel)

# Projection Layer initialization (Optimization variables) (training)
xp = cp.Variable(shape=(args.num_rollouts*sys.state_dim, args.horizon))
up = cp.Variable(shape=(args.num_rollouts*sys.in_dim, args.horizon))
constr = [xp[::sys.state_dim, :] >= sys.x_min[0], xp[1::sys.state_dim, :] >= sys.x_min[1], xp[2::sys.state_dim, :] >= sys.x_min[2], xp[3::sys.state_dim, :] >= sys.x_min[3], # State lower bounds
          xp[::sys.state_dim, :] <= sys.x_max[0], xp[1::sys.state_dim, :] <= sys.x_max[1], xp[2::sys.state_dim, :] <= sys.x_max[2], xp[3::sys.state_dim, :] <= sys.x_max[3], # State upper bounds
          up[::sys.in_dim, :] >= sys.u_min[0], up[1::sys.in_dim, :] >= sys.u_min[1],# Input lower bounds
          up[::sys.in_dim, :] <= sys.u_max[0], up[1::sys.in_dim, :] <= sys.u_max[1] # Input upper bounds
          ]
# Projection Layer initialization (Optimization variables) (validation)
xp_v = cp.Variable(shape=(args.num_rollouts_valid*sys.state_dim, args.horizon))
up_v = cp.Variable(shape=(args.num_rollouts_valid*sys.in_dim, args.horizon))
constr_v = [xp_v[::sys.state_dim, :] >= sys.x_min[0], xp_v[1::sys.state_dim, :] >= sys.x_min[1], xp_v[2::sys.state_dim, :] >= sys.x_min[2], xp_v[3::sys.state_dim, :] >= sys.x_min[3], # State lower bounds
          xp_v[::sys.state_dim, :] <= sys.x_max[0], xp_v[1::sys.state_dim, :] <= sys.x_max[1], xp_v[2::sys.state_dim, :] <= sys.x_max[2], xp_v[3::sys.state_dim, :] <= sys.x_max[3], # State upper bounds
          up_v[::sys.in_dim, :] >= sys.u_min[0], up_v[1::sys.in_dim, :] >= sys.u_min[1],# Input lower bounds
          up_v[::sys.in_dim, :] <= sys.u_max[0], up_v[1::sys.in_dim, :] <= sys.u_max[1] # Input upper bounds
          ]


# Initialization of the projected state and input
# Rollout with the initialized controller parameters
with torch.no_grad():
    all_x_logs_t, _, all_u_logs_t = sys.rollout(
        controller=ctl, data=train_data, train=False,
    )

xp_opt_t, up_opt_t, lambda_x, lambda_u = admm_f._proj(all_x_logs_t, all_u_logs_t, xp, up, constr, lambda_x_prev = 0, lambda_u_prev = 0, num_rollouts=args.num_rollouts, args=args, sys = sys)

# Initializing ADMM dataset
train_data_admm = TensorDataset(torch.cat((train_data, xp_opt_t.to(torch.float32), up_opt_t.to(torch.float32), lambda_x.to(torch.float32), lambda_u.to(torch.float32)), dim=2))
# batch the data
seed_for_batching = 2022
generator = torch.Generator() # Fixing a generator for the train_dataloader
generator.manual_seed(seed_for_batching)
train_dataloader = DataLoader(train_data_admm, batch_size=args.batch_size, shuffle=True, generator=generator) # Training on minibatch based on batch_size
# DataLoader for plot (the batch is all the training data)
train_dataloader_epoch = DataLoader(train_data_admm, batch_size=train_data.shape[0], shuffle=False)

# ------------ 6. Loss ------------
args.Q = 1*torch.eye(sys.state_dim)
args.R = 0.1*torch.eye(sys.in_dim)

loss_fn = RobotsLoss(
    Q = args.Q , R = args.R,
    rho = rho,
    alpha_obst=args.alpha_obst, delta_obst = args.delta_obst
)

# ------------ 7. Optimizer ------------
# Initialization of the Lagrangian multiplier for validation
lambda_x_v = 0
lambda_u_v = 0

assert not (valid_data is None and args.return_best)

optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_it, gamma=args.scheduler_gamma)

# ------------ 8. Training ------------
print('------------ Begin training ------------')
best_valid_loss = 1e6 # Value given if no better ones are determined
best_params = ctl.state_dict() # Parameter initializations
t = time.time()

# ------------ 9. ADMM initializations ------------
# Convergence flag
conv_flag = False
# Initialization
f_best = False  # Flag for plotting strings
it_best = None
epoch_best = 0
i = 0
# Initialization of the list for plotting
LOSS_TOT = np.array([])
LOSS_PER = np.array([])
LOSS_CONSTR = np.array([])
LOSS_OBS = np.array([])
PRI_RES = np.array([])
EPS_PRI_RES = np.array([])
DUAL_RES = np.array([])
EPS_DUAL_RES = np.array([])
RHO = np.array([])

if args.load:
    filename_load ='PB_current_parameters_with_None_epochs_and_initial_std_0.1'
    data = sio.loadmat(filename_load + '.mat')
    xp_opt_t = data.get('xp_opt_t')
    up_opt_t = data.get('up_opt_t')
    lambda_x = data.get('lambda_x'),
    lambda_u = data.get('lambda_u')
    i = data.get('it')
    LOSS_TOT = data.get('LOSS_TOT'),
    LOSS_PER = data.get('LOSS_PER'),
    LOSS_CONSTR = data.get('LOSS_CONSTR'),
    LOSS_OBS = data.get('LOSS_OBS'),
    PRI_RES = data.get('pri_res_plt'),
    DUAL_RES = data.get('dual_res_plt'),
    EPS_PRI_RES = data.get('eps_pri_plt'),
    EPS_DUAL_RES = data.get('eps_dual_plt'),
    RHO = data.get('rho_plt')
    checkpoint = torch.load(filename_load + '.pth')
    ctl.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Projected variables at the previous iteration (needed for ADMM)
xu_logs_t_m = torch.zeros(args.num_rollouts*(sys.state_dim+sys.in_dim), args.horizon)

# stop_operator is TRUE if an operator decided to stop the training
stop_operator = False

# ------------ 10. ADMM learning algorithm ------------
while i <= args.max_iter-1 and (not conv_flag) and (not stop_operator):
    # Performance boosting Step
    for epoch in range(1+args.epochs):
        # iterate over all data batches
        for train_data_batch in train_dataloader:
            optimizer.zero_grad()
            # Extracting the data from the batch
            train_data_batch = train_data_batch[0]
            data_4_training = train_data_batch[:, :, :sys.state_dim]
            data_proj_x = train_data_batch[:, :, sys.state_dim : 2*sys.state_dim]
            data_proj_u = train_data_batch[:, :, 2*sys.state_dim: 2*sys.state_dim + sys.in_dim]
            data_lambda_x = train_data_batch[:, :, 2*sys.state_dim + sys.in_dim: 3*sys.state_dim + sys.in_dim]
            data_lambda_u = train_data_batch[:, :, 3*sys.state_dim + sys.in_dim:]

            # simulate over horizon steps
            x_log, _, u_log = sys.rollout(
                controller=ctl, data=data_4_training, train=True,
            )

            # loss of this rollout
            loss_tot_roll, loss_per_roll, loss_constr_roll, loss_obs_roll = loss_fn.forward(x_log, u_log, data_proj_x, data_proj_u, data_lambda_x, data_lambda_u)
            # take a step
            loss_tot_roll.backward()
            optimizer.step()

        LOSS_TOT = np.append(LOSS_TOT, loss_tot_roll.detach().numpy())
        LOSS_PER = np.append(LOSS_PER, loss_per_roll.detach().numpy())
        LOSS_CONSTR = np.append(LOSS_CONSTR, loss_constr_roll.detach().numpy())
        LOSS_OBS = np.append(LOSS_OBS, loss_obs_roll.detach().numpy())

        # print info
        if epoch % args.log_epoch == 0:
                 t = time.time()

    # Retrieving optimal state and input for the training dataset
    with torch.no_grad():
        all_x_logs_t, _, all_u_logs_t = sys.rollout(
            controller=ctl, data=train_data, train=False,
        )

    all_x_logs_t_4opt = all_x_logs_t.permute(0, 2, 1).reshape(-1, all_x_logs_t.permute(0, 2, 1).shape[2])
    all_u_logs_t_4opt = all_u_logs_t.permute(0, 2, 1).reshape(-1, all_u_logs_t.permute(0, 2, 1).shape[2])

    # Projection Step
    obj = cp.Minimize(cp.sum_squares(xp - all_x_logs_t_4opt) + cp.sum_squares(up - all_u_logs_t_4opt))

    proj = cp.Problem(obj, constr)
    proj.solve()
    xp_opt = xp.value
    up_opt = up.value
    # Creating the 1d-tensors
    xp_opt_a = torch.tensor(xp_opt)
    up_opt_a = torch.tensor(up_opt)
    # Creating the 2d-tensors
    xp_opt_t = xp_opt_a.reshape(args.num_rollouts, sys.state_dim, args.horizon).permute(0, 2, 1)
    up_opt_t = up_opt_a.reshape(args.num_rollouts, sys.in_dim, args.horizon).permute(0, 2, 1)

    # Lagrangian multiplier update
    lambda_x += all_x_logs_t - xp_opt_t
    lambda_u += all_u_logs_t - up_opt_t

    # Creating the following for dimensional convenience
    lambda_x_a = lambda_x.permute(0, 2, 1).reshape(-1, lambda_x.permute(0, 2, 1).shape[2])
    lambda_u_a = lambda_u.permute(0, 2, 1).reshape(-1, lambda_u.permute(0, 2, 1).shape[2])

    xu_logs_a = torch.cat((all_x_logs_t_4opt, all_u_logs_t_4opt), dim=0)
    xu_proj_a = torch.cat((xp_opt_a, up_opt_a), dim=0)
    lambda_xu_a = torch.cat((lambda_x_a, lambda_u_a), dim=0)
    # Computing the residuals
    res_pri = torch.norm(xu_proj_a.view(-1) - xu_logs_a.view(-1), p=2)
    res_dual = torch.norm(-loss_fn.rho * (xu_proj_a.view(-1) - xu_logs_t_m.view(-1)), p=2)
    # Saving for plotting
    PRI_RES = np.append(PRI_RES, res_pri.detach().numpy())
    DUAL_RES = np.append(DUAL_RES, res_dual.detach().numpy())

    eps_pri, eps_dual = admm_f._tolerances(xu_logs_a.view(-1), xu_proj_a.view(-1), lambda_xu_a.view(-1))
    eps_pri_t = torch.tensor([eps_pri], dtype=torch.float32).reshape(1, 1)
    eps_dual_t = torch.tensor([eps_dual], dtype=torch.float32).reshape(1, 1)
    # Saving for plotting
    EPS_PRI_RES = np.append(EPS_PRI_RES, eps_pri)
    EPS_DUAL_RES = np.append(EPS_DUAL_RES, eps_dual)

    # Extracting the current learning rate
    for param_group in optimizer.param_groups:
         lr_current = param_group['lr']

    msg_ADMM = 'ADMM iteration %.4f: Complete! Residuals: - primal: %.4f - dual: %.4f' % (i, res_pri.item(),
                                                                                     res_dual.item())
    msg_ADMM += ' ---||--- Tolerances: - primal: %.4f - dual: %.4f' % (eps_pri, eps_dual)
    msg_ADMM += ' ---||--- Penalty: - rho: %.4f ---||--- Learning Rate: - l_r: %.8f' % (loss_fn.rho, lr_current)
    print(msg_ADMM)

    # Saving the penalty parameter if plotting
    RHO = np.append(RHO, torch.tensor([loss_fn.rho], dtype=torch.float32).detach().numpy())


    if res_pri <= eps_pri and res_dual <= eps_dual:
        conv_flag = True
    else:
        # Updating for dual residual computation
        xu_logs_t_m = xu_proj_a
        # Updating penalty parameter
        if args.admm_penalty_update:
            rho_next, lambda_xu_next = admm_f._update_penalty(loss_fn.rho, res_pri, res_dual, lambda_xu_a)
            # Updating the loss function and lagrangian multipliers
            loss_fn.rho = rho_next
            lambda_x_a_next, lambda_u_a_next = torch.split(lambda_xu_next, [lambda_x_a.shape[0], lambda_u_a.shape[0]], dim=0)
            lambda_x = lambda_x_a_next.reshape(args.num_rollouts, sys.state_dim, args.horizon).permute(0, 2, 1)
            lambda_u = lambda_u_a_next.reshape(args.num_rollouts, sys.in_dim, args.horizon).permute(0, 2, 1)
        # Updating dataset
        train_data_admm = TensorDataset(torch.cat(
            (train_data, xp_opt_t.to(torch.float32), up_opt_t.to(torch.float32), lambda_x.to(torch.float32),
             lambda_u.to(torch.float32)), dim=2))
        # batch the data
        train_dataloader = DataLoader(train_data_admm, batch_size=args.batch_size, shuffle=True,
                                      generator=generator)  # Training on minibatch based on batch_size
        # DataLoader for plotting
        train_dataloader_epoch = DataLoader(train_data_admm, batch_size=train_data.shape[0], shuffle=False)

    if (i % 100) == 0:
        print('Saving Parameters')
        file_name = 'PB_current_parameters_with_' + str(plant_state_init) + '_epochs_and_initial_std_' + str(args.cont_init_std)
        saving_dict = {
            'xp_opt_t': xp_opt_t,
            'up_opt_t': up_opt_t,
            'lambda_x': lambda_x,
            'lambda_u': lambda_u,
            'it': i,
            'LOSS_TOT': LOSS_TOT,
            'LOSS_PER': LOSS_PER,
            'LOSS_CONSTR': LOSS_CONSTR,
            'LOSS_OBS': LOSS_OBS,
            'pri_res_plt': PRI_RES,
            'dual_res_plt': DUAL_RES,
            'eps_pri_plt': EPS_PRI_RES,
            'eps_dual_plt': EPS_DUAL_RES,
            'rho_plt': RHO
        }

        file_name_mat = file_name + '.mat'
        savemat(file_name_mat, saving_dict)
        checkpoint = {
            'model_state_dict': copy.deepcopy(ctl.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        file_name_pth = file_name + '.pth'
        torch.save(checkpoint, file_name_pth)

    i += 1

    if lr_current > args.learning_rate_bound:
        scheduler.step()

    if (i > 2000) and ((i % 50) == 0):
        stop_operator = input("Press 1 to stop, otherwise press enter.")


# ------------ 11. Plotting Results ------------

# Plot noise-free trajectories vs time after training
x_log_nf, _, u_log_nf = sys.rollout(ctl, plot_data)  # Noise-free plot
plot_traj_vs_time(t_ext, x_log_nf[0, :, :], u_log_nf[0, :, :],
                  x_max=sys.x_max, x_min=sys.x_min, u_max=sys.u_max, u_min=sys.u_min,
                  plt_opt=plt_set,
                  it_ADMM=it_best, epoch_num=epoch_best, best=True, singl=True, str_title='Noise-free')

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
                       it_ADMM=it_best, epoch_num=epoch_best, best=True, singl=True, str_title='Training')

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



# Losses
# loss_tr_per_plt_epoch = torch.sum(loss_tr_per_plt, 1) / args.num_rollouts
# plt_loss_vs_epochs(i * (args.epochs + 1), loss_tr_per_plt_epoch.unsqueeze(0), y_label=["Value"],
#                    lgd_label=["Performance Loss in Training"], plt_opt=plt_set, singl=True)
# # Mean and std for performance loss in training over epochs
# plt_loss_mean_and_std(i * (args.epochs + 1), loss_tr_per_plt, str_title='Performance Loss in Training',
#                       str_x_axis='$n_e$ [-]', plt_opt=plt_set)

plt.plot(LOSS_TOT)
plt.show()
plt.plot(LOSS_PER)
plt.show()
plt.plot(LOSS_CONSTR)
plt.show()
plt.plot(LOSS_OBS)
plt.show()


# Recovering testing data
with torch.no_grad():
    x_log_test, _, u_log_test = sys.rollout(
        controller=ctl, data=test_data, train=False,
    )
plot_traj_mean_and_std(t_ext, x_log_test, u_log_test, x_max=sys.x_max, x_min=sys.x_min, u_max=sys.u_max, u_min=sys.u_min, plt_opt=plt_set,
                       it_ADMM=it_best, epoch_num=epoch_best, best=True, singl=True, str_title='Testing')

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


print('------------ Checking ADMM metrics ------------')
print("Residuals: - primal: ", res_pri.item(), " - dual: ", res_dual.item())
plt.plot(PRI_RES, label='Primal residual')
plt.plot(EPS_PRI_RES, label='Primal tolerance')
plt.show()
plt.plot(DUAL_RES, label='Dual residual')
plt.plot(EPS_DUAL_RES, label='Dual tolerance')
plt.show()
plt.plot(RHO)
plt.show()

print('------------ Checking Constraints for Training data ------------')
sys.checking_constraints(x_log=x_log_tr, u_log=u_log_tr)
print('------------ Checking Constraints for Testing data ------------')
sys.checking_constraints(x_log=x_log_test, u_log=u_log_test)

input("Press Enter to save...")

# ------------ 12. Saving Results ------------
if args.saving:

    if args.std_noise_plant == 0:
        file_name = 'PB_admm'
    else:
        file_name = 'PB_admm_noise'

    file_name += '_delta_obs' + str(args.delta_obst) + '_eps_abs' + str(args.eps_abs) + 'eps_rel' + str(args.eps_rel)

    saving_dict = {
        'admm': True,
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
        'args_admm_penalty_update':  args.admm_penalty_update,
        'args_eps_abs': args.eps_abs,
        'args_eps_rel': args.eps_rel,
        'args_max_iter': args.max_iter,
        'x_nf': x_log_nf,  # Noise-free state evolution
        'u_nf': u_log_nf,  # Noise-free input evolution
        'x_tr': x_log_tr,  # State evolution with training data
        'u_tr': u_log_tr,  # Input evolution with training data
        'x_test': x_log_test,  # State evolution with testing data
        'u_test': u_log_test,  # Input evolution with testing data
        'LOSS_TOT': LOSS_TOT,  # Total training loss over ADMM iterations
        'LOSS_PER': LOSS_PER,  # Performance training loss over ADMM iterations
        'LOSS_CONSTR': LOSS_CONSTR,  # Constraint training loss over ADMM iterations
        'LOSS_OBS': LOSS_OBS,  # Obstacle training loss over ADMM iterations
        'pri_res_plt': PRI_RES,
        'dual_res_plt': DUAL_RES,
        'eps_pri_plt': EPS_PRI_RES,
        'eps_dual_plt': EPS_DUAL_RES,
        'rho': RHO,
        'it_end': i,
        'it_best': it_best,
        'epoch_best': epoch_best,
        'sys_n_agents': sys.n_agents,
        'sys_x_max': sys.x_max,
        'sys_x_min': sys.x_min,
        'sys_u_max': sys.u_max,
        'sys_u_min': sys.u_min,
        'minibatch_seed': seed_for_batching,
        'std_parameters': args.cont_init_std,
        'scheduler_it': args.scheduler_it,
        'scheduler_gamma': args.scheduler_gamma,
        'alpha_obs': args.alpha_obst,
        'delta_obs': args.delta_obst
    }
    if args.admm_penalty_update:
        file_name += '_online_rho_' + '_mu' + str(args.mu) + '_tau_incr' + str(args.tau_incr) + '_tau_decr' + str(
            args.tau_decr) + '_init' + str(args.rho_init)
        saving_dict['args_rho_init'] = args.rho_init
        saving_dict['args_mu'] = args.mu
        saving_dict['args_tau_incr'] = args.tau_incr
        saving_dict['args_tau_decr'] = args.tau_decr
    else:
        file_name += '_fixed_rho' + str(args.rho)

    if torch.equal(train_data, valid_data):
        file_name += 'SAME'

    # file_name += '.pkl'
    # saving_data(saving_dict, file_name)

    file_name += '.mat'
    savemat(file_name, saving_dict)
