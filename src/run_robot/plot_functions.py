import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# plt.rcParams['text.usetex'] = True

def plot_trajectories(
    x, save=False, filename='', T=100, obst=False,
    x_max = None, x_min = None,
    dots=False, circles=False, radius_robot=1, f=5,
    obstacle_centers=None, obstacle_radius=None,
    plt_opt = None, it_ADMM = None, epoch_num = None,
    best = False, col=None, str_title = None
):

    if plt_opt is not None:
        plt.rcParams['figure.figsize'] = (plt_opt.plot_dim_x, plt_opt.plot_dim_y)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = plt_opt.fontname
        plt.rcParams['axes.titlesize'] = plt_opt.title_font_dim
        plt.rcParams['axes.labelsize'] = plt_opt.font_dim
        plt.rcParams['lines.linewidth'] = plt_opt.thick
        plt.rcParams['axes.grid'] = plt_opt.grid

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles
    if col is None:
        col = ['tab:blue', 'tab:orange']

    ax.plot(
        x[:T+1,0].detach(), x[:T+1,1].detach(),
        color=col[0], linewidth=1
    )
    ax.plot(
        x[T:,0].detach(), x[T:,1].detach(),
        color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
        )
    ax.plot(
        x[0,0].detach(), x[0,1].detach(),
        color=col[0], marker='8'
    )

    # Plotting constraints
    if x_max is not None:
        if x_max[0] is not None:
            ax.axvline(x_max[0], color='k', linestyle=':')
        if x_max[1] is not None:
            ax.axhline(x_max[1], color='k', linestyle=':')

    if x_min is not None:
        if x_min[0] is not None:
            ax.axvline(x_min[0], color='k', linestyle=':')
        if x_min[1] is not None:
            ax.axhline(x_min[1], color='k', linestyle=':')

    if dots:
        for j in range(T):
            ax.plot(
                x[j, 0].detach(), x[j, 1].detach(),
                color=col[0], marker='o'
            )
    if circles:
        r = radius_robot
        circle = plt.Circle((x[T-1, 0].detach(), x[T-1, 1].detach()),
                            r, color=col[0], alpha=0.5, zorder=10
                           )
        ax.add_patch(circle)
    if obstacle_centers is not None:
        r = obstacle_radius[0,0]
        circle = plt.Circle((obstacle_centers[0,0],obstacle_centers[0,1]),
                            r, color='k', alpha=0.1, zorder=10
                            )
        ax.add_patch(circle)

    ax.set_xlabel(r'$p_x(t)$ [m]')
    ax.set_ylabel(r'$p_y(t)$ [m]')


    if it_ADMM is not None or epoch_num is not None:
        admm_string = 'ADMM iteration: ' + str(it_ADMM) + ', ' if it_ADMM is not None else ''
        if best:
            plt.suptitle('Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                         fontsize=plt_opt.title_font_dim, weight='bold')
        else:
            plt.suptitle(admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim)
    else:
        plt.suptitle('Optimal Input and State - '+ str_title)

    if save:
        plt.savefig(filename + '.pdf', format='pdf')
    plt.show()


def plot_trajectories_mean_and_std(
    x, save=False, filename='', T=100, obst=False,
    x_max = None, x_min = None,
    dots=False, circles=False, radius_robot=1, f=5,
    obstacle_centers=None, obstacle_radius=None,
    plt_opt = None, it_ADMM = None, epoch_num = None,
    best = False, col=None, str_title = None, ell = 0
):

    if plt_opt is not None:
        plt.rcParams['figure.figsize'] = (plt_opt.plot_dim_x, plt_opt.plot_dim_y)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = plt_opt.fontname
        plt.rcParams['axes.titlesize'] = plt_opt.title_font_dim
        plt.rcParams['axes.labelsize'] = plt_opt.font_dim
        plt.rcParams['lines.linewidth'] = plt_opt.thick
        plt.rcParams['axes.grid'] = plt_opt.grid

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles
    if col is None:
        col = ['tab:blue', 'tab:orange']

    # Mean
    x_mean = torch.mean(x, dim=0)
    # Std
    x_std = torch.std(x, dim=0)

    ax.scatter(
        x_mean[:T+1,0].detach(), x_mean[:T+1,1].detach(),
        color=col[0], linewidth=1
    )
    if ell:
        sigma = 2  # Cambia questo per 1, 2, 3, ecc.
        for t in (range(T)):
            ellipse = Ellipse((x_mean[t, 0].item(), x_mean[t, 1].item()), width=2 * sigma * x_std[t, 0].item(), height=2 * sigma * x_std[t, 1].item(),
                              alpha=0.5, facecolor=col[0], linewidth=0)
            ax.add_patch(ellipse)
    else:
        ax.errorbar(x_mean[:T+1, 0].detach(), x_mean[:T+1, 1].detach(),
                    xerr = x_std[:T + 1, 0].detach(), yerr = x_std[:T + 1, 1].detach(),
                    fmt='o', color=col[0])
    ax.plot( # Final point
        x_mean[T:,0].detach(), x_mean[T:,1].detach(),
        color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
        )
    ax.plot( # Initial point
        x_mean[0,0].detach(), x_mean[0,1].detach(),
        color=col[0], marker='8'
    )

    # Plotting constraints
    if x_max is not None:
        if x_max[0] is not None:
            ax.axvline(x_max[0], color='k', linestyle=':')
        if x_max[1] is not None:
            ax.axhline(x_max[1], color='k', linestyle=':')

    if x_min is not None:
        if x_min[0] is not None:
            ax.axvline(x_min[0], color='k', linestyle=':')
        if x_min[1] is not None:
            ax.axhline(x_min[1], color='k', linestyle=':')

    if dots:
        for j in range(T):
            ax.plot(
                x[j, 0].detach(), x[j, 1].detach(),
                color=col[0], marker='o'
            )
    if circles:
        r = radius_robot
        circle = plt.Circle((x[T-1, 0].detach(), x[T-1, 1].detach()),
                            r, color=col[0], alpha=0.5, zorder=10
                           )
        ax.add_patch(circle)
    if obstacle_centers is not None:
        r = obstacle_radius[0,0]
        circle = plt.Circle((obstacle_centers[0,0],obstacle_centers[0,1]),
                            r, color='k', alpha=0.1, zorder=10
                            )
        ax.add_patch(circle)

    ax.set_xlabel(r'$p_x(t)$ [m]')
    ax.set_ylabel(r'$p_y(t)$ [m]')


    if it_ADMM is not None or epoch_num is not None:
        admm_string = 'ADMM iteration: ' + str(it_ADMM) + ', ' if it_ADMM is not None else ''
        if best:
            plt.suptitle('Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                         fontsize=plt_opt.title_font_dim, weight='bold')
        else:
            plt.suptitle(admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim)
    else:
        plt.suptitle('Optimal Input and State - '+ str_title)

    if save:
        plt.savefig(filename + '.pdf', format='pdf')
    plt.show()


def plot_traj_vs_time(t_end, x, u=None, save=False, filename='', u_bar=None, x_bar=None,
                      x_max = None, x_min = None, u_max = None, u_min = None, plt_opt = None, it_ADMM = None,
                      epoch_num = None, best = False, singl = True, col=None, comp = False, label = None, str_title = None):

    # Initializing the color
    if col is None:
        col = ['b', 'c']

    if plt_opt is not None:
        plt.rcParams['figure.figsize'] = (plt_opt.plot_dim_x, plt_opt.plot_dim_y)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = plt_opt.fontname
        plt.rcParams['axes.titlesize'] = plt_opt.title_font_dim
        plt.rcParams['axes.labelsize'] = plt_opt.font_dim
        plt.rcParams['lines.linewidth'] = plt_opt.thick
        plt.rcParams['axes.grid'] = plt_opt.grid

    t = torch.linspace(0,t_end-1, t_end)
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    if u is not None and u.dim() == 2:
        u = u.unsqueeze(-1)
    assert x.dim() == 3
    if u is not None:
        p = 3
    else:
        p = 2
    if u_bar is not None:
        u_bar = float(u_bar)

    if not singl:
        # Plotting the input
        plt.subplot(p, 1, 1)
        plt.subplots_adjust(hspace=1/p)
    else:
        plt.figure(1)

    for i in range(u.shape[1]): # When the input given to the function is for the different datapoints, we can plot them with the for cycle w.r.t. u.shape[0]
        plt.plot(t, u[:, i, 0].detach(), color = col[i], label = label)
    if u_max is not None:
        for ind, constr in enumerate(u_max):
            plt.axhline(y=constr, color = col[ind], linestyle=':')
    if u_min is not None:
        for ind, constr in enumerate(u_min):
            plt.axhline(y=constr, color = col[ind], linestyle=':')

    if u_bar is not None:
        plt.plot(t, (-u_bar * torch.ones_like(t)), 'k--', linewidth=0.1)
    if singl:
        plt.title('Optimal Input' + ', ' + str_title)

    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$u(t)$ [N]')
    plt.xlim(t[0].item(), t[-1].item())
    if label is not None:
        plt.legend()


    if not singl:
        # Plotting the state
        plt.subplot(p ,1, 2)
    else:
        plt.figure(2)
    for i in range(x.shape[1])[:2]: # When the state given to the function is for the different datapoints, we can plot them with the for cycle w.r.t. u.shape[0]
        plt.plot(t, x[:, i,0].detach(), color = col[i], label = label)
    if x_max is not None:
        for ind, constr in enumerate(x_max[:2]):
            plt.axhline(y=constr, color=col[ind], linestyle=':')

    if x_min is not None:
        for ind, constr in enumerate(x_min[:2]):
            plt.axhline(y=constr, color=col[ind], linestyle=':')

    if x_bar is not None:
        plt.plot(t, (x_bar * torch.ones_like(t)), 'k--', linewidth=0.1)
        plt.plot(t, torch.zeros_like(t), 'k--', linewidth=0.1)
    if singl:
        plt.title('Position Evolution'  + ', ' + str_title)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$p(t)$ [m]')
    plt.xlim(t[0].item(), t[-1].item())
    if label is not None:
        plt.legend()

    if not singl:
        # Plotting the state
        plt.subplot(p ,1, 3)
    else:
        plt.figure(3)
    for i in range(x.shape[1])[2:]: # When the state given to the function is for the different datapoints, we can plot them with the for cycle w.r.t. u.shape[0]
        plt.plot(t, x[:,i,0].detach(), color = col[i-2], label = label)
    if x_max is not None:
        for ind, constr in enumerate(x_max[2:]):
            plt.axhline(y=constr, color=col[ind-2], linestyle=':')

    if x_min is not None:
        for ind, constr in enumerate(x_min[2:]):
            plt.axhline(y=constr, color=col[ind-2], linestyle=':')

    if x_bar is not None:
        plt.plot(t, (x_bar * torch.ones_like(t)), 'k--', linewidth=0.1)
        plt.plot(t, torch.zeros_like(t), 'k--', linewidth=0.1)
    if singl:
        plt.title('Speed Evolution'  + ', ' + str_title)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$q(t)$ [m]')
    plt.xlim(t[0].item(), t[-1].item())
    if label is not None:
        plt.legend()

    if it_ADMM is not None or epoch_num is not None:
        admm_string = 'ADMM iteration: ' + str(it_ADMM) + ', ' if it_ADMM is not None else ''
        if not singl:
            if best:
                plt.suptitle('Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
            else:
                plt.suptitle(admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim)
        else:
            if best:
                plt.figure(1)
                plt.suptitle('Input - Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
                plt.figure(2)
                plt.suptitle('Position Evolution - Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
                plt.figure(3)
                plt.suptitle('Speed Evolution - Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
            else:
                plt.figure(1)
                plt.suptitle('Input - ' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim)
                plt.figure(2)
                plt.suptitle('Position Evolution - ' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim)
                plt.figure(3)
                plt.suptitle('Speed Evolution - Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
    else:
        plt.suptitle('Optimal Input and State - '+ str_title)


    # Saving the plots
    if not comp:
        plt.show()
        # plt.close()
        if save:
           plt.savefig('' + filename + '.pdf', format='pdf')


def plot_traj_mean_and_std(t_end, x, u=None, save=False, filename='', u_bar=None, x_bar=None,
                      x_max = None, x_min = None, u_max = None, u_min = None, plt_opt = None, it_ADMM = None,
                      epoch_num = None, best = False, singl = True, col = None, comp = False, label = None, str_title = None):
    if plt_opt is not None:
        plt.rcParams['figure.figsize'] = (plt_opt.plot_dim_x, plt_opt.plot_dim_y)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = plt_opt.fontname
        plt.rcParams['axes.titlesize'] = plt_opt.title_font_dim
        plt.rcParams['axes.labelsize'] = plt_opt.font_dim
        plt.rcParams['lines.linewidth'] = plt_opt.thick
        plt.rcParams['axes.grid'] = plt_opt.grid

    # Initializing the color
    if col is None:
        col = ['b', 'c']

    t = torch.linspace(0,t_end-1, t_end)
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    if u is not None and u.dim() == 2:
        u = u.unsqueeze(-1)
    assert x.dim() == 3
    if u is not None:
        p = 3
    else:
        p = 2
    if u_bar is not None:
        u_bar = float(u_bar)

    # Mean
    x_mean = torch.mean(x, dim=0)
    u_mean = torch.mean(u, dim=0)
    # Std
    x_std = torch.std(x, dim=0)
    u_std = torch.std(u, dim=0)

    if not singl:
        # Plotting the input
        plt.subplot(p, 1, 1)
        plt.subplots_adjust(hspace=1/p)
    else:
        plt.figure(1)
    for i in range(u_mean.shape[-1]):
        plt.plot(t, u_mean[:, i].detach(), color = col[i], label = label)
        plt.fill_between(t, u_mean[:, i].detach() - u_std[:, i].detach() , u_mean[:, i].detach() + u_std[:, i].detach(), color=col[i],
                         alpha=0.5)

    if u_max is not None:
        for ind, constr in enumerate(u_max):
            plt.axhline(y=constr, color = col[ind], linestyle=':')
    if u_min is not None:
        for ind, constr in enumerate(u_min):
            plt.axhline(y=constr, color = col[ind], linestyle=':')

    if u_bar is not None:
        plt.plot(t, (-u_bar * torch.ones_like(t)), 'k--', linewidth=0.1)
    if singl:
        plt.title('Optimal Input - Mean and Std, ' + ', ' + str_title)

    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$u(t)$ [N]')
    plt.xlim(t[0].item(), t[-1].item())
    if  label is not None:
        plt.legend()


    if not singl:
        # Plotting the state
        plt.subplot(p ,1, 2)
    else:
        plt.figure(2)
    for i in range(x_mean.shape[-1])[:2]:
        plt.plot(t, x_mean[:, i].detach(), color=col[i], label=label)
        plt.fill_between(t, x_mean[:, i].detach() - x_std[:, i].detach(),
                         x_mean[:, i].detach() + x_std[:, i].detach(), color=col[i],
                         alpha=0.5)
    if x_max is not None:
        for ind, constr in enumerate(x_max[:2]):
            plt.axhline(y=constr, color=col[ind], linestyle=':')

    if x_min is not None:
        for ind, constr in enumerate(x_min[:2]):
            plt.axhline(y=constr, color=col[ind], linestyle=':')

    if x_bar is not None:
        plt.plot(t, (x_bar * torch.ones_like(t)), 'k--', linewidth=0.1)
        plt.plot(t, torch.zeros_like(t), 'k--', linewidth=0.1)
    if singl:
        plt.title('Position Evolution - Mean and Std, ' + ', ' + str_title)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$p(t)$ [m]')
    plt.xlim(t[0].item(), t[-1].item())



    if not singl:
        # Plotting the state
        plt.subplot(p, 1, 3)
    else:
        plt.figure(3)
    for i in range(x_mean.shape[-1])[2:]:
        plt.plot(t, x_mean[:, i].detach(), color=col[i - 2], label=label)
        plt.fill_between(t, x_mean[:, i].detach() - x_std[:, i].detach(),
                         x_mean[:, i].detach() + x_std[:, i].detach(), color=col[i - 2],
                         alpha=0.5)
    if x_max is not None:
        for ind, constr in enumerate(x_max[:2]):
            plt.axhline(y=constr, color=col[ind - 2], linestyle=':')

    if x_min is not None:
        for ind, constr in enumerate(x_min[:2]):
            plt.axhline(y=constr, color=col[ind - 2], linestyle=':')

    if x_bar is not None:
        plt.plot(t, (x_bar * torch.ones_like(t)), 'k--', linewidth=0.1)
        plt.plot(t, torch.zeros_like(t), 'k--', linewidth=0.1)
    if singl:
        plt.title('Speed Evolution - Mean and Std, ' + ', ' + str_title)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$q(t)$ [m]')
    plt.xlim(t[0].item(), t[-1].item())

    if it_ADMM is not None or epoch_num is not None:
        admm_string = 'ADMM iteration: ' + str(it_ADMM) + ', ' if it_ADMM is not None else ''
        if not singl:
            if best:
                plt.suptitle('Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
            else:
                plt.suptitle(admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim)
        else:
            if best:
                plt.figure(1)
                plt.suptitle('Input - Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
                plt.figure(2)
                plt.suptitle('Position Evolution - Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
                plt.figure(3)
                plt.suptitle('Speed Evolution - Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
            else:
                plt.figure(1)
                plt.suptitle('Input - ' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim)
                plt.figure(2)
                plt.suptitle('Position Evolution - ' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim)
                plt.figure(3)
                plt.suptitle('Speed Evolution - Best result @' + admm_string + 'epoch: ' + str(epoch_num) + ', ' + str_title,
                             fontsize=plt_opt.title_font_dim, weight='bold')
    else:
        plt.suptitle('Optimal Input and State - Mean and Std, ' + ', ' + str_title)

    if label is not None:
        plt.legend()
    # Saving the plots
    if not comp:
        plt.show()
        # plt.close()
        if save:
           plt.savefig('' + filename + '.pdf', format='pdf')

def plt_loss_vs_epochs(num_epochs, losses, y_label, lgd_label = None, save=False, filename='', plt_opt = None, singl = True, col = None, comp = False):
    # Data in quantities are stored as (type_quantity, values, 1)
    if plt_opt is not None:
        plt.rcParams['figure.figsize'] = (plt_opt.plot_dim_x, plt_opt.plot_dim_y)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = plt_opt.fontname
        plt.rcParams['axes.titlesize'] = plt_opt.title_font_dim
        plt.rcParams['axes.labelsize'] = plt_opt.font_dim
        plt.rcParams['lines.linewidth'] = plt_opt.thick
        plt.rcParams['axes.grid'] = plt_opt.grid

    n_e = np.linspace(1,num_epochs, num_epochs)

    if col is None:
        cmap = mpl.cm.get_cmap('Paired')
        col = [cmap(i) for i in range(len(losses))]
    else:
        col = np.full((len(losses),), col)

    # together = np.array([[1 , 2]])
    together = np.array([ ])

    i = 0 # Plot index
    j = 1 # Subplot index/figure index
    # Plotting the quantities
    while i < len(losses):
        if not singl:
           plt.subplot(len(losses) - len(together), 1, j)
        else:
            plt.figure(j)
        plt.plot(n_e, losses[i], label=lgd_label[i], color = col[i])
        plt.xlabel('$n_e$ [-]')
        plt.ylabel(y_label[i])
        plt.xlim(n_e[0].item(), n_e[-1].item())
        if i in together:
            plt.plot(n_e, losses[i+1], '--', label=lgd_label[i+1], color = col[i+1])
            i += 2
        else:
            i += 1
        plt.legend()
        j += 1

    if not singl:
        plt.subplots_adjust(hspace = 1/(len(losses) - len(together) - 1.5))

    # Saving the plots
    if not comp:
        plt.show()
        # plt.close()
        if save:
           plt.savefig('' + filename + '.pdf', format='pdf')

def plt_loss_mean_and_std(qty, loss, str_title = None, str_x_axis = '[-]', lgd_label=None, save=False, filename='', plt_opt=None, col=None, comp=False):
    if plt_opt is not None:
        plt.rcParams['figure.figsize'] = (plt_opt.plot_dim_x, plt_opt.plot_dim_y)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = plt_opt.fontname
        plt.rcParams['axes.titlesize'] = plt_opt.title_font_dim
        plt.rcParams['axes.labelsize'] = plt_opt.font_dim
        plt.rcParams['lines.linewidth'] = plt_opt.thick
        plt.rcParams['axes.grid'] = plt_opt.grid

        loss = loss.unsqueeze(0)
        losses_mean = torch.mean(loss, dim=2)
        losses_std = torch.std(loss, dim=2)
        # qty could be the number of epochs or iteration
        qty = np.linspace(1, qty, qty)

        plt.figure(1)
        plt.plot(qty, losses_mean[0, : ].detach(), color=col, label=lgd_label)
        plt.fill_between(qty, losses_mean[0, : ].squeeze().detach() - losses_std[0, : ].squeeze().detach(),
                             losses_mean[0, : ].squeeze().detach() + losses_std[0, : ].squeeze().detach(), color=col,
                             alpha=0.5)

        plt.title(str_title + ' - Mean and Std over the dataset')

        plt.xlabel(str_x_axis)
        plt.ylabel(r'Value')
        plt.xlim(qty[0].item(), qty[-1].item())
        if lgd_label is not None:
            plt.legend()


        if lgd_label is not None:
            plt.legend()
        # Saving the plots
        if not comp:
            plt.show()
            # plt.close()
            if save:
                plt.savefig('' + filename + '.pdf', format='pdf')


def plt_quantity_vs_it(it_end, quantities, y_label, lgd_label, save=False, filename='', plt_opt = None, singl = True, col = None, comp = False):
    # Data in quantities are stored as (type_quantity, values, 1)
    if plt_opt is not None:
        plt.rcParams['figure.figsize'] = (plt_opt.plot_dim_x, plt_opt.plot_dim_y)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = plt_opt.fontname
        plt.rcParams['axes.titlesize'] = plt_opt.title_font_dim
        plt.rcParams['axes.labelsize'] = plt_opt.font_dim - 2
        plt.rcParams['lines.linewidth'] = plt_opt.thick
        plt.rcParams['axes.grid'] = plt_opt.grid

    it = np.linspace(1,it_end, it_end)

    if col is None:
        cmap = mpl.cm.get_cmap('Paired')
        col = [cmap(i) for i in range(len(quantities))]
    else:
        col = np.full((len(quantities),), col)

    together = np.array([[0 , 1], [2 , 3]])

    i = 0 # Plot index
    j = 1 # Subplot index/figure index
    # Plotting the quantities
    while i < len(quantities):
        if not singl:
           plt.subplot(len(quantities) - len(together), 1, j)
        else:
            plt.figure(j)
        plt.plot(it, quantities[i], label=lgd_label[i], color = col[i])
        plt.xlabel('iter [-]')
        plt.ylabel(y_label[i])
        plt.xlim(it[0].item(), it[-1].item())
        if i in together:
            plt.plot(it, quantities[i+1], '--', label=lgd_label[i+1], color = col[i+1])
            i += 2
        else:
            i += 1
        plt.legend()
        j += 1

    if not singl:
        plt.subplots_adjust(hspace = 1/(len(quantities) - len(together) - 1.5))

    # Saving the plots
    if not comp:
        plt.show()
        # plt.close()
        if save:
           plt.savefig('' + filename + '.pdf', format='pdf')