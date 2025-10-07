import torch
import math
import cvxpy as cp



class AdmmFun:
    def __init__(self, dim_proj, dim_dec, eps_abs = 1e-4 , eps_rel = 1e-3, mu = 0, tau_incr = 2, tau_decr = 2 ):
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.dim_proj = dim_proj # dimensions of the variables to be projected
        self.dim_dec = dim_dec # dimensions for all the decision variables of the original problem
        # Update of the penalty parameter
        self.mu = mu
        self.tau_incr = tau_incr
        self.tau_decr = tau_decr


    def _tolerances(self, var, var_proj, l_mult):
        norm_var = torch.norm(var, p=2) # original variables
        norm_var_proj = torch.norm(var_proj, p=2) # projected variables
        norm_l_mult = torch.norm(l_mult, p=2) # lagrangian multipliers variables
        eps_pri = math.sqrt(self.dim_proj)*self.eps_abs + self.eps_rel*max(norm_var.item(), norm_var_proj.item())
        eps_dual = math.sqrt(self.dim_dec)*self.eps_abs + self.eps_rel*norm_l_mult.item()

        return eps_pri, eps_dual

    def _update_penalty(self, rho, res_pri, res_dual, lagrangian_mult):
        if res_pri > self.mu*res_dual:
            rho_next = rho*self.tau_incr
            lagrangian_mult_update = lagrangian_mult/self.tau_incr
        elif res_dual >  self.mu*res_pri:
            rho_next = rho/self.tau_decr
            lagrangian_mult_update = lagrangian_mult*self.tau_decr
        else:
            rho_next = rho
            lagrangian_mult_update = lagrangian_mult

        return rho_next, lagrangian_mult_update

    def _proj(self, x_log_t, u_log_t, xp: cp.Variable, up: cp.Variable, constr: list, lambda_x_prev, lambda_u_prev, num_rollouts, args, sys):

        all_x_logs_t_4opt = x_log_t.permute(0, 2, 1).reshape(-1, x_log_t.permute(0, 2, 1).shape[2])
        all_u_logs_t_4opt = u_log_t.permute(0, 2, 1).reshape(-1, u_log_t.permute(0, 2, 1).shape[2])

        # (Initialization) Projection Step
        obj = cp.Minimize(cp.sum_squares(xp - all_x_logs_t_4opt) + cp.sum_squares(up - all_u_logs_t_4opt))

        proj = cp.Problem(obj, constr)
        proj.solve()
        xp_opt = xp.value
        up_opt = up.value
        # Extracting the (initial) projected variables
        xp_opt_a = torch.tensor(xp_opt)
        up_opt_a = torch.tensor(up_opt)
        # Modifying the shape
        xp_opt_t = xp_opt_a.reshape(num_rollouts, sys.state_dim, args.horizon).permute(0, 2, 1)
        up_opt_t = up_opt_a.reshape(num_rollouts, sys.in_dim, args.horizon).permute(0, 2, 1)
        # (initial) Lagrangian multiplier
        lambda_x = lambda_x_prev + x_log_t - xp_opt_t
        lambda_u = lambda_u_prev + u_log_t - up_opt_t

        return xp_opt_t, up_opt_t, lambda_x, lambda_u


