import torch
import math
from assistive_functions import to_tensor


class RobotsLoss():
    def __init__(
        self, Q, R,
        alpha_barrier_x = None, alpha_barrier_u = None,
        sys = None,
        alpha_obst=None,
        delta_obst=None,
        radius_robot=0.25,
        obstacle_centers=None, obstacle_radius=None
    ):


        self.Q, self.R = to_tensor(Q), to_tensor(R)
        self.alpha_BF_x = alpha_barrier_x
        self.alpha_BF_u = alpha_barrier_u

        self.sys = sys

        self.alpha_obst, self.radius_robot = alpha_obst, radius_robot
        self.delta_obst = delta_obst

        # Obstacle Avoidance
        if obstacle_centers is None:
            self.obstacle_centers = torch.tensor([[1., 0.5]])
        else:
            self.obstacle_centers = obstacle_centers
        self.n_obstacles = self.obstacle_centers.shape[0]
        if obstacle_radius is None:
            self.obstacle_radius = torch.tensor([[0.5]])
        else:
            self.obstacle_radius = obstacle_radius
        assert self.n_obstacles == self.obstacle_radius.shape[0]

    def forward(self, xs, us, for_plt = False):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)

        Return:
            - loss of shape (1, 1).
        """
        # batch
        x_batch = xs.reshape(*xs.shape, 1)
        u_batch = us.reshape(*us.shape, 1)

        # Barrier function loss
        loss_barrier_u_max, loss_barrier_u_min = self.loss_barrier_function_u(u_batch)
        loss_barrier_x_max, loss_barrier_x_min = self.loss_barrier_function_x(x_batch, u_batch, self.sys)
        loss_constr = loss_barrier_x_max + loss_barrier_x_min + loss_barrier_u_max + loss_barrier_u_min
        if not for_plt and (self.alpha_BF_x > 0 or self.alpha_BF_u > 0):
            loss_constr = torch.sum(loss_constr, 0) / xs.shape[0]  # shape = (1, 1)


        # Performance Loss
        loss_x,loss_u = self.loss_quadratic(x_batch, u_batch)
        loss_per = loss_x + loss_u
        if not for_plt:
            loss_per = torch.sum(loss_per, 0)/xs.shape[0]                   # shape = (1, 1)

       # obstacle avoidance loss
        if self.alpha_obst is None:
            loss_obst = 0
        else:
            loss_obst = self.alpha_obst * self.f_loss_obst(x_batch) # shape = (S, 1, 1)
            if not for_plt:
                loss_obst = torch.sum(loss_obst, 0) / xs.shape[0]

        # sum up all losses (average over the samples)
        loss_val = loss_per + loss_constr + loss_obst                # shape = (1, 1)

        return loss_val, loss_per, loss_constr, loss_obst

    def loss_quadratic(self, x_batch, u_batch):
        # loss states = sum_{t=1}^T (x_t-xbar)^T Q (x_t-xbar)
        x_batch_centered = x_batch  # - self.xbar TODO see how to menage
        xTQx = torch.matmul(
            torch.matmul(x_batch_centered.transpose(-1, -2), self.Q),
            x_batch_centered
        )  # shape = (S, T, 1, 1)
        loss_x = torch.sum(xTQx, 1)
        # loss control actions = sum_{t=1}^T u_t^T R u_t
        uTRu = torch.matmul(
            torch.matmul(u_batch.transpose(-1, -2), self.R),
            u_batch
        )  # shape = (S, T, 1, 1)
        loss_u = torch.sum(uTRu, 1)
        return loss_x, loss_u

    def f_loss_obst(self, x_batch):
        """
        Obstacle avoidance loss.
        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.
        Return:
            - collision avoidance loss of shape (1, 1).
        """

        min_sec_dist = self.delta_obst * (self.radius_robot + self.obstacle_radius[0,0])
        # compute pairwise distances
        distance_sq = self.get_pairwise_distance_sq(x_batch)              # shape = (S, T, n_agents, n_agents)
        # compute and sum up loss when two agents are too close
        loss_obs = (1/(distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2))).sum((-1, -2))/2        # shape = (S, T)

        loss_obs = loss_obs.sum(1)
        # reshape to S,1,1
        loss_obs = loss_obs.reshape(-1,1,1)
        return loss_obs

    def get_pairwise_distance_sq(self, x_batch):
        """
        Squared distance between run_robot and obstacle.
        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.
        Return:
            - matrix of shape (S, T, 1, n_obstacles) of squared pairwise distances.
        """
        # collision avoidance:
        x_robot = x_batch[:, :, 0:1, :]  # shape = (S, T, 1, 1)
        y_robot = x_batch[:, :, 1:2, :]  # shape = (S, T, 1, 1)
        deltaqx = x_robot.repeat(1, 1, 1, self.n_obstacles) - self.obstacle_centers[:,0].repeat(1, x_robot.shape[1], 1, 1)   # shape = (S, T, 1, n_obstacles)
        deltaqy = y_robot.repeat(1, 1, 1, self.n_obstacles) - self.obstacle_centers[:,1].repeat(1, x_robot.shape[1], 1, 1)   # shape = (S, T, 1, n_obstacles)
        distance_sq = deltaqx ** 2 + deltaqy ** 2             # shape = (S, T, 1, n_obstacles)
        return distance_sq

    def loss_barrier_function_u(self, u_batch): # Barrier function for the input
        loss_u_max = 0
        loss_u_min = 0

        if self.alpha_BF_u > 0:
            u_max = self.sys.u_max
            u_min = self.sys.u_min

            for i, constr in enumerate(u_max):
                if not math.isinf(u_max[i]):
                    loss_u_max += self.alpha_BF_u * torch.sum(torch.relu(u_batch[:, :, i] - u_max[i]), 1)

            for i, constr in enumerate(u_min):
                if not math.isinf(u_min[i]):
                    loss_u_min += self.alpha_BF_u * torch.sum(torch.relu(-u_batch[:, :, i] - u_min[i]), 1)

        return loss_u_max, loss_u_min

    def loss_barrier_function_x(self, x_batch, u_batch, sys): # Barrier function for the state
        loss_barrier_x_max = 0
        loss_barrier_x_min = 0
        gamma = 0.2
        if self.alpha_BF_x > 0:

            x_max = self.sys.x_max
            x_min = self.sys.x_min

            x_next = sys.forward(t=0, x=x_batch, u=u_batch, w=torch.zeros_like(x_batch))
            x_next = x_next.reshape(x_batch.size())

            for i, constr in enumerate(x_max):
                if not math.isinf(x_max[i]):
                    h_next_max = x_max[i] - x_next[:, :, i]
                    h_max = x_max[i] - x_batch[:, :, i]
                    loss_barrier_x_max += (self.alpha_BF_x * torch.relu((1 - gamma) * h_max - h_next_max))

            for i, constr in enumerate(x_min):
                if not math.isinf(x_min[i]):
                    h_next_min = x_next[:, :, i] - x_min[i]
                    h_min = x_batch[:, :, i] - x_min[i]
                    loss_barrier_x_min += (self.alpha_BF_x * torch.relu((1 - gamma) * h_min - h_next_min))


            # average over the time horizon T. shape = (S, T, 1)
            loss_barrier_x_max = torch.sum(loss_barrier_x_max, 1).unsqueeze(2)
            loss_barrier_x_min = torch.sum(loss_barrier_x_min, 1).unsqueeze(2)

        return loss_barrier_x_max, loss_barrier_x_min