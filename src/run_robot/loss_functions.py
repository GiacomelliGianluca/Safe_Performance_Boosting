import torch
from assistive_functions import to_tensor


class RobotsLoss():
    def __init__(
        self, Q, alpha_u=1,
        alpha_obst=None,
        radius_robot=0.25,
        obstacle_centers=None, obstacle_radius=None
    ):
        self.Q, self.R = to_tensor(Q), to_tensor(alpha_u)
        self.alpha_obst, self.radius_robot = alpha_obst, radius_robot
        # define obstacles
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

    def forward(self, xs, us):
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
        # loss states = 1/T sum_{t=1}^T (x_t)^T Q (x_t)
        xTQx = torch.matmul(
            torch.matmul(x_batch.transpose(-1, -2), self.Q),
            x_batch
        )   # shape = (S, T, 1, 1)
        loss_x = torch.sum(xTQx, 1) / x_batch.shape[1]    # average over the time horizon. shape = (S, 1, 1)
        # loss control actions = 1/T sum_{t=1}^T u_t^T R u_t
        uTRu = self.R * torch.matmul(
            u_batch.transpose(-1, -2),
            u_batch
        )   # shape = (S, T, 1, 1)
        loss_u = torch.sum(uTRu, 1) / x_batch.shape[1]    # average over the time horizon. shape = (S, 1, 1)
        # obstacle avoidance loss
        if self.alpha_obst is None:
            loss_obst = 0
        else:
            loss_obst = self.alpha_obst * self.f_loss_obst(x_batch) # shape = (S, 1, 1)
        # sum up all losses
        loss_val = loss_x + loss_u + loss_obst            # shape = (S, 1, 1)
        # average over the samples
        loss_val = torch.sum(loss_val, 0)/xs.shape[0]       # shape = (1, 1)
        return loss_val

    def f_loss_obst(self, x_batch):
        """
        Obstacle avoidance loss.
        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.
        Return:
            - collision avoidance loss of shape (1, 1).
        """
        min_sec_dist = 1.2 * (self.radius_robot + self.obstacle_radius[0,0])
        # compute pairwise distances
        distance_sq = self.get_pairwise_distance_sq(x_batch)              # shape = (S, T, n_agents, n_agents)
        # compute and sum up loss when two agents are too close
        loss_obs = (1/(distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2))).sum((-1, -2))/2        # shape = (S, T)
        # average over time steps
        loss_obs = loss_obs.sum(1)/loss_obs.shape[1]
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
