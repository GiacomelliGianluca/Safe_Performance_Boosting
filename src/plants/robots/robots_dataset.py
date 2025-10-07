import torch
from plants.custom_dataset import CustomDataset


class RobotsDataset(CustomDataset):
    def __init__(self, random_seed, horizon, std_ini=0.2, std_noise=0, n_agents=1):
        # experiment and file names
        exp_name = 'run_robot'
        if std_noise == 0:
            file_name = 'data_T' + str(horizon) + '_stdini' + str(std_ini) + '_agents' + str(n_agents) + '_RS' + str(
                random_seed) + '.pkl'
        else:
            file_name = 'data_T' + str(horizon) + '_stdini' + str(std_ini) + '_stdnoise' + str(std_noise) + '_agents' + str(n_agents) + '_RS' + str(
                random_seed) + '.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini  #Std for initial condition
        self.std_noise = std_noise  #Std for measurement noise
        self.n_agents = n_agents

        # initial state
        self.x0 = torch.tensor([2, 2, 0, 0])
        self.xbar = torch.zeros(4)

    # ---- data generation ----
    def _generate_data(self, num_samples):
        state_dim = 4 * self.n_agents
        data = torch.zeros(num_samples, self.horizon, state_dim)
        for rollout_num in range(num_samples):
            data[rollout_num, 0, :2] = \
                (self.x0[:2] - self.xbar[:2]) + self.std_ini * torch.randn(self.x0[:2].shape) # Noise only on position
            if self.std_noise != 0:
                data[rollout_num, 1:, :] = self.std_noise * torch.randn(self.horizon -1, state_dim)

        assert data.shape[0] == num_samples
        return data
