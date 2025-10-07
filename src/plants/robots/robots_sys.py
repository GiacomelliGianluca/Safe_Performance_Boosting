import torch
import numpy as np
import torch.nn.functional as F


class RobotsSystem(torch.nn.Module):
    def __init__(self, n_agents, xbar: torch.Tensor, linear_plant: bool, x_init=None, u_init=None,
                 prestabilized=True, k: float = 1.0, x_min = None, x_max = None, u_min = None, u_max = None
                 ):
        """
        Args:
            xbar:           Concatenated nominal equilibrium point of all agents.
            linear_plant:   If True, a linearized model of the system is used.
                            Otherwise, the model is nonlinear due to the dependence of friction on the speed.
            x_init:         Concatenated initial point of all agents. Default to xbar when None.
            u_init:         Initial input to the plant. Defaults to zero when None.
            k (float):      Gain of the pre-stabilizing controller (acts as a spring constant).
        """
        super().__init__()

        self.n_agents = n_agents
        self.linear_plant = linear_plant
        self.register_buffer('xbar', xbar.reshape(1, -1))  # Reference allocation shape = (1, state_dim)
        # initial state
        x_init = self.xbar.detach().clone() if x_init is None else x_init.reshape(1, -1)   #  Initial state allocation shape = (1, state_dim)
        self.register_buffer('x_init', x_init)
        if u_init is None:
            u_init = torch.zeros(1, int(self.xbar.shape[1]/2))
        else:
            u_init.reshape(1, -1)  # Initial Input allocation shape = (1, in_dim)
        self.register_buffer('u_init', u_init)
        # check dimensions
        self.state_dim = 4 * self.n_agents
        self.in_dim = 2 * self.n_agents
        assert self.xbar.shape[1] == self.state_dim and self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.in_dim

        self.prestabilized = prestabilized


        self.tau_s= 0.05  #Sampling time
        self.mass = 1.0
        self.k = k
        self.b = 1.0
        self.b2 = None if self.linear_plant else 0.1

        # B matrix (Control input influence on the system)
        self.B = self.tau_s * torch.kron(
            torch.eye(self.n_agents),
            torch.tensor([[0, 0], [0, 0], [1 / self.mass, 0], [0, 1 / self.mass]]),
        )

        # Constraints
        if x_min is None:
            self.x_min = np.array((self.state_dim, 1), -np.inf)
        else:
            self.x_min = np.array(x_min)

        if x_max is None:
            self.x_max = np.array((self.state_dim, 1), np.inf)
        else:
            self.x_max = np.array(x_max)

        if u_min is None:
            self.u_min = np.array((self.in_dim, 1), np.inf)
        else:
            self.u_min = np.array(u_min)

        if u_max is None:
            self.u_max = np.array((self.in_dim, 1), np.inf)
        else:
            self.u_max = np.array(u_max)


    def A_matrix(self, x):
        """
        Constructs the system matrix (A matrix) based on the current state of the system, including
        damping, prestabilization, and mass-spring parameters.

        Args:
            x (torch.Tensor): The current state of the system.

        Returns:
            torch.Tensor: The computed A matrix.
        """
        b1 = self.b
        m, k = self.mass, self.k

        A1 = torch.eye(4 * self.n_agents)
        A2 = torch.cat(
            (
                torch.cat((torch.zeros(2, 2), torch.eye(2)), dim=1),
                torch.cat(
                    (
                        torch.diag(torch.tensor([-k / m, -k / m])),
                        torch.diag(torch.tensor([-b1 / m, -b1 / m])),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        A2 = torch.kron(torch.eye(self.n_agents), A2)
        A = A1 + self.tau_s * A2
        return A

    def noiseless_forward(self, t, x: torch.Tensor, u: torch.Tensor):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state of the noise-free dynamics.
        """
        x = x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.in_dim)

        A_x = self.A_matrix(x)
        mask = torch.cat([torch.zeros(2), torch.ones(2)]).repeat(self.n_agents)
        if self.prestabilized:
            # State evolution: x_{t+1} = A(x) @ (x - target_positions) + B @ u + target_positions
            f = (
                    F.linear(x - self.xbar, A_x)
                    + F.linear(u, self.B)
                    + self.tau_s
                    * self.b2
                    / self.mass
                    * mask
                    * torch.tanh(x - self.xbar)
                    + self.xbar
            )
        else:
            # State evolution: x_{t+1} = A(x) @ x + B @ u
            f = F.linear(x, A_x) + F.linear(u, self.B)
        return f

    # shape = (batch_size, 1, state_dim); f corresponds with x_{t+1}

    def forward(self, t, x, u, w):
        """
        forward of the plant with the process noise.
        Args:
            - t (int):          current time step
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)
        Returns:
            next state.
        """
        return self.noiseless_forward(t, x, u) + w.view(-1, 1, self.state_dim)

    # simulation
    def rollout(self, controller, data, train=False):
        """
        rollout REN for rollouts of the process noise
        Args:
            - data: sequence of disturbance samples of shape (batch_size, T, state_dim).
        Return:
            - x_log of shape (batch_size, T, state_dim)
            - u_log of shape (batch_size, T, in_dim)
        """

        # init
        controller.reset()
        x = self.x_init.detach().clone().repeat(data.shape[0], 1, 1)
        u = self.u_init.detach().clone().repeat(data.shape[0], 1, 1)

        # Simulate
        if train:
            for t in range(data.shape[1]):
                x = self.forward(t=t, x=x, u=u, w=data[:, t:t+1, :])    # shape = (batch_size, 1, state_dim)
                u = controller(t, x)                                       # shape = (batch_size, 1, in_dim)

                if t == 0:
                    x_log, u_log = x, u
                else:
                    x_log = torch.cat((x_log, x), 1)
                    u_log = torch.cat((u_log, u), 1)
        else:
            with torch.no_grad():
                for t in range(data.shape[1]):
                    x = self.forward(t=t, x=x, u=u, w=data[:, t:t + 1, :])  # shape = (batch_size, 1, state_dim)
                    u = controller(t, x)  # shape = (batch_size, 1, in_dim)

                    if t == 0:
                        x_log, u_log = x, u
                    else:
                        x_log = torch.cat((x_log, x), 1)
                        u_log = torch.cat((u_log, u), 1)
        controller.reset()

        return x_log, None, u_log

    def checking_constraints(self, x_log, u_log):
        all_x_max_safe = []

        for i, constr in enumerate(self.x_max):
            all_x_max_safe.append(torch.all(x_log[:, :, i] <= self.x_max[i]))

        if all(all_x_max_safe):
            print("Maximum constraint on the state x not violated :)")
            viol_x_max = None
        else:
            viol_x_max = []
            for index, is_safe in enumerate(all_x_max_safe):
                if not is_safe:
                    viol_x_max.append(x_log[:, :, index] > self.x_max[index])
                    ind_viol_x_max = viol_x_max[index].nonzero(as_tuple=True)
                    epoch_viol_x_max = ind_viol_x_max[0]
                    inst_viol_x_max = ind_viol_x_max[1]
                    print("Maximum constraint for the element " + str(index+1) + "of the state x are violated for:")
                    for ds, inst in zip(epoch_viol_x_max, inst_viol_x_max):
                        print(f"Data sample: " + str(ds.item()) + " Instant: " + str(inst.item()) + " \n")
                else:
                    viol_x_max.append('Ok')
                    print("Maximum constraint for the element " + str(index+1) + " of the state x not violated ")


        all_x_min_safe = []
        for i, constr in enumerate(self.x_min):
            all_x_min_safe.append(torch.all(x_log[:, :, i] >= self.x_min[i]))

        if all(all_x_min_safe):
            print("Minimum constraint on the state x not violated :)")
            viol_x_min = None
        else:
            viol_x_min = []
            for index, is_safe in enumerate(all_x_min_safe):
                if not is_safe:
                    viol_x_min.append(x_log[:, :, index] < self.x_min[index])
                    ind_viol_x_min = viol_x_min[index].nonzero(as_tuple=True)
                    epoch_viol_x_min = ind_viol_x_min[0]
                    inst_viol_x_min = ind_viol_x_min[1]
                    print("Minimum constraint for the element " + str(index+1) + " of the state x are violated for:")
                    for ds, inst in zip(epoch_viol_x_min, inst_viol_x_min):
                        print(f"Data sample: " + str(ds.item()) + " Instant: " + str(inst.item()) + " \n")
                else:
                    viol_x_min.append('Ok')
                    print("Minimum constraint for the element " + str(index+1) + " of the state x not violated ")

        all_u_max_safe = []
        for i, constr in enumerate(self.u_max):
            all_u_max_safe.append(torch.all(u_log[:, :, i] <= self.u_max[i]))


        if all(all_u_max_safe):
            print("Maximum constraint on the input u not violated :)")
            viol_u_max = None
        else:
            viol_u_max = []
            for index, is_safe in enumerate(all_u_max_safe):
                if not is_safe:
                    viol_u_max.append(u_log[:, :, index] > self.u_max[index])
                    ind_viol_u_max = viol_u_max[index].nonzero(as_tuple=True)
                    epoch_viol_u_max = ind_viol_u_max[0]
                    inst_viol_u_max = ind_viol_u_max[1]
                    print("Maximum constraint for the element " + str(index+1) + " of the input u are violated for:")
                    for ds, inst in zip(epoch_viol_u_max, inst_viol_u_max):
                        print(f"Element " + str(index+1) + " of the input u - Data sample: " + str(ds.item()) + " Instant: " + str(inst.item()) + " \n")
                else:
                    viol_u_max.append('Ok')
                    print("Maximum constraint for the element " + str(index+1) + " of the input u not violated ")


        all_u_min_safe = []
        for i, constr in enumerate(self.u_min):
            all_u_min_safe.append(torch.all(u_log[:, :, i] >= self.u_min[i]))


        if all(all_u_min_safe):
            print("Minimum constraint on the input u not violated :)")
            viol_u_min = None
        else:
            viol_u_min = []
            for index, is_safe in enumerate(all_u_min_safe):
                if not is_safe:
                    viol_u_min.append(u_log[:, :, index] < self.u_min[index])
                    ind_viol_u_min = viol_u_min[index].nonzero(as_tuple=True)
                    epoch_viol_u_min = ind_viol_u_min[0]
                    inst_viol_u_min = ind_viol_u_min[1]
                    print("Minimum constraint for the element " + str(index+1) + " of the input u are violated for:")
                    for ds, inst in zip(epoch_viol_u_min, inst_viol_u_min):
                        print(f"Element " + str(index+1) + " of the input u - Data sample: " + str(ds.item()) + " Instant: " + str(inst.item()) + " \n")
                else:
                    viol_u_min.append('Ok')
                    print("Minimum constraint for the element " + str(index+1) + " of the input u not violated ")

        return viol_x_max, viol_x_min, viol_u_max, viol_u_min

