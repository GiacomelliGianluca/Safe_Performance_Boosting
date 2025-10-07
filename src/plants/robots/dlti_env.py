import gymnasium as gym
from gymnasium import spaces
import torch


class DiscreteTransferFunctionEnv(gym.Env):
    """
    A custom reinforcement learning environment that simulates a discrete-time linear system
    governed by a state-space representation. The environment supports linear-quadratic regulation
    (LQR) with optional constraints on the state and control inputs.

    The system dynamics are defined by:
        x_{k+1} = Ax_k + Bu_k
        y_k = Cx_k + Du_k

    where:
    - A is the state transition matrix.
    - B is the input matrix.
    - C is the output matrix (optional, default is identity matrix for fully observed systems).
    - D is the feedthrough matrix (optional, default is zero).

    Cost is calculated as:
        J = x.T @ Q @ x + u.T @ R @ u

    State, control, and initial conditions can be constrained using the provided limits.

    Args:
        A (torch.Tensor or None): State transition matrix.
        B (torch.Tensor or None): Input matrix.
        Q (torch.Tensor): State cost matrix for LQR.
        R (torch.Tensor): Input cost matrix for LQR.
        dt (float): Time step for the system.
        C (torch.Tensor, optional): Output matrix. Defaults to identity matrix.
        D (torch.Tensor, optional): Feedthrough matrix. Defaults to zero.
        initial_state_low (torch.Tensor, optional): Lower bound for the initial state.
        initial_state_high (torch.Tensor, optional): Upper bound for the initial state.
        state_limit_low (torch.Tensor, optional): Lower bound for state constraints.
        state_limit_high (torch.Tensor, optional): Upper bound for state constraints.
        control_limit_low (torch.Tensor, optional): Lower bound for control input constraints.
        control_limit_high (torch.Tensor, optional): Upper bound for control input constraints.
    """

    def __init__(
        self,
        A,
        B,
        Q,
        R,
        dt,
        C=None,
        D=None,
        initial_state_low=None,
        initial_state_high=None,
        state_limit_low=None,
        state_limit_high=None,
        control_limit_low=None,
        control_limit_high=None,
    ):
        super(DiscreteTransferFunctionEnv, self).__init__()

        # Check if A and B are None, handle dynamically constructed A and B in subclasses
        self.A = (
            A.clone().detach()
            if isinstance(A, torch.Tensor)
            else torch.tensor(A, dtype=torch.float32)
            if A is not None
            else None
        )
        self.B = (
            B.clone().detach()
            if isinstance(B, torch.Tensor)
            else torch.tensor(B, dtype=torch.float32)
            if B is not None
            else None
        )
        self.Q = (
            Q.clone().detach()
            if isinstance(Q, torch.Tensor)
            else torch.tensor(Q, dtype=torch.float32)
        )
        self.R = (
            R.clone().detach()
            if isinstance(R, torch.Tensor)
            else torch.tensor(R, dtype=torch.float32)
        )
        self.dt = dt

        self.n_states = self.Q.shape[0]
        self.n_inputs = self.R.shape[0]

        if C is None:
            self.C = torch.eye(self.n_states)  # Fully Observed System
        else:
            self.C = (
                C.clone().detach()
                if isinstance(C, torch.Tensor)
                else torch.tensor(C, dtype=torch.float32)
            )
        self.n_outputs = self.C.shape[0]

        if D is None:
            self.D = torch.zeros(
                (self.n_outputs, self.n_inputs)
            )  # Physically Realisable System (D = 0)
        else:
            self.D = (
                D.clone().detach()
                if isinstance(D, torch.Tensor)
                else torch.tensor(D, dtype=torch.float32)
            )

        # State Constraints
        self.state_limit_low = (
            state_limit_low.clone().detach()
            if isinstance(state_limit_low, torch.Tensor)
            else torch.tensor(state_limit_low, dtype=torch.float32)
            if state_limit_low is not None
            else -torch.inf * torch.ones((self.n_states,))
        )
        self.state_limit_high = (
            state_limit_high.clone().detach()
            if isinstance(state_limit_high, torch.Tensor)
            else torch.tensor(state_limit_high, dtype=torch.float32)
            if state_limit_high is not None
            else torch.inf * torch.ones((self.n_states,))
        )

        # Control Input Constraints
        self.control_limit_low = (
            control_limit_low.clone().detach()
            if isinstance(control_limit_low, torch.Tensor)
            else torch.tensor(control_limit_low, dtype=torch.float32)
            if control_limit_low is not None
            else -torch.inf * torch.ones((self.n_inputs,))
        )
        self.control_limit_high = (
            control_limit_high.clone().detach()
            if isinstance(control_limit_high, torch.Tensor)
            else torch.tensor(control_limit_high, dtype=torch.float32)
            if control_limit_high is not None
            else torch.inf * torch.ones((self.n_inputs,))
        )

        # Initial State bounds
        self.initial_state_low = (
            initial_state_low.clone().detach()
            if isinstance(initial_state_low, torch.Tensor)
            else torch.tensor(initial_state_low, dtype=torch.float32)
            if initial_state_low is not None
            else self.state_limit_low
        )
        self.initial_state_high = (
            initial_state_high.clone().detach()
            if isinstance(initial_state_high, torch.Tensor)
            else torch.tensor(initial_state_high, dtype=torch.float32)
            if initial_state_high is not None
            else self.state_limit_high
        )

        # Define action space and observation space
        self.action_space = spaces.Box(
            low=self.control_limit_low.numpy(),
            high=self.control_limit_high.numpy(),
            shape=(self.n_inputs,),
        )
        self.observation_space = spaces.Box(
            low=self.state_limit_low.numpy(),
            high=self.state_limit_high.numpy(),
            shape=(self.n_states,),
        )

        # Initialize state
        self.state = None

    def reset(self):
        """
        Resets the environment to a random initial state within the specified bounds.

        Returns:
            torch.Tensor: The initial state of the system.
        """
        self.state = (
            torch.rand(self.n_states)
            * (self.initial_state_high - self.initial_state_low)
            + self.initial_state_low
        )
        return self.state

    def step(self, action):
        """
        Executes a single time step of the system dynamics given the control input (action).

        Args:
            action (numpy.ndarray or torch.Tensor): The control input applied to the system.

        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
                - next_state (torch.Tensor): The next state of the system.
                - reward (float): The negative cost associated with the current state and action.
                - terminated (bool): False, since the environment does not terminate.
                - truncated (bool): False, since the environment does not truncate episodes.
                - info (dict): An empty dictionary (for compatibility with gym environments).
        """
        action = torch.tensor(action, dtype=torch.float32)
        action = torch.clamp(
            action, min=self.control_limit_low, max=self.control_limit_high
        )

        # Discrete-time system update
        if self.A is None or self.B is None:
            raise ValueError(
                "A and B matrices must be defined dynamically in the subclass."
            )

        self.state = self.A @ self.state + self.B @ action
        self.state = torch.clamp(
            self.state, min=self.state_limit_low, max=self.state_limit_high
        )
        output = self.C @ self.state + self.D @ action

        # Calculate the LQR cost
        cost = self.state.T @ self.Q @ self.state + action.T @ self.R @ action
        reward = -cost.item()

        # No termination or truncation conditions
        terminated = False
        truncated = False

        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Prints the current state of the system.

        Args:
            mode (str, optional): Render mode. Defaults to 'human'.
        """
        print(f"State: {self.state}")

    def close(self):
        """
        Cleans up the environment, called when the environment is closed.
        """
        pass
