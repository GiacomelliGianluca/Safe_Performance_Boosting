import torch
import torch.nn as nn
import numpy as np

from .contractive_ren import ContractiveREN



class PerfBoostController(nn.Module):
    """
    Performance boosting controller, following the paper:
        "Learning to Boost the Performance of Stable Nonlinear Systems".
    Implements a state-feedback controller with stability guarantees.
    NOTE: When used in closed-loop, the controller input is the measured state of the plant
          and the controller output is the input to the plant.
    This controller has a memory for the last input ("self.last_input") and the last output ("self.last_output").
    """

    def __init__(self,
                 noiseless_forward,
                 input_init: torch.Tensor,
                 output_init: torch.Tensor,
                 nn_type: str = "REN",
                 non_linearity: str = None,
                 # acyclic REN properties
                 dim_internal: int = 8,
                 dim_nl: int = 8,
                 initialization_std: float = 0.5,
                 pos_def_tol: float = 0.001,
                 contraction_rate_lb: float = 1.0,
                 ren_internal_state_init=None,
                 ):
        """
         Args:
            noiseless_forward:            System dynamics without process noise. It can be TV.
            input_init (torch.Tensor):    Initial input to the controller.
            output_init (torch.Tensor):   Initial output from the controller before anything is calculated.
            nn_type (str):                Which NN model to use for the Emme operator (Options: 'REN' or 'SSM')
            non_linearity (str):          Non-linearity used in SSMs for scaffolding.
            ##### the following are the same as AcyclicREN args:
            dim_internal (int):           Internal state (x) dimension.
            dim_nl (int):                 Dimension of the input ("v") and output ("w") of the NL static block of REN.
            initialization_std (float):   [Optional] Weight initialization. Set to 0.1 by default.
            pos_def_tol (float):          [Optional] Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float):  [Optional] Lower bound on the contraction rate. Default to 1.
            ren_internal_state_init (torch.Tensor): [Optional] Initial state of the REN. Default to 0 when None.
        """
        super().__init__()

        # set initial conditions
        self.input_init = input_init.reshape(1, -1)
        self.output_init = output_init.reshape(1, -1)

        # set dimensions
        self.dim_in = self.input_init.shape[-1]
        self.dim_out = self.output_init.shape[-1]

        # set type of nn for emme
        self.nn_type = nn_type
        # define Emme as REN or SSM
        if nn_type == "REN":
            self.emme = ContractiveREN(
                dim_in=self.dim_in, dim_out=self.dim_out, dim_internal=dim_internal,
                dim_nl=dim_nl, initialization_std=initialization_std,
                internal_state_init=ren_internal_state_init,
                pos_def_tol=pos_def_tol, contraction_rate_lb=contraction_rate_lb
            )
        else:
            raise ValueError("Model for emme not implemented")

        # define the system dynamics without process noise
        self.noiseless_forward = noiseless_forward

        # Internal variables
        self.t = None
        self.last_input = None
        self.last_output = None
        # Initialize internal variables
        self.reset()

    def reset(self):
        """
        set time to 0 and reset to initial state.
        """
        self.t = 0  # time
        self.last_input = self.input_init.detach().clone()
        self.last_output = self.output_init.detach().clone()
        self.emme.reset()  # reset emme states to the initial value

    def forward(self, t,  input_t: torch.Tensor):
        """
        Forward pass of the controller.

        Args:
            input_t (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).
            # NOTE: when used in closed-loop, "input_t" is the measured states.

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """

        # apply noiseless forward to get noise less input (noise less state of the plant)
        u_noiseless = self.noiseless_forward(
            t=self.t,
            x=self.last_input,  # last input to the controller is the last state of the plant
            u=self.last_output  # last output of the controller is the last input to the plant
        )  # shape = (batch_size, 1, self.dim_in)

        # reconstruct the noise
        w_ = input_t - u_noiseless  # shape = (batch_size, 1, self.dim_in)

        # apply REN or SSM
        output = self.emme.forward(w_)
        output = output  # shape = (batch_size, 1, self.dim_out)

        # update internal states
        self.last_input, self.last_output = input_t, output
        self.t += 1
        return output
