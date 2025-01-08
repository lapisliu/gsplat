import torch
from typing import Dict, Union
from ..cuda._wrapper import customized_fused_adam_update, fused_adam_init, fused_adam_free


class CustomizedFusedAdam:
    def __init__(self, betas, eps=1e-8, weight_decay=0.0):
        """
        Initializes the fused Adam optimizer with the given parameters.
        Assumes all groups share the same betas, eps, and weight_decay values.
        """
        self.param_list = []
        self.grad_list = []
        self.exp_avg_list = []
        self.exp_avg_sq_list = []
        self.lr_list = []
        self.step_counter = 0

        # Initialize kernel values
        fused_adam_init(betas[0], betas[1], eps, weight_decay)

    def clear_lists(self):
        """Clears the internal state lists before each optimization step."""
        self.param_list.clear()
        self.grad_list.clear()
        self.exp_avg_list.clear()
        self.exp_avg_sq_list.clear()
        self.lr_list.clear()

    def process_optimizer(self, optimizer):
        """
        Processes a single optimizer to extract parameter data and update the state lists.
        """
        for group in optimizer.param_groups:
            lr = group['lr']
            self.lr_list.append(lr)

            assert len(group['params']) == 1, "more than one tensor in group"
            param = group['params'][0]
            if param.grad is None:
                print("No gradient for parameter")
                continue

            state = optimizer.state[param]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param.data, dtype=param.dtype, device=param.device)
                state['exp_avg_sq'] = torch.zeros_like(param.data, dtype=param.dtype, device=param.device)
            state['step'] += 1
            self.step_counter = state['step']

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            self.param_list.append(param.data.contiguous())
            self.grad_list.append(param.grad.contiguous())
            self.exp_avg_list.append(exp_avg.data.contiguous())
            self.exp_avg_sq_list.append(exp_avg_sq.data.contiguous())

    def step(self, optimizers: Union[Dict[str, torch.optim.Optimizer], torch.optim.Optimizer]):
        """
        Performs a single optimization step.
        Each group in optimizer.param_groups must contain the following keys:
        - 'lr': float
        optimizer.state[param] must contain the following keys:
        - 'step': int
        - 'exp_avg': torch.Tensor
        - 'exp_avg_sq': torch.Tensor
        """
        self.clear_lists()

        if isinstance(optimizers, dict):
            for optimizer in optimizers.values():
                self.process_optimizer(optimizer)
        else:
            self.process_optimizer(optimizers)

        assert len(self.param_list) == len(self.grad_list) == len(self.exp_avg_list) == len(self.exp_avg_sq_list) == len(self.lr_list), (
            "Number of parameters, gradients, exp_avgs, exp_avg_sqs, and learning rates must match."
        )
        assert len(self.param_list) > 0, "No parameters to optimize."
        assert len(self.param_list) <= 6, "Number of parameters must be 6 or less."

        customized_fused_adam_update(
            self.param_list,
            self.grad_list,
            self.exp_avg_list,
            self.exp_avg_sq_list,
            self.lr_list,
            self.step_counter
        )

    def __del__(self):
        """Frees the resources allocated for the fused Adam kernel."""
        fused_adam_free()
