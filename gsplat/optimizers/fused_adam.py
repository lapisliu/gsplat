import torch
from ..cuda._wrapper import customized_fused_adam_update, fused_adam_init, fused_adam_free


class CustomizedFusedAdam(torch.optim.Adam):
    def __init__(self, params, betas, eps=1e-8, lr=1e-3, weight_decay=0.0):
        super(CustomizedFusedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.param_list = []
        self.grad_list = []
        self.exp_avg_list = []
        self.exp_avg_sq_list = []
        self.lr_list = []

        # assume all groups have the same betas, eps, and weight_decay, and they don't change
        # init the values to kernel
        fused_adam_init(betas[0], betas[1], eps, weight_decay)

    def step(self, closure=None):
        self.param_list.clear()
        self.grad_list.clear()
        self.exp_avg_list.clear()
        self.exp_avg_sq_list.clear()
        self.lr_list.clear()

        step = 0

        for group in self.param_groups:
            lr = group['lr']

            self.lr_list.append(lr)

            assert len(group['params']) == 1, "more than one tensor in group"
            p = group['params'][0]
            if p.grad is None:
                continue

            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data, dtype=p.dtype, device=p.device)
                state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=p.dtype, device=p.device)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            step = state['step']

            self.param_list.append(p.data.contiguous())
            self.grad_list.append(p.grad.contiguous())
            self.exp_avg_list.append(exp_avg.data.contiguous())
            self.exp_avg_sq_list.append(exp_avg_sq.data.contiguous())

        if hasattr(self, 'verbose') and self.verbose:
            print(f"Launching fused kernel with {len(self.param_list)} parameters.")

        customized_fused_adam_update(
            self.param_list, self.grad_list, self.exp_avg_list, self.exp_avg_sq_list,
            self.lr_list, step
        )

    def __del__(self):
        fused_adam_free()
