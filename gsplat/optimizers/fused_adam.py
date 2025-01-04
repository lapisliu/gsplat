import torch
from ..cuda._wrapper import fuse_adam_step_multi_tensor, customized_fused_adam_update, fused_adam_init, fused_adam_free


class FusedAdamMultiTensor(torch.optim.Optimizer):
    def __init__(self, params, betas, eps=1e-8, lr=1e-3, weight_decay=0.0, chunk_size=1000000):
        beta_1, beta_2 = betas
        self.chunk_size = chunk_size
        defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=eps, weight_decay=weight_decay)
        super(FusedAdamMultiTensor, self).__init__(params, defaults)

    def step(self, closure=None):

        param_list = []
        grad_list = []
        exp_avg_list = []
        exp_avg_sq_list = []
        lr_list = []
        beta_1_list = []
        beta_2_list = []
        eps_list = []
        weight_decay_list = []
        tensor_to_group = []
        group_idx = 0
        tot_num_elems = 0
        step = 0

        for group in self.param_groups:
            lr = group['lr']
            beta_1, beta_2 = group['beta_1'], group['beta_2']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']

            lr_list.append(lr)
            beta_1_list.append(beta_1)
            beta_2_list.append(beta_2)
            eps_list.append(epsilon)
            weight_decay_list.append(weight_decay)

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=p.dtype, device=p.device)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=p.dtype, device=p.device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                param_list.append(p.data.contiguous())
                grad_list.append(p.grad.contiguous())
                exp_avg_list.append(exp_avg.data.contiguous())
                exp_avg_sq_list.append(exp_avg_sq.data.contiguous())
                step = state['step']

                tot_num_elems += p.numel()
                tensor_to_group.append(group_idx)

            group_idx += 1

        if hasattr(self, 'verbose') and self.verbose:
            print(f"Launching fused kernel with {tot_num_elems} elements and {len(param_list)} parameters.")

        fuse_adam_step_multi_tensor(
            [param_list, grad_list, exp_avg_list, exp_avg_sq_list], step,
            lr_list, beta_1_list, beta_2_list,
            eps_list, weight_decay_list, tensor_to_group, tot_num_elems, self.chunk_size)


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
