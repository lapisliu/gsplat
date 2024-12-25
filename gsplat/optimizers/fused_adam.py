import torch
from ..cuda._wrapper import fuse_adam_step_single_tensor, fuse_adam_step_multi_tensor

class FusedAdamSingleTensor(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay=weight_decay)
        super(FusedAdamSingleTensor, self).__init__(params, defaults)
        # print(lr, beta_1, beta_2, eps, weight_decay)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=p.dtype, device=p.device)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=p.dtype, device=p.device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                lr = group['lr']
                beta_1, beta_2 = group['beta_1'], group['beta_2']
                epsilon = group['epsilon']
                weight_decay = group['weight_decay']
                state['step'] += 1
                fuse_adam_step_single_tensor(
                    p.data, grad.data, exp_avg.data, exp_avg_sq.data, state['step'], 
                    lr, beta_1, beta_2, epsilon, weight_decay)



class FusedAdamMultiTensor(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, weight_decay=0.0, chunk_size=1000000):
        self.chunk_size = chunk_size
        defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay=weight_decay)
        super(FusedAdamMultiTensor, self).__init__(params, defaults)

    def step(self):

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

        fuse_adam_step_multi_tensor(
            [param_list, grad_list, exp_avg_list, exp_avg_sq_list], step, 
            lr_list, beta_1_list, beta_2_list, 
            eps_list, weight_decay_list, tensor_to_group, tot_num_elems, self.chunk_size)