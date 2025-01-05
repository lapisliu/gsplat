#include "bindings.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <vector>
// #include <cuda_profiler_api.h>

// gt = dt + lambda * t
// mt = beta_1 * mt + (1 - beta_1) * gt
// vt = beta_2 * vt + (1 - beta_2) * gt ^ 2
// mt_hat = mt / (1 - beta_1 ^ step)
// vt_hat = vt / (1 - beta_2 ^ step)
// if amsgrad: not implement now
// out = t - lr * mt_hat / (sqrt(vt_hat) + epsilon)

namespace gsplat {

namespace cg = cooperative_groups;

__constant__ float const_lr[6]; // Assuming the number of parameters (groups) is
                                // less than or equal to 6

//assuming these are the same for all parameters
__constant__ float const_beta1;
__constant__ float const_beta2;
__constant__ float const_correction1; //precomputed the bias corrections
__constant__ float const_correction2;
__constant__ float const_epsilon;
__constant__ float const_weight_decay;

__constant__ long const_data_point_to_group[6];

float fused_adam_kernel_beta1;
float fused_adam_kernel_beta2;
float fused_adam_kernel_prev_correction1;
float fused_adam_kernel_prev_correction2;

float **fused_adam_kernel_d_params, **fused_adam_kernel_d_grads, **fused_adam_kernel_d_moment1, **fused_adam_kernel_d_moment2;

void fused_adam_init(
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay
) {
    cudaMemcpyToSymbol(const_beta1, &beta1, sizeof(float));
    cudaMemcpyToSymbol(const_beta2, &beta2, sizeof(float));
    cudaMemcpyToSymbol(const_epsilon, &epsilon, sizeof(float));
    cudaMemcpyToSymbol(const_weight_decay, &weight_decay, sizeof(float));
    fused_adam_kernel_beta1 = beta1;
    fused_adam_kernel_beta2 = beta2;
    fused_adam_kernel_prev_correction1 = 1 - beta1;
    fused_adam_kernel_prev_correction2 = 1 - beta2;
    cudaMemcpyToSymbol(
            const_correction1, &fused_adam_kernel_prev_correction1, sizeof(float)
    );
    cudaMemcpyToSymbol(
            const_correction2, &fused_adam_kernel_prev_correction2, sizeof(float)
    );
    cudaMalloc(&fused_adam_kernel_d_params, 6 * sizeof(float *));
    cudaMalloc(&fused_adam_kernel_d_grads, 6 * sizeof(float *));
    cudaMalloc(&fused_adam_kernel_d_moment1, 6 * sizeof(float *));
    cudaMalloc(&fused_adam_kernel_d_moment2, 6 * sizeof(float *));
}

void fused_adam_free() {
    cudaFree(fused_adam_kernel_d_params);
    cudaFree(fused_adam_kernel_d_grads);
    cudaFree(fused_adam_kernel_d_moment1);
    cudaFree(fused_adam_kernel_d_moment2);
}

__global__ void op_customized_fused_adam_kernel(
    float **params,
    float **grads,
    float **moment1,
    float **moment2,
    long tot_num_elems,
    int num_params
) {
    __shared__ float *shared_params[6];
    __shared__ float *shared_grads[6];
    __shared__ float *shared_moment1[6];
    __shared__ float *shared_moment2[6];

    if (threadIdx.x == 0) {
        for (int i = 0; i < num_params; ++i) {
            shared_params[i] = params[i];
            shared_grads[i] = grads[i];
            shared_moment1[i] = moment1[i];
            shared_moment2[i] = moment2[i];
        }
    }
    __syncthreads();

    int stride_x = blockDim.x * gridDim.x;
    int group_idx = 0;
    for (long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < tot_num_elems;
         idx += stride_x) {

        for (int i = group_idx; i < num_params; ++i) {
            if (idx < const_data_point_to_group[i]) {
                group_idx = i;
                break;
            }
        }
        int cur_idx = (int)(
            idx -
            (group_idx == 0 ? 0 : const_data_point_to_group[group_idx - 1])
        );

        if (cur_idx < 0) {
            return;
        }

        float p = shared_params[group_idx][cur_idx];
        float g = shared_grads[group_idx][cur_idx];
        float m = shared_moment1[group_idx][cur_idx];
        float v = shared_moment2[group_idx][cur_idx];

        g += const_weight_decay * p;

        m = const_beta1 * m + (1 - const_beta1) * g;
        v = const_beta2 * v + (1 - const_beta2) * g * g;
        float m_hat = m / const_correction1;
        float v_hat = v / const_correction2;

        shared_params[group_idx][cur_idx] = p + (-const_lr[group_idx] * m_hat / (sqrtf(v_hat) + const_epsilon));
        shared_moment1[group_idx][cur_idx] = m;
        shared_moment2[group_idx][cur_idx] = v;
    }
}

void customized_fused_adam_update(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<float> lr,
    int step
) {

    int num_params = params.size();
    if (num_params > 6) {
        printf("The number of parameters should be less than or equal to 6\n");
        return;
    }

    cudaMemcpyToSymbol(const_lr, lr.data(), num_params * sizeof(float));
    if(fused_adam_kernel_prev_correction1 != 1.0) {
        fused_adam_kernel_prev_correction1 = (1.0 - powf(fused_adam_kernel_beta1, step));
        cudaMemcpyToSymbol(
            const_correction1, &fused_adam_kernel_prev_correction1, sizeof(float)
        );
    }
    if(fused_adam_kernel_prev_correction2 != 1.0) {
        fused_adam_kernel_prev_correction2 = (1.0 - powf(fused_adam_kernel_beta2, step));
        cudaMemcpyToSymbol(
            const_correction2, &fused_adam_kernel_prev_correction2, sizeof(float)
        );
    }

    long data_point_to_group[6]; // param[i] belongs to param group j if
                                 // data_point_to_group[j-1] <= i < data_point_to_group[j]
    long tot_num_elems = 0;
    for (int i = 0; i < num_params; i++) {
        tot_num_elems += params[i].numel();
        data_point_to_group[i] = tot_num_elems;
    }
    cudaMemcpyToSymbol(
        const_data_point_to_group, data_point_to_group, num_params * sizeof(long)
    );

    std::vector<float *> param_ptrs(num_params);
    std::vector<float *> grad_ptrs(num_params);
    std::vector<float *> exp_avg_ptrs(num_params);
    std::vector<float *> exp_avg_sq_ptrs(num_params);

    for (int i = 0; i < num_params; i++) {
        param_ptrs[i] = params[i].data_ptr<float>();
        grad_ptrs[i] = grads[i].data_ptr<float>();
        exp_avg_ptrs[i] = exp_avgs[i].data_ptr<float>();
        exp_avg_sq_ptrs[i] = exp_avg_sqs[i].data_ptr<float>();
    }

    cudaMemcpy(fused_adam_kernel_d_params, param_ptrs.data(), num_params * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(fused_adam_kernel_d_grads, grad_ptrs.data(), num_params * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(fused_adam_kernel_d_moment1, exp_avg_ptrs.data(), num_params * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(fused_adam_kernel_d_moment2, exp_avg_sq_ptrs.data(), num_params * sizeof(float *), cudaMemcpyHostToDevice);

    int num_blocks =  std::min(FUSED_ADAM_MAX_NUM_BLOCK,(int)((tot_num_elems + FUSED_ADAM_BLOCK_SIZE - 1) / FUSED_ADAM_BLOCK_SIZE));
    op_customized_fused_adam_kernel<<<num_blocks, FUSED_ADAM_BLOCK_SIZE>>>(
        fused_adam_kernel_d_params, fused_adam_kernel_d_grads, fused_adam_kernel_d_moment1, fused_adam_kernel_d_moment2, tot_num_elems, num_params
    );


    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(launchErr));
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace gsplat