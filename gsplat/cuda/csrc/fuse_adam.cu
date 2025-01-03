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

__global__ void op_adam_multi_tensor_kernel(
    TensorInfo tis, int step, int num_params, int tot_num_elems
) {

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;

    for (int idx = thread_idx; idx < tot_num_elems; idx += stride_x * ILP) {
        int param_idx[ILP]; // tensor idx in tensorInfo list

#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
            param_idx[ii] = num_params - 1;
        }

        int ii_idx = 0; // idx within ILP
        int global_idx = idx;
        int j = 0;
        // #pragma unroll
        while (j < num_params) { // iterate until g_idx < start_idx
            if (global_idx < tis.start_idx[j]) {
                param_idx[ii_idx] = j - 1;
                ii_idx += 1;

                if (ii_idx >= ILP)
                    break;
                else
                    global_idx += stride_x;
            } else {
                j += 1;
            }
        }

        float r_p[ILP];
        float r_g[ILP];
        float r_m[ILP];
        float r_v[ILP];
        float r_l[ILP];
        float r_b1[ILP];
        float r_b2[ILP];
        float r_e[ILP];
        float r_w[ILP];

#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
            int i = idx + stride_x * ii - tis.start_idx[param_idx[ii]];
            if (i < tis.size[param_idx[ii]]) {
                r_p[ii] = tis.param_addr[param_idx[ii]][i];
                r_g[ii] = tis.grad_addr[param_idx[ii]][i];
                r_m[ii] = tis.m_addr[param_idx[ii]][i];
                r_v[ii] = tis.v_addr[param_idx[ii]][i];
                r_l[ii] = tis.lr[param_idx[ii]];
                r_b1[ii] = tis.beta_1[param_idx[ii]];
                r_b2[ii] = tis.beta_2[param_idx[ii]];
                r_e[ii] = tis.epsilon[param_idx[ii]];
                r_w[ii] = tis.weight_decay[param_idx[ii]];
            } else {
                r_g[ii] = 0.0;
                r_p[ii] = 0.0;
                r_m[ii] = 0.0;
                r_v[ii] = 0.0;
                r_l[ii] = 0.0;
                r_b1[ii] = 0.0;
                r_b2[ii] = 0.0;
                r_e[ii] = 0.0;
                r_w[ii] = 0.0;
            }
        }
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
            if (r_w[ii] != 0.0)
                r_g[ii] += r_w[ii] * r_p[ii];

            r_m[ii] = r_b1[ii] * r_m[ii] + (1 - r_b1[ii]) * r_g[ii];
            r_v[ii] = r_b2[ii] * r_v[ii] + (1 - r_b2[ii]) * r_g[ii] * r_g[ii];
            float mt_hat = r_m[ii] / (1 - powf(r_b1[ii], step));
            float vt_hat = r_v[ii] / (1 - powf(r_b2[ii], step));
            r_p[ii] = r_p[ii] - r_l[ii] * mt_hat / (sqrtf(vt_hat) + r_e[ii]);
        }
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
            int i = idx + stride_x * ii - tis.start_idx[param_idx[ii]];
            if (i < tis.size[param_idx[ii]]) {
                tis.param_addr[param_idx[ii]][i] = r_p[ii];
                tis.m_addr[param_idx[ii]][i] = r_m[ii];
                tis.v_addr[param_idx[ii]][i] = r_v[ii];
            }
        }
    }
}

// chuncked version
void FuseAdamStepCUDAMultiTensor(
    std::vector<std::vector<torch::Tensor>> tensor_list,
    int step,
    std::vector<float> lr,
    std::vector<float> beta_1,
    std::vector<float> beta_2,
    std::vector<float> epsilon,
    std::vector<float> weight_decay,
    std::vector<int> tensor_to_group,
    long tot_num_elems,
    int ADAM_CHUNK_SIZE
) {

    std::vector<torch::Tensor> pp = tensor_list[0];
    std::vector<torch::Tensor> grad = tensor_list[1];
    std::vector<torch::Tensor> m = tensor_list[2];
    std::vector<torch::Tensor> v = tensor_list[3];

    int total_params = 0;
    for (int i = 0; i < tensor_list[0].size(); i++) {
        total_params += tensor_list[0][i].numel();
    }

    int num_params = tensor_list[0].size();
    int tot_num_chunks =
        (int)(tot_num_elems + ADAM_CHUNK_SIZE - 1) / ADAM_CHUNK_SIZE;

    int num_threads = ADAM_BLOCK_SIZE;
    int num_blocks =
        min(MAX_NUM_BLOCK,
            (int)(tot_num_elems + num_threads - 1) / num_threads);

    int param_idx = 0;          // global idx of params, linear probing
    long param_offset = 0;      // offset in the current parameter
    int chunk_length = 0;       // offset / final length in the current chunk
    int param_idx_in_chunk = 0; // the idx of params in the current chunk
    TensorInfo tis;

    for (int chunk = 0; chunk < tot_num_chunks; chunk++) {
        for (int t = param_idx;
             t < min(param_idx + MAX_NUM_PARAMS_PER_CHUNK, num_params);
             t++) {
            long tensor_length = tensor_list[0][t].numel();

            tis.param_addr[param_idx_in_chunk] =
                pp[t].data<float>() + param_offset;
            tis.grad_addr[param_idx_in_chunk] =
                grad[t].data<float>() + param_offset;
            tis.m_addr[param_idx_in_chunk] = m[t].data<float>() + param_offset;
            tis.v_addr[param_idx_in_chunk] = v[t].data<float>() + param_offset;
            tis.start_idx[param_idx_in_chunk] = chunk_length;
            tis.lr[param_idx_in_chunk] = lr[tensor_to_group[t]];
            tis.beta_1[param_idx_in_chunk] = beta_1[tensor_to_group[t]];
            tis.beta_2[param_idx_in_chunk] = beta_2[tensor_to_group[t]];
            tis.epsilon[param_idx_in_chunk] = epsilon[tensor_to_group[t]];
            tis.weight_decay[param_idx_in_chunk] =
                weight_decay[tensor_to_group[t]];

            if (tensor_length - param_offset >=
                ADAM_CHUNK_SIZE - chunk_length) {
                tis.size[param_idx_in_chunk] = ADAM_CHUNK_SIZE - chunk_length;
                param_offset += ADAM_CHUNK_SIZE - chunk_length;
                chunk_length = ADAM_CHUNK_SIZE;
                param_idx_in_chunk += 1;
                param_idx = t;
                if (param_offset == tensor_length) {
                    param_offset = 0;
                    param_idx = t + 1;
                }
                break;
            } else {
                tis.size[param_idx_in_chunk] =
                    (int)(tensor_length - param_offset);
                chunk_length += (int)(tensor_length - param_offset);
                param_idx_in_chunk += 1;
                param_offset = 0;
                param_idx = t + 1;
                if (param_idx_in_chunk >= MAX_NUM_PARAMS_PER_CHUNK)
                    break;
            }
        }
        op_adam_multi_tensor_kernel<<<num_blocks, num_threads>>>(
            tis, step, param_idx_in_chunk, chunk_length
        );
        chunk_length = 0;
        param_idx_in_chunk = 0;
    }
    cudaDeviceSynchronize();
}

__constant__ float const_lr[6]; // Assuming the number of parameters (groups) is
                                // less than or equal to 6
__constant__ float const_beta1[6];
__constant__ float const_beta2[6];
__constant__ float const_correction1[6];
__constant__ float const_correction2[6];
__constant__ float const_epsilon[6];
__constant__ float const_weight_decay[6];
__constant__ int const_data_point_to_group[6];

__global__ void op_customized_fused_adam_kernel(
    float **params,
    float **grads,
    float **moment1,
    float **moment2,
    int tot_num_elems,
    int num_params
) {

    int stride_x = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < tot_num_elems;
         idx += stride_x) {
        int group_idx = 0;
#pragma unroll
        for (int i = 0; i < num_params; ++i) {
            if (idx < const_data_point_to_group[i]) { //const_data_point_to_group[num_params-1] should be tot_num_elems,
                                                      // so the last group will be handled correctly
                group_idx = i;
                break;
            }
        }
        int cur_idx =
            idx -
            (group_idx == 0 ? 0 : const_data_point_to_group[group_idx - 1]);

        if (cur_idx < 0 ) {
            return;
        }

        float p = params[group_idx][cur_idx];
        float g = grads[group_idx][cur_idx];
        float m = moment1[group_idx][cur_idx];
        float v = moment2[group_idx][cur_idx];

        g += const_weight_decay[group_idx] * p;

        m = const_beta1[group_idx] * m + (1 - const_beta1[group_idx]) * g;
        v = const_beta2[group_idx] * v + (1 - const_beta2[group_idx]) * g * g;
        float m_hat = m / const_correction1[group_idx];
        float v_hat = v / const_correction2[group_idx];

        params[group_idx][cur_idx] = p + (-const_lr[group_idx] * m_hat / (sqrtf(v_hat) + const_epsilon[group_idx]));
        moment1[group_idx][cur_idx] = m;
        moment2[group_idx][cur_idx] = v;
    }
}

void customized_fused_adam_update(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    int step,
    std::vector<float> lr,
    std::vector<float> beta_1,
    std::vector<float> beta_2,
    std::vector<float> epsilon,
    std::vector<float> weight_decay
) {

    int num_params = params.size();
    if (num_params > 6) {
        printf("The number of parameters should be less than or equal to 6\n");
        return;
    }

    cudaMemcpyToSymbol(const_lr, lr.data(), num_params * sizeof(float));
    cudaMemcpyToSymbol(const_beta1, beta_1.data(), num_params * sizeof(float));
    cudaMemcpyToSymbol(const_beta2, beta_2.data(), num_params * sizeof(float));
    cudaMemcpyToSymbol(
        const_epsilon, epsilon.data(), num_params * sizeof(float)
    );
    cudaMemcpyToSymbol(
        const_weight_decay, weight_decay.data(), num_params * sizeof(float)
    );

    int data_point_to_group[6]; // param[i] belongs to param group j if
                                // data_point_to_group[j-1] <= i <
                                // data_point_to_group[j]
    int tot_num_elems = 0;
    for (int i = 0; i < num_params; i++) {
        tot_num_elems += params[i].numel();
        data_point_to_group[i] = tot_num_elems;
    }

    cudaMemcpyToSymbol(
        const_data_point_to_group, data_point_to_group, num_params * sizeof(int)
    );

    //precaculate the correction factor
    float correction1[6];
    float correction2[6];
    for (int i = 0; i < num_params; i++) {
        correction1[i] = 1 - powf(beta_1[i], step);
        correction2[i] = 1 - powf(beta_2[i], step);
    }
    cudaMemcpyToSymbol(
        const_correction1, correction1, num_params * sizeof(float)
    );
    cudaMemcpyToSymbol(
        const_correction2, correction2, num_params * sizeof(float)
    );

    int num_threads = 256;
    int num_blocks =  (int)((tot_num_elems + num_threads - 1) / num_threads);

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

    float **d_params, **d_grads, **d_moment1, **d_moment2;
    cudaMalloc(&d_params, num_params * sizeof(float *));
    cudaMalloc(&d_grads, num_params * sizeof(float *));
    cudaMalloc(&d_moment1, num_params * sizeof(float *));
    cudaMalloc(&d_moment2, num_params * sizeof(float *));
    cudaMemcpy(d_params, param_ptrs.data(), num_params * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grads, grad_ptrs.data(), num_params * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moment1, exp_avg_ptrs.data(), num_params * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moment2, exp_avg_sq_ptrs.data(), num_params * sizeof(float *), cudaMemcpyHostToDevice);

    op_customized_fused_adam_kernel<<<num_blocks, num_threads>>>(
        d_params, d_grads, d_moment1, d_moment2, tot_num_elems, num_params
        );


    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(launchErr));
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_params);
    cudaFree(d_grads);
    cudaFree(d_moment1);
    cudaFree(d_moment2);
}
} // namespace gsplat