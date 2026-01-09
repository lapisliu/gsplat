#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Projection4DGS.cuh"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_4dgs_fused_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] optional
    const scalar_t *__restrict__ opacities, // [B, N]
    const scalar_t *__restrict__ ts,       // [B, N, 1]
    const scalar_t *__restrict__ quats_t,  // [B, N, 4]
    const scalar_t *__restrict__ scales_t, // [B, N, 1]
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const scalar_t *__restrict__ timestamps, // [B, C, 1]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,         // [B, C, N, 2]
    scalar_t *__restrict__ means2d,      // [B, C, N, 2]
    scalar_t *__restrict__ depths,       // [B, C, N]
    scalar_t *__restrict__ conics,       // [B, C, N, 3]
    scalar_t *__restrict__ weighted_opacities, // [B, C, N]
    scalar_t *__restrict__ compensations // [B, C, N] optional
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    quats += bid * N * 4 + gid * 4;
    scales += bid * N * 3 + gid * 3;
    ts += bid * N * 1 + gid * 1;
    quats_t += bid * N * 4 + gid * 4;
    scales_t += bid * N * 1 + gid * 1;
    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;
    timestamps += bid * C * 1 + cid * 1;

    vec3 mean_3d = glm::make_vec3(means);
    mat3 covar(0.f);
    float marginal_t;

    computeCov3D_conditional(glm::make_vec3(scales), *scales_t, 1.0f,
                             glm::make_vec4(quats), glm::make_vec4(quats_t), &covar, mean_3d, ts[0], timestamps[0], marginal_t);

    if (marginal_t < 0.05f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, mean_3d, mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float weighted_opacity = opacities[bid * N + gid] * marginal_t;
    if (compensations != nullptr) {
        // we assume compensation term will be applied later on.
        weighted_opacity *= compensation;
    }
    if (weighted_opacity < ALPHA_THRESHOLD) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }
    // Compute opacity-aware bounding box.
    // https://arxiv.org/pdf/2402.00525 Section B.2
    float extend = min(3.33f, sqrt(2.0f * __logf(weighted_opacity / ALPHA_THRESHOLD)));

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

    if ((radius_x <= radius_clip && radius_y <= radius_clip) ||
        (radius_x > sqrt(image_width * image_width + image_height * image_height) && radius_y > sqrt(image_width * image_width + image_height * image_height))) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    weighted_opacities[idx] = weighted_opacity;
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

void launch_projection_4dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::Tensor opacities,            // [..., N]
    const at::Tensor ts,                   // [..., N, 1]
    const at::Tensor quats_t,              // [..., N, 4]
    const at::Tensor scales_t,             // [..., N, 1]
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const at::Tensor timestamps,           // [..., C, 1]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                      // [..., C, N, 2]
    at::Tensor means2d,                    // [..., C, N, 2]
    at::Tensor depths,                     // [..., C, N]
    at::Tensor conics,                     // [..., C, N, 3]
    at::Tensor weighted_opacities,         // [..., C, N]
    at::optional<at::Tensor> compensations // [..., C, N] optional
) {
    uint32_t N = means.size(-2);    // number of gaussians
    uint32_t C = viewmats.size(-3); // number of cameras
    uint32_t B = means.numel() / (N * 3);    // number of batches

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_4dgs_fused_fwd_kernel",
        [&]() {
            projection_4dgs_fused_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    B,
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    quats.has_value() ? quats.value().data_ptr<scalar_t>()
                                      : nullptr,
                    scales.has_value() ? scales.value().data_ptr<scalar_t>()
                                       : nullptr,
                    opacities.data_ptr<scalar_t>(),
                    ts.data_ptr<scalar_t>(),
                    quats_t.data_ptr<scalar_t>(),
                    scales_t.data_ptr<scalar_t>(),
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    timestamps.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<scalar_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),
                    weighted_opacities.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr
                );
        }
    );
}

template <typename scalar_t>
__global__ void projection_4dgs_fused_bwd_kernel(
    // fwd inputs
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] optional
    const scalar_t *__restrict__ opacities, // [B, N]
    const scalar_t *__restrict__ ts,       // [B, N, 1]
    const scalar_t *__restrict__ quats_t,  // [B, N, 4]
    const scalar_t *__restrict__ scales_t, // [B, N, 1]
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const scalar_t *__restrict__ timestamps, // [B, C, 1]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int32_t *__restrict__ radii,          // [B, C, N, 2]
    const scalar_t *__restrict__ conics,        // [B, C, N, 3]
    const scalar_t *__restrict__ compensations, // [B, C, N] optional
    // grad outputs
    const scalar_t *__restrict__ v_means2d,       // [B, C, N, 2]
    const scalar_t *__restrict__ v_depths,        // [B, C, N]
    const scalar_t *__restrict__ v_conics,        // [B, C, N, 3]
    const scalar_t *__restrict__ v_weighted_opacities, // [B, C, N]
    const scalar_t *__restrict__ v_compensations, // [B, C, N] optional
    // grad inputs
    scalar_t *__restrict__ v_means,   // [B, N, 3]
    scalar_t *__restrict__ v_covars,  // [B, N, 6] optional
    scalar_t *__restrict__ v_quats,   // [B, N, 4] optional
    scalar_t *__restrict__ v_scales,  // [B, N, 3] optional
    scalar_t *__restrict__ v_opacities, // [B, N]
    scalar_t *__restrict__ v_ts,        // [B, N, 1]
    scalar_t *__restrict__ v_quats_t,  // [B, N, 4]
    scalar_t *__restrict__ v_scales_t, // [B, N, 1]
    scalar_t *__restrict__ v_viewmats // [B, C, 4, 4] optional
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    quats += bid * N * 4 + gid * 4;
    scales += bid * N * 3 + gid * 3;
    opacities += bid * N + gid;
    ts += bid * N * 1 + gid * 1;
    quats_t += bid * N * 4 + gid * 4;
    scales_t += bid * N * 1 + gid * 1;

    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;
    timestamps += bid * C * 1 + cid * 1;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;
    v_weighted_opacities += idx;


    // vjp: compute the inverse of the 2d covariance
    mat2 covar2d_inv = mat2(conics[0], conics[1], conics[1], conics[2]);
    mat2 v_covar2d_inv =
        mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    // check magnitude of v_covar2d
//    {
//        float dbg_abs_v_covar2d =
//            fabsf(v_covar2d[0][0]) + fabsf(v_covar2d[0][1]) +
//            fabsf(v_covar2d[1][0]) + fabsf(v_covar2d[1][1]);
//        if (dbg_abs_v_covar2d > 1e2f) {
//            printf("Warning: v_covar2d has too large magnitude: %f\n", dbg_abs_v_covar2d);
//        }
//    }

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

    // transform Gaussian to camera space
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    vec3 means_3d = glm::make_vec3(means);
    mat3 covar(0.f);
    vec4 quat = glm::make_vec4(quats);
    vec3 scale = glm::make_vec3(scales);
    vec4 quat_t = glm::make_vec4(quats_t);
    float marginal_t;
    computeCov3D_conditional(scale, *scales_t, 1.0f,
                             quat, quat_t, &covar, means_3d, ts[0], timestamps[0], marginal_t);

    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // vjp: perspective projection
    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3 v_covar_c(0.f);
    vec3 v_mean_c(0.f);

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    }
//    // check the magnitude of v_covar_c
//    {
//        float dbg_abs_v_covar_c =
//            fabsf(v_covar_c[0][0]) + fabsf(v_covar_c[0][1]) + fabsf(v_covar_c[0][2]) +
//            fabsf(v_covar_c[1][0]) + fabsf(v_covar_c[1][1]) + fabsf(v_covar_c[1][2]) +
//            fabsf(v_covar_c[2][0]) + fabsf(v_covar_c[2][1]) + fabsf(v_covar_c[2][2]);
//        if (dbg_abs_v_covar_c > 1e2f) {
//            printf("Warning: v_covar_c has too large magnitude: %f\n", dbg_abs_v_covar_c);
//        }
//    }

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // vjp: transform Gaussian covariance to camera space
    vec3 v_mean(0.f);
    mat3 v_covar(0.f);
    mat3 v_R(0.f);
    vec3 v_t(0.f);
    posW2C_VJP(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);

    // check the magnitude of v_covar
//    {
//        float dbg_abs_v_covar =
//            fabsf(v_covar[0][0]) + fabsf(v_covar[0][1]) + fabsf(v_covar[0][2]) +
//            fabsf(v_covar[1][0]) + fabsf(v_covar[1][1]) + fabsf(v_covar[1][2]) +
//            fabsf(v_covar[2][0]) + fabsf(v_covar[2][1]) + fabsf(v_covar[2][2]);
//        if (dbg_abs_v_covar > 1e2f) {
//            printf("Warning: v_covar has too large magnitude: %f\n", dbg_abs_v_covar);
//        }
//    }

    float    d_opacity = 0.f;
    float    d_t     = 0.f;
    vec3     d_scale(0.f);
    float    d_scale_t = 0.f;
    vec4     d_quat(0.f);
    vec4     d_quat_t(0.f);

    // pack v_covar (grad wrt world cov3D) to 6-tuple for the conditional-time bwd
    float d_cov3D_6[6];
    pack_sym3_to_six(v_covar, d_cov3D_6);

    // run the conditional-time backward (returns false if masked out by marginal_t threshold)
    const bool used =
        computeCov3D_conditional_bwd(
            /*scale*/     scale,
            /*scale_t*/   *scales_t,
            /*mod*/       1.0f,
            /*rot*/       quat,
            /*rot_r*/     quat_t,
            /*t*/         ts[0],
            /*timestamp*/ timestamps[0],
            /*opacity*/   opacities[0],
            /*d_cov3D*/   d_cov3D_6,
            /*d_mean_w_shifted*/ v_mean,
            /*d_weighted_opacity_in*/ v_weighted_opacities[0],
            /*out grads:*/
            /*d_opacity*/   d_opacity,
            /*d_t_out*/     d_t,
            /*d_scale*/     d_scale,
            /*d_scale_t*/   d_scale_t,
            /*d_rot*/       d_quat,
            /*d_rot_r*/     d_quat_t
        );

//    {
//        float dbg_abs_v_quat =
//            fabsf(d_quat_t[0]) + fabsf(d_quat_t[1]) + fabsf(d_quat_t[2]) + fabsf(d_quat_t[3]);
//        float dbg_abs_v_scale_t = fabsf(d_scale_t);
//
//        if (dbg_abs_v_quat > 1e2f || dbg_abs_v_scale_t > 1e2f) {
//            printf("Warning: d_quat_t or d_scale_t has too large magnitude: %f, %f\n",
//                   dbg_abs_v_quat, dbg_abs_v_scale_t);
//            printf("quat: %f, %f, %f, %f\n", quat.x, quat.y, quat.z, quat.w);
//            printf("quat_t: %f, %f, %f, %f\n", quat_t.x, quat_t.y, quat_t.z, quat_t.w);
//            printf("marginal_t: %f\n", marginal_t);
//        }
//    }

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (used){
        if (v_means != nullptr) {
            warpSum(v_mean, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_means += bid * N * 3 + gid * 3;
#pragma unroll
                for (uint32_t i = 0; i < 3; i++) {
                    gpuAtomicAdd(v_means + i, v_mean[i]);
                }
            }
        }
        if (v_ts != nullptr) {
            warpSum(d_t, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_ts += bid * N + gid;
                gpuAtomicAdd(v_ts, d_t);
            }
        }
        if (v_covars != nullptr) {
            // Output gradients w.r.t. the covariance matrix
            warpSum(v_covar, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_covars += bid * N * 6 + gid * 6;
                gpuAtomicAdd(v_covars, v_covar[0][0]);
                gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
                gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
                gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
                gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
                gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
            }
        } else {
            // Directly output gradients w.r.t. the quaternion and scale
            warpSum(d_quat, warp_group_g);
            warpSum(d_quat_t, warp_group_g);
            warpSum(d_scale, warp_group_g);
            warpSum(d_scale_t, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_quats += bid * N * 4 + gid * 4;
                v_quats_t += bid * N * 4 + gid * 4;
                v_scales += bid * N * 3 + gid * 3;
                v_scales_t += bid * N + gid;

#pragma unroll
                for (uint32_t i = 0; i < 4; i++) {
                    gpuAtomicAdd(v_quats + i, d_quat[i]);
                    gpuAtomicAdd(v_quats_t + i, d_quat_t[i]);
                }
#pragma unroll
                for (uint32_t i = 0; i < 3; i++) {
                    gpuAtomicAdd(v_scales + i, d_scale[i]);
                }
                gpuAtomicAdd(v_scales_t, d_scale_t);
            }
        }
        if (v_opacities != nullptr) {
            warpSum(d_opacity, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_opacities += bid * N + gid;
                gpuAtomicAdd(v_opacities, d_opacity);
            }
        }
    }

    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += bid * C * 16 + cid * 16;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

void launch_projection_4dgs_fused_bwd_kernel(
    // inputs
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::Tensor opacities,            // [..., N]
    const at::Tensor ts,                   // [..., N, 1]
    const at::Tensor quats_t,              // [..., N, 4]
    const at::Tensor scales_t,             // [..., N, 1]
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const at::Tensor timestamps,           // [..., C, 1]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [..., C, N, 2]
    const at::Tensor conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> compensations, // [..., C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [..., C, N, 2]
    const at::Tensor v_depths,                      // [..., C, N]
    const at::Tensor v_conics,                      // [..., C, N, 3]
    const at::Tensor v_weighted_opacities,          // [..., C, N]
    const at::optional<at::Tensor> v_compensations, // [..., C, N] optional
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_means,   // [..., N, 3]
    at::Tensor v_covars,  // [..., N, 6]
    at::Tensor v_quats,   // [..., N, 4]
    at::Tensor v_scales,  // [..., N, 3]
    at::Tensor v_opacities, // [..., N]
    at::Tensor v_ts,        // [..., N, 1]
    at::Tensor v_quats_t,  // [..., N, 4]
    at::Tensor v_scales_t, // [..., N, 1]
    at::Tensor v_viewmats // [..., C, 4, 4]
) {
    uint32_t N = means.size(-2);    // number of gaussians
    uint32_t C = viewmats.size(-3); // number of cameras
    uint32_t B = means.numel() / (N * 3); // number of batches

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_4dgs_fused_bwd_kernel",
        [&]() {
            projection_4dgs_fused_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    B,
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr
                                       : quats.value().data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : scales.value().data_ptr<scalar_t>(),
                    opacities.data_ptr<scalar_t>(),
                    ts.data_ptr<scalar_t>(),
                    quats_t.data_ptr<scalar_t>(),
                    scales_t.data_ptr<scalar_t>(),
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    timestamps.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    v_means2d.data_ptr<scalar_t>(),
                    v_depths.data_ptr<scalar_t>(),
                    v_conics.data_ptr<scalar_t>(),
                    v_weighted_opacities.data_ptr<scalar_t>(),
                    v_compensations.has_value()
                        ? v_compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    v_means.data_ptr<scalar_t>(),
                    covars.has_value() ? v_covars.data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr : v_quats.data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : v_scales.data_ptr<scalar_t>(),
                    v_opacities.data_ptr<scalar_t>(),
                    v_ts.data_ptr<scalar_t>(),
                    v_quats_t.data_ptr<scalar_t>(),
                    v_scales_t.data_ptr<scalar_t>(),
                    viewmats_requires_grad ? v_viewmats.data_ptr<scalar_t>()
                                           : nullptr
                );
        }
    );
}

} // namespace gsplat
