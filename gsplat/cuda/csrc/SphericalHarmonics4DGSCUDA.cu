#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"

namespace gsplat {

__device__ constexpr float kPI = 3.14159265358979323846f;

namespace cg = cooperative_groups;

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

template <typename scalar_t>
__device__ void sh4d_coeffs_to_color_fast(
    const uint32_t degree,       // spatial SH degree (0..3)
    const uint32_t degree_t,     // temporal degree (0..N_harmonics), cosine-only
    const uint32_t c,            // color channel 0..2
    const vec3 &dir_in,   // direction (will be normalized)
    const float dir_t,           // relative time: (ts[idx] - timestamp)
    const float time_duration,   // total time span for 1 cycle
    const scalar_t *coeffs,         // [K, 3] = [(degree+1)^2 * (degree_t+1), 3]
    scalar_t *colors                // [3]
) {
    // normalize direction
    float inv_norm = rsqrtf(dir_in.x * dir_in.x +
                            dir_in.y * dir_in.y +
                            dir_in.z * dir_in.z);
    float x = dir_in.x * inv_norm;
    float y = dir_in.y * inv_norm;
    float z = dir_in.z * inv_norm;

    // Clamp spatial degree to at most 3
    uint32_t deg = degree;
    if (deg > 3u) deg = 3u;

    // Precompute spatial SH basis scalars in gsplat’s ordering (degree-major)
    // up to degree<=3 (16 terms max).
    float basis[16];  // used up to (deg+1)^2
#pragma unroll
    for (int i = 0; i < 16; ++i) basis[i] = 0.0f;

    // l=0 (k = 0)
    basis[0] = 0.2820947917738781f; // SH_C0

    if (deg >= 1u) {
        // l=1 (k = 1..3)
        // 0.48860251190292f ≈ SH_C1
        basis[1] = 0.48860251190292f * (-y);
        basis[2] = 0.48860251190292f * ( z);
        basis[3] = 0.48860251190292f * (-x);
    }
    if (deg >= 2u) {
        // l=2 (k = 4..8)
        float z2  = z * z;
        float fTmp0B = -1.092548430592079f * z;
        float fC1 = x * x - y * y;
        float fS1 = 2.0f * x * y;
        float pSH6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
        float pSH7 = fTmp0B * x;
        float pSH5 = fTmp0B * y;
        float pSH8 = 0.5462742152960395f * fC1;
        float pSH4 = 0.5462742152960395f * fS1;

        basis[4] = pSH4;
        basis[5] = pSH5;
        basis[6] = pSH6;
        basis[7] = pSH7;
        basis[8] = pSH8;
    }
    if (deg >= 3u) {
        // l=3 (k = 9..15)
        float z2  = z * z;
        float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
        float fTmp1B = 1.445305721320277f * z;
        float x2 = x * x, y2 = y * y;
        float fC1 = x2 - y2;
        float fS1 = 2.0f * x * y;
        float fC2 = x * fC1 - y * fS1;
        float fS2 = x * fS1 + y * fC1;

        float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
        float pSH13 = fTmp0C * x;
        float pSH11 = fTmp0C * y;
        float pSH14 = fTmp1B * fC1;
        float pSH10 = fTmp1B * fS1;
        float pSH15 = -0.5900435899266435f * fC2;
        float pSH9  = -0.5900435899266435f * fS2;

        basis[ 9] = pSH9;
        basis[10] = pSH10;
        basis[11] = pSH11;
        basis[12] = pSH12;
        basis[13] = pSH13;
        basis[14] = pSH14;
        basis[15] = pSH15;
    }

    const uint32_t Kspatial = (deg + 1u) * (deg + 1u);

    // accumulate n = 0 block (static/DC)
    float result = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < Kspatial; ++k) {
        result += basis[k] * coeffs[k * 3u + c];
    }

    // temporal harmonics: cosine only
    // block n has offset n*Kspatial (n>=1)
    if (degree_t > 0u) {
        // scale argument: theta = 2*pi*n*(dir_t/time_duration)
        // guard against div by zero
        float inv_T = (time_duration > 0.0f) ? (1.0f / time_duration) : 0.0f;
#pragma unroll 1
        for (uint32_t n = 1u; n <= degree_t; ++n) {
            float theta = 2.0f * (float)kPI * (float)n * dir_t * inv_T;
            float tcos = cosf(theta);
            const uint32_t base = n * Kspatial;
#pragma unroll
            for (uint32_t k = 0u; k < Kspatial; ++k) {
                result += tcos * basis[k] * coeffs[(base + k) * 3u + c];
            }
        }
    }

    colors[c] = result;
}

template <typename scalar_t>
__device__ void sh4d_coeffs_to_color_fast_vjp(
    const uint32_t degree,        // spatial degree (0..3, will be clamped)
    const uint32_t degree_t,      // temporal degree (#cos harmonics, 0..N)
    const uint32_t c,             // channel 0..2
    const vec3 &dir,       // [3]
    const scalar_t *coeffs,       // [K, 3] with K = (degree+1)^2 * (degree_t+1)
    const scalar_t *v_colors,     // [3]
    // time inputs
    const scalar_t ts_val,        // ts[idx]
    const scalar_t timestamp,     // scalar
    const scalar_t time_duration, // scalar
    // outputs
    scalar_t *v_coeffs,           // [K, 3]
    vec3 *v_dir,           // [3] optional
    scalar_t *v_ts_scalar         // scalar contribution for this (elem,chan)
) {
    // Clamp degree to at most 3
    uint32_t deg = degree;
    if (deg > 3u) deg = 3u;

    // up to degree 3 (16 terms)
    float inorm = rsqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    float x = dir.x * inorm, y = dir.y * inorm, z = dir.z * inorm;

    // local accumulators
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;
    float v_ts_local = 0.f;
    const float v = v_colors[c];

    const uint32_t Kspatial = (deg + 1u) * (deg + 1u);

    auto apply_block = [&](uint32_t base, float scale, bool accumulate_vdir) {
        // l=0 (always present)
        v_coeffs[(base + 0)*3 + c] += scale * 0.2820947917738781f * v;
        // no dir grads at l=0

        if (deg < 1u) return;

        // l=1
        // coefficients
        v_coeffs[(base + 1)*3 + c] += scale * (-0.48860251190292f * y) * v;
        v_coeffs[(base + 2)*3 + c] += scale * ( 0.48860251190292f * z) * v;
        v_coeffs[(base + 3)*3 + c] += scale * (-0.48860251190292f * x) * v;

        if (accumulate_vdir && v_dir != nullptr) {
            v_x += scale * (-0.48860251190292f) * coeffs[(base + 3)*3 + c] * v;
            v_y += scale * (-0.48860251190292f) * coeffs[(base + 1)*3 + c] * v;
            v_z += scale * ( 0.48860251190292f) * coeffs[(base + 2)*3 + c] * v;
        }
        if (deg < 2u) return;

        // l=2
        float z2 = z*z;
        float fTmp0B = -1.092548430592079f * z;
        float fC1 = x*x - y*y;
        float fS1 = 2.f * x * y;
        float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
        float pSH7 = fTmp0B * x;
        float pSH5 = fTmp0B * y;
        float pSH8 = 0.5462742152960395f * fC1;
        float pSH4 = 0.5462742152960395f * fS1;

        v_coeffs[(base + 4)*3 + c] += scale * pSH4 * v;
        v_coeffs[(base + 5)*3 + c] += scale * pSH5 * v;
        v_coeffs[(base + 6)*3 + c] += scale * pSH6 * v;
        v_coeffs[(base + 7)*3 + c] += scale * pSH7 * v;
        v_coeffs[(base + 8)*3 + c] += scale * pSH8 * v;

        if (accumulate_vdir && v_dir != nullptr) {
            float fTmp0B_z = -1.092548430592079f;
            float fC1_x = 2.f * x, fC1_y = -2.f * y;
            float fS1_x = 2.f * y, fS1_y = 2.f * x;
            float pSH6_z = 2.f * 0.9461746957575601f * z;
            float pSH7_x = fTmp0B, pSH7_z = fTmp0B_z * x;
            float pSH5_y = fTmp0B, pSH5_z = fTmp0B_z * y;
            float pSH8_x = 0.5462742152960395f * fC1_x;
            float pSH8_y = 0.5462742152960395f * fC1_y;
            float pSH4_x = 0.5462742152960395f * fS1_x;
            float pSH4_y = 0.5462742152960395f * fS1_y;

            v_x += scale * v * ( pSH4_x * coeffs[(base + 4)*3 + c]
                                + pSH8_x * coeffs[(base + 8)*3 + c]
                                + pSH7_x * coeffs[(base + 7)*3 + c]);
            v_y += scale * v * ( pSH4_y * coeffs[(base + 4)*3 + c]
                                + pSH8_y * coeffs[(base + 8)*3 + c]
                                + pSH5_y * coeffs[(base + 5)*3 + c]);
            v_z += scale * v * ( pSH6_z * coeffs[(base + 6)*3 + c]
                                + pSH7_z * coeffs[(base + 7)*3 + c]
                                + pSH5_z * coeffs[(base + 5)*3 + c]);
        }
        if (deg < 3u) return;

        // l=3
        float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
        float fTmp1B = 1.445305721320277f * z;
        float fC2 = x * fC1 - y * fS1;
        float fS2 = x * fS1 + y * fC1;
        float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
        float pSH13 = fTmp0C * x;
        float pSH11 = fTmp0C * y;
        float pSH14 = fTmp1B * fC1;
        float pSH10 = fTmp1B * fS1;
        float pSH15 = -0.5900435899266435f * fC2;
        float pSH9  = -0.5900435899266435f * fS2;

        v_coeffs[(base +  9)*3 + c] += scale * pSH9  * v;
        v_coeffs[(base + 10)*3 + c] += scale * pSH10 * v;
        v_coeffs[(base + 11)*3 + c] += scale * pSH11 * v;
        v_coeffs[(base + 12)*3 + c] += scale * pSH12 * v;
        v_coeffs[(base + 13)*3 + c] += scale * pSH13 * v;
        v_coeffs[(base + 14)*3 + c] += scale * pSH14 * v;
        v_coeffs[(base + 15)*3 + c] += scale * pSH15 * v;

        if (accumulate_vdir && v_dir != nullptr) {
            float fTmp0C_z = -2.285228997322329f * 2.f * z;
            float fTmp1B_z = 1.445305721320277f;
            float fC1_x = 2.f * x, fC1_y = -2.f * y;
            float fS1_x = 2.f * y, fS1_y = 2.f * x;
            float fC2_x = fC1 + x * fC1_x - y * fS1_x;
            float fC2_y = x * fC1_y - fS1 - y * fS1_y;
            float fS2_x = fS1 + x * fS1_x + y * fC1_x;
            float fS2_y = x * fS1_y + fC1 + y * fC1_y;
            float pSH12_z = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
            float pSH13_x = fTmp0C, pSH13_z = fTmp0C_z * x;
            float pSH11_y = fTmp0C, pSH11_z = fTmp0C_z * y;
            float pSH14_x = fTmp1B * fC1_x;
            float pSH14_y = fTmp1B * fC1_y;
            float pSH14_z = fTmp1B_z * fC1;
            float pSH10_x = fTmp1B * fS1_x;
            float pSH10_y = fTmp1B * fS1_y;
            float pSH10_z = fTmp1B_z * fS1;
            float pSH15_x = -0.5900435899266435f * fC2_x;
            float pSH15_y = -0.5900435899266435f * fC2_y;
            float pSH9_x  = -0.5900435899266435f * fS2_x;
            float pSH9_y  = -0.5900435899266435f * fS2_y;

            v_x += scale * v * ( pSH9_x  * coeffs[(base +  9)*3 + c]
                                + pSH15_x * coeffs[(base + 15)*3 + c]
                                + pSH10_x * coeffs[(base + 10)*3 + c]
                                + pSH14_x * coeffs[(base + 14)*3 + c]
                                + pSH13_x * coeffs[(base + 13)*3 + c]);

            v_y += scale * v * ( pSH9_y  * coeffs[(base +  9)*3 + c]
                                + pSH15_y * coeffs[(base + 15)*3 + c]
                                + pSH10_y * coeffs[(base + 10)*3 + c]
                                + pSH14_y * coeffs[(base + 14)*3 + c]
                                + pSH11_y * coeffs[(base + 11)*3 + c]);

            v_z += scale * v * ( pSH12_z * coeffs[(base + 12)*3 + c]
                                + pSH13_z * coeffs[(base + 13)*3 + c]
                                + pSH11_z * coeffs[(base + 11)*3 + c]
                                + pSH14_z * coeffs[(base + 14)*3 + c]
                                + pSH10_z * coeffs[(base + 10)*3 + c]);
        }
    };

    // ---- apply static block (n=0) ----
    apply_block(/*base=*/0u, /*scale=*/1.0f, /*accumulate_vdir=*/true);

    // ---- temporal blocks (n>=1) ----
    if (degree_t > 0u) {
        const float Tinv = (time_duration > 0.0f) ? (1.0f / time_duration) : 0.0f;
        const float dir_t = ts_val - timestamp;

        for (uint32_t n = 1; n <= degree_t; ++n) {
            const uint32_t base = n * Kspatial;
            float theta = 2.0f * (float)kPI * (float)n * dir_t * Tinv;
            float c_n = cosf(theta);
            float s_n = sinf(theta);

            // 1) coeffs grads + dir grads scaled by c_n
            apply_block(base, c_n, /*accumulate_vdir=*/(v_dir != nullptr));

            // 2) ts grad via d/dts cos(theta) = -sin(theta) * dtheta/dts
            // dtheta/dts = 2π n / T
            if (v_ts_scalar != nullptr && time_duration > 0.0f) {
                float sum_block = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < Kspatial; ++k) {
                    // Recompute scalar SH basis b_k at this (x,y,z), degree<=3
                    float b = 0.f;
                    if (k == 0) {
                        b = 0.2820947917738781f;
                    } else if (deg >= 1u && k <= 3u) {
                        if      (k == 1u) b = -0.48860251190292f * y;
                        else if (k == 2u) b =  0.48860251190292f * z;
                        else             b = -0.48860251190292f * x;
                    } else if (deg >= 2u && k <= 8u) {
                        float z2 = z*z;
                        float fTmp0B = -1.092548430592079f * z;
                        float fC1 = x*x - y*y;
                        float fS1 = 2.f * x * y;
                        if      (k == 4u) b = 0.5462742152960395f * fS1;
                        else if (k == 5u) b = fTmp0B * y;
                        else if (k == 6u) b = 0.9461746957575601f * z2 - 0.3153915652525201f;
                        else if (k == 7u) b = fTmp0B * x;
                        else              b = 0.5462742152960395f * fC1;
                    } else if (deg >= 3u && k <= 15u) {
                        float z2 = z*z;
                        float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                        float fTmp1B = 1.445305721320277f * z;
                        float fC1 = x*x - y*y;
                        float fS1 = 2.f * x * y;
                        float fC2 = x * fC1 - y * fS1;
                        float fS2 = x * fS1 + y * fC1;
                        if      (k ==  9u) b = -0.5900435899266435f * fS2;
                        else if (k == 10u) b =  fTmp1B * fS1;
                        else if (k == 11u) b =  fTmp0C * y;
                        else if (k == 12u) { float z2a=z*z; b = z*(1.865881662950577f*z2a - 1.119528997770346f); }
                        else if (k == 13u) b =  fTmp0C * x;
                        else if (k == 14u) b =  fTmp1B * fC1;
                        else               b = -0.5900435899266435f * fC2;
                    }
                    sum_block += b * coeffs[(base + k)*3 + c];
                }
                float dtheta_dts = (time_duration > 0.0f)
                                       ? (2.0f * (float)kPI * (float)n / time_duration)
                                       : 0.0f;
                v_ts_local += (-s_n) * dtheta_dts * v * sum_block;
            }
        }
    }

    // write v_dir (project back through normalization), if requested
    if (v_dir != nullptr) {
        vec3 dir_n{x, y, z};
        vec3 v_dir_n{v_x, v_y, v_z};
        vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;
        v_dir->x = v_d.x; v_dir->y = v_d.y; v_dir->z = v_d.z;
    }

    if (v_ts_scalar != nullptr) {
        *v_ts_scalar = v_ts_local;
    }
}

template <typename scalar_t>
__global__ void spherical_harmonics4d_fwd_kernel(
    const uint32_t N,
    const uint32_t K,                 // must equal (deg+1)^2 * (deg_t+1)
    const uint32_t degrees_spatial,   // spatial degree
    const uint32_t degrees_temporal,  // temporal degree (cos-only)
    const vec3 *__restrict__ dirs, // [N, 3]
    const scalar_t *__restrict__ ts,         // [N] per-point time (e.g., gaussian ts)
    const scalar_t *__restrict__ timestamps,  // [N]
    const scalar_t time_duration,            // scalar
    const scalar_t *__restrict__ coeffs,     // [N, K, 3]
    const bool *__restrict__ masks,   // [N] (optional)
    scalar_t *__restrict__ colors            // [N, 3]
) {
    // parallelize over N * 3 (channels)
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) return;

    uint32_t elem_id = idx / 3;
    uint32_t c = idx % 3;

    if (masks != nullptr && !masks[elem_id]) {
        return;
    }

    const vec3 d = dirs[elem_id];
    const scalar_t dir_t = ts[elem_id] - timestamps[elem_id];

    sh4d_coeffs_to_color_fast(
        degrees_spatial,
        degrees_temporal,
        c,
        d,
        dir_t,
        time_duration,
        coeffs + elem_id * K * 3,
        colors + elem_id * 3
    );
}
void launch_spherical_harmonics4d_fwd_kernel(
    // inputs
    const uint32_t degrees_spatial,
    const uint32_t degrees_temporal,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::Tensor ts,                 // [... ]
    const at::Tensor timestamps,        // [... ]
    const float time_duration,           // scalar
    const at::optional<at::Tensor> masks, // [...]
    // outputs
    at::Tensor colors // [..., 2]
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;

    // parallelize over N * 3
    int64_t n_elements = N * 3;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        dirs.scalar_type(),
        "spherical_harmonics4d_fwd_kernel",
        [&]() {
            spherical_harmonics4d_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    K,
                    degrees_spatial,
                    degrees_temporal,
                    reinterpret_cast<vec3 *>(dirs.data_ptr<scalar_t>()),
                    ts.data_ptr<scalar_t>(),
                    timestamps.data_ptr<scalar_t>(),
                    time_duration,
                    coeffs.data_ptr<scalar_t>(),
                    masks.has_value() ? masks.value().data_ptr<bool>()
                                      : nullptr,
                    colors.data_ptr<scalar_t>()
                );
        }
    );
}

template <typename scalar_t>
__global__ void spherical_harmonics4d_bwd_kernel(
    const uint32_t N,
    const uint32_t K,                    // (degree+1)^2 * (degree_t+1)
    const uint32_t degree_spatial,
    const uint32_t degree_temporal,
    const vec3 *__restrict__ dirs,    // [N, 3]
    const scalar_t *__restrict__ ts,            // [N]
    const scalar_t *__restrict__ timestamps,    // [N]
    const scalar_t time_duration,               // scalar
    const scalar_t *__restrict__ coeffs,        // [N, K, 3]
    const bool *__restrict__ masks,      // [N] optional
    const scalar_t *__restrict__ v_colors,      // [N, 3]
    // outputs
    scalar_t *__restrict__ v_coeffs,            // [N, K, 3]
    scalar_t *__restrict__ v_dirs,              // [N, 3] optional
    scalar_t *__restrict__ v_ts
)  {
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) return;

    uint32_t elem_id = idx / 3;
    uint32_t c = idx % 3;

    if (masks != nullptr && !masks[elem_id]) {
        return;
    }

    vec3 v_dir_local = {0.f, 0.f, 0.f};
    scalar_t v_ts_local = 0.f;

    sh4d_coeffs_to_color_fast_vjp(
        degree_spatial,
        degree_temporal,
        c,
        dirs[elem_id],
        coeffs + elem_id * K * 3,
        v_colors + elem_id * 3,
        // time inputs
        ts[elem_id],
        timestamps[elem_id],
        time_duration,
        // outputs
        v_coeffs + elem_id * K * 3,
        v_dirs == nullptr ? nullptr : &v_dir_local,
        v_ts   == nullptr ? nullptr : &v_ts_local
    );

    if (v_dirs != nullptr) {
        gpuAtomicAdd(v_dirs + elem_id * 3 + 0, v_dir_local.x);
        gpuAtomicAdd(v_dirs + elem_id * 3 + 1, v_dir_local.y);
        gpuAtomicAdd(v_dirs + elem_id * 3 + 2, v_dir_local.z);
    }
    if (v_ts != nullptr) {
        gpuAtomicAdd(v_ts + elem_id, v_ts_local);
    }
}

void launch_spherical_harmonics4d_bwd_kernel(
    // inputs
    const uint32_t degree_spatial,
    const uint32_t degree_temporal,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::Tensor ts,                 // [... ]
    const at::Tensor timestamps,        // [... ]
    const float time_duration,           // scalar
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    // outputs
    at::Tensor v_coeffs,                // [..., K, 3]
    at::optional<at::Tensor> v_dirs,    // [..., 3]
    at::Tensor v_ts                     // [...]
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;

    // parallelize over N * 3
    int64_t n_elements = N * 3;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        dirs.scalar_type(),
        "spherical_harmonics4d_bwd_kernel",
        [&]() {
            spherical_harmonics4d_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    K,
                    degree_spatial,
                    degree_temporal,
                    reinterpret_cast<vec3 *>(dirs.data_ptr<scalar_t>()),
                    ts.data_ptr<scalar_t>(),
                    timestamps.data_ptr<scalar_t>(),
                    time_duration,
                    coeffs.data_ptr<scalar_t>(),
                    masks.has_value() ? masks.value().data_ptr<bool>()
                                      : nullptr,
                    v_colors.data_ptr<scalar_t>(),
                    v_coeffs.data_ptr<scalar_t>(),
                    v_dirs.has_value() ? v_dirs.value().data_ptr<scalar_t>()
                                       : nullptr,
                    v_ts.data_ptr<scalar_t>()
                );
        }
    );
}

} // namespace gsplat
