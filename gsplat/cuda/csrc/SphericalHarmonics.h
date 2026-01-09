#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    // outputs
    at::Tensor colors // [..., 2]
);

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    // outputs
    at::Tensor v_coeffs,            // [..., K, 3]
    at::optional<at::Tensor> v_dirs // [..., 3]
);

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
);

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
);

} // namespace gsplat