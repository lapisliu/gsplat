#pragma once

#include "Common.h"
#include "Utils.cuh"

namespace gsplat {

inline __device__ void pack_sym3_to_six(const mat3& G, float out6[6]) {
    out6[0] = G[0][0];                    // xx
    out6[1] = G[0][1] + G[1][0];          // xy (doubled)
    out6[2] = G[0][2] + G[2][0];          // xz (doubled)
    out6[3] = G[1][1];                    // yy
    out6[4] = G[1][2] + G[2][1];          // yz (doubled)
    out6[5] = G[2][2];                    // zz
}


inline __device__ vec4 quat_normalize(const vec4& q) {
    float inv_norm = rsqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return vec4(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm, q.w * inv_norm);
}

inline __device__ void quat_normalize_vjp(
    const vec4& q,           // unnormalized input
    const vec4& v_q_norm,    // upstream grad w.r.t. normalized q
    vec4&       v_q_accum    // accumulate into grad w.r.t. original q
){
    float inv_norm = rsqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    vec4 qn = vec4(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm, q.w * inv_norm);
    float dot = glm::dot(v_q_norm, qn);
    vec4 proj = vec4(
        v_q_norm.x - dot * qn.x,
        v_q_norm.y - dot * qn.y,
        v_q_norm.z - dot * qn.z,
        v_q_norm.w - dot * qn.w
    );
    v_q_accum += proj * inv_norm;
}

inline __device__ void computeCov3D_conditional(
    const vec3  scale,
    const float scale_t,
    float       mod,
    const vec4  rot,
    const vec4  rot_r,
    mat3*       cov3D,
    vec3&       p_orig,
    const float t,
    const float timestamp,
    float&      marginal_t
){
    // scaling matrix
    float dt = timestamp - t;
    mat4 S(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;
    S[3][3] = mod * scale_t;

    // Normalize quaternions
    vec4 rot_n   = quat_normalize(rot);
    vec4 rot_r_n = quat_normalize(rot_r);

    float a = rot_n.x, b = rot_n.y, c = rot_n.z, d = rot_n.w;
    float p = rot_r_n.x, q = rot_r_n.y, r = rot_r_n.z, s = rot_r_n.w;

    mat4 M_l(
        a,  b, -c,  d,
        -b,  a,  d,  c,
        c, -d,  a,  b,
        -d, -c, -b,  a
    );

    mat4 M_r(
        p,  q, -r, -s,
        -q,  p,  s, -r,
        r, -s,  p, -q,
        s,  r,  q,  p
    );

    mat4 R     = M_r * M_l;
    mat4 M     = S * R;
    mat4 Sigma = glm::transpose(M) * M;

    // Clamp denominator to avoid blow-ups
    float cov_t      = Sigma[3][3];
    float cov_t_safe = glm::max(cov_t, 1e-12f);
    float expo       = -0.5f * (dt * dt / cov_t_safe);
    marginal_t       = __expf(expo);

    mat3  cov11 = mat3(Sigma);
    vec3  cov12 = vec3(Sigma[0][3], Sigma[1][3], Sigma[2][3]);

    *cov3D = cov11 - (glm::outerProduct(cov12, cov12) / cov_t_safe);

    vec3 delta_mean = (cov12 / cov_t_safe) * dt;
    p_orig.x += delta_mean.x;
    p_orig.y += delta_mean.y;
    p_orig.z += delta_mean.z;
}

inline __device__ bool computeCov3D_conditional_bwd(
    // forward inputs
    const vec3& scale,          // scale.xyz
    const float scale_t,        // temporal scale
    const float mod,            // typically 1
    const vec4& rot,            // left quat
    const vec4& rot_r,          // right quat
    const float t,              // gaussian time
    const float timestamp,      // camera time
    const float opacity,        // alpha

    // upstream grads w.r.t. outputs of the forward “conditional” block:
    const float d_cov3D[6],     // Schur 3x3 as xx,xy,xz,yy,yz,zz
    const vec3& d_mean_w_shifted,
    const float d_weighted_opacity_in,

    // outputs (accumulate into these)
    float& d_opacity,   // += dL/d alpha
    float& d_t_out,     // += dL/d t (gaussian time)
    vec3&  d_scale,     // += dL/d scale.xyz
    float& d_scale_t,   // += dL/d scale_t
    vec4&  d_rot,       // += dL/d rot (left quat)
    vec4&  d_rot_r      // += dL/d rot_r (right quat)
) {
    const float dt = timestamp - t;

    mat4 S(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;
    S[3][3] = mod * scale_t;

    // Normalize quaternions
    const vec4 rot_n   = quat_normalize(rot);
    const vec4 rot_r_n = quat_normalize(rot_r);

    const float a = rot_n.x, b = rot_n.y, c = rot_n.z, d = rot_n.w;
    const float p = rot_r_n.x, q = rot_r_n.y, r = rot_r_n.z, s = rot_r_n.w;

    mat4 M_l(
        a,  b, -c,  d,
        -b,  a,  d,  c,
        c, -d,  a,  b,
        -d, -c, -b,  a
    );

    mat4 M_r(
        p,  q, -r, -s,
        -q,  p,  s, -r,
        r, -s,  p, -q,
        s,  r,  q,  p
    );

    const mat4 R     = M_r * M_l;
    const mat4 M     = S * R;
    const mat4 Sigma = glm::transpose(M) * M;

    const float cov_t     = Sigma[3][3];
    const float inv_cov_t = 1.0f / glm::max(cov_t, 1e-12f);
    const float marginal_t = __expf(-0.5f * (dt * dt * inv_cov_t));

    const bool mask = (marginal_t > 0.05f);
    if (!mask) {
        return false;
    }

    const vec3 cov12(Sigma[0][3], Sigma[1][3], Sigma[2][3]);

    // ----- Backprop to cov12 and cov_t from d_cov3D (Schur complement) -----
    vec3  g_cov12(0.0f);
    float g_cov_t = 0.0f;

    const float d_xx = d_cov3D[0];
    const float d_xy = d_cov3D[1];
    const float d_xz = d_cov3D[2];
    const float d_yy = d_cov3D[3];
    const float d_yz = d_cov3D[4];
    const float d_zz = d_cov3D[5];

    g_cov12 = -vec3(
                  d_xx * cov12.x + d_xy * cov12.y * 0.5f + d_xz * cov12.z * 0.5f,
                  d_xy * cov12.x * 0.5f + d_yy * cov12.y + d_yz * cov12.z * 0.5f,
                  d_xz * cov12.x * 0.5f + d_yz * cov12.y * 0.5f + d_zz * cov12.z
              ) * (2.0f * inv_cov_t);

    g_cov_t += ( cov12.x * cov12.x * d_xx
                + cov12.x * cov12.y * d_xy
                + cov12.x * cov12.z * d_xz
                + cov12.y * cov12.y * d_yy
                + cov12.y * cov12.z * d_yz
                + cov12.z * cov12.z * d_zz ) * (inv_cov_t * inv_cov_t);

    // ----- Opacity path: weighted_opacity = alpha * marginal_t -----
    const float d_marginal_t = d_weighted_opacity_in * opacity;
    d_opacity += d_weighted_opacity_in * marginal_t;

    // marginal_t = exp(-0.5 * dt^2 / cov_t)
    const float dmarginal_dcovt = marginal_t * (dt * dt) * 0.5f * (inv_cov_t * inv_cov_t);
    const float dmarginal_ddt   = -marginal_t * dt * inv_cov_t;
    g_cov_t += dmarginal_dcovt * d_marginal_t;
    float g_dt = dmarginal_ddt * d_marginal_t;

    // ----- Mean shift path: delta_mean = (cov12 / cov_t) * dt -----
    g_cov12 += d_mean_w_shifted * (dt * inv_cov_t);
    g_cov_t  += -glm::dot(d_mean_w_shifted, cov12) * dt * (inv_cov_t * inv_cov_t);
    g_dt     +=  glm::dot(d_mean_w_shifted, cov12) * inv_cov_t;

    // propagate dt to t (gaussian time): dt = timestamp - t
    d_t_out += -g_dt;

    // ----- Build dL/dSigma from pieces -----
    mat4 g_Sigma(0.0f);
    g_Sigma[0][0] = d_xx;
    g_Sigma[0][1] = 0.5f * d_xy; g_Sigma[1][0] = g_Sigma[0][1];
    g_Sigma[0][2] = 0.5f * d_xz; g_Sigma[2][0] = g_Sigma[0][2];
    g_Sigma[1][1] = d_yy;
    g_Sigma[1][2] = 0.5f * d_yz; g_Sigma[2][1] = g_Sigma[1][2];
    g_Sigma[2][2] = d_zz;

    g_Sigma[0][3] = 0.5f * g_cov12.x; g_Sigma[3][0] = g_Sigma[0][3];
    g_Sigma[1][3] = 0.5f * g_cov12.y; g_Sigma[3][1] = g_Sigma[1][3];
    g_Sigma[2][3] = 0.5f * g_cov12.z; g_Sigma[3][2] = g_Sigma[2][3];
    g_Sigma[3][3] = g_cov_t;

    // ----- Backprop Sigma = M^T M  to M -----
    mat4 g_M = 2.0f * M * g_Sigma;

    // Split M = S * R ...
    const mat4 Rt  = glm::transpose(R);
    const mat4 g_Mt = glm::transpose(g_M);

    d_scale.x += glm::dot(Rt[0], g_Mt[0]) * mod;
    d_scale.y += glm::dot(Rt[1], g_Mt[1]) * mod;
    d_scale.z += glm::dot(Rt[2], g_Mt[2]) * mod;
    d_scale_t += glm::dot(Rt[3], g_Mt[3]) * mod;

    mat4 g_Mt_scaled = g_Mt;
    g_Mt_scaled[0] *= (mod * scale.x);
    g_Mt_scaled[1] *= (mod * scale.y);
    g_Mt_scaled[2] *= (mod * scale.z);
    g_Mt_scaled[3] *= (mod * scale_t);

    const mat4 g_ml = g_Mt_scaled * M_r; // grad wrt M_l
    const mat4 g_mr = M_l * g_Mt_scaled; // grad wrt M_r

    vec4 d_rot_norm(0.0f), d_rot_r_norm(0.0f);

    // Quaternion left (rot) mapping
    d_rot_norm.x += g_ml[0][0] + g_ml[1][1] + g_ml[2][2] + g_ml[3][3];
    d_rot_norm.y += -g_ml[0][1] + g_ml[1][0] - g_ml[2][3] + g_ml[3][2];
    d_rot_norm.z +=  g_ml[0][2] - g_ml[1][3] - g_ml[2][0] + g_ml[3][1];
    d_rot_norm.w += -g_ml[0][3] - g_ml[1][2] + g_ml[2][1] + g_ml[3][0];

    // Quaternion right (rot_r) mapping
    d_rot_r_norm.x += g_mr[0][0] + g_mr[1][1] + g_mr[2][2] + g_mr[3][3];
    d_rot_r_norm.y += -g_mr[0][1] + g_mr[1][0] + g_mr[2][3] - g_mr[3][2];
    d_rot_r_norm.z +=  g_mr[0][2] + g_mr[1][3] - g_mr[2][0] - g_mr[3][1];
    d_rot_r_norm.w +=  g_mr[0][3] - g_mr[1][2] + g_mr[2][1] - g_mr[3][0];

    quat_normalize_vjp(rot,   d_rot_norm,   d_rot);
    quat_normalize_vjp(rot_r, d_rot_r_norm, d_rot_r);

    return true; // mask passed
}

} // namespace gsplat
