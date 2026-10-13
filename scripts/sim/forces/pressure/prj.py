"""PRJ solver (Projected Jacobi): constant-density pass (IISPH) and optional
divergence-free pass (DFSPH)."""

import time
import warp as wp
from .pressure_base import PressureBase, compute_active_mask
from scripts.sim.math_utils import (
    add_vec3, scale_vec3,
    add_scalar, coef_wise_mul, coef_wise_div, div_vec3_by_scalar, max_zero
)


# ============================================================================
# PRJ Kernels
# ============================================================================

@wp.kernel
def pj_head_fused(
    dt2_lam: wp.array(dtype=float),
    m: wp.array(dtype=float),
    dt_v_0: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    dt_v: wp.array(dtype=wp.vec3),
):
    """Fused head of one PRJ iteration:  dt_v^l = dt_v_0 − Δt² J^T λ / m."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        return

    dt2_lam_i = dt2_lam[p_i]
    m_i = m[p_i]
    dt2_JT_lam_i = wp.vec3(float(0.0), float(0.0), float(0.0))

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        if is_dynamic[p_j] == 1:
            dt2_JT_lam_i += (m[p_j] * dt2_lam_i + m_i * dt2_lam[p_j]) * grad_W
        else:
            dt2_JT_lam_i += (m[p_j] * dt2_lam_i) * grad_W

    denom = m_i
    if denom < 1e-6:
        denom = 1e-6
    dt_v[p_i] = dt_v_0[p_i] - dt2_JT_lam_i / denom


@wp.kernel
def pj_tail_fused(
    dt_v: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=float),
    D_inv: wp.array(dtype=float),
    m: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    dt2_lam: wp.array(dtype=float),
    omega: float,
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    result_buf: wp.array(dtype=float),
    cnt_buf: wp.array(dtype=float),
):
    """Fused tail of one PRJ iteration: r_i = c_i + Δt (J v^l)_i,
    Δt²λ_i ← max(Δt²λ_i + ω D⁻¹_i r_i, 0), and accumulates avg(max(r_i / ρ₀_i, 0))."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        return

    # dt_Jv_i = sum_j m_j (dt_v_i − dt_v_j) · ∇W_ij
    dt_v_i = dt_v[p_i]
    dt_Jv_i = float(0.0)
    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        if is_dynamic[p_j] == 1:
            dt_Jv_i += m[p_j] * wp.dot(dt_v_i - dt_v[p_j], grad_W)
        else:
            dt_Jv_i += m[p_j] * wp.dot(dt_v_i, grad_W)

    # compute r_i = dt_Jv_i + c_i
    r_i = dt_Jv_i + c[p_i]
    # cumulative Δt² λ update + non-negativity clamp
    dt2_lam[p_i] = wp.max(dt2_lam[p_i] + omega * r_i * D_inv[p_i], 0.0)

    # density-error contribution: avg(max(r_i / ρ₀_i, 0))
    rho0 = density0[p_i]
    if rho0 < 1e-6:
        rho0 = 1e-6
    wp.atomic_add(result_buf, 0, wp.max(r_i / rho0, 0.0))
    wp.atomic_add(cnt_buf, 0, 1.0)


# ============================================================================
# PRJ Class
# ============================================================================

class PRJ(PressureBase):
    """Projected Jacobi solver for the constant-density constraint (IISPH-style)."""

    def __init__(self, particle_system, neighbour_search, config):
        super().__init__(particle_system, neighbour_search, config)

        self.outer_iter = 0
        self.active = wp.zeros(self.ps.particle_max_num, dtype=float)
        self.warm_start = False
        self.omega = 0.5  # PRJ relaxation step size (conservative default)

        # DF (divergence-free) pass: off by default; only PRJ has one.
        self.enable_df = False
        self.df_num_threshold = 25
        self.df_tol_opt = 1e-2
        self.stats_df_outer_iters = 0
        self.stats_df_total_cost_ms = 0.0
        # Cumulative DF Lagrange multipliers (Δt²-scaled λ_df).
        self.dt2_lam_df = wp.zeros(self.ps.particle_max_num, dtype=float)
        self.dt2_lam_df_previous = wp.zeros(self.ps.particle_max_num, dtype=float)

    def solve_PJ(self):
        # Aliases: D_inv = D⁻¹, dt_v = Δt v, dt2_lam = Δt² λ (PressureBase buffers).
        D_inv     = self.k           # D⁻¹
        c         = self.c           # constraint
        dt_v_0    = self.s           # Δt · v^0 (v^*)
        dt_v      = self.dx          # Δt · v^l
        dt2_lam   = self.p           # Δt² · cumulative λ

        wp.copy(dt_v, dt_v_0)
        # outer_iter counts updates applied to the velocity, not loop passes: each
        # pass's head applies the lambda accumulated by the earlier passes, so the
        # first pass only measures the initial residual.
        self.outer_iter = 0
        self.reset_stats()

        for n_pass in range(self.max_iteration_opt + 1):
            t_iter = time.perf_counter()
            self.outer_iter = n_pass

            # dt_v = dt_v_0 − dt^2 J^T λ / m
            wp.launch(
                kernel=pj_head_fused,
                dim=self.ps.particle_num,
                inputs=[
                    dt2_lam, self.ps.m, dt_v_0, self.ps.is_dynamic,
                    self.ps.particle_num,
                    int(self.ps.cache_size),
                    self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW,
                ],
                outputs=[dt_v],
            )

            # dt_Jv -> r=dt_Jv+c -> λ update -> clamp -> (/ρ₀) -> reduction
            self.result_buffer.zero_()
            self.cnt_buffer.zero_()
            wp.launch(
                kernel=pj_tail_fused,
                dim=self.ps.particle_num,
                inputs=[
                    dt_v, c, D_inv, self.ps.m, self.ps.density0, dt2_lam,
                    self.omega, self.ps.is_dynamic, self.ps.particle_num,
                    int(self.ps.cache_size),
                    self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW,
                    self.result_buffer, self.cnt_buffer,
                ],
            )
            err = self.read_avg_error()
            wp.synchronize()
            self.stats_density_errors.append(err)
            self.stats_iter_costs.append((time.perf_counter() - t_iter) * 1000.0)

            converged = (err < self.tol_opt) and (not self.use_fixed_iters)
            if (converged and self.outer_iter >= 1) or self.outer_iter >= self.max_iteration_opt:
                wp.copy(self.p_previous, dt2_lam)
                break

        self.stats_outer_iters = self.outer_iter
        self.stats_final_density_error = err
        self.check_converged(err, self.outer_iter)

    def solve_df(self, dt):
        """Divergence-free second projection: enforce Jv ≤ 0 (DFSPH-style).

        Same structure as solve_PJ but with constraint Jv (no `c` term),
        masked by `self.active` (particles with enough neighbors).
        `self.dt2_lam_df` is the Δt²-scaled cumulative λ for this pass.
        """
        t_df = time.perf_counter()
        n = self.ps.particle_num

        # Aliases (see solve_PJ)
        D_inv     = self.k
        dt_v_0    = self.s
        dt_v      = self.dx
        dt_Jv     = self.Jx
        dt2_JT_lam    = self.tmp
        D_inv_r   = self.dp
        dt2_lam_df    = self.dt2_lam_df

        # Compute active mask once per DF solve
        wp.launch(kernel=compute_active_mask, dim=n,
                  inputs=[self.ps.neighbor_count, self.ps.is_dynamic,
                          n, self.df_num_threshold],
                  outputs=[self.active])

        # dt_v_0 = dt v
        wp.launch(kernel=scale_vec3, dim=n,
                  inputs=[dt_v_0, dt, self.ps.v, self.ps.is_dynamic, n])

        # λ_df: cold (zero) or warm from prior-frame DF result
        # warm start: dt2_lam_df = 0.5 * dt2_lam_df_previous
        dt2_lam_df.zero_()
        if self.warm_start:
            wp.launch(
                kernel=add_scalar,
                dim=n,
                inputs=[dt2_lam_df, dt2_lam_df, 0.5, self.dt2_lam_df_previous, self.ps.is_dynamic, n],
            )

        wp.copy(dt_v, dt_v_0)

        # df_iter = updates applied so far (same convention as solve_PJ).
        for df_iter in range(self.max_iteration_opt + 1):
            # dt2_JT_lam = J^T λ_df
            self.compute_J_tr_x(dt2_JT_lam, dt2_lam_df)

            # dt2_JT_lam = dt2_JT_lam / m
            wp.launch(kernel=div_vec3_by_scalar, dim=n,
                      inputs=[dt2_JT_lam, dt2_JT_lam, self.ps.m, self.ps.is_dynamic, n])

            # dt_v = dt_v_0 − dt2_JT_lam
            wp.launch(kernel=add_vec3, dim=n,
                      inputs=[dt_v, dt_v_0, -1.0, dt2_JT_lam, self.ps.is_dynamic, n])

            # compute dt_Jv, masked by active set
            self.compute_J_x(dt_Jv, dt_v)
            wp.launch(kernel=coef_wise_mul, dim=n,
                      inputs=[dt_Jv, dt_Jv, self.active, self.ps.is_dynamic, n])

            # D_inv_r = D_inv dt_Jv
            wp.launch(kernel=coef_wise_mul, dim=n,
                      inputs=[D_inv_r, dt_Jv, D_inv, self.ps.is_dynamic, n])

            # λ_df ← λ_df + ω · D_inv_r
            wp.launch(kernel=add_scalar, dim=n,
                      inputs=[dt2_lam_df, dt2_lam_df, self.omega, D_inv_r, self.ps.is_dynamic, n])

            # λ_df ← max(λ_df, 0)
            wp.launch(kernel=max_zero, dim=n,
                      inputs=[dt2_lam_df, self.ps.is_dynamic, n])

            # Convergence: avg(max(dt_Jv / ρ₀, 0))
            wp.launch(kernel=coef_wise_div, dim=n,
                      inputs=[self.t, dt_Jv, self.ps.density0, self.ps.is_dynamic, n])
            err = self.compute_avg_density_error(self.t)

            df_converged = (err < self.df_tol_opt) and (not self.use_fixed_iters)
            if (df_converged and df_iter >= 1) or df_iter >= self.max_iteration_opt:
                wp.copy(self.dt2_lam_df_previous, dt2_lam_df)
                break

        # v = dt_v / dt, and the divergence for the visualizer
        wp.launch(kernel=scale_vec3, dim=n,
                  inputs=[self.ps.v, 1.0 / dt, dt_v, self.ps.is_dynamic, n])
        self.compute_divergence()

        wp.synchronize()
        self.stats_df_outer_iters = df_iter
        self.stats_df_total_cost_ms = (time.perf_counter() - t_df) * 1000.0

    def solve(self, dt):
        """Main PRJ solve."""
        t_frame = time.perf_counter()

        # Aliases (see solve_PJ)
        dt_v_0    = self.s
        dt_v      = self.dx
        dt2_lam   = self.p           # Δt² · cumulative λ

        # c = ρ − ρ₀
        self.compute_constraint()

        # dt_v_0 = dt v
        wp.launch(
            kernel=scale_vec3,
            dim=self.ps.particle_num,
            inputs=[dt_v_0, dt, self.ps.v, self.ps.is_dynamic, self.ps.particle_num],
            outputs=[]
        )

        # Initialize cumulative λ (cold or warm-started)
        # warm start: dt2_lam = 0.5 * self.p_previous
        dt2_lam.zero_()
        if self.warm_start:
            wp.launch(
                kernel=add_scalar,
                dim=self.ps.particle_num,
                inputs=[dt2_lam, dt2_lam, 0.5, self.p_previous, self.ps.is_dynamic, self.ps.particle_num],
                outputs=[]
            )

        self.solve_PJ()

        self.stats_outer_iters = self.outer_iter

        # v = dt_v / dt
        wp.launch(
            kernel=scale_vec3,
            dim=self.ps.particle_num,
            inputs=[self.ps.v, 1.0 / dt, dt_v, self.ps.is_dynamic, self.ps.particle_num],
            outputs=[]
        )

        # Final density error with hard max: avg(max((c + Δt J v_final)/ρ_0, 0))
        self.stats_final_density_error = self.final_density_error(dt_v)

        wp.synchronize()
        self.stats_total_cost_ms = (time.perf_counter() - t_frame) * 1000.0

        # Update divergence for visualization: Jv (post-correction, unscaled v)
        self.compute_divergence()
