"""PBF solver (position-based)."""

import time
import warp as wp
from .pressure_base import PressureBase
from scripts.sim.sph_kernel import cubic_kernel


# ============================================================================
# PBF Kernels
# ============================================================================

@wp.kernel
def compute_lambda(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    m_inv: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    support_radius: float,
    omega: float,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=float),
    c: wp.array(dtype=float),
    lda: wp.array(dtype=float),
):
    """Density, constraint c = max(rho-rho0,0) and lambda = omega*c/(Aii+1e-6) with omega
    folded in (downstream applies a bare J^T lambda), Aii = (J M^-1 J^T)_ii."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    lda[p_i] = 0.0
    c[p_i] = 0.0

    if is_dynamic[p_i] == 0:
        return

    pos_i = x[p_i]
    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    # Compute density
    rho = m[p_i] * cubic_kernel(0.0, support_radius)

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        r_norm = wp.length(pos_i - x[p_j])
        rho += m[p_j] * cubic_kernel(r_norm, support_radius)

    density[p_i] = rho

    # Compute constraint
    constraint = rho - density0[p_i]
    c[p_i] = wp.max(constraint, 0.0)

    # Compute Aii for the lambda denominator
    Aii = float(0.0)
    J_ii = wp.vec3(float(0.0), float(0.0), float(0.0))

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        J_ij = m[p_j] * grad_W
        if is_dynamic[p_j] == 1:
            Aii += wp.dot(J_ij, J_ij) * m_inv[p_j]
        J_ii -= J_ij

    Aii += wp.dot(J_ii, J_ii) * m_inv[p_i]

    # Regularization: k=1e6 rescales lda = max(c,0)/(Aii+1e-6) to avoid the tiny eps.
    k = 1e6
    lda[p_i] = omega * k * wp.max(c[p_i], 0.0) / (k * Aii + 1.0)


@wp.kernel
def pbf_head_fused(
    lda: wp.array(dtype=float),
    m: wp.array(dtype=float),
    m_inv: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    x: wp.array(dtype=wp.vec3),
):
    """Fused PBF position update in one launch: sol = -m_inv * J^T * lda; x += sol."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        return

    lda_i = lda[p_i]
    m_i = m[p_i]
    JT_lda = wp.vec3(float(0.0), float(0.0), float(0.0))

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    # compute J^T λ
    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        if is_dynamic[p_j] == 1:
            JT_lda += (m[p_j] * lda_i + m_i * lda[p_j]) * grad_W
        else:
            JT_lda += (m[p_j] * lda_i) * grad_W

    # sol = -M_inv * J^T λ ; x += sol
    sol_i = (-1.0) * m_inv[p_i] * JT_lda
    x[p_i] = x[p_i] + sol_i


@wp.kernel
def pbf_tail_fused(
    c: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    result_buf: wp.array(dtype=float),
    cnt_buf: wp.array(dtype=float),
):
    """Avg density error: atomic-add reduction of max(c/rho0, 0) + count."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        return

    rho0 = density0[p_i]
    if rho0 < 1e-6:
        rho0 = 1e-6
    c_norm = c[p_i] / rho0
    wp.atomic_add(result_buf, 0, wp.max(c_norm, 0.0))
    wp.atomic_add(cnt_buf, 0, 1.0)


# ============================================================================
# PBF Class
# ============================================================================

class PBF(PressureBase):
    """Position-Based Fluids solver (Jacobi position update)."""

    is_position_based = True

    def __init__(self, particle_system, neighbour_search, config):
        super().__init__(particle_system, neighbour_search, config)

        # PBF uses its own relaxation factor; the scene's `omega` is IP's.
        self.omega = 0.5

        max_particles = self.ps.particle_max_num
        self.lda = wp.zeros(max_particles, dtype=float)  # lambda buffer

    def _update_positions(self):
        """One Jacobi update: x += -M^-1 J^T lda (omega already folded into lda)."""
        wp.launch(
            kernel=pbf_head_fused,
            dim=self.ps.particle_num,
            inputs=[
                self.lda, self.ps.m, self.ps.m_inv,
                self.ps.is_dynamic, self.ps.particle_num,
                int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW,
            ],
            outputs=[self.ps.x],
        )

    def solve(self, dt):
        """Position-based density solve. Framework pre-advects and rebuilds neighbors
        first; x is at the predicted position, ps.x_old holds the pre-advection one."""
        t_frame = time.perf_counter()

        n = self.ps.particle_num
        opt_iter = 0          # position updates applied so far
        err = 0.0
        pos_res = 0.0
        self.reset_stats()
        # Max-dx criterion: checked AFTER the update it measures (see below).
        use_dx = self.convergence_mode == 0

        for _ in range(self.max_iteration_opt + 1):
            t_iter = time.perf_counter()
            # Recompute gradW with updated positions (neighbor indices unchanged)
            self.ns.recompute_gradW(
                self.ps.x,
                self.ps.particle_num,
                int(self.ps.cache_size),
                self.ps.neighbor_count,
                self.ps.neighbor_ids,
                self.ps.neighbor_gradW,
            )

            # Density, constraint c, lambda (omega folded in).
            wp.launch(
                kernel=compute_lambda,
                dim=self.ps.particle_num,
                inputs=[
                    self.ps.x, self.ps.m, self.ps.m_inv, self.ps.density0,
                    self.ps.is_dynamic, self.ps.particle_num, self.ps.support_radius,
                    self.omega, int(self.ps.cache_size),
                    self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW
                ],
                outputs=[self.ps.density, self.c, self.lda]
            )

            # Fused tail: c/density0 (in register) + atomic-add reduction.
            self.result_buffer.zero_()
            self.cnt_buffer.zero_()
            wp.launch(
                kernel=pbf_tail_fused,
                dim=self.ps.particle_num,
                inputs=[
                    self.c, self.ps.density0, self.ps.is_dynamic,
                    self.ps.particle_num,
                    self.result_buffer, self.cnt_buffer,
                ],
            )
            err = self.read_avg_error()
            wp.synchronize()
            self.stats_density_errors.append(err)

            converged = (err < self.tol_opt) and (not self.use_fixed_iters) and (not use_dx)

            # Density criterion: checked before the update, once at least one update is done.
            if (converged and opt_iter >= 1) or opt_iter >= self.max_iteration_opt:
                self.stats_iter_costs.append((time.perf_counter() - t_iter) * 1000.0)
                break

            wp.copy(self.x_prev, self.ps.x, count=n)
            self._update_positions()

            opt_iter += 1
            # Position residual of this update: max_i ||x^{l+1} - x^l|| / h.
            pos_res = self.compute_max_pos_update(self.x_prev)
            self.stats_pos_residuals.append(pos_res)
            self.stats_iter_costs.append((time.perf_counter() - t_iter) * 1000.0)

            if use_dx:
                # Max-dx criterion (>= 2 updates), the cap, or a spent fixed budget.
                if ((not self.use_fixed_iters) and pos_res < self.pos_tol and opt_iter > 1) \
                        or opt_iter >= self.max_iteration_opt:
                    break

        # v = (x_corrected - x_old) / dt, and the divergence for the visualizer
        self.update_velocity_from_positions(dt)
        self.compute_divergence()

        self.stats_outer_iters = opt_iter
        self.stats_total_cost_ms = (time.perf_counter() - t_frame) * 1000.0
        # In max-dx mode err is the density error measured just before the last update.
        self.stats_final_density_error = err
        self.stats_final_pos_residual = pos_res
        self.check_converged(err, opt_iter, pos_res)
