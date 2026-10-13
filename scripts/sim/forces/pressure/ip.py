"""
IP solver (Implicit Projection, velocity-based).
Kee et al., SIGGRAPH Asia 2026.
"""

import time
import warp as wp
from .pressure_base import PressureBase
from scripts.sim.math_utils import (
    add_vec3, scale_vec3, fill_vec3,
    coef_wise_div,
    pcg_update_x_r_scalar,
)

# ============================================================================
# IP Kernels
# ============================================================================

@wp.kernel
def compute_Jx_pcg(
    x_val: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    dfdt: wp.array(dtype=float),
    k: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    Jx: wp.array(dtype=float)
):
    """Compute J*x for PCG, scaled by dfdt and k."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    Jx[p_i] = 0.0
    if is_dynamic[p_i] == 0:
        return

    if dfdt[p_i] > 0.0:
        val_i = x_val[p_i]
        result = float(0.0)

        base = p_i * cache_size
        cnt = neighbor_count[p_i]
        if cnt > cache_size:
            cnt = cache_size

        for j in range(cnt):
            p_j = neighbor_ids[base + j]
            grad_W = neighbor_gradW[base + j]
            val_ij = m[p_j] * grad_W
            if is_dynamic[p_j] == 1:
                result += wp.dot(val_i - x_val[p_j], val_ij)
            else:
                result += wp.dot(val_i, val_ij)

        Jx[p_i] = result * dfdt[p_i] * k[p_i]


@wp.kernel
def compute_Ax_part1(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    Ax: wp.array(dtype=wp.vec3)
):
    """Ax = m * x — the mass term of the projection operator
    `A = M + omega J^T D^-1 phi' J`."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        Ax[p_i] = wp.vec3(0.0, 0.0, 0.0)
        return

    Ax[p_i] = m[p_i] * x[p_i]


@wp.kernel
def compute_Ax_part2(
    Jx: wp.array(dtype=float),
    m: wp.array(dtype=float),
    dfdt: wp.array(dtype=float),
    omega: float,
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    Ax: wp.array(dtype=wp.vec3)
):
    """Add omega * J^T * (Jx) to Ax using atomic operations."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        return
    if dfdt[p_i] <= 0.0:
        return

    Jx_i = Jx[p_i]

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        val_ij = m[p_j] * grad_W
        contrib = omega * Jx_i * val_ij

        wp.atomic_add(Ax, p_i, contrib)
        if is_dynamic[p_j] == 1:
            wp.atomic_sub(Ax, p_j, contrib)


@wp.kernel
def ip_pack_fused(
    d: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=float),
    m: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    f: wp.array(dtype=float),
    dfdt: wp.array(dtype=float),
):
    """Fused IP outer-loop pack: t = J d + c, f = max(t, 0) (hard max),
    dfdt = 1 if t >= 1e-6 else 0 (its Heaviside derivative)."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        t[p_i] = 0.0
        f[p_i] = 0.0
        dfdt[p_i] = 0.0
        return

    val_i = d[p_i]
    Jd_i = float(0.0)
    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        if is_dynamic[p_j] == 1:
            Jd_i += m[p_j] * wp.dot(val_i - d[p_j], grad_W)
        else:
            Jd_i += m[p_j] * wp.dot(val_i, grad_W)

    t_i = Jd_i + c[p_i]
    t[p_i] = t_i
    f[p_i] = wp.max(t_i, 0.0)
    if t_i >= 1e-6:
        dfdt[p_i] = 1.0
    else:
        dfdt[p_i] = 0.0


@wp.kernel
def ip_rhs_fused(
    f: wp.array(dtype=float),
    k: wp.array(dtype=float),
    m: wp.array(dtype=float),
    omega: float,
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    rhs: wp.array(dtype=wp.vec3),
):
    """Fused IP rhs: rhs = omega J^T (f k). The products f k stay in registers,
    so f itself is not modified."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        rhs[p_i] = wp.vec3(0.0, 0.0, 0.0)
        return

    f_k_i = f[p_i] * k[p_i]
    m_i = m[p_i]
    result = wp.vec3(0.0, 0.0, 0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        if is_dynamic[p_j] == 1:
            f_k_j = f[p_j] * k[p_j]
            result += (m[p_j] * f_k_i + m_i * f_k_j) * grad_W
        else:
            result += (m[p_j] * f_k_i) * grad_W

    rhs[p_i] = omega * result


# ============================================================================
# IP Class
# ============================================================================

class IP(PressureBase):
    """Velocity-based density solver: implicit projection with a CG inner solve."""

    def __init__(self, particle_system, neighbour_search, config):
        super().__init__(particle_system, neighbour_search, config)

        self.tol_pcg = float(config.get_cfg("tol_pcg", 1e-6))
        self.max_iteration_pcg = 1000
        self.D_inv = self.k  # 1 / (Aii + eps), refreshed by compute_Aii() before each solve

        max_particles = self.ps.particle_max_num

        # f = max(c + J d, 0), the IP source term, and its derivative dfdt.
        self.f = wp.zeros(max_particles, dtype=float)
        self.dfdt  = wp.zeros(max_particles, dtype=float)

        # CG buffers
        self.a     = wp.zeros(max_particles, dtype=wp.vec3)
        self.Ap    = wp.zeros(max_particles, dtype=wp.vec3)
        self.r_pcg = wp.zeros(max_particles, dtype=wp.vec3)
        self.p_pcg = wp.zeros(max_particles, dtype=wp.vec3)

    def compute_Ax(self, Ax, Jx, x):
        """Compute A*x = M*x + omega * J^T*diag(dfdt*k)*J*x for PCG."""
        # First compute Jx (J * x) with dfdt and k scaling
        wp.launch(
            kernel=compute_Jx_pcg,
            dim=self.ps.particle_num,
            inputs=[
                x, self.ps.m, self.dfdt, self.D_inv, self.ps.is_dynamic,
                self.ps.particle_num,
                int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW
            ],
            outputs=[Jx]
        )

        # Ax = m * x
        wp.launch(
            kernel=compute_Ax_part1,
            dim=self.ps.particle_num,
            inputs=[
                x, self.ps.m,
                self.ps.is_dynamic, self.ps.particle_num
            ],
            outputs=[Ax]
        )

        # Add omega * J^T * Jx to Ax
        wp.launch(
            kernel=compute_Ax_part2,
            dim=self.ps.particle_num,
            inputs=[
                Jx, self.ps.m, self.dfdt, self.omega,
                self.ps.is_dynamic, self.ps.particle_num,
                int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW
            ],
            outputs=[Ax]
        )

    def PCG(self, x, b):
        """CG on A x = b from x = 0 (identity preconditioner).
        Returns the number of CG iterations."""
        r = self.r_pcg
        p = self.p_pcg
        Ap = self.Ap
        Jp = self.dp
        n = self.ps.particle_num

        # r = b (x = 0)
        wp.copy(r, b)

        # rr = r^T r
        rr_old = self.dot(r, r)
        pcg_iter = 0

        # Guards below are pure divide-by-zero floors (1e-30), NOT value clamps -- a
        # larger floor could engage on a legitimately small p^T A p and silently
        # distort alpha.
        if rr_old > 1e-30:
            # p = r
            wp.copy(p, r)
            for _ in range(self.max_iteration_pcg):
                self.compute_Ax(Ap, Jp, p)
                pAp = self.dot(p, Ap)
                if pAp < 0.0:
                    raise RuntimeError(f"[IP] non-positive definite matrix at PCG iter {pcg_iter} (pAp={pAp:.2e})")

                alpha = rr_old / max(pAp, 1e-30)

                # x += alpha*p, r -= alpha*Ap
                wp.launch(kernel=pcg_update_x_r_scalar, dim=n,
                          inputs=[x, r, alpha, p, Ap, self.ps.is_dynamic, n])

                rr_new = self.dot(r, r)
                pcg_iter += 1
                if (rr_new < self.tol_pcg and pcg_iter >= 2) \
                        or pcg_iter >= self.max_iteration_pcg:
                    break

                beta = rr_new / max(rr_old, 1e-30)
                # p = r + beta * p
                wp.launch(kernel=add_vec3, dim=n,
                          inputs=[p, r, beta, p, self.ps.is_dynamic, n])
                rr_old = rr_new

        return pcg_iter

    def solve(self, dt):
        """Outer loop: each update solves (M + ω Jᵀ diag(f′ k) J) sol = rhs with CG and
        sets d -= sol. Stops on the average density error, checked before each update
        once at least one update is done."""
        t_frame = time.perf_counter()
        self.compute_constraint()
        self.reset_stats()

        n = self.ps.particle_num
        d, c, rhs, sol = self.dx, self.c, self.tmp, self.a

        # d = dt * v
        wp.launch(kernel=scale_vec3, dim=n, inputs=[d, dt, self.ps.v, self.ps.is_dynamic, n])

        opt_iter = 0  # updates applied
        while True:
            t_iter = time.perf_counter()

            # t = J d + c, f = max(t, 0), dfdt = heaviside(t)
            wp.launch(
                kernel=ip_pack_fused,
                dim=n,
                inputs=[
                    d, c, self.ps.m, self.ps.is_dynamic, n,
                    int(self.ps.cache_size),
                    self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW,
                ],
                outputs=[self.t, self.f, self.dfdt],
            )

            # err = avg(f / rho0)
            wp.launch(kernel=coef_wise_div, dim=n,
                      inputs=[self.t, self.f, self.ps.density0, self.ps.is_dynamic, n])
            err = self.compute_avg_density_error(self.t)
            self.stats_density_errors.append(err)

            converged = err < self.tol_opt and not self.use_fixed_iters
            if (converged and opt_iter >= 1) or opt_iter >= self.max_iteration_opt:
                self.stats_iter_costs.append((time.perf_counter() - t_iter) * 1000.0)
                break

            # rhs = omega J^T (f k), then (M + omega J^T diag(dfdt k) J) sol = rhs
            wp.launch(
                kernel=ip_rhs_fused,
                dim=n,
                inputs=[
                    self.f, self.D_inv, self.ps.m, self.omega,
                    self.ps.is_dynamic, n,
                    int(self.ps.cache_size),
                    self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW,
                ],
                outputs=[rhs],
            )
            wp.launch(kernel=fill_vec3, dim=n,
                      inputs=[sol, wp.vec3(0.0, 0.0, 0.0), self.ps.is_dynamic, n])
            self.stats_pcg_iters.append(self.PCG(sol, rhs))

            # d -= sol
            wp.launch(kernel=add_vec3, dim=n, inputs=[d, d, -1.0, sol, self.ps.is_dynamic, n])
            opt_iter += 1
            wp.synchronize()
            self.stats_iter_costs.append((time.perf_counter() - t_iter) * 1000.0)

        self.stats_outer_iters = opt_iter
        self.check_converged(err, opt_iter)

        # v = d / dt
        wp.launch(kernel=scale_vec3, dim=n, inputs=[self.ps.v, 1.0 / dt, d, self.ps.is_dynamic, n])

        # Final density error at the corrected d, and divergence for visualization
        self.stats_final_density_error = self.final_density_error(d)
        self.compute_divergence()
        wp.synchronize()
        self.stats_total_cost_ms = (time.perf_counter() - t_frame) * 1000.0
