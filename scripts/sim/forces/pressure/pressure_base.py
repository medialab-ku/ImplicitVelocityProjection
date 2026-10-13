"""Base class for pressure solvers."""

import warp as wp
from scripts.sim.sph_kernel import cubic_kernel
from scripts.sim.math_utils import add_scalar, coef_wise_div, dot_vec3_tiled


# ============================================================================
# PressureBase Kernels
# ============================================================================

@wp.kernel
def compute_density_kernel(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    support_radius: float,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    density: wp.array(dtype=float)
):
    """SPH density: rho_i = sum_j m_j * W(r_ij, h)"""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        return

    pos_i = x[p_i]
    rho = float(0.0)
    rho = rho + m[p_i] * cubic_kernel(0.0, support_radius)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        pos_j = x[p_j]
        r_norm = wp.length(pos_i - pos_j)
        rho += m[p_j] * cubic_kernel(r_norm, support_radius)

    density[p_i] = rho


@wp.kernel
def compute_Aii_kernel(
    m: wp.array(dtype=float),
    m_inv: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    k: wp.array(dtype=float)
):
    """Diagonal D^-1 of J M^-1 J^T: k = 1 / (Aii + eps)"""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        k[p_i] = 0.0
        return

    eps = 1e-3
    Aii = float(0.0)
    J_ii = wp.vec3(float(0.0), float(0.0), float(0.0))

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        J_ij = m[p_j] * grad_W
        if is_dynamic[p_j] == 1:
            Aii += wp.dot(J_ij, J_ij) * m_inv[p_j]
        J_ii -= J_ij

    Aii += wp.dot(J_ii, J_ii) * m_inv[p_i]
    k[p_i] = 1.0 / (Aii + eps)



@wp.kernel
def compute_J_x_kernel(
    x_val: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    ret: wp.array(dtype=float)
):
    """Jacobian * vector: J(x)_i = sum_j m_j * (x_i - x_j) . grad_W_ij"""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        ret[p_i] = 0.0
        return

    val_i = x_val[p_i]
    result = float(0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        if is_dynamic[p_j] == 1:
            result += m[p_j] * wp.dot(val_i - x_val[p_j], grad_W)
        else:
            result += m[p_j] * wp.dot(val_i, grad_W)

    ret[p_i] = result


@wp.kernel
def compute_J_tr_x_kernel(
    x_val: wp.array(dtype=float),
    m: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    ret: wp.array(dtype=wp.vec3)
):
    """Jacobian transpose * scalar: J^T(x)_i = sum_j (m_j*x_i + m_i*x_j) * grad_W_ij"""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        ret[p_i] = wp.vec3(0.0, 0.0, 0.0)
        return

    val_i = x_val[p_i]
    result = wp.vec3(float(0.0), float(0.0), float(0.0))

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        grad_W = neighbor_gradW[base + j]
        if is_dynamic[p_j] == 1:
            result += (m[p_j] * val_i + m[p_i] * x_val[p_j]) * grad_W
        else:
            result += (m[p_j] * val_i) * grad_W

    ret[p_i] = result


@wp.kernel
def compute_constraint_kernel(
    density: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    c: wp.array(dtype=float)
):
    """Constraint: c = density - density0"""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        c[p_i] = 0.0
        return
    c[p_i] = density[p_i] - density0[p_i]


@wp.kernel
def compute_avg_density_error_kernel(
    x: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    result: wp.array(dtype=float),
    count: wp.array(dtype=float)
):
    """Reduction for the average density error: result += max(x, 0), count += 1
    over dynamic particles (x is already divided by rho0)."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        return
    wp.atomic_add(result, 0, wp.max(x[p_i], 0.0))
    wp.atomic_add(count, 0, 1.0)


@wp.kernel
def reduce_max_dx_sq(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    out: wp.array(dtype=float),
):
    """out[0] = max_i ||x[i] - x_prev[i]||^2 over dynamic particles."""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    d = x[i] - x_prev[i]
    wp.atomic_max(out, 0, wp.dot(d, d))


@wp.kernel
def compute_active_mask(
    neighbor_count: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    threshold: int,
    active: wp.array(dtype=float),
):
    """active_i = 1 if dynamic and #neighbors > threshold, else 0."""
    i = wp.tid()
    if i >= num_particles:
        return
    if is_dynamic[i] == 1 and neighbor_count[i] > threshold:
        active[i] = 1.0
    else:
        active[i] = 0.0


@wp.kernel
def update_divergence(
    Jv: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    divergence: wp.array(dtype=float)
):
    """divergence = max(Jv, 0) (dynamic particles), shown by the visualizer."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 1:
        divergence[p_i] = wp.max(Jv[p_i], 0.0)
    else:
        divergence[p_i] = 0.0


@wp.kernel
def update_velocities(
    x: wp.array(dtype=wp.vec3),
    x_old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    dt: float
):
    """v = (x - x_old) / dt"""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 1:
        v[p_i] = (x[p_i] - x_old[p_i]) / dt
    else:
        v[p_i] = wp.vec3(0.0, 0.0, 0.0)


# ============================================================================
# PressureBase Class
# ============================================================================

class PressureBase:
    # Default: velocity-based. Position-based subclasses override to True.
    is_position_based = False

    def __init__(self, particle_system, neighbour_search, config):
        self.ps = particle_system
        self.ns = neighbour_search

        max_particles = self.ps.particle_max_num

        # Solver buffers
        self.tmp = wp.zeros(max_particles, dtype=wp.vec3)
        self.dx = wp.zeros(max_particles, dtype=wp.vec3)
        self.s = wp.zeros(max_particles, dtype=wp.vec3)

        self.t = wp.zeros(max_particles, dtype=float)
        self.Jx = wp.zeros(max_particles, dtype=float)
        self.dp = wp.zeros(max_particles, dtype=float)
        self.c = wp.zeros(max_particles, dtype=float)
        self.p = wp.zeros(max_particles, dtype=float)
        self.p_previous = wp.zeros(max_particles, dtype=float)
        self.k = wp.zeros(max_particles, dtype=float)

        # Reduction buffers (device + host mirror) shared by `compute_avg_density_error`
        self.result_buffer = wp.zeros(1, dtype=float)
        self.cnt_buffer = wp.zeros(1, dtype=float)
        self.result_host = wp.zeros(1, dtype=float, device="cpu")
        self.cnt_host = wp.zeros(1, dtype=float, device="cpu")
        self.dot_buffer = wp.zeros(1, dtype=float)
        self.dot_host = wp.zeros(1, dtype=float, device="cpu")

        # Solver parameters. max_iteration_opt caps the outer iterations (pressure
        # updates) per step; in fixed-iteration mode (use_fixed_iters) every solver
        # runs exactly that many and accepts the result without a convergence demand.
        self.max_iteration_opt = int(config.get_cfg("maxOuterIters", 1000))
        self.use_fixed_iters = bool(config.get_cfg("fixedOuterIters", False))
        self.tol_opt = float(config.get_cfg("densityErrorTol", 1e-3))
        self.omega = float(config.get_cfg("omega", 10.0))

        # Position-based solvers can stop on the max position update instead:
        #   convergence_mode 1 = avg density error < tol_opt (default)
        #                    0 = max_i ||x^{l+1} - x^l|| / h < pos_tol
        if self.is_position_based:
            self.convergence_mode = int(config.get_cfg("convergenceMode", 1))
            self.pos_tol = float(config.get_cfg("posTol", 1e-3))
            self.dx_max_sq_buf = wp.zeros(1, dtype=float)
            self.dx_max_sq_host = wp.zeros(1, dtype=float, device="cpu")
            # Pre-update snapshot of x for the residual ||x^{l+1} - x^l||.
            self.x_prev = wp.zeros(max_particles, dtype=wp.vec3)

        # Per-step stats, filled by solve(). pos_residuals (max dx / h per update)
        # is filled by the position-based solvers only.
        self.stats_density_errors = []
        self.stats_iter_costs = []
        self.stats_pcg_iters = []
        self.stats_pos_residuals = []
        self.stats_outer_iters = 0
        self.stats_final_density_error = 0.0
        self.stats_final_pos_residual = 0.0
        self.stats_total_cost_ms = 0.0

    def compute_density(self):
        wp.launch(
            kernel=compute_density_kernel,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.x, self.ps.m, self.ps.is_dynamic,
                self.ps.particle_num, self.ps.support_radius,
                int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids
            ],
            outputs=[self.ps.density]
        )

    def compute_Aii(self):
        wp.launch(
            kernel=compute_Aii_kernel,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.m, self.ps.m_inv, self.ps.is_dynamic,
                self.ps.particle_num,
                int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW
            ],
            outputs=[self.k]
        )


    def compute_J_x(self, ret, x):
        wp.launch(
            kernel=compute_J_x_kernel,
            dim=self.ps.particle_num,
            inputs=[
                x, self.ps.m, self.ps.is_dynamic,
                self.ps.particle_num,
                int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW
            ],
            outputs=[ret]
        )

    def compute_J_tr_x(self, ret, x):
        wp.launch(
            kernel=compute_J_tr_x_kernel,
            dim=self.ps.particle_num,
            inputs=[
                x, self.ps.m, self.ps.is_dynamic,
                self.ps.particle_num,
                int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW
            ],
            outputs=[ret]
        )

    def compute_constraint(self):
        wp.launch(
            kernel=compute_constraint_kernel,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.density, self.ps.density0, self.ps.is_dynamic,
                self.ps.particle_num
            ],
            outputs=[self.c]
        )

    def read_avg_error(self):
        """Host readback of a sum / count reduction in result_buffer / cnt_buffer."""
        wp.copy(self.result_host, self.result_buffer)
        wp.copy(self.cnt_host, self.cnt_buffer)
        total = float(self.result_host.numpy()[0])
        count = float(self.cnt_host.numpy()[0])
        return total / count if count > 0.0 else 0.0

    def compute_avg_density_error(self, x):
        """avg(max(x, 0)) over dynamic particles (x already divided by rho0)."""
        self.result_buffer.zero_()
        self.cnt_buffer.zero_()
        wp.launch(
            kernel=compute_avg_density_error_kernel,
            dim=self.ps.particle_num,
            inputs=[x, self.ps.is_dynamic, self.ps.particle_num],
            outputs=[self.result_buffer, self.cnt_buffer]
        )
        return self.read_avg_error()

    def dot(self, a, b):
        """Dot product of two vec3 arrays over dynamic particles (one host roundtrip)."""
        self.dot_buffer.zero_()
        wp.launch(
            kernel=dot_vec3_tiled,
            dim=self.ps.particle_num,
            inputs=[a, b, self.ps.is_dynamic, self.ps.particle_num],
            outputs=[self.dot_buffer],
            block_dim=128,
        )
        wp.copy(self.dot_host, self.dot_buffer)
        return float(self.dot_host.numpy()[0])

    def final_density_error(self, d):
        """Velocity-based solvers: avg(max((c + J d) / rho0, 0)) at the corrected
        displacement d. Uses the Jx and t buffers."""
        n = self.ps.particle_num
        self.compute_J_x(self.Jx, d)
        wp.launch(kernel=add_scalar, dim=n,
                  inputs=[self.t, self.Jx, 1.0, self.c, self.ps.is_dynamic, n])
        wp.launch(kernel=coef_wise_div, dim=n,
                  inputs=[self.t, self.t, self.ps.density0, self.ps.is_dynamic, n])
        return self.compute_avg_density_error(self.t)

    def compute_divergence(self):
        """ps.divergence = max(J v, 0) for the visualizer. Uses the Jx buffer."""
        self.compute_J_x(self.Jx, self.ps.v)
        wp.launch(
            kernel=update_divergence,
            dim=self.ps.particle_num,
            inputs=[self.Jx, self.ps.is_dynamic, self.ps.particle_num],
            outputs=[self.ps.divergence]
        )

    def update_velocity_from_positions(self, dt):
        """Position-based solvers: v = (x - x_old) / dt after the projection."""
        wp.launch(
            kernel=update_velocities,
            dim=self.ps.particle_num,
            inputs=[self.ps.x, self.ps.x_old, self.ps.v, self.ps.is_dynamic,
                    self.ps.particle_num, dt],
        )

    def compute_max_pos_update(self, x_prev):
        """max_i ||x_i - x_prev_i|| / h over dynamic particles (one host roundtrip)."""
        n = self.ps.particle_num
        self.dx_max_sq_buf.zero_()
        wp.launch(
            kernel=reduce_max_dx_sq,
            dim=n,
            inputs=[self.ps.x, x_prev, self.ps.is_dynamic, n],
            outputs=[self.dx_max_sq_buf],
        )
        wp.copy(self.dx_max_sq_host, self.dx_max_sq_buf)
        return float(self.dx_max_sq_host.numpy()[0]) ** 0.5 / self.ps.support_radius

    def convergence_metric(self, err, pos_res):
        """(value, tol, label) of the active outer stopping criterion."""
        if getattr(self, "convergence_mode", 1) == 0:
            return pos_res, self.pos_tol, "pos residual"
        return err, self.tol_opt, "density err"

    def check_converged(self, err, n_updates, pos_res=0.0):
        """Raise if the update cap was hit above tolerance. Fixed-iteration mode
        accepts the result as-is (no convergence demand)."""
        metric, tol, name = self.convergence_metric(err, pos_res)
        if (not self.use_fixed_iters) and metric >= tol and n_updates >= self.max_iteration_opt:
            raise RuntimeError(
                f"[{type(self).__name__}] does not converge! {name}={metric:.2e} "
                f"(tol={tol:.2e}) after {n_updates} iters"
            )

    def reset_stats(self):
        """Clear the per-update stat lists at the start of a solve."""
        self.stats_density_errors = []
        self.stats_iter_costs = []
        self.stats_pcg_iters = []
        self.stats_pos_residuals = []

    def solve(self, dt):
        """Override in subclass."""
        pass
