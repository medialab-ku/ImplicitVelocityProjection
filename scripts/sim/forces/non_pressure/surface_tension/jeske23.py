"""
Jeske 2023 implicit surface tension for WarpSPH simulation.

Implements "Implicit Surface Tension for SPH Fluid Simulation"
(Jeske et al., ACM TOG 2023).

Solves  M v^{t+1} = v^t + b  via matrix-free PCG (correction form),
where M = I + dt^2 (sigma/m_i) sum_j m_bar_ij g_tilde_ij (v_i - v_j).

The surface tension kernel W^st is a clamped cubic spline with support h_st = 2d
and normalization correction theta = 10/7.
"""

import warp as wp
from .surface_tension import SurfaceTension
from scripts.sim.math_utils import add_vec3, dot_vec3


# ============================================================================
# Warp device functions: surface tension kernel W^st
# ============================================================================

@wp.func
def compute_Wst(r_ij: float, d: float, h_st: float) -> float:
    """Surface tension kernel value.

    W^st(r; h_st) = theta * W_cubic(max(r, d); h_st)
    where theta = 10/7 (3D normalization correction).
    """
    theta = 10.0 / 7.0
    r_clamped = wp.max(r_ij, d)
    q = r_clamped / h_st
    # W_cubic(r; h) = (8 / (pi h^3)) * f(q)
    norm = 8.0 / (3.14159265358979 * h_st * h_st * h_st)
    fq = float(0.0)
    if q <= 0.5:
        fq = 1.0 - 6.0 * q * q + 6.0 * q * q * q
    elif q <= 1.0:
        t = 1.0 - q
        fq = 2.0 * t * t * t
    return theta * norm * fq


@wp.func
def compute_grad_Wst(x_ij: wp.vec3, r_ij: float, d: float, h_st: float) -> wp.vec3:
    """Surface tension kernel gradient w.r.t. x_i.

    If r_ij <= d: gradient is zero (kernel is constant below d).
    If r_ij > d:  theta * grad W_cubic(x_ij; h_st).
    """
    theta = 10.0 / 7.0
    if r_ij <= d or r_ij < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    q = r_ij / h_st
    if q > 1.0:
        return wp.vec3(0.0, 0.0, 0.0)
    norm = 8.0 / (3.14159265358979 * h_st * h_st * h_st)
    fpq = float(0.0)
    if q <= 0.5:
        fpq = -12.0 * q + 18.0 * q * q
    else:
        t = 1.0 - q
        fpq = -6.0 * t * t
    # grad_i W = norm * (1/h) * f'(q) * (x_ij / r_ij)
    return theta * norm * (1.0 / h_st) * fpq * (x_ij / r_ij)


# ============================================================================
# Jeske23 Kernels
# ============================================================================

@wp.kernel
def compute_divergence_jeske23(
    v: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    material_fluid: int,
    div_arr: wp.array(dtype=float),
):
    """Precompute divergence div_i = sum_j m_j * dot(v_ij, grad_W_ij) (cached gradient).

    Uses the STANDARD SPH kernel gradient (neighbor_gradW), not the ST kernel.
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    if is_dynamic[p_i] == 0 or material[p_i] != material_fluid:
        div_arr[p_i] = 0.0
        return

    v_i = v[p_i]
    div_val = float(0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        if material[p_j] != material_fluid:
            continue
        v_ij = v_i - v[p_j]
        grad_W = neighbor_gradW[base + j]
        div_val = div_val + m[p_j] * wp.dot(v_ij, grad_W)

    div_arr[p_i] = div_val


@wp.kernel
def compute_g_tilde_jeske23(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=float),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    div_arr: wp.array(dtype=float),
    dt: float,
    particle_diameter: float,
    support_radius_st: float,
    material_fluid: int,
    g_tilde: wp.array(dtype=float),
):
    """Precompute g_tilde_ij for each fluid neighbor pair.

    g_tilde_ij = W^st_ij / rho_bar_ij
                 + dt * [ dot(grad W^st_ij, v_ij) / rho_bar_ij
                          - W^st_ij / (2 rho_bar_ij^2) * (div_i + div_j) ]
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    if is_dynamic[p_i] == 0 or material[p_i] != material_fluid:
        for j in range(cnt):
            g_tilde[base + j] = 0.0
        return

    x_i = x[p_i]
    v_i = v[p_i]
    rho_i = density[p_i]
    div_i = div_arr[p_i]
    d = particle_diameter
    h_st = support_radius_st

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        if material[p_j] != material_fluid:
            g_tilde[base + j] = 0.0
            continue

        x_ij = x_i - x[p_j]
        r_ij = wp.length(x_ij)

        # Skip pairs outside ST kernel support
        if r_ij >= h_st:
            g_tilde[base + j] = 0.0
            continue

        v_ij = v_i - v[p_j]
        rho_j = density[p_j]
        rho_bar = 0.5 * (rho_i + rho_j)
        div_j = div_arr[p_j]

        Wst = compute_Wst(r_ij, d, h_st)
        grad_Wst = compute_grad_Wst(x_ij, r_ij, d, h_st)

        # g_ij = W^st / rho_bar
        g_ij = Wst / rho_bar

        # dg_ij = term_kernel - term_density
        term_kernel = wp.dot(grad_Wst, v_ij) / rho_bar
        term_density = Wst / (2.0 * rho_bar * rho_bar) * (div_i + div_j)
        g_tilde[base + j] = g_ij + dt * (term_kernel - term_density)

@wp.kernel
def compute_rhs_jeske23(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    g_tilde: wp.array(dtype=float),
    sigma: float,
    dt: float,
    material_fluid: int,
    b: wp.array(dtype=wp.vec3),
):
    """Compute extra RHS vector b_i.

    b_i = -dt * (sigma / m_i) * sum_j m_bar_ij * x_ij * g_tilde_ij
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    if is_dynamic[p_i] == 0 or material[p_i] != material_fluid:
        b[p_i] = wp.vec3(0.0, 0.0, 0.0)
        return

    x_i = x[p_i]
    m_i = m[p_i]
    factor = -dt * sigma / m_i

    acc = wp.vec3(0.0, 0.0, 0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        if material[p_j] != material_fluid:
            continue

        g_t = g_tilde[base + j]
        if g_t == 0.0:
            continue

        x_ij = x_i - x[p_j]
        m_bar = 0.5 * (m_i + m[p_j])
        acc = acc + m_bar * x_ij * g_t

    b[p_i] = factor * acc


@wp.kernel
def compute_Av_jeske23(
    v_in: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    g_tilde: wp.array(dtype=float),
    sigma: float,
    dt: float,
    material_fluid: int,
    Av: wp.array(dtype=wp.vec3),
):
    """Matrix-vector product (Mv)_i.

    (Mv)_i = v_i + dt^2 * (sigma / m_i) * sum_j m_bar_ij * g_tilde_ij * v_ij

    Sign is POSITIVE (unlike Weiler18's NEGATIVE).
    v_in is the CG iterate, NOT self.ps.v.
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    if is_dynamic[p_i] == 0 or material[p_i] != material_fluid:
        Av[p_i] = v_in[p_i]
        return

    v_i = v_in[p_i]
    m_i = m[p_i]
    dt2 = dt * dt
    factor = dt2 * sigma / m_i

    sum_val = wp.vec3(0.0, 0.0, 0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        if material[p_j] != material_fluid:
            continue

        g_t = g_tilde[base + j]
        if g_t == 0.0:
            continue

        v_ij = v_i - v_in[p_j]
        m_bar = 0.5 * (m_i + m[p_j])
        sum_val = sum_val + m_bar * g_t * v_ij

    # POSITIVE sign
    Av[p_i] = v_i + factor * sum_val


@wp.kernel
def compute_precond_jeske23(
    m: wp.array(dtype=float),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    g_tilde: wp.array(dtype=float),
    sigma: float,
    dt: float,
    material_fluid: int,
    precond_inv: wp.array(dtype=wp.mat33),
):
    """Compute block Jacobi preconditioner inverse.

    M_ii = I3 * [1 + dt^2 * (sigma / m_i) * sum_j m_bar_ij * g_tilde_ij]
    M_ii^{-1} = I3 * [1 / diag_val]
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    I3 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    if is_dynamic[p_i] == 0 or material[p_i] != material_fluid:
        precond_inv[p_i] = I3
        return

    m_i = m[p_i]
    dt2 = dt * dt
    factor = dt2 * sigma / m_i

    diag_sum = float(0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        if material[p_j] != material_fluid:
            continue
        m_bar = 0.5 * (m_i + m[p_j])
        diag_sum = diag_sum + m_bar * g_tilde[base + j]

    diag_val = 1.0 + factor * diag_sum
    # Protect against zero/negative diagonal
    if diag_val < 1.0e-12:
        diag_val = 1.0e-12
    inv_val = 1.0 / diag_val
    precond_inv[p_i] = inv_val * I3


@wp.kernel
def apply_precond_jeske23(
    precond_inv: wp.array(dtype=wp.mat33),
    r: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    z: wp.array(dtype=wp.vec3),
):
    """z = M^{-1} * r  (per-particle 3x3 mat-vec)."""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    z[i] = precond_inv[i] * r[i]


# ============================================================================
# Jeske23 Class
# ============================================================================

class Jeske23(SurfaceTension):
    """Implicit surface tension solver (Jeske et al. 2023).

    Solves for the surface-tension velocity via backward Euler + PCG.
    Uses the correction form:  M * dv = rhs,  where rhs = v_old - M*v_old + b,
    then v_new = v_old + dv.
    """

    def __init__(self, simulation_data, neighbour_search, config):
        super().__init__(simulation_data, neighbour_search, config)

        max_particles = self.ps.particle_max_num
        cs = int(self.ps.cache_size)

        # Surface tension kernel parameters
        self.particle_diameter = 2.0 * self.ps.particle_radius  # d
        self.support_radius_st = 2.0 * self.particle_diameter   # h_st = 2d = 4 * particle_radius

        # Verify neighbor search covers ST support radius
        if self.support_radius_st > self.ps.support_radius:
            raise RuntimeError(
                f"Surface tension support radius h_st={self.support_radius_st:.4f} "
                f"> SPH support radius h={self.ps.support_radius:.4f}. "
                f"Neighbor search cannot find all required pairs."
            )

        # PCG work arrays
        self.dv     = wp.zeros(max_particles, dtype=wp.vec3)
        self.r_cg   = wp.zeros(max_particles, dtype=wp.vec3)
        self.z_cg   = wp.zeros(max_particles, dtype=wp.vec3)
        self.p_cg   = wp.zeros(max_particles, dtype=wp.vec3)
        self.Ap_cg  = wp.zeros(max_particles, dtype=wp.vec3)
        self.rhs_cg = wp.zeros(max_particles, dtype=wp.vec3)

        # Extra RHS from surface tension
        self.b_st = wp.zeros(max_particles, dtype=wp.vec3)

        # Block Jacobi preconditioner
        self.precond_inv = wp.zeros(max_particles, dtype=wp.mat33)

        # Precomputed arrays
        self.div_arr = wp.zeros(max_particles, dtype=float)
        self.g_tilde = wp.zeros(max_particles * cs, dtype=float)

        # Reduction buffers
        self.dot_buffer = wp.zeros(1, dtype=float)
        self.dot_host = wp.zeros(1, dtype=float, device="cpu")

        # Solver parameters
        self.max_cg_iterations = 100
        self.tol = 1e-5

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def dot(self, a, b):
        """Dot product over dynamic particles (GPU -> CPU read-back)."""
        self.dot_buffer.zero_()
        wp.launch(
            kernel=dot_vec3,
            dim=self.ps.particle_num,
            inputs=[a, b, self.ps.is_dynamic, self.ps.particle_num],
            outputs=[self.dot_buffer],
        )
        wp.copy(self.dot_host, self.dot_buffer)
        return float(self.dot_host.numpy()[0])

    def compute_Av(self, Av, v_in, sigma, dt):
        """Launch the matrix-vector product kernel."""
        wp.launch(
            kernel=compute_Av_jeske23,
            dim=self.ps.particle_num,
            inputs=[
                v_in, self.ps.m,
                self.ps.material, self.ps.is_dynamic,
                self.ps.particle_num, int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids,
                self.g_tilde,
                sigma, dt, self.ps.material_fluid,
            ],
            outputs=[Av],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Preconditioner
    # ------------------------------------------------------------------

    def compute_preconditioner(self, sigma, dt):
        """Compute block Jacobi preconditioner."""
        wp.launch(
            kernel=compute_precond_jeske23,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.m,
                self.ps.material, self.ps.is_dynamic,
                self.ps.particle_num, int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids,
                self.g_tilde,
                sigma, dt, self.ps.material_fluid,
            ],
            outputs=[self.precond_inv],
            device=self.device,
        )

    def apply_preconditioner(self, z, r):
        """z = M^{-1} * r (block Jacobi)."""
        wp.launch(
            kernel=apply_precond_jeske23,
            dim=self.ps.particle_num,
            inputs=[self.precond_inv, r, self.ps.is_dynamic,
                    self.ps.particle_num],
            outputs=[z],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # PCG solver (block Jacobi preconditioned CG)
    # ------------------------------------------------------------------

    def PCG(self, x, b, sigma, dt):
        """Preconditioned Conjugate Gradient: solve M*x = b from x = 0."""
        r = self.r_cg
        z = self.z_cg
        p = self.p_cg
        Ap = self.Ap_cg
        n = self.ps.particle_num

        # r = b  (since x = 0)
        wp.copy(r, b)

        # z = M^{-1} r
        self.apply_preconditioner(z, r)

        # p = z
        wp.copy(p, z)

        rz_old = self.dot(r, z)
        cg_iter = 0

        if rz_old > 1e-12:
            for _ in range(self.max_cg_iterations):
                self.compute_Av(Ap, p, sigma, dt)
                pAp = self.dot(p, Ap)

                if pAp <= 0.0:
                    break

                alpha = rz_old / max(pAp, 1e-12)

                # x += alpha * p
                wp.launch(add_vec3, dim=n,
                          inputs=[x, x, alpha, p,
                                  self.ps.is_dynamic, n])
                # r -= alpha * Ap
                wp.launch(add_vec3, dim=n,
                          inputs=[r, r, -alpha, Ap,
                                  self.ps.is_dynamic, n])

                r_norm = self.dot(r, r)
                cg_iter += 1
                if (r_norm < self.tol and cg_iter >= 2) \
                        or cg_iter >= self.max_cg_iterations:
                    break

                # z = M^{-1} r
                self.apply_preconditioner(z, r)

                rz_new = self.dot(r, z)
                beta = rz_new / max(rz_old, 1e-12)

                # p = z + beta * p
                wp.launch(add_vec3, dim=n,
                          inputs=[p, z, beta, p,
                                  self.ps.is_dynamic, n])
                rz_old = rz_new

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def solve(self, sigma: float, _coeff_adh: float, dt: float):
        """Solve implicit surface tension for one time step.

        1. Precompute divergence div_i
        2. Precompute g_tilde_ij
        3. Compute extra RHS vector b_i
        4. Compute preconditioner M_ii^{-1}
        5. Compute rhs_correction:
           a. M * v^t  -> Ap_cg
           b. rhs = v^t - Ap_cg
           c. rhs += b
        6. PCG solve  M * dv = rhs
        7. v^{t+1} = v^t + dv
        """
        n = self.ps.particle_num
        cs = int(self.ps.cache_size)

        # 1. Precompute divergence (uses standard SPH kernel)
        wp.launch(
            kernel=compute_divergence_jeske23,
            dim=n,
            inputs=[
                self.ps.v, self.ps.m,
                self.ps.material, self.ps.is_dynamic,
                n, cs,
                self.ps.neighbor_count, self.ps.neighbor_ids,
                self.ps.neighbor_gradW,
                self.ps.material_fluid,
            ],
            outputs=[self.div_arr],
            device=self.device,
        )

        # 2. Precompute g_tilde (uses ST kernel, computed on-the-fly)
        wp.launch(
            kernel=compute_g_tilde_jeske23,
            dim=n,
            inputs=[
                self.ps.x, self.ps.v, self.ps.density,
                self.ps.material, self.ps.is_dynamic,
                n, cs,
                self.ps.neighbor_count, self.ps.neighbor_ids,
                self.div_arr,
                dt, self.particle_diameter, self.support_radius_st,
                self.ps.material_fluid,
            ],
            outputs=[self.g_tilde],
            device=self.device,
        )

        # 3. Compute extra RHS b_i
        wp.launch(
            kernel=compute_rhs_jeske23,
            dim=n,
            inputs=[
                self.ps.x, self.ps.m,
                self.ps.material, self.ps.is_dynamic,
                n, cs,
                self.ps.neighbor_count, self.ps.neighbor_ids,
                self.g_tilde,
                sigma, dt, self.ps.material_fluid,
            ],
            outputs=[self.b_st],
            device=self.device,
        )

        # 4. Compute preconditioner
        self.compute_preconditioner(sigma, dt)

        # 5. Correction form: rhs = v_old - M*v_old + b, solved from dv = 0
        self.dv.zero_()

        # 5a. M * v^t -> Ap_cg
        self.compute_Av(self.Ap_cg, self.ps.v, sigma, dt)

        # 5b. rhs = v^t - Ap_cg
        wp.launch(add_vec3, dim=n,
                  inputs=[self.rhs_cg, self.ps.v, -1.0, self.Ap_cg,
                          self.ps.is_dynamic, n])

        # 5c. rhs += b
        wp.launch(add_vec3, dim=n,
                  inputs=[self.rhs_cg, self.rhs_cg, 1.0, self.b_st,
                          self.ps.is_dynamic, n])

        # 6. PCG solve M * dv = rhs
        self.PCG(self.dv, self.rhs_cg, sigma, dt)

        # 7. v += dv
        wp.launch(add_vec3, dim=n,
                  inputs=[self.ps.v, self.ps.v, 1.0, self.dv,
                          self.ps.is_dynamic, n])
