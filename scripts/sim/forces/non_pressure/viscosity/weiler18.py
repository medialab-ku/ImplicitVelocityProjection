"""
Weiler 2018 implicit viscosity for WarpSPH simulation.

Implements "A Physically Consistent Implicit Viscosity Solver for SPH Fluids"
(Weiler et al., Computer Graphics Forum 37(2), 2018).

Solves  (I - dt/rho * L_visc) v_new = v_old  via matrix-free CG,
where L_visc is the SPH Laplacian:
  L_i(v) = 10 * mu * sum_j (m_j/rho_j) * (v_ij . x_ij) / (|x_ij|^2 + 0.01h^2) * grad_W_ij
"""

import warp as wp
from .viscosity import Viscosity
from scripts.sim.math_utils import add_vec3, dot_vec3


# ============================================================================
# Weiler18 Kernels
# ============================================================================

@wp.kernel
def compute_precond_weiler18(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    density: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    support_radius: float,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    visc_coeff: float,
    dt: float,
    material_fluid: int,
    precond_inv: wp.array(dtype=wp.mat33),
):
    """Compute inverse of diagonal 3x3 block Jacobi preconditioner (Eq. 10).

    Diagonal block of system matrix (I - dt*A):
      M_i = I - (dt*mu_i/rho_i) * d * sum_j (m_j/rho_j)/(r^2+eps^2) * gradW_ij (x) x_ij^T
    Stores M_i^{-1} for direct application in PCG.
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    I3 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    if is_dynamic[p_i] == 0 or material[p_i] != material_fluid:
        precond_inv[p_i] = I3
        return

    x_i = x[p_i]
    rho_i = density[p_i]
    eps2 = 0.01 * support_radius * support_radius
    d = float(10.0)
    mu_i = visc_coeff * density0[p_i]
    factor = dt * mu_i / rho_i

    # L_ii = d * sum_j (m_j/rho_j) / (r^2 + eps^2) * gradW_ij (x) x_ij^T
    L_ii = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        if material[p_j] != material_fluid:
            continue

        x_ij = x_i - x[p_j]
        r_sq = wp.dot(x_ij, x_ij)
        grad_W = neighbor_gradW[base + j]

        c = d * (m[p_j] / density[p_j]) / (r_sq + eps2)
        L_ii = L_ii + c * wp.outer(grad_W, x_ij)

    # M_i = I - factor * L_ii, store inverse
    M_i = I3 - factor * L_ii
    precond_inv[p_i] = wp.inverse(M_i)


@wp.kernel
def apply_precond_weiler18(
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


@wp.kernel
def compute_Av_weiler18(
    x: wp.array(dtype=wp.vec3),
    v_in: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    density: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    support_radius: float,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    visc_coeff: float,
    dt: float,
    material_fluid: int,
    Av: wp.array(dtype=wp.vec3),
):
    """Matrix-vector product: (Av)_i = v_i - (dt*mu_i/rho_i) * L_i(v).

    A = I - dt/rho * L_visc  (SPD for positive viscosity).
    Only fluid-fluid interactions contribute to the Laplacian.
    Non-fluid / non-dynamic particles get identity: Av = v_in.
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    if is_dynamic[p_i] == 0 or material[p_i] != material_fluid:
        Av[p_i] = v_in[p_i]
        return

    v_i = v_in[p_i]
    x_i = x[p_i]
    rho_i = density[p_i]

    # eps^2 = 0.01 * h^2  (regularization, per paper Eq. 5)
    eps2 = 0.01 * support_radius * support_radius
    # d = 2(dim+2) = 10 for 3D
    d = float(10.0)
    # mu_i = kinematic_visc * rest_density_i
    mu_i = visc_coeff * density0[p_i]
    # combined factor:  dt * mu / rho_i
    factor = dt * mu_i / rho_i

    laplacian = wp.vec3(0.0, 0.0, 0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        if material[p_j] != material_fluid:
            continue

        x_ij = x_i - x[p_j]
        v_ij = v_i - v_in[p_j]
        r_sq = wp.dot(x_ij, x_ij)

        grad_W = neighbor_gradW[base + j]
        vx = wp.dot(v_ij, x_ij)

        laplacian += d * (m[p_j] / density[p_j]) * (vx / (r_sq + eps2)) * grad_W

    # (Av)_i = v_i  -  factor * laplacian
    Av[p_i] = v_i - factor * laplacian


# ============================================================================
# Weiler18 Class
# ============================================================================

class Weiler18(Viscosity):
    """Implicit viscosity solver (Weiler et al. 2018).

    Solves for the viscous velocity via backward Euler + CG.
    Uses the correction form:  A * dv = rhs,  where rhs = v_old - A*v_old,
    then v_new = v_old + dv.  Initial guess dv = 0, or the previous dv with warm_start.
    """

    def __init__(self, simulation_data, neighbour_search, config):
        super().__init__(simulation_data, neighbour_search, config)

        max_particles = self.ps.particle_max_num

        # PCG work arrays
        self.dv     = wp.zeros(max_particles, dtype=wp.vec3)  # correction (PCG solution)
        self.r_cg   = wp.zeros(max_particles, dtype=wp.vec3)  # residual
        self.z_cg   = wp.zeros(max_particles, dtype=wp.vec3)  # preconditioned residual
        self.p_cg   = wp.zeros(max_particles, dtype=wp.vec3)  # search direction
        self.Ap_cg  = wp.zeros(max_particles, dtype=wp.vec3)  # mat-vec result
        self.rhs_cg = wp.zeros(max_particles, dtype=wp.vec3)  # right-hand side

        # Block Jacobi preconditioner: M_i^{-1} per particle (Eq. 10)
        self.precond_inv = wp.zeros(max_particles, dtype=wp.mat33)

        # Reduction buffers
        self.dot_buffer = wp.zeros(1, dtype=float)
        self.dot_host = wp.zeros(1, dtype=float, device="cpu")

        # Solver parameters
        self.max_cg_iterations = 1000
        self.tol = 1e-5
        self.use_preconditioner = True
        self.warm_start = False

    def reset(self):
        """Zero all work buffers."""
        self.dv.zero_()
        self.r_cg.zero_()
        self.z_cg.zero_()
        self.p_cg.zero_()
        self.Ap_cg.zero_()
        self.rhs_cg.zero_()

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

    def compute_Av(self, Av, v_in, coeff, dt):
        """Launch the matrix-vector product kernel."""
        wp.launch(
            kernel=compute_Av_weiler18,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.x, v_in, self.ps.m,
                self.ps.density, self.ps.density0,
                self.ps.material, self.ps.is_dynamic,
                self.ps.particle_num, self.ps.support_radius,
                int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids,
                self.ps.neighbor_gradW,
                coeff, dt, self.ps.material_fluid,
            ],
            outputs=[Av],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Preconditioner
    # ------------------------------------------------------------------

    def compute_preconditioner(self, coeff, dt):
        """Compute block Jacobi preconditioner: M_i^{-1} per particle."""
        wp.launch(
            kernel=compute_precond_weiler18,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.x, self.ps.m,
                self.ps.density, self.ps.density0,
                self.ps.material, self.ps.is_dynamic,
                self.ps.particle_num, self.ps.support_radius,
                int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids,
                self.ps.neighbor_gradW,
                coeff, dt, self.ps.material_fluid,
            ],
            outputs=[self.precond_inv],
            device=self.device,
        )

    def apply_preconditioner(self, z, r):
        """z = M^{-1} * r.  Identity (copy) when preconditioner is off."""
        if self.use_preconditioner:
            wp.launch(
                kernel=apply_precond_weiler18,
                dim=self.ps.particle_num,
                inputs=[self.precond_inv, r, self.ps.is_dynamic,
                        self.ps.particle_num],
                outputs=[z],
                device=self.device,
            )
        else:
            wp.copy(z, r)

    # ------------------------------------------------------------------
    # PCG solver (block Jacobi preconditioned CG)
    # ------------------------------------------------------------------

    def PCG(self, x, b, coeff, dt, warm):
        """Preconditioned Conjugate Gradient: solve A*x = b.

        Uses 3x3 block Jacobi preconditioner (Eq. 10).
        warm=False: x=0 on entry, r_0 = b.
        warm=True:  x=dv(t-dt), r_0 = b - A·x.
        """
        r = self.r_cg
        z = self.z_cg
        p = self.p_cg
        Ap = self.Ap_cg
        n = self.ps.particle_num

        if warm:
            # r = b - A·x
            self.compute_Av(Ap, x, coeff, dt)
            wp.launch(add_vec3, dim=n,
                      inputs=[r, b, -1.0, Ap,
                              self.ps.is_dynamic, n])
        else:
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
                self.compute_Av(Ap, p, coeff, dt)
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
    # public interface  (same signature as Viscosity.solve)
    # ------------------------------------------------------------------

    def solve(self, coeff: float, dt: float):
        """Solve implicit viscosity for one time step.

        1.  Compute block Jacobi preconditioner (Eq. 10)
        2.  Compute  rhs = v_old - A * v_old
        3.  PCG solve  A * dv = rhs   with  dv_0 = 0 (previous dv with warm_start)
        4.  v_new = v_old + dv
        """
        n = self.ps.particle_num

        # -- compute preconditioner (skip for plain CG) --
        if self.use_preconditioner:
            self.compute_preconditioner(coeff, dt)

        # -- correction formulation: A·dv = rhs, rhs = v_old - A·v_old --
        warm = self.warm_start
        if not warm:
            self.dv.zero_()
        # rhs = v_old - A * v_old
        self.compute_Av(self.Ap_cg, self.ps.v, coeff, dt)
        wp.launch(add_vec3, dim=n,
                  inputs=[self.rhs_cg, self.ps.v, -1.0, self.Ap_cg,
                          self.ps.is_dynamic, n])

        # solve A * dv = rhs
        self.PCG(self.dv, self.rhs_cg, coeff, dt, warm)

        # v += dv
        wp.launch(add_vec3, dim=n,
                  inputs=[self.ps.v, self.ps.v, 1.0, self.dv,
                          self.ps.is_dynamic, n])
