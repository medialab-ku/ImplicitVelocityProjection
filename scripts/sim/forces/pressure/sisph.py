"""
SISPH pressure solver — Semi-Implicit SPH (He et al., Eurographics 2025).
Ported to Python/Warp, with modifications, from PeriDyno's SemiImplicitDensitySolver.cu (Apache-2.0).
"""

import time
import warp as wp

from .pressure_base import PressureBase
from scripts.sim.sph_kernel import cubic_kernel, spiky_kernel_derivative


# ============================================================================
# Bulk-energy helpers (paper Sec. 4.3)
# ============================================================================

@wp.func
def sisph_Bdot(lam: float, energy_type: int) -> float:
    """Full first derivative B'(lambda) = B'+(lambda) + B'-(lambda).
    B0: lambda - 1,  B1: lambda^2 - 1,  B2: lambda^3 - 1  (B'- = -1 for all)."""
    bplus = lam
    if energy_type == 1:
        bplus = lam * lam
    elif energy_type == 2:
        bplus = lam * lam * lam
    return bplus - 1.0


@wp.func
def sisph_Benergy(lam: float, energy_type: int) -> float:
    """Bulk energy B(lambda) (paper Sec. 4.3), for the line-search psi.
    B0 = 1/2 (lambda-1)^2,  B1 = (lambda^3-1)/3 - lambda + 1,
    B2 = (lambda^4-1)/4 - lambda + 1.  All >= 0 for lambda >= 1."""
    d = lam - 1.0
    if energy_type == 1:
        return (lam * lam * lam - 1.0) / 3.0 - lam + 1.0
    elif energy_type == 2:
        return (lam * lam * lam * lam - 1.0) / 4.0 - lam + 1.0
    return 0.5 * d * d


# ============================================================================
# SISPH Kernels
# ============================================================================

@wp.kernel
def sisph_density_lambda(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    support_radius: float,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    density: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    c: wp.array(dtype=float),
):
    """rho_i (cubic) + clamped lambda_i = max(rho_i, rho0)/rho0 and the
    compression c_i = max(rho_i - rho0, 0) (the convergence error), at the current x.
    Recomputed every outer iteration (paper Algorithm 1: compute rho using Eq. 6)."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        density[p_i] = 0.0
        lam[p_i] = 1.0
        c[p_i] = 0.0
        return

    pos_i = x[p_i]
    rho = m[p_i] * cubic_kernel(0.0, support_radius)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for k in range(cnt):
        p_j = neighbor_ids[base + k]
        rho += m[p_j] * cubic_kernel(wp.length(pos_i - x[p_j]), support_radius)

    density[p_i] = rho
    rho0 = density0[p_i]
    if rho0 < 1e-6:
        rho0 = 1e-6
    rho_clamped = wp.max(rho, rho0)          # clamp: lambda >= 1 (Sec. 4.3)
    lam[p_i] = rho_clamped / rho0
    c[p_i] = wp.max(rho - rho0, 0.0)


@wp.kernel
def sisph_jacobi_step(
    x: wp.array(dtype=wp.vec3),      # current iterate x^k
    x_star: wp.array(dtype=wp.vec3),  # inertial target x* (frozen)
    m: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    support_radius: float,
    cache_size: int,
    dt: float,
    stiffness: float,                 # mu
    energy_type: int,
    conserve_momentum: int,
    use_line_search: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    f_out: wp.array(dtype=wp.vec3),   # fixed-point map result f(x^k)
):
    """One semi-implicit Jacobi step (paper Eqs. 12/14) + optional line search
    (Eqs. 19-20)."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_dynamic[p_i] == 0:
        f_out[p_i] = x[p_i]
        return

    m_i = m[p_i]
    if m_i < 1e-12:
        f_out[p_i] = x[p_i]
        return
    rho0 = density0[p_i]
    if rho0 < 1e-6:
        rho0 = 1e-6

    pos_i = x[p_i]
    lam_i = lam[p_i]
    Bdot_i = sisph_Bdot(lam_i, energy_type)

    A = stiffness * dt * dt / rho0
    inv_m_i = 1.0 / m_i

    num = x_star[p_i] - pos_i          # inertial term (x* - x_i)
    denom = float(1.0)                 # momentum coefficient normalized to 1

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for k in range(cnt):
        p_j = neighbor_ids[base + k]
        d = pos_i - x[p_j]             # x_i - x_j
        r2 = wp.dot(d, d)
        if r2 < 1e-12:
            continue
        gW = spiky_kernel_derivative(d, support_radius)   # W'(r) (x_i-x_j)/r
        s = wp.dot(gW, d) / r2         # (dW/dr)/r  (scalar, < 0)

        # i's own constraint: base_i = A (m_j/m_i) s
        base_i = A * (m[p_j] * inv_m_i) * s
        A_full = Bdot_i * base_i
        A_plus = (-1.0) * base_i       # B'-(lam_i) = -1

        # momentum-conserving transpose term (Eq. 14): j's constraint reacting on i
        if conserve_momentum == 1 and is_dynamic[p_j] == 1:
            base_j = A * s             # (m_i/m_i) = 1 after momentum normalization
            Bdot_j = sisph_Bdot(lam[p_j], energy_type)
            A_full += Bdot_j * base_j
            A_plus += (-1.0) * base_j

        num += A_full * (-d)           # A_full (x_j - x_i)
        denom += A_plus

    if denom < 1e-6:
        denom = 1e-6
    dx = num / denom

    alpha = float(1.0)
    if use_line_search == 1:
        num_sq = wp.dot(num, num)
        if num_sq > 1e-20:
            dstar = pos_i - x_star[p_i]
            # psi / mom_i = 1/2 ||x - x*||^2 + mu B(lambda) dt^2 / m_i
            psi_over_mom = 0.5 * wp.dot(dstar, dstar) \
                + stiffness * sisph_Benergy(lam_i, energy_type) * (dt * dt * inv_m_i)
            alpha = psi_over_mom * denom / num_sq
            if alpha > 1.0:
                alpha = 1.0

    f_out[p_i] = pos_i + alpha * dx


@wp.kernel
def sisph_chebyshev_blend(
    f: wp.array(dtype=wp.vec3),        # fixed-point result f(x^k) = q^hat^{k+1}
    x_k: wp.array(dtype=wp.vec3),      # x^k = q^k
    x_km1: wp.array(dtype=wp.vec3),    # x^{k-1}
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    omega: float,
    gamma: float,
    x_out: wp.array(dtype=wp.vec3),
):
    """Chebyshev semi-iterative update with Wang [Wan15] under-relaxation (Eq. 21):
        x^{k+1} = omega ( gamma (f - x^k) + x^k - x^{k-1} ) + x^{k-1}.
    gamma < 1 damps the divergence that Chebyshev over-relaxation causes under very
    large stress (Wang Sec. 5). gamma = 1 recovers paper He et al. Eq. 22
    (= omega (f - x^{k-1}) + x^{k-1})."""
    i = wp.tid()
    if i >= num_particles:
        return
    if is_dynamic[i] == 0:
        return
    x_out[i] = omega * (gamma * (f[i] - x_k[i]) + x_k[i] - x_km1[i]) + x_km1[i]


@wp.kernel
def sisph_residual(
    f: wp.array(dtype=wp.vec3),
    x: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    g_out: wp.array(dtype=wp.vec3),
):
    """Anderson residual g = f(x) - x."""
    i = wp.tid()
    if i >= num_particles:
        return
    if is_dynamic[i] == 0:
        g_out[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    g_out[i] = f[i] - x[i]


@wp.kernel
def sisph_anderson_dots(
    g: wp.array(dtype=wp.vec3),        # g_k
    g_prev: wp.array(dtype=wp.vec3),   # g_{k-1}
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    out: wp.array(dtype=float),        # out[0] += <dg, g_k>, out[1] += <dg, dg>
):
    """Least-squares reduction for Anderson n=1: dg = g_k - g_{k-1}."""
    i = wp.tid()
    if i >= num_particles:
        return
    if is_dynamic[i] == 0:
        return
    dg = g[i] - g_prev[i]
    wp.atomic_add(out, 0, wp.dot(dg, g[i]))
    wp.atomic_add(out, 1, wp.dot(dg, dg))


@wp.kernel
def sisph_anderson_blend(
    f: wp.array(dtype=wp.vec3),        # f_k
    f_prev: wp.array(dtype=wp.vec3),   # f_{k-1}
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    omega: float,
    x_out: wp.array(dtype=wp.vec3),    # x^{k+1} = f_k + omega (f_{k-1} - f_k)
):
    """Anderson n=1 blend (paper Eq. 21)."""
    i = wp.tid()
    if i >= num_particles:
        return
    if is_dynamic[i] == 0:
        return
    x_out[i] = f[i] + omega * (f_prev[i] - f[i])


# ============================================================================
# SISPH Class
# ============================================================================

class SISPH(PressureBase):
    """Semi-Implicit SPH density solver (position-based)."""

    is_position_based = True

    def __init__(self, particle_system, neighbour_search, config):
        super().__init__(particle_system, neighbour_search, config)

        max_particles = self.ps.particle_max_num

        # mu: bulk-energy weight (paper Eq. 7), A+_ij = mu dt^2/rho0 |dW/dr|/r;
        # larger mu -> stiffer, more incompressible.
        self.stiffness = float(config.get_cfg("sisphStiffness", 1000000.0))

        # Bulk energy B0/B1/B2 (paper Sec. 4.3). 0 default (needs no line search).
        self.energy_type = int(config.get_cfg("sisphEnergyType", 0))

        # Momentum conservation (paper Eq. 14). On by default (PeriDyno default;
        # off oscillates, Fig. 4).
        self.conserve_momentum = bool(config.get_cfg("sisphConserveMomentum", True))

        # Line search alpha (paper Eqs. 19-20). Off by default (B0 doesn't need it).
        self.use_line_search = bool(config.get_cfg("sisphLineSearch", False))

        # Acceleration: 0 none, 1 Chebyshev, 2 Anderson (paper Sec. 4.3).
        self.accel_type = int(config.get_cfg("sisphAccel", 0))
        # Chebyshev spectral radius rho (paper uses 0.9).
        self.cheby_rho = float(config.get_cfg("sisphChebyRho", 0.9))
        # Chebyshev stabilization from Wang [Wan15] (cited by the paper for Chebyshev):
        self.cheby_delay = int(config.get_cfg("sisphChebyDelay", 10))
        self.cheby_gamma = float(config.get_cfg("sisphChebyGamma", 0.9))
        # Master toggle for the Wang stabilization. Off -> S=0, gamma=1.0 (i.e. the
        # plain paper He et al. Eq. 22 Chebyshev), which reproduces the velocity-spike
        # artifact under large stress. On (default) -> use cheby_delay / cheby_gamma.
        self.cheby_stabilize = bool(config.get_cfg("sisphChebyStabilize", True))

        # Inertial target x* = advected x, frozen for the whole solve (= self.s
        # from PressureBase, reused). Per-particle fixed-point result f(x^k).
        self.f = wp.zeros(max_particles, dtype=wp.vec3)
        self.lam = wp.zeros(max_particles, dtype=float)

        # Acceleration state buffers.
        self.x_km1 = wp.zeros(max_particles, dtype=wp.vec3)   # Chebyshev x^{k-1}
        self.g_curr = wp.zeros(max_particles, dtype=wp.vec3)  # Anderson g_k
        self.g_prev = wp.zeros(max_particles, dtype=wp.vec3)  # Anderson g_{k-1}
        self.f_prev = wp.zeros(max_particles, dtype=wp.vec3)  # Anderson f_{k-1}
        self.accel_dots = wp.zeros(2, dtype=float)
        self.accel_dots_host = wp.zeros(2, dtype=float, device="cpu")

        # x_prev (PressureBase) holds the start-of-iteration x^k: acceleration
        # reference and max-dx residual.

    def solve(self, dt):
        """Semi-implicit successive substitution density solve (Algorithm 1).
        Framework pre-advects and rebuilds neighbors first, so on entry ps.x is
        the predicted position (= x*), ps.x_old = x^t."""
        t_frame = time.perf_counter()
        n = self.ps.particle_num

        # Effective Chebyshev stabilization (Wang [Wan15]); disabled -> plain Eq. 22.
        _cheby_S = self.cheby_delay if self.cheby_stabilize else 0
        _cheby_gamma = self.cheby_gamma if self.cheby_stabilize else 1.0

        # x* = predicted position, frozen for the whole solve (self.s reused).
        wp.copy(self.s, self.ps.x, count=n)
        # Chebyshev x^{-1} := x^0 (so k=0 with omega=1 reproduces f_0).
        if self.accel_type == 1:
            wp.copy(self.x_km1, self.ps.x, count=n)

        opt_iter = 0
        err = 0.0
        pos_res = 0.0
        omega_prev = 1.0
        self.reset_stats()

        for _ in range(self.max_iteration_opt):
            t_iter = time.perf_counter()

            # --- density rho(cubic) + clamped lambda at current x ---
            wp.launch(
                kernel=sisph_density_lambda,
                dim=n,
                inputs=[
                    self.ps.x, self.ps.m, self.ps.density0, self.ps.is_dynamic, n,
                    self.ps.support_radius, int(self.ps.cache_size),
                    self.ps.neighbor_count, self.ps.neighbor_ids,
                ],
                outputs=[self.ps.density, self.lam, self.c],
            )

            # --- semi-implicit Jacobi step + line search -> f = f(x^k) ---
            wp.launch(
                kernel=sisph_jacobi_step,
                dim=n,
                inputs=[
                    self.ps.x, self.s, self.ps.m, self.lam, self.ps.density0,
                    self.ps.is_dynamic, n, self.ps.support_radius,
                    int(self.ps.cache_size), dt, self.stiffness,
                    self.energy_type,
                    1 if self.conserve_momentum else 0,
                    1 if self.use_line_search else 0,
                    self.ps.neighbor_count, self.ps.neighbor_ids,
                ],
                outputs=[self.f],
            )

            opt_iter += 1
            wp.copy(self.x_prev, self.ps.x, count=n)   # x^k: acceleration blend + dx residual

            # --- acceleration -> x^{k+1} ---
            if self.accel_type == 1:
                # Chebyshev (paper Eq. 22-23) with Wang [Wan15] delay S:
                rho2 = self.cheby_rho * self.cheby_rho
                c = opt_iter - _cheby_S
                if c <= 1:
                    omega = 1.0
                    omega_prev = 1.0
                elif c == 2:
                    omega = 2.0 / (2.0 - rho2)
                    omega_prev = omega
                else:
                    omega = 4.0 / (4.0 - rho2 * omega_prev)
                    omega_prev = omega
                # x^{k+1} = omega ( gamma (f - x^k) + x^k - x^{k-1} ) + x^{k-1};
                # then x^{k-1} <- x^k.  (gamma = Wang under-relaxation, Eq. 21.)
                wp.launch(
                    kernel=sisph_chebyshev_blend,
                    dim=n,
                    inputs=[self.f, self.x_prev, self.x_km1, self.ps.is_dynamic, n,
                            omega, _cheby_gamma],
                    outputs=[self.ps.x],
                )
                wp.copy(self.x_km1, self.x_prev, count=n)
            elif self.accel_type == 2:
                # Anderson n=1 (Eq. 21). g_k = f_k - x_k.
                wp.launch(
                    kernel=sisph_residual, dim=n,
                    inputs=[self.f, self.x_prev, self.ps.is_dynamic, n],
                    outputs=[self.g_curr],
                )
                if opt_iter == 1:
                    wp.copy(self.ps.x, self.f, count=n)         # x_1 = f_0
                else:
                    self.accel_dots.zero_()
                    wp.launch(
                        kernel=sisph_anderson_dots, dim=n,
                        inputs=[self.g_curr, self.g_prev, self.ps.is_dynamic, n],
                        outputs=[self.accel_dots],
                    )
                    wp.copy(self.accel_dots_host, self.accel_dots)
                    dvals = self.accel_dots_host.numpy()
                    numer, denom = float(dvals[0]), float(dvals[1])
                    omega = numer / denom if abs(denom) > 1e-20 else 0.0
                    wp.launch(
                        kernel=sisph_anderson_blend, dim=n,
                        inputs=[self.f, self.f_prev, self.ps.is_dynamic, n, omega],
                        outputs=[self.ps.x],
                    )
                wp.copy(self.g_prev, self.g_curr, count=n)
                wp.copy(self.f_prev, self.f, count=n)
            else:
                # no acceleration: x^{k+1} = f
                wp.copy(self.ps.x, self.f, count=n)

            # --- convergence: avg density error (from c at x^k) + max ||x^{k+1}-x^k|| / h ---
            err = self.compute_avg_density_error(self.c) / float(self.ps.cfg.density0)
            pos_res = self.compute_max_pos_update(self.x_prev)
            wp.synchronize()
            self.stats_density_errors.append(err)
            self.stats_pos_residuals.append(pos_res)
            self.stats_iter_costs.append((time.perf_counter() - t_iter) * 1000.0)

            metric, tol, _ = self.convergence_metric(err, pos_res)
            if (not self.use_fixed_iters) and \
                    (((metric < tol) and (opt_iter > 1)) or (opt_iter >= self.max_iteration_opt)):
                break

        # Reconstruct velocity from the corrected positions: v = (x - x_old)/dt,
        # and the divergence for the visualizer.
        self.update_velocity_from_positions(dt)
        self.compute_divergence()

        self.stats_outer_iters = opt_iter
        self.stats_total_cost_ms = (time.perf_counter() - t_frame) * 1000.0
        self.stats_final_density_error = err
        self.stats_final_pos_residual = pos_res

        self.check_converged(err, opt_iter, pos_res)
