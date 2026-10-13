"""
Surface tension force for WarpSPH simulation (simple explicit version).

Implements surface tension using:
- Cohesion force (fluid-fluid): Akinci 2013
- Curvature force (surface bending): Akinci 2013
- Adhesion force (fluid-rigid boundary): Akinci 2013
"""

import warp as wp


# ============================================================================
# Helper functions for surface tension kernels
# ============================================================================

@wp.func
def cohesion_term(r: wp.vec3, h: float) -> float:
    """
    Cohesion kernel C(r) from Akinci 2013.

    C(r) = (32 / (pi * h^9)) * {
        ((h - rn) * rn)^3 - h^6/64   if rn <= 0.5*h
        ((h - rn) * rn)^3            if 0.5*h < rn <= h
        0                             otherwise
    }

    Args:
        r: Position difference vector
        h: Support radius

    Returns:
        Cohesion kernel value
    """
    rn = wp.length(r)
    res = 0.0

    k = 32.0 / (3.14159265359 * wp.pow(h, 9.0))

    if rn <= 0.0 or rn > h:
        res = 0.0
    elif rn > 0.5 * h:
        t = (h - rn) * rn
        res = k * (t * t * t)
    else:
        t = 2.0 * (h - rn) * rn
        h6_64 = wp.pow(h, 6.0) / 64.0
        res = k * ((t * t * t) - h6_64)

    return res


@wp.func
def adhesion_term(r: wp.vec3, h: float) -> float:
    """
    Adhesion kernel A(r) from Akinci 2013.

    A(r) = (0.007 / h^3.25) * sqrt(sqrt(-4*rn^2/h + 6*rn - 2*h))
           for 0.5*h < rn <= h, else 0

    Args:
        r: Position difference vector
        h: Support radius

    Returns:
        Adhesion kernel value
    """
    rn = wp.length(r)
    res = 0.0

    k = 0.007 / wp.pow(h, 3.25)
    arg = -4.0 * rn * rn / h + 6.0 * rn - 2.0 * h
    arg = wp.max(arg, 0.0)

    if rn <= h and rn > 0.5 * h:
        # Fourth root: sqrt(sqrt(arg))
        res = k * wp.sqrt(wp.sqrt(arg))

    return res


# ============================================================================
# Surface tension kernels
# ============================================================================

@wp.kernel
def compute_surface_normals(
    m: wp.array(dtype=float),
    density: wp.array(dtype=float),
    material: wp.array(dtype=int),
    num_particles: int,
    support_radius: float,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
    material_fluid: int,
    # Output
    n: wp.array(dtype=wp.vec3)
):
    """
    Compute surface normals for fluid particles.

    n_i = h * sum_j (m_j / rho_j) * grad_W(x_i - x_j, h)

    The normal field is used for curvature-based surface tension force.

    Args:
        m: Particle masses
        density: Particle densities
        material: Material type per particle
        num_particles: Number of active particles
        support_radius: SPH support radius (h)
        material_fluid: Fluid material type constant
        n: Output surface normal vectors
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    # Only compute normals for fluid particles
    if material[p_i] != material_fluid:
        n[p_i] = wp.vec3(0.0, 0.0, 0.0)
        return

    h = support_radius
    normal = wp.vec3(0.0, 0.0, 0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        # Only consider fluid neighbors for normal computation
        if material[p_j] != material_fluid:
            continue

        rho_j = density[p_j]
        grad_W = neighbor_gradW[base + j]

        # Accumulate normal contribution
        normal = normal + (m[p_j] / (rho_j + 1e-12)) * grad_W

    # Scale by support radius
    n[p_i] = h * normal


@wp.kernel
def compute_surface_tension(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    density: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    n: wp.array(dtype=wp.vec3),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    support_radius: float,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    coeff_sf: float,
    coeff_adh: float,
    dt: float,
    material_fluid: int,
    material_rigid: int
):
    """
    Compute surface tension forces (cohesion + curvature + adhesion).

    Fluid-Fluid interactions:
    - Cohesion: f_coh = -coeff_sf * m_i * m_j * C(r) * r_hat
    - Curvature: f_curv = -coeff_sf * m_i * (n_i - n_j)

    Fluid-Rigid interactions:
    - Adhesion: f_adh = -coeff_adh * m_i * m_j * A(r) * r_hat

    All with neighborhood deficiency correction: K_ij = 2 * rho0_i / (rho_i + rho_j)

    Args:
        x: Particle positions
        v: Particle velocities (updated in-place)
        m: Particle masses
        density: Current particle densities
        density0: Rest densities
        n: Surface normal vectors (precomputed)
        material: Material type per particle
        is_dynamic: Dynamic flag per particle
        num_particles: Number of active particles
        support_radius: SPH support radius (h)
        coeff_sf: Surface tension coefficient
        coeff_adh: Adhesion coefficient
        dt: Time step
        material_fluid: Fluid material type constant
        material_rigid: Rigid material type constant
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    # Skip static rigid particles
    if is_dynamic[p_i] == 0:
        return

    # Only process fluid particles
    if material[p_i] != material_fluid:
        return

    h = support_radius
    x_i = x[p_i]
    v_i = v[p_i]
    n_i = n[p_i]
    rho_i = density[p_i]
    rho0_i = density0[p_i]
    m_i = m[p_i]

    acc = wp.vec3(0.0, 0.0, 0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        x_j = x[p_j]
        r = x_i - x_j
        rn = wp.length(r)

        # Unit direction vector (protected against zero distance)
        inv_rn = 1.0 / wp.max(rn, 1e-6 * h)
        r_hat = r * inv_rn

        # ---------- Fluid-Fluid: Cohesion & Curvature ----------
        if material[p_j] == material_fluid:
            n_j = n[p_j]
            rho_j = density[p_j]
            m_j = m[p_j]

            # Cohesion force
            C_ij = cohesion_term(r, h)
            f_coh = -coeff_sf * m_i * m_j * C_ij * r_hat

            # Curvature force
            f_curv = -coeff_sf * m_i * (n_i - n_j)

            # Neighborhood deficiency correction K_ij (Akinci 2013)
            K_ij = 2.0 * rho0_i / (rho_i + rho_j)

            acc = acc + K_ij * (f_coh + f_curv)

        # ---------- Fluid-Rigid: Adhesion ----------
        elif material[p_j] == material_rigid:
            m_j = m[p_j]

            # Adhesion force
            A_ij = adhesion_term(r, h)
            f_adh = -coeff_adh * m_i * m_j * A_ij * r_hat

            acc = acc + f_adh

    # Update velocity
    v[p_i] = v_i + acc * dt


class SurfaceTension:
    """
    Surface tension force solver for SPH simulation.
    Implements surface tension using cohesion, curvature, and adhesion forces
    from Akinci 2013.
    """

    def __init__(self, simulation_data, neighbour_search, config):
        """
        Initialize surface tension solver.

        Args:
            simulation_data: SimulationData instance
            neighbour_search: NeighbourSearch instance with HashGrid
            config: SimConfig instance
        """
        self.ps = simulation_data
        self.ns = neighbour_search
        self.cfg = config
        self.device = wp.get_device()

    def solve(self, coeff_sf: float, coeff_adh: float, dt: float):
        """
        Apply surface tension forces to fluid particles.

        Two-phase computation:
        1. Compute surface normals for all fluid particles
        2. Compute cohesion, curvature (fluid-fluid) and adhesion (fluid-rigid) forces

        Args:
            coeff_sf: Surface tension coefficient (typically 0.01 - 0.1)
            coeff_adh: Adhesion coefficient (typically same as coeff_sf)
            dt: Time step
        """
        # Phase 1: Compute surface normals
        wp.launch(
            kernel=compute_surface_normals,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.m,
                self.ps.density,
                self.ps.material,
                self.ps.particle_num,
                self.ps.support_radius,
                int(self.ps.cache_size),
                self.ps.neighbor_count,
                self.ps.neighbor_ids,
                self.ps.neighbor_gradW,
                self.ps.material_fluid
            ],
            outputs=[self.ps.n],
            device=self.device
        )

        # Phase 2: Compute surface tension forces (cohesion + curvature + adhesion)
        wp.launch(
            kernel=compute_surface_tension,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.x,
                self.ps.v,
                self.ps.m,
                self.ps.density,
                self.ps.density0,
                self.ps.n,
                self.ps.material,
                self.ps.is_dynamic,
                self.ps.particle_num,
                self.ps.support_radius,
                int(self.ps.cache_size),
                self.ps.neighbor_count,
                self.ps.neighbor_ids,
                coeff_sf,
                coeff_adh,
                dt,
                self.ps.material_fluid,
                self.ps.material_rigid
            ],
            device=self.device
        )
