"""
Viscosity force for WarpSPH simulation (simple explicit version): the SPH
Laplacian viscosity with the K_ij neighbourhood-deficiency correction.
"""

import warp as wp


@wp.kernel
def compute_viscosity(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
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
    coeff: float,
    dt: float,
    # Material constants
    material_fluid: int
):
    """
    Compute the explicit SPH viscosity force.

    f_v = k_visc * coeff * (m_j / rho_j) * (v_ij . r) / (r^2 + eps^2) * grad_W

    Uses neighborhood deficiency correction: K_ij = 2 * rho0_i / (rho_i + rho_j)

    Args:
        x: Particle positions
        v: Particle velocities (updated in-place)
        m: Particle masses
        density: Current particle densities
        density0: Rest densities
        material: Material type per particle
        is_dynamic: Dynamic flag per particle
        num_particles: Number of active particles
        support_radius: SPH support radius (h)
        coeff: Viscosity coefficient
        dt: Time step
        material_fluid: Fluid material type constant
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
    rho_i = density[p_i]
    rho0_i = density0[p_i]

    # Regularization term to prevent division by zero
    eps2 = (0.01 * h) * (0.01 * h)

    # Viscosity coefficient: 2 * (dim + 2) where dim = 3
    k_visc = 2.0 * (3.0 + 2.0)  # = 10.0

    acc = wp.vec3(0.0, 0.0, 0.0)

    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for j in range(cnt):
        p_j = neighbor_ids[base + j]
        # Only fluid-fluid interaction for viscosity
        if material[p_j] != material_fluid:
            continue

        r = x_i - x[p_j]
        rn = wp.length(r)

        v_j = v[p_j]
        rho_j = density[p_j]

        # Velocity difference dot position difference
        v_ij = v_i - v_j
        vxy = wp.dot(v_ij, r)

        # Neighborhood deficiency correction K_ij (Akinci 2013)
        K_ij = 2.0 * rho0_i / (rho_i + rho_j)

        # Compute viscosity force
        # f_v = k_visc * coeff * (m_j / rho_j) * (v_ij . r) / (r^2 + eps^2) * grad_W
        grad_W = neighbor_gradW[base + j]
        f_v = k_visc * coeff * (m[p_j] / rho_j) * (vxy / (rn * rn + eps2)) * grad_W

        acc = acc + K_ij * f_v

    # Update velocity
    v[p_i] = v_i + acc * dt


class Viscosity:
    """
    Explicit SPH viscosity (Laplacian form with the K_ij deficiency correction).
    """

    def __init__(self, simulation_data, neighbour_search, config):
        """
        Initialize viscosity solver.

        Args:
            simulation_data: SimulationData instance
            neighbour_search: NeighbourSearch instance with HashGrid
            config: SimConfig instance
        """
        self.ps = simulation_data
        self.ns = neighbour_search
        self.cfg = config
        self.device = wp.get_device()

    def reset(self):
        """Reset solver state. Override in subclasses with internal buffers."""
        pass

    def solve(self, coeff: float, dt: float):
        """
        Apply viscosity force to fluid particles.

        Args:
            coeff: Viscosity coefficient (typically 0.01 - 0.1)
            dt: Time step
        """
        wp.launch(
            kernel=compute_viscosity,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.x,
                self.ps.v,
                self.ps.m,
                self.ps.density,
                self.ps.density0,
                self.ps.material,
                self.ps.is_dynamic,
                self.ps.particle_num,
                self.ps.support_radius,
                int(self.ps.cache_size),
                self.ps.neighbor_count,
                self.ps.neighbor_ids,
                self.ps.neighbor_gradW,
                coeff,
                dt,
                self.ps.material_fluid
            ],
            device=self.device
        )
