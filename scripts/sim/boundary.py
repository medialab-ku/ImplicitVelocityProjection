"""
Boundary enforcement for WarpSPH simulation.

Handles particle-boundary collisions with reflection.
"""

import warp as wp


@wp.func
def simulate_collisions(v: wp.vec3, collision_normal: wp.vec3) -> wp.vec3:
    """
    Apply collision response to velocity.

    Args:
        v: Current velocity
        collision_normal: Normalized collision normal

    Returns:
        Updated velocity after collision
    """
    c_f = 0.5  # Restitution coefficient
    v_new = v - (1.0 + c_f) * wp.dot(v, collision_normal) * collision_normal
    return v_new


@wp.kernel
def enforce_boundary(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    domain_size: wp.vec3,
    padding: float
):
    """
    Enforce box boundary constraints with collision response.
    Clamps particles to domain boundaries and applies velocity reflection.

    Args:
        x: Particle positions
        v: Particle velocities
        is_dynamic: Dynamic flags (1 = dynamic, 0 = static)
        num_particles: Number of particles
        domain_size: Domain dimensions (width, height, depth)
        padding: Boundary padding distance
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    if is_dynamic[p_i] == 0:
        return

    pos = x[p_i]
    collision_normal = wp.vec3(0.0, 0.0, 0.0)

    # Check X boundaries
    if pos[0] > domain_size[0] - padding:
        collision_normal = collision_normal + wp.vec3(1.0, 0.0, 0.0)
        x[p_i] = wp.vec3(domain_size[0] - padding, pos[1], pos[2])
        pos = x[p_i]
    if pos[0] <= padding:
        collision_normal = collision_normal + wp.vec3(-1.0, 0.0, 0.0)
        x[p_i] = wp.vec3(padding, pos[1], pos[2])
        pos = x[p_i]

    # Check Y boundaries
    if pos[1] > domain_size[1] - padding:
        collision_normal = collision_normal + wp.vec3(0.0, 1.0, 0.0)
        x[p_i] = wp.vec3(pos[0], domain_size[1] - padding, pos[2])
        pos = x[p_i]
    if pos[1] <= padding:
        collision_normal = collision_normal + wp.vec3(0.0, -1.0, 0.0)
        x[p_i] = wp.vec3(pos[0], padding, pos[2])
        pos = x[p_i]

    # Check Z boundaries
    if pos[2] > domain_size[2] - padding:
        collision_normal = collision_normal + wp.vec3(0.0, 0.0, 1.0)
        x[p_i] = wp.vec3(pos[0], pos[1], domain_size[2] - padding)
        pos = x[p_i]
    if pos[2] <= padding:
        collision_normal = collision_normal + wp.vec3(0.0, 0.0, -1.0)
        x[p_i] = wp.vec3(pos[0], pos[1], padding)
        pos = x[p_i]

    # Apply collision response
    collision_normal_length = wp.length(collision_normal)
    if collision_normal_length > 1e-6:
        v[p_i] = simulate_collisions(v[p_i], collision_normal / collision_normal_length)
