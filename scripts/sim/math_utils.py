"""Math utility kernels for SPH computations."""

import warp as wp


# ============================================================================
# Vector operations (vec3)
# ============================================================================

@wp.kernel
def add_vec3(
    ret: wp.array(dtype=wp.vec3),
    v0: wp.array(dtype=wp.vec3),
    scale: float,
    v1: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int
):
    """ret = v0 + scale * v1"""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    ret[i] = v0[i] + scale * v1[i]


@wp.kernel
def scale_vec3(
    ret: wp.array(dtype=wp.vec3),
    scale: float,
    v0: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int
):
    """ret = scale * v0"""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    ret[i] = scale * v0[i]


# ============================================================================
# Scalar operations
# ============================================================================

@wp.kernel
def add_scalar(
    ret: wp.array(dtype=float),
    v0: wp.array(dtype=float),
    scale: float,
    v1: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int
):
    """ret = v0 + scale * v1"""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    ret[i] = v0[i] + scale * v1[i]


@wp.kernel
def coef_wise_mul(
    ret: wp.array(dtype=float),
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int
):
    """ret = x * y (element-wise)"""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    ret[i] = x[i] * y[i]


@wp.kernel
def coef_wise_div(
    ret: wp.array(dtype=float),
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int
):
    """ret = x / y (element-wise)"""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    denom = y[i]
    if denom < 1e-6:
        denom = 1e-6
    ret[i] = x[i] / denom


@wp.kernel
def div_vec3_by_scalar(
    ret: wp.array(dtype=wp.vec3),
    x: wp.array(dtype=wp.vec3),
    y: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int
):
    """ret = x / y (vec3 / scalar)"""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    denom = y[i]
    if denom < 1e-6:
        denom = 1e-6
    ret[i] = x[i] / denom


@wp.kernel
def max_zero(
    x: wp.array(dtype=float),
    is_dynamic: wp.array(dtype=int),
    num_particles: int
):
    """x = max(x, 0) in-place"""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    x[i] = wp.max(x[i], 0.0)


# ============================================================================
# Fill and reductions
# ============================================================================

@wp.kernel
def fill_vec3(
    x: wp.array(dtype=wp.vec3),
    val: wp.vec3,
    is_dynamic: wp.array(dtype=int),
    num_particles: int
):
    """x = val for all dynamic particles"""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    x[i] = val


@wp.kernel
def dot_vec3(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    result: wp.array(dtype=float)
):
    """Dot product of vec3 arrays using atomic add."""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    wp.atomic_add(result, 0, wp.dot(a[i], b[i]))


@wp.kernel
def dot_vec3_tiled(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    result: wp.array(dtype=float),
):
    """Dot product of vec3 arrays using tile-based cooperative reduction."""
    i = wp.tid()
    val = 0.0
    if i < num_particles and is_dynamic[i] == 1:
        val = wp.dot(a[i], b[i])
    t = wp.tile(val)
    s = wp.tile_sum(t)
    wp.tile_atomic_add(result, s)


# ============================================================================
# CG helper
# ============================================================================

@wp.kernel
def pcg_update_x_r_scalar(
    x: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    alpha: float,
    p: wp.array(dtype=wp.vec3),
    Ap: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
):
    """Fused CG update: x += alpha*p, r -= alpha*Ap."""
    i = wp.tid()
    if i >= num_particles or is_dynamic[i] == 0:
        return
    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]
