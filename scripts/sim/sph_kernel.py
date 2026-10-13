"""
SPH kernel functions for WarpSPH simulation.
"""

import warp as wp


@wp.func
def cubic_kernel(r_norm: float, h: float) -> float:

    res = 0.0

    k = 8.0 / 3.14159265359  # 8/π
    k = k / (h * h * h)      # Normalize by h³
    q = r_norm / h

    if q <= 1.0:
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
        else:
            factor = 1.0 - q
            res = k * 2.0 * factor * factor * factor

    return res


@wp.func
def cubic_kernel_derivative(r: wp.vec3, h: float) -> wp.vec3:

    k = 8.0 / 3.14159265359  # 8/π
    k = 6.0 * k / (h * h * h)  # 6 * 8/π / h³

    r_norm = wp.length(r)
    q = r_norm / h

    res = wp.vec3(0.0, 0.0, 0.0)

    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)

        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q

    return res


@wp.func
def spiky_kernel_derivative(r: wp.vec3, h: float) -> wp.vec3:

    k = 45.0 / (3.14159265359 * h * h * h * h * h * h)
    r_norm = wp.length(r)
    if r_norm < 1e-6:
        r_norm = 1e-6
    grad_q = r / r_norm
    t = h - r_norm
    return -k * (t * t) * grad_q
