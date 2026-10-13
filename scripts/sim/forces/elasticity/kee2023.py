"""Kee 2023 elasticity solver for WarpSPH (Kee et al., Eurographics 2023)."""

import warp as wp
import numpy as np

from scripts.sim.sph_kernel import cubic_kernel, cubic_kernel_derivative


# Material type codes (mirror SPlisHSPlasH Elasticity_Kee2023::m_materialType)
MATERIAL_COROTATED = 0
MATERIAL_SNH = 1
MATERIAL_NAMES = {MATERIAL_COROTATED: "Co-rotated", MATERIAL_SNH: "Stable NH"}

# Body-entry keys that override the scene's material for that body.
MATERIAL_KEYS = ("youngsModulus", "poissonsRatio")


def lame_parameters(E: float, nu: float):
    """(μ, λ) from Young's modulus E and Poisson's ratio ν."""
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


# ============================================================================
# Init kernels (every kernel launches dim = #deformables; p_i = ids[tid])
# ============================================================================


@wp.kernel
def snapshot_rest_neighbors_kernel(
    ids: wp.array(dtype=int),
    object_id: wp.array(dtype=int),
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    reg_id: wp.array(dtype=int),
    cache_size: int,
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
):
    """Copy current neighbors into the rest snapshot: same registration batch AND same object only,
    so a body can never be glued to an earlier one that happens to share its object id."""
    p_i = ids[wp.tid()]

    obj_i = object_id[p_i]
    reg_i = reg_id[p_i]
    n_i = neighbor_count[p_i]
    count = int(0)
    for k in range(n_i):
        p_j = neighbor_ids[p_i * cache_size + k]
        if reg_id[p_j] == reg_i and object_id[p_j] == obj_i and p_j != p_i:
            rest_neighbor_ids[p_i * cache_size + count] = p_j
            count = count + 1
    rest_neighbor_count[p_i] = count


@wp.kernel
def symmetrize_rest_neighbors_kernel(
    ids: wp.array(dtype=int),
    cache_size: int,
    snapshot_count: wp.array(dtype=int),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
):
    """For each (i, j) in the snapshot, ensure (j, i) is also present.

    `snapshot_count` is the post-snapshot, pre-symmetrize count read at kernel
    entry; iterating against it (instead of the live `rest_neighbor_count`) avoids
    re-symmetrizing neighbors added by this same pass.

    Each particle i is the sole writer of "i into j's list", so the per-thread
    add for a given (i, j) pair is unique — the atomic_add on j's count claims a
    slot; if the slot is in range it is written, otherwise it is left for the
    overflow check (Python side) to flag.
    """
    p_i = ids[wp.tid()]

    n_i = snapshot_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        # Is p_i already in p_j's list?
        n_j = snapshot_count[p_j]
        found = int(0)
        for m in range(n_j):
            if rest_neighbor_ids[p_j * cache_size + m] == p_i:
                found = 1
        if found == 0:
            slot = wp.atomic_add(rest_neighbor_count, p_j, 1)
            if slot < cache_size:
                rest_neighbor_ids[p_j * cache_size + slot] = p_i


@wp.kernel
def compute_V0_rest_kernel(
    ids: wp.array(dtype=int),
    x0: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    support_radius: float,
    V0_rest: wp.array(dtype=float),
):
    """V0_i = 1 / ( W(0) + sum_{j in N_i^0} W(|x_ij0|) )"""
    p_i = ids[wp.tid()]

    x_i0 = x0[p_i]
    sum_W = cubic_kernel(0.0, support_radius)
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        r_ij0 = x_i0 - x0[p_j]
        sum_W = sum_W + cubic_kernel(wp.length(r_ij0), support_radius)

    V0_rest[p_i] = 1.0 / sum_W


@wp.kernel
def compute_rest_density_kernel(
    ids: wp.array(dtype=int),
    x0: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    support_radius: float,
    density0: wp.array(dtype=float),
):
    """Per-particle rest density from the body's own rest configuration:
        ρ_rest,i = m_i W(0) + Σ_{j∈N0(i)} m_j W(|x_i0 - x_j0|)
    Stored as density0_i, so the density constraint c_i = ρ_i - density0_i is zero at rest and
    turns positive as soon as another body (or a wall) enters the kernel support."""
    p_i = ids[wp.tid()]
    x_i0 = x0[p_i]
    rho = m[p_i] * cubic_kernel(0.0, support_radius)
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        rho = rho + m[p_j] * cubic_kernel(wp.length(x_i0 - x0[p_j]), support_radius)
    density0[p_i] = rho


@wp.kernel
def compute_matrix_L_kernel(
    ids: wp.array(dtype=int),
    x0: wp.array(dtype=wp.vec3),
    V0_rest: wp.array(dtype=float),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    support_radius: float,
    L: wp.array(dtype=wp.mat33),
    rank: wp.array(dtype=int),
):
    """L_i = pinv( sum_j V0_j gradW(x_ij0) (x) (-x_ij0) )    [x_ij0 = x_i0 - x_j0]

    Pseudo-inverse via SVD: a rest neighbourhood that is coplanar/collinear (thin
    features, few neighbours) cannot measure strain along its null directions.
    rank_i = number of supported directions (3 = full); the polar kernels complete
    the missing modes of F covariantly (see _complete_null_modes) so such particles
    are strain-free at rest and under rigid motion, instead of wp.inverse's silent
    zero matrix.
    """
    p_i = ids[wp.tid()]

    x_i0 = x0[p_i]
    Dm = wp.mat33(0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0)
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        r_ij0 = x_i0 - x0[p_j]
        gW = cubic_kernel_derivative(r_ij0, support_radius)
        Dm = Dm + V0_rest[p_j] * wp.outer(gW, -r_ij0)

    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s = wp.vec3(0.0, 0.0, 0.0)
    wp.svd3(Dm, U, s, V)
    s_max = wp.max(wp.abs(s[0]), wp.max(wp.abs(s[1]), wp.abs(s[2])))
    tol = 1.0e-4 * s_max
    inv = wp.vec3(0.0, 0.0, 0.0)
    keep = wp.vec3(0.0, 0.0, 0.0)
    for k in range(3):
        if wp.abs(s[k]) > tol:
            inv[k] = 1.0 / s[k]
            keep[k] = 1.0
    L[p_i] = V @ wp.diag(inv) @ wp.transpose(U)
    rank[p_i] = int(keep[0] + keep[1] + keep[2])


@wp.kernel
def precompute_VjLi_gradW_kernel(
    ids: wp.array(dtype=int),
    x0: wp.array(dtype=wp.vec3),
    V0_rest: wp.array(dtype=float),
    L: wp.array(dtype=wp.mat33),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    support_radius: float,
    VjLi_gradW: wp.array(dtype=wp.vec3),
):
    """Cache V_j * (L_i @ gradW(x_ij0)) per rest neighbor pair, used by compute_F."""
    p_i = ids[wp.tid()]

    x_i0 = x0[p_i]
    L_i = L[p_i]
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        r_ij0 = x_i0 - x0[p_j]
        gW = cubic_kernel_derivative(r_ij0, support_radius)
        VjLi_gradW[p_i * cache_size + k] = V0_rest[p_j] * (L_i @ gW)


@wp.kernel
def precompute_K_ze_kernel(
    ids: wp.array(dtype=int),
    x0: wp.array(dtype=wp.vec3),
    V0_rest: wp.array(dtype=float),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    support_radius: float,
    alpha_ze: float,
    mu: wp.array(dtype=float),
    K_ze: wp.array(dtype=float),
):
    """Per-pair ZE stiffness (as in SPlisHSPlasH Elasticity_Kee2023):
        K_ij = α μ_i V0_i V0_j W(|x_ij0|) / |x_ij0|²

    α and μ are baked in here. Refresh via refresh_K_ze() if either changes.
    Material-agnostic: same K_ze for Co-rotated and SNH (ZE penalty acts on the
    non-affine pair-displacement residual, not on the energy density Ψ).
    """
    p_i = ids[wp.tid()]

    x_i0 = x0[p_i]
    V_i = V0_rest[p_i]
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        r_ij0 = x_i0 - x0[p_j]
        r2 = wp.dot(r_ij0, r_ij0)
        W = cubic_kernel(wp.sqrt(r2), support_radius)
        if r2 > 1.0e-20:
            K_ze[p_i * cache_size + k] = alpha_ze * mu[p_i] * V_i * V0_rest[p_j] * W / r2
        else:
            K_ze[p_i * cache_size + k] = 0.0


@wp.kernel
def precompute_G_sum_kernel(
    ids: wp.array(dtype=int),
    rest_neighbor_count: wp.array(dtype=int),
    VjLi_gradW: wp.array(dtype=wp.vec3),
    cache_size: int,
    G_sum: wp.array(dtype=wp.vec3),
):
    """G_sum[i] = Σ_l G_il, cached so the scatter kernel doesn't re-traverse
    N(i) to compute the i-side coefficient (-1 + G_sum[i] · d_ij)."""
    p_i = ids[wp.tid()]
    n_i = rest_neighbor_count[p_i]
    s = wp.vec3(0.0, 0.0, 0.0)
    for k in range(n_i):
        s = s + VjLi_gradW[p_i * cache_size + k]
    G_sum[p_i] = s


@wp.kernel
def precompute_Vj_gradW_kernel(
    ids: wp.array(dtype=int),
    x0: wp.array(dtype=wp.vec3),
    V0_rest: wp.array(dtype=float),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    support_radius: float,
    Vj_gradW: wp.array(dtype=wp.vec3),
):
    """Cache V_j * gradW(x_ij0) per rest neighbor pair (no L_i correction).

    Used by the gradient and matvec where L_i applies to P_i (not the kernel
    gradient): force_i ∝ Σ_j (P_i L_i + P_j L_j) · (V_j ∇W_ij).
    """
    p_i = ids[wp.tid()]

    x_i0 = x0[p_i]
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        r_ij0 = x_i0 - x0[p_j]
        gW = cubic_kernel_derivative(r_ij0, support_radius)
        Vj_gradW[p_i * cache_size + k] = V0_rest[p_j] * gW


@wp.kernel
def compute_F_kernel(
    ids: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    VjLi_gradW: wp.array(dtype=wp.vec3),
    cache_size: int,
    F: wp.array(dtype=wp.mat33),
):
    """F_i = sum_j (x_j - x_i) (x) V_j L_i gradW(x_ij0)   (null modes of a rank-deficient
    rest set are exactly 0 here; the polar kernels complete them)"""
    p_i = ids[wp.tid()]

    x_i = x[p_i]
    F_i = wp.mat33(0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0)
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        F_i = F_i + wp.outer(x[p_j] - x_i, VjLi_gradW[p_i * cache_size + k])

    F[p_i] = F_i


# ============================================================================
# Per-step kernels (co-rotated or Stable Neo-Hookean, velocity-based Newton + PCG)
# ============================================================================


@wp.kernel
def compute_x_tilde_kernel(
    ids: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    dt: float,
    x_tilde: wp.array(dtype=wp.vec3),
):
    """Inertial predictor: x̃_i = x_i + dt v_i."""
    p_i = ids[wp.tid()]
    x_tilde[p_i] = x[p_i] + dt * v[p_i]


@wp.func
def _complete_null_modes(s: wp.vec3, rank: int) -> wp.vec3:
    """Set the (3 - rank) structurally-zero singular values of F to 1: directions the
    rest neighbourhood cannot observe read as undeformed and rotate with the body."""
    out = wp.vec3(s[0], s[1], s[2])
    if rank >= 3:
        return out
    a0 = wp.abs(s[0])
    a1 = wp.abs(s[1])
    a2 = wp.abs(s[2])
    if rank == 2:
        if a0 <= a1 and a0 <= a2:
            out[0] = 1.0
        elif a1 <= a2:
            out[1] = 1.0
        else:
            out[2] = 1.0
    elif rank == 1:
        if a0 >= a1 and a0 >= a2:
            out[1] = 1.0
            out[2] = 1.0
        elif a1 >= a2:
            out[0] = 1.0
            out[2] = 1.0
        else:
            out[0] = 1.0
            out[1] = 1.0
    else:
        out = wp.vec3(1.0, 1.0, 1.0)
    return out


@wp.kernel
def compute_polar_P_PL_kernel(
    ids: wp.array(dtype=int),
    F: wp.array(dtype=wp.mat33),
    L: wp.array(dtype=wp.mat33),
    rank: wp.array(dtype=int),
    mu: wp.array(dtype=float),
    lambda_: wp.array(dtype=float),
    U_out: wp.array(dtype=wp.mat33),
    V_out: wp.array(dtype=wp.mat33),
    sigma_out: wp.array(dtype=wp.vec3),
    R_out: wp.array(dtype=wp.mat33),
    PL_out: wp.array(dtype=wp.mat33),
):
    """SVD F → (U, σ, V), fix det(U V^T) = +1, compute R, FCR stress P, and PL = P·L.

    Co-rotated P (Kee2023 / SPlisHSPlasH form):
        P = 2μ(F - R) + λ(tr(R^T F) - 3) R
    Storing PL = P·L avoids re-multiplying L in the matvec / gradient.

    Caches U, V, σ per particle so the per-particle 9×9 Hessian can be applied
    on the fly in the PCG matvec without re-doing the SVD each iteration.
    """
    p_i = ids[wp.tid()]

    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sigma = wp.vec3(0.0, 0.0, 0.0)
    wp.svd3(F[p_i], U, sigma, V)

    # Reflection fix: ensure R = U V^T has det = +1.
    # If det(U V^T) < 0, flip the last column of U and negate the last σ.
    R_try = U @ wp.transpose(V)
    if wp.determinant(R_try) < 0.0:
        U = wp.mat33(U[0, 0], U[0, 1], -U[0, 2],
                     U[1, 0], U[1, 1], -U[1, 2],
                     U[2, 0], U[2, 1], -U[2, 2])
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])

    R = U @ wp.transpose(V)

    F_i = F[p_i]
    if rank[p_i] < 3:
        # Rank-deficient rest set: complete the null modes covariantly, store F back
        # (the ZE build, the SNH Hessian and the preconditioner read the completed F).
        sigma = _complete_null_modes(sigma, rank[p_i])
        F_i = U @ wp.diag(sigma) @ wp.transpose(V)
        F[p_i] = F_i
    trRtF = (R[0, 0] * F_i[0, 0] + R[1, 0] * F_i[1, 0] + R[2, 0] * F_i[2, 0]
           + R[0, 1] * F_i[0, 1] + R[1, 1] * F_i[1, 1] + R[2, 1] * F_i[2, 1]
           + R[0, 2] * F_i[0, 2] + R[1, 2] * F_i[1, 2] + R[2, 2] * F_i[2, 2])

    P = 2.0 * mu[p_i] * (F_i - R) + lambda_[p_i] * (trRtF - 3.0) * R

    U_out[p_i] = U
    V_out[p_i] = V
    sigma_out[p_i] = sigma
    R_out[p_i] = R
    PL_out[p_i] = P @ L[p_i]


@wp.kernel
def compute_polar_P_PL_snh_kernel(
    ids: wp.array(dtype=int),
    F: wp.array(dtype=wp.mat33),
    L: wp.array(dtype=wp.mat33),
    rank: wp.array(dtype=int),
    mu: wp.array(dtype=float),
    lambda_: wp.array(dtype=float),
    U_out: wp.array(dtype=wp.mat33),
    V_out: wp.array(dtype=wp.mat33),
    sigma_out: wp.array(dtype=wp.vec3),
    R_out: wp.array(dtype=wp.mat33),
    PL_out: wp.array(dtype=wp.mat33),
):
    """Stable Neo-Hookean (Smith 2018) first-Piola stress and PL = P·L.

        P = μ(1 - 1/(I_C+1)) F + λ(J - α) cof(F)
        α = 1 + μ/λ - μ/(4λ),  I_C = ‖F‖²_F,  J = det(F)

    R is not needed by P here, but the SNH Hessian matvec also reuses (U, σ, V)
    so we still run the SVD and stash R for buffer-shape compatibility.
    """
    p_i = ids[wp.tid()]

    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sigma = wp.vec3(0.0, 0.0, 0.0)
    wp.svd3(F[p_i], U, sigma, V)

    R_try = U @ wp.transpose(V)
    if wp.determinant(R_try) < 0.0:
        U = wp.mat33(U[0, 0], U[0, 1], -U[0, 2],
                     U[1, 0], U[1, 1], -U[1, 2],
                     U[2, 0], U[2, 1], -U[2, 2])
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])

    R = U @ wp.transpose(V)

    F_i = F[p_i]
    if rank[p_i] < 3:
        sigma = _complete_null_modes(sigma, rank[p_i])
        F_i = U @ wp.diag(sigma) @ wp.transpose(V)
        F[p_i] = F_i
    f0 = wp.vec3(F_i[0, 0], F_i[1, 0], F_i[2, 0])
    f1 = wp.vec3(F_i[0, 1], F_i[1, 1], F_i[2, 1])
    f2 = wp.vec3(F_i[0, 2], F_i[1, 2], F_i[2, 2])

    # cof(F) columns = cross products of F columns
    c0 = wp.cross(f1, f2)
    c1 = wp.cross(f2, f0)
    c2 = wp.cross(f0, f1)
    cofF = wp.mat33(c0[0], c1[0], c2[0],
                    c0[1], c1[1], c2[1],
                    c0[2], c1[2], c2[2])

    I_C = (f0[0]*f0[0] + f0[1]*f0[1] + f0[2]*f0[2]
         + f1[0]*f1[0] + f1[1]*f1[1] + f1[2]*f1[2]
         + f2[0]*f2[0] + f2[1]*f2[1] + f2[2]*f2[2])
    J = wp.determinant(F_i)
    mu_i = mu[p_i]
    lam_i = lambda_[p_i]
    alpha = 1.0 + mu_i / lam_i - mu_i / (4.0 * lam_i)

    P = mu_i * (1.0 - 1.0 / (I_C + 1.0)) * F_i + lam_i * (J - alpha) * cofF

    U_out[p_i] = U
    V_out[p_i] = V
    sigma_out[p_i] = sigma
    R_out[p_i] = R
    PL_out[p_i] = P @ L[p_i]


@wp.kernel
def compute_gradient_kernel(
    ids: wp.array(dtype=int),
    V0_rest: wp.array(dtype=float),
    PL: wp.array(dtype=wp.mat33),
    Vj_gradW: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    gradient: wp.array(dtype=wp.vec3),
):
    """Variational gradient of the elastic energy w.r.t. position:
        ∇Ψ_k = -V_k Σ_j V_j (P_k L_k + P_j L_j) ∇W_kj

    Sign / structure derivation (using F_i[a,b] = Σ_j (x_j - x_i)[a] G_ij[b]
    with G_ij = V_j L_i ∇W_ij):
        ∂E/∂x_k = Σ_i V_i P_i : ∂F_i/∂x_k
                = -V_k Σ_j V_j (P_k L_k + P_j L_j) ∇W_kj
        (rest-neighbor symmetry collapses the i-side sum to k's own neighbors)
    """
    p_i = ids[wp.tid()]

    V_i = V0_rest[p_i]
    PL_i = PL[p_i]
    accum = wp.vec3(0.0, 0.0, 0.0)
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        g = Vj_gradW[p_i * cache_size + k]
        accum = accum + (PL_i + PL[p_j]) @ g

    gradient[p_i] = -V_i * accum


# ----------------------------------------------------------------------------
# Zero-energy mode (hourglass) penalty — matrix-free, GATHER form, no atomics.
#
# E_ZE = (1/2) Σ_i Σ_{j∈N(i)} K_ij · |r_ij|²,   r_ij = (x_j − x_i) − F_i d_ij
#
# The force ∇E_ZE = S_ZE·x (S_ZE = H̃ K H̃ᵀ) factors per owner through a single
# per-particle 3×3 matrix Y_i = Σ_j y_ij ⊗ d_ij  (y_ij = K_ij r_ij):
#
#   force on p = (Y_p·G_sum[p] − Ysum_p)            # p as owner of its pairs
#              + Σ_{i ∈ N(p)} (y_ip − Y_i·G_ip)     # p as neighbour of owner i
#
# Computed in TWO kernels, each with ONE inner loop, NO atomics:
#
#   build : per owner i, loop N(i) once → store y_ij, accumulate Y_i, Ysum_i
#   gather: per particle p, loop N(p) once → read y_ip, Y_i, G_ip and assemble
#
# The gather needs, for each (p,k) with i = N(p)[k], the slot m where p sits in
# N(i) — so it can fetch y_ip = y_ze[i·C+m] and G_ip = VjLi_gradW[i·C+m]. That
# reverse-slot map is a rest-config constant, built once in initialize().
#
# Each thread writes only its own grad[p] → no atomic contention. Translation
# invariance holds because G_sum[i] = Σ_k G_ik exactly (same precompute order).
# ----------------------------------------------------------------------------


@wp.kernel
def build_reverse_slot_kernel(
    ids: wp.array(dtype=int),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    reverse_slot: wp.array(dtype=int),
):
    """reverse_slot[i·C+k] = m  such that  N(j)[m] = i,  where j = N(i)[k].

    Uses rest-neighbour symmetry (j ∈ N(i) ⟺ i ∈ N(j), enforced in
    initialize()). If not found (shouldn't happen post-symmetrize) leaves -1.
    """
    p_i = ids[wp.tid()]
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        n_j = rest_neighbor_count[p_j]
        m_found = int(-1)
        for m in range(n_j):
            if rest_neighbor_ids[p_j * cache_size + m] == p_i:
                m_found = m
        reverse_slot[p_i * cache_size + k] = m_found


@wp.kernel
def ze_build_Y_gradient_kernel(
    ids: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    F: wp.array(dtype=wp.mat33),
    K_ze: wp.array(dtype=float),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    y_ze: wp.array(dtype=wp.vec3),
    Y_ze: wp.array(dtype=wp.mat33),
    Ysum_ze: wp.array(dtype=wp.vec3),
):
    """Build pass (gradient): per owner i, one loop over N(i):
        store y_ij = K_ij r_ij,  Y_i = Σ_j y_ij ⊗ d_ij,  Ysum_i = Σ_j y_ij,
    using the material F_i."""
    p_i = ids[wp.tid()]
    x_i = x[p_i]
    x_i0 = x0[p_i]
    F_i = F[p_i]
    n_i = rest_neighbor_count[p_i]

    Y_i = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Ysum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        d_ij = x0[p_j] - x_i0
        r = (x[p_j] - x_i) - F_i @ d_ij
        K = K_ze[p_i * cache_size + k]
        y = K * r
        y_ze[p_i * cache_size + k] = y
        Y_i = Y_i + wp.outer(y, d_ij)
        Ysum = Ysum + y

    Y_ze[p_i] = Y_i
    Ysum_ze[p_i] = Ysum


@wp.kernel
def ze_build_Y_matvec_kernel(
    ids: wp.array(dtype=int),
    p_in: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    K_ze: wp.array(dtype=float),
    VjLi_gradW: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    y_ze: wp.array(dtype=wp.vec3),
    Y_ze: wp.array(dtype=wp.mat33),
    Ysum_ze: wp.array(dtype=wp.vec3),
):
    """Build pass (matvec): same as the gradient build but the residual uses
    δF_i = Σ_l (p_l − p_i) ⊗ G_il in place of the material F_i."""
    p_i = ids[wp.tid()]
    p_i_vec = p_in[p_i]
    x_i0 = x0[p_i]
    n_i = rest_neighbor_count[p_i]

    dF = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for k in range(n_i):
        p_l = rest_neighbor_ids[p_i * cache_size + k]
        dF = dF + wp.outer(p_in[p_l] - p_i_vec, VjLi_gradW[p_i * cache_size + k])

    Y_i = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Ysum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        d_ij = x0[p_j] - x_i0
        z = (p_in[p_j] - p_i_vec) - dF @ d_ij
        y = K_ze[p_i * cache_size + k] * z
        y_ze[p_i * cache_size + k] = y
        Y_i = Y_i + wp.outer(y, d_ij)
        Ysum = Ysum + y

    Y_ze[p_i] = Y_i
    Ysum_ze[p_i] = Ysum


@wp.kernel
def ze_gather_kernel(
    ids: wp.array(dtype=int),
    y_ze: wp.array(dtype=wp.vec3),
    Y_ze: wp.array(dtype=wp.mat33),
    Ysum_ze: wp.array(dtype=wp.vec3),
    G_sum: wp.array(dtype=wp.vec3),
    VjLi_gradW: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    reverse_slot: wp.array(dtype=int),
    cache_size: int,
    scale: float,
    out: wp.array(dtype=wp.vec3),
):
    """Gather pass (shared by gradient & matvec — reads only the build outputs):

        out[p] += scale · [ (Y_p·G_sum[p] − Ysum_p)
                            + Σ_{i ∈ N(p)} (y_ip − Y_i·G_ip) ]

    Each thread owns out[p] exclusively — no atomics. y_ip and G_ip come from
    owner i's pair slot m = reverse_slot[p·C+k] (where p = N(i)[m])."""
    p_i = ids[wp.tid()]
    n_i = rest_neighbor_count[p_i]

    # p as owner of its own pairs
    f = Y_ze[p_i] @ G_sum[p_i] - Ysum_ze[p_i]

    # p as neighbour of each owner i ∈ N(p)
    for k in range(n_i):
        p_owner = rest_neighbor_ids[p_i * cache_size + k]
        m = reverse_slot[p_i * cache_size + k]
        slot = p_owner * cache_size + m
        y_ip = y_ze[slot]
        G_ip = VjLi_gradW[slot]
        f = f + (y_ip - Y_ze[p_owner] @ G_ip)

    out[p_i] = out[p_i] + scale * f


# ----------------------------------------------------------------------------
# Matrix-free Newton matvec: A·p = M·p + dt² K·p
#
# K·p is the action of ∇²Ψ on a 3-vector-per-particle field p. Computed in
# three stages, mirroring SPlisHSPlasH Elasticity_Kee2023::newtonMatvec:
#
#   (1) δF_k = Σ_j (p_j - p_k) ⊗ G_kj             [G_kj = V_j L_k ∇W_kj]
#   (2) δP_k = H_k : δF_k                          [per-particle 9×9, applied
#                                                   on-the-fly via the
#                                                   rank-1 eigen-decomposition
#                                                   of the co-rotated Hessian]
#   (3) δPL_k = δP_k · L_k,
#       (K·p)_k = -V_k Σ_j V_j (δPL_k + δPL_j) ∇W_kj
#
# Co-rotated material: H is
#     H = 2μ H_ARAP + λ vec(R) vec(R)^T,
#     H_ARAP : δF = Σ_{eig k} ⟨q_k, δF⟩ q_k     (skew/sym off-diag + diag basis)
# with PSD-clamped eigenvalues λ_k = max(0, 1 - 2/(σ_a + σ_b)) on the three
# skew-symmetric "twist" modes and unit eigenvalues on the six remaining modes
# (see SPlisHSPlasH::computeCorotatedHessian9x9).
# ----------------------------------------------------------------------------


@wp.func
def _ddot33(A: wp.mat33, B: wp.mat33) -> float:
    """Frobenius inner product trace(A^T B) = Σ_{ab} A[a,b] B[a,b]."""
    return (A[0, 0] * B[0, 0] + A[0, 1] * B[0, 1] + A[0, 2] * B[0, 2]
          + A[1, 0] * B[1, 0] + A[1, 1] * B[1, 1] + A[1, 2] * B[1, 2]
          + A[2, 0] * B[2, 0] + A[2, 1] * B[2, 1] + A[2, 2] * B[2, 2])


@wp.func
def _outer_cols(U: wp.mat33, a: int, V: wp.mat33, b: int) -> wp.mat33:
    """Outer product u_a v_b^T where u_a = U[:, a], v_b = V[:, b]."""
    u0 = U[0, a]; u1 = U[1, a]; u2 = U[2, a]
    v0 = V[0, b]; v1 = V[1, b]; v2 = V[2, b]
    return wp.mat33(u0 * v0, u0 * v1, u0 * v2,
                    u1 * v0, u1 * v1, u1 * v2,
                    u2 * v0, u2 * v1, u2 * v2)


@wp.func
def _apply_corot_hessian(
    U: wp.mat33, V: wp.mat33, sigma: wp.vec3, R: wp.mat33,
    mu: float, lambda_: float, dF: wp.mat33,
) -> wp.mat33:
    """Apply the 9×9 co-rotated Hessian to δF (returned as 3×3 δP).

    Eigen-decomposition (Lin et al. 2022, "iARAP"):
        H = 2μ Σ_k λ_k q_k q_k^T  +  λ vec(R) vec(R)^T

    Modes (3×3 basis matrices in vectorized form):
        Twist   T_ab = (u_a v_b^T - u_b v_a^T)/√2,  λ = max(0, 1 - 2/(σ_a+σ_b))
        Flip    F_ab = (u_a v_b^T + u_b v_a^T)/√2,  λ = 1
        Scaling S_a  = u_a v_a^T,                    λ = 1
    for (a, b) ∈ {(0,1), (1,2), (0,2)} on twist/flip and a ∈ {0,1,2} on scaling.
    """
    sqrt2inv = float(0.7071067811865476)

    u0v1 = _outer_cols(U, 0, V, 1)
    u1v0 = _outer_cols(U, 1, V, 0)
    u1v2 = _outer_cols(U, 1, V, 2)
    u2v1 = _outer_cols(U, 2, V, 1)
    u0v2 = _outer_cols(U, 0, V, 2)
    u2v0 = _outer_cols(U, 2, V, 0)

    T01 = sqrt2inv * (u0v1 - u1v0)
    T12 = sqrt2inv * (u1v2 - u2v1)
    T02 = sqrt2inv * (u0v2 - u2v0)

    Fl01 = sqrt2inv * (u0v1 + u1v0)
    Fl12 = sqrt2inv * (u1v2 + u2v1)
    Fl02 = sqrt2inv * (u0v2 + u2v0)

    S0 = _outer_cols(U, 0, V, 0)
    S1 = _outer_cols(U, 1, V, 1)
    S2 = _outer_cols(U, 2, V, 2)

    lam01 = wp.max(0.0, 1.0 - 2.0 / (sigma[0] + sigma[1]))
    lam12 = wp.max(0.0, 1.0 - 2.0 / (sigma[1] + sigma[2]))
    lam02 = wp.max(0.0, 1.0 - 2.0 / (sigma[0] + sigma[2]))

    arap = (lam01 * _ddot33(T01, dF)) * T01 \
         + (lam12 * _ddot33(T12, dF)) * T12 \
         + (lam02 * _ddot33(T02, dF)) * T02 \
         + _ddot33(Fl01, dF) * Fl01 \
         + _ddot33(Fl12, dF) * Fl12 \
         + _ddot33(Fl02, dF) * Fl02 \
         + _ddot33(S0,  dF) * S0  \
         + _ddot33(S1,  dF) * S1  \
         + _ddot33(S2,  dF) * S2

    return (2.0 * mu) * arap + (lambda_ * _ddot33(R, dF)) * R


@wp.func
def _apply_snh_hessian(
    U: wp.mat33, V: wp.mat33, sigma: wp.vec3, F: wp.mat33,
    mu: float, lambda_: float, dF: wp.mat33,
) -> wp.mat33:
    """Apply the PSD-projected Stable Neo-Hookean 9×9 Hessian (Smith 2018 Eq. 22)
    to δF:
        H = μ_T I  +  μ_F vec(F) vec(F)^T  +  λ vec(cof) vec(cof)^T  +  γ_H H_vol
    with
        I_C = ‖F‖²,  J = det(F),  IC1 = I_C + 1,
        μ_T = μ (1 - 1/IC1),     μ_F = 2μ / IC1²,
        γ_H = λ(J − 1) − 0.75μ   [≡ λ(J − α) for α = 1 + μ/λ − μ/(4λ)]

    H_vol uses the volumetric eigendecomposition (same modes as the corotated
    volume Hessian — 6 twist/flip with eigenvalues ±σᵢ + 3 cubic-root scaling).
    Each PSD-clamped mode contributes a rank-1 outer product.
    """
    sqrt2inv = float(0.7071067811865476)
    eps_tol = float(1.0e-12)
    pi_const = float(3.14159265358979323846)

    f0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    c0 = wp.cross(f1, f2)
    c1 = wp.cross(f2, f0)
    c2 = wp.cross(f0, f1)
    cofF = wp.mat33(c0[0], c1[0], c2[0],
                    c0[1], c1[1], c2[1],
                    c0[2], c1[2], c2[2])

    I_C = (f0[0]*f0[0] + f0[1]*f0[1] + f0[2]*f0[2]
         + f1[0]*f1[0] + f1[1]*f1[1] + f1[2]*f1[2]
         + f2[0]*f2[0] + f2[1]*f2[1] + f2[2]*f2[2])
    J = wp.determinant(F)
    IC1 = I_C + 1.0
    mu_T = mu * (1.0 - 1.0 / IC1)
    mu_F = 2.0 * mu / (IC1 * IC1)
    gamma_H = lambda_ * (J - 1.0) - 0.75 * mu

    # Term 1: μ_T · δF
    dP = mu_T * dF
    # Term 2: μ_F · (F : δF) · F
    dP = dP + (mu_F * _ddot33(F, dF)) * F
    # Term 3: λ · (cof : δF) · cof
    dP = dP + (lambda_ * _ddot33(cofF, dF)) * cofF

    # Term 4: γ_H H_vol — same eigen-basis as corotated volume Hessian,
    # but with the SNH-specific eigenvalues. PSD-clamp by `scaled > 0`.
    u0v1 = _outer_cols(U, 0, V, 1)
    u1v0 = _outer_cols(U, 1, V, 0)
    u1v2 = _outer_cols(U, 1, V, 2)
    u2v1 = _outer_cols(U, 2, V, 1)
    u0v2 = _outer_cols(U, 0, V, 2)
    u2v0 = _outer_cols(U, 2, V, 0)
    T01 = sqrt2inv * (u0v1 - u1v0)
    T12 = sqrt2inv * (u1v2 - u2v1)
    T02 = sqrt2inv * (u0v2 - u2v0)
    Fl01 = sqrt2inv * (u0v1 + u1v0)
    Fl12 = sqrt2inv * (u1v2 + u2v1)
    Fl02 = sqrt2inv * (u0v2 + u2v0)

    # Pairing (matches SPlisHSPlasH volD[0..5] enumeration):
    #   T12 ↔ +σ0,  Fl12 ↔ −σ0,
    #   T02 ↔ +σ1,  Fl02 ↔ −σ1,
    #   T01 ↔ +σ2,  Fl01 ↔ −σ2.
    s0 = sigma[0]; s1 = sigma[1]; s2 = sigma[2]
    e_T12 = gamma_H * s0
    if e_T12 > 0.0:
        dP = dP + (e_T12 * _ddot33(T12, dF)) * T12
    e_Fl12 = -gamma_H * s0
    if e_Fl12 > 0.0:
        dP = dP + (e_Fl12 * _ddot33(Fl12, dF)) * Fl12
    e_T02 = gamma_H * s1
    if e_T02 > 0.0:
        dP = dP + (e_T02 * _ddot33(T02, dF)) * T02
    e_Fl02 = -gamma_H * s1
    if e_Fl02 > 0.0:
        dP = dP + (e_Fl02 * _ddot33(Fl02, dF)) * Fl02
    e_T01 = gamma_H * s2
    if e_T01 > 0.0:
        dP = dP + (e_T01 * _ddot33(T01, dF)) * T01
    e_Fl01 = -gamma_H * s2
    if e_Fl01 > 0.0:
        dP = dP + (e_Fl01 * _ddot33(Fl01, dF)) * Fl01

    # 3 mixed-scaling modes via depressed-cubic eigenvalues.
    if I_C > eps_tol:
        sqrtIC3 = wp.sqrt(I_C / 3.0)
        cosArg = 3.0 * J / I_C * wp.sqrt(3.0 / I_C)
        clampedCos = wp.max(-1.0, wp.min(1.0, cosArg))
        theta = wp.acos(clampedCos)
        for k in range(3):
            eps_k = 2.0 * sqrtIC3 * wp.cos((theta + 2.0 * pi_const * float(k)) / 3.0)
            scaled = gamma_H * eps_k
            if scaled > 0.0:
                d0 = s0 * s2 + s1 * eps_k
                d1 = s1 * s2 + s0 * eps_k
                d2 = eps_k * eps_k - s2 * s2
                norm_k = wp.sqrt(d0*d0 + d1*d1 + d2*d2)
                if norm_k > eps_tol:
                    inv_n = 1.0 / norm_k
                    diag_k = wp.vec3(d0 * inv_n, d1 * inv_n, d2 * inv_n)
                    # Q = U · diag(diag_k) · V^T
                    UD = wp.mat33(
                        U[0, 0]*diag_k[0], U[0, 1]*diag_k[1], U[0, 2]*diag_k[2],
                        U[1, 0]*diag_k[0], U[1, 1]*diag_k[1], U[1, 2]*diag_k[2],
                        U[2, 0]*diag_k[0], U[2, 1]*diag_k[1], U[2, 2]*diag_k[2],
                    )
                    Q = UD @ wp.transpose(V)
                    dP = dP + (scaled * _ddot33(Q, dF)) * Q

    return dP


@wp.kernel
def matvec_compute_dPL_snh_kernel(
    ids: wp.array(dtype=int),
    p_in: wp.array(dtype=wp.vec3),
    L: wp.array(dtype=wp.mat33),
    U: wp.array(dtype=wp.mat33),
    V: wp.array(dtype=wp.mat33),
    sigma: wp.array(dtype=wp.vec3),
    F: wp.array(dtype=wp.mat33),
    VjLi_gradW: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    mu: wp.array(dtype=float),
    lambda_: wp.array(dtype=float),
    dPL_out: wp.array(dtype=wp.mat33),
):
    """SNH version of matvec stage 1: δPL_i = (H_SNH_i : δF_i) · L_i."""
    p_i = ids[wp.tid()]

    p_i_vec = p_in[p_i]
    dF = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        g = VjLi_gradW[p_i * cache_size + k]
        dF = dF + wp.outer(p_in[p_j] - p_i_vec, g)

    dP = _apply_snh_hessian(U[p_i], V[p_i], sigma[p_i], F[p_i], mu[p_i], lambda_[p_i], dF)
    dPL_out[p_i] = dP @ L[p_i]


@wp.kernel
def matvec_compute_dPL_kernel(
    ids: wp.array(dtype=int),
    p_in: wp.array(dtype=wp.vec3),
    L: wp.array(dtype=wp.mat33),
    U: wp.array(dtype=wp.mat33),
    V: wp.array(dtype=wp.mat33),
    sigma: wp.array(dtype=wp.vec3),
    R: wp.array(dtype=wp.mat33),
    VjLi_gradW: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    mu: wp.array(dtype=float),
    lambda_: wp.array(dtype=float),
    dPL_out: wp.array(dtype=wp.mat33),
):
    """Stage 1 of A·p: compute δPL_i = (H_i : δF_i) · L_i, where
        δF_i = Σ_j (p_j - p_i) ⊗ G_ij,    G_ij = V_j L_i ∇W_ij.
    """
    p_i = ids[wp.tid()]

    p_i_vec = p_in[p_i]
    dF = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        g = VjLi_gradW[p_i * cache_size + k]
        dF = dF + wp.outer(p_in[p_j] - p_i_vec, g)

    dP = _apply_corot_hessian(U[p_i], V[p_i], sigma[p_i], R[p_i], mu[p_i], lambda_[p_i], dF)
    dPL_out[p_i] = dP @ L[p_i]


@wp.kernel
def matvec_assemble_Ap_kernel(
    ids: wp.array(dtype=int),
    p_in: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    V0_rest: wp.array(dtype=float),
    dPL: wp.array(dtype=wp.mat33),
    Vj_gradW: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    cache_size: int,
    dt2: float,
    Ap_out: wp.array(dtype=wp.vec3),
):
    """Stage 2 of A·p: (A·p)_k = m_k p_k - dt² V_k Σ_j V_j (δPL_k + δPL_j) ∇W_kj.

    Sign matches the gradient: ∇²Ψ on a position perturbation gives the same
    -V_k Σ_j V_j ( · L_k +  · L_j) ∇W_kj structure as ∇Ψ, but with P → δP.
    """
    p_i = ids[wp.tid()]

    V_i = V0_rest[p_i]
    dPL_i = dPL[p_i]
    accum = wp.vec3(0.0, 0.0, 0.0)
    n_i = rest_neighbor_count[p_i]
    for k in range(n_i):
        p_j = rest_neighbor_ids[p_i * cache_size + k]
        g = Vj_gradW[p_i * cache_size + k]
        accum = accum + (dPL_i + dPL[p_j]) @ g

    Ap_out[p_i] = m[p_i] * p_in[p_i] - dt2 * V_i * accum


# ----------------------------------------------------------------------------
# PCG plumbing kernels
#
# 3×3 block Jacobi preconditioner: the exact diagonal blocks of A = M + dt² (K + S_ZE),
#     M_pre_inv[i] = (m_i I + dt² (K_ii + k_ze,i I))^{-1}
# ----------------------------------------------------------------------------


@wp.func
def _apply_hessian(
    material_type: int, U: wp.mat33, V: wp.mat33, sigma: wp.vec3, R: wp.mat33, F: wp.mat33,
    mu: float, lambda_: float, dF: wp.mat33,
) -> wp.mat33:
    """δP = H : δF for the active material (same PSD-projected H as the matvec)."""
    if material_type == 1:
        return _apply_snh_hessian(U, V, sigma, F, mu, lambda_, dF)
    return _apply_corot_hessian(U, V, sigma, R, mu, lambda_, dF)


@wp.kernel
def compute_block_precond_kernel(
    ids: wp.array(dtype=int),
    m: wp.array(dtype=float),
    V0_rest: wp.array(dtype=float),
    L: wp.array(dtype=wp.mat33),
    U: wp.array(dtype=wp.mat33),
    V: wp.array(dtype=wp.mat33),
    sigma: wp.array(dtype=wp.vec3),
    R: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    VjLi_gradW: wp.array(dtype=wp.vec3),
    Vj_gradW: wp.array(dtype=wp.vec3),
    G_sum: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    reverse_slot: wp.array(dtype=int),
    cache_size: int,
    mu: wp.array(dtype=float),
    lambda_: wp.array(dtype=float),
    material_type: int,
    ze_diag: wp.array(dtype=float),
    dt2: float,
    M_pre_inv: wp.array(dtype=wp.mat33),
):
    """3×3 block Jacobi:  M_pre_inv[q] = (m_q I + dt² (K_qq + k_ze,q I))^{-1}.

    K_qq is the matvec applied to a unit displacement e_c on particle q alone.
    Then δF_q = -e_c ⊗ Gsum_q, δF_j = e_c ⊗ G_jq for j ∈ N(q), all others 0, so
        K_qq e_c = -V_q [ (H_q : (-e_c ⊗ Gsum_q)) L_q · S_q
                         + Σ_j (H_j : (e_c ⊗ G_jq)) L_j · V_j ∇W_qj ],
        S_q = Σ_j V_j ∇W_qj,   G_jq = VjLi_gradW[j, slot of q in N(j)].
    """
    q = ids[wp.tid()]
    mt = material_type
    n_q = rest_neighbor_count[q]

    S_q = wp.vec3(0.0, 0.0, 0.0)
    for k in range(n_q):
        S_q = S_q + Vj_gradW[q * cache_size + k]

    e0 = wp.vec3(1.0, 0.0, 0.0)
    e1 = wp.vec3(0.0, 1.0, 0.0)
    e2 = wp.vec3(0.0, 0.0, 1.0)

    # q-side term: columns (H_q : (-e_c ⊗ Gsum_q)) L_q · S_q
    Uq = U[q]; Vq = V[q]; sq = sigma[q]; Rq = R[q]; Fq = F[q]; Lq = L[q]
    mq = mu[q]; lq = lambda_[q]
    gs = G_sum[q]
    c0 = (_apply_hessian(mt, Uq, Vq, sq, Rq, Fq, mq, lq, -wp.outer(e0, gs)) @ Lq) @ S_q
    c1 = (_apply_hessian(mt, Uq, Vq, sq, Rq, Fq, mq, lq, -wp.outer(e1, gs)) @ Lq) @ S_q
    c2 = (_apply_hessian(mt, Uq, Vq, sq, Rq, Fq, mq, lq, -wp.outer(e2, gs)) @ Lq) @ S_q

    # neighbour terms: columns (H_j : (e_c ⊗ G_jq)) L_j · V_j ∇W_qj
    for k in range(n_q):
        j = rest_neighbor_ids[q * cache_size + k]
        G_jq = VjLi_gradW[j * cache_size + reverse_slot[q * cache_size + k]]
        g = Vj_gradW[q * cache_size + k]
        Uj = U[j]; Vj = V[j]; sj = sigma[j]; Rj = R[j]; Fj = F[j]; Lj = L[j]
        mj = mu[j]; lj = lambda_[j]
        c0 = c0 + (_apply_hessian(mt, Uj, Vj, sj, Rj, Fj, mj, lj, wp.outer(e0, G_jq)) @ Lj) @ g
        c1 = c1 + (_apply_hessian(mt, Uj, Vj, sj, Rj, Fj, mj, lj, wp.outer(e1, G_jq)) @ Lj) @ g
        c2 = c2 + (_apply_hessian(mt, Uj, Vj, sj, Rj, Fj, mj, lj, wp.outer(e2, G_jq)) @ Lj) @ g

    # K_qq[:, c] = -V_q c_c; symmetrized (the exact block is symmetric; PCG needs a symmetric M)
    K = -V0_rest[q] * (wp.outer(c0, e0) + wp.outer(c1, e1) + wp.outer(c2, e2))
    K = 0.5 * (K + wp.transpose(K))
    M = wp.identity(n=3, dtype=float) * (m[q] + dt2 * ze_diag[q]) + dt2 * K
    M_pre_inv[q] = wp.inverse(M)


@wp.kernel
def precompute_ze_diag_kernel(
    ids: wp.array(dtype=int),
    x0: wp.array(dtype=wp.vec3),
    K_ze: wp.array(dtype=float),
    VjLi_gradW: wp.array(dtype=wp.vec3),
    G_sum: wp.array(dtype=wp.vec3),
    rest_neighbor_count: wp.array(dtype=int),
    rest_neighbor_ids: wp.array(dtype=int),
    reverse_slot: wp.array(dtype=int),
    cache_size: int,
    ze_diag: wp.array(dtype=float),
):
    """Diagonal block of the ZE Hessian, k_ze,q I (rest-config constant).

    E_ZE = ½ Σ_i Σ_{j∈N(i)} K_ij |r_ij|²,  r_ij = (x_j - x_i) - F_i d_ij,  d_ij = x_j0 - x_i0.
    A unit displacement e_c on q makes every residual a multiple of e_c:
        owner q:          r_qj = -e_c (1 - Gsum_q · d_qj)
        owner i ∈ N(q):   r_il =  e_c (δ_lq - G_iq · d_il)
    so the block is scalar:
        k_ze,q = Σ_j K_qj (1 - Gsum_q · d_qj)²  +  Σ_{i∈N(q)} Σ_{l∈N(i)} K_il (δ_lq - G_iq · d_il)²
    """
    q = ids[wp.tid()]
    x_q0 = x0[q]
    gs = G_sum[q]
    n_q = rest_neighbor_count[q]
    kz = float(0.0)
    for k in range(n_q):
        j = rest_neighbor_ids[q * cache_size + k]
        s_q = 1.0 - wp.dot(gs, x0[j] - x_q0)
        kz = kz + K_ze[q * cache_size + k] * s_q * s_q

        i = j
        G_iq = VjLi_gradW[i * cache_size + reverse_slot[q * cache_size + k]]
        x_i0 = x0[i]
        n_i = rest_neighbor_count[i]
        for l in range(n_i):
            p_l = rest_neighbor_ids[i * cache_size + l]
            s_il = -wp.dot(G_iq, x0[p_l] - x_i0)
            if p_l == q:
                s_il = s_il + 1.0
            kz = kz + K_ze[i * cache_size + l] * s_il * s_il
    ze_diag[q] = kz


@wp.kernel
def pcg_init_kernel(
    ids: wp.array(dtype=int),
    gradient: wp.array(dtype=wp.vec3),
    dt: float,
    M_pre_inv: wp.array(dtype=wp.mat33),
    dv: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
):
    """Initialize PCG with dv=0:  r = b - A·0 = b = -dt · ∇Ψ.

    Then z = M_pre_inv * r and p = z. We carry r and z separately so the
    final r·z dot product can be computed against the preconditioned
    residual, the standard PCG initialization.
    """
    p_i = ids[wp.tid()]
    b = -dt * gradient[p_i]
    dv[p_i] = wp.vec3(0.0, 0.0, 0.0)
    r[p_i] = b
    z_i = M_pre_inv[p_i] @ b
    z[p_i] = z_i
    p[p_i] = z_i


@wp.kernel
def reduce_dot_kernel(
    ids: wp.array(dtype=int),
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    """out[0] += Σ_i a[i] · b[i] (atomic reduction)."""
    p_i = ids[wp.tid()]
    wp.atomic_add(out, 0, wp.dot(a[p_i], b[p_i]))


@wp.kernel
def pcg_update_dv_r_kernel(
    ids: wp.array(dtype=int),
    alpha: float,
    p: wp.array(dtype=wp.vec3),
    Ap: wp.array(dtype=wp.vec3),
    dv: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
):
    """PCG step: dv += α p,  r -= α A p."""
    p_i = ids[wp.tid()]
    dv[p_i] = dv[p_i] + alpha * p[p_i]
    r[p_i] = r[p_i] - alpha * Ap[p_i]


@wp.kernel
def pcg_apply_precond_kernel(
    ids: wp.array(dtype=int),
    M_pre_inv: wp.array(dtype=wp.mat33),
    r: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
):
    """z = M_pre_inv · r (3×3 block per particle)."""
    p_i = ids[wp.tid()]
    z[p_i] = M_pre_inv[p_i] @ r[p_i]


@wp.kernel
def pcg_update_p_kernel(
    ids: wp.array(dtype=int),
    beta: float,
    z: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
):
    """PCG direction update: p = z + β p."""
    p_i = ids[wp.tid()]
    p[p_i] = z[p_i] + beta * p[p_i]


@wp.kernel
def _add_vec3_kernel(
    ids: wp.array(dtype=int),
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
):
    """dst[i] += src[i]. Merges the ZE-only gradient into the composite gradient."""
    p_i = ids[wp.tid()]
    dst[p_i] = dst[p_i] + src[p_i]


@wp.kernel
def mask_fixed_kernel(
    ids: wp.array(dtype=int),
    is_fixed: wp.array(dtype=int),
    vec: wp.array(dtype=wp.vec3),
):
    """Zero vec[i] wherever is_fixed[i]==1. Enforces the fixed-vertex
    constraint by masking the Newton-system gradient and matvec output:
    with gradient[fixed]=0 and (A·p)[fixed]=0, PCG keeps dv[fixed]=0, so
    fixed particles receive no velocity update."""
    p_i = ids[wp.tid()]
    if is_fixed[p_i] == 1:
        vec[p_i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _reduce_max_abs_vec3_kernel(
    ids: wp.array(dtype=int),
    a: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    """out[0] = max over i of ||a[i]||_inf  (max absolute component)."""
    p_i = ids[wp.tid()]
    v = a[p_i]
    m = wp.max(wp.abs(v[0]), wp.max(wp.abs(v[1]), wp.abs(v[2])))
    wp.atomic_max(out, 0, m)


@wp.kernel
def reduce_max_principal_strain_kernel(
    ids: wp.array(dtype=int),
    sigma: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    """out[0] = max_i max_k |sigma_ik - 1|: rotation-invariant strain (negative sigma = inverted)."""
    p_i = ids[wp.tid()]
    s = sigma[p_i]
    e = wp.max(wp.abs(s[0] - 1.0), wp.max(wp.abs(s[1] - 1.0), wp.abs(s[2] - 1.0)))
    wp.atomic_max(out, 0, e)


@wp.kernel
def apply_velocity_update_kernel(
    ids: wp.array(dtype=int),
    dv: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
):
    """v ← v + Δv (linearized at x̃, so Δv is the descent step from v_pred)."""
    p_i = ids[wp.tid()]
    v[p_i] = v[p_i] + dv[p_i]


# ============================================================================
# Class
# ============================================================================


class ElasticityKee2023:
    """Velocity-based SPH elasticity (Kee 2023; co-rotated or Stable Neo-Hookean),
    one Newton iteration per step by default.

    Deformable particles are addressed through the compact index list
    `ids[0:count]` (any arena slots, any order). Bodies are registered at load
    time (SolidBodies) or when a ScheduledSpawn fires; per-particle buffers are
    indexed by particle index, so registration never moves data.
    """

    def __init__(self, simulation_data, neighbour_search, config):
        self.ps = simulation_data

        N = self.ps.particle_max_num
        C = self.ps.cache_size

        # Deformable registry: compact index list + per-particle registration id
        # (0 = not deformable, k0 + 1 for the batch registered at ids[k0:k1]).
        self.ids = wp.zeros(N, dtype=int)
        self.ids_np = np.zeros(N, dtype=np.int32)
        self.reg_np = np.zeros(N, dtype=np.int32)
        self.count = 0
        self.reg_id = wp.zeros(N, dtype=int)
        self._snapshot_count = wp.zeros(N, dtype=int)

        # Rest-config state
        self.V0_rest = wp.zeros(N, dtype=float)
        self.L = wp.zeros(N, dtype=wp.mat33)
        self.rank = wp.zeros(N, dtype=int)         # supported rest directions per particle (3 = full)
        self.F = wp.zeros(N, dtype=wp.mat33)
        self.rest_neighbor_count = wp.zeros(N, dtype=int)
        self.rest_neighbor_ids = wp.zeros(N * C, dtype=int)
        self.VjLi_gradW = wp.zeros(N * C, dtype=wp.vec3)
        self.Vj_gradW = wp.zeros(N * C, dtype=wp.vec3)

        # Zero-energy mode (ZE / hourglass) penalty buffers — material-agnostic.
        # Applied matrix-free via the GATHER form (build-Y + offset-gather, no atomics).
        self.K_ze = wp.zeros(N * C, dtype=float)       # rest-config per-pair stiffness
        self.G_sum = wp.zeros(N, dtype=wp.vec3)        # Σ_l G_il at rest
        self.y_ze = wp.zeros(N * C, dtype=wp.vec3)     # per-pair y_ij = K_ij r_ij (build → gather)
        self.Y_ze = wp.zeros(N, dtype=wp.mat33)        # per-particle Y_i = Σ_j y_ij ⊗ d_ij
        self.Ysum_ze = wp.zeros(N, dtype=wp.vec3)      # per-particle Σ_j y_ij
        # Reverse-neighbour offset: reverse_slot[i·C+k] = slot of i in N(N(i)[k]).
        # Rest-config constant, built at registration. Lets the gather pass
        # fetch the owner's pair data with no search and no atomics.
        self.rest_neighbor_reverse_slot = wp.zeros(N * C, dtype=int)
        # ZE-only gradient buffer, gathered into and then merged into the composite gradient.
        self.gradient_ze = wp.zeros(N, dtype=wp.vec3)

        # Per-step caches: SVD components + R + PL (so matvec can re-apply H)
        self.U = wp.zeros(N, dtype=wp.mat33)
        self.V_svd = wp.zeros(N, dtype=wp.mat33)
        self.sigma = wp.zeros(N, dtype=wp.vec3)
        self.R = wp.zeros(N, dtype=wp.mat33)
        self.PL = wp.zeros(N, dtype=wp.mat33)

        # Newton system: gradient ∇Ψ, preconditioner, PCG buffers
        self.x_tilde = wp.zeros(N, dtype=wp.vec3)
        self.gradient = wp.zeros(N, dtype=wp.vec3)
        self.M_pre_inv = wp.zeros(N, dtype=wp.mat33)
        self.ze_diag = wp.zeros(N, dtype=float)        # rest-config ZE Hessian diagonal (block = k_ze I)
        self.dv = wp.zeros(N, dtype=wp.vec3)
        self.r = wp.zeros(N, dtype=wp.vec3)
        self.z = wp.zeros(N, dtype=wp.vec3)
        self.p = wp.zeros(N, dtype=wp.vec3)
        self.Ap = wp.zeros(N, dtype=wp.vec3)
        self.dPL = wp.zeros(N, dtype=wp.mat33)
        self._scalar_buf = wp.zeros(1, dtype=float)

        # Material parameters (read from scene config; converted from E, ν).
        # mu / lambda_ are the scene defaults; the kernels read the per-particle
        # mu_p / lambda_p, which a body entry's own youngsModulus / poissonsRatio override.
        E = float(config.get_cfg("youngsModulus", 5.0e5))
        nu = float(config.get_cfg("poissonsRatio", 0.45))
        self.mu, self.lambda_ = lame_parameters(E, nu)
        self.mu_np = np.full(N, self.mu, dtype=np.float32)
        self.lambda_np = np.full(N, self.lambda_, dtype=np.float32)
        self.custom_np = np.zeros(N, dtype=bool)
        self.mu_p = wp.array(self.mu_np, dtype=float)
        self.lambda_p = wp.array(self.lambda_np, dtype=float)
        self.max_iter_cg = int(config.get_cfg("maxIterCG", 100))
        self.tol_cg = float(config.get_cfg("tolCG", 1e-4))
        # 0 = Co-rotated (FCR), 1 = Stable Neo-Hookean (Smith 2018)
        self.material_type = int(config.get_cfg("materialType", 0))
        # Zero-energy mode penalty strength (0 disables, matches SPlisHSPlasH
        # m_alpha default). Material-agnostic: applies the same matrix-free
        # H^T K_ze H operator under both Co-rotated and SNH dispatch.
        self.alpha_ze = float(config.get_cfg("alphaZE", 0.0))
        # Newton outer loop (mirrors Kee2023 stepElasticitySolver): 1 iteration by
        # default; more re-linearise and stop on ‖Δv·dt‖_∞ < tol_newton.
        self.max_iter_newton = int(config.get_cfg("maxIterNewton", 1))
        self.tol_newton = float(config.get_cfg("tolNewton", 1.0e-6))

        # Last-solve stats (shown by the GUI)
        self.last_cg_iters = 0
        self.last_cg_residual = 0.0
        self.last_max_strain = 0.0  # max |sigma(F) - 1| at the last linearisation point

    # ------------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------------

    def register(self, start: int, n: int, material=None):
        """Append arena slots [start, start+n) to the deformable set. Returns (k0, k1).
        `material` (the body's scene entry) may carry its own youngsModulus / poissonsRatio."""
        if n <= 0:
            return self.count, self.count
        k0 = self.count
        k1 = k0 + int(n)
        if k1 > self.ps.particle_max_num:
            raise RuntimeError(f"[ElasticityKee2023] registry overflow: {k1} > {self.ps.particle_max_num}")
        self.ids_np[k0:k1] = np.arange(start, start + n, dtype=np.int32)
        self.reg_np[start:start + n] = k0 + 1
        wp.copy(self.ids, wp.array(self.ids_np[k0:k1], dtype=int, device=self.ids.device),
                dest_offset=k0, count=int(n))
        self.count = k1
        self._upload_membership()
        self._set_material(start, int(n), material)
        return k0, k1

    def _set_material(self, start: int, n: int, material):
        """Write (μ, λ) of slots [start, start+n): the entry's values, else the scene default."""
        custom = material is not None and any(k in material for k in MATERIAL_KEYS)
        if custom:
            nu_d = self.lambda_ / (2.0 * (self.lambda_ + self.mu))   # scene default E, ν
            E_d = 2.0 * self.mu * (1.0 + nu_d)
            mu, lam = lame_parameters(float(material.get("youngsModulus", E_d)),
                                      float(material.get("poissonsRatio", nu_d)))
        else:
            mu, lam = self.mu, self.lambda_
        sl = slice(start, start + n)
        self.mu_np[sl], self.lambda_np[sl], self.custom_np[sl] = mu, lam, custom
        for dst, src in ((self.mu_p, self.mu_np), (self.lambda_p, self.lambda_np)):
            wp.copy(dst, wp.array(src[sl], dtype=float, device=dst.device), dest_offset=start, count=n)

    def set_default_material(self, mu: float, lam: float):
        """Change the scene-default (μ, λ); bodies with their own material keep it.
        Call refresh_K_ze() afterwards (K_ze caches μ)."""
        self.mu, self.lambda_ = float(mu), float(lam)
        self.mu_np[~self.custom_np] = self.mu
        self.lambda_np[~self.custom_np] = self.lambda_
        self.mu_p.assign(self.mu_np)
        self.lambda_p.assign(self.lambda_np)

    def truncate(self, count: int):
        """Forget every particle registered after the first `count` (Framework.reset)."""
        self.count = int(count)
        self._upload_membership()

    def _upload_membership(self):
        flags = np.zeros(self.ps.particle_max_num, dtype=np.int32)
        sel = self.ids_np[:self.count]
        flags[sel] = self.reg_np[sel]
        self.reg_id.assign(flags)

    # ------------------------------------------------------------------------
    # Rest-configuration init
    # ------------------------------------------------------------------------

    def initialize(self):
        """Rest init for every registered particle (x == x0 required). Idempotent."""
        if self.count == 0:
            print("[ElasticityKee2023] no deformable particles; init skipped")
            return
        st = self.initialize_range(0, self.count)
        print(f"[ElasticityKee2023] {st['count']} deformable particles, max rest neighbors "
              f"after symmetrize: {st['max_rest_neighbors']} (cache_size={self.ps.cache_size}), "
              f"rank-deficient rest sets: {st['rank_deficient']}")
        n_custom = int(self.custom_np[self.ids_np[:self.count]].sum())
        print(f"[ElasticityKee2023] material = {MATERIAL_NAMES.get(self.material_type, '?')}, "
              f"μ = {self.mu:.3e}, λ = {self.lambda_:.3e} (default; {n_custom} particles override), "
              f"α_ZE = {self.alpha_ze:.2e}, PCG: maxIter = {self.max_iter_cg}, tol = {self.tol_cg:.1e}")

    def initialize_range(self, k0: int, k1: int) -> dict:
        """Build rest neighbours, V0, L, cached gradients and ZE data for ids[k0:k1].

        Reads the live neighbour cache, so the body must sit at its rest shape
        (x == x0) when this runs: at load/reset for SolidBodies, or right after
        the neighbour rebuild of the step a ScheduledSpawn fired. Returns the
        per-range stats the callers print.
        """
        ps = self.ps
        D = int(k1 - k0)
        if D <= 0:
            return {"count": 0, "max_rest_neighbors": 0, "rank_deficient": 0}
        ids = self.ids[k0:k1]
        C = int(ps.cache_size)
        h = float(ps.support_radius)

        # 1. Snapshot rest neighbours (same registration batch + same object only).
        wp.launch(
            kernel=snapshot_rest_neighbors_kernel,
            dim=D,
            inputs=[ids, ps.object_id, ps.neighbor_count, ps.neighbor_ids, self.reg_id, C],
            outputs=[self.rest_neighbor_count, self.rest_neighbor_ids],
        )

        # 2. Symmetrize against the post-snapshot counts (indexed by particle index).
        wp.copy(self._snapshot_count, self.rest_neighbor_count)
        wp.launch(
            kernel=symmetrize_rest_neighbors_kernel,
            dim=D,
            inputs=[ids, C, self._snapshot_count],
            outputs=[self.rest_neighbor_count, self.rest_neighbor_ids],
        )

        # Overflow check: any deformable exceeding cache_size is fatal.
        sel = self.ids_np[k0:k1]
        counts_np = self.rest_neighbor_count.numpy()[sel]
        max_count = int(counts_np.max())
        if max_count > C:
            offender = int(sel[int(np.argmax(counts_np))])
            raise RuntimeError(
                f"[ElasticityKee2023] rest neighbor overflow: particle {offender} "
                f"has {counts_np.max()} symmetric neighbors but cache_size={C}. "
                f"Raise the scene's neighborCacheSize or check its particle spacing "
                f"vs supportRadius ({h})."
            )
        # 3. V0_rest, L, V_j L_i gradW, V_j gradW.
        wp.launch(
            kernel=compute_V0_rest_kernel,
            dim=D,
            inputs=[ids, ps.x0, self.rest_neighbor_count, self.rest_neighbor_ids, C, h],
            outputs=[self.V0_rest],
        )
        wp.launch(
            kernel=compute_rest_density_kernel,
            dim=D,
            inputs=[ids, ps.x0, ps.m, self.rest_neighbor_count, self.rest_neighbor_ids, C, h],
            outputs=[ps.density0],
        )
        wp.launch(
            kernel=compute_matrix_L_kernel,
            dim=D,
            inputs=[ids, ps.x0, self.V0_rest, self.rest_neighbor_count, self.rest_neighbor_ids, C, h],
            outputs=[self.L, self.rank],
        )
        n_deficient = int((self.rank.numpy()[sel] < 3).sum())
        wp.launch(
            kernel=precompute_VjLi_gradW_kernel,
            dim=D,
            inputs=[ids, ps.x0, self.V0_rest, self.L, self.rest_neighbor_count, self.rest_neighbor_ids, C, h],
            outputs=[self.VjLi_gradW],
        )
        wp.launch(
            kernel=precompute_Vj_gradW_kernel,
            dim=D,
            inputs=[ids, ps.x0, self.V0_rest, self.rest_neighbor_count, self.rest_neighbor_ids, C, h],
            outputs=[self.Vj_gradW],
        )

        # Zero-energy mode rest-config precompute. K_ze caches (α, μ); if either
        # changes at runtime, call refresh_K_ze() to rebuild.
        wp.launch(
            kernel=precompute_G_sum_kernel,
            dim=D,
            inputs=[ids, self.rest_neighbor_count, self.VjLi_gradW, C],
            outputs=[self.G_sum],
        )
        # Reverse-neighbour offset map for the ZE gather and the block preconditioner
        # (rest-config constant); built before K_ze, whose refresh also rebuilds ze_diag.
        wp.launch(
            kernel=build_reverse_slot_kernel,
            dim=D,
            inputs=[ids, self.rest_neighbor_count, self.rest_neighbor_ids, C],
            outputs=[self.rest_neighbor_reverse_slot],
        )
        self._refresh_K_ze_range(k0, k1)
        return {"count": D, "max_rest_neighbors": max_count, "rank_deficient": n_deficient}

    def refresh_K_ze(self):
        """Rebuild per-pair ZE stiffness K_ij = α μ V0_i V0_j W(|x_ij0|) / |x_ij0|²
        for every registered particle. Call after any change to α_ZE or μ (E/ν)."""
        self._refresh_K_ze_range(0, self.count)

    def _refresh_K_ze_range(self, k0: int, k1: int):
        D = int(k1 - k0)
        if D <= 0:
            return
        C = int(self.ps.cache_size)
        h = float(self.ps.support_radius)
        wp.launch(
            kernel=precompute_K_ze_kernel,
            dim=D,
            inputs=[
                self.ids[k0:k1], self.ps.x0, self.V0_rest,
                self.rest_neighbor_count, self.rest_neighbor_ids, C, h,
                self.alpha_ze, self.mu_p,
            ],
            outputs=[self.K_ze],
        )
        wp.launch(
            kernel=precompute_ze_diag_kernel,
            dim=D,
            inputs=[
                self.ids[k0:k1], self.ps.x0, self.K_ze, self.VjLi_gradW, self.G_sum,
                self.rest_neighbor_count, self.rest_neighbor_ids,
                self.rest_neighbor_reverse_slot, C,
            ],
            outputs=[self.ze_diag],
        )

    # ------------------------------------------------------------------------
    # Per-step solver
    # ------------------------------------------------------------------------

    def _matvec(self, p_in, Ap_out, dt: float):
        """Matrix-free A·p_in → Ap_out  (A = M + dt² K, K evaluated at x̃).

        Two passes:
            1. δPL_i = (H_i : δF_i) · L_i,  δF_i = Σ_j (p_j - p_i) ⊗ V_j L_i ∇W_ij
            2. (A·p)_k = m_k p_k - dt² V_k Σ_j V_j (δPL_k + δPL_j) ∇W_kj

        Stage 1 dispatches on material_type — co-rotated uses (U, V, σ, R) and
        the ARAP+vol eigen-basis; SNH uses (U, V, σ, F) and the Smith 2018
        4-term decomposition.
        """
        ps = self.ps
        D = self.count
        ids = self.ids
        C = int(ps.cache_size)
        dt2 = dt * dt

        if self.material_type == MATERIAL_SNH:
            wp.launch(
                kernel=matvec_compute_dPL_snh_kernel,
                dim=D,
                inputs=[
                    ids, p_in, self.L, self.U, self.V_svd, self.sigma, self.F,
                    self.VjLi_gradW, self.rest_neighbor_count, self.rest_neighbor_ids,
                    C, self.mu_p, self.lambda_p,
                ],
                outputs=[self.dPL],
            )
        else:
            wp.launch(
                kernel=matvec_compute_dPL_kernel,
                dim=D,
                inputs=[
                    ids, p_in, self.L, self.U, self.V_svd, self.sigma, self.R,
                    self.VjLi_gradW, self.rest_neighbor_count, self.rest_neighbor_ids,
                    C, self.mu_p, self.lambda_p,
                ],
                outputs=[self.dPL],
            )
        wp.launch(
            kernel=matvec_assemble_Ap_kernel,
            dim=D,
            inputs=[
                ids, p_in, ps.m, self.V0_rest, self.dPL, self.Vj_gradW,
                self.rest_neighbor_count, self.rest_neighbor_ids, C, dt2,
            ],
            outputs=[Ap_out],
        )
        # ZE matvec contribution: Ap += dt² · (H̃ K H̃ᵀ · p) via build-Y +
        # offset-gather (no atomics). The gather kernel is shared with the
        # gradient — only the build differs (δF instead of material F).
        if self.alpha_ze > 0.0:
            wp.launch(
                kernel=ze_build_Y_matvec_kernel,
                dim=D,
                inputs=[
                    ids, p_in, ps.x0, self.K_ze, self.VjLi_gradW,
                    self.rest_neighbor_count, self.rest_neighbor_ids, C,
                ],
                outputs=[self.y_ze, self.Y_ze, self.Ysum_ze],
            )
            wp.launch(
                kernel=ze_gather_kernel,
                dim=D,
                inputs=[
                    ids, self.y_ze, self.Y_ze, self.Ysum_ze, self.G_sum,
                    self.VjLi_gradW, self.rest_neighbor_count,
                    self.rest_neighbor_ids, self.rest_neighbor_reverse_slot,
                    C, dt2,
                ],
                outputs=[Ap_out],
            )

        # Fixed-vertex constraint: (A·p)[fixed] = 0 so PCG never gives fixed
        # particles a nonzero Δv. With dv0=0 and r[fixed]=0 (gradient masked),
        # the fixed entries of p, z, Ap, r stay 0 throughout iteration.
        self._mask_fixed(Ap_out)

    def _mask_fixed(self, vec):
        """Zero vec[i] wherever is_fixed[i]==1. Cheap one-kernel pass; no-op
        when no particle is fixed (kernel just tests the flag)."""
        D = self.count
        if D == 0:
            return
        wp.launch(
            kernel=mask_fixed_kernel,
            dim=D,
            inputs=[self.ids, self.ps.is_fixed],
            outputs=[vec],
        )

    def _reduce_dot(self, a, b) -> float:
        """Compute Σ_i a[i] · b[i] over deformable particles (host-side scalar)."""
        D = self.count
        self._scalar_buf.zero_()
        wp.launch(
            kernel=reduce_dot_kernel,
            dim=D,
            inputs=[self.ids, a, b],
            outputs=[self._scalar_buf],
        )
        return float(self._scalar_buf.numpy()[0])

    def solve(self, dt: float):
        """Velocity-based Newton solve for elasticity.

        Outer Newton loop (max_iter_newton ≥ 1):
            repeat:
                1. x̃ = x + dt v               (re-linearise at current v)
                2. F = F(x̃) → (U, σ, V) → R → P → PL = P·L
                3. ∇Ψ_k = material_grad + (ZE_grad if α > 0)
                4. PCG: (M + dt² K) Δv = -dt ∇Ψ
                5. v ← v + Δv
              until max|Δv·dt| < tol_newton  (or max_iter_newton reached)

        max_iter_newton = 1 is a single velocity-based Newton step; larger values
        re-linearise as Kee2023's stepElasticitySolver does.
        """
        ps = self.ps
        D = self.count
        if D == 0:
            self.last_cg_iters = 0
            self.last_cg_residual = 0.0
            self.last_max_strain = 0.0
            return
        C = int(ps.cache_size)

        max_dv_inf = 0.0
        for _ in range(self.max_iter_newton):
            self._newton_step(ps, D, C, dt)
            # Termination on ||Δv · dt||_inf (= position increment for this step)
            self._scalar_buf.zero_()
            wp.launch(
                kernel=_reduce_max_abs_vec3_kernel,
                dim=D,
                inputs=[self.ids, self.dv],
                outputs=[self._scalar_buf],
            )
            max_dv_inf = float(self._scalar_buf.numpy()[0]) * dt
            if max_dv_inf < self.tol_newton:
                break

        # Diagnostic: principal strain at the last linearisation point x~ (rotation-invariant).
        self._scalar_buf.zero_()
        wp.launch(
            kernel=reduce_max_principal_strain_kernel,
            dim=D,
            inputs=[self.ids, self.sigma],
            outputs=[self._scalar_buf],
        )
        self.last_max_strain = float(self._scalar_buf.numpy()[0])

        # Fixed vertices stay static: zero their velocity so they don't advect.
        # (Gravity / other non-pressure forces add to v before elasticity; the
        # Newton step's dv is already masked to 0 for them, but their pre-solve
        # velocity must also be cleared or they'd drift on the advection step.)
        self._mask_fixed(ps.v)

    def assemble_gradient(self, dt: float):
        """Linearise at x̃ = x + dt·v and assemble the elastic gradient ∇Ψ
        (material + optional ZE) into self.gradient, masked on fixed vertices.

        Populates self.F, self.U/V_svd/sigma/R/PL too, so a subsequent _matvec
        re-applies the Hessian at the same linearisation point.
        """
        ps = self.ps
        D = self.count
        if D == 0:
            return
        ids = self.ids
        C = int(ps.cache_size)

        # 1. Inertial predictor x̃ = x + dt v
        wp.launch(
            kernel=compute_x_tilde_kernel,
            dim=D,
            inputs=[ids, ps.x, ps.v, dt],
            outputs=[self.x_tilde],
        )

        # 2. F at x̃, then SVD + material-specific P → PL
        wp.launch(
            kernel=compute_F_kernel,
            dim=D,
            inputs=[
                ids, self.x_tilde, self.rest_neighbor_count, self.rest_neighbor_ids,
                self.VjLi_gradW, C,
            ],
            outputs=[self.F],
        )
        p_pl_kernel = (
            compute_polar_P_PL_snh_kernel if self.material_type == MATERIAL_SNH
            else compute_polar_P_PL_kernel
        )
        wp.launch(
            kernel=p_pl_kernel,
            dim=D,
            inputs=[ids, self.F, self.L, self.rank, self.mu_p, self.lambda_p],
            outputs=[self.U, self.V_svd, self.sigma, self.R, self.PL],
        )

        # 3. Gradient ∇Ψ (material) + optional ZE contribution
        wp.launch(
            kernel=compute_gradient_kernel,
            dim=D,
            inputs=[
                ids, self.V0_rest, self.PL, self.Vj_gradW,
                self.rest_neighbor_count, self.rest_neighbor_ids, C,
            ],
            outputs=[self.gradient],
        )
        if self.alpha_ze > 0.0:
            # ZE gradient via the gather form (build-Y then offset-gather, no
            # atomics), gathered into gradient_ze and merged into the composite gradient.
            self.gradient_ze.zero_()
            wp.launch(
                kernel=ze_build_Y_gradient_kernel,
                dim=D,
                inputs=[
                    ids, self.x_tilde, ps.x0, self.F, self.K_ze,
                    self.rest_neighbor_count, self.rest_neighbor_ids, C,
                ],
                outputs=[self.y_ze, self.Y_ze, self.Ysum_ze],
            )
            wp.launch(
                kernel=ze_gather_kernel,
                dim=D,
                inputs=[
                    ids, self.y_ze, self.Y_ze, self.Ysum_ze, self.G_sum,
                    self.VjLi_gradW, self.rest_neighbor_count,
                    self.rest_neighbor_ids, self.rest_neighbor_reverse_slot,
                    C, 1.0,
                ],
                outputs=[self.gradient_ze],
            )
            wp.launch(
                kernel=_add_vec3_kernel,
                dim=D,
                inputs=[ids, self.gradient_ze],
                outputs=[self.gradient],
            )

        # Fixed-vertex constraint: zero gradient[fixed] so r0 = -dt·∇Ψ has zero
        # fixed entries → PCG keeps dv[fixed]=0 by induction (combined with the
        # matvec masking in _matvec). One mask is the load-bearing line.
        self._mask_fixed(self.gradient)

    def _newton_step(self, ps, D: int, C: int, dt: float):
        """One Newton iteration: re-linearise + PCG + v += Δv."""
        self.assemble_gradient(dt)
        ids = self.ids

        # 4. PCG  for  (M + dt² K) Δv = -dt ∇Ψ
        dt2 = dt * dt
        wp.launch(
            kernel=compute_block_precond_kernel,
            dim=D,
            inputs=[
                ids, ps.m, self.V0_rest, self.L, self.U, self.V_svd, self.sigma, self.R,
                self.F, self.VjLi_gradW, self.Vj_gradW, self.G_sum,
                self.rest_neighbor_count, self.rest_neighbor_ids,
                self.rest_neighbor_reverse_slot, C, self.mu_p, self.lambda_p,
                self.material_type, self.ze_diag, dt2,
            ],
            outputs=[self.M_pre_inv],
        )
        wp.launch(
            kernel=pcg_init_kernel,
            dim=D,
            inputs=[ids, self.gradient, dt, self.M_pre_inv],
            outputs=[self.dv, self.r, self.z, self.p],
        )

        # PCG main loop. Convergence: ‖r‖ / ‖r0‖ < tol.
        rz_old = self._reduce_dot(self.r, self.z)
        r0_norm_sq = self._reduce_dot(self.r, self.r)
        if r0_norm_sq <= 0.0:
            self.last_cg_iters = 0
            self.last_cg_residual = 0.0
            return
        tol_sq = self.tol_cg * self.tol_cg * r0_norm_sq

        iters = 0
        r_norm_sq = r0_norm_sq
        for it in range(self.max_iter_cg):
            self._matvec(self.p, self.Ap, dt)
            pAp = self._reduce_dot(self.p, self.Ap)
            if pAp <= 0.0:
                print(f"[ElasticityKee2023] PCG: pAp <= 0 at iter {it} ({pAp:.3e}); "
                      f"matrix not SPD — stopping early")
                break
            alpha = rz_old / pAp

            wp.launch(
                kernel=pcg_update_dv_r_kernel,
                dim=D,
                inputs=[ids, alpha, self.p, self.Ap],
                outputs=[self.dv, self.r],
            )

            r_norm_sq = self._reduce_dot(self.r, self.r)
            iters = it + 1
            if r_norm_sq < tol_sq:
                break

            wp.launch(
                kernel=pcg_apply_precond_kernel,
                dim=D,
                inputs=[ids, self.M_pre_inv, self.r],
                outputs=[self.z],
            )
            rz_new = self._reduce_dot(self.r, self.z)
            beta = rz_new / rz_old
            wp.launch(
                kernel=pcg_update_p_kernel,
                dim=D,
                inputs=[ids, beta, self.z],
                outputs=[self.p],
            )
            rz_old = rz_new

        self.last_cg_iters = iters
        self.last_cg_residual = float(np.sqrt(max(r_norm_sq, 0.0) / max(r0_norm_sq, 1e-30)))

        # 5. Velocity update
        wp.launch(
            kernel=apply_velocity_update_kernel,
            dim=D,
            inputs=[ids, self.dv],
            outputs=[ps.v],
        )

    def validate_F_at_rest(self, k0: int = 0, k1=None) -> float:
        """Principal strain max_i max_k |sigma_k(F_i(x0)) - 1| over ids[k0:k1] (rank-deficient
        particles completed like in the solve). Must be ~0: checks L, the rest lists and
        the neighbour cache against each other."""
        ps = self.ps
        if k1 is None:
            k1 = self.count
        D = int(k1 - k0)
        if D <= 0:
            return 0.0
        ids = self.ids[k0:k1]
        C = int(ps.cache_size)

        wp.launch(
            kernel=compute_F_kernel,
            dim=D,
            inputs=[
                ids, ps.x0, self.rest_neighbor_count, self.rest_neighbor_ids,
                self.VjLi_gradW, C,
            ],
            outputs=[self.F],
        )
        wp.launch(
            kernel=compute_polar_P_PL_kernel,
            dim=D,
            inputs=[ids, self.F, self.L, self.rank, self.mu_p, self.lambda_p],
            outputs=[self.U, self.V_svd, self.sigma, self.R, self.PL],
        )
        out = wp.zeros(1, dtype=float)
        wp.launch(
            kernel=reduce_max_principal_strain_kernel,
            dim=D,
            inputs=[ids, self.sigma],
            outputs=[out],
        )
        return float(out.numpy()[0])
