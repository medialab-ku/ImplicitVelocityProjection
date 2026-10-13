"""
Neighbor search for WarpSPH simulation using Warp's HashGrid.
"""

import warp as wp
import math
from .sph_kernel import spiky_kernel_derivative


@wp.kernel
def build_neighbor_cache(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    num_particles: int,
    support_radius: float,
    cache_size: int,
    # Outputs
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
):
    """
    Build the fixed-size neighbor list from HashGrid.

    neighbor_ids and neighbor_gradW are flattened 2D arrays:
        base = i * cache_size
        neighbor_ids[base + k] = j
        neighbor_gradW[base + k] = gradW(x_i - x_j)
    """
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    x_i = x[p_i]
    base = p_i * cache_size
    cnt = int(0)

    neighbors = wp.hash_grid_query(grid, x_i, support_radius)
    for p_j in neighbors:
        if p_j == p_i:
            continue
        r = x_i - x[p_j]
        rn = wp.length(r)
        if rn < support_radius:
            if cnt < cache_size:
                neighbor_ids[base + cnt] = p_j
                neighbor_gradW[base + cnt] = spiky_kernel_derivative(r, support_radius)
                cnt += 1

    # neighbor_count is capped at cache_size
    neighbor_count[p_i] = cnt


@wp.kernel
def update_neighbor_gradW(
    x: wp.array(dtype=wp.vec3),
    num_particles: int,
    support_radius: float,
    cache_size: int,
    neighbor_count: wp.array(dtype=int),
    neighbor_ids: wp.array(dtype=int),
    neighbor_gradW: wp.array(dtype=wp.vec3),
):
    """Recompute gradW from existing neighbor indices and updated positions."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    x_i = x[p_i]
    base = p_i * cache_size
    cnt = neighbor_count[p_i]
    if cnt > cache_size:
        cnt = cache_size

    for k in range(cnt):
        p_j = neighbor_ids[base + k]
        r = x_i - x[p_j]
        neighbor_gradW[base + k] = spiky_kernel_derivative(r, support_radius)


class NeighbourSearch:
    """
    HashGrid-based neighbor search using Warp's built-in spatial acceleration.

    Call build() and build_cache() every step: the HashGrid only generates candidates,
    and all solvers read the cached neighbor_ids / neighbor_gradW arrays.

    Reference:
        - Warp SPH Example: https://github.com/NVIDIA/warp/blob/main/warp/examples/core/example_sph.py
        - Warp HashGrid Discussion: https://github.com/NVIDIA/warp/discussions/62
    """

    def __init__(self, domain_size: tuple, support_radius: float):
        """
        Initialize HashGrid-based neighbor search.

        Args:
            domain_size: Simulation domain size (width, height, depth)
            support_radius: SPH support radius (smoothing length)
        """
        self.support_radius = support_radius

        # Calculate grid dimensions based on support radius
        # Each grid cell should be approximately support_radius in size
        grid_dim_x = max(1, int(math.ceil(domain_size[0] / support_radius)))
        grid_dim_y = max(1, int(math.ceil(domain_size[1] / support_radius)))
        grid_dim_z = max(1, int(math.ceil(domain_size[2] / support_radius)))

        # Create Warp HashGrid
        # This is a GPU-optimized spatial acceleration structure
        self.grid = wp.HashGrid(grid_dim_x, grid_dim_y, grid_dim_z)

        print(f"NeighbourSearch initialized with grid dimensions: {grid_dim_x}x{grid_dim_y}x{grid_dim_z}")
        print(f"Support radius: {support_radius:.4f}")

    def build(self, positions: wp.array, num_particles: int):
        """
        Build/rebuild the spatial hash grid with current particle positions.
        This must be called every frame before neighbor queries.

        Args:
            positions: Warp array of particle positions (vec3), sized to
                particle_max_num
            num_particles: Number of active particles
        """
        n = int(num_particles)
        if n < len(positions):
            positions = positions[:n]
        self.grid.build(positions, self.support_radius)

    def build_cache(
        self,
        positions: wp.array,
        num_particles: int,
        cache_size: int,
        neighbor_count: wp.array,
        neighbor_ids: wp.array,
        neighbor_gradW: wp.array,
    ):
        """
        Build fixed-size neighbor cache (count/ids/gradW) from current HashGrid.
        Must be called after build().
        """
        wp.launch(
            kernel=build_neighbor_cache,
            dim=num_particles,
            inputs=[
                self.grid.id,
                positions,
                num_particles,
                self.support_radius,
                cache_size,
            ],
            outputs=[neighbor_count, neighbor_ids, neighbor_gradW],
        )

    def recompute_gradW(
        self,
        positions: wp.array,
        num_particles: int,
        cache_size: int,
        neighbor_count: wp.array,
        neighbor_ids: wp.array,
        neighbor_gradW: wp.array,
    ):
        """Recompute gradW from existing neighbor indices and updated positions."""
        wp.launch(
            kernel=update_neighbor_gradW,
            dim=num_particles,
            inputs=[
                positions,
                num_particles,
                self.support_radius,
                cache_size,
                neighbor_count,
                neighbor_ids,
            ],
            outputs=[neighbor_gradW],
        )
