"""
Particle data structure for WarpSPH simulation.

Stores all particle arrays (positions, velocities, densities, etc.)
and provides methods for particle initialization.
"""

import warp as wp
import numpy as np
from scripts.io.config_builder import SimConfig


# Helper function for particle addition
@wp.func
def add_particle(
    p: int,
    object_id: int,
    x: wp.vec3,
    v: wp.vec3,
    density: float,
    material: int,
    is_dynamic: int,
    color: wp.vec4,
    m_V0: float,
    # Output arrays
    object_id_arr: wp.array(dtype=int),
    x_arr: wp.array(dtype=wp.vec3),
    x_old_arr: wp.array(dtype=wp.vec3),
    x0_arr: wp.array(dtype=wp.vec3),
    v_arr: wp.array(dtype=wp.vec3),
    m_arr: wp.array(dtype=float),
    m_inv_arr: wp.array(dtype=float),
    density_arr: wp.array(dtype=float),
    density0_arr: wp.array(dtype=float),
    material_arr: wp.array(dtype=int),
    is_dynamic_arr: wp.array(dtype=int),
    color_arr: wp.array(dtype=wp.vec4)
):

    object_id_arr[p] = object_id
    x_arr[p] = x
    x_old_arr[p] = x
    x0_arr[p] = x
    v_arr[p] = v

    density0_arr[p] = density
    density_arr[p] = density
    m_arr[p] = m_V0 * density

    material_arr[p] = material
    is_dynamic_arr[p] = is_dynamic

    if is_dynamic == 1:
        m_inv_arr[p] = 1.0 / m_arr[p]
    else:
        m_inv_arr[p] = 0.0
        m_arr[p] = m_arr[p] * 10.0  # Static particles have higher mass

    color_arr[p] = color


# Kernel for adding multiple particles
@wp.kernel
def add_particles_kernel(
    particle_num_start: int,
    new_particles_num: int,
    object_id: int,
    new_positions: wp.array(dtype=wp.vec3),
    new_velocities: wp.array(dtype=wp.vec3),
    new_densities: wp.array(dtype=float),
    new_materials: wp.array(dtype=int),
    new_is_dynamic: wp.array(dtype=int),
    new_colors: wp.array(dtype=wp.vec4),
    m_V0: float,
    # Output arrays
    object_id_arr: wp.array(dtype=int),
    x_arr: wp.array(dtype=wp.vec3),
    x_old_arr: wp.array(dtype=wp.vec3),
    x0_arr: wp.array(dtype=wp.vec3),
    v_arr: wp.array(dtype=wp.vec3),
    m_arr: wp.array(dtype=float),
    m_inv_arr: wp.array(dtype=float),
    density_arr: wp.array(dtype=float),
    density0_arr: wp.array(dtype=float),
    material_arr: wp.array(dtype=int),
    is_dynamic_arr: wp.array(dtype=int),
    color_arr: wp.array(dtype=wp.vec4)
):
    """
    Kernel to add multiple particles.
    """
    i = wp.tid()
    if i >= new_particles_num:
        return

    p = particle_num_start + i

    add_particle(
        p, object_id,
        new_positions[i],
        new_velocities[i],
        new_densities[i],
        new_materials[i],
        new_is_dynamic[i],
        new_colors[i],
        m_V0,
        object_id_arr, x_arr, x_old_arr, x0_arr, v_arr,
        m_arr, m_inv_arr, density_arr, density0_arr,
        material_arr, is_dynamic_arr, color_arr
    )


class SimulationData:
    """
    Particle data structure
    """

    def __init__(self, config: SimConfig):

        self.cfg = config
        self.dim = 3  # 3D simulation

        # Domain info
        self.domain_size = np.array([config.domain_size[0], config.domain_size[1], config.domain_size[2]], dtype=np.float32)

        # Material types
        self.material_rigid = 0
        self.material_fluid = 1

        # Particle properties
        self.particle_radius = config.particle_radius
        self.particle_diameter = config.particle_diameter
        self.support_radius = config.support_radius
        self.padding = self.support_radius

        # Particle volume for mass calculation
        self.m_V0_init = 0.8 * self.particle_diameter ** self.dim
        # Deformables are voxelized at pitch a·d (a = `deformableSamplingFactor`, default 1);
        # same 0.8 volume convention per voxel, so a body's mass does not depend on a.
        self.deformable_sampling = float(config.get_cfg("deformableSamplingFactor", 1.0))
        self.m_V0_solid = 0.8 * (self.deformable_sampling * self.particle_diameter) ** self.dim

        # Tracking
        self.num_objects = 1
        self.num_rigid_bodies = 0
        self.rigid_particle_num = 0  # shown by the GUI
        self.particle_num = 0
        self.particle_max_num = 0

        # Deformable particles occupy the contiguous prefix [0, num_deformable).
        # SceneLoader adds SolidBodies first and sets this; everything else appends after.
        self.num_deformable = 0

        # Neighbor cache (fixed-size neighbor list)
        # Filled every frame from HashGrid by NeighbourSearch.
        # Denser deformables (deformableSamplingFactor < 1) need more slots: `neighborCacheSize`.
        self.cache_size = int(config.get_cfg("neighborCacheSize", 60))
        self.neighbor_count = None          # wp.array(int) length particle_max_num
        self.neighbor_ids = None            # wp.array(int) length particle_max_num * cache_size
        self.neighbor_gradW = None          # wp.array(wp.vec3) length particle_max_num * cache_size

        # Arrays will be allocated later
        self.allocated = False

    def allocate(self, particle_max_num: int, num_objects: int = 1, num_rigid_bodies: int = 0):
        """
        Allocate memory for particle arrays.

        Args:
            particle_max_num: Maximum number of particles
            num_objects: Number of objects
            num_rigid_bodies: Number of rigid bodies
        """
        self.particle_max_num = int(particle_max_num)
        self.num_objects = int(max(num_objects, 1))
        self.num_rigid_bodies = int(num_rigid_bodies)

        print(f"Allocating arrays for {self.particle_max_num} particles")
        print(f"Number of rigid bodies: {self.num_rigid_bodies}")

        # Particle related properties
        self.object_id = wp.zeros(self.particle_max_num, dtype=int)
        self.x = wp.zeros(self.particle_max_num, dtype=wp.vec3)
        self.x_old = wp.zeros(self.particle_max_num, dtype=wp.vec3)
        self.x0 = wp.zeros(self.particle_max_num, dtype=wp.vec3)
        self.v = wp.zeros(self.particle_max_num, dtype=wp.vec3)
        self.v0 = wp.zeros(self.particle_max_num, dtype=wp.vec3)  # initial v, for reset

        self.m = wp.zeros(self.particle_max_num, dtype=float)
        self.m_inv = wp.zeros(self.particle_max_num, dtype=float)

        self.n = wp.zeros(self.particle_max_num, dtype=wp.vec3)  # Surface normal
        self.density = wp.zeros(self.particle_max_num, dtype=float)
        self.density0 = wp.zeros(self.particle_max_num, dtype=float)
        self.divergence = wp.zeros(self.particle_max_num, dtype=float)

        self.material = wp.zeros(self.particle_max_num, dtype=int)
        self.color = wp.zeros(self.particle_max_num, dtype=wp.vec4)
        self.is_dynamic = wp.zeros(self.particle_max_num, dtype=int)
        # Elasticity fixed-vertex flag: stays dynamic, but the elastic Newton step is masked to 0.
        self.is_fixed = wp.zeros(self.particle_max_num, dtype=int)
        # Initial-position offset applied after reset (x = x0 + x_perturbation); x0 stays clean.
        self.x_perturbation = wp.zeros(self.particle_max_num, dtype=wp.vec3)
        self.is_kinematic = wp.zeros(self.particle_max_num, dtype=int)

        # Neighbor cache buffers (flattened 2D: [i * cache_size + k])
        self.neighbor_count = wp.zeros(self.particle_max_num, dtype=int)
        self.neighbor_ids = wp.zeros(self.particle_max_num * self.cache_size, dtype=int)
        self.neighbor_gradW = wp.zeros(self.particle_max_num * self.cache_size, dtype=wp.vec3)

        self.allocated = True
        print("SimulationData allocated successfully")

    def add_particles(
        self,
        object_id: int,
        new_particles_num: int,
        new_particles_positions: np.ndarray,
        new_particles_velocity: np.ndarray,
        new_particle_density: np.ndarray,
        new_particles_material: np.ndarray,
        new_particles_is_dynamic: np.ndarray,
        new_particles_color: np.ndarray,
        m_V0: float = None
    ):
        """
        Add multiple particles to the simulation.
        Python wrapper that launches the kernel.

        Args:
            object_id: Object ID
            new_particles_num: Number of particles to add
            new_particles_positions: (N, 3) array of positions
            new_particles_velocity: (N, 3) array of velocities
            new_particle_density: (N,) array of densities
            new_particles_material: (N,) array of material types
            new_particles_is_dynamic: (N,) array of dynamic flags
            new_particles_color: (N, 3) or (N, 4) array of colors
            m_V0: particle volume (default m_V0_init; deformables pass m_V0_solid)
        """
        if not self.allocated:
            raise RuntimeError("Must call allocate() before adding particles")

        if self.particle_num + new_particles_num > self.particle_max_num:
            print(f"Warning: Cannot add {new_particles_num} particles. "
                  f"Only {self.particle_max_num - self.particle_num} slots available.")
            return

        # Convert numpy arrays to Warp arrays
        positions_wp = wp.array(new_particles_positions, dtype=wp.vec3)
        velocities_wp = wp.array(new_particles_velocity, dtype=wp.vec3)
        densities_wp = wp.array(new_particle_density, dtype=float)
        materials_wp = wp.array(new_particles_material, dtype=int)
        is_dynamic_wp = wp.array(new_particles_is_dynamic, dtype=int)

        # Handle color array (convert to vec4 if needed)
        if new_particles_color.shape[1] == 3:
            # Add alpha channel
            colors_rgba = np.hstack([new_particles_color, np.full((new_particles_num, 1), 255)])
        else:
            colors_rgba = new_particles_color
        colors_wp = wp.array(colors_rgba, dtype=wp.vec4)

        # Launch kernel to add particles
        wp.launch(
            kernel=add_particles_kernel,
            dim=new_particles_num,
            inputs=[
                self.particle_num,
                new_particles_num,
                object_id,
                positions_wp,
                velocities_wp,
                densities_wp,
                materials_wp,
                is_dynamic_wp,
                colors_wp,
                self.m_V0_init if m_V0 is None else float(m_V0)
            ],
            outputs=[
                self.object_id, self.x, self.x_old, self.x0, self.v,
                self.m, self.m_inv, self.density, self.density0,
                self.material, self.is_dynamic, self.color
            ]
        )

        self.particle_num += new_particles_num


    def add_cube(
        self,
        object_id,
        lower_corner,
        cube_size,
        material,
        is_dynamic,
        color=(0, 0, 0),
        density=None,
        velocity=None
    ):
        """
        Add a rectangular block of particles (cube/cuboid).

        Args:
            object_id: Object ID
            lower_corner: Bottom-left-front corner position (x, y, z)
            cube_size: Size of the block (width, height, depth)
            material: Material type (0=rigid, 1=fluid)
            is_dynamic: 1 for dynamic, 0 for static
            color: RGB color tuple (r, g, b)
            density: Rest density (defaults to 1000.0)
            velocity: Initial velocity (vx, vy, vz) (defaults to zero)
        """
        from functools import reduce

        # Generate particle positions in a grid
        num_dim = []
        for i in range(self.dim):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i], self.particle_diameter))

        num_new_particles = reduce(lambda x, y: x * y, [len(n) for n in num_dim])

        # Create meshgrid of positions
        new_positions = np.array(
            np.meshgrid(*num_dim, sparse=False, indexing='ij'),
            dtype=np.float32
        )
        new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()

        # Velocity array
        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        # Material and dynamic arrays
        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)

        # Color array
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)

        # Density array
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.0)

        # Add particles to system
        self.add_particles(
            object_id,
            num_new_particles,
            new_positions,
            velocity_arr,
            density_arr,
            material_arr,
            is_dynamic_arr,
            color_arr
        )
