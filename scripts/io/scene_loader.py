"""
Scene loader for WarpSPH simulation.

Handles loading scene definitions from JSON and voxelizing .obj meshes into particles.
"""

import pathlib

import numpy as np
import trimesh as tm
import warp as wp
from .config_builder import SimConfig
from scripts.sim.simulation_data import SimulationData
from scripts.sim.utils.scheduled_spawn import DEFORMABLE_MATERIALS


def _color255(color) -> np.ndarray:
    """Scene RGB colour as int 0-255; colours given in 0-1 are scaled up."""
    c = np.asarray(color, dtype=np.float32)[:3]
    return (c * 255.0 if c.max() <= 1.0 else c).astype(np.int32)


@wp.kernel
def _randomize_positions_kernel(
    particle_start: int,
    n: int,
    new_positions: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    x_perturbation: wp.array(dtype=wp.vec3),
):
    """x_perturbation[i] = new_position - x0[i] over the just-added slice.
    Framework.initialize() applies it after the elasticity rest-config init."""
    p_local = wp.tid()
    if p_local >= n:
        return
    p_i = particle_start + p_local
    x_perturbation[p_i] = new_positions[p_local] - x0[p_i]


class SceneLoader:
    """
    Scene loader that processes JSON scene files and creates particles.
    """

    def __init__(self, config: SimConfig):
        """
        Initialize scene loader.

        Args:
            config: SimConfig instance
        """
        self.cfg = config
        self.dim = 3  # 3D simulation
        self.particle_diameter = config.particle_diameter
        # Deformable voxel pitch factor (pitch = a·d); particles get m_V0 = 0.8 (a·d)³ (SimulationData).
        self.deformable_sampling = float(config.get_cfg("deformableSamplingFactor", 1.0))

    def get_scene_name(self, scene_file_path: str) -> str:
        """
        Extract scene name from file path.

        Args:
            scene_file_path: Path to scene file

        Returns:
            Scene name
        """
        return pathlib.Path(scene_file_path).stem if scene_file_path else "scene"

    def prepare_scene(self):
        """
        Parse scene configuration and compute particle counts.

        Returns:
            Dictionary with scene data including particle counts
        """
        fluid_blocks = list(self.cfg.get_fluid_blocks() or [])
        fluid_bodies = list(self.cfg.get_fluid_bodies() or [])
        rigid_bodies = list(self.cfg.get_rigid_bodies() or [])
        solid_bodies = list(self.cfg.get_solid_bodies() or [])

        object_ids = []
        object_map = {}

        fluid_cnt = 0
        rigid_cnt = 0
        solid_cnt = 0

        # Process Fluid Blocks
        for fluid in fluid_blocks:
            object_ids.append(int(fluid["objectId"]))
            object_map[fluid["objectId"]] = fluid
            t = np.array(fluid.get("translation", [0, 0, 0]), dtype=np.float32)
            s = np.array(fluid.get("scale", [1, 1, 1]), dtype=np.float32)
            start = np.array(fluid["start"], dtype=np.float32) + t
            size = (np.array(fluid["end"], dtype=np.float32) - np.array(fluid["start"], dtype=np.float32)) * s
            end = start + size
            particle_num = self.compute_cube_particle_num(start, end)
            fluid["worldStart"] = start
            fluid["worldEnd"] = end
            fluid["particleNum"] = particle_num
            fluid_cnt += particle_num

        # Process Fluid Bodies (.obj meshes)
        for fb in fluid_bodies:
            object_ids.append(int(fb["objectId"]))
            object_map[fb["objectId"]] = fb
            voxelized_points_np = self.load_mesh(fb, a=1.0)
            fb["particleNum"] = voxelized_points_np.shape[0]
            fb["voxelizedPoints"] = voxelized_points_np
            fluid_cnt += voxelized_points_np.shape[0]

        # Process Rigid Bodies (.obj meshes)
        for rigid_body in rigid_bodies:
            object_ids.append(int(rigid_body["objectId"]))
            object_map[rigid_body["objectId"]] = rigid_body
            voxelized_points_np = self.load_mesh(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            rigid_cnt += voxelized_points_np.shape[0]

        # Process Deformable Bodies (.obj meshes)
        for sb in solid_bodies:
            object_ids.append(int(sb["objectId"]))
            object_map[sb["objectId"]] = sb
            voxelized_points_np = self.load_mesh(sb, a=self.deformable_sampling)
            sb["particleNum"] = voxelized_points_np.shape[0]
            sb["voxelizedPoints"] = voxelized_points_np
            solid_cnt += voxelized_points_np.shape[0]

        # Process Generated Bodies (procedural rigid shapes)
        generated_bodies = list(self.cfg.get_generated_bodies() or [])
        for gb in generated_bodies:
            obj_id = int(gb["objectId"])
            object_ids.append(obj_id)
            object_map[obj_id] = gb
            positions = self._generate_body(gb)
            gb["generatedPositions"] = positions
            gb["particleNum"] = positions.shape[0]
            rigid_cnt += positions.shape[0]

        # Process ScheduledSpawns (pre-voxelize each mesh; expand batch entries)
        raw_spawn_defs = list(self.cfg.get_scheduled_spawns() or [])
        spawn_defs = []
        spawn_capacity = 0
        for sp in raw_spawn_defs:
            deformable = str(sp.get("material", "fluid")).lower() in DEFORMABLE_MATERIALS
            voxelized = self.load_mesh(sp, a=self.deformable_sampling if deformable else 1.0)
            num = int(sp.get("spawnNum", 1))
            start_time = float(sp.get("startTime", sp.get("time", 0.0)))
            interval = float(sp.get("interval", 1.0))
            base_obj_id = int(sp["objectId"])
            # Optional "grid" replicates the body into a lattice centred on `translation`;
            # all copies of one wave spawn together, and `spawnNum` counts the waves.
            grid_offsets = self._spawn_grid_offsets(sp)
            slots = [voxelized + off for off in grid_offsets]
            has_grid = len(slots) > 1
            colors = sp.get("colorCycle") or [sp.get("color", [10, 100, 200])]
            # With a grid, one object id per grid slot is reused by every wave, so long
            # streams stay inside the visualizer's 256-id tables; without a grid each
            # wave gets its own id.
            for i in range(len(slots) if has_grid else num):
                oid = base_obj_id + i
                if oid in object_ids:
                    print(f"Warning: ScheduledSpawn objectId {oid} collides with an existing "
                          f"object; give each spawn stream its own id range")
                else:
                    object_ids.append(oid)
            # Optional "waveOffset" shifts wave k by k * offset (e.g. raise the spawn
            # height as the pile grows).
            wave_offset = np.array(sp.get("waveOffset", [0.0, 0.0, 0.0]), dtype=np.float32)
            base_translation = np.array(sp.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            for k in range(num):
                shift = k * wave_offset
                for g, positions in enumerate(slots):
                    entry = dict(sp)
                    entry["voxelizedPoints"] = (positions + shift) if k else positions
                    entry["translation"] = (base_translation + shift).tolist()
                    entry["particleNum"] = positions.shape[0]
                    entry["time"] = start_time + k * interval
                    entry["objectId"] = base_obj_id + (g if has_grid else k)
                    entry["color"] = colors[(k + g) % len(colors)]  # rotates per wave for any slot count
                    spawn_defs.append(entry)
                    object_map[entry["objectId"]] = entry
                    spawn_capacity += positions.shape[0]

        # Generate boundary wall particles
        boundary_layers = int(self.cfg.scene_data.get("Configuration", {}).get("boundaryLayers", 0))
        boundary_cnt = 0
        boundary_positions = None
        if boundary_layers > 0:
            boundary_positions = self._generate_boundary_walls(boundary_layers)
            boundary_cnt = boundary_positions.shape[0]
            rigid_cnt += boundary_cnt

        num_objects = (max(object_ids) + 1) if object_ids else 1
        particle_max_num = fluid_cnt + rigid_cnt + solid_cnt + spawn_capacity

        return {
            "fluid_blocks": fluid_blocks,
            "fluid_bodies": fluid_bodies,
            "rigid_bodies": rigid_bodies,
            "generated_bodies": generated_bodies,
            "solid_bodies": solid_bodies,
            "spawn_defs": spawn_defs,
            "boundary_positions": boundary_positions,
            "particle_max_num": int(particle_max_num),
            "fluid_particle_num": int(fluid_cnt),
            "rigid_particle_num": int(rigid_cnt),
            "solid_particle_num": int(solid_cnt),
            "num_objects": int(num_objects),
            "num_rigid_bodies": int(len(rigid_bodies)),
            "object_map": object_map,
            "object_ids": object_ids
        }

    def populate_scene(self, ps: SimulationData, scene_data: dict):
        """
        Create particles from scene data.

        Args:
            ps: SimulationData instance
            scene_data: Scene data dictionary from prepare_scene()
        """
        # Allocate memory
        ps.allocate(
            particle_max_num=scene_data["particle_max_num"],
            num_objects=scene_data["num_objects"],
            num_rigid_bodies=scene_data["num_rigid_bodies"]
        )

        ps.rigid_particle_num = scene_data["rigid_particle_num"]

        print("\nPopulating scene:")
        print(f"  Fluid particles: {scene_data['fluid_particle_num']}")
        print(f"  Rigid particles: {scene_data['rigid_particle_num']}")
        print(f"  Solid particles: {scene_data['solid_particle_num']}")

        # Solid (deformable) bodies go first: Framework registers the load-time bodies
        # as the prefix [0, num_deformable). Tagged material=fluid so they behave as
        # plain fluid when elasticity is off.
        for solid_body in scene_data["solid_bodies"]:
            obj_id = int(solid_body["objectId"])
            num_particles_obj = solid_body["particleNum"]
            voxelized_points_np = solid_body["voxelizedPoints"]
            velocity = np.array(solid_body.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
            density = float(solid_body.get("density", 1000.0))
            color = _color255(solid_body.get("color", [200, 80, 80]))

            particle_start = int(ps.particle_num)
            ps.add_particles(
                obj_id,
                num_particles_obj,
                voxelized_points_np,
                np.stack([velocity for _ in range(num_particles_obj)]),
                density * np.ones(num_particles_obj, dtype=np.float32),
                np.ones(num_particles_obj, dtype=np.int32),   # material=fluid (1)
                np.ones(num_particles_obj, dtype=np.int32),   # is_dynamic=1
                np.stack([color for _ in range(num_particles_obj)]),
                m_V0=ps.m_V0_solid,
            )

            # Optional initial-state randomization, staged into ps.x_perturbation.
            #   randomizeMagnitude > 0 : x0 + uniform[-d, d]^3, clamped to the AABB
            #   randomize = true       : uniform-random inside the AABB
            do_random = bool(solid_body.get("randomize", False))
            magnitude = float(solid_body.get("randomizeMagnitude", 0.0))
            if do_random or magnitude > 0.0:
                seed = int(solid_body.get("randomizeSeed", 0))
                clamp = solid_body.get("randomizeClampAABB")
                if clamp is None:
                    lo = voxelized_points_np.min(axis=0).astype(np.float32)
                    hi = voxelized_points_np.max(axis=0).astype(np.float32)
                else:
                    lo = np.array(clamp[:3], dtype=np.float32)
                    hi = np.array(clamp[3:], dtype=np.float32)
                rng = np.random.default_rng(seed)
                if magnitude > 0.0:
                    new_positions_np = (
                        voxelized_points_np
                        + rng.uniform(-magnitude, magnitude, size=(num_particles_obj, 3)).astype(np.float32)
                    )
                    new_positions_np = np.clip(new_positions_np, lo, hi)
                    mode = f"perturbed by +/-{magnitude*1000:.1f} mm"
                else:
                    new_positions_np = (
                        lo + rng.uniform(size=(num_particles_obj, 3)).astype(np.float32) * (hi - lo)
                    )
                    mode = "FULLY randomized"
                new_positions_wp = wp.array(new_positions_np, dtype=wp.vec3, device=wp.get_device())
                wp.launch(
                    kernel=_randomize_positions_kernel,
                    dim=num_particles_obj,
                    inputs=[particle_start, num_particles_obj, new_positions_wp, ps.x0],
                    outputs=[ps.x_perturbation],
                )
                print(f"  Solid body {obj_id}: {mode} {num_particles_obj} positions "
                      f"in box {lo.tolist()}..{hi.tolist()} (seed {seed})")

            # Optional fixed vertices: rest positions inside fixedAABB
            # [xmin,ymin,zmin, xmax,ymax,zmax] get is_fixed=1 (inclusive).
            fixed_aabb = solid_body.get("fixedAABB")
            if fixed_aabb is not None:
                f_lo = np.array(fixed_aabb[:3], dtype=np.float32)
                f_hi = np.array(fixed_aabb[3:], dtype=np.float32)
                inside = np.all(
                    (voxelized_points_np >= f_lo) & (voxelized_points_np <= f_hi), axis=1
                ).astype(np.int32)
                n_fixed = int(inside.sum())
                if n_fixed > 0:
                    is_fixed_host = ps.is_fixed.numpy()
                    is_fixed_host[particle_start:particle_start + num_particles_obj] = inside
                    ps.is_fixed = wp.array(is_fixed_host, dtype=int, device=wp.get_device())
                print(f"  Solid body {obj_id}: fixed {n_fixed} / {num_particles_obj} "
                      f"particles inside AABB {f_lo.tolist()}..{f_hi.tolist()}")
        ps.num_deformable = ps.particle_num

        # Add fluid blocks
        for fluid in scene_data["fluid_blocks"]:
            obj_id = int(fluid["objectId"])
            start = fluid["worldStart"]
            end = fluid["worldEnd"]
            velocity = np.array(fluid.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
            density = float(fluid.get("density", 1000.0))
            color = tuple(_color255(fluid.get("color", [10, 100, 200])))

            ps.add_cube(
                object_id=obj_id,
                lower_corner=tuple(start),
                cube_size=tuple(end - start),
                material=ps.material_fluid,
                is_dynamic=1,
                color=color,
                density=density,
                velocity=tuple(velocity)
            )

        # Add fluid bodies (voxelized .obj meshes)
        for fluid_body in scene_data["fluid_bodies"]:
            obj_id = int(fluid_body["objectId"])
            num_particles_obj = fluid_body["particleNum"]
            voxelized_points_np = fluid_body["voxelizedPoints"]
            velocity = np.array(fluid_body.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
            density = float(fluid_body.get("density", 1000.0))
            color = _color255(fluid_body.get("color", [10, 100, 200]))

            # Per-particle velocity: v_i = v_linear + omega x (r_i - r_cm)
            omega = np.array(fluid_body.get("angularVelocity", [0.0, 0.0, 0.0]), dtype=np.float32)
            if np.linalg.norm(omega) > 0.0:
                r_cm = voxelized_points_np.mean(axis=0)
                r_rel = voxelized_points_np - r_cm
                velocities = np.cross(omega, r_rel).astype(np.float32) + velocity
            else:
                velocities = np.stack([velocity for _ in range(num_particles_obj)])

            ps.add_particles(
                obj_id,
                num_particles_obj,
                voxelized_points_np,
                velocities,
                density * np.ones(num_particles_obj, dtype=np.float32),
                np.ones(num_particles_obj, dtype=np.int32),  # material=fluid
                np.ones(num_particles_obj, dtype=np.int32),  # is_dynamic=1
                np.stack([color for _ in range(num_particles_obj)])
            )

        # Add rigid bodies (voxelized .obj meshes)
        for rigid_body in scene_data["rigid_bodies"]:
            obj_id = int(rigid_body["objectId"])
            is_dynamic = int(bool(rigid_body.get("isDynamic", False)))
            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            desired_velocity = np.array(rigid_body.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
            velocity = desired_velocity if is_dynamic else np.zeros(3, dtype=np.float32)
            density = float(rigid_body.get("density", 1000.0))
            color = _color255(rigid_body.get("color", [255, 255, 255]))

            ps.add_particles(
                obj_id,
                num_particles_obj,
                voxelized_points_np,
                np.stack([velocity for _ in range(num_particles_obj)]),
                density * np.ones(num_particles_obj, dtype=np.float32),
                np.zeros(num_particles_obj, dtype=np.int32),  # material=rigid
                is_dynamic * np.ones(num_particles_obj, dtype=np.int32),
                np.stack([color for _ in range(num_particles_obj)])
            )

        # Add generated bodies (procedural rigid shapes)
        for gb in scene_data["generated_bodies"]:
            obj_id = int(gb["objectId"])
            positions = gb["generatedPositions"]
            n = positions.shape[0]
            is_dynamic = int(bool(gb.get("isDynamic", False)))
            density = float(gb.get("density", 1000.0))
            color = _color255(gb.get("color", [200, 200, 200]))

            ps.add_particles(
                obj_id, n, positions,
                np.zeros((n, 3), dtype=np.float32),
                density * np.ones(n, dtype=np.float32),
                np.zeros(n, dtype=np.int32),    # material=rigid
                is_dynamic * np.ones(n, dtype=np.int32),
                np.stack([color for _ in range(n)])
            )
            print(f"  Generated body {obj_id}: {n} particles")

        # Add boundary wall particles
        boundary_positions = scene_data.get("boundary_positions")
        if boundary_positions is not None:
            n = boundary_positions.shape[0]
            # Use the next available object ID for boundary walls
            boundary_obj_id = scene_data["num_objects"]
            density = float(self.cfg.density0)
            color = np.array([180, 180, 180], dtype=np.int32)
            ps.add_particles(
                boundary_obj_id,
                n,
                boundary_positions,
                np.zeros((n, 3), dtype=np.float32),
                density * np.ones(n, dtype=np.float32),
                np.zeros(n, dtype=np.int32),    # material=rigid
                np.zeros(n, dtype=np.int32),    # is_dynamic=0
                np.stack([color for _ in range(n)])
            )
            print(f"  Boundary wall particles: {n}")

        # Initial velocities, restored by Framework.reset().
        wp.copy(ps.v0, ps.v)

    @staticmethod
    def _spawn_grid_offsets(sp) -> np.ndarray:
        """Offsets of a ScheduledSpawn's grid copies, centred on its translation.

        "grid": {"counts": [nx, ny, nz], "spacing": [dx, dy, dz]}. Absent or all-ones
        counts give a single copy at the original position.
        """
        grid = sp.get("grid")
        if not grid:
            return np.zeros((1, 3), dtype=np.float32)
        counts = [max(1, int(c)) for c in grid.get("counts", [1, 1, 1])]
        spacing = [float(s) for s in grid.get("spacing", [0.0, 0.0, 0.0])]
        axes = [(np.arange(n) - 0.5 * (n - 1)) * s for n, s in zip(counts, spacing)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    def _generate_body(self, gb):
        """Generate a procedural rigid body shape.

        Supported types: "cube" (hollow shell of boundary particles).

        Args:
            gb: Generated body dict with type, translation, size, layers.

        Returns:
            (N, 3) float32 array of particle positions.
        """
        body_type = gb.get("type", "cube")
        translation = np.array(gb.get("translation", [0, 0, 0]), dtype=np.float32)
        size = np.array(gb.get("size", [1, 1, 1]), dtype=np.float32)
        layers = int(gb.get("layers", 2))
        diam = self.particle_diameter

        if body_type == "cube":
            half = size / 2.0
            start = translation - half
            end = translation + half
            positions = []
            for layer in range(layers):
                offset = layer * diam
                s = start - offset
                e = end + offset
                xs = np.arange(s[0], e[0] + diam * 0.5, diam)
                ys = np.arange(s[1], e[1] + diam * 0.5, diam)
                zs = np.arange(s[2], e[2] + diam * 0.5, diam)

                # 6 faces
                gx, gz = np.meshgrid(xs, zs, indexing='ij')
                positions.append(np.stack([gx.ravel(), np.full(gx.size, s[1]), gz.ravel()], axis=1))
                positions.append(np.stack([gx.ravel(), np.full(gx.size, e[1]), gz.ravel()], axis=1))

                gy, gz2 = np.meshgrid(ys, zs, indexing='ij')
                positions.append(np.stack([np.full(gy.size, s[0]), gy.ravel(), gz2.ravel()], axis=1))
                positions.append(np.stack([np.full(gy.size, e[0]), gy.ravel(), gz2.ravel()], axis=1))

                gx2, gy2 = np.meshgrid(xs, ys, indexing='ij')
                positions.append(np.stack([gx2.ravel(), gy2.ravel(), np.full(gx2.size, s[2])], axis=1))
                positions.append(np.stack([gx2.ravel(), gy2.ravel(), np.full(gx2.size, e[2])], axis=1))

            return np.concatenate(positions, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown generated body type: {body_type}")

    def _generate_boundary_walls(self, num_layers):
        """Generate boundary particles along domain walls.

        Args:
            num_layers: Number of particle layers at each wall.

        Returns:
            (N, 3) float32 array of boundary particle positions.
        """
        diam = self.particle_diameter
        start = np.array(self.cfg.domain_start, dtype=np.float32)
        end = start + np.array(self.cfg.domain_size, dtype=np.float32)

        positions = []
        for layer in range(num_layers):
            offset = layer * diam

            # Generate a full grid for this layer's shell
            xs = np.arange(start[0] - offset, end[0] + offset + diam * 0.5, diam)
            ys = np.arange(start[1] - offset, end[1] + offset + diam * 0.5, diam)
            zs = np.arange(start[2] - offset, end[2] + offset + diam * 0.5, diam)

            # Bottom wall (y = start_y - offset)
            gx, gz = np.meshgrid(xs, zs, indexing='ij')
            bottom = np.stack([gx.ravel(), np.full(gx.size, start[1] - offset), gz.ravel()], axis=1)
            positions.append(bottom)

            # Top wall (y = end_y + offset)
            top = np.stack([gx.ravel(), np.full(gx.size, end[1] + offset), gz.ravel()], axis=1)
            positions.append(top)

            # Left wall (x = start_x - offset)
            gy, gz2 = np.meshgrid(ys, zs, indexing='ij')
            left = np.stack([np.full(gy.size, start[0] - offset), gy.ravel(), gz2.ravel()], axis=1)
            positions.append(left)

            # Right wall (x = end_x + offset)
            right = np.stack([np.full(gy.size, end[0] + offset), gy.ravel(), gz2.ravel()], axis=1)
            positions.append(right)

            # Front wall (z = start_z - offset)
            gx2, gy2 = np.meshgrid(xs, ys, indexing='ij')
            front = np.stack([gx2.ravel(), gy2.ravel(), np.full(gx2.size, start[2] - offset)], axis=1)
            positions.append(front)

            # Back wall (z = end_z + offset)
            back = np.stack([gx2.ravel(), gy2.ravel(), np.full(gx2.size, end[2] + offset)], axis=1)
            positions.append(back)

        all_positions = np.concatenate(positions, axis=0).astype(np.float32)
        print(f"  Generated {all_positions.shape[0]} boundary wall particles ({num_layers} layers)")
        return all_positions

    @staticmethod
    def load_transformed_mesh(obj: dict):
        """Load OBJ and apply scale → rotate → translate (the same pipeline used for voxelization)."""
        mesh = tm.load(obj["geometryFile"], force="mesh")
        mesh.apply_scale(obj.get("scale", [1, 1, 1]))
        angle = obj.get("rotationAngle", 0) / 360 * 2 * np.pi
        direction = obj.get("rotationAxis", [0, 1, 0])
        if abs(angle) > 1e-6:
            rot_matrix = tm.transformations.rotation_matrix(
                angle, direction, mesh.vertices.mean(axis=0)
            )
            mesh.apply_transform(rot_matrix)
        offset = np.array(obj.get("translation", [0, 0, 0]))
        mesh.vertices += offset
        return mesh

    def load_mesh(self, obj: dict, a: float = 1.0) -> np.ndarray:
        """
        Load .obj mesh file and voxelize it into particles.

        Args:
            obj: Object dictionary with geometryFile, scale, translation, rotation
            a: Voxel spacing factor (multiplies particle_diameter)

        Returns:
            Array of voxelized particle positions (N, 3)
        """
        obj_id = obj["objectId"]

        mesh = SceneLoader.load_transformed_mesh(obj)

        # Repair mesh (fill holes)
        try:
            tm.repair.fill_holes(mesh)
        except Exception as e:
            print(f"Warning: Could not repair mesh for object {obj_id}: {e}")

        # Voxelize mesh
        voxelized_mesh = mesh.voxelized(
            pitch=a * self.particle_diameter,
            max_iter=15
        ).fill()

        voxelized_points_np = voxelized_mesh.points
        print(f"Mesh object {obj_id} loaded. Particle count: {voxelized_points_np.shape[0]}")

        return voxelized_points_np

    def setup_scheduled_spawn_system(self, ps, scene_data):
        """Create ScheduledSpawnSystem from spawn definitions."""
        from scripts.sim.utils.scheduled_spawn import ScheduledSpawnSystem, ScheduledSpawn

        defs = scene_data.get("spawn_defs", [])
        if not defs:
            return None

        spawn_system = ScheduledSpawnSystem(ps)
        verbose = len(defs) <= 20
        for sp in defs:
            positions = sp["voxelizedPoints"]
            spawn_system.add(ScheduledSpawn(
                time=float(sp.get("time", 0.0)),
                object_id=int(sp["objectId"]),
                positions=positions,
                velocity=sp.get("velocity", [0.0, 0.0, 0.0]),
                density=float(sp.get("density", 1000.0)),
                color=_color255(sp.get("color", [10, 100, 200])),
                material=str(sp.get("material", "fluid")).lower(),
                is_dynamic=bool(sp.get("isDynamic", True)),
                elastic_params={k: sp[k] for k in ("youngsModulus", "poissonsRatio") if k in sp},
            ))
            if verbose:
                print(f"  Scheduled spawn @ t={sp.get('time', 0.0)}s: "
                      f"{positions.shape[0]} particles (objectId={sp['objectId']}, "
                      f"material={sp.get('material', 'fluid')})")
        if not verbose:
            times = [float(sp.get("time", 0.0)) for sp in defs]
            total = sum(int(sp["particleNum"]) for sp in defs)
            print(f"  Scheduled spawns: {len(defs)} bodies, {total} particles, "
                  f"t = {min(times):.2f}..{max(times):.2f}s")
        return spawn_system

    def compute_cube_particle_num(self, start, end):
        """
        Compute number of particles needed for a cube.

        Args:
            start: Start position (x, y, z)
            end: End position (x, y, z)

        Returns:
            Number of particles
        """
        num_dim = [np.arange(start[i], end[i], self.particle_diameter) for i in range(self.dim)]
        n = 1
        for arr in num_dim:
            n *= len(arr)
        return int(n)
