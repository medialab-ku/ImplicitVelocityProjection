"""Scheduled body spawn system.

Injects meshes, voxelized by SceneLoader at startup, as fluid, rigid or deformable
particles at scheduled simulation times.
"""

import numpy as np

# ScheduledSpawn `material` values that spawn an elastic body (fluid-tagged
# particles registered with the elasticity solver by Framework.forward()).
DEFORMABLE_MATERIALS = ("solid", "deformable", "elastic")


class ScheduledSpawn:
    """A single scheduled spawn."""

    def __init__(self, time, object_id, positions, velocity, density, color,
                 material="fluid", is_dynamic=True, elastic_params=None):
        self.time = float(time)
        self.object_id = int(object_id)
        # Per-body youngsModulus / poissonsRatio for ElasticityKee2023.register.
        self.elastic_params = dict(elastic_params) if elastic_params else {}
        self.positions = positions.astype(np.float32, copy=False)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.density = float(density)
        self.color = np.array(color, dtype=np.int32)
        # material code: 0 = rigid; fluid AND deformable are 1 (deformables are
        # fluid-tagged, the elasticity solver tracks them by index).
        self.material_code = 0 if material == "rigid" else 1
        self.is_deformable = material in DEFORMABLE_MATERIALS
        self.is_dynamic = 1 if (is_dynamic or self.is_deformable) else 0
        self.fired = False


class ScheduledSpawnSystem:
    """Manages scheduled spawns: pre-voxelized meshes injected at set times."""

    def __init__(self, ps):
        self.ps = ps
        self.spawns = []
        self._sorted = True
        # (start, n, elastic_params) of deformable bodies spawned since the last pop.
        self.pending_deformable = []
        # Rigid particles scheduled / spawned so far; spawned ones are counted into
        # ps.rigid_particle_num (GUI stats).
        self.rigid_capacity = 0
        self.rigid_spawned = 0

    def add(self, spawn: ScheduledSpawn):
        self.spawns.append(spawn)
        if spawn.material_code == 0:
            self.rigid_capacity += spawn.positions.shape[0]
        self._sorted = False

    def reset(self):
        self.pending_deformable = []
        self.ps.rigid_particle_num -= self.rigid_spawned
        self.rigid_spawned = 0
        for s in self.spawns:
            s.fired = False

    def pop_pending_deformable(self):
        """Return and clear the deformable ranges spawned since the last call."""
        out = self.pending_deformable
        self.pending_deformable = []
        return out

    def step(self, t):
        """Fire any spawns whose time has arrived. Returns #particles added."""
        if not self._sorted:
            # Fire in time order (stable), so particle slots follow the spawn schedule.
            self.spawns.sort(key=lambda s: s.time)
            self._sorted = True
        added = 0
        for s in self.spawns:
            if s.fired:
                continue
            if t < s.time:
                continue
            n = s.positions.shape[0]
            remain = int(self.ps.particle_max_num) - int(self.ps.particle_num)
            if remain <= 0:
                s.fired = True
                continue
            if s.is_deformable and n > remain:
                # A partial body has no valid rest configuration; skip it instead.
                print(f"[spawn] not enough slots for deformable object {s.object_id} "
                      f"({n} > {remain}); skipped")
                s.fired = True
                continue
            n = min(n, remain)
            P = s.positions[:n]
            V = np.tile(s.velocity[None, :], (n, 1)).astype(np.float32)
            start = int(self.ps.particle_num)
            self.ps.add_particles(
                object_id=s.object_id,
                new_particles_num=n,
                new_particles_positions=P,
                new_particles_velocity=V,
                new_particle_density=np.full(n, s.density, dtype=np.float32),
                new_particles_material=np.full(n, s.material_code, dtype=np.int32),
                new_particles_is_dynamic=np.full(n, s.is_dynamic, dtype=np.int32),
                new_particles_color=np.stack([
                    np.full(n, s.color[0], dtype=np.int32),
                    np.full(n, s.color[1], dtype=np.int32),
                    np.full(n, s.color[2], dtype=np.int32),
                ], axis=1),
                m_V0=self.ps.m_V0_solid if s.is_deformable else None,
            )
            s.fired = True
            added += n
            if s.material_code == 0:
                self.ps.rigid_particle_num += n
                self.rigid_spawned += n
            if s.is_deformable:
                self.pending_deformable.append((start, n, s.elastic_params))
        return added
