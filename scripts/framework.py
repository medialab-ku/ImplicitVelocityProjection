"""
Framework orchestrator for WarpSPH simulation.
"""

import math
import time
import warp as wp
import numpy as np
from scripts.sim.boundary import enforce_boundary as enforce_boundary_kernel
from scripts.sim.forces.non_pressure.viscosity.viscosity import Viscosity
from scripts.sim.forces.non_pressure.viscosity.weiler18 import Weiler18
from scripts.sim.forces.non_pressure.surface_tension.surface_tension import SurfaceTension
from scripts.sim.forces.non_pressure.surface_tension.jeske23 import Jeske23
from scripts.sim.forces.pressure.prj import PRJ
from scripts.sim.forces.pressure.ip import IP
from scripts.sim.forces.pressure.sisph import SISPH
from scripts.sim.forces.pressure.pbf import PBF
from scripts.sim.forces.elasticity.kee2023 import ElasticityKee2023
from scripts.sim.utils.scheduled_spawn import DEFORMABLE_MATERIALS


# ============================================================================
# Physics kernels
# ============================================================================

@wp.kernel
def apply_gravity(
    v: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    is_kinematic: wp.array(dtype=int),
    num_particles: int,
    gravity: wp.vec3,
    dt: float
):
    """v += gravity * dt for dynamic particles."""
    p_i = wp.tid()
    if p_i >= num_particles:
        return
    if is_kinematic[p_i] == 1 or is_dynamic[p_i] == 0:
        return
    v[p_i] = v[p_i] + gravity * dt


@wp.kernel
def compute_max_velocity_sq(
    v: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    result: wp.array(dtype=float),
):
    """Compute max(|v_i|^2) over dynamic particles via atomic max."""
    i = wp.tid()
    if i >= num_particles:
        return
    if is_dynamic[i] == 0:
        return
    v_sq = wp.dot(v[i], v[i])
    wp.atomic_max(result, 0, v_sq)


@wp.kernel
def fill_vec3_far(x: wp.array(dtype=wp.vec3), far: wp.vec3, n: int):
    i = wp.tid()
    if i >= n:
        return
    x[i] = far


@wp.kernel
def zero_beyond_int(arr: wp.array(dtype=int), start: int, end: int):
    i = wp.tid()
    if i < start or i >= end:
        return
    arr[i] = 0


@wp.kernel
def zero_beyond_float(arr: wp.array(dtype=float), start: int, end: int):
    i = wp.tid()
    if i < start or i >= end:
        return
    arr[i] = 0.0


@wp.kernel
def _apply_perturbation_kernel(
    perturbation: wp.array(dtype=wp.vec3),
    n: int,
    x: wp.array(dtype=wp.vec3),
    x_old: wp.array(dtype=wp.vec3),
):
    """x += perturbation, x_old += perturbation over [0, n). Zero outside randomized bodies."""
    p_i = wp.tid()
    if p_i >= n:
        return
    delta = perturbation[p_i]
    x[p_i] = x[p_i] + delta
    x_old[p_i] = x_old[p_i] + delta


@wp.kernel
def _pin_fixed_kernel(
    ids: wp.array(dtype=int),
    is_fixed: wp.array(dtype=int),
    x_old: wp.array(dtype=wp.vec3),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
):
    """fixedAABB anchors: x = x_old (pre-step position), v = 0. Undoes pressure/boundary moves."""
    p_i = ids[wp.tid()]
    if is_fixed[p_i] == 0:
        return
    x[p_i] = x_old[p_i]
    v[p_i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def advect_position(
    x: wp.array(dtype=wp.vec3),
    x_old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    dt: float
):
    """x_old = x; x += v * dt"""
    p_i = wp.tid()
    if p_i >= num_particles:
        return

    x_old[p_i] = x[p_i]

    if is_dynamic[p_i] == 0:
        v[p_i] = wp.vec3(0.0, 0.0, 0.0)

    x[p_i] = x[p_i] + dt * v[p_i]


# ============================================================================
# Framework class
# ============================================================================

class Framework:
    """Per-step simulation orchestrator; see forward() for the step order."""

    def __init__(self, simulation_data, neighbour_search, config):
        self.ps = simulation_data
        self.ns = neighbour_search
        self.cfg = config

        self.dt = config.dt
        self.dt_default = config.dt
        self.use_cfl = bool(config.get_cfg("useCFL", False))
        self.cfl_factor = float(config.get_cfg("cflFactor", 0.4))
        self.cfl_min_dt = float(config.get_cfg("cflMinDt", 1e-4))
        self.v_max_sq_buffer = wp.zeros(1, dtype=float)
        self.v_max_sq_host = wp.zeros(1, dtype=float, device="cpu")

        self.time = 0.0
        self.initial_particle_num = simulation_data.particle_num

        # Per-step stage timings shown in the GUI
        self.stats_pressure_ms = 0.0
        self.stats_elasticity_ms = 0.0

        self.g = wp.vec3(config.gravity[0], config.gravity[1], config.gravity[2])

        # Pressure solver: 0=PRJ, 1=IP, 3=SISPH, 5=PBF
        _pressure_classes = {0: PRJ, 1: IP, 3: SISPH, 5: PBF}
        pressure_method = int(config.get_cfg("pressureMethod", 1))
        if pressure_method not in _pressure_classes:
            raise ValueError(f"Unknown pressureMethod {pressure_method} (valid: {sorted(_pressure_classes)}).")
        pressure_cls = _pressure_classes[pressure_method]
        self.pressure = pressure_cls(simulation_data, neighbour_search, config)

        # Non-pressure force solvers
        _visc_classes = {0: Viscosity, 1: Weiler18}
        visc_cls = _visc_classes.get(config.viscosity_method, Viscosity)
        self.viscosity = visc_cls(simulation_data, neighbour_search, config)

        _st_classes = {0: SurfaceTension, 1: Jeske23}
        st_cls = _st_classes.get(config.surface_tension_method, SurfaceTension)
        self.surface_tension = st_cls(simulation_data, neighbour_search, config)

        # Coefficients (from scene config)
        self.viscosity_coeff = config.viscosity_coeff
        self.surface_tension_coeff = config.surface_tension_coeff

        # Elasticity (Kee et al. 2023). Created when the scene has SolidBodies or any
        # ScheduledSpawn with a deformable material; fluid-only scenes are untouched.
        _D = int(simulation_data.num_deformable)
        _spawns_solid = any(
            str(sp.get("material", "fluid")).lower() in DEFORMABLE_MATERIALS
            for sp in (config.get_scheduled_spawns() or [])
        )
        self.elasticity = (
            ElasticityKee2023(simulation_data, neighbour_search, config)
            if (_D > 0 or _spawns_solid) else None
        )
        # Load-time SolidBodies sit in [0, num_deformable); spawned bodies register later.
        self._initial_deformable = _D
        if self.elasticity is not None and _D > 0:
            solids = list(config.get_solid_bodies() or [])
            counts = [int(sb.get("particleNum", 0)) for sb in solids]
            if sum(counts) != _D:
                solids, counts = [{}], [_D]
            start = 0
            for sb, n in zip(solids, counts):
                self.elasticity.register(start, n, sb)
                start += n
        # GUI/debug toggles: skip the elastic solve, or skip the pressure solve
        # (deformable-only scenes where the density projection fights elasticity).
        self.elasticity_enabled = True
        self.pressure_enabled = bool(config.get_cfg("pressureEnabled", True))
        # Elastic rest data is built once per reset() (needs x == x0); see initialize().
        self._rest_ready = False
        self._has_fixed = bool(simulation_data.is_fixed.numpy()[:_D].any()) if _D > 0 else False

        # Scheduled spawn system (set by main.py)
        self.scheduled_spawn_system = None

        # Animation engine (set by main.py)
        self.animation = None

        self.device = wp.get_device()

    def reset(self):
        """Reset simulation to initial state."""
        ps = self.ps

        # Restore particle count to initial value (before any spawn)
        ps.particle_num = self.initial_particle_num

        # Defensive: park stale spawned slots far outside the domain and clear their
        # flags (the neighbour search only reads [0, particle_num) anyway).
        N = ps.particle_num
        MAX = ps.particle_max_num
        far = wp.vec3(1.0e6, 1.0e6, 1.0e6)
        wp.launch(kernel=fill_vec3_far, dim=MAX, inputs=[ps.x, far, MAX])
        wp.launch(kernel=fill_vec3_far, dim=MAX, inputs=[ps.x_old, far, MAX])
        ps.v.zero_()

        # Clear stale spawned-slot flags/masses beyond initial particle range
        wp.launch(kernel=zero_beyond_int, dim=MAX, inputs=[ps.is_dynamic, N, MAX])
        wp.launch(kernel=zero_beyond_int, dim=MAX, inputs=[ps.is_kinematic, N, MAX])
        wp.launch(kernel=zero_beyond_float, dim=MAX, inputs=[ps.m, N, MAX])
        wp.launch(kernel=zero_beyond_float, dim=MAX, inputs=[ps.m_inv, N, MAX])

        wp.copy(ps.x, ps.x0, count=N)
        wp.copy(ps.x_old, ps.x0, count=N)
        wp.copy(ps.v, ps.v0, count=N)

        # Zero out computed fields

        # Reset density to rest density
        wp.copy(ps.density, ps.density0)

        # Reset time
        self.time = 0.0

        # Reset viscosity solver internal state
        self.viscosity.reset()

        # Zero solver persistent buffers to avoid stale data after spawn slots are reused.
        # Some buffers are solver-specific (dt2_lam_df/dt2_lam_df_previous → PRJ only)
        # so only zero them when present.
        for _name in ("p", "p_previous", "dt2_lam_df", "dt2_lam_df_previous", "dx", "k"):
            _buf = getattr(self.pressure, _name, None)
            if _buf is not None:
                _buf.zero_()

        if self.scheduled_spawn_system is not None:
            self.scheduled_spawn_system.reset()

        # Re-initialize: elastic rest data (x == x0 here) + neighbor search, density, Aii.
        # Spawned deformables are gone with the particle count, so drop their registration.
        if self.elasticity is not None:
            self.elasticity.truncate(self._initial_deformable)
        self._rest_ready = False
        self.initialize()

    def _rebuild_neighbors_and_density(self):
        """Neighbor grid + cache + density + Aii at the current x."""
        self.ns.build(self.ps.x, self.ps.particle_num)
        self.ns.build_cache(
            self.ps.x, self.ps.particle_num, int(self.ps.cache_size),
            self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW,
        )
        self.pressure.compute_density()
        self.pressure.compute_Aii()

    def _init_rest_state(self):
        """One-shot elastic rest-config init (rest neighbours, V0, L, ...). Needs x == x0."""
        ps = self.ps
        el = self.elasticity
        sel = el.ids_np[:el.count]
        dev = float(np.abs(ps.x.numpy()[sel] - ps.x0.numpy()[sel]).max()) if el.count > 0 else 0.0
        if dev > 1e-6:
            raise RuntimeError(
                f"[ElasticityKee2023] rest init needs x == x0 on the deformable set "
                f"(max |x - x0| = {dev:.3e}); call Framework.reset() first"
            )
        self.ns.build(ps.x, ps.particle_num)
        self.ns.build_cache(
            ps.x, ps.particle_num, int(ps.cache_size),
            ps.neighbor_count, ps.neighbor_ids, ps.neighbor_gradW,
        )
        self.elasticity.initialize()
        err = self.elasticity.validate_F_at_rest()
        print(f"[ElasticityKee2023] rest principal strain = {err:.3e}")
        if err > 1e-3:
            print("[ElasticityKee2023] WARNING: rest state is not strain-free: rest lists, L and "
                  "the neighbour cache disagree (was x == x0 when the cache was built?)")
        # Loader-staged initial perturbation goes on AFTER the rest init; x0 stays clean.
        N = int(ps.particle_num)
        wp.launch(
            kernel=_apply_perturbation_kernel,
            dim=N,
            inputs=[ps.x_perturbation, N],
            outputs=[ps.x, ps.x_old],
        )

    def initialize(self):
        """Initialize simulation state. Elastic rest data is built only on the first
        call after construction / reset(); later calls are a pure neighbour/density
        rebuild at the current x."""
        if self.elasticity is not None and not self._rest_ready:
            self._init_rest_state()
            self._rest_ready = True
        self._rebuild_neighbors_and_density()

    def register_pending_deformables(self):
        """Register deformable bodies spawned since the last call and build their rest
        data from the current neighbour cache. The bodies must still sit at their rest
        shape (x == x0), so call this right after the spawn and the neighbour rebuild."""
        if self.elasticity is None or self.scheduled_spawn_system is None:
            return
        pending = self.scheduled_spawn_system.pop_pending_deformable()
        if not pending:
            return
        ps = self.ps
        x_np = ps.x.numpy()
        x0_np = ps.x0.numpy()
        n_particles, max_nb, n_deficient, max_err = 0, 0, 0, 0.0
        for start, n, elastic_params in pending:
            dev = float(np.abs(x_np[start:start + n] - x0_np[start:start + n]).max())
            if dev > 1e-6:
                raise RuntimeError(
                    f"[ElasticityKee2023] spawned body [{start}, {start + n}) is no longer at rest "
                    f"(max |x - x0| = {dev:.3e}); register it before moving it"
                )
            k0, k1 = self.elasticity.register(start, n, elastic_params)
            st = self.elasticity.initialize_range(k0, k1)
            n_particles += st["count"]
            max_nb = max(max_nb, st["max_rest_neighbors"])
            n_deficient += st["rank_deficient"]
            max_err = max(max_err, self.elasticity.validate_F_at_rest(k0, k1))
        print(f"[ElasticityKee2023] spawned {len(pending)} bodies @ t={self.time:.3f}s: "
              f"{n_particles} particles, max rest neighbors {max_nb}, rank-deficient "
              f"{n_deficient}, rest principal strain <= {max_err:.3e} "
              f"(total deformables {self.elasticity.count})")

    def _clear_pressure_stats(self):
        """Zero the solver's per-step stats when the pressure solve is skipped."""
        s = self.pressure
        s.stats_outer_iters = 0
        s.stats_density_errors = []
        s.stats_pcg_iters = []
        s.stats_iter_costs = []
        s.stats_total_cost_ms = 0.0
        s.stats_final_density_error = 0.0
        s.stats_pos_residuals = []
        s.stats_final_pos_residual = 0.0
        if hasattr(s, "stats_df_outer_iters"):
            s.stats_df_outer_iters = 0
            s.stats_df_total_cost_ms = 0.0

    def enforce_boundary(self):
        """Enforce domain boundary constraints."""
        domain_size = wp.vec3(
            self.ps.domain_size[0],
            self.ps.domain_size[1],
            self.ps.domain_size[2]
        )

        wp.launch(
            kernel=enforce_boundary_kernel,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.x,
                self.ps.v,
                self.ps.is_dynamic,
                self.ps.particle_num,
                domain_size,
                self.ps.padding
            ],
            device=self.device
        )

    def compute_non_pressure_forces(self):
        """Apply gravity and other non-pressure forces."""
        # Apply gravity
        wp.launch(
            kernel=apply_gravity,
            dim=self.ps.particle_num,
            inputs=[
                self.ps.v,
                self.ps.is_dynamic,
                self.ps.is_kinematic,
                self.ps.particle_num,
                self.g,
                self.dt
            ],
            device=self.device
        )

        # Apply surface tension (cohesion + curvature; adhesion uses the same coefficient)
        self.surface_tension.solve(self.surface_tension_coeff, self.surface_tension_coeff, self.dt)

        # Apply viscosity
        self.viscosity.solve(self.viscosity_coeff, self.dt)

    def forward(self):
        """Execute one simulation time step."""

        # CFL adaptive time step: dt = cfl_factor * h / v_max
        if self.use_cfl:
            self.v_max_sq_buffer.zero_()
            wp.launch(kernel=compute_max_velocity_sq, dim=self.ps.particle_num,
                      inputs=[self.ps.v, self.ps.is_dynamic, self.ps.particle_num],
                      outputs=[self.v_max_sq_buffer])
            wp.copy(self.v_max_sq_host, self.v_max_sq_buffer)
            v_max_sq = float(self.v_max_sq_host.numpy()[0])
            if math.isfinite(v_max_sq) and v_max_sq > 1e-16:
                v_max = v_max_sq ** 0.5
                self.dt = min(self.dt_default, self.cfl_factor * self.ps.support_radius / v_max)
            else:
                self.dt = self.dt_default
            # Floor dt to prevent stalls at extreme velocities
            self.dt = max(self.dt, self.cfl_min_dt)

        # Apply animation before physics (sets is_kinematic, updates positions/velocities)
        if self.animation is not None:
            self.animation.apply(self.time, self.dt)

        # Non-pressure forces; sync so the stage timings below exclude them
        self.compute_non_pressure_forces()
        wp.synchronize()

        # Elasticity: velocity-based Newton + matrix-free PCG, v += dv.
        # Operator-split before the pressure stage, so position-based solvers
        # advect with the elastic velocity and then project density on x.
        if self.elasticity is not None and self.elasticity_enabled:
            t0 = time.perf_counter()
            self.elasticity.solve(self.dt)
            wp.synchronize()
            self.stats_elasticity_ms = (time.perf_counter() - t0) * 1000.0
        else:
            self.stats_elasticity_ms = 0.0

        # Position-based solvers (PBF, SISPH) work on
        # advected positions: advect + rebuild neighbors BEFORE solve().
        # Velocity-based solvers (PRJ, IP) advect AFTER solve().
        # Dispatch by the `is_position_based` class attribute on the solver.
        is_position_based = getattr(self.pressure, "is_position_based", False)
        if is_position_based:
            wp.launch(
                kernel=advect_position,
                dim=self.ps.particle_num,
                inputs=[self.ps.x, self.ps.x_old, self.ps.v,
                        self.ps.is_dynamic, self.ps.particle_num, self.dt],
                device=self.device
            )
            self.enforce_boundary()
            self.ns.build(self.ps.x, self.ps.particle_num)
            self.ns.build_cache(
                self.ps.x, self.ps.particle_num, int(self.ps.cache_size),
                self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW,
            )

        t0 = time.perf_counter()
        if self.pressure_enabled:
            self.pressure.solve(self.dt)

            # DF solve (PRJ only; after CD solve, before advection)
            if getattr(self.pressure, "enable_df", False):
                self.pressure.solve_df(self.dt)
        else:
            # Pure "skip projection": position-based solvers still get v from the
            # advected positions (as PBF.solve would), and stale stats are cleared.
            if is_position_based:
                self.pressure.update_velocity_from_positions(self.dt)
            self._clear_pressure_stats()

        wp.synchronize()
        self.stats_pressure_ms = (time.perf_counter() - t0) * 1000.0

        # Advect + enforce boundary (skipped for position-based, already done above)
        if not is_position_based:
            wp.launch(
                kernel=advect_position,
                dim=self.ps.particle_num,
                inputs=[
                    self.ps.x,
                    self.ps.x_old,
                    self.ps.v,
                    self.ps.is_dynamic,
                    self.ps.particle_num,
                    self.dt
                ],
                device=self.device
            )
            self.enforce_boundary()

        # fixedAABB anchors: undo any pressure/boundary displacement (x_old = pre-step x).
        if self._has_fixed:
            wp.launch(
                kernel=_pin_fixed_kernel,
                dim=int(self.elasticity.count),
                inputs=[self.elasticity.ids, self.ps.is_fixed, self.ps.x_old],
                outputs=[self.ps.x, self.ps.v],
            )

        # Clear kinematic flags after physics (re-enable dynamics for next frame)
        if self.animation is not None:
            self.animation.clear_kinematic()

        # Spawn bodies (before neighbor rebuild so new particles are included)
        if self.scheduled_spawn_system is not None:
            self.scheduled_spawn_system.step(self.time)

        # Neighbourhood search, neighbour cache, density and Aii
        self.ns.build(self.ps.x, self.ps.particle_num)
        self.ns.build_cache(
            self.ps.x, self.ps.particle_num, int(self.ps.cache_size),
            self.ps.neighbor_count, self.ps.neighbor_ids, self.ps.neighbor_gradW,
        )
        # Deformable bodies spawned this step sit at their rest shape: register them
        # from the cache that was just rebuilt.
        self.register_pending_deformables()
        self.pressure.compute_density()
        self.pressure.compute_Aii()

        self.time += self.dt
