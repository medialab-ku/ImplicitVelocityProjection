"""
ImGui-based GUI manager for WarpSPH.
Handles stats display and solver settings UI.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from enum import IntEnum
import imgui
from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer

if TYPE_CHECKING:
    import warp.render


class SolverType(IntEnum):
    """Pressure solver types, in GUI combo order (indices into SOLVER_NAMES)."""
    PRJ = 0
    IP = 1
    PBF = 2
    SISPH = 3


SOLVER_NAMES = ["PRJ (IISPH, DFSPH)", "IP", "PBF", "SISPH"]

class ViscosityType(IntEnum):
    """Available viscosity solver types."""
    EXPLICIT = 0
    IMPLICIT = 1


VISCOSITY_NAMES = ["Explicit", "Implicit (Weiler18)"]


class SurfaceTensionType(IntEnum):
    """Available surface tension solver types."""
    AKINCI13 = 0
    JESKE23 = 1


SURFACE_TENSION_NAMES = ["Akinci13", "Implicit (Jeske23)"]

# Poisson's ratio range for the elasticity panel (lambda is singular at 0.5).
NU_MIN, NU_MAX = -0.49, 0.499


@dataclass
class StatsState:
    """Container for simulation statistics."""
    frame: int = 0
    sim_time: float = 0.0
    is_running: bool = False


@dataclass
class SolverSettings:
    """Container for solver settings that can be modified via UI."""
    # Solver type
    solver_type: int = SolverType.PRJ
    viscosity_type: int = ViscosityType.EXPLICIT
    viscosity_coeff: float = 0.01
    visc_tol: int = 5
    visc_use_preconditioner: bool = True
    visc_warm_start: bool = False
    visc_max_cg_iterations: int = 1000
    # Surface tension
    surface_tension_type: int = SurfaceTensionType.AKINCI13
    surface_tension_coeff: float = 0.01
    st_tol: int = 5              # -log10(tol), for Jeske23
    st_max_cg_iterations: int = 100
    # Time step
    dt: float = 0.002
    use_cfl: bool = False
    cfl_factor: float = 0.4


    # Divergence-free solve (PRJ only)
    enable_df: bool = False
    df_num_threshold: int = 25
    df_tol_opt: float = 1e-2

    # Common solver settings (density)
    tol_opt: float = 1e-3
    max_iteration_opt: int = 1000
    omega: float = 0.5

    # PRJ-specific
    warm_start: bool = False

    fixed_outer_iters: bool = False  # every solver: run max_iteration_opt, ignore convergence
    # Position-based solvers: outer stopping criterion (0 = max dx, 1 = avg density err)
    convergence_mode: int = 1
    pos_tol: float = 1e-3            # max-dx tol, x support radius

    # SISPH-specific
    stiffness: float = 1000000.0       # mu, raw penalty stiffness (Eq. 7)
    sisph_energy_type: int = 0         # bulk energy: 0=B0 quadratic, 1=B1 cubic, 2=B2 quartic (Sec. 4.3)
    sisph_conserve_momentum: bool = True   # momentum conservation (Eq. 14; off oscillates, Fig. 4)
    sisph_line_search: bool = False    # step length alpha^k (Eqs. 19-20; B0 needs none)
    sisph_accel_type: int = 0          # acceleration: 0=None, 1=Chebyshev, 2=Anderson (Sec. 4.3)
    sisph_cheby_rho: float = 0.9       # Chebyshev spectral radius rho (paper 0.9)
    sisph_cheby_stabilize: bool = True # master toggle for Wang [Wan15] stabilization
    sisph_cheby_delay: int = 10        # Wang [Wan15] delay S: no accel for first S iters
    sisph_cheby_gamma: float = 0.9     # Wang [Wan15] under-relaxation gamma in [0.6, 0.9]

    # IP-specific
    tol_pcg: float = 1e-6   # inner CG tol (||r||^2 < tol_pcg)
    max_iteration_pcg: int = 1000

    # Elasticity (Kee 2023, velocity-based Newton + matrix-free PCG)
    elasticity_enabled: bool = True
    pressure_enabled: bool = True
    elasticity_material: int = 0      # 0 = Co-rotated, 1 = Stable NH
    elasticity_E: float = 5.0e5       # Young's modulus (Pa)
    elasticity_nu: float = 0.45       # Poisson's ratio
    elasticity_alpha_ze: float = 0.0  # zero-energy-mode penalty (0 = off)
    elasticity_max_cg: int = 100
    elasticity_tol_cg: float = 1e-4


class GUI:
    """ImGui-based GUI manager for simulation stats and solver settings."""

    def __init__(
        self,
        renderer: "warp.render.OpenGLRenderer | None" = None,
        framework=None,
        visualizer=None,
        title: str = "Stats",
        window_pos: tuple[int, int] = (10, 10),
    ):
        """
        Args:
            renderer: Warp OpenGLRenderer to draw the panel on
            framework: Framework whose solver settings the panel edits
            visualizer: Visualizer whose display settings the panel edits
            title: Panel window title
            window_pos: (x, y) initial panel position
        """
        self.title = title
        self.window_pos = window_pos
        self.state = StatsState()
        self.solver_settings = SolverSettings()
        # Main-panel page selector: 0=solver, 1=visualization.
        self.page = 0
        self.is_available = False
        self.renderer = None
        self.framework = framework
        self.visualizer = visualizer

        # FPS tracking
        self.fps = 0.0
        self.fps_frames = 0
        self.fps_last = time.perf_counter()

        # Deferred initialization - will be done when renderer is attached
        self.imgui = None
        self.impl = None

        # Initialize solver settings from framework if provided
        if framework is not None:
            self.sync_settings_from_framework()

        if renderer is not None:
            self.attach(renderer)

    def attach(self, renderer: "warp.render.OpenGLRenderer"):
        """
        Args:
            renderer: Warp OpenGLRenderer instance
        """
        if self.is_available:
            return  # Already attached

        self.renderer = renderer
        self.imgui = imgui

        # Create ImGui context
        self.imgui.create_context()

        # Create ImGui renderer using the Warp renderer's pyglet window
        self.impl = PygletProgrammablePipelineRenderer(self.renderer.window)

        # Register our render callback with Warp's 2D callback system
        self.renderer.render_2d_callbacks.append(self.render_callback)

        self.is_available = True

    def update(self, *, frame: int, sim_time: float, is_running: bool):
        """
        Update simulation statistics.

        Args:
            frame: Current frame number
            sim_time: Current simulation time in seconds
            is_running: Whether the simulation is currently running
        """
        self.state.frame = int(frame)
        self.state.sim_time = float(sim_time)
        self.state.is_running = bool(is_running)

    def tick_fps(self):
        """Update FPS calculation."""
        self.fps_frames += 1
        now = time.perf_counter()
        dt = now - self.fps_last
        if dt >= 0.5:  # Update every 0.5 seconds
            self.fps = self.fps_frames / max(dt, 1e-9)
            self.fps_frames = 0
            self.fps_last = now

    def render_callback(self):
        """
        ImGui render callback.
        This is called automatically by the Warp renderer during end_frame().
        """
        if not self.is_available or self.renderer is None:
            return

        self.tick_fps()

        # Get OpenGL module from the renderer
        gl = self.renderer.gl

        # ImGui needs the window size in logical points and the framebuffer/point
        # ratio separately, so layout and click targets stay right on HiDPI monitors.
        io = self.imgui.get_io()
        window = self.renderer.window
        win_w, win_h = window.get_size()
        fb_w, fb_h = window.get_framebuffer_size()
        io.display_size = float(win_w), float(win_h)
        io.display_fb_scale = (
            (fb_w / win_w) if win_w > 0 else 1.0,
            (fb_h / win_h) if win_h > 0 else 1.0,
        )

        # Disable Warp's mouse/keyboard when ImGui wants input
        self.renderer.enable_mouse_interaction = not io.want_capture_mouse
        self.renderer.enable_keyboard_interaction = not io.want_capture_keyboard

        self.imgui.new_frame()
        self.draw_main_panel()
        self.imgui.render()

        # Save relevant OpenGL state
        blend_enabled = gl.glIsEnabled(gl.GL_BLEND)
        depth_test_enabled = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        cull_face_enabled = gl.glIsEnabled(gl.GL_CULL_FACE)

        # Set up blending for ImGui rendering
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)

        # Render ImGui draw data
        self.impl.render(self.imgui.get_draw_data())

        # Restore OpenGL state
        if blend_enabled:
            gl.glEnable(gl.GL_BLEND)
        else:
            gl.glDisable(gl.GL_BLEND)

        if depth_test_enabled:
            gl.glEnable(gl.GL_DEPTH_TEST)
        else:
            gl.glDisable(gl.GL_DEPTH_TEST)

        if cull_face_enabled:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

    def _draw_stats_block(self):
        """Stats at the top of the panel: FPS, particle counts, frame, sim time."""
        fps_color = (0.2, 1.0, 0.2, 1.0) if self.fps >= 30 else (1.0, 0.5, 0.2, 1.0)
        self.imgui.text_colored(f"FPS: {self.fps:6.1f}", *fps_color)
        ps = self.framework.ps
        boundary = ps.rigid_particle_num
        fluid = ps.particle_num - boundary
        spawns = self.framework.scheduled_spawn_system
        rigid_pending = (spawns.rigid_capacity - spawns.rigid_spawned) if spawns is not None else 0
        fluid_max = ps.particle_max_num - boundary - rigid_pending
        self.imgui.text(f"Fluid:    {fluid:>7d} / {fluid_max:>7d}")
        self.imgui.text(f"Boundary: {boundary:>7d}")
        self.imgui.text(f"Frame: {self.state.frame:>8d}")

        self.imgui.separator()

        if self.state.is_running:
            self.imgui.text_colored("Sim Time:", 0.4, 0.8, 1.0, 1.0)
            self.imgui.same_line()
            self.imgui.text_colored(f"{self.state.sim_time:8.4f} s", 0.4, 1.0, 0.4, 1.0)
            self.imgui.text_colored("RUNNING", 0.2, 1.0, 0.2, 1.0)
        else:
            self.imgui.text(f"Sim Time: {self.state.sim_time:8.4f} s")
            self.imgui.text_colored("PAUSED", 1.0, 0.8, 0.2, 1.0)

    def get_current_solver_type(self) -> int:
        """Determine current solver type from framework."""
        if self.framework is None:
            return SolverType.PRJ

        solver = self.framework.pressure
        class_name = type(solver).__name__

        if class_name == "IP":
            return SolverType.IP
        elif class_name == "PBF":
            return SolverType.PBF
        elif class_name == "SISPH":
            return SolverType.SISPH
        return SolverType.PRJ

    def sync_settings_from_framework(self):
        """Sync solver settings from framework to UI state."""
        if self.framework is None:
            return

        fw = self.framework
        solver = fw.pressure
        settings = self.solver_settings

        # Solver type
        settings.solver_type = self.get_current_solver_type()
        settings.viscosity_type = self.get_current_viscosity_type()
        settings.surface_tension_type = self.get_current_surface_tension_type()
        settings.viscosity_coeff = fw.viscosity_coeff
        settings.surface_tension_coeff = fw.surface_tension_coeff
        if hasattr(fw.surface_tension, 'tol'):
            settings.st_tol = round(-math.log10(max(fw.surface_tension.tol, 1e-15)))
        if hasattr(fw.surface_tension, 'max_cg_iterations'):
            settings.st_max_cg_iterations = fw.surface_tension.max_cg_iterations
        if hasattr(fw.viscosity, 'tol'):
            settings.visc_tol = round(-math.log10(max(fw.viscosity.tol, 1e-15)))
        if hasattr(fw.viscosity, 'use_preconditioner'):
            settings.visc_use_preconditioner = fw.viscosity.use_preconditioner
        if hasattr(fw.viscosity, 'warm_start'):
            settings.visc_warm_start = fw.viscosity.warm_start
        if hasattr(fw.viscosity, 'max_cg_iterations'):
            settings.visc_max_cg_iterations = fw.viscosity.max_cg_iterations

        # Common settings
        settings.dt = fw.dt_default
        settings.use_cfl = fw.use_cfl
        settings.cfl_factor = fw.cfl_factor
        settings.tol_opt = solver.tol_opt
        settings.max_iteration_opt = solver.max_iteration_opt
        settings.fixed_outer_iters = solver.use_fixed_iters
        settings.omega = solver.omega

        # PRJ-specific
        if hasattr(solver, 'warm_start'):
            settings.warm_start = solver.warm_start
        if hasattr(solver, 'enable_df'):
            settings.enable_df = solver.enable_df
            settings.df_num_threshold = solver.df_num_threshold
            settings.df_tol_opt = solver.df_tol_opt

        # SISPH-specific
        if hasattr(solver, 'stiffness'):
            settings.stiffness = solver.stiffness
        if hasattr(solver, 'energy_type'):
            settings.sisph_energy_type = solver.energy_type
        if hasattr(solver, 'conserve_momentum'):
            settings.sisph_conserve_momentum = solver.conserve_momentum
        if hasattr(solver, 'use_line_search'):
            settings.sisph_line_search = solver.use_line_search
        if hasattr(solver, 'accel_type'):
            settings.sisph_accel_type = solver.accel_type
        if hasattr(solver, 'cheby_rho'):
            settings.sisph_cheby_rho = solver.cheby_rho
        if hasattr(solver, 'cheby_stabilize'):
            settings.sisph_cheby_stabilize = solver.cheby_stabilize
        if hasattr(solver, 'cheby_delay'):
            settings.sisph_cheby_delay = solver.cheby_delay
        if hasattr(solver, 'cheby_gamma'):
            settings.sisph_cheby_gamma = solver.cheby_gamma
        if hasattr(solver, 'convergence_mode'):
            settings.convergence_mode = solver.convergence_mode
            settings.pos_tol = solver.pos_tol
        if hasattr(solver, 'tol_pcg'):
            settings.tol_pcg = solver.tol_pcg
        if hasattr(solver, 'max_iteration_pcg'):
            settings.max_iteration_pcg = solver.max_iteration_pcg

        # Elasticity: recover (E, nu) from (mu, lambda).
        settings.pressure_enabled = bool(getattr(fw, "pressure_enabled", True))
        el = getattr(fw, "elasticity", None)
        if el is not None:
            settings.elasticity_enabled = bool(fw.elasticity_enabled)
            settings.elasticity_material = int(el.material_type)
            mu = float(el.mu)
            lam = float(el.lambda_)
            denom = 2.0 * (lam + mu)
            nu = lam / denom if denom > 0 else 0.0
            settings.elasticity_nu = nu
            settings.elasticity_E = 2.0 * mu * (1.0 + nu)
            settings.elasticity_alpha_ze = float(el.alpha_ze)
            settings.elasticity_max_cg = int(el.max_iter_cg)
            settings.elasticity_tol_cg = float(el.tol_cg)

    def sync_settings_to_framework(self):
        """Sync UI state to framework solver settings."""
        if self.framework is None:
            return

        fw = self.framework
        solver = fw.pressure
        settings = self.solver_settings

        # Common settings
        fw.dt_default = settings.dt
        fw.dt = settings.dt
        fw.use_cfl = settings.use_cfl
        fw.cfl_factor = settings.cfl_factor
        fw.viscosity_coeff = settings.viscosity_coeff
        fw.surface_tension_coeff = settings.surface_tension_coeff
        if hasattr(fw.surface_tension, 'tol'):
            fw.surface_tension.tol = 10.0 ** (-settings.st_tol)
        if hasattr(fw.surface_tension, 'max_cg_iterations'):
            fw.surface_tension.max_cg_iterations = settings.st_max_cg_iterations
        if hasattr(fw.viscosity, 'tol'):
            fw.viscosity.tol = 10.0 ** (-settings.visc_tol)
        if hasattr(fw.viscosity, 'use_preconditioner'):
            fw.viscosity.use_preconditioner = settings.visc_use_preconditioner
        if hasattr(fw.viscosity, 'warm_start'):
            fw.viscosity.warm_start = settings.visc_warm_start
        if hasattr(fw.viscosity, 'max_cg_iterations'):
            fw.viscosity.max_cg_iterations = settings.visc_max_cg_iterations
        solver.tol_opt = settings.tol_opt
        solver.max_iteration_opt = settings.max_iteration_opt
        solver.use_fixed_iters = settings.fixed_outer_iters
        solver.omega = settings.omega
        if hasattr(solver, 'enable_df'):  # PRJ only
            solver.enable_df = settings.enable_df
            solver.df_num_threshold = settings.df_num_threshold
            solver.df_tol_opt = settings.df_tol_opt

        # PRJ-specific
        if hasattr(solver, 'warm_start'):
            solver.warm_start = settings.warm_start

        # SISPH-specific
        if hasattr(solver, 'stiffness'):
            solver.stiffness = settings.stiffness
        if hasattr(solver, 'energy_type'):
            solver.energy_type = settings.sisph_energy_type
        if hasattr(solver, 'conserve_momentum'):
            solver.conserve_momentum = settings.sisph_conserve_momentum
        if hasattr(solver, 'use_line_search'):
            solver.use_line_search = settings.sisph_line_search
        if hasattr(solver, 'accel_type'):
            solver.accel_type = settings.sisph_accel_type
        if hasattr(solver, 'cheby_rho'):
            solver.cheby_rho = settings.sisph_cheby_rho
        if hasattr(solver, 'cheby_stabilize'):
            solver.cheby_stabilize = settings.sisph_cheby_stabilize
        if hasattr(solver, 'cheby_delay'):
            solver.cheby_delay = settings.sisph_cheby_delay
        if hasattr(solver, 'cheby_gamma'):
            solver.cheby_gamma = settings.sisph_cheby_gamma
        if hasattr(solver, 'convergence_mode'):
            solver.convergence_mode = settings.convergence_mode
            solver.pos_tol = settings.pos_tol
        if hasattr(solver, 'tol_pcg'):
            solver.tol_pcg = settings.tol_pcg
        if hasattr(solver, 'max_iteration_pcg'):
            solver.max_iteration_pcg = settings.max_iteration_pcg

        # Elasticity: mu, lambda derived from (E, nu); kernels read them at launch.
        fw.pressure_enabled = bool(settings.pressure_enabled)
        el = getattr(fw, "elasticity", None)
        if el is not None:
            fw.elasticity_enabled = bool(settings.elasticity_enabled)
            el.material_type = int(settings.elasticity_material)
            E = float(settings.elasticity_E)
            nu = max(NU_MIN, min(NU_MAX, float(settings.elasticity_nu)))
            new_mu = E / (2.0 * (1.0 + nu))
            new_lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            new_alpha_ze = float(settings.elasticity_alpha_ze)
            # Rewrite (mu, lambda) only when E/nu really changed, so an unrelated widget
            # cannot alter a scene value through the E/nu <-> mu/lambda round trip.
            def _same(a, b):
                return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1e-30)
            mat_dirty = not (_same(new_mu, el.mu) and _same(new_lam, el.lambda_))
            ze_dirty = mat_dirty or (new_alpha_ze != el.alpha_ze)  # K_ze caches (alpha, mu)
            if mat_dirty:
                el.set_default_material(new_mu, new_lam)
            el.alpha_ze = new_alpha_ze
            if ze_dirty:
                el.refresh_K_ze()
            el.max_iter_cg = int(settings.elasticity_max_cg)
            el.tol_cg = float(settings.elasticity_tol_cg)


    def switch_solver(self, new_type: int):
        """Switch to a different solver type."""
        if self.framework is None:
            return

        current_type = self.get_current_solver_type()
        if current_type == new_type:
            return

        from scripts.sim.forces.pressure.prj import PRJ
        from scripts.sim.forces.pressure.pbf import PBF
        from scripts.sim.forces.pressure.ip import IP
        from scripts.sim.forces.pressure.sisph import SISPH

        fw = self.framework
        ps = fw.ps
        ns = fw.ns
        cfg = fw.cfg
        old_solver = fw.pressure

        # Create new solver
        if new_type == SolverType.PRJ:
            fw.pressure = PRJ(ps, ns, cfg)
            print("Switched to PRJ solver")
        elif new_type == SolverType.PBF:
            fw.pressure = PBF(ps, ns, cfg)
            print("Switched to PBF solver")
        elif new_type == SolverType.IP:
            fw.pressure = IP(ps, ns, cfg)
            print("Switched to Implicit Projection solver")
        elif new_type == SolverType.SISPH:
            fw.pressure = SISPH(ps, ns, cfg)
            print("Switched to SISPH solver")

        # Re-initialize solver state
        fw.pressure.compute_density()
        fw.pressure.compute_Aii()

        # Shared settings carry over; solver-specific ones come from the new solver.
        # The max-dx criterion carries over only between position-based solvers.
        s = self.solver_settings
        s.solver_type = new_type
        shared = (s.tol_opt, s.max_iteration_opt, s.fixed_outer_iters)
        conv = (s.convergence_mode, s.pos_tol) if hasattr(old_solver, "convergence_mode") else None
        self.sync_settings_from_framework()
        s.tol_opt, s.max_iteration_opt, s.fixed_outer_iters = shared
        if conv is not None:
            s.convergence_mode, s.pos_tol = conv
        self.sync_settings_to_framework()

    def get_current_viscosity_type(self) -> int:
        """Determine current viscosity solver type from framework."""
        if self.framework is None:
            return ViscosityType.EXPLICIT
        class_name = type(self.framework.viscosity).__name__
        if class_name == "Weiler18":
            return ViscosityType.IMPLICIT
        return ViscosityType.EXPLICIT

    def switch_viscosity(self, new_type: int):
        """Switch viscosity solver at runtime."""
        if self.framework is None:
            return
        current_type = self.get_current_viscosity_type()
        if current_type == new_type:
            return

        from scripts.sim.forces.non_pressure.viscosity.viscosity import Viscosity
        from scripts.sim.forces.non_pressure.viscosity.weiler18 import Weiler18

        fw = self.framework
        if new_type == ViscosityType.EXPLICIT:
            fw.viscosity = Viscosity(fw.ps, fw.ns, fw.cfg)
            print("Switched to explicit viscosity")
        elif new_type == ViscosityType.IMPLICIT:
            fw.viscosity = Weiler18(fw.ps, fw.ns, fw.cfg)
            print("Switched to implicit viscosity (Weiler18)")

        self.solver_settings.viscosity_type = new_type
        self.sync_settings_from_framework()
        self.sync_settings_to_framework()

    def get_current_surface_tension_type(self) -> int:
        """Determine current surface tension solver type from framework."""
        if self.framework is None:
            return SurfaceTensionType.AKINCI13
        class_name = type(self.framework.surface_tension).__name__
        if class_name == "Jeske23":
            return SurfaceTensionType.JESKE23
        return SurfaceTensionType.AKINCI13

    def switch_surface_tension(self, new_type: int):
        """Switch surface tension solver at runtime."""
        if self.framework is None:
            return
        current_type = self.get_current_surface_tension_type()
        if current_type == new_type:
            return

        from scripts.sim.forces.non_pressure.surface_tension.surface_tension import SurfaceTension
        from scripts.sim.forces.non_pressure.surface_tension.jeske23 import Jeske23

        fw = self.framework
        if new_type == SurfaceTensionType.AKINCI13:
            fw.surface_tension = SurfaceTension(fw.ps, fw.ns, fw.cfg)
            print("Switched to Akinci13 surface tension")
        elif new_type == SurfaceTensionType.JESKE23:
            fw.surface_tension = Jeske23(fw.ps, fw.ns, fw.cfg)
            print("Switched to implicit surface tension (Jeske23)")

        self.solver_settings.surface_tension_type = new_type
        # Push current settings to the new instance
        self.sync_settings_from_framework()
        self.sync_settings_to_framework()

    _PAGE_NAMES = ("solver settings", "visualization")

    def draw_main_panel(self):
        """Single ImGui window: stats + page slider + selected page body."""
        if self.framework is None:
            return

        # Top-left placement.
        self.imgui.set_next_window_position(
            self.window_pos[0], self.window_pos[1], self.imgui.ONCE
        )
        flags = (
            self.imgui.WINDOW_NO_SAVED_SETTINGS
            | self.imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )
        expanded, _ = self.imgui.begin(self.title, flags=flags)

        if expanded:
            self._draw_stats_block()
            self.imgui.separator()

            c, new_page = self.imgui.slider_int("page", int(self.page), 0, len(self._PAGE_NAMES) - 1)
            if c:
                self.page = int(new_page)
            self.imgui.text(f"  -> {self._PAGE_NAMES[self.page]}")
            self.imgui.separator()

            if self.page == 0:
                self._draw_solver_settings_block()
            elif self.page == 1:
                self._draw_visualization_block()

        self.imgui.end()

    def _tol_pcg_slider(self, settings):
        """IP's CG tolerance (absolute ||r||^2) as a decade slider, tol_pcg = 10^-n.
        Returns True if the user changed it."""
        exp = 6
        if settings.tol_pcg > 0.0:
            exp = int(round(-math.log10(settings.tol_pcg)))
        exp = min(max(exp, 1), 12)
        c, exp_new = self.imgui.slider_int("CG tol", exp, 1, 12, "1e-%d")
        if c:
            settings.tol_pcg = 10.0 ** (-exp_new)
        return c

    def _draw_solver_settings_block(self):
        """Solver settings page."""
        if self.framework is None:
            return
        settings = self.solver_settings
        changed = False

        # Time step slider
        c, settings.dt = self.imgui.slider_float(
            "dt", settings.dt, 0.001, 0.03, "%.4f"
        )
        changed = changed or c
        c, settings.use_cfl = self.imgui.checkbox(
            "CFL adaptive dt", settings.use_cfl
        )
        changed = changed or c
        if settings.use_cfl:
            c, settings.cfl_factor = self.imgui.slider_float(
                "CFL factor", settings.cfl_factor, 0.1, 1.0, "%.2f"
            )
            changed = changed or c
            self.imgui.text(f"  dt = {self.framework.dt:.6f}")

        # --- Density Solver group ---
        self.imgui.separator()
        self.imgui.text("Constant Density Solver")
        solver = self.framework.pressure
        dens_err = getattr(solver, 'stats_final_density_error', 0.0)
        if hasattr(solver, 'convergence_mode'):
            # Position-based solvers stop on max dx (position-native) or on the
            # avg density error (the velocity-based solvers' criterion).
            c, settings.convergence_mode = self.imgui.combo(
                "convergence criterion", settings.convergence_mode,
                ["max dx", "avg density err"]
            )
            changed = changed or c
            pos_res = getattr(solver, 'stats_final_pos_residual', 0.0)
            if settings.convergence_mode == 1:
                c, settings.tol_opt = self.imgui.input_float(
                    "density error tol", settings.tol_opt, 0.0001, 0.1, "%.6f"
                )
                changed = changed or c
                self.imgui.text(f"  actual (avg density err): {dens_err:.6e}")
                self.imgui.text(f"  max dx / h: {pos_res:.3e}")
            else:
                c, settings.pos_tol = self.imgui.input_float(
                    "position tol (x h)", settings.pos_tol, 1e-5, 1e-2, "%.1e"
                )
                changed = changed or c
                self.imgui.text(f"  actual (max dx / h): {pos_res:.3e}")
                self.imgui.text(f"  avg density err: {dens_err:.3e}")
        else:
            c, settings.tol_opt = self.imgui.input_float(
                "density error tol", settings.tol_opt, 0.0001, 0.1, "%.6f"
            )
            changed = changed or c
            self.imgui.text(f"  actual: {dens_err:.6e}")

        c, settings.max_iteration_opt = self.imgui.input_int(
            "max outer iter", settings.max_iteration_opt
        )
        settings.max_iteration_opt = max(1, settings.max_iteration_opt)
        changed = changed or c
        # Ignore the convergence criterion and run exactly "max outer iter" updates
        # (PRJ's DF pass too).
        c, settings.fixed_outer_iters = self.imgui.checkbox(
            "fixed outer iters (ignore convergence)", settings.fixed_outer_iters
        )
        changed = changed or c
        outer_iters = getattr(self.framework.pressure, 'stats_outer_iters', 0)
        self.imgui.text(f"  outer iters: {outer_iters}")
        pressure_ms = getattr(self.framework, 'stats_pressure_ms', 0.0)
        self.imgui.text(f"  elapsed: {pressure_ms:.3f} ms")


        c, new_solver_type = self.imgui.combo(
            "##solver_type", settings.solver_type, SOLVER_NAMES
        )
        if c and new_solver_type != settings.solver_type:
            self.switch_solver(new_solver_type)

        # Divergence-free pass (DFSPH): PRJ only
        if settings.solver_type == SolverType.PRJ:
            c, settings.enable_df = self.imgui.checkbox(
                "Divergence-Free Solve", settings.enable_df
            )
            changed = changed or c
            if settings.enable_df:
                c, settings.df_num_threshold = self.imgui.slider_int(
                    "DF #neighbor threshold", settings.df_num_threshold, 0, 80
                )
                changed = changed or c
                c, settings.df_tol_opt = self.imgui.input_float(
                    "DF error tol", settings.df_tol_opt, 0.0001, 0.1, "%.6f"
                )
                changed = changed or c
                df_iters = getattr(self.framework.pressure, 'stats_df_outer_iters', 0)
                self.imgui.text(f"  DF iters: {df_iters}")

        # PRJ-specific settings
        if settings.solver_type == SolverType.PRJ:
            c, settings.omega = self.imgui.input_float(
                "omega", settings.omega, 0.1, 1.0, "%.2f"
            )
            changed = changed or c
            c, settings.warm_start = self.imgui.checkbox(
                "warm start", settings.warm_start
            )
            changed = changed or c

        # PBF-specific settings
        elif settings.solver_type == SolverType.PBF:
            c, settings.omega = self.imgui.input_float(
                "omega", settings.omega, 0.1, 1.0, "%.2f"
            )
            changed = changed or c

        # IP-specific settings
        elif settings.solver_type == SolverType.IP:
            c, settings.omega = self.imgui.input_float(
                "omega", settings.omega, 0.1, 1000.0, "%.2f"
            )
            changed = changed or c

            c = self._tol_pcg_slider(settings)
            changed = changed or c

            c, settings.max_iteration_pcg = self.imgui.slider_int(
                "max CG iter", settings.max_iteration_pcg, 1, 1000
            )
            changed = changed or c
            pcg_iters = list(getattr(self.framework.pressure, 'stats_pcg_iters', []))
            total_pcg = int(sum(pcg_iters))
            avg_pcg = (total_pcg / max(len(pcg_iters), 1)) if pcg_iters else 0.0
            self.imgui.text(f"  total: {total_pcg}    avg/outer: {avg_pcg:.2f}")

        # SISPH-specific settings
        elif settings.solver_type == SolverType.SISPH:
            # mu: bulk-energy weight (Eq. 7); larger -> stiffer, more incompressible.
            c, settings.stiffness = self.imgui.input_float(
                "stiffness (mu)", settings.stiffness, 100.0, 100000.0, "%.1f"
            )
            changed = changed or c

            # Bulk energy B0/B1/B2 (Sec. 4.3); B0 (default) needs no line search.
            c, settings.sisph_energy_type = self.imgui.combo(
                "bulk energy", settings.sisph_energy_type,
                ["B0 (quadratic)", "B1 (cubic)", "B2 (quartic)"]
            )
            changed = changed or c

            # Momentum conservation (Eq. 14) adds the transpose A_ji terms; off -> Eq. 12.
            c, settings.sisph_conserve_momentum = self.imgui.checkbox(
                "conserve momentum (Eq. 14)", settings.sisph_conserve_momentum
            )
            changed = changed or c

            # Step-length line search alpha^k (Eqs. 19-20), for B1/B2.
            c, settings.sisph_line_search = self.imgui.checkbox(
                "line search (alpha^k)", settings.sisph_line_search
            )
            changed = changed or c

            # Convergence acceleration (Sec. 4.3): Chebyshev or Anderson (n = 1).
            c, settings.sisph_accel_type = self.imgui.combo(
                "acceleration", settings.sisph_accel_type,
                ["None", "Chebyshev", "Anderson"]
            )
            changed = changed or c
            if settings.sisph_accel_type == 1:
                c, settings.sisph_cheby_rho = self.imgui.slider_float(
                    "Chebyshev rho", settings.sisph_cheby_rho, 0.0, 0.999, "%.3f"
                )
                changed = changed or c
                # Wang [Wan15] stabilization: delay S + under-relaxation gamma; off -> plain Eq. 22.
                c, settings.sisph_cheby_stabilize = self.imgui.checkbox(
                    "Wang stabilization (delay + under-relax)", settings.sisph_cheby_stabilize
                )
                changed = changed or c
                if settings.sisph_cheby_stabilize:
                    c, settings.sisph_cheby_delay = self.imgui.input_int(
                        "Chebyshev delay S", settings.sisph_cheby_delay
                    )
                    changed = changed or c
                    c, settings.sisph_cheby_gamma = self.imgui.slider_float(
                        "Chebyshev gamma (under-relax)", settings.sisph_cheby_gamma, 0.5, 1.0, "%.2f"
                    )
                    changed = changed or c
                else:
                    self.imgui.text("  S=0, gamma=1.0 (plain Eq. 22)")

        # --- Viscosity Solver group ---
        self.imgui.separator()
        self.imgui.text("Viscosity Solver")
        c, new_visc_type = self.imgui.combo(
            "##viscosity_type", settings.viscosity_type, VISCOSITY_NAMES
        )
        if c and new_visc_type != settings.viscosity_type:
            self.switch_viscosity(new_visc_type)

        c, settings.viscosity_coeff = self.imgui.input_float(
            "viscosity", settings.viscosity_coeff, 0.0, 0.0, "%.8f"
        )
        changed = changed or c

        if settings.viscosity_type == ViscosityType.IMPLICIT:
            c, settings.visc_tol = self.imgui.slider_int(
                "visc CG tol", settings.visc_tol, 1, 7
            )
            changed = changed or c

            c, settings.visc_max_cg_iterations = self.imgui.slider_int(
                "visc CG max iter", settings.visc_max_cg_iterations, 1, 1000
            )
            changed = changed or c

            c, settings.visc_use_preconditioner = self.imgui.checkbox(
                "block Jacobi precond", settings.visc_use_preconditioner
            )
            changed = changed or c

            c, settings.visc_warm_start = self.imgui.checkbox(
                "warm start##visc", settings.visc_warm_start
            )
            changed = changed or c

        # --- Surface Tension group ---
        self.imgui.separator()
        self.imgui.text("Surface Tension")
        c, new_st_type = self.imgui.combo(
            "##st_type", settings.surface_tension_type, SURFACE_TENSION_NAMES
        )
        if c and new_st_type != settings.surface_tension_type:
            self.switch_surface_tension(new_st_type)

        c, settings.surface_tension_coeff = self.imgui.input_float(
            "surf. tension", settings.surface_tension_coeff, 0.0, 0.0, "%.8f"
        )
        changed = changed or c

        if settings.surface_tension_type == SurfaceTensionType.JESKE23:
            c, settings.st_tol = self.imgui.slider_int(
                "ST CG tol", settings.st_tol, 1, 7
            )
            changed = changed or c

            c, settings.st_max_cg_iterations = self.imgui.slider_int(
                "ST CG max iter", settings.st_max_cg_iterations, 1, 1000
            )
            changed = changed or c

        # --- Elasticity (Kee 2023) group: only when the scene has deformable bodies ---
        el = getattr(self.framework, "elasticity", None)
        if el is not None:
            self.imgui.separator()
            self.imgui.text("Elasticity (Kee2023)")

            c, settings.elasticity_enabled = self.imgui.checkbox(
                "enable elasticity", settings.elasticity_enabled
            )
            changed = changed or c

            c, settings.pressure_enabled = self.imgui.checkbox(
                "enable pressure solve", settings.pressure_enabled
            )
            changed = changed or c

            c, settings.elasticity_material = self.imgui.combo(
                "material", settings.elasticity_material, ["Co-rotated", "Stable NH"]
            )
            changed = changed or c

            c, settings.elasticity_E = self.imgui.input_float(
                "default E (Pa)", settings.elasticity_E, 1.0e3, 1.0e6, "%.3e"
            )
            changed = changed or c

            c, settings.elasticity_nu = self.imgui.slider_float(
                "default nu", settings.elasticity_nu, NU_MIN, NU_MAX, "%.3f"
            )
            changed = changed or c

            nu_clamp = max(NU_MIN, min(NU_MAX, settings.elasticity_nu))
            mu_disp = settings.elasticity_E / (2.0 * (1.0 + nu_clamp))
            lam_disp = settings.elasticity_E * nu_clamp / ((1.0 + nu_clamp) * (1.0 - 2.0 * nu_clamp))
            self.imgui.text(f"  mu = {mu_disp:.3e}    lambda = {lam_disp:.3e}")

            c, settings.elasticity_alpha_ze = self.imgui.input_float(
                "ZE alpha (hourglass)", settings.elasticity_alpha_ze, 0.0, 1.0, "%.4f"
            )
            changed = changed or c

            c, settings.elasticity_tol_cg = self.imgui.input_float(
                "elast PCG tol", settings.elasticity_tol_cg, 1e-5, 1e-2, "%.1e"
            )
            changed = changed or c

            c, settings.elasticity_max_cg = self.imgui.slider_int(
                "elast PCG max iter", settings.elasticity_max_cg, 1, 1000
            )
            changed = changed or c

            cg_iters = int(getattr(el, "last_cg_iters", 0))
            cg_residual = float(getattr(el, "last_cg_residual", 0.0))
            strain = float(getattr(el, "last_max_strain", 0.0))
            elast_ms = float(getattr(self.framework, "stats_elasticity_ms", 0.0))
            self.imgui.text(
                f"  CG iters: {cg_iters:3d}  residual: {cg_residual:.2e}  "
                f"max strain: {strain:.2e}  cost: {elast_ms:.2f} ms"
            )

        # Sync changes to framework
        if changed:
            self.sync_settings_to_framework()


    def _draw_visualization_block(self):
        """Visualization controls (color mode, heatmap, invisibility, mesh view)."""
        if self.visualizer is None:
            self.imgui.text("(no visualizer attached)")
            return

        from scripts.visualization.visualize import COLOR_MODE_NAMES, HEATMAP_FIELD_NAMES

        vsettings = self.visualizer.settings

        self.imgui.text("Color Mode")
        c, new_mode = self.imgui.combo(
            "##color_mode", int(vsettings.color_mode), COLOR_MODE_NAMES
        )
        if c:
            vsettings.color_mode = int(new_mode)

        if int(vsettings.color_mode) == 0:
            self.imgui.text("Heatmap Field")
            c, new_field = self.imgui.combo(
                "##heatmap_field", int(vsettings.heatmap_field), HEATMAP_FIELD_NAMES
            )
            if c:
                vsettings.heatmap_field = int(new_field)

            vis = self.visualizer
            if int(vsettings.heatmap_field) == 0:  # Velocity
                c, vis.vmin_v = self.imgui.input_float("vmin", vis.vmin_v, 0.01, 1.0, "%.2f")
                c, vis.vmax_v = self.imgui.input_float("vmax", vis.vmax_v, 0.01, 1.0, "%.2f")
            elif int(vsettings.heatmap_field) == 1:  # Divergence
                c, vis.vmin_div = self.imgui.input_float("vmin", vis.vmin_div, 0.01, 1.0, "%.2f")
                c, vis.vmax_div = self.imgui.input_float("vmax", vis.vmax_div, 0.01, 1.0, "%.2f")
            else:  # Density
                c, vis.vmin_den = self.imgui.input_float("vmin", vis.vmin_den, 0.01, 1.0, "%.2f")
                c, vis.vmax_den = self.imgui.input_float("vmax", vis.vmax_den, 0.01, 1.0, "%.2f")

        self.imgui.separator()
        self.imgui.text("Invisible Objects")

        c, inv = self.imgui.checkbox("invisible", bool(vsettings.invisible))
        if c:
            vsettings.invisible = bool(inv)

        if getattr(self.visualizer, "invisible_object_ids", None):
            self.imgui.text(f"IDs: {list(self.visualizer.invisible_object_ids)}")
            if vsettings.invisible:
                self.imgui.text("Mode: hidden")
            else:
                self.imgui.text("Mode: shown")
        else:
            self.imgui.text("No transparentObjects in config")

        self.imgui.separator()
        c, mv = self.imgui.checkbox(
            "mesh view (static rigids)", bool(vsettings.mesh_visualize)
        )
        if c:
            vsettings.mesh_visualize = bool(mv)
        n_meshes = len(getattr(self.visualizer, "static_rigid_meshes", []))
        self.imgui.text(f"  loaded meshes: {n_meshes}")

        self.imgui.separator()
        c, hf = self.imgui.checkbox("hide fluid", bool(vsettings.hide_fluid))
        if c:
            vsettings.hide_fluid = bool(hf)
        if getattr(self.visualizer, "has_deformables", False):
            c, hd = self.imgui.checkbox("hide deformables", bool(vsettings.hide_deformables))
            if c:
                vsettings.hide_deformables = bool(hd)
        else:
            self.imgui.text("  no deformables in scene: hides all fluid")

    def close(self):
        """Clean up resources."""
        if not self.is_available:
            return

        # Remove callback from renderer
        if self.renderer is not None and hasattr(self.renderer, 'render_2d_callbacks'):
            self.renderer.render_2d_callbacks.remove(self.render_callback)

        # Shutdown ImGui
        if self.impl is not None:
            self.impl.shutdown()


        self.is_available = False
        self.renderer = None
        self.impl = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
