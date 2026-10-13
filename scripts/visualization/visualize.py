"""
Rendering-side visualization for WarpSPH.

This module owns visualization state (coloring, heatmap, invisible objects)
and populates GPU buffers for rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np
import warp as wp


class ColorMode(IntEnum):
    """Particle coloring mode."""

    HEATMAP = 0
    ORIGINAL = 1


class HeatmapField(IntEnum):
    """Scalar field used in heatmap mode."""

    VELOCITY = 0
    DIVERGENCE = 1
    DENSITY = 2


COLOR_MODE_NAMES = ["Heatmap", "Original"]
HEATMAP_FIELD_NAMES = ["Velocity", "Divergence", "Density"]


@dataclass
class VisualizationSettings:
    """Visualization settings controlled by GUI."""

    color_mode: int = int(ColorMode.HEATMAP)
    heatmap_field: int = int(HeatmapField.DENSITY)

    # If True, objects listed in config.transparentObjects are hidden (particles
    # and mesh view); if False, they are drawn with their scene colour.
    invisible: bool = True

    # When True, static rigid bodies are rendered as their original triangle
    # meshes (loaded from `geometryFile`) and their particle representation is
    # hidden. Animated rigids and fluid particles are unaffected.
    mesh_visualize: bool = False

    # Hide plain fluid / deformable particles independently (both are material 1).
    hide_fluid: bool = False
    hide_deformables: bool = False


@wp.func
def clamp01(x: float) -> float:
    return wp.min(wp.max(x, 0.0), 1.0)


@wp.func
def lerp3(a: wp.vec3, b: wp.vec3, t: float) -> wp.vec3:
    return a * (1.0 - t) + b * t


@wp.func
def heat_vel_color(s: float) -> wp.vec3:
    # blue -> white
    return lerp3(wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 1.0, 1.0), s)


@wp.func
def heat_div_color(s: float) -> wp.vec3:
    # white (no divergence) -> red (high positive divergence)
    return wp.vec3(1.0, 1.0 - s, 1.0 - s)


@wp.func
def heat_den_color(s: float) -> wp.vec3:
    # white -> red
    if s >= 0.5:
        return lerp3(wp.vec3(1.0, 1.0, 1.0), wp.vec3(1.0, 0.0, 0.0), (s - 0.5) * 2.0)
    return wp.vec3(1.0, 1.0, 1.0)


@wp.kernel
def update_visualization_buffers(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    divergence: wp.array(dtype=float),
    density: wp.array(dtype=float),
    density0: wp.array(dtype=float),
    color_rgba_255: wp.array(dtype=wp.vec4),
    object_id: wp.array(dtype=int),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    num_particles: int,
    particle_max_num: int,
    invisible_obj_table: wp.array(dtype=int),
    invisible_enabled: int,
    mesh_hide_table: wp.array(dtype=int),
    mesh_view_enabled: int,
    reg_id: wp.array(dtype=int),
    has_deformables: int,
    hide_fluid: int,
    hide_deformables: int,
    color_mode: int,
    heatmap_field: int,
    # Normalization parameters
    vmin_v: float,
    vmax_v: float,
    vmin_div: float,
    vmax_div: float,
    vmin_den: float,
    vmax_den: float,
    # Position used to park hidden particles, outside the far plane
    hide_coord: float,
    # Outputs
    x_vis: wp.array(dtype=wp.vec3),
    color_vis: wp.array(dtype=wp.vec3),
):
    """Populate render buffers (positions + RGB colors)."""
    p_i = wp.tid()
    if p_i >= particle_max_num:
        return

    far = wp.vec3(hide_coord, hide_coord, hide_coord)

    if p_i >= num_particles:
        x_vis[p_i] = far
        color_vis[p_i] = wp.vec3(0.0, 0.0, 0.0)
        return

    pos = x[p_i]

    c4 = color_rgba_255[p_i]
    col = wp.vec3(c4[0], c4[1], c4[2]) * (1.0 / 255.0)

    # Heatmap applies to every particle except static rigid ones
    apply_heatmap = (color_mode == int(ColorMode.HEATMAP)) and not (
        material[p_i] == 0 and is_dynamic[p_i] == 0
    )

    if apply_heatmap:
        s = float(0.0)
        if heatmap_field == int(HeatmapField.VELOCITY):
            vv = v[p_i]
            mag = wp.sqrt(wp.dot(vv, vv))
            s = (mag - vmin_v) / (vmax_v - vmin_v + 1e-8)
            s = clamp01(s)
            col = heat_vel_color(s)
        elif heatmap_field == int(HeatmapField.DIVERGENCE):
            val = divergence[p_i]
            s = (val - vmin_div) / (vmax_div - vmin_div + 1e-8)
            s = clamp01(s)
            col = heat_div_color(s)
        else:
            val = density[p_i] - density0[p_i]
            s = (val - vmin_den) / (vmax_den - vmin_den + 1e-8)
            s = clamp01(s)
            col = heat_den_color(s)

    # Invisible objects handling (config.transparentObjects)
    oid = object_id[p_i]
    is_invisible_obj = 0
    if 0 <= oid and oid < invisible_obj_table.shape[0]:
        is_invisible_obj = invisible_obj_table[oid]

    if is_invisible_obj == 1:
        if invisible_enabled == 1:
            pos = far
            col = wp.vec3(0.0, 0.0, 0.0)
        # else: keep the particle's assigned color (from JSON) so transparent
        # objects render with their configured color when visibility is enabled.

    # Mesh-view hides the particle representation of static rigids (the OBJ
    # mesh is drawn separately by Visualizer.render).
    if mesh_view_enabled == 1 and 0 <= oid and oid < mesh_hide_table.shape[0]:
        if mesh_hide_table[oid] == 1:
            pos = far
            col = wp.vec3(0.0, 0.0, 0.0)

    # Deformables share material 1 with fluid; reg_id > 0 tells them apart.
    if material[p_i] == 1:
        is_def = 0
        if has_deformables == 1:
            if reg_id[p_i] > 0:
                is_def = 1
        if (is_def == 0 and hide_fluid == 1) or (is_def == 1 and hide_deformables == 1):
            pos = far
            col = wp.vec3(0.0, 0.0, 0.0)

    x_vis[p_i] = pos
    color_vis[p_i] = col


class Visualizer:
    """Rendering-side visualization (colors, masking, domain box)."""

    def __init__(self, ps, config):
        self.ps = ps
        self.config = config
        self.settings = VisualizationSettings()

        # Normalization
        self.vmin_v, self.vmax_v = 0.0, 1.5
        self.vmin_div, self.vmax_div = 0.0, 5.0
        self.vmin_den, self.vmax_den = -50.0, 50.0

        # Park hidden particles well beyond the renderer far plane
        # (main.py sets it to 3x the domain diagonal).
        diag = float(np.linalg.norm(np.asarray(config.domain_size, dtype=np.float64)))
        self.hide_coord = 20.0 * max(diag, 1e-6)

        # Invisible object ids (config.transparentObjects)
        self.invisible_object_ids = list(config.get_transparent_objects())
        self.max_object_id = 256
        self.set_invisible_objects(self.invisible_object_ids)

        # Static-rigid triangle meshes loaded from `geometryFile`, populated by
        # set_static_rigid_meshes(). Each entry: (object_id, vertices, indices, color).
        # When settings.mesh_visualize is True these are rendered and the matching
        # particles are hidden via `mesh_hide_table`.
        self.static_rigid_meshes: list[tuple[int, np.ndarray, np.ndarray, tuple[float, float, float]]] = []
        self.mesh_hide_table = wp.zeros(self.max_object_id, dtype=int, device=wp.get_device())
        self._mesh_pos = {}  # last on/off-screen position pushed per mesh instance

        # Per-particle deformable registry id (elasticity.reg_id, 0 = fluid);
        # set by set_deformables(). Placeholder when the scene has none.
        self.reg_id = wp.zeros(1, dtype=int, device=wp.get_device())
        self.has_deformables = False

        # Render buffers
        max_n = int(self.ps.particle_max_num)
        self.x_vis = wp.zeros(max_n, dtype=wp.vec3, device=wp.get_device())
        self.color_vis = wp.zeros(max_n, dtype=wp.vec3, device=wp.get_device())

        # CUDA-GL interop for color buffers (registered after first render)
        self._color_gl_registered = False
        self._color1_cuda_buffer = None
        self._color2_cuda_buffer = None

        # Domain box geometry
        ds = config.domain_start
        de = tuple(s + sz for s, sz in zip(config.domain_start, config.domain_size))
        self.domain_box_vertices = np.array(
            [
                [ds[0], ds[1], ds[2]],
                [de[0], ds[1], ds[2]],
                [de[0], de[1], ds[2]],
                [ds[0], de[1], ds[2]],
                [ds[0], ds[1], de[2]],
                [de[0], ds[1], de[2]],
                [de[0], de[1], de[2]],
                [ds[0], de[1], de[2]],
            ],
            dtype=np.float32,
        )
        self.domain_box_indices = np.array(
            [
                0, 1, 1, 2, 2, 3, 3, 0,
                4, 5, 5, 6, 6, 7, 7, 4,
                0, 4, 1, 5, 2, 6, 3, 7,
            ],
            dtype=np.int32,
        )

    def set_deformables(self, elasticity):
        """Share the elasticity registry's reg_id so fluid and deformables can be hidden apart."""
        if elasticity is None:
            return
        self.reg_id = elasticity.reg_id
        self.has_deformables = True

    def set_invisible_objects(self, object_ids: Iterable[int]):
        """Update the invisible object table from a list of object IDs."""
        table = np.zeros((self.max_object_id,), dtype=np.int32)
        for oid in object_ids:
            oid_i = int(oid)
            if 0 <= oid_i < self.max_object_id:
                table[oid_i] = 1
        self.invisible_obj_table = wp.array(table, dtype=int, device=wp.get_device())

    def set_static_rigid_meshes(self, rigid_bodies, animated_obj_ids=None):
        """Load triangle meshes for static, non-animated rigid bodies with a geometryFile.

        These are rendered as meshes (instead of voxelized particles) when
        `settings.mesh_visualize` is True. Particles whose object_id appears in
        `mesh_hide_table` are then pushed off-screen by the update kernel.
        """
        from scripts.io.scene_loader import SceneLoader
        animated = set(int(o) for o in (animated_obj_ids or []))
        self.static_rigid_meshes = []
        hide = np.zeros((self.max_object_id,), dtype=np.int32)
        for rb in rigid_bodies or []:
            oid = int(rb.get("objectId", -1))
            geom = rb.get("geometryFile")
            if oid < 0 or oid in animated or not geom or rb.get("isDynamic", False):
                continue
            try:
                mesh = SceneLoader.load_transformed_mesh(rb)
            except Exception as e:
                print(f"[mesh-view] could not load mesh for object {oid}: {e}")
                continue
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            c = np.asarray(rb.get("color", [200, 200, 200]), dtype=np.float32)
            if c.max() > 1.0:
                c = c / 255.0
            color = (float(c[0]), float(c[1]), float(c[2]))
            self.static_rigid_meshes.append((oid, verts, faces, color))
            if 0 <= oid < self.max_object_id:
                hide[oid] = 1
        self.mesh_hide_table = wp.array(hide, dtype=int, device=wp.get_device())

    def update(self):
        """Update render buffers from current particle state."""
        wp.launch(
            kernel=update_visualization_buffers,
            dim=int(self.ps.particle_max_num),
            inputs=[
                self.ps.x,
                self.ps.v,
                self.ps.divergence,
                self.ps.density,
                self.ps.density0,
                self.ps.color,
                self.ps.object_id,
                self.ps.material,
                self.ps.is_dynamic,
                int(self.ps.particle_num),
                int(self.ps.particle_max_num),
                self.invisible_obj_table,
                1 if self.settings.invisible else 0,
                self.mesh_hide_table,
                1 if self.settings.mesh_visualize else 0,
                self.reg_id,
                1 if self.has_deformables else 0,
                1 if self.settings.hide_fluid else 0,
                1 if self.settings.hide_deformables else 0,
                int(self.settings.color_mode),
                int(self.settings.heatmap_field),
                float(self.vmin_v),
                float(self.vmax_v),
                float(self.vmin_div),
                float(self.vmax_div),
                float(self.vmin_den),
                float(self.vmax_den),
                float(self.hide_coord),
            ],
            outputs=[self.x_vis, self.color_vis],
            device=wp.get_device(),
        )

    def _try_register_color_gl_buffers(self, instancer):
        """Register GL color buffers with CUDA for zero-copy GPU→GPU updates."""
        try:
            buf1 = instancer.instance_color1_buffer
            buf2 = instancer.instance_color2_buffer
            if buf1 is not None and buf2 is not None:
                device = wp.get_device()
                self._color1_cuda_buffer = wp.RegisteredGLBuffer(
                    int(buf1.value), device,
                    flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                )
                self._color2_cuda_buffer = wp.RegisteredGLBuffer(
                    int(buf2.value), device,
                    flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                )
                print("Color buffers registered with CUDA-GL interop")
        except Exception as e:
            print(f"CUDA-GL interop for colors not available: {e}")
            self._color1_cuda_buffer = None
            self._color2_cuda_buffer = None
        self._color_gl_registered = True

    def render(self, renderer, *, domain_line_radius: float):
        """Render particles and domain box. Must be called between begin_frame/end_frame."""
        # NOTE:
        # Warp OpenGLRenderer.render_points() will index colors[0] on first registration.
        # Passing a wp.array here will crash (wp.array disallows Python indexing).
        # So we register/update points without per-instance colors, then update the
        # instancer colors via ShapeInstancer.update_instances().
        renderer.render_points(
            name="particles",
            points=self.x_vis,
            radius=self.ps.particle_radius,
            colors=(1.0, 1.0, 1.0),
        )

        instancer = getattr(renderer, "_shape_instancers", {}).get("particles", None)
        if instancer is not None:
            if not self._color_gl_registered:
                self._try_register_color_gl_buffers(instancer)

            if self._color1_cuda_buffer is not None:
                # Fast path: GPU→GPU via CUDA-GL interop (no CPU round-trip).
                # Fallback below runs only if interop registration failed.
                n = instancer.num_instances
                mapped1 = self._color1_cuda_buffer.map(dtype=wp.vec3, shape=(n,))
                wp.copy(mapped1, self.color_vis)
                self._color1_cuda_buffer.unmap()

                mapped2 = self._color2_cuda_buffer.map(dtype=wp.vec3, shape=(n,))
                wp.copy(mapped2, self.color_vis)
                self._color2_cuda_buffer.unmap()
            else:
                # Original path: GPU→CPU→GPU
                colors_np = self.color_vis.numpy()
                instancer.update_instances(colors1=colors_np, colors2=colors_np)

        renderer.render_line_list(
            name="domain_box",
            vertices=self.domain_box_vertices,
            indices=self.domain_box_indices,
            color=(1.0, 0.8, 0.0),
            radius=domain_line_radius,
        )

        # Warp ignores `visible=` at draw time, so a mesh is hidden by moving its
        # instance off-screen (vertices are already in world space). Register each
        # mesh once, then only move its instance when its visibility changes.
        hidden_ids = set(self.invisible_object_ids) if self.settings.invisible else set()
        for oid, verts, faces, color in self.static_rigid_meshes:
            name = f"rigid_mesh_{oid}"
            shown = self.settings.mesh_visualize and oid not in hidden_ids
            pos = (0.0, 0.0, 0.0) if shown else (1.0e6, 1.0e6, 1.0e6)
            if name not in self._mesh_pos:
                renderer.render_mesh(
                    name=name,
                    points=verts,
                    indices=faces.reshape(-1),
                    colors=color,
                    pos=pos,
                    smooth_shading=True,
                )
            elif self._mesh_pos[name] != pos:
                renderer.update_shape_instance(name, pos=pos)
            self._mesh_pos[name] = pos

