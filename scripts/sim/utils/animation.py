"""Keyframe animation for rigid bodies."""

import numpy as np
import warp as wp


@wp.kernel
def apply_transform_kernel(
    x: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    is_kinematic: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    object_id_arr: wp.array(dtype=int),
    num_particles: int,
    target_object_id: int,
    pivot: wp.vec3,
    R: wp.mat33,
    translation: wp.vec3,
    dt: float,
):
    """new_x = pivot + T + R * (x0 - pivot)"""
    p = wp.tid()
    if p >= num_particles:
        return
    if object_id_arr[p] != target_object_id:
        return

    d = x0[p] - pivot
    new_x = pivot + translation + R * d
    if dt > 0.0:
        v[p] = (new_x - x[p]) / dt
    x[p] = new_x
    is_kinematic[p] = 1
    is_dynamic[p] = 0


@wp.kernel
def clear_kinematic_kernel(
    is_kinematic: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    material: wp.array(dtype=int),
    num_particles: int,
):
    """Reset kinematic flags after physics step."""
    p = wp.tid()
    if p >= num_particles:
        return
    is_kinematic[p] = 0
    if material[p] != 0:
        is_dynamic[p] = 1


def _lerp_vec3(keys, t):
    """Linear interpolation on a list of {"t": float, "value": [x,y,z]} keyframes."""
    if not keys:
        return np.zeros(3, dtype=np.float32)
    if t <= keys[0]["t"]:
        return np.array(keys[0]["value"], dtype=np.float32)
    if t >= keys[-1]["t"]:
        return np.array(keys[-1]["value"], dtype=np.float32)
    for i in range(len(keys) - 1):
        t0, t1 = keys[i]["t"], keys[i + 1]["t"]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            v0 = np.array(keys[i]["value"], dtype=np.float32)
            v1 = np.array(keys[i + 1]["value"], dtype=np.float32)
            return v0 + alpha * (v1 - v0)
    return np.array(keys[-1]["value"], dtype=np.float32)


def _lerp_scalar(keys, t):
    """Linear interpolation on a list of {"t": float, "value": float} keyframes."""
    if not keys:
        return 0.0
    if t <= keys[0]["t"]:
        return float(keys[0]["value"])
    if t >= keys[-1]["t"]:
        return float(keys[-1]["value"])
    for i in range(len(keys) - 1):
        t0, t1 = keys[i]["t"], keys[i + 1]["t"]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            return keys[i]["value"] + alpha * (keys[i + 1]["value"] - keys[i]["value"])
    return float(keys[-1]["value"])


def _remap_time(t, loop_mode, loop_start, loop_end):
    """Remap time according to loop mode."""
    if loop_mode == "none" or loop_end <= loop_start:
        return t
    period = loop_end - loop_start
    if t < loop_start:
        return loop_start
    elapsed = t - loop_start
    if loop_mode == "cycle":
        return loop_start + (elapsed % period)
    elif loop_mode == "pingpong":
        phase = elapsed % (2.0 * period)
        if phase < period:
            return loop_start + phase
        else:
            return loop_end - (phase - period)
    return t


def _axis_angle_to_matrix(axis, angle_rad):
    """Rodrigues' rotation formula: axis (unit vec3), angle in radians -> 3x3 matrix."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ], dtype=np.float32)


def _resolve_named_pivot(name, obj, track_cfg):
    """Named pivots: "object_restCOM" = mean rest position of the object's
    particles; "world" = track_cfg["pivot_world"]. Falls back to the translation."""
    fallback = obj.get("translation", [0, 0, 0])
    if name == "world":
        return track_cfg.get("pivot_world", track_cfg.get("pivotWorld", fallback))
    pts = obj.get("voxelizedPoints", obj.get("generatedPositions"))
    if pts is not None and len(pts) > 0:
        return np.asarray(pts, dtype=np.float32).mean(axis=0)
    return fallback


def _parse_axis(axis_cfg):
    """Parse rotation axis from string or list."""
    if isinstance(axis_cfg, str):
        return {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis_cfg.lower()]
    return list(axis_cfg)


class AnimationTrack:
    def __init__(self, object_id, channels, pivot, rotation_axis,
                 loop_mode="none", loop_start=0.0, loop_end=0.0):
        self.object_id = object_id
        self.channels = channels
        self.pivot = np.array(pivot, dtype=np.float32)
        self.rotation_axis = np.array(rotation_axis, dtype=np.float32)
        n = np.linalg.norm(self.rotation_axis)
        if n > 1e-12:
            self.rotation_axis /= n
        self.loop_mode = loop_mode
        self.loop_start = loop_start
        self.loop_end = loop_end


def parse_tracks(obj_id, obj):
    """AnimationTracks of one scene entry (its "Animation" block); [] if none."""
    tracks = []
    for track_cfg in (obj.get("Animation") or {}).get("tracks", []):
        loop_cfg = track_cfg.get("loop", {})
        # Pivot: vec3, or a name ("object_restCOM"/"world"); default = translation
        pivot = track_cfg.get("pivot", obj.get("translation", [0, 0, 0]))
        if isinstance(pivot, str):
            pivot = _resolve_named_pivot(pivot, obj, track_cfg)
        tracks.append(AnimationTrack(
            object_id=int(obj_id),
            channels=track_cfg.get("channels", {}),
            pivot=pivot,
            rotation_axis=_parse_axis(track_cfg.get("rotation_axis", "y")),
            loop_mode=loop_cfg.get("mode", "none"),
            loop_start=float(loop_cfg.get("start", 0.0)),
            loop_end=float(loop_cfg.get("end", 0.0)),
        ))
    return tracks


def track_transform(track, t):
    """(R, translation) of a track at time t: new_x = pivot + T + R (x0 - pivot)."""
    t_local = _remap_time(t, track.loop_mode, track.loop_start, track.loop_end)
    trans = _lerp_vec3(track.channels.get("translation", []), t_local)
    angle_deg = _lerp_scalar(track.channels.get("rotation_angle_deg", []), t_local)
    R = _axis_angle_to_matrix(track.rotation_axis, float(angle_deg) * np.pi / 180.0)
    return R, trans


class AnimationEngine:
    def __init__(self, ps):
        self.ps = ps
        self.tracks = []

    def build(self, scene_data):
        """Parse animation tracks from scene data object_map."""
        self.tracks = []
        for obj_id, obj in scene_data.get("object_map", {}).items():
            self.tracks.extend(parse_tracks(obj_id, obj))

    def apply(self, t, dt):
        """Apply all animation tracks at time t."""
        ps = self.ps
        for track in self.tracks:
            R, trans = track_transform(track, t)

            wp.launch(
                kernel=apply_transform_kernel,
                dim=ps.particle_num,
                inputs=[
                    ps.x, ps.x0, ps.v, ps.is_kinematic, ps.is_dynamic,
                    ps.object_id, ps.particle_num,
                    track.object_id,
                    wp.vec3(float(track.pivot[0]), float(track.pivot[1]), float(track.pivot[2])),
                    wp.mat33(
                        float(R[0, 0]), float(R[0, 1]), float(R[0, 2]),
                        float(R[1, 0]), float(R[1, 1]), float(R[1, 2]),
                        float(R[2, 0]), float(R[2, 1]), float(R[2, 2]),
                    ),
                    wp.vec3(float(trans[0]), float(trans[1]), float(trans[2])),
                    dt,
                ],
            )

    def clear_kinematic(self):
        """Reset kinematic flags after physics step."""
        ps = self.ps
        wp.launch(
            kernel=clear_kinematic_kernel,
            dim=ps.particle_num,
            inputs=[ps.is_kinematic, ps.is_dynamic, ps.material, ps.particle_num],
        )

    def has_tracks(self):
        return len(self.tracks) > 0
