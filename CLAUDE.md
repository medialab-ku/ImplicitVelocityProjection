# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Demo code for **Implicit Velocity Projection (IP)** (Kee et al., SIGGRAPH Asia 2026), built as a GPU SPH fluid simulator on NVIDIA Warp with an interactive OpenGL + ImGui viewer. IP is the method of the repository; PRJ (IISPH, plus a DFSPH pass), PBF and SISPH are baselines for comparison. The code calls the method `IP` throughout (class `IP`, `ip.py`, GUI label "IP"). There is no headless mode, sweep or test suite; the viewer is the only entry point.

## Commands

Setup (Python 3.11): `pip install -r requirements.txt`, then `unzip data/models.zip -d data/` (scene geometry → `data/models/`). Warp needs a CUDA GPU.

```
python -m scripts.main --scene_file data/scenes/dambreak_dragon.json
```

Run it as a module from the repo root; `python scripts/main.py` breaks the package imports. `--scene_file` is the only option; the default scene `pillar100K` needs about 28 GB of RAM to load. Controls: SPACE start/pause, R reset, mouse rotate, WASD/E/Q camera, ESC exit.

Scenes (`data/scenes/`), all starting with IP and implicit viscosity / surface tension: `dambreak_dragon`, `small_pillar` (smallest), `pillar100K`, `stack` (fluid bodies spawned in waves plus an animated plunger). None of them has deformable bodies; the elasticity solver runs only for scenes with `SolidBodies` or solid `ScheduledSpawns`.

## IP ([scripts/sim/forces/pressure/ip.py](scripts/sim/forces/pressure/ip.py))

Velocity-based (`is_position_based = False`); selected by `"pressureMethod": 1` (the default). With `d = Δt v`, `c = ρ − ρ₀`, `D⁻¹ = k = 1/(Aii + eps)` and `φ = max(c + J d, 0)`, each outer iteration solves

`(M + ω Jᵀ diag(φ′ D⁻¹) J) δ = ω Jᵀ D⁻¹ φ`, then `d ← d − δ`,

and finally sets `v = d / Δt`. In `IP.solve(dt)`:

- `ip_pack_fused`: `t = J d + c`, `f = max(t, 0)`, `dfdt = φ′` (Heaviside) in one launch. The average of `f / ρ₀` is the density error, checked **before** each update; the loop stops once it is below `densityErrorTol` and at least one update was applied, or at `maxOuterIters`.
- `ip_rhs_fused`: `rhs = ω Jᵀ (f k)` in one launch.
- `PCG`: plain CG (identity preconditioner, starts from zero) on `A = M + ω Jᵀ diag(dfdt k) J`, applied matrix-free by `compute_Ax` (`compute_Jx_pcg` → `compute_Ax_part1` → `compute_Ax_part2`; part 2 is the exact transpose of the Jx kernel, so `A` stays symmetric). It stops when `‖r‖² < tol_pcg` (absolute, after at least 2 iterations) or at `max_iteration_pcg` (1000); `pAp < 0` raises.
- After the loop: `check_converged`, `stats_final_density_error` at the final `d`, and `compute_divergence()` for the viewer.

Keys: `omega` (10), `tol_pcg` (1e-6), `densityErrorTol` (1e-3), `maxOuterIters` (1000), `fixedOuterIters` (false). The GUI IP panel edits `omega`, the CG tolerance (a decade slider) and the CG iteration cap, and shows total / average CG iterations; the shared group above it edits the density tolerance and the outer-iteration settings. `stats_outer_iters` counts updates applied; `stats_pcg_iters` holds one CG count per update.

## Architecture

### One time step ([scripts/framework.py](scripts/framework.py))

`Framework.forward()`:

1. Optional CFL `dt` (`useCFL`; capped at `timeStepSize`, floored at `cflMinDt`).
2. Keyframe animations set kinematic rigid positions.
3. Gravity, surface tension, viscosity, then elasticity (`v += Δv`) when the scene has deformables.
4. Pressure solve `pressure.solve(dt)`, then PRJ's divergence-free pass `solve_df(dt)` if it is enabled (both skipped when `pressureEnabled` is false). Position-based solvers (`is_position_based = True`) advect, enforce the boundary and rebuild neighbours **before** the solve; velocity-based solvers (IP, PRJ) advect **after** it.
5. Scheduled spawns; newly spawned deformables are registered after the neighbour rebuild.
6. Neighbour search into the fixed-size cache.
7. Density and the diagonal `k = 1/(Aii + eps)` read by IP and PRJ.

The GUI shows `stats_pressure_ms` and `stats_elasticity_ms`, timed around stages 4 and 3.

### Pressure solvers ([scripts/sim/forces/pressure/](scripts/sim/forces/pressure/))

| Solver | `pressureMethod` | GUI index | Class (file) | Kind |
| --- | --- | --- | --- | --- |
| **IP** | 1 (default) | 1 | `IP` (`ip.py`) | velocity |
| IISPH / DFSPH | 0 | 0 "PRJ (IISPH, DFSPH)" | `PRJ` (`prj.py`) | velocity |
| PBF | 5 | 2 | `PBF` (`pbf.py`) | position |
| SISPH | 3 | 3 | `SISPH` (`sisph.py`) | position |

An unknown `pressureMethod` raises.

- **PRJ** (baseline): projected Jacobi on the constant-density constraint with fused head/tail kernels, `omega` 0.5, optional warm start. Only PRJ has a divergence-free pass (`enable_df`, `df_num_threshold`, `df_tol_opt`; the GUI's "Divergence-Free Solve"), which reuses `maxOuterIters`.
- **PBF** (baseline) pins `omega = 0.5` and ignores the scene's `omega`.
- **SISPH** (baseline) keys: `sisphStiffness` (1e6), `sisphEnergyType` (0 = B0, 1 = B1, 2 = B2), `sisphConserveMomentum` (on), `sisphLineSearch` (off), `sisphAccel` (0 none, 1 Chebyshev, 2 Anderson) with `sisphChebyRho` / `sisphChebyStabilize` / `sisphChebyDelay` / `sisphChebyGamma`.
- **Stopping**: every solver stops on the average density error (`densityErrorTol`) or at `maxOuterIters`, and raises `RuntimeError` at the cap above tolerance (the viewer then resets). `fixedOuterIters` runs exactly `maxOuterIters` updates. Position-based solvers can instead stop on `max_i ‖x^{l+1} − x^l‖ / h < posTol` (`convergenceMode: 0`).

`PressureBase` ([pressure_base.py](scripts/sim/forces/pressure/pressure_base.py)) owns the shared buffers (`p`, `dx`, `k`, `x_prev`, …), the `stats_*` fields, the shared kernels (density, `Aii`, `J x`, `Jᵀ x`, active mask) and the helpers `compute_constraint`, `compute_J_x` / `compute_J_tr_x`, `compute_avg_density_error` / `read_avg_error`, `dot`, `final_density_error`, `compute_divergence`, `update_velocity_from_positions(dt)`, `compute_max_pos_update(x_prev)`, `convergence_metric`, `check_converged(err, n_updates, pos_res)` and `reset_stats()`.

**Adding a solver.**
1. Subclass `PressureBase`. In `solve(dt)`, call `reset_stats()` first, fill `stats_density_errors` / `stats_iter_costs` (and `stats_pcg_iters` for an inner solve), and end with `check_converged(...)`.
2. If it moves positions directly, set `is_position_based = True`; snapshot `x` into `x_prev` before each update, append `compute_max_pos_update(x_prev)` to `stats_pos_residuals`, stop via `convergence_metric(err, pos_res)`, and recover `v` with `update_velocity_from_positions(dt)`.
3. Register it in `_pressure_classes` in `framework.py`.
4. In `scripts/visualization/gui.py`, add it to `SolverType` / `SOLVER_NAMES`, the class-name mapping, `switch_solver` and a settings panel.

### Non-pressure forces and elasticity

- `viscosityMethod` / `surfaceTensionMethod`: 1 = `Weiler18` / `Jeske23` (implicit, CG; all shipped scenes), 0 = simple explicit versions.
- `ElasticityKee2023` ([kee2023.py](scripts/sim/forces/elasticity/kee2023.py)): co-rotated (`materialType: 0`) or Stable Neo-Hookean (`1`), one Newton step per time step (`maxIterNewton`), matrix-free PCG (`maxIterCG`, `tolCG`), material `youngsModulus` / `poissonsRatio` (overridable per body), optional `alphaZE`. It exists only when the scene has `SolidBodies` or a `ScheduledSpawns` entry with `"material": "solid"`. Each deformable particle's `density0` is its body's own rest SPH density, so the density constraint engages only at contact. Bodies are voxelized at pitch `deformableSamplingFactor · d`; denser sampling needs a larger `neighborCacheSize`. Per-body `fixedAABB` pins particles and `randomize` perturbs the start.

### Particle data and neighbour cache

`SimulationData` ([simulation_data.py](scripts/sim/simulation_data.py)) holds every per-particle Warp array sized to `particle_max_num`; `particle_num` is the live count and grows as scheduled spawns fire. `material == 1` is fluid (deformables too). Neighbours are cached per step in `neighbor_ids[i*cache_size + k]` / `neighbor_gradW[...]`, and kernels loop over `min(neighbor_count[i], cache_size)` (`neighborCacheSize`, default 60). `NeighbourSearch.build()` slices `positions[:num_particles]`, and **this slice is load-bearing**: `wp.HashGrid` folds cell coordinates with `%`, so unused slots would wrap onto real cells and slow every lookup there.

### Scenes

JSON with `Configuration` plus `RigidBodies`, `SolidBodies`, `FluidBlocks`, `FluidBodies`, `ScheduledSpawns`, `GeneratedBodies`. `SceneLoader` ([scene_loader.py](scripts/io/scene_loader.py)) voxelizes the OBJ in `geometryFile` with trimesh. Objects can carry `Animation.tracks[]` (`translation` / `rotation_angle_deg` keyframes, `rotation_axis`, `pivot`, `loop`). A `ScheduledSpawns` entry fires `spawnNum` waves every `interval` from `startTime`, optionally on a `grid`, with `waveOffset` and `colorCycle`; a rigid entry with `isDynamic: false` and a translation track drives a plunger. `transparentObjects` lists object ids the viewer skips. `Configuration` keys outside the `SimConfig` dataclass are read on demand with `config.get_cfg(name, default)`, so each solver's keys are documented where it reads them.

## Conventions

- GPU-first: hot code is Warp kernels (`@wp.kernel` / `@wp.func`). Add kernels rather than CPU loops over particle arrays.
- Comments stay short: at most about three lines per block.
- Results are not bit-reproducible (float atomics); compare against run-to-run noise, not exact values.
- Verify a change by running a scene in the viewer. For a GUI-free check, set up `SceneLoader`, `SimulationData`, `NeighbourSearch` and `Framework` as `scripts/main.py` does and call `Framework.forward()` for a few steps.

## Porting from SPlisHSPlasH

The reference implementation is [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) (C++/OpenMP, CPU-based), expected as a sibling checkout at `../SPlisHSPlasH`. When porting solver logic from it:

1. Identify the target algorithm in SPlisHSPlasH.
2. Ask Claude to explain the algorithm's loop structure and data dependencies before writing any code.
3. Port one function at a time — never the whole file at once.
