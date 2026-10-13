"""
WarpSPH - Main simulation entry point
"""

import argparse
import numpy as np
import warp as wp
import warp.render
from pyglet.window import key
from pyglet import event
from scripts.io.config_builder import SimConfig
from scripts.sim.simulation_data import SimulationData
from scripts.sim.neighbour_search import NeighbourSearch
from scripts.io.scene_loader import SceneLoader
from scripts.framework import Framework
from scripts.visualization.gui import GUI
from scripts.visualization.visualize import Visualizer


def main():
    """Main simulation entry point with scene loading and interactive visualization."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WarpSPH simulation runner")
    parser.add_argument(
        "--scene_file",
        "-s",
        dest="scene_file",
        default="./data/scenes/pillar100K.json",
        help="Path to scene JSON file",
    )
    args = parser.parse_args()

    scene_file = args.scene_file

    print(f"Loading scene: {scene_file}")

    # Initialize Warp
    wp.init()
    print(f"Warp initialized on device: {wp.get_device()}")

    # Load configuration from JSON scene file
    config = SimConfig.from_json(scene_file)

    print("\nConfiguration:")
    print(f"  Domain: {config.domain_size} starting at {config.domain_start}")
    print(f"  Particle radius: {config.particle_radius}")
    print(f"  Support radius: {config.support_radius}")
    print(f"  Time step: {config.dt}")
    print(f"  Gravity: {config.gravity}")

    # Create scene loader
    loader = SceneLoader(config)
    scene_name = loader.get_scene_name(scene_file)
    scene_data = loader.prepare_scene()

    print(f"\nScene: {scene_name}")

    # Create simulation data
    ps = SimulationData(config)

    # Populate scene: allocates memory and creates particles
    loader.populate_scene(ps, scene_data)

    # Create neighbor search
    ns = NeighbourSearch(
        domain_size=config.domain_size,
        support_radius=config.support_radius
    )

    # Create framework
    fw = Framework(ps, ns, config)
    fw.scheduled_spawn_system = loader.setup_scheduled_spawn_system(ps, scene_data)

    # Setup animation
    from scripts.sim.utils.animation import AnimationEngine
    anim = AnimationEngine(ps)
    anim.build(scene_data)
    if anim.has_tracks():
        fw.animation = anim

    fw.initialize()
    print("\nSimulation initialized\n")

    # Create visualization manager
    visualizer = Visualizer(ps=ps, config=config)
    visualizer.set_deformables(fw.elasticity)
    _animated_ids = {t.object_id for t in (fw.animation.tracks if fw.animation else [])}
    visualizer.set_static_rigid_meshes(scene_data.get("rigid_bodies", []), _animated_ids)
    print(
        f"Transparent objects: {visualizer.invisible_object_ids if visualizer.invisible_object_ids else 'None'}"
    )
    print(f"Static rigid meshes loaded for mesh view: {len(visualizer.static_rigid_meshes)}")

    # Create interactive OpenGL renderer
    domain_center = tuple(s + sz/2 for s, sz in zip(config.domain_start, config.domain_size))
    ground_y = config.domain_start[1] + config.domain_size[1] * 0.15
    horiz_dim = max(config.domain_size[0], config.domain_size[2])
    cam_distance = horiz_dim * 2.0
    # Clip planes and camera speeds scale with the domain diagonal so a
    # scene stays framed and navigable at any particle size. The near/far
    # ratio is fixed to keep depth precision scale-invariant.
    domain_diag = float(np.linalg.norm(np.asarray(config.domain_size, dtype=np.float64)))
    domain_diag = max(domain_diag, 1e-6)

    base_title = f"WarpSPH - {scene_name}"
    renderer = warp.render.OpenGLRenderer(
        title=base_title,
        screen_width=1280,
        screen_height=720,
        up_axis="Y",
        near_plane=domain_diag * 1e-3,
        far_plane=domain_diag * 3.0,
        camera_pos=(domain_center[0] + cam_distance, ground_y + cam_distance * 0.5, domain_center[2] + cam_distance),
        camera_front=(-0.6, -0.3, -0.6),
        fps=60,
        vsync=False,
        show_info=False,
        draw_grid=True,
        draw_sky=False,
        background_color=(0.10, 0.10, 0.11),
    )

    # Reduce sphere mesh resolution: 32×32 (default) → 4×4 segments per particle
    orig_sphere = type(renderer)._create_sphere_mesh
    type(renderer)._create_sphere_mesh = staticmethod(
        lambda r=1.0, nlat=4, nlon=4, rw=False: orig_sphere(r, nlat, nlon, rw)
    )

    # Warp flies at (_camera_speed * 100) units/s, so this crosses the domain
    # diagonal in about 4 s at any scene scale. QE below reuses the same value.
    camera_speed = domain_diag * 2.5e-3
    renderer._camera_speed = camera_speed

    print("Interactive renderer started")
    print("Controls:")
    print("  Mouse     - Rotate camera")
    print("  WASD      - Move camera (horizontal)")
    print("  E/Q       - Move camera (up/down)")
    print("  SPACEBAR  - Start/Pause simulation")
    print("  R         - Reset simulation")
    print("  ESC       - Exit")
    print("-" * 60)
    print("\nPress SPACEBAR to start the simulation...")

    frame = 0
    is_running = False

    key_q_pressed = False
    key_e_pressed = False

    # Create GUI as overlay on renderer window
    gui = GUI(renderer=renderer, framework=fw, visualizer=visualizer, title="WarpSPH")

    # Key handler for custom controls
    def on_key_press(symbol, modifiers):
        nonlocal is_running, key_q_pressed, key_e_pressed, frame

        # Leave keys to the panel while one of its fields has keyboard focus.
        if not getattr(renderer, "enable_keyboard_interaction", True):
            return

        if symbol == key.SPACE:
            is_running = not is_running
            state = "RUNNING" if is_running else "PAUSED"
            print(f"\nSimulation {state}")
            return event.EVENT_HANDLED

        elif symbol == key.R:
            fw.reset()
            frame = 0
            is_running = False
            print("\nSimulation RESET")
            return event.EVENT_HANDLED

        elif symbol == key.Q:
            key_q_pressed = True
            return event.EVENT_HANDLED
        elif symbol == key.E:
            key_e_pressed = True
            return event.EVENT_HANDLED

    def on_key_release(symbol, modifiers):
        nonlocal key_q_pressed, key_e_pressed

        if symbol == key.Q:
            key_q_pressed = False
            return event.EVENT_HANDLED
        elif symbol == key.E:
            key_e_pressed = False
            return event.EVENT_HANDLED

    renderer.window.push_handlers(on_key_press=on_key_press, on_key_release=on_key_release)

    # Main simulation loop (end_frame() below pumps events and redraws the window)
    while renderer.is_running():
        if key_q_pressed or key_e_pressed:
            step = camera_speed * renderer._frame_speed
            if key_q_pressed:
                step = -step
            renderer.camera_pos = (
                renderer.camera_pos[0],
                renderer.camera_pos[1] + step,
                renderer.camera_pos[2]
            )

        if is_running:
            try:
                fw.forward()
            except RuntimeError as e:
                print(f"[GUI] step failed: {e} - resetting and stopping")
                fw.reset()
                frame = 0
                is_running = False
                wp.synchronize()
                continue
            wp.synchronize()
            frame += 1

        gui.update(frame=frame, sim_time=float(fw.time), is_running=is_running)

        visualizer.update()

        renderer.begin_frame()
        visualizer.render(renderer, domain_line_radius=ps.particle_radius * 0.3)
        renderer.end_frame()

    gui.close()
    print("\n" + "-" * 60)
    print("Simulation finished!")


if __name__ == "__main__":
    main()
