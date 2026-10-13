"""
Configuration system for WarpSPH simulation.
"""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimConfig:
    """
    Simulation configuration parameters.
    """

    # Simulation domain
    domain_size: tuple = (1.0, 1.0, 1.0)
    domain_start: tuple = (0.0, 0.0, 0.0)

    # Particle properties
    particle_radius: float = 0.01
    support_radius: float = 0.04  # 4 * particle_radius
    particle_diameter: float = 0.02  # 2 * particle_radius
    density0: float = 1000.0  # Water density (kg/m^3)

    # Simulation parameters
    # Scene JSON key: "timeStepSize"
    dt: float = 0.0016  # Time step (seconds)

    # Physics
    gravity: tuple = (0.0, -9.8, 0.0)  # Gravity acceleration (m/s^2)

    # Solver selection (0=explicit, 1=weiler18)
    viscosity_method: int = 0

    # Surface tension method (0=Akinci13, 1=Jeske23)
    surface_tension_method: int = 0

    # Coefficients (from scene JSON, used as initial values)
    viscosity_coeff: float = 0.01
    surface_tension_coeff: float = 0.01

    # Scene data (loaded from JSON)
    scene_data: Optional[dict] = field(default=None)

    def __post_init__(self):
        """
        Compute derived parameters.
        """
        # Update derived parameters
        self.particle_diameter = 2.0 * self.particle_radius
        self.support_radius = 4.0 * self.particle_radius

    @classmethod
    def from_json(cls, scene_file_path: str):
        """
        Create SimConfig from JSON scene file.

        Args:
            scene_file_path: Path to JSON scene file

        Returns:
            SimConfig instance with parameters loaded from JSON
        """
        with open(scene_file_path, "r") as f:
            scene_data = json.load(f)

        cfg = scene_data.get("Configuration", {})

        # Parse domain
        domain_start = tuple(cfg.get("domainStart", [0.0, 0.0, 0.0]))
        domain_end = tuple(cfg.get("domainEnd", [1.0, 1.0, 1.0]))
        domain_size = tuple(e - s for s, e in zip(domain_start, domain_end))

        # Parse simulation parameters
        particle_radius = float(cfg.get("particleRadius", 0.01))
        dt = float(cfg.get("timeStepSize", 0.0016))
        density0 = float(cfg.get("density0", 1000.0))
        gravity = tuple(cfg.get("gravitation", [0.0, -9.8, 0.0]))
        viscosity_method = int(cfg.get("viscosityMethod", 0))
        surface_tension_method = int(cfg.get("surfaceTensionMethod", 0))
        viscosity_coeff = float(cfg.get("viscosityCoeff", 0.01))
        surface_tension_coeff = float(cfg.get("surfaceTensionCoeff", 0.01))

        # Create config instance
        config = cls(
            domain_size=domain_size,
            domain_start=domain_start,
            particle_radius=particle_radius,
            dt=dt,
            density0=density0,
            gravity=gravity,
            viscosity_method=viscosity_method,
            surface_tension_method=surface_tension_method,
            viscosity_coeff=viscosity_coeff,
            surface_tension_coeff=surface_tension_coeff,
            scene_data=scene_data
        )

        return config

    def get_cfg(self, name: str, default=None):
        """
        Get configuration parameter from scene data.
        
        Args:
            name: Configuration parameter name
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        if self.scene_data is None:
            return default
        return self.scene_data.get("Configuration", {}).get(name, default)

    def get_rigid_bodies(self):
        """Get rigid body definitions from scene data."""
        if self.scene_data is None:
            return []
        return self.scene_data.get("RigidBodies", [])

    def get_solid_bodies(self):
        """Get solid body definitions from scene data."""
        if self.scene_data is None:
            return []
        return self.scene_data.get("SolidBodies", [])

    def get_fluid_blocks(self):
        """Get fluid block definitions from scene data."""
        if self.scene_data is None:
            return []
        return self.scene_data.get("FluidBlocks", [])

    def get_fluid_bodies(self):
        """Get fluid body definitions from scene data."""
        if self.scene_data is None:
            return []
        return self.scene_data.get("FluidBodies", [])

    def get_scheduled_spawns(self):
        """Get scheduled spawn definitions from scene data."""
        if self.scene_data is None:
            return []
        return self.scene_data.get("ScheduledSpawns", [])

    def get_generated_bodies(self):
        """Get procedurally generated rigid body definitions."""
        if self.scene_data is None:
            return []
        return self.scene_data.get("GeneratedBodies", [])

    def get_transparent_objects(self):
        """Get list of object IDs that should be transparent (not rendered)."""
        if self.scene_data is None:
            return []
        return self.scene_data.get("Configuration", {}).get("transparentObjects", [])
