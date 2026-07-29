"""Contact configuration helpers for Isaac G1 calibration.

The values here are runtime calibration presets only. They do not modify the
MuJoCo XML, Isaac USD assets, DDS protocol, or WBC code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IsaacContactConfig:
    """Runtime contact/solver parameters applied by IsaacSimulationBackend."""

    name: str
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.8
    contact_offset: float | None = None
    rest_offset: float | None = None
    solver_position_iteration_count: int | None = None
    solver_velocity_iteration_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
            "contact_offset": self.contact_offset,
            "rest_offset": self.rest_offset,
            "solver_position_iteration_count": self.solver_position_iteration_count,
            "solver_velocity_iteration_count": self.solver_velocity_iteration_count,
        }


ISAAC_DEFAULT_CONTACT_CONFIG = IsaacContactConfig(
    name="isaac_default",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.8,
    contact_offset=None,
    rest_offset=None,
    solver_position_iteration_count=None,
    solver_velocity_iteration_count=None,
)


MUJOCO_EQUIVALENT_CONTACT_CONFIG = IsaacContactConfig(
    name="mujoco_equivalent",
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
    contact_offset=0.02,
    rest_offset=0.001,
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=4,
)


HIGH_SOLVER_CONTACT_CONFIG = IsaacContactConfig(
    name="mujoco_high_solver",
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
    contact_offset=0.02,
    rest_offset=0.001,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=8,
)


CONTACT_PRESETS: dict[str, IsaacContactConfig] = {
    "default": ISAAC_DEFAULT_CONTACT_CONFIG,
    "mujoco": MUJOCO_EQUIVALENT_CONTACT_CONFIG,
    "high_solver": HIGH_SOLVER_CONTACT_CONFIG,
}


def build_contact_config(
    *,
    preset: str = "default",
    friction: float | None = None,
    static_friction: float | None = None,
    dynamic_friction: float | None = None,
    restitution: float | None = None,
    contact_offset: float | None = None,
    rest_offset: float | None = None,
    solver_position_iteration_count: int | None = None,
    solver_velocity_iteration_count: int | None = None,
) -> IsaacContactConfig:
    """Build a contact config from a named preset plus optional overrides."""

    if preset not in CONTACT_PRESETS:
        raise ValueError(f"unknown contact preset {preset!r}; options={sorted(CONTACT_PRESETS)}")
    base = CONTACT_PRESETS[preset]
    static = static_friction if static_friction is not None else base.static_friction
    dynamic = dynamic_friction if dynamic_friction is not None else base.dynamic_friction
    if friction is not None:
        static = friction
        dynamic = friction
    return IsaacContactConfig(
        name=base.name if friction is None else f"{base.name}_friction_{friction:g}",
        static_friction=float(static),
        dynamic_friction=float(dynamic),
        restitution=base.restitution if restitution is None else float(restitution),
        contact_offset=base.contact_offset if contact_offset is None else float(contact_offset),
        rest_offset=base.rest_offset if rest_offset is None else float(rest_offset),
        solver_position_iteration_count=(
            base.solver_position_iteration_count
            if solver_position_iteration_count is None
            else int(solver_position_iteration_count)
        ),
        solver_velocity_iteration_count=(
            base.solver_velocity_iteration_count
            if solver_velocity_iteration_count is None
            else int(solver_velocity_iteration_count)
        ),
    )
