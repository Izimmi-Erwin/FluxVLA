"""Robot interface adapters for simulator-backed Unitree DDS bridges."""

from gear_sonic.robot_interface.base import RobotInterface

__all__ = [
    "RobotInterface",
    "IsaacUnitreeBridge",
]


def __getattr__(name):
    if name == "IsaacUnitreeBridge":
        from gear_sonic.robot_interface.isaac_unitree_bridge import IsaacUnitreeBridge

        return IsaacUnitreeBridge
    raise AttributeError(name)
