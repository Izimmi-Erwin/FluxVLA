"""Common robot interface used by simulator-specific Unitree bridge adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class RobotInterface(ABC):
    """Abstract boundary between a simulator and Unitree-compatible DDS I/O.

    The interface is intentionally small. Existing MuJoCo code still owns the
    physics/control step; this layer only describes the robot I/O contract that
    Isaac must reproduce later.
    """

    @abstractmethod
    def start(self) -> None:
        """Start robot communication resources."""

    @abstractmethod
    def stop(self) -> None:
        """Stop robot communication resources."""

    @abstractmethod
    def publish_low_state(self, obs: Mapping[str, Any]) -> None:
        """Publish body low-state from simulator observations."""

    @abstractmethod
    def receive_low_cmd(self) -> Any:
        """Return the latest low-level body command."""

    @abstractmethod
    def publish_imu(self, obs: Mapping[str, Any]) -> None:
        """Publish IMU state from simulator observations."""

    @abstractmethod
    def publish_hand_state(self, obs: Mapping[str, Any]) -> None:
        """Publish left/right hand state from simulator observations."""

    @abstractmethod
    def reset(self) -> None:
        """Reset bridge-side command/state bookkeeping."""

    def receive_left_hand_cmd(self) -> Any:
        """Return the latest left-hand command."""
        raise NotImplementedError

    def receive_right_hand_cmd(self) -> Any:
        """Return the latest right-hand command."""
        raise NotImplementedError

    def publish_odometry(self, obs: Mapping[str, Any]) -> None:
        """Publish base odometry from simulator observations."""
        raise NotImplementedError

    def publish_left_hand_state(self, obs: Mapping[str, Any]) -> None:
        """Publish left-hand state from simulator observations."""
        raise NotImplementedError

    def publish_right_hand_state(self, obs: Mapping[str, Any]) -> None:
        """Publish right-hand state from simulator observations."""
        raise NotImplementedError

    def read_joint_state(self) -> Any:
        """Read simulator joint positions, velocities, and torques."""
        raise NotImplementedError

    def apply_joint_command(self, command: Any) -> None:
        """Apply a body/hand joint command to the simulator articulation."""
        raise NotImplementedError
