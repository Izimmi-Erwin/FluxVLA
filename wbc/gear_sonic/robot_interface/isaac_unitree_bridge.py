"""Unitree DDS bridge for the Isaac Sonic backend.

This bridge only replaces the simulator-side Unitree DDS I/O.  It does not
start FluxVLA, cameras, WBC processes, or evaluation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterable, Mapping

import numpy as np

from gear_sonic.robot_interface.base import RobotInterface
from gear_sonic.robot_interface.joint_mapping import SONIC_BODY_JOINT_NAMES


DEFAULT_UNITREE_SDK_PATH = Path("/home/limx/Erwin/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python")
DEFAULT_DDS_SITE_PACKAGES = Path("/home/limx/miniconda3/envs/fluxvla/lib/python3.10/site-packages")


@dataclass
class IsaacLowCommand:
    """Decoded 29-DOF body command from ``rt/lowcmd`` in Sonic body order."""

    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray
    mode: np.ndarray | None = None
    reserve: np.ndarray | None = None
    mode_pr: int | None = None
    mode_machine: int | None = None
    message_reserve: list[int] = field(default_factory=list)
    crc: int | None = None
    received_action_dim: int = 29
    body_dim: int = 29
    hand_dim: int = 0
    received_time: float = field(default_factory=time.time)
    sequence_id: int = 0


@dataclass
class IsaacDdsBridgeStats:
    publish_count: int = 0
    imu_publish_count: int = 0
    odometry_publish_count: int = 0
    receive_count: int = 0
    apply_count: int = 0
    last_publish_time: float | None = None
    last_receive_time: float | None = None
    last_apply_time: float | None = None
    last_error: str = ""
    last_received_action_dim: int = 0
    last_applied_joint_dim: int = 0
    first_receive_time: float | None = None
    first_apply_time: float | None = None


class IsaacUnitreeBridge(RobotInterface):
    """DDS bridge that exposes Isaac articulation state as Unitree low-level I/O."""

    def __init__(
        self,
        articulation: Any | None = None,
        joint_mapping: Any | None = None,
        *,
        domain_id: int = 0,
        network_interface: str | None = None,
        num_body_motor: int = 29,
        motor_mode: int = 1,
        default_temperature: int = 0,
        unitree_sdk_path: str | Path | None = DEFAULT_UNITREE_SDK_PATH,
        dds_site_packages: str | Path | None = DEFAULT_DDS_SITE_PACKAGES,
        log_dir: str | Path | None = None,
        verbose: bool = True,
        performance_monitor: Any | None = None,
        receive_log_interval: int = 500,
        apply_log_interval: int = 500,
        configure_drives_on_start: bool = True,
        drive_kp_scale: float = 1.0,
        drive_kd_scale: float = 1.0,
        control_mode: str = "position",
        robot_asset: Any | None = None,
    ):
        self.articulation = articulation
        self.joint_mapping = joint_mapping
        self.domain_id = domain_id
        self.network_interface = network_interface
        self.num_body_motor = num_body_motor
        self.motor_mode = motor_mode
        self.default_temperature = default_temperature
        self.unitree_sdk_path = Path(unitree_sdk_path).expanduser() if unitree_sdk_path else None
        self.dds_site_packages = Path(dds_site_packages).expanduser() if dds_site_packages else None
        self.log_dir = Path(log_dir).expanduser().resolve() if log_dir else None
        self.verbose = verbose
        self.performance_monitor = performance_monitor
        self.receive_log_interval = max(1, int(receive_log_interval))
        self.apply_log_interval = max(1, int(apply_log_interval))
        self.configure_drives_on_start = configure_drives_on_start
        self.drive_kp_scale = float(drive_kp_scale)
        self.drive_kd_scale = float(drive_kd_scale)
        self.control_mode = str(control_mode).strip().lower()
        if self.control_mode not in ("position", "effort"):
            raise ValueError(f"unsupported Isaac control_mode={control_mode!r}; expected position or effort")
        self.robot_asset = robot_asset

        self.stats = IsaacDdsBridgeStats()
        self.low_cmd: Any | None = None
        self.latest_low_command: IsaacLowCommand | None = None
        self._new_low_cmd = False
        self._low_cmd_received = False
        self._low_cmd_lock = threading.Lock()
        self._low_cmd_sequence = 0
        self._last_latency_sequence = -1
        self._started = False

        self.low_state: Any | None = None
        self.odo_state: Any | None = None
        self.secondary_imu_state: Any | None = None
        self.low_state_puber: Any | None = None
        self.odo_state_puber: Any | None = None
        self.secondary_imu_puber: Any | None = None
        self.low_cmd_suber: Any | None = None
        self.left_hand_state: Any | None = None
        self.right_hand_state: Any | None = None
        self.left_hand_state_puber: Any | None = None
        self.right_hand_state_puber: Any | None = None
        self.left_hand_cmd: Any | None = None
        self.right_hand_cmd: Any | None = None
        self.left_hand_cmd_suber: Any | None = None
        self.right_hand_cmd_suber: Any | None = None
        self.num_hand_motor = 7
        self._left_hand_cmd_lock = threading.Lock()
        self._right_hand_cmd_lock = threading.Lock()
        self.left_hand_cmd_received = False
        self.right_hand_cmd_received = False
        self.new_left_hand_cmd = False
        self.new_right_hand_cmd = False
        self.hand_state_publish_count = 0

        self._LowStateType = None
        self._OdoStateType = None
        self._IMUStateType = None
        self._LowCmdType = None
        self._HandStateType = None
        self._HandCmdType = None
        self._ChannelFactoryInitialize = None
        self._ChannelPublisher = None
        self._ChannelSubscriber = None
        self._ArticulationActionType = None
        self._drives_configured = False
        self._timeline_header_written = False
        self._lowcmd_snapshot_header_written = False
        self._effort_command_header_written = False
        self._lowstate_state_header_written = False
        self._imu_angular_velocity_header_written = False
        self._standing_pose_error_header_written = False
        self._contact_views: dict[str, Any] = {}
        self._contact_error = ""
        self.last_command_tau_by_isaac: np.ndarray | None = None
        self.last_applied_tau_by_isaac: np.ndarray | None = None
        self.last_published_lowstate: dict[str, Any] | None = None
        self.last_imu_angular_velocity: dict[str, Any] | None = None
        self._joint_mapping_validated = False
        self.startup_state_snapshots: list[dict[str, Any]] = []

    def start(self) -> None:
        """Initialize Unitree DDS publishers/subscriber."""

        if self._started:
            return
        self._load_unitree_dds()

        assert self._ChannelFactoryInitialize is not None
        assert self._ChannelPublisher is not None
        assert self._ChannelSubscriber is not None
        assert self._LowStateType is not None
        assert self._OdoStateType is not None
        assert self._IMUStateType is not None
        assert self._LowCmdType is not None
        assert self._HandStateType is not None
        assert self._HandCmdType is not None

        if self.network_interface:
            init_ok = self._ChannelFactoryInitialize(self.domain_id, self.network_interface)
        else:
            init_ok = self._ChannelFactoryInitialize(self.domain_id)
        if init_ok is False:
            raise RuntimeError(
                f"ChannelFactoryInitialize failed: domain_id={self.domain_id} "
                f"network_interface={self.network_interface!r}"
            )

        self.low_state_puber = self._ChannelPublisher("rt/lowstate", self._LowStateType)
        self.low_state_puber.Init()
        self.odo_state_puber = self._ChannelPublisher("rt/odostate", self._OdoStateType)
        self.odo_state_puber.Init()
        self.secondary_imu_puber = self._ChannelPublisher("rt/secondary_imu", self._IMUStateType)
        self.secondary_imu_puber.Init()
        self.left_hand_state_puber = self._ChannelPublisher("rt/dex3/left/state", self._HandStateType)
        self.left_hand_state_puber.Init()
        self.right_hand_state_puber = self._ChannelPublisher("rt/dex3/right/state", self._HandStateType)
        self.right_hand_state_puber.Init()

        self.low_cmd_suber = self._ChannelSubscriber("rt/lowcmd", self._LowCmdType)
        self.low_cmd_suber.Init(self._low_cmd_handler, 1)
        self.left_hand_cmd_suber = self._ChannelSubscriber("rt/dex3/left/cmd", self._HandCmdType)
        self.left_hand_cmd_suber.Init(self._left_hand_cmd_handler, 1)
        self.right_hand_cmd_suber = self._ChannelSubscriber("rt/dex3/right/cmd", self._HandCmdType)
        self.right_hand_cmd_suber.Init(self._right_hand_cmd_handler, 1)

        if self.articulation is not None and self.joint_mapping is not None:
            self.validate_joint_mapping_or_raise()

        if self.configure_drives_on_start and self.articulation is not None and self.joint_mapping is not None:
            self.configure_joint_drives(kp_scale=self.drive_kp_scale, kd_scale=self.drive_kd_scale)

        self._started = True
        self._log(
            "dds_publish.log",
            "DDS publishers initialized: rt/lowstate, rt/odostate, rt/secondary_imu, "
            "rt/dex3/left/state, rt/dex3/right/state",
        )
        self._log("dds_receive.log", "DDS subscribers initialized: rt/lowcmd, rt/dex3/left/cmd, rt/dex3/right/cmd")
        self.write_lowcmd_field_usage_report()
        hand_available = bool(getattr(self.robot_asset, "hand_dof_available", False))
        hand_count = len(getattr(self.robot_asset, "hand_joint_names", ())) if self.robot_asset is not None else 0
        self._log(
            "dds_dimensions.log",
            (
                f"body_motor_count={self.num_body_motor} lowstate_body_slots={self.num_body_motor} "
                f"lowcmd_body_slots={self.num_body_motor} hand_dof_available={hand_available} "
                f"hand_dof_count={hand_count} dex3_state_slots={self.num_hand_motor} "
                "hand_state_source=zero_when_usd_has_no_hand_dofs"
            ),
        )

    def stop(self) -> None:
        for endpoint in (
            self.low_cmd_suber,
            self.left_hand_cmd_suber,
            self.right_hand_cmd_suber,
            self.low_state_puber,
            self.odo_state_puber,
            self.secondary_imu_puber,
            self.left_hand_state_puber,
            self.right_hand_state_puber,
        ):
            close = getattr(endpoint, "Close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    self.stats.last_error = repr(exc)
        self._started = False

    def reset(self) -> None:
        with self._low_cmd_lock:
            self.low_cmd = None
            self.latest_low_command = None
            self._new_low_cmd = False
            self._low_cmd_received = False
            self._low_cmd_sequence = 0
            self._last_latency_sequence = -1
        with self._left_hand_cmd_lock:
            self.left_hand_cmd = None
            self.left_hand_cmd_received = False
            self.new_left_hand_cmd = False
        with self._right_hand_cmd_lock:
            self.right_hand_cmd = None
            self.right_hand_cmd_received = False
            self.new_right_hand_cmd = False
        self.hand_state_publish_count = 0
        self.stats = IsaacDdsBridgeStats()

    def peek_low_cmd(self) -> IsaacLowCommand | None:
        """Return the latest command without consuming the new-command flag."""

        with self._low_cmd_lock:
            return self.latest_low_command

    def receive_low_cmd(self) -> IsaacLowCommand | None:
        """Return and log the newest 29-body low command, if one arrived."""

        with self._low_cmd_lock:
            if not self._new_low_cmd:
                return None
            command = self.latest_low_command
            self._new_low_cmd = False

        if command is None:
            return None

        if self.stats.receive_count == 1 or self.stats.receive_count % self.receive_log_interval == 0:
            self._log(
                "dds_receive.log",
                (
                    f"receive_count={self.stats.receive_count} sequence_id={command.sequence_id} "
                    f"received_action_dim={command.received_action_dim} "
                    f"body_dim={command.body_dim} hand_dim={command.hand_dim} "
                    f"target_position={_preview(command.q)} "
                    f"kp={_preview(command.kp)} kd={_preview(command.kd)} torque={_preview(command.tau)}"
                ),
            )
        return command

    def receive_left_hand_cmd(self) -> Any:
        with self._left_hand_cmd_lock:
            return self.left_hand_cmd

    def receive_right_hand_cmd(self) -> Any:
        with self._right_hand_cmd_lock:
            return self.right_hand_cmd

    def publish_low_state(self, obs: Mapping[str, Any]) -> None:
        """Publish Unitree ``rt/lowstate`` from an Isaac articulation state."""

        self._require_started()
        self._ensure_messages()

        joint_position_isaac = _float_array(obs.get("joint_position", []), self.num_body_motor)
        joint_velocity_isaac = _float_array(obs.get("joint_velocity", []), self.num_body_motor)
        # TODO(isaac-dds): replace zeros when a validated Isaac torque read API is selected.
        joint_torque_isaac = _float_array(obs.get("joint_torque", []), self.num_body_motor, default=0.0)
        joint_position = self._isaac_to_unitree_body_order(joint_position_isaac, label="joint_position")
        joint_velocity = self._isaac_to_unitree_body_order(joint_velocity_isaac, label="joint_velocity")
        joint_torque = self._isaac_to_unitree_body_order(joint_torque_isaac, label="joint_torque")
        tick = _tick_from_obs(obs)
        omega_world, omega_body = _root_angular_velocity_world_body(obs)

        for index in range(self.num_body_motor):
            motor_state = self.low_state.motor_state[index]
            motor_state.mode = int(self.motor_mode)
            motor_state.q = float(joint_position[index])
            motor_state.dq = float(joint_velocity[index])
            motor_state.ddq = 0.0
            motor_state.tau_est = float(joint_torque[index])
            _assign_temperature(motor_state, self.default_temperature)

        self.last_published_lowstate = {
            "time": float(obs.get("time", float("nan"))),
            "joint_position": joint_position.copy(),
            "joint_velocity": joint_velocity.copy(),
            "joint_torque": joint_torque.copy(),
            "joint_position_isaac": joint_position_isaac.copy(),
            "joint_velocity_isaac": joint_velocity_isaac.copy(),
            "joint_torque_isaac": joint_torque_isaac.copy(),
            "root_position": _float_array(obs.get("root_position", []), 3, default=float("nan")),
            "root_quaternion": _float_array(obs.get("root_quaternion", []), 4, default=float("nan")),
            "root_linear_velocity": _float_array(obs.get("root_linear_velocity", []), 3, default=float("nan")),
            "root_angular_velocity": omega_body.copy(),
            "root_angular_velocity_frame": "body",
            "root_angular_velocity_world": omega_world.copy(),
            "root_angular_velocity_body": omega_body.copy(),
            "tick": tick,
            "publish_count": self.stats.publish_count + 1,
        }

        self.low_state.tick = tick
        self.publish_imu(obs)
        self.publish_odometry(obs)
        ok = self.low_state_puber.Write(self.low_state)
        if not ok:
            raise RuntimeError("failed to publish rt/lowstate")

        self.stats.publish_count += 1
        self.stats.last_publish_time = time.time()
        if self.stats.publish_count == 1 or self.stats.publish_count % 500 == 0:
            self._log(
                "dds_publish.log",
                (
                    f"lowstate publish_count={self.stats.publish_count} tick={tick} "
                    f"unitree_order_q={_preview(joint_position)} unitree_order_dq={_preview(joint_velocity)} "
                    "tau_est=default_zero TODO(read Isaac joint torque)"
                ),
            )
        self.publish_hand_state(obs)

    def _isaac_to_unitree_body_order(self, values_by_isaac: np.ndarray, *, label: str) -> np.ndarray:
        if self.joint_mapping is None:
            raise RuntimeError("joint_mapping is required before publishing Unitree lowstate")
        values = np.asarray(values_by_isaac, dtype=np.float32).reshape(-1)
        required_isaac_dof = max(self.joint_mapping.sonic_body_to_isaac_dof) + 1
        if values.size < required_isaac_dof:
            raise RuntimeError(
                f"{label} has too few Isaac DOF values for name-based Unitree mapping: "
                f"values={values.size} required={required_isaac_dof}"
            )
        return values[np.asarray(self.joint_mapping.sonic_body_to_isaac_dof, dtype=np.int64)].astype(np.float32)

    def publish_imu(self, obs: Mapping[str, Any]) -> None:
        """Publish ``rt/secondary_imu`` and mirror it into ``low_state.imu_state``."""

        self._require_started()
        self._ensure_messages()

        quaternion = _float_array(obs.get("root_quaternion", [1.0, 0.0, 0.0, 0.0]), 4)
        omega_world, omega_body = _root_angular_velocity_world_body(obs)
        # Isaac exposes root linear velocity here, not linear acceleration.
        # TODO(isaac-dds): publish measured/validated linear acceleration if a sensor or API is added.
        accelerometer = _float_array(obs.get("root_linear_acceleration", [0.0, 0.0, 0.0]), 3)

        self.secondary_imu_state.quaternion[:] = quaternion.tolist()
        self.secondary_imu_state.gyroscope[:] = omega_world.tolist()
        self.secondary_imu_state.accelerometer[:] = accelerometer.tolist()
        self.secondary_imu_state.temperature = int(self.default_temperature)

        self.low_state.imu_state.quaternion[:] = quaternion.tolist()
        self.low_state.imu_state.gyroscope[:] = omega_body.tolist()
        self.low_state.imu_state.accelerometer[:] = accelerometer.tolist()
        self.low_state.imu_state.temperature = int(self.default_temperature)

        self.last_imu_angular_velocity = {
            "time": float(obs.get("time", float("nan"))),
            "frame_published_to_dds": "body",
            "omega_world": omega_world.copy(),
            "omega_body": omega_body.copy(),
        }
        self._record_imu_angular_velocity_frame(obs, quaternion, omega_world, omega_body)

        ok = self.secondary_imu_puber.Write(self.secondary_imu_state)
        if not ok:
            raise RuntimeError("failed to publish rt/secondary_imu")
        self.stats.imu_publish_count += 1

    def publish_odometry(self, obs: Mapping[str, Any]) -> None:
        self._require_started()
        self._ensure_messages()

        tick = _tick_from_obs(obs)
        self.odo_state.tick = tick
        self.odo_state.position[:] = _float_array(obs.get("root_position", [0.0, 0.0, 0.0]), 3).tolist()
        self.odo_state.orientation[:] = _float_array(obs.get("root_quaternion", [1.0, 0.0, 0.0, 0.0]), 4).tolist()
        self.odo_state.linear_velocity[:] = _float_array(obs.get("root_linear_velocity", [0.0, 0.0, 0.0]), 3).tolist()
        self.odo_state.angular_velocity[:] = _float_array(obs.get("root_angular_velocity", [0.0, 0.0, 0.0]), 3).tolist()
        ok = self.odo_state_puber.Write(self.odo_state)
        if not ok:
            raise RuntimeError("failed to publish rt/odostate")
        self.stats.odometry_publish_count += 1

    def _record_imu_angular_velocity_frame(
        self,
        obs: Mapping[str, Any],
        quaternion: np.ndarray,
        omega_world: np.ndarray,
        omega_body: np.ndarray,
    ) -> None:
        path = self._log_path("imu_angular_velocity_frame.csv")
        if path is None:
            return
        publish_index = int(self.stats.imu_publish_count + 1)
        if publish_index != 1 and publish_index % 5 != 0:
            return
        row = {
            "timestamp": f"{time.time():.9f}",
            "simulation_time": f"{float(obs.get('time', float('nan'))):.9f}",
            "publish_index": publish_index,
            "source": "Isaac articulation.get_angular_velocity()",
            "source_frame": "world",
            "published_frame": "body",
            "lowstate_published_frame": "body",
            "secondary_imu_published_frame": "world",
            "quaternion_format": "wxyz",
            "quaternion_w": float(quaternion[0]) if quaternion.shape[0] >= 4 else float("nan"),
            "quaternion_x": float(quaternion[1]) if quaternion.shape[0] >= 4 else float("nan"),
            "quaternion_y": float(quaternion[2]) if quaternion.shape[0] >= 4 else float("nan"),
            "quaternion_z": float(quaternion[3]) if quaternion.shape[0] >= 4 else float("nan"),
            "omega_world_x": float(omega_world[0]),
            "omega_world_y": float(omega_world[1]),
            "omega_world_z": float(omega_world[2]),
            "omega_world_norm": float(np.linalg.norm(omega_world)),
            "omega_body_x": float(omega_body[0]),
            "omega_body_y": float(omega_body[1]),
            "omega_body_z": float(omega_body[2]),
            "omega_body_norm": float(np.linalg.norm(omega_body)),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or not self._imu_angular_velocity_header_written
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if write_header:
                writer.writeheader()
                self._imu_angular_velocity_header_written = True
            writer.writerow(row)
        if publish_index == 1 or publish_index % 500 == 0:
            self._log(
                "dds_publish.log",
                (
                    "imu_angular_velocity_frame "
                    "source_frame=world published_frame=body "
                    f"omega_world={_preview(omega_world, limit=3)} "
                    f"omega_body={_preview(omega_body, limit=3)}"
                ),
            )

    def publish_hand_state(self, obs: Mapping[str, Any]) -> None:
        self.publish_left_hand_state(obs)
        self.publish_right_hand_state(obs)
        self.hand_state_publish_count += 1
        if self.hand_state_publish_count == 1 or self.hand_state_publish_count % 500 == 0:
            left_q = _float_array(obs.get("left_hand_q", []), self.num_hand_motor, default=0.0)
            right_q = _float_array(obs.get("right_hand_q", []), self.num_hand_motor, default=0.0)
            self._log(
                "dds_publish.log",
                (
                    f"dex3_state publish_count={self.hand_state_publish_count} "
                    f"left_q={_preview(left_q)} right_q={_preview(right_q)} "
                    "source=obs_or_zero_for_29dof_usd"
                ),
            )

    def publish_left_hand_state(self, obs: Mapping[str, Any]) -> None:
        self._require_started()
        self._ensure_messages()
        left_q = _float_array(obs.get("left_hand_q", []), self.num_hand_motor, default=0.0)
        left_dq = _float_array(obs.get("left_hand_dq", []), self.num_hand_motor, default=0.0)
        left_ddq = _float_array(obs.get("left_hand_ddq", []), self.num_hand_motor, default=0.0)
        left_tau = _float_array(obs.get("left_hand_tau_est", []), self.num_hand_motor, default=0.0)
        for i in range(self.num_hand_motor):
            motor_state = self.left_hand_state.motor_state[i]
            motor_state.q = float(left_q[i])
            motor_state.dq = float(left_dq[i])
            motor_state.ddq = float(left_ddq[i])
            motor_state.tau_est = float(left_tau[i])
            _assign_temperature(motor_state, self.default_temperature)
        ok = self.left_hand_state_puber.Write(self.left_hand_state)
        if not ok:
            raise RuntimeError("failed to publish rt/dex3/left/state")

    def publish_right_hand_state(self, obs: Mapping[str, Any]) -> None:
        self._require_started()
        self._ensure_messages()
        right_q = _float_array(obs.get("right_hand_q", []), self.num_hand_motor, default=0.0)
        right_dq = _float_array(obs.get("right_hand_dq", []), self.num_hand_motor, default=0.0)
        right_ddq = _float_array(obs.get("right_hand_ddq", []), self.num_hand_motor, default=0.0)
        right_tau = _float_array(obs.get("right_hand_tau_est", []), self.num_hand_motor, default=0.0)
        for i in range(self.num_hand_motor):
            motor_state = self.right_hand_state.motor_state[i]
            motor_state.q = float(right_q[i])
            motor_state.dq = float(right_dq[i])
            motor_state.ddq = float(right_ddq[i])
            motor_state.tau_est = float(right_tau[i])
            _assign_temperature(motor_state, self.default_temperature)
        ok = self.right_hand_state_puber.Write(self.right_hand_state)
        if not ok:
            raise RuntimeError("failed to publish rt/dex3/right/state")

    def read_joint_state(self) -> Any:
        if self.articulation is None:
            raise RuntimeError("Isaac articulation is not attached to IsaacUnitreeBridge")
        return {
            "joint_position": _to_list(self.articulation.get_joint_positions()),
            "joint_velocity": _to_list(self.articulation.get_joint_velocities()),
            "root_position": _to_list(self.articulation.get_world_pose()[0]),
            "root_quaternion": _to_list(self.articulation.get_world_pose()[1]),
            "root_linear_velocity": _to_list(self.articulation.get_linear_velocity()),
            "root_angular_velocity": _to_list(self.articulation.get_angular_velocity()),
        }

    def apply_joint_command(self, command: Any) -> None:
        """Forward lowcmd to Isaac using the configured control mode."""

        if command is None:
            return
        if self.articulation is None:
            raise RuntimeError("Isaac articulation is not attached to IsaacUnitreeBridge")
        if self.joint_mapping is None:
            raise RuntimeError("joint_mapping is required before applying Isaac lowcmd")

        applied_joint_dim = len(self.joint_mapping.sonic_body_to_isaac_dof)
        if self.control_mode == "effort":
            joint_q = np.asarray(self.articulation.get_joint_positions(), dtype=np.float32).copy()
            joint_dq = np.asarray(self.articulation.get_joint_velocities(), dtype=np.float32).copy()
            command_tau_by_isaac = self._compute_isaac_effort_command(command, joint_q, joint_dq)
            self._apply_isaac_joint_efforts(command_tau_by_isaac)
            self.last_command_tau_by_isaac = command_tau_by_isaac.copy()
            self.last_applied_tau_by_isaac = command_tau_by_isaac.copy()
            self._record_effort_command(command, joint_q, joint_dq, command_tau_by_isaac)
        else:
            target_by_isaac = np.asarray(self.articulation.get_joint_positions(), dtype=np.float32).copy()
            if target_by_isaac.shape[0] < self.num_body_motor:
                raise RuntimeError(f"Isaac articulation has too few DOFs: {target_by_isaac.shape[0]}")

            for sonic_index, isaac_index in enumerate(self.joint_mapping.sonic_body_to_isaac_dof):
                target_by_isaac[isaac_index] = float(command.q[sonic_index])

            self._apply_isaac_position_targets(target_by_isaac)
            self.last_command_tau_by_isaac = None
            self.last_applied_tau_by_isaac = None
        self.stats.apply_count += 1
        self.stats.last_applied_joint_dim = int(applied_joint_dim)
        self.stats.last_apply_time = time.time()
        if self.stats.first_apply_time is None:
            self.stats.first_apply_time = self.stats.last_apply_time
        if self.performance_monitor is not None:
            count_latency = command.sequence_id != self._last_latency_sequence
            self.performance_monitor.record_apply(
                receive_time=command.received_time if count_latency else None,
                apply_time=self.stats.last_apply_time,
            )
            if count_latency:
                self._last_latency_sequence = command.sequence_id
        if self.stats.apply_count == 1 or self.stats.apply_count % self.apply_log_interval == 0:
            self._log(
                "joint_command.log",
                (
                    f"apply_count={self.stats.apply_count} sequence_id={command.sequence_id} "
                    f"control_mode={self.control_mode} "
                    f"received_action_dim={getattr(command, 'received_action_dim', len(command.q))} "
                    f"applied_joint_dim={applied_joint_dim} "
                    f"lowcmd_mode={_preview(getattr(command, 'mode', []))} "
                    f"target_position_sonic={_preview(command.q)} "
                    f"command_tau={_preview(self.last_command_tau_by_isaac)} "
                    f"applied_tau={_preview(self.last_applied_tau_by_isaac)}"
                ),
            )

    def _apply_isaac_position_targets(self, target_by_isaac: np.ndarray) -> None:
        """Apply full-DOF position targets using the Isaac articulation API."""

        articulation_view = getattr(self.articulation, "_articulation_view", None)
        view_set_targets = getattr(articulation_view, "set_joint_position_targets", None)
        if view_set_targets is not None:
            view_set_targets(np.expand_dims(target_by_isaac, axis=0))
            return

        set_targets = getattr(self.articulation, "set_joint_position_targets", None)
        if set_targets is not None:
            set_targets(target_by_isaac)
            return

        apply_action = getattr(self.articulation, "apply_action", None)
        if apply_action is not None:
            if self._ArticulationActionType is None:
                try:
                    from isaacsim.core.utils.types import ArticulationAction
                except ModuleNotFoundError:
                    from omni.isaac.core.utils.types import ArticulationAction
                self._ArticulationActionType = ArticulationAction

            apply_action(self._ArticulationActionType(joint_positions=target_by_isaac))
            return

        raise RuntimeError(
            "Isaac articulation does not expose apply_action or set_joint_position_targets"
        )

    def _compute_isaac_effort_command(
        self,
        command: IsaacLowCommand,
        joint_q: np.ndarray,
        joint_dq: np.ndarray,
    ) -> np.ndarray:
        if joint_q.shape[0] < self.num_body_motor or joint_dq.shape[0] < self.num_body_motor:
            raise RuntimeError(
                f"Isaac articulation has too few DOFs for effort control: "
                f"q={joint_q.shape[0]} dq={joint_dq.shape[0]} required={self.num_body_motor}"
            )
        command_tau_by_isaac = np.zeros_like(joint_q, dtype=np.float32)
        for sonic_index, isaac_index in enumerate(self.joint_mapping.sonic_body_to_isaac_dof):
            q_des = float(command.q[sonic_index])
            dq_des = float(command.dq[sonic_index])
            kp = float(command.kp[sonic_index])
            kd = float(command.kd[sonic_index])
            tau_ff = float(command.tau[sonic_index])
            q = float(joint_q[isaac_index])
            dq = float(joint_dq[isaac_index])
            command_tau_by_isaac[isaac_index] = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
        return command_tau_by_isaac

    def _apply_isaac_joint_efforts(self, effort_by_isaac: np.ndarray) -> None:
        articulation_view = getattr(self.articulation, "_articulation_view", None)
        view_set_efforts = getattr(articulation_view, "set_joint_efforts", None)
        if view_set_efforts is None:
            raise RuntimeError("Isaac articulation does not expose _articulation_view.set_joint_efforts")
        view_set_efforts(np.expand_dims(np.asarray(effort_by_isaac, dtype=np.float32), axis=0))

    def _record_effort_command(
        self,
        command: IsaacLowCommand,
        joint_q: np.ndarray,
        joint_dq: np.ndarray,
        effort_by_isaac: np.ndarray,
    ) -> None:
        path = self._log_path("joint_effort_command.csv")
        if path is None:
            return
        row: dict[str, Any] = {
            "timestamp": f"{time.time():.9f}",
            "apply_count": int(self.stats.apply_count + 1),
            "sequence_id": int(command.sequence_id),
            "control_mode": self.control_mode,
            "received_action_dim": int(getattr(command, "received_action_dim", len(command.q))),
            "applied_joint_dim": int(len(self.joint_mapping.sonic_body_to_isaac_dof)),
        }
        for sonic_index, isaac_index in enumerate(self.joint_mapping.sonic_body_to_isaac_dof):
            joint_name = self.joint_mapping.sonic_body_joint_names[sonic_index]
            row[f"{joint_name}.joint_q"] = float(joint_q[isaac_index])
            row[f"{joint_name}.joint_dq"] = float(joint_dq[isaac_index])
            row[f"{joint_name}.command_tau"] = float(effort_by_isaac[isaac_index])
            row[f"{joint_name}.applied_tau"] = float(effort_by_isaac[isaac_index])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or not self._effort_command_header_written
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if write_header:
                writer.writeheader()
                self._effort_command_header_written = True
            writer.writerow(row)

    def configure_joint_drives(
        self,
        *,
        kp_scale: float | None = None,
        kd_scale: float | None = None,
        force: bool = False,
    ) -> list[Any]:
        """Configure Isaac articulation drive gains from Sonic PD mapping."""

        if self.articulation is None:
            raise RuntimeError("Isaac articulation is not attached to IsaacUnitreeBridge")
        if self.joint_mapping is None:
            raise RuntimeError("joint_mapping is required before configuring Isaac drives")
        if self._drives_configured and not force:
            return []

        from gear_sonic.robot_interface.pd_mapping import build_pd_mapping

        kp_scale_value = self.drive_kp_scale if kp_scale is None else float(kp_scale)
        kd_scale_value = self.drive_kd_scale if kd_scale is None else float(kd_scale)
        rows = build_pd_mapping(
            self.articulation.dof_names,
            kp_scale=kp_scale_value,
            kd_scale=kd_scale_value,
        )

        articulation_view = getattr(self.articulation, "_articulation_view", None)
        if articulation_view is None:
            raise RuntimeError("Isaac articulation does not expose _articulation_view")

        old_kps, old_kds = articulation_view.get_gains()
        old_efforts = articulation_view.get_max_efforts()
        old_kps = np.asarray(old_kps, dtype=np.float32)
        old_kds = np.asarray(old_kds, dtype=np.float32)
        old_efforts = np.asarray(old_efforts, dtype=np.float32)

        joint_indices = np.asarray([row.isaac_index for row in rows], dtype=np.int64)
        new_kps = np.asarray([[row.kp for row in rows]], dtype=np.float32)
        new_kds = np.asarray([[row.kd for row in rows]], dtype=np.float32)
        new_efforts = np.asarray([[row.effort_limit for row in rows]], dtype=np.float32)

        self._log(
            "isaac_pd_after.log",
            f"configure_joint_drives kp_scale={kp_scale_value} kd_scale={kd_scale_value}",
        )
        for local_index, row in enumerate(rows):
            isaac_index = row.isaac_index
            message = (
                f"joint={row.isaac_joint} old_stiffness={float(old_kps[0, isaac_index]):.6f} "
                f"new_stiffness={row.kp:.6f} old_damping={float(old_kds[0, isaac_index]):.6f} "
                f"new_damping={row.kd:.6f} old_effort={float(old_efforts[0, isaac_index]):.6f} "
                f"new_effort={row.effort_limit:.6f}"
            )
            self._log("isaac_pd_after.log", message)
            if local_index < 6:
                print(f"[IsaacUnitreeBridge] {message}")

        articulation_view.set_gains(kps=new_kps, kds=new_kds, joint_indices=joint_indices)
        articulation_view.set_max_efforts(new_efforts, joint_indices=joint_indices)
        self._drives_configured = True
        return rows

    def read_joint_drive_parameters(self) -> list[dict[str, Any]]:
        """Read current Isaac drive gains/limits for all articulation DOFs."""

        if self.articulation is None:
            raise RuntimeError("Isaac articulation is not attached to IsaacUnitreeBridge")

        articulation_view = getattr(self.articulation, "_articulation_view", None)
        if articulation_view is None:
            raise RuntimeError("Isaac articulation does not expose _articulation_view")

        kps, kds = articulation_view.get_gains()
        efforts = articulation_view.get_max_efforts()
        drive_types = getattr(articulation_view, "get_drive_types", lambda: None)()
        dof_types = getattr(articulation_view, "get_dof_types", lambda: None)()

        kps = np.asarray(kps, dtype=np.float32)
        kds = np.asarray(kds, dtype=np.float32)
        efforts = np.asarray(efforts, dtype=np.float32)
        drive_types_array = None if drive_types is None else np.asarray(drive_types)

        rows: list[dict[str, Any]] = []
        for index, name in enumerate(self.articulation.dof_names):
            rows.append(
                {
                    "index": index,
                    "joint": name,
                    "stiffness": float(kps[0, index]),
                    "damping": float(kds[0, index]),
                    "max_effort": float(efforts[0, index]),
                    "drive_type": _drive_type_at(drive_types_array, index),
                    "dof_type": _dof_type_at(dof_types, index),
                }
            )
        return rows

    def log_joint_drive_parameters(self, filename: str = "isaac_pd_before.log") -> list[dict[str, Any]]:
        """Write current Isaac drive gains/limits to a bridge log file."""

        rows = self.read_joint_drive_parameters()
        self._log(filename, "index joint stiffness damping max_effort drive_type dof_type")
        for row in rows:
            self._log(
                filename,
                (
                    f"{row['index']} {row['joint']} {row['stiffness']:.6f} "
                    f"{row['damping']:.6f} {row['max_effort']:.6f} "
                    f"{row['drive_type']} {row['dof_type']}"
                ),
            )
        return rows

    def write_drive_parameters_csv(self, filename: str = "isaac_drive_parameters.csv") -> list[dict[str, Any]]:
        rows = self.read_joint_drive_parameters()
        path = self._log_path(filename)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["index", "joint", "stiffness", "damping", "max_effort", "drive_type", "dof_type"],
                )
                writer.writeheader()
                writer.writerows(rows)
        return rows

    def write_lowcmd_field_usage_report(self, filename: str = "lowcmd_field_usage.csv") -> list[dict[str, Any]]:
        rows = lowcmd_field_usage_rows(control_mode=self.control_mode)
        path = self._log_path(filename)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["field", "dds_lowcmd_exists", "decoded_by_bridge", "isaac_uses", "lost", "notes"],
                )
                writer.writeheader()
                writer.writerows(rows)
        return rows

    def write_state_mapping_csv(self, filename: str = "state_joint_mapping.csv") -> list[dict[str, Any]]:
        rows = self.state_mapping_rows()
        path = self._log_path(filename)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "index",
                        "isaac_joint",
                        "dds_joint",
                        "wbc_joint",
                        "isaac_dof_index",
                        "dds_index",
                        "unitree_joint_index",
                        "match",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
        return rows

    def write_motor35_mapping_csv(self, filename: str = "motor35_mapping.csv") -> list[dict[str, Any]]:
        rows = self.motor35_mapping_rows()
        path = self._log_path(filename)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["index", "motor_name", "used_ignored", "notes"])
                writer.writeheader()
                writer.writerows(rows)
        return rows

    def state_mapping_rows(self) -> list[dict[str, Any]]:
        if self.joint_mapping is None:
            return []
        rows: list[dict[str, Any]] = []
        for unitree_index, expected_joint in enumerate(self.joint_mapping.sonic_body_joint_names):
            isaac_index = int(self.joint_mapping.sonic_body_to_isaac_dof[unitree_index])
            isaac_joint = self.joint_mapping.isaac_dof_names[isaac_index]
            rows.append(
                {
                    "index": unitree_index,
                    "isaac_joint": isaac_joint,
                    "dds_joint": expected_joint,
                    "wbc_joint": expected_joint,
                    "isaac_dof_index": isaac_index,
                    "dds_index": unitree_index,
                    "unitree_joint_index": unitree_index,
                    "match": isaac_joint == expected_joint,
                    "notes": (
                        "Bridge writes this Isaac joint by name into the same Unitree/DDS body slot."
                    ),
                }
            )
        return rows

    def validate_joint_mapping_or_raise(self) -> list[dict[str, Any]]:
        if self.joint_mapping is None:
            raise RuntimeError("joint_mapping validation failed: joint_mapping is not attached")
        expected_unitree_order = list(SONIC_BODY_JOINT_NAMES)
        actual_unitree_order = list(self.joint_mapping.sonic_body_joint_names)
        if actual_unitree_order != expected_unitree_order:
            raise RuntimeError(
                "joint mapping validation failed: unexpected Unitree/WBC body joint order "
                f"actual={actual_unitree_order} expected={expected_unitree_order}"
            )
        rows = self.state_mapping_rows()
        if len(rows) != self.num_body_motor:
            raise RuntimeError(
                f"joint mapping validation failed: rows={len(rows)} expected={self.num_body_motor}"
            )
        failures = [row for row in rows if not bool(row["match"])]
        print("[IsaacUnitreeBridge] joint mapping validation:")
        for row in rows:
            status = "PASS" if row["match"] else "FAIL"
            print(
                "[IsaacUnitreeBridge] "
                f"joint_name={row['wbc_joint']} isaac_index={row['isaac_dof_index']} "
                f"dds_index={row['dds_index']} status={status}"
            )
        if failures:
            details = "; ".join(
                f"{row['wbc_joint']}: isaac_joint={row['isaac_joint']} "
                f"isaac_index={row['isaac_dof_index']} dds_index={row['dds_index']}"
                for row in failures
            )
            raise RuntimeError(f"joint mapping validation failed: {details}")
        self._joint_mapping_validated = True
        self._log(
            "joint_mapping_validation.log",
            f"current_isaac_order={list(self.joint_mapping.isaac_dof_names)}",
        )
        self._log(
            "joint_mapping_validation.log",
            f"expected_unitree_order={actual_unitree_order}",
        )
        self._log("joint_mapping_validation.log", "joint mapping validation: 29/29 PASS")
        for row in rows:
            self._log(
                "joint_mapping_validation.log",
                (
                    f"joint_name={row['wbc_joint']} isaac_index={row['isaac_dof_index']} "
                    f"dds_index={row['dds_index']} status=PASS"
                ),
            )
        return rows

    def motor35_mapping_rows(self) -> list[dict[str, Any]]:
        names = list(self.joint_mapping.sonic_body_joint_names) if self.joint_mapping is not None else []
        rows: list[dict[str, Any]] = []
        for index in range(35):
            if index < self.num_body_motor and index < len(names):
                rows.append(
                    {
                        "index": index,
                        "motor_name": names[index],
                        "used_ignored": "used_by_bridge_body",
                        "notes": "Bridge publishes/decodes body motor slot.",
                    }
                )
            else:
                rows.append(
                    {
                        "index": index,
                        "motor_name": "",
                        "used_ignored": "ignored_by_bridge_extra_slot",
                        "notes": "unitree_hg LowState/LowCmd has 35 slots; official G1 Isaac asset has 29 body DOF.",
                    }
                )
        return rows
 
    def record_lowstate_state_snapshot(
        self,
        label: str,
        *,
        simulation_time: float,
        state: Any,
        physics_dt: float | None = None,
    ) -> dict[str, Any]:
        path = self._log_path("lowstate_state_snapshot.csv")
        if path is None:
            return {}
        if self.joint_mapping is None:
            return {}

        joint_position = np.asarray(state.joint_position, dtype=np.float32)
        joint_velocity = np.asarray(state.joint_velocity, dtype=np.float32)
        root_position = np.asarray(state.root_position, dtype=np.float64)
        root_quaternion = np.asarray(state.root_quaternion, dtype=np.float64)
        root_angular_velocity = np.asarray(state.root_angular_velocity, dtype=np.float64)
        roll, pitch, yaw = _quat_wxyz_to_rpy(root_quaternion)
        reference_q = _standing_reference_pose(self.num_body_motor)
        contacts = self.read_contact_metrics(physics_dt or (1.0 / 500.0))
        foot_geometry = self.read_foot_collision_geometry_metrics()
        published = self.last_published_lowstate or {}
        published_q = _float_array(published.get("joint_position", []), self.num_body_motor, default=float("nan"))
        published_dq = _float_array(published.get("joint_velocity", []), self.num_body_motor, default=float("nan"))
        published_omega_body = _float_array(
            published.get("root_angular_velocity_body", published.get("root_angular_velocity", [])),
            3,
            default=float("nan"),
        )
        published_omega_world = _float_array(
            published.get("root_angular_velocity_world", []),
            3,
            default=float("nan"),
        )

        row: dict[str, Any] = {
            "label": label,
            "timestamp": f"{time.time():.9f}",
            "simulation_time": f"{float(simulation_time):.9f}",
            "lowstate_simulation_time": published.get("time", ""),
            "publish_count": published.get("publish_count", ""),
            "receive_count": int(self.stats.receive_count),
            "apply_count": int(self.stats.apply_count),
            "root_position_x": float(root_position[0]) if root_position.shape[0] >= 3 else float("nan"),
            "root_position_y": float(root_position[1]) if root_position.shape[0] >= 3 else float("nan"),
            "root_position_z": float(root_position[2]) if root_position.shape[0] >= 3 else float("nan"),
            "root_height": float(root_position[2]) if root_position.shape[0] >= 3 else float("nan"),
            "root_quaternion_format": "wxyz",
            "root_quaternion_w": float(root_quaternion[0]) if root_quaternion.shape[0] >= 4 else float("nan"),
            "root_quaternion_x": float(root_quaternion[1]) if root_quaternion.shape[0] >= 4 else float("nan"),
            "root_quaternion_y": float(root_quaternion[2]) if root_quaternion.shape[0] >= 4 else float("nan"),
            "root_quaternion_z": float(root_quaternion[3]) if root_quaternion.shape[0] >= 4 else float("nan"),
            "root_quaternion_norm": float(np.linalg.norm(root_quaternion)) if root_quaternion.shape[0] >= 4 else float("nan"),
            "root_roll": roll,
            "root_pitch": pitch,
            "root_yaw": yaw,
            "angular_velocity_source": "Isaac articulation.get_angular_velocity()",
            "angular_velocity_frame_assumption": "world",
            "root_angular_velocity_x": float(root_angular_velocity[0]) if root_angular_velocity.shape[0] >= 3 else float("nan"),
            "root_angular_velocity_y": float(root_angular_velocity[1]) if root_angular_velocity.shape[0] >= 3 else float("nan"),
            "root_angular_velocity_z": float(root_angular_velocity[2]) if root_angular_velocity.shape[0] >= 3 else float("nan"),
            "angular_velocity_norm": float(np.linalg.norm(root_angular_velocity)) if root_angular_velocity.shape[0] >= 3 else float("nan"),
            "angular_velocity_frame_published_to_lowstate": published.get("root_angular_velocity_frame", "body"),
            "published_omega_world_x": float(published_omega_world[0]),
            "published_omega_world_y": float(published_omega_world[1]),
            "published_omega_world_z": float(published_omega_world[2]),
            "published_omega_body_x": float(published_omega_body[0]),
            "published_omega_body_y": float(published_omega_body[1]),
            "published_omega_body_z": float(published_omega_body[2]),
            "published_omega_body_norm": float(np.linalg.norm(published_omega_body)),
            "angular_velocity_frame_expected_by_wbc": "body",
            **contacts,
            **foot_geometry,
        }
        for sonic_index, joint_name in enumerate(self.joint_mapping.sonic_body_joint_names):
            isaac_index = int(self.joint_mapping.sonic_body_to_isaac_dof[sonic_index])
            isaac_q = float(joint_position[isaac_index])
            isaac_dq = float(joint_velocity[isaac_index])
            ref_q = float(reference_q[sonic_index])
            row[f"{joint_name}.isaac_dof_index"] = isaac_index
            row[f"{joint_name}.dds_index_expected_by_wbc"] = sonic_index
            row[f"{joint_name}.isaac_q"] = isaac_q
            row[f"{joint_name}.isaac_dq"] = isaac_dq
            row[f"{joint_name}.wbc_reference_q"] = ref_q
            row[f"{joint_name}.q_error_vs_wbc_reference"] = isaac_q - ref_q
            row[f"{joint_name}.published_lowstate_q"] = float(published_q[sonic_index])
            row[f"{joint_name}.published_lowstate_dq"] = float(published_dq[sonic_index])
            row[f"{joint_name}.wbc_read_q_from_dds_index"] = float(published_q[sonic_index])
            row[f"{joint_name}.wbc_read_dq_from_dds_index"] = float(published_dq[sonic_index])

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or not self._lowstate_state_header_written
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if write_header:
                writer.writeheader()
                self._lowstate_state_header_written = True
            writer.writerow(row)
        self.startup_state_snapshots.append(dict(row))
        return row

    def record_standing_pose_error(
        self,
        label: str,
        *,
        simulation_time: float,
        state: Any,
        target_joint_positions_by_isaac: Any,
    ) -> list[dict[str, Any]]:
        path = self._log_path("standing_pose_error.csv")
        if path is None or self.joint_mapping is None:
            return []

        target_by_isaac = np.asarray(target_joint_positions_by_isaac, dtype=np.float32).reshape(-1)
        actual_by_isaac = np.asarray(state.joint_position, dtype=np.float32).reshape(-1)
        velocity_by_isaac = np.asarray(state.joint_velocity, dtype=np.float32).reshape(-1)
        rows: list[dict[str, Any]] = []
        errors: list[float] = []
        timestamp = f"{time.time():.9f}"
        for dds_index, joint_name in enumerate(self.joint_mapping.sonic_body_joint_names):
            isaac_index = int(self.joint_mapping.sonic_body_to_isaac_dof[dds_index])
            target_q = float(target_by_isaac[isaac_index]) if isaac_index < target_by_isaac.size else float("nan")
            actual_q = float(actual_by_isaac[isaac_index]) if isaac_index < actual_by_isaac.size else float("nan")
            actual_dq = float(velocity_by_isaac[isaac_index]) if isaac_index < velocity_by_isaac.size else float("nan")
            error = actual_q - target_q
            errors.append(abs(error))
            rows.append(
                {
                    "label": label,
                    "timestamp": timestamp,
                    "simulation_time": f"{float(simulation_time):.9f}",
                    "dds_index": dds_index,
                    "joint_name": joint_name,
                    "isaac_index": isaac_index,
                    "target_q": target_q,
                    "actual_q": actual_q,
                    "actual_dq": actual_dq,
                    "error": error,
                    "abs_error": abs(error),
                }
            )

        max_abs_error = float(max(errors)) if errors else float("nan")
        mean_abs_error = float(np.mean(errors)) if errors else float("nan")
        for row in rows:
            row["max_abs_error_this_label"] = max_abs_error
            row["mean_abs_error_this_label"] = mean_abs_error

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or not self._standing_pose_error_header_written
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
            if write_header and rows:
                writer.writeheader()
                self._standing_pose_error_header_written = True
            if rows:
                writer.writerows(rows)
        self._log(
            "standing_mode.log",
            f"standing_pose_error label={label} max_abs_error={max_abs_error:.9f} mean_abs_error={mean_abs_error:.9f}",
        )
        return rows

    def record_timeline_event(
        self,
        event: str,
        *,
        simulation_time: float | None = None,
        state: Any | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        path = self._log_path("timeline.csv")
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        root_position = _float_array(getattr(state, "root_position", []), 3, default=float("nan")) if state is not None else np.full(3, float("nan"))
        quaternion = _float_array(getattr(state, "root_quaternion", []), 4, default=float("nan")) if state is not None else np.full(4, float("nan"))
        roll, pitch, yaw = _quat_wxyz_to_rpy(quaternion)
        row = {
            "event": event,
            "timestamp": f"{time.time():.9f}",
            "monotonic_time": f"{time.monotonic():.9f}",
            "simulation_time": "" if simulation_time is None else f"{float(simulation_time):.9f}",
            "root_height": f"{float(root_position[2]):.9f}",
            "root_roll": f"{roll:.9f}",
            "root_pitch": f"{pitch:.9f}",
            "root_yaw": f"{yaw:.9f}",
            "details": json.dumps(dict(details or {}), sort_keys=True),
        }
        write_header = not path.exists() or not self._timeline_header_written
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if write_header:
                writer.writeheader()
                self._timeline_header_written = True
            writer.writerow(row)

    def read_contact_metrics(self, physics_dt: float) -> dict[str, Any]:
        for attempt in range(2):
            try:
                return self._read_contact_metrics_once(physics_dt)
            except Exception as exc:
                self._contact_views = {}
                self._contact_error = repr(exc)
                if attempt == 0:
                    continue
                return {
                    "left_foot_contact": "",
                    "right_foot_contact": "",
                    "left_contact_count": "",
                    "right_contact_count": "",
                    "left_normal_force": "",
                    "right_normal_force": "",
                    "normal_force": "",
                    "left_contact_lowest_point_x": "",
                    "left_contact_lowest_point_y": "",
                    "left_contact_lowest_point_z": "",
                    "left_contact_distance_min_m": "",
                    "right_contact_lowest_point_x": "",
                    "right_contact_lowest_point_y": "",
                    "right_contact_lowest_point_z": "",
                    "right_contact_distance_min_m": "",
                    "contact_error": repr(exc),
                }
        return {}

    def _read_contact_metrics_once(self, physics_dt: float) -> dict[str, Any]:
        if not self._contact_views:
            from isaacsim.core.api.sensors import RigidContactView
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import UsdPhysics

            stage = get_current_stage()
            ground_paths: list[str] = []
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if ("Ground" in path or "ground" in path) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    ground_paths.append(path)
            if not ground_paths:
                ground_paths = ["/World/defaultGroundPlane/GroundPlane/CollisionPlane"]

            foot_paths = {
                "left": "/World/G1/left_ankle_roll_link",
                "right": "/World/G1/right_ankle_roll_link",
            }
            if self.robot_asset is not None and getattr(self.robot_asset, "foot_prim_paths", None):
                for path in self.robot_asset.foot_prim_paths:
                    if "left" in path:
                        foot_paths["left"] = path
                    elif "right" in path:
                        foot_paths["right"] = path
            for side, expr in foot_paths.items():
                view = RigidContactView(
                    prim_paths_expr=expr,
                    filter_paths_expr=ground_paths,
                    name=f"wbc_timeline_{side}_foot_contact_{int(time.time() * 1000)}",
                    prepare_contact_sensors=True,
                    max_contact_count=256,
                )
                view.initialize()
                self._contact_views[side] = view
        left = _contact_metrics(self._contact_views["left"], physics_dt)
        right = _contact_metrics(self._contact_views["right"], physics_dt)
        self._contact_error = ""
        return {
            "left_foot_contact": int(left["contact_count"]) > 0,
            "right_foot_contact": int(right["contact_count"]) > 0,
            "left_contact_count": int(left["contact_count"]),
            "right_contact_count": int(right["contact_count"]),
            "left_normal_force": float(left["normal_force_n"]),
            "right_normal_force": float(right["normal_force_n"]),
            "normal_force": float(left["normal_force_n"]) + float(right["normal_force_n"]),
            "left_contact_lowest_point_x": left["lowest_point_x"],
            "left_contact_lowest_point_y": left["lowest_point_y"],
            "left_contact_lowest_point_z": left["lowest_point_z"],
            "left_contact_distance_min_m": left["distance_min_m"],
            "right_contact_lowest_point_x": right["lowest_point_x"],
            "right_contact_lowest_point_y": right["lowest_point_y"],
            "right_contact_lowest_point_z": right["lowest_point_z"],
            "right_contact_distance_min_m": right["distance_min_m"],
            "contact_error": "",
        }

    def read_foot_collision_geometry_metrics(self) -> dict[str, Any]:
        rows: dict[str, Any] = {
            "foot_collision_geometry_source": "usd_collision_bbox_world_bound",
            "foot_collision_geometry_error": "",
        }
        for side in ("left", "right"):
            rows.update(
                {
                    f"{side}_foot_lowest_collision_point_x": "",
                    f"{side}_foot_lowest_collision_point_y": "",
                    f"{side}_foot_lowest_collision_point_z": "",
                    f"{side}_foot_ground_distance_m": "",
                    f"{side}_foot_collision_prim_count": 0,
                }
            )
        try:
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import UsdGeom, UsdPhysics

            stage = get_current_stage()
            bbox_cache = UsdGeom.BBoxCache(0.0, ["default", "render", "proxy", "guide"])
            foot_paths = {
                "left": "/World/G1/left_ankle_roll_link",
                "right": "/World/G1/right_ankle_roll_link",
            }
            if self.robot_asset is not None and getattr(self.robot_asset, "foot_prim_paths", None):
                for path in self.robot_asset.foot_prim_paths:
                    if "left" in path:
                        foot_paths["left"] = path
                    elif "right" in path:
                        foot_paths["right"] = path

            for side, foot_path in foot_paths.items():
                lowest: tuple[float, float, float] | None = None
                count = 0
                prefix = foot_path.rstrip("/") + "/"
                for prim in stage.Traverse():
                    path = str(prim.GetPath())
                    if path != foot_path and not path.startswith(prefix):
                        continue
                    has_collision = (
                        prim.HasAPI(UsdPhysics.CollisionAPI)
                        or prim.GetAttribute("physics:collisionEnabled").IsValid()
                    )
                    if not has_collision:
                        continue
                    count += 1
                    try:
                        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                        minimum = box.GetMin()
                        maximum = box.GetMax()
                        candidate = (
                            float((minimum[0] + maximum[0]) * 0.5),
                            float((minimum[1] + maximum[1]) * 0.5),
                            float(minimum[2]),
                        )
                    except Exception:
                        continue
                    if lowest is None or candidate[2] < lowest[2]:
                        lowest = candidate
                rows[f"{side}_foot_collision_prim_count"] = count
                if lowest is not None:
                    rows[f"{side}_foot_lowest_collision_point_x"] = lowest[0]
                    rows[f"{side}_foot_lowest_collision_point_y"] = lowest[1]
                    rows[f"{side}_foot_lowest_collision_point_z"] = lowest[2]
                    rows[f"{side}_foot_ground_distance_m"] = lowest[2]
        except Exception as exc:
            rows["foot_collision_geometry_error"] = repr(exc)
        return rows

    def record_lowcmd_snapshot(
        self,
        *,
        simulation_time: float,
        state: Any,
        command: IsaacLowCommand,
        joint_indices: Mapping[str, int],
        physics_dt: float,
    ) -> dict[str, Any]:
        path = self._log_path("lowcmd_snapshot.csv")
        if path is None:
            return {}

        joint_position = np.asarray(state.joint_position, dtype=np.float32)
        joint_velocity = np.asarray(state.joint_velocity, dtype=np.float32)
        root_position = np.asarray(state.root_position, dtype=np.float64)
        root_quaternion = np.asarray(state.root_quaternion, dtype=np.float64)
        root_angular_velocity = np.asarray(state.root_angular_velocity, dtype=np.float64)
        roll, pitch, yaw = _quat_wxyz_to_rpy(root_quaternion)
        contacts = self.read_contact_metrics(physics_dt)
        row: dict[str, Any] = {
            "timestamp": f"{time.time():.9f}",
            "simulation_time": f"{float(simulation_time):.9f}",
            "receive_count": int(self.stats.receive_count),
            "apply_count": int(self.stats.apply_count),
            "sequence_id": int(command.sequence_id),
            "received_action_dim": int(command.received_action_dim),
            "applied_joint_dim": int(self.stats.last_applied_joint_dim),
            "root_height": float(root_position[2]) if root_position.shape[0] >= 3 else float("nan"),
            "root_position_x": float(root_position[0]) if root_position.shape[0] >= 3 else float("nan"),
            "root_position_y": float(root_position[1]) if root_position.shape[0] >= 3 else float("nan"),
            "root_position_z": float(root_position[2]) if root_position.shape[0] >= 3 else float("nan"),
            "root_quaternion_w": float(root_quaternion[0]) if root_quaternion.shape[0] >= 4 else float("nan"),
            "root_quaternion_x": float(root_quaternion[1]) if root_quaternion.shape[0] >= 4 else float("nan"),
            "root_quaternion_y": float(root_quaternion[2]) if root_quaternion.shape[0] >= 4 else float("nan"),
            "root_quaternion_z": float(root_quaternion[3]) if root_quaternion.shape[0] >= 4 else float("nan"),
            "root_roll": roll,
            "root_pitch": pitch,
            "root_yaw": yaw,
            "root_angular_velocity_x": float(root_angular_velocity[0]) if root_angular_velocity.shape[0] >= 3 else float("nan"),
            "root_angular_velocity_y": float(root_angular_velocity[1]) if root_angular_velocity.shape[0] >= 3 else float("nan"),
            "root_angular_velocity_z": float(root_angular_velocity[2]) if root_angular_velocity.shape[0] >= 3 else float("nan"),
            **contacts,
        }
        for joint_name, sonic_index in joint_indices.items():
            isaac_index = sonic_index
            if self.joint_mapping is not None:
                isaac_index = int(self.joint_mapping.sonic_body_to_isaac_dof[sonic_index])
            actual_q = float(joint_position[isaac_index])
            actual_dq = float(joint_velocity[isaac_index])
            lowcmd_q = float(command.q[sonic_index])
            row[f"{joint_name}.actual_q"] = actual_q
            row[f"{joint_name}.actual_dq"] = actual_dq
            row[f"{joint_name}.lowcmd_q"] = lowcmd_q
            row[f"{joint_name}.lowcmd_dq"] = float(command.dq[sonic_index])
            row[f"{joint_name}.lowcmd_kp"] = float(command.kp[sonic_index])
            row[f"{joint_name}.lowcmd_kd"] = float(command.kd[sonic_index])
            row[f"{joint_name}.lowcmd_tau"] = float(command.tau[sonic_index])
            row[f"{joint_name}.q_error"] = actual_q - lowcmd_q
            if self.last_command_tau_by_isaac is not None:
                row[f"{joint_name}.command_tau"] = float(self.last_command_tau_by_isaac[isaac_index])
            if self.last_applied_tau_by_isaac is not None:
                row[f"{joint_name}.applied_tau"] = float(self.last_applied_tau_by_isaac[isaac_index])

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or not self._lowcmd_snapshot_header_written
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if write_header:
                writer.writeheader()
                self._lowcmd_snapshot_header_written = True
            writer.writerow(row)
        return row

    def cmd_received(self) -> bool:
        with self._low_cmd_lock:
            return self._low_cmd_received

    def _low_cmd_handler(self, msg: Any) -> None:
        received_action_dim = _motor_cmd_dim(msg)
        if received_action_dim is not None and received_action_dim < self.num_body_motor:
            self.stats.last_error = (
                f"rt/lowcmd motor_cmd has too few slots: "
                f"received_action_dim={received_action_dim} body_required={self.num_body_motor}"
            )
            self._log("dds_dimensions.log", f"DDS_DIMENSION_FAIL {self.stats.last_error}")
            return

        if received_action_dim is None:
            received_action_dim_value = self.num_body_motor
        else:
            received_action_dim_value = int(received_action_dim)
        body_dim = self.num_body_motor
        hand_dim = max(0, received_action_dim_value - body_dim)

        with self._low_cmd_lock:
            self._low_cmd_sequence += 1
            command = IsaacLowCommand(
                q=np.asarray([msg.motor_cmd[i].q for i in range(self.num_body_motor)], dtype=np.float32),
                dq=np.asarray([msg.motor_cmd[i].dq for i in range(self.num_body_motor)], dtype=np.float32),
                kp=np.asarray([msg.motor_cmd[i].kp for i in range(self.num_body_motor)], dtype=np.float32),
                kd=np.asarray([msg.motor_cmd[i].kd for i in range(self.num_body_motor)], dtype=np.float32),
                tau=np.asarray([msg.motor_cmd[i].tau for i in range(self.num_body_motor)], dtype=np.float32),
                mode=np.asarray([msg.motor_cmd[i].mode for i in range(self.num_body_motor)], dtype=np.int32),
                reserve=np.asarray([getattr(msg.motor_cmd[i], "reserve", 0) for i in range(self.num_body_motor)], dtype=np.uint32),
                mode_pr=int(getattr(msg, "mode_pr", 0)),
                mode_machine=int(getattr(msg, "mode_machine", 0)),
                message_reserve=_to_list(getattr(msg, "reserve", [])),
                crc=int(getattr(msg, "crc", 0)),
                received_action_dim=received_action_dim_value,
                body_dim=body_dim,
                hand_dim=hand_dim,
                sequence_id=self._low_cmd_sequence,
            )
            self.low_cmd = msg
            self.latest_low_command = command
            self._low_cmd_received = True
            self._new_low_cmd = True
            self.stats.receive_count += 1
            self.stats.last_received_action_dim = received_action_dim_value
            self.stats.last_receive_time = command.received_time
            if self.stats.first_receive_time is None:
                self.stats.first_receive_time = command.received_time
        if self.performance_monitor is not None:
            self.performance_monitor.record_dds_receive()
        if self.stats.receive_count == 1 or self.stats.receive_count % self.receive_log_interval == 0:
            self._log(
                "dds_dimensions.log",
                (
                    f"received_action_dim={received_action_dim_value} "
                    f"body_dim={body_dim} hand_dim={hand_dim} "
                    f"applied_joint_dim={body_dim}"
                ),
            )

    def _left_hand_cmd_handler(self, msg: Any) -> None:
        with self._left_hand_cmd_lock:
            self.left_hand_cmd = msg
            self.left_hand_cmd_received = True
            self.new_left_hand_cmd = True

    def _right_hand_cmd_handler(self, msg: Any) -> None:
        with self._right_hand_cmd_lock:
            self.right_hand_cmd = msg
            self.right_hand_cmd_received = True
            self.new_right_hand_cmd = True

    def _load_unitree_dds(self) -> None:
        for path in (self.dds_site_packages, self.unitree_sdk_path):
            if path and path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandCmd_ as HandCmd_default,
            unitree_hg_msg_dds__HandState_ as HandState_default,
            unitree_hg_msg_dds__IMUState_ as IMUState_default,
            unitree_hg_msg_dds__LowCmd_,
            unitree_hg_msg_dds__LowState_ as LowState_default,
            unitree_hg_msg_dds__OdoState_ as OdoState_default,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, IMUState_, LowCmd_, LowState_, OdoState_

        self._ChannelFactoryInitialize = ChannelFactoryInitialize
        self._ChannelPublisher = ChannelPublisher
        self._ChannelSubscriber = ChannelSubscriber
        self._LowStateType = LowState_
        self._OdoStateType = OdoState_
        self._IMUStateType = IMUState_
        self._LowCmdType = LowCmd_
        self._HandStateType = HandState_
        self._HandCmdType = HandCmd_
        self.low_state = LowState_default()
        self.odo_state = OdoState_default()
        self.secondary_imu_state = IMUState_default()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.left_hand_state = HandState_default()
        self.right_hand_state = HandState_default()
        self.left_hand_cmd = HandCmd_default()
        self.right_hand_cmd = HandCmd_default()

    def _ensure_messages(self) -> None:
        if (
            self.low_state is None
            or self.odo_state is None
            or self.secondary_imu_state is None
            or self.left_hand_state is None
            or self.right_hand_state is None
        ):
            self._load_unitree_dds()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("IsaacUnitreeBridge.start() must be called first")

    def _log(self, filename: str, message: str) -> None:
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / filename).open("a", encoding="utf-8") as file:
                file.write(message.rstrip() + "\n")
        if self.verbose:
            print(f"[IsaacUnitreeBridge] {message}")

    def _log_path(self, filename: str) -> Path | None:
        if self.log_dir is None:
            return None
        return self.log_dir / filename


def _float_array(value: Any, size: int, default: float = 0.0) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return np.full(size, default, dtype=np.float32)
    if array.size < size:
        padded = np.full(size, default, dtype=np.float32)
        padded[: array.size] = array
        return padded
    return array[:size]


def _motor_cmd_dim(msg: Any) -> int | None:
    motor_cmd = getattr(msg, "motor_cmd", None)
    if motor_cmd is None:
        return None
    try:
        return int(len(motor_cmd))
    except Exception:
        return None


def lowcmd_field_usage_rows(control_mode: str = "position") -> list[dict[str, Any]]:
    effort_mode = str(control_mode).strip().lower() == "effort"
    return [
        {
            "field": "mode_pr",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": False,
            "lost": True,
            "notes": "Logged only; not forwarded to Isaac articulation.",
        },
        {
            "field": "mode_machine",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": False,
            "lost": True,
            "notes": "Logged only; not forwarded to Isaac articulation.",
        },
        {
            "field": "motor_cmd[].mode",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": False,
            "lost": True,
            "notes": "Decoded/logged; Isaac write path does not consume motor mode.",
        },
        {
            "field": "motor_cmd[].q",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": True,
            "lost": False,
            "notes": (
                "Used in effort equation kp*(q_des-q)+kd*(dq_des-dq)+tau_ff."
                if effort_mode
                else "Mapped body 29 q values to Isaac set_joint_position_targets()."
            ),
        },
        {
            "field": "motor_cmd[].dq",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": effort_mode,
            "lost": not effort_mode,
            "notes": (
                "Used in effort equation kp*(q_des-q)+kd*(dq_des-dq)+tau_ff."
                if effort_mode
                else "Decoded/logged; not written as velocity target."
            ),
        },
        {
            "field": "motor_cmd[].kp",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": effort_mode,
            "lost": not effort_mode,
            "notes": (
                "Used in effort equation kp*(q_des-q)+kd*(dq_des-dq)+tau_ff."
                if effort_mode
                else "Decoded/logged; Isaac drive gains are configured separately before control."
            ),
        },
        {
            "field": "motor_cmd[].kd",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": effort_mode,
            "lost": not effort_mode,
            "notes": (
                "Used in effort equation kp*(q_des-q)+kd*(dq_des-dq)+tau_ff."
                if effort_mode
                else "Decoded/logged; Isaac drive gains are configured separately before control."
            ),
        },
        {
            "field": "motor_cmd[].tau",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": effort_mode,
            "lost": not effort_mode,
            "notes": (
                "Used as tau_ff in effort equation kp*(q_des-q)+kd*(dq_des-dq)+tau_ff."
                if effort_mode
                else "Decoded/logged; not applied as effort feed-forward."
            ),
        },
        {
            "field": "motor_cmd[].reserve",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": False,
            "lost": True,
            "notes": "Decoded/logged only.",
        },
        {
            "field": "reserve[4]",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": False,
            "lost": True,
            "notes": "Decoded/logged only.",
        },
        {
            "field": "crc",
            "dds_lowcmd_exists": True,
            "decoded_by_bridge": True,
            "isaac_uses": False,
            "lost": True,
            "notes": "Decoded/logged only; launcher disables CRC check on GR00T side.",
        },
        {
            "field": "weight",
            "dds_lowcmd_exists": False,
            "decoded_by_bridge": False,
            "isaac_uses": False,
            "lost": False,
            "notes": "No weight field exists in unitree_hg LowCmd_/MotorCmd_ IDL.",
        },
    ]


def _quat_wxyz_to_rpy(quaternion: Any) -> tuple[float, float, float]:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.size < 4 or not np.all(np.isfinite(q[:4])):
        return float("nan"), float("nan"), float("nan")
    norm = float(np.linalg.norm(q[:4]))
    if norm <= 0.0:
        return float("nan"), float("nan"), float("nan")
    w, x, y, z = (q[:4] / norm).tolist()
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))
    sinp = 2.0 * (w * y - z * x)
    pitch = float(np.sign(sinp) * np.pi / 2.0) if abs(sinp) >= 1.0 else float(np.arcsin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def _root_angular_velocity_world_body(obs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    omega_world = _float_array(obs.get("root_angular_velocity", [0.0, 0.0, 0.0]), 3)
    quaternion = _float_array(obs.get("root_quaternion", [1.0, 0.0, 0.0, 0.0]), 4)
    rotation_world_body = _quat_wxyz_to_rotation_world_body(quaternion)
    omega_body = rotation_world_body.T @ omega_world.astype(np.float64)
    return omega_world.astype(np.float32), omega_body.astype(np.float32)


def _quat_wxyz_to_rotation_world_body(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.size < 4 or not np.all(np.isfinite(q[:4])):
        return np.eye(3, dtype=np.float64)
    norm = float(np.linalg.norm(q[:4]))
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = (q[:4] / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _contact_metrics(view: Any, physics_dt: float) -> dict[str, Any]:
    matrix = view.get_contact_force_matrix(dt=physics_dt)
    if matrix is None:
        return {
            "normal_force_n": 0.0,
            "friction_force_n": 0.0,
            "contact_count": 0,
            "lowest_point_x": "",
            "lowest_point_y": "",
            "lowest_point_z": "",
            "distance_min_m": "",
        }
    force = np.asarray(matrix, dtype=np.float32).reshape(-1, 3).sum(axis=0)
    data = view.get_contact_force_data(dt=physics_dt)
    contact_count = 0
    lowest_point: tuple[float, float, float] | None = None
    distance_min: float | str = ""
    if data is not None:
        _forces, points, _normals, distances, counts, _starts = data
        contact_count = int(np.sum(np.asarray(counts, dtype=np.int64).reshape(-1)))
        if contact_count > 0:
            points_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            valid_points = points_array[:contact_count]
            if valid_points.size:
                index = int(np.argmin(valid_points[:, 2]))
                lowest_point = tuple(float(v) for v in valid_points[index].tolist())
            distances_array = np.asarray(distances, dtype=np.float32).reshape(-1)
            if distances_array.size:
                distance_min = float(np.min(distances_array[:contact_count]))
    return {
        "normal_force_n": float(force[2]),
        "friction_force_n": float(np.linalg.norm(force[:2])),
        "contact_count": contact_count,
        "lowest_point_x": "" if lowest_point is None else lowest_point[0],
        "lowest_point_y": "" if lowest_point is None else lowest_point[1],
        "lowest_point_z": "" if lowest_point is None else lowest_point[2],
        "distance_min_m": distance_min,
    }


def _standing_reference_pose(size: int) -> np.ndarray:
    try:
        from gear_sonic.robot_interface.pd_mapping import SONIC_STANDING_POSE

        return _float_array(SONIC_STANDING_POSE, size, default=float("nan"))
    except Exception:
        return np.full(size, float("nan"), dtype=np.float32)


def _tick_from_obs(obs: Mapping[str, Any]) -> int:
    if "time" in obs:
        return int(float(obs["time"]) * 1e3)
    return int(time.time() * 1e3)


def _assign_temperature(motor_state: Any, temperature: int) -> None:
    if isinstance(getattr(motor_state, "temperature", None), list):
        motor_state.temperature[:] = [int(temperature)] * len(motor_state.temperature)
    else:
        motor_state.temperature = int(temperature)


def _preview(values: Iterable[Any] | None, limit: int = 6) -> list[float]:
    if values is None:
        return []
    array = np.asarray(list(values), dtype=np.float32).reshape(-1)
    return [float(value) for value in array[:limit]]


def _drive_type_at(drive_types: np.ndarray | None, index: int) -> str:
    if drive_types is None:
        return "unknown"
    try:
        value = drive_types[0, index] if drive_types.ndim >= 2 else drive_types[index]
    except Exception:
        return "unknown"
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def _dof_type_at(dof_types: Any, index: int) -> str:
    if dof_types is None:
        return "unknown"
    try:
        value = dof_types[index]
    except Exception:
        return "unknown"
    return str(value)


def _to_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float, str, bool)):
        return [value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return list(value)
