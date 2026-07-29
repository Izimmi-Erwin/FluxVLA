"""Isaac Sim backend for the Sonic simulation server.

This backend is intentionally limited to the first Isaac migration milestone:
start Isaac Sim, load the G1 USD, acquire an articulation, validate body joint
mapping, reset, step, and read state. It does not connect DDS, WBC, FluxVLA, or
camera publishing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import importlib.util
from pathlib import Path
import sys
from threading import RLock
import time
from typing import Any

import numpy as np

from gear_sonic.physics.contact_config import IsaacContactConfig
from gear_sonic.physics.passive_config import PassiveDynamicsConfig, build_passive_mapping
from gear_sonic.robot_interface.joint_mapping import (
    SONIC_BODY_JOINT_NAMES,
    build_joint_mapping,
)
from gear_sonic.robot_interface.isaac_robot_assets import (
    DEFAULT_ROBOT_MODEL,
    get_robot_asset,
)

from .protocol import JsonZmqSimulationServer, SimulationBackend


DEFAULT_G1_USD_PATH = get_robot_asset(DEFAULT_ROBOT_MODEL).usd_path
DEFAULT_SONICSTAR_TASK_SCENE_LAYER_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "scenes"
    / "sonicstar"
    / "sonicstar_task_scene.usda"
)
DEFAULT_FLUXVLA_SITE_PACKAGES = Path(
    "/home/limx/miniconda3/envs/fluxvla/lib/python3.10/site-packages"
)


def _ensure_optional_site_packages(*module_names: str) -> None:
    """Expose optional runtime deps when Isaac Python does not ship them.

    Isaac Sim's bundled Python in this workspace does not include the msgpack /
    cv2 stack used by the MuJoCo camera protocol, while the local FluxVLA env
    does.  Adding that site-packages path keeps the wire protocol identical
    without modifying FluxVLA, WBC, USD, or control parameters.
    """

    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if not missing:
        return
    site_packages = DEFAULT_FLUXVLA_SITE_PACKAGES
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


class IsaacImagePublisher:
    """Isaac camera publisher with the same ZMQ/msgpack schema as MuJoCo port 5555."""

    def __init__(
        self,
        *,
        port: int = 5555,
        camera_name: str = "ego_view",
        width: int = 640,
        height: int = 480,
        image_dt: float = 1.0 / 30.0,
        verbose: bool = True,
    ):
        self.port = int(port)
        self.camera_name = str(camera_name)
        self.width = int(width)
        self.height = int(height)
        self.image_dt = float(image_dt)
        self.verbose = bool(verbose)
        self.sensor_server = None
        self.camera = None
        self._last_publish_time = 0.0
        self._fallback_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def start(self, backend: "IsaacSimulationBackend") -> None:
        _ensure_optional_site_packages("msgpack", "msgpack_numpy", "cv2", "zmq")
        from gear_sonic.utils.mujoco_sim.sensor_server import SensorServer

        self.sensor_server = SensorServer()
        self.sensor_server.start_server(port=self.port)
        self._try_create_camera(backend)
        print(
            "[IsaacImagePublisher] publishing MuJoCo-compatible camera stream "
            f"camera={self.camera_name!r} port={self.port} shape=({self.height},{self.width},3)"
        )

    def stop(self) -> None:
        if self.sensor_server is not None:
            try:
                self.sensor_server.stop_server()
            except Exception:
                pass
        self.sensor_server = None
        self.camera = None

    def maybe_publish(self, backend: "IsaacSimulationBackend") -> None:
        if self.sensor_server is None:
            return
        now = time.time()
        if now - self._last_publish_time < self.image_dt:
            return
        self._last_publish_time = now
        self.publish(backend=backend, timestamp=now)

    def publish(self, *, backend: "IsaacSimulationBackend", timestamp: float | None = None) -> None:
        if self.sensor_server is None:
            return
        from gear_sonic.utils.mujoco_sim.sensor_server import ImageMessageSchema, ImageUtils

        image = self._read_camera_image()
        timestamp = time.time() if timestamp is None else float(timestamp)
        image_msg = ImageMessageSchema(
            timestamps={self.camera_name: timestamp},
            images={self.camera_name: image},
        )
        serialized_data = image_msg.serialize()
        # MuJoCo's ImagePublishProcess also includes a legacy top-level key
        # with the same JPEG payload. Keep it for byte-schema compatibility.
        serialized_data[self.camera_name] = ImageUtils.encode_image(image)
        self.sensor_server.send_message(serialized_data)

    def _try_create_camera(self, backend: "IsaacSimulationBackend") -> None:
        try:
            try:
                from isaacsim.sensors.camera import Camera
            except Exception:
                from omni.isaac.sensor import Camera

            try:
                from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

                # External front-left view aimed roughly at the tabletop/robot area.
                orientation = euler_angles_to_quats(
                    np.asarray([60.0, 0.0, 135.0], dtype=np.float32),
                    degrees=True,
                )
            except Exception:
                orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            self.camera = Camera(
                prim_path=f"/World/{self.camera_name}_camera",
                name=f"{self.camera_name}_camera",
                position=np.asarray([1.2, -1.2, 1.35], dtype=np.float32),
                orientation=orientation,
                resolution=(self.width, self.height),
                frequency=max(1, int(round(1.0 / self.image_dt))),
            )
            initialize = getattr(self.camera, "initialize", None)
            if initialize is not None:
                initialize()
            if backend.world is not None:
                backend.world.step(render=True)
            print(f"[IsaacImagePublisher] Isaac camera initialized at /World/{self.camera_name}_camera")
        except Exception as exc:
            self.camera = None
            print(
                "[IsaacImagePublisher] warning: Isaac Camera unavailable; "
                f"publishing black fallback frames. error={exc!r}"
            )

    def _read_camera_image(self) -> np.ndarray:
        if self.camera is None:
            return self._fallback_frame

        try:
            rgba = None
            for getter_name in ("get_rgba", "get_rgb"):
                getter = getattr(self.camera, getter_name, None)
                if getter is not None:
                    rgba = getter()
                    if rgba is not None:
                        break
            if rgba is None:
                return self._fallback_frame
            image = np.asarray(rgba)
            if image.size == 0:
                return self._fallback_frame
            if image.ndim == 3 and image.shape[-1] >= 3:
                image = image[..., :3]
            image = np.asarray(image)
            if image.dtype != np.uint8:
                max_value = float(np.nanmax(image)) if image.size else 0.0
                if max_value <= 1.0:
                    image = image * 255.0
                image = np.clip(image, 0, 255).astype(np.uint8)
            if image.shape[:2] != (self.height, self.width):
                try:
                    import cv2

                    image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                except Exception:
                    return self._fallback_frame
            return np.ascontiguousarray(image)
        except Exception as exc:
            if self.verbose:
                print(f"[IsaacImagePublisher] warning: failed to read camera frame: {exc!r}")
            return self._fallback_frame


class IsaacKeyboardCommandSubscriber:
    """Subscribe to the shared FluxVLA keyboard channel for simulator-only keys.

    ``send_keyboard_cmd.py`` owns the PUB side on port 5580. FluxVLA consumes
    ``k/i/p`` from that channel. Isaac only mirrors the MuJoCo viewer keys that
    are simulator-local: ``9`` and ``Backspace``. All other keys are ignored so
    the FluxVLA / GR00T command path remains unchanged.
    """

    SIM_KEYS = {"9", "backspace", "\b", "\x7f"}

    def __init__(
        self,
        backend: "IsaacSimulationBackend",
        *,
        host: str = "localhost",
        port: int = 5580,
    ):
        self.backend = backend
        self.host = host
        self.port = int(port)
        self.endpoint = f"tcp://{host}:{self.port}"
        self._ctx: Any | None = None
        self._socket: Any | None = None
        self._zmq: Any | None = None
        self._running = False

    def start(self) -> bool:
        if self._running:
            return True
        try:
            _ensure_optional_site_packages("zmq")
            import zmq

            self._zmq = zmq
            self._ctx = zmq.Context()
            self._socket = self._ctx.socket(zmq.SUB)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.setsockopt(zmq.RCVTIMEO, 0)
            self._socket.connect(self.endpoint)
            self._running = True
            print(
                "[IsaacKeyboard] connected to shared keyboard channel "
                f"{self.endpoint}; handling only 9/backspace"
            )
            return True
        except Exception as exc:
            print(f"[IsaacKeyboard] disabled: {exc!r}")
            self.stop()
            return False

    def poll_once(self) -> bool:
        if not self._running or self._socket is None or self._zmq is None:
            return False
        try:
            key = self._socket.recv_string(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return False
        except Exception as exc:
            print(f"[IsaacKeyboard] receive failed; disabling subscriber: {exc!r}")
            self.stop()
            return False

        normalized = str(key).lower()
        if normalized in self.SIM_KEYS:
            self.backend.keyboard(normalized)
            return True
        return False

    def stop(self) -> None:
        self._running = False
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass
            self._socket = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None
        self._zmq = None


class IsaacRealtimeStatePublisher:
    """Optional Isaac-side publisher for GR00T/FluxVLA g1_debug frames on port 5557.

    The normal four-terminal flow should leave this disabled because
    ``g1_deploy_onnx_ref`` already owns 5557 and publishes the canonical C++
    payload. This class is a protocol-compatible fallback for Isaac-only audits.
    """

    def __init__(self, *, port: int = 5557, topic: str = "g1_debug", verbose: bool = True):
        self.port = int(port)
        self.topic = str(topic)
        self.verbose = bool(verbose)
        self.context = None
        self.socket = None
        self.index = 0
        self._last_config_publish = 0.0

    def start(self, backend: "IsaacSimulationBackend") -> None:
        _ensure_optional_site_packages("msgpack", "zmq")
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{self.port}")
        print(
            "[IsaacRealtimeStatePublisher] optional g1_debug publisher running "
            f"at tcp://*:{self.port} topic={self.topic!r}"
        )

    def stop(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
        if self.context is not None:
            try:
                self.context.term()
            except Exception:
                pass
        self.socket = None
        self.context = None

    def publish(self, backend: "IsaacSimulationBackend", state: IsaacRobotState | None = None) -> None:
        if self.socket is None:
            return
        _ensure_optional_site_packages("msgpack", "zmq")
        import msgpack
        import zmq

        state = backend.read_joint_state() if state is None else state
        now = time.time()
        if now - self._last_config_publish >= 2.0:
            self._send(
                "robot_config",
                {
                    "control_loop_type": "cpp",
                    "simulator": "isaac",
                    "robot_model": backend.robot_model,
                    "body_dof": len(SONIC_BODY_JOINT_NAMES),
                    "topic_prefix": self.topic,
                },
                msgpack_module=msgpack,
                zmq_module=zmq,
            )
            self._last_config_publish = now

        payload = self._build_debug_payload(backend=backend, state=state, timestamp=now)
        self._send(self.topic, payload, msgpack_module=msgpack, zmq_module=zmq)
        self.index += 1

    def _send(self, topic: str, payload: dict[str, Any], *, msgpack_module: Any, zmq_module: Any) -> None:
        packed = msgpack_module.packb(payload, use_bin_type=True)
        # Match the C++ ZMQ output handler: one frame, topic prefix prepended
        # directly to the msgpack payload.
        frame = topic.encode("utf-8") + packed
        try:
            self.socket.send(frame, flags=zmq_module.NOBLOCK)
        except zmq_module.Again:
            if self.verbose:
                print(f"[IsaacRealtimeStatePublisher] warning: dropped {topic} frame")

    def _build_debug_payload(
        self,
        *,
        backend: "IsaacSimulationBackend",
        state: IsaacRobotState,
        timestamp: float,
    ) -> dict[str, Any]:
        body_q = self._sonic_order(state.joint_position, backend)
        body_dq = self._sonic_order(state.joint_velocity, backend)
        base_quat = _fit_float_list(state.root_quaternion, 4, fill=0.0)
        if len(base_quat) == 4 and not any(base_quat):
            base_quat = [1.0, 0.0, 0.0, 0.0]
        base_ang_vel = _fit_float_list(state.root_angular_velocity, 3, fill=0.0)
        base_pos = _fit_float_list(state.root_position, 3, fill=0.0)
        zero3 = [0.0, 0.0, 0.0]
        zero4 = [1.0, 0.0, 0.0, 0.0]
        zero7 = [0.0] * 7
        zero29 = [0.0] * len(SONIC_BODY_JOINT_NAMES)
        return {
            "control_loop_type": "cpp",
            "index": int(self.index),
            "ros_timestamp": float(timestamp),
            "base_quat": base_quat,
            "base_ang_vel": base_ang_vel,
            "body_torso_quat": base_quat,
            "body_torso_ang_vel": base_ang_vel,
            "body_q": body_q,
            "body_dq": body_dq,
            "left_hand_q": zero7,
            "left_hand_dq": zero7,
            "right_hand_q": zero7,
            "right_hand_dq": zero7,
            "last_action": zero29,
            "last_left_hand_action": zero7,
            "last_right_hand_action": zero7,
            "token_state": [],
            "motor_temperature": [0.0] * 58,
            "base_trans_target": base_pos,
            "base_quat_target": base_quat,
            "body_q_target": body_q,
            "base_trans_measured": base_pos,
            "base_quat_measured": base_quat,
            "body_q_measured": body_q,
            "left_hand_q_measured": zero7,
            "right_hand_q_measured": zero7,
            "vr_3point_position": zero3 * 3,
            "vr_3point_orientation": zero4 * 3,
            "vr_3point_compliance": zero3,
        }

    def _sonic_order(self, values: list[float], backend: "IsaacSimulationBackend") -> list[float]:
        if backend.joint_mapping is None:
            return _fit_float_list(values, len(SONIC_BODY_JOINT_NAMES), fill=0.0)
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        ordered = []
        for isaac_index in backend.joint_mapping.sonic_body_to_isaac_dof:
            ordered.append(float(array[isaac_index]) if isaac_index < array.size else 0.0)
        return ordered


@dataclass
class IsaacElasticBandState:
    """MuJoCo ElasticBand runtime equivalent for Isaac.

    Mirrors ``gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge.ElasticBand``:
    force and torque are applied to the target rigid body in world coordinates.
    """

    kp_pos: float = 10000.0
    kd_pos: float = 1000.0
    kp_ang: float = 1000.0
    kd_ang: float = 10.0
    point: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    length: float = 0.0
    enabled: bool = True
    target_link: str = ""
    target_prim_path: str = ""
    force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    last_error: str | None = None

    def compute(
        self,
        *,
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta_x = self.point - position
        force = self.kp_pos * (delta_x + np.asarray([0.0, 0.0, self.length], dtype=np.float32))
        force = force + self.kd_pos * (-linear_velocity)
        rotvec = _quat_wxyz_to_rotvec(orientation_wxyz)
        torque = -self.kp_ang * rotvec - self.kd_ang * angular_velocity
        self.force = np.asarray(force, dtype=np.float32)
        self.torque = np.asarray(torque, dtype=np.float32)
        return self.force, self.torque

    def as_dict(self) -> dict[str, Any]:
        return {
            "elastic_band_enabled": bool(self.enabled),
            "elastic_band_target_link": self.target_link,
            "elastic_band_target_prim_path": self.target_prim_path,
            "elastic_band_anchor": [float(x) for x in self.point],
            "elastic_band_length": float(self.length),
            "elastic_band_force": [float(x) for x in self.force],
            "elastic_band_torque": [float(x) for x in self.torque],
            "elastic_band_last_error": self.last_error,
            "elastic_band_max_force": None,
        }


@dataclass
class IsaacJointMappingResult:
    """Mapping from Sonic body order to Isaac articulation DOF indices."""

    sonic_body_joint_names: list[str]
    isaac_dof_names: list[str]
    sonic_body_to_isaac_dof: list[int]

    def as_rows(self) -> list[dict[str, Any]]:
        rows = []
        for sonic_idx, isaac_idx in enumerate(self.sonic_body_to_isaac_dof):
            rows.append(
                {
                    "sonic_body_index": sonic_idx,
                    "sonic_body_joint": self.sonic_body_joint_names[sonic_idx],
                    "isaac_dof_index": isaac_idx,
                    "isaac_dof_name": self.isaac_dof_names[isaac_idx],
                }
            )
        return rows


@dataclass
class IsaacRobotState:
    """Small serializable snapshot of the Isaac G1 articulation state."""

    joint_position: list[float] = field(default_factory=list)
    joint_velocity: list[float] = field(default_factory=list)
    root_position: list[float] = field(default_factory=list)
    root_quaternion: list[float] = field(default_factory=list)
    root_linear_velocity: list[float] = field(default_factory=list)
    root_angular_velocity: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "joint_position": self.joint_position,
            "joint_velocity": self.joint_velocity,
            "root_position": self.root_position,
            "root_quaternion": self.root_quaternion,
            "root_linear_velocity": self.root_linear_velocity,
            "root_angular_velocity": self.root_angular_velocity,
        }


@dataclass
class IsaacBackspaceSnapshot:
    """State restored by Isaac's MuJoCo-window Backspace equivalent."""

    source: str
    root_position: np.ndarray
    root_quaternion: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_linear_velocity: np.ndarray
    root_angular_velocity: np.ndarray
    scene_xform_ops: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class IsaacSimulationBackend(SimulationBackend):
    """Minimal Isaac Sim backend for G1 load/reset/state-read validation."""

    def __init__(
        self,
        robot_model: str | None = None,
        usd_path: str | Path | None = None,
        prim_path: str | None = None,
        articulation_root_path: str | None = None,
        headless: bool = True,
        physics_dt: float = 1.0 / 200.0,
        rendering_dt: float = 1.0 / 60.0,
        startup_steps: int = 5,
        dds_bridge: Any | None = None,
        contact_config: IsaacContactConfig | None = None,
        passive_config: PassiveDynamicsConfig | None = None,
        scene_layer_path: str | Path | None = None,
        collision_layer_path: str | Path | None = None,
    ):
        self.robot_asset = get_robot_asset(robot_model)
        self.robot_model = self.robot_asset.name
        self.usd_path = Path(usd_path).expanduser().resolve() if usd_path else self.robot_asset.usd_path
        self.scene_layer_path = (
            Path(scene_layer_path).expanduser().resolve() if scene_layer_path else None
        )
        self.collision_layer_path = (
            Path(collision_layer_path).expanduser().resolve() if collision_layer_path else None
        )
        self.prim_path = prim_path or self.robot_asset.prim_path
        self.requested_articulation_root_path = articulation_root_path or self.robot_asset.articulation_root
        self.headless = headless
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt
        self.startup_steps = startup_steps
        self.contact_config = contact_config
        self.passive_config = passive_config

        self.simulation_app = None
        self.world = None
        self.ground_plane = None
        self.articulation = None
        self.articulation_root_path: str | None = None
        self.joint_mapping: IsaacJointMappingResult | None = None
        self.dds_bridge = dds_bridge
        self.contact_report: dict[str, Any] = {}
        self.passive_report: dict[str, Any] = {}
        self.scene_layer_report: dict[str, Any] = {}
        self.collision_layer_report: dict[str, Any] = {}
        self.robot_asset_report: dict[str, Any] = self.robot_asset.as_dict()
        self.joint_mapping_report: dict[str, Any] = {}
        self.image_publish_process: IsaacImagePublisher | None = None
        self.realtime_state_publisher: IsaacRealtimeStatePublisher | None = None
        self.keyboard_subscriber: IsaacKeyboardCommandSubscriber | None = None
        self.elastic_band = IsaacElasticBandState(
            target_link=Path(self.robot_asset.root_body_prim_path).name,
            target_prim_path=self.robot_asset.root_body_prim_path,
        )
        self._elastic_band_rigid_prim = None
        self._elastic_band_trace_path: Path | None = None
        self._elastic_band_trace_header_written = False
        self._reset_trace_path: Path | None = None
        self._reset_trace_header_written = False
        self._backspace_snapshot: IsaacBackspaceSnapshot | None = None
        self._control_lock = RLock()
        self._running = False
        self._sim_time = 0.0

    def start(self) -> None:
        """Launch Isaac, load G1 USD, initialize physics, and validate joints."""

        if self._running:
            return

        if not self.usd_path.exists():
            raise FileNotFoundError(f"G1 USD not found: {self.usd_path}")
        if self.usd_path == self.robot_asset.usd_path:
            self.robot_asset_report = self.robot_asset.as_dict()
        else:
            self.robot_asset_report = self.robot_asset.as_dict()
            self.robot_asset_report["usd_path_override"] = str(self.usd_path)

        # Isaac imports must happen after SimulationApp starts and must stay
        # lazy so normal Python can import this module for static checks.
        from isaacsim import SimulationApp

        self.simulation_app = SimulationApp({"headless": self.headless})

        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from isaacsim.core.utils.prims import get_articulation_root_api_prim_path
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, update_stage
        from pxr import UsdPhysics

        self.world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        self.world.set_simulation_dt(physics_dt=self.physics_dt, rendering_dt=self.rendering_dt)
        ground_kwargs = {}
        if self.contact_config is not None:
            ground_kwargs = {
                "static_friction": self.contact_config.static_friction,
                "dynamic_friction": self.contact_config.dynamic_friction,
                "restitution": self.contact_config.restitution,
            }
        self.ground_plane = self.world.scene.add_default_ground_plane(**ground_kwargs)
        if self.contact_config is not None:
            self.configure_contact_parameters(self.contact_config, phase="ground")

        if self.scene_layer_path is not None:
            self.apply_scene_layer(self.scene_layer_path)

        print(f"[IsaacBackend] robot model: {self.robot_model}")
        print(f"[IsaacBackend] robot_usd_path: {self.usd_path}")
        print(f"[IsaacBackend] loading USD: {self.usd_path}")
        print(f"[IsaacBackend] reference prim: {self.prim_path}")
        print(f"[IsaacBackend] configured articulation root: {self.requested_articulation_root_path}")
        add_reference_to_stage(usd_path=str(self.usd_path), prim_path=self.prim_path)
        if self.collision_layer_path is not None:
            self.apply_collision_layer(self.collision_layer_path)

        for _ in range(60):
            update_stage()
            if omni.usd.get_context().get_stage_loading_status()[2] == 0:
                break

        self.articulation_root_path = self._resolve_articulation_root_path(
            get_articulation_root_api_prim_path=get_articulation_root_api_prim_path,
            get_current_stage=get_current_stage,
            usd_physics=UsdPhysics,
        )
        print(f"[IsaacBackend] articulation root: {self.articulation_root_path}")

        self.articulation = self.world.scene.add(
            SingleArticulation(prim_path=self.articulation_root_path, name="sonic_g1")
        )
        self._elastic_band_rigid_prim = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=self.elastic_band.target_prim_path,
                name="sonic_g1_elastic_band_target",
            )
        )
        self.world.reset()

        for _ in range(max(0, self.startup_steps)):
            self.world.step(render=not self.headless)
            self._sim_time += self.physics_dt

        if self.contact_config is not None:
            self.configure_contact_parameters(self.contact_config, phase="articulation")
        if self.passive_config is not None:
            self.configure_passive_dynamics(self.passive_config, phase="startup")

        self._running = True
        self._validate_joint_mapping()
        self._validate_configured_prims()
        self._capture_backspace_snapshot(source="isaac_world_reset")
        self._print_robot_summary()

    def apply_scene_layer(self, scene_layer_path: str | Path) -> dict[str, Any]:
        """Load an optional SonicStar task scene USD layer.

        The layer is sublayered into the anonymous runtime stage before the G1
        USD is referenced. Robot USD files and collision calibration layers are
        left separate.
        """

        layer_path = Path(scene_layer_path).expanduser().resolve()
        report: dict[str, Any] = {
            "requested_layer": str(layer_path),
            "applied": False,
            "errors": [],
        }
        if not layer_path.exists():
            report["errors"].append(f"scene layer not found: {layer_path}")
            self.scene_layer_report = report
            raise FileNotFoundError(report["errors"][-1])
        try:
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import Sdf

            layer = Sdf.Layer.FindOrOpen(str(layer_path))
            if layer is None:
                raise RuntimeError(f"failed to open scene layer: {layer_path}")
            stage = get_current_stage()
            root_layer = stage.GetRootLayer()
            sublayers = list(root_layer.subLayerPaths)
            if str(layer_path) not in sublayers:
                root_layer.subLayerPaths.append(str(layer_path))
            report["applied"] = True
            report["root_layer"] = root_layer.identifier
            report["sublayer_count"] = len(root_layer.subLayerPaths)
            report["layer_default_prim"] = str(layer.defaultPrim)
            report["layer_root_prims"] = [str(prim.path) for prim in layer.rootPrims]
        except Exception as exc:
            report["errors"].append(repr(exc))
            self.scene_layer_report = report
            raise

        self.scene_layer_report = report
        print(f"[IsaacBackend] scene layer: {report}")
        return report

    def apply_collision_layer(self, collision_layer_path: str | Path) -> dict[str, Any]:
        """Load an optional Isaac-only USD layer for collision calibration.

        The layer is sublayered into the anonymous runtime stage. Source robot
        USD files are not edited or saved.
        """

        layer_path = Path(collision_layer_path).expanduser().resolve()
        report: dict[str, Any] = {
            "requested_layer": str(layer_path),
            "applied": False,
            "errors": [],
        }
        if not layer_path.exists():
            report["errors"].append(f"collision layer not found: {layer_path}")
            self.collision_layer_report = report
            raise FileNotFoundError(report["errors"][-1])
        try:
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import Sdf

            layer = Sdf.Layer.FindOrOpen(str(layer_path))
            if layer is None:
                raise RuntimeError(f"failed to open collision layer: {layer_path}")
            stage = get_current_stage()
            root_layer = stage.GetRootLayer()
            sublayers = list(root_layer.subLayerPaths)
            if str(layer_path) not in sublayers:
                root_layer.subLayerPaths.append(str(layer_path))
            report["applied"] = True
            report["root_layer"] = root_layer.identifier
            report["sublayer_count"] = len(root_layer.subLayerPaths)
            if self.prim_path != "/World/G1":
                report["warning"] = (
                    "default calibration layer authors /World/G1 paths; "
                    f"current prim_path={self.prim_path!r}"
                )
        except Exception as exc:
            report["errors"].append(repr(exc))
            self.collision_layer_report = report
            raise

        self.collision_layer_report = report
        print(f"[IsaacBackend] collision layer: {report}")
        return report

    def stop(self) -> None:
        self.close()

    def close(self) -> None:
        """Close Isaac resources."""

        self._running = False
        if self.keyboard_subscriber is not None:
            self.keyboard_subscriber.stop()
            self.keyboard_subscriber = None
        self.stop_realtime_state_publisher()
        self.stop_image_publish()
        if self.world is not None:
            try:
                self.world.stop()
            except Exception:
                pass
        if self.simulation_app is not None:
            self.simulation_app.close()
        self.world = None
        self.articulation = None
        self.simulation_app = None

    def is_running(self) -> bool:
        return self._running

    def reset(self) -> None:
        """Reset the Isaac world and revalidate the articulation handle."""

        if self.world is None or self.articulation is None:
            raise RuntimeError("Isaac articulation must exist before configuring passive dynamics")
        with self._control_lock:
            self.world.reset()
            for _ in range(max(1, self.startup_steps)):
                self.world.step(render=not self.headless)
                self._sim_time += self.physics_dt
            self._validate_joint_mapping()
            self._capture_backspace_snapshot(source="isaac_full_world_reset")

    def initialize_articulation_state_for_control(
        self,
        *,
        joint_positions: Any,
        root_position: Any,
        root_orientation: Any,
        configure_drives: Any | None = None,
        target_writer: Any | None = None,
        state_audit_writer: Any | None = None,
        reset_world: bool = True,
        startup_warmup_steps: int = 20,
        state_commit_steps: int = 20,
        target_commit_steps: int = 20,
        render: bool = False,
    ) -> dict[str, Any]:
        """Canonical Isaac control initialization sequence.

        The sequence is intentionally separated into reset, drive config,
        state teleport, velocity clearing, state commit, target activation,
        and target/contact settle.  It does not change DDS, WBC, FluxVLA, or
        the source USD.
        """

        self._require_started()
        if self.world is None or self.articulation is None:
            raise RuntimeError("Isaac articulation must exist before control initialization")

        import numpy as np

        joint_positions_array = np.asarray(joint_positions, dtype=np.float32)
        root_position_array = np.asarray(root_position, dtype=np.float32)
        root_orientation_array = np.asarray(root_orientation, dtype=np.float32)
        report: dict[str, Any] = {
            "sequence": [],
            "reset_world": bool(reset_world),
            "startup_warmup_steps": int(startup_warmup_steps),
            "state_commit_steps": int(state_commit_steps),
            "target_commit_steps": int(target_commit_steps),
        }

        def step_many(label: str, count: int) -> None:
            count = max(0, int(count))
            for _ in range(count):
                self.world.step(render=render)
                self._sim_time += self.physics_dt
            report["sequence"].append({"operation": label, "steps": count})

        if reset_world:
            with self._control_lock:
                self.world.reset()
                self._sim_time = 0.0
                self._validate_joint_mapping()
                report["sequence"].append({"operation": "reset", "steps": 0})

        step_many("startup_warmup", startup_warmup_steps)

        if configure_drives is not None:
            drive_rows = configure_drives()
            report["drive_rows"] = len(drive_rows) if drive_rows is not None else 0
            report["sequence"].append({"operation": "configure_drives", "steps": 0})

        with self._control_lock:
            self.articulation.set_world_pose(position=root_position_array, orientation=root_orientation_array)
            report["sequence"].append({"operation": "set_world_pose", "steps": 0})

            self.articulation.set_joint_positions(joint_positions_array)
            report["sequence"].append({"operation": "set_joint_positions", "steps": 0})
            if state_audit_writer is not None:
                state_audit_writer(
                    label="T0_after_set_joint_positions",
                    simulation_time=self._sim_time,
                    state=self.read_joint_state(),
                    target_joint_positions_by_isaac=joint_positions_array,
                )

            zero_joint_velocities = np.zeros_like(joint_positions_array, dtype=np.float32)
            zero_root_velocity = np.zeros(3, dtype=np.float32)
            self.articulation.set_joint_velocities(zero_joint_velocities)
            self.articulation.set_linear_velocity(zero_root_velocity)
            self.articulation.set_angular_velocity(zero_root_velocity)
            self._capture_backspace_snapshot(
                source="canonical_control_init",
                root_position=root_position_array,
                root_quaternion=root_orientation_array,
                joint_position=joint_positions_array,
                joint_velocity=zero_joint_velocities,
                root_linear_velocity=zero_root_velocity,
                root_angular_velocity=zero_root_velocity,
            )
            report["sequence"].append({"operation": "clear_velocities", "steps": 0})

        step_many("state_commit", state_commit_steps)
        if state_audit_writer is not None:
            state_audit_writer(
                label="T1_after_state_commit",
                simulation_time=self._sim_time,
                state=self.read_joint_state(),
                target_joint_positions_by_isaac=joint_positions_array,
            )

        if target_writer is not None:
            target_writer(joint_positions_array)
        else:
            articulation_view = getattr(self.articulation, "_articulation_view", None)
            view_set_targets = getattr(articulation_view, "set_joint_position_targets", None)
            if view_set_targets is not None:
                view_set_targets(np.expand_dims(joint_positions_array, axis=0))
            elif hasattr(self.articulation, "set_joint_position_targets"):
                self.articulation.set_joint_position_targets(joint_positions_array)
            else:
                raise RuntimeError("Isaac articulation does not expose a joint target writer")
        report["sequence"].append({"operation": "set_joint_position_targets", "steps": 0})

        step_many("target_commit", target_commit_steps)
        return report

    def step(self, render: bool | None = None) -> None:
        self._require_started()
        should_render = (not self.headless) if render is None else render
        with self._control_lock:
            self._apply_elastic_band_force()
            self.world.step(render=should_render)
            self._sim_time += self.physics_dt

    def render(self) -> None:
        self.step(render=True)

    def start_image_publish_subprocess(
        self,
        *,
        start_method: str = "spawn",
        camera_port: int = 5555,
        camera_name: str = "ego_view",
        image_dt: float = 1.0 / 30.0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        """Start Isaac's MuJoCo-compatible camera publisher.

        The name mirrors the MuJoCo simulator API even though Isaac publishes
        in-process; the wire protocol is identical to MuJoCo's 5555 publisher.
        ``start_method`` is accepted for API compatibility and intentionally
        unused.
        """

        del start_method
        self._require_started()
        if self.image_publish_process is not None:
            return
        publisher = IsaacImagePublisher(
            port=camera_port,
            camera_name=camera_name,
            width=width,
            height=height,
            image_dt=image_dt,
        )
        publisher.start(self)
        self.image_publish_process = publisher

    def stop_image_publish(self) -> None:
        if self.image_publish_process is not None:
            self.image_publish_process.stop()
        self.image_publish_process = None

    def start_realtime_state_publisher(
        self,
        *,
        port: int = 5557,
        topic: str = "g1_debug",
    ) -> None:
        """Start optional Isaac-side 5557 publisher.

        Do not enable this while ``g1_deploy_onnx_ref`` is running, because the
        C++ process normally owns 5557 and emits the canonical GR00T payload.
        """

        self._require_started()
        if self.realtime_state_publisher is not None:
            return
        publisher = IsaacRealtimeStatePublisher(port=port, topic=topic)
        publisher.start(self)
        self.realtime_state_publisher = publisher

    def stop_realtime_state_publisher(self) -> None:
        if self.realtime_state_publisher is not None:
            self.realtime_state_publisher.stop()
        self.realtime_state_publisher = None

    def spin(
        self,
        *,
        frequency_hz: int = 500,
        duration_s: float | None = None,
        dds_bridge: Any | None = None,
        log_dir: str | Path | None = None,
        control_server: Any | None = None,
    ) -> Any:
        """Run the Isaac DDS bridge loop.

        Loop body:

            physics step -> read joint state -> publish lowstate/IMU -> receive lowcmd -> apply target

        This does not start FluxVLA, WBC, or evaluation logic. Optional camera
        and state publishers can be started before entering the loop.
        """

        self._require_started()
        if frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")

        bridge = dds_bridge or self.dds_bridge
        owns_bridge = bridge is None
        if bridge is None:
            from gear_sonic.robot_interface.isaac_unitree_bridge import IsaacUnitreeBridge

            bridge = IsaacUnitreeBridge(
                articulation=self.articulation,
                joint_mapping=self.joint_mapping,
                log_dir=log_dir,
            )
        else:
            bridge.articulation = self.articulation
            bridge.joint_mapping = self.joint_mapping
            if log_dir is not None and getattr(bridge, "log_dir", None) is None:
                bridge.log_dir = Path(log_dir)

        self.dds_bridge = bridge
        bridge.start()
        self.keyboard_subscriber = IsaacKeyboardCommandSubscriber(self)
        self.keyboard_subscriber.start()
        if log_dir is not None:
            self._elastic_band_trace_path = Path(log_dir) / "elastic_band_trace.csv"
            self._reset_trace_path = Path(log_dir) / "reset_trace.csv"
        if control_server is not None and not getattr(control_server, "_running", False):
            control_server.start(as_thread=False, serve=False)

        period_s = 1.0 / float(frequency_hz)
        start_time = time.monotonic()
        next_tick = start_time
        active_command = None
        try:
            while self._running:
                if duration_s is not None and time.monotonic() - start_time >= duration_s:
                    break

                render_for_publishers = self.image_publish_process is not None
                self.step(render=(not self.headless) or render_for_publishers)
                performance_monitor = getattr(bridge, "performance_monitor", None)
                if performance_monitor is not None:
                    performance_monitor.record_physics_step()
                state = self.read_joint_state()
                if self.image_publish_process is not None:
                    self.image_publish_process.maybe_publish(self)
                if self.realtime_state_publisher is not None:
                    self.realtime_state_publisher.publish(self, state)
                obs = state.as_dict()
                obs["time"] = self._sim_time
                bridge.publish_low_state(obs)
                command = bridge.receive_low_cmd()
                if command is not None:
                    active_command = command
                elif active_command is not None:
                    active_command = bridge.peek_low_cmd() or active_command

                if active_command is not None:
                    bridge.apply_joint_command(active_command)

                if control_server is not None:
                    # Process 5590 control requests in the Isaac main loop.
                    # This avoids Python-thread starvation while preserving
                    # the same JSON/ZMQ protocol used by MuJoCo.
                    control_server.poll_once()
                if self.keyboard_subscriber is not None:
                    # Mirror MuJoCo viewer-only keys from send_keyboard_cmd.py:
                    # 9 toggles ElasticBand, Backspace performs runtime reset.
                    self.keyboard_subscriber.poll_once()

                next_tick += period_s
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
                    # Keep the 5590 eval-control thread responsive if the
                    # Isaac/DDS step loop overruns its nominal period.
                    time.sleep(0)
        finally:
            if self.keyboard_subscriber is not None:
                self.keyboard_subscriber.stop()
                self.keyboard_subscriber = None
            if owns_bridge:
                bridge.stop()

        return bridge.stats

    def read_joint_state(self) -> IsaacRobotState:
        """Read joint and root state from the Isaac articulation."""

        self._require_started()
        joint_position = self._tolist(self.articulation.get_joint_positions())
        joint_velocity = self._tolist(self.articulation.get_joint_velocities())
        root_position, root_quaternion = self.articulation.get_world_pose()
        root_linear_velocity = self._tolist(self.articulation.get_linear_velocity())
        root_angular_velocity = self._tolist(self.articulation.get_angular_velocity())

        return IsaacRobotState(
            joint_position=self._tolist(rootless(joint_position)),
            joint_velocity=self._tolist(rootless(joint_velocity)),
            root_position=self._tolist(root_position),
            root_quaternion=self._tolist(root_quaternion),
            root_linear_velocity=root_linear_velocity,
            root_angular_velocity=root_angular_velocity,
        )

    def configure_passive_dynamics(
        self,
        passive_config: PassiveDynamicsConfig | None = None,
        *,
        phase: str = "runtime",
    ) -> dict[str, Any]:
        """Apply MuJoCo-style passive dynamics to the Isaac runtime stage.

        This configures three runtime equivalents:

        - MuJoCo joint damping -> optional Isaac drive damping offset.
        - MuJoCo armature -> ``physxJoint:armature`` USD attribute.
        - MuJoCo frictionloss -> ``physxJoint:jointFriction`` USD attribute.

        The source USD files are not saved or modified.
        """

        if self.world is None or self.articulation is None:
            raise RuntimeError("Isaac articulation must exist before configuring passive dynamics")
        config = passive_config or self.passive_config
        if config is None:
            raise ValueError("passive_config must be provided")

        rows = build_passive_mapping(list(self.articulation.dof_names), config=config)
        before = self._read_passive_parameters(rows)
        applied: list[str] = []
        errors: list[str] = []

        if config.apply_drive_damping_offset:
            try:
                articulation_view = getattr(self.articulation, "_articulation_view", None)
                if articulation_view is None:
                    raise RuntimeError("Isaac articulation does not expose _articulation_view")
                kps, kds = articulation_view.get_gains()
                import numpy as np

                kps_array = np.asarray(kps, dtype=np.float32)
                kds_array = np.asarray(kds, dtype=np.float32)
                joint_indices = np.asarray([row.isaac_index for row in rows], dtype=np.int64)
                new_kds = np.asarray(
                    [[float(kds_array[0, row.isaac_index]) + row.damping_offset for row in rows]],
                    dtype=np.float32,
                )
                articulation_view.set_gains(
                    kps=kps_array[:, joint_indices],
                    kds=new_kds,
                    joint_indices=joint_indices,
                )
                applied.append("drive_damping_offset")
            except Exception as exc:
                errors.append(f"drive_damping_offset={exc!r}")

        try:
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import Sdf

            stage = get_current_stage()
            joint_prims = self._joint_prim_by_name(stage)
            for row in rows:
                prim = joint_prims.get(row.isaac_joint)
                if prim is None:
                    errors.append(f"{row.isaac_joint}:joint_prim_not_found")
                    continue
                if config.apply_joint_armature:
                    attr = prim.GetAttribute("physxJoint:armature")
                    if not attr:
                        attr = prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float)
                    attr.Set(float(row.armature))
                    applied.append(f"{row.isaac_joint}:armature")
                if config.apply_joint_friction:
                    attr = prim.GetAttribute("physxJoint:jointFriction")
                    if not attr:
                        attr = prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float)
                    attr.Set(float(row.joint_friction))
                    applied.append(f"{row.isaac_joint}:joint_friction")
        except Exception as exc:
            errors.append(f"usd_joint_attributes={exc!r}")

        after = self._read_passive_parameters(rows)
        report = {
            "phase": phase,
            "config": config.as_dict(),
            "before": before,
            "after": after,
            "applied_count": len(applied),
            "applied_preview": applied[:12],
            "errors": errors,
        }
        self.passive_report = report
        print(f"[IsaacBackend] passive config phase={phase}: {config.as_dict()}")
        print(f"[IsaacBackend] passive applied_count={len(applied)} errors={errors[:6]}")
        return report

    def configure_contact_parameters(
        self,
        contact_config: IsaacContactConfig | None = None,
        *,
        phase: str = "runtime",
    ) -> dict[str, Any]:
        """Apply Isaac runtime contact parameters and return before/after values.

        This intentionally only touches Isaac runtime material/solver settings.
        It does not edit USD files, MuJoCo XML, DDS, or WBC code.
        """

        config = contact_config or self.contact_config
        if config is None:
            raise ValueError("contact_config must be provided")

        before = self._read_contact_parameters()
        applied: list[str] = []
        errors: list[str] = []

        if self.ground_plane is not None:
            try:
                from isaacsim.core.api.materials import PhysicsMaterial

                material = PhysicsMaterial(
                    prim_path="/World/Physics_Materials/sonic_ground_contact_material",
                    static_friction=config.static_friction,
                    dynamic_friction=config.dynamic_friction,
                    restitution=config.restitution,
                )
                self.ground_plane.apply_physics_material(material)
                applied.extend(["ground_static_friction", "ground_dynamic_friction", "ground_restitution"])
            except Exception as exc:
                errors.append(f"ground_material={exc!r}")

            ground_collision = getattr(self.ground_plane, "collision_geometry_prim", None)
            if config.contact_offset is not None and ground_collision is not None:
                try:
                    ground_collision.set_contact_offset(float(config.contact_offset))
                    applied.append("ground_contact_offset")
                except Exception as exc:
                    errors.append(f"ground_contact_offset={exc!r}")

            if config.rest_offset is not None and ground_collision is not None:
                try:
                    ground_collision.set_rest_offset(float(config.rest_offset))
                    applied.append("ground_rest_offset")
                except Exception as exc:
                    errors.append(f"ground_rest_offset={exc!r}")

        if self.articulation is not None:
            if config.solver_position_iteration_count is not None:
                try:
                    self.articulation.set_solver_position_iteration_count(
                        int(config.solver_position_iteration_count)
                    )
                    applied.append("articulation_solver_position_iteration_count")
                except Exception as exc:
                    errors.append(f"solver_position_iteration_count={exc!r}")

            if config.solver_velocity_iteration_count is not None:
                try:
                    self.articulation.set_solver_velocity_iteration_count(
                        int(config.solver_velocity_iteration_count)
                    )
                    applied.append("articulation_solver_velocity_iteration_count")
                except Exception as exc:
                    errors.append(f"solver_velocity_iteration_count={exc!r}")

        after = self._read_contact_parameters()
        report = {
            "phase": phase,
            "config": config.as_dict(),
            "before": before,
            "after": after,
            "applied": applied,
            "errors": errors,
        }
        self.contact_report = report
        print(f"[IsaacBackend] contact config phase={phase}: {config.as_dict()}")
        print(f"[IsaacBackend] contact before: {before}")
        print(f"[IsaacBackend] contact after: {after}")
        if errors:
            print(f"[IsaacBackend] contact config warnings: {errors}")
        return report

    def _read_contact_parameters(self) -> dict[str, Any]:
        values: dict[str, Any] = {}

        if self.ground_plane is not None:
            try:
                material = self.ground_plane.get_applied_physics_material()
                values["ground_static_friction"] = material.get_static_friction()
                values["ground_dynamic_friction"] = material.get_dynamic_friction()
                values["ground_restitution"] = material.get_restitution()
            except Exception as exc:
                values["ground_material_error"] = repr(exc)

            ground_collision = getattr(self.ground_plane, "collision_geometry_prim", None)
            for name, getter in (
                ("ground_contact_offset", "get_contact_offset"),
                ("ground_rest_offset", "get_rest_offset"),
            ):
                if ground_collision is not None and hasattr(ground_collision, getter):
                    try:
                        values[name] = getattr(ground_collision, getter)()
                    except Exception as exc:
                        values[f"{name}_error"] = repr(exc)
                else:
                    values[name] = "unavailable"

        if self.articulation is not None:
            for name, getter in (
                ("articulation_solver_position_iteration_count", "get_solver_position_iteration_count"),
                ("articulation_solver_velocity_iteration_count", "get_solver_velocity_iteration_count"),
            ):
                if hasattr(self.articulation, getter):
                    try:
                        values[name] = int(getattr(self.articulation, getter)())
                    except Exception as exc:
                        values[f"{name}_error"] = repr(exc)
                else:
                    values[name] = "unavailable"

        return values

    def _read_passive_parameters(self, rows: list[Any]) -> dict[str, Any]:
        values: dict[str, Any] = {"rows": []}

        drive_damping_by_index: dict[int, float] = {}
        try:
            articulation_view = getattr(self.articulation, "_articulation_view", None)
            if articulation_view is not None:
                _, kds = articulation_view.get_gains()
                for row in rows:
                    drive_damping_by_index[row.isaac_index] = float(kds[0, row.isaac_index])
        except Exception as exc:
            values["drive_damping_error"] = repr(exc)

        joint_prims: dict[str, Any] = {}
        try:
            from isaacsim.core.utils.stage import get_current_stage

            joint_prims = self._joint_prim_by_name(get_current_stage())
        except Exception as exc:
            values["joint_prim_error"] = repr(exc)

        for row in rows:
            prim = joint_prims.get(row.isaac_joint)
            armature = None
            joint_friction = None
            prim_path = None
            if prim is not None:
                prim_path = str(prim.GetPath())
                armature = _get_usd_attr_value(prim, "physxJoint:armature")
                joint_friction = _get_usd_attr_value(prim, "physxJoint:jointFriction")
            values["rows"].append(
                {
                    "sonic_joint": row.sonic_joint,
                    "isaac_joint": row.isaac_joint,
                    "isaac_index": row.isaac_index,
                    "prim_path": prim_path,
                    "drive_damping": drive_damping_by_index.get(row.isaac_index),
                    "armature": armature,
                    "joint_friction": joint_friction,
                }
            )
        return values

    def _joint_prim_by_name(self, stage: Any) -> dict[str, Any]:
        joint_names = set(list(self.articulation.dof_names) if self.articulation is not None else [])
        result: dict[str, Any] = {}
        prefix = self.prim_path.rstrip("/") + "/"
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path != self.prim_path and not path.startswith(prefix):
                continue
            name = prim.GetName()
            if name in joint_names:
                result[name] = prim
        return result

    def get_task_state(self, task_name: str = "cup_to_bin") -> dict[str, Any]:
        """Return a minimal state snapshot for smoke tests.

        Full cup/bin task state is intentionally not implemented in this phase.
        """

        state = self.read_joint_state()
        root_height = state.root_position[2] if len(state.root_position) >= 3 else None
        result = {
            "task_name": task_name,
            "sim_time": self._sim_time,
            "robot_root_pos": state.root_position,
            "robot_root_quat": state.root_quaternion,
            "robot_root_height": root_height,
            "band_enabled": bool(self.elastic_band.enabled),
            "cup_pos": None,
            "bin_pos": None,
            "todo": "cup/bin task state is not implemented in Isaac milestone 1",
        }
        result.update(self.elastic_band.as_dict())
        return result

    def band_on(self) -> bool:
        return self.set_elastic_band_enabled(True)

    def band_off(self) -> bool:
        return self.set_elastic_band_enabled(False)

    def keyboard(self, key: str) -> None:
        normalized = str(key).lower()
        if normalized == "9":
            self.set_elastic_band_enabled(not self.elastic_band.enabled)
            return
        if normalized in ("backspace", "\b", "\x7f"):
            self.mujoco_backspace_reset()
            return
        print(f"[IsaacBackend] keyboard command not handled by Isaac backend: {key!r}")

    def set_elastic_band_enabled(self, enabled: bool) -> bool:
        with self._control_lock:
            self.elastic_band.enabled = bool(enabled)
            if not self.elastic_band.enabled:
                self.elastic_band.force = np.zeros(3, dtype=np.float32)
                self.elastic_band.torque = np.zeros(3, dtype=np.float32)
            print(f"ElasticBand enable: {self.elastic_band.enabled}")
        return True

    def mujoco_backspace_reset(self) -> None:
        """MuJoCo viewer Backspace equivalent, without restarting control processes.

        MuJoCo calls ``mj_resetData`` + clears ``xfrc_applied`` + ``mj_forward``.
        For Isaac, keep the running World/DDS/lowcmd bridge intact and restore
        the captured model/canonical initial robot state plus task-object xforms.
        ElasticBand enable state is intentionally preserved; only the currently
        applied force/torque cache is zeroed before the next physics step
        recomputes it, matching MuJoCo's ``xfrc_applied[:] = 0`` behavior.
        """

        self._require_started()
        if self.articulation is None:
            raise RuntimeError("Isaac articulation is unavailable")
        with self._control_lock:
            if self._backspace_snapshot is None:
                self._capture_backspace_snapshot(source="lazy_pre_backspace")
            assert self._backspace_snapshot is not None
            snapshot = self._backspace_snapshot
            before = self.read_joint_state().as_dict()
            band_before = bool(self.elastic_band.enabled)
            self._restore_scene_xform_ops(snapshot.scene_xform_ops)
            self.articulation.set_world_pose(
                position=np.asarray(snapshot.root_position, dtype=np.float32),
                orientation=np.asarray(snapshot.root_quaternion, dtype=np.float32),
            )
            self.articulation.set_joint_positions(np.asarray(snapshot.joint_position, dtype=np.float32))
            self.articulation.set_joint_velocities(np.asarray(snapshot.joint_velocity, dtype=np.float32))
            self.articulation.set_linear_velocity(np.asarray(snapshot.root_linear_velocity, dtype=np.float32))
            self.articulation.set_angular_velocity(np.asarray(snapshot.root_angular_velocity, dtype=np.float32))
            self.elastic_band.force = np.zeros(3, dtype=np.float32)
            self.elastic_band.torque = np.zeros(3, dtype=np.float32)
            after = self.read_joint_state().as_dict()
            self._write_reset_trace(
                command="keyboard_backspace",
                snapshot_source=snapshot.source,
                before=before,
                after=after,
                elastic_band_before=band_before,
                elastic_band_after=bool(self.elastic_band.enabled),
                object_xform_count=sum(len(v) for v in snapshot.scene_xform_ops.values()),
            )
            print(
                "[IsaacBackend] MuJoCo Backspace reset applied "
                f"snapshot={snapshot.source!r} elastic_band_preserved={self.elastic_band.enabled}"
            )

    def _capture_backspace_snapshot(
        self,
        *,
        source: str,
        root_position: Any | None = None,
        root_quaternion: Any | None = None,
        joint_position: Any | None = None,
        joint_velocity: Any | None = None,
        root_linear_velocity: Any | None = None,
        root_angular_velocity: Any | None = None,
    ) -> None:
        if self.articulation is None:
            return
        if root_position is None or root_quaternion is None:
            root_position, root_quaternion = self.articulation.get_world_pose()
        if joint_position is None:
            joint_position = self.articulation.get_joint_positions()
        if joint_velocity is None:
            joint_velocity = self.articulation.get_joint_velocities()
        if root_linear_velocity is None:
            root_linear_velocity = self.articulation.get_linear_velocity()
        if root_angular_velocity is None:
            root_angular_velocity = self.articulation.get_angular_velocity()

        self._backspace_snapshot = IsaacBackspaceSnapshot(
            source=str(source),
            root_position=np.asarray(root_position, dtype=np.float32).reshape(3).copy(),
            root_quaternion=np.asarray(root_quaternion, dtype=np.float32).reshape(4).copy(),
            joint_position=np.asarray(joint_position, dtype=np.float32).reshape(-1).copy(),
            joint_velocity=np.asarray(joint_velocity, dtype=np.float32).reshape(-1).copy(),
            root_linear_velocity=np.asarray(root_linear_velocity, dtype=np.float32).reshape(3).copy(),
            root_angular_velocity=np.asarray(root_angular_velocity, dtype=np.float32).reshape(3).copy(),
            scene_xform_ops=self._capture_scene_xform_ops(),
        )

    def _capture_scene_xform_ops(self) -> dict[str, list[dict[str, Any]]]:
        if self.scene_layer_path is None:
            return {}
        try:
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import UsdGeom

            stage = get_current_stage()
            root_path = "/World/SonicStarTask"
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim or not root_prim.IsValid():
                return {}
            result: dict[str, list[dict[str, Any]]] = {}
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if path != root_path and not path.startswith(root_path + "/"):
                    continue
                xformable = UsdGeom.Xformable(prim)
                if not xformable:
                    continue
                ops = []
                for op in xformable.GetOrderedXformOps():
                    attr = op.GetAttr()
                    ops.append({"attr_name": attr.GetName(), "value": attr.Get()})
                if ops:
                    result[path] = ops
            return result
        except Exception as exc:
            print(f"[IsaacBackend] warning: failed to capture scene xforms for Backspace: {exc!r}")
            return {}

    def _restore_scene_xform_ops(self, scene_xform_ops: dict[str, list[dict[str, Any]]]) -> None:
        if not scene_xform_ops:
            return
        try:
            from isaacsim.core.utils.stage import get_current_stage

            stage = get_current_stage()
            for path, ops in scene_xform_ops.items():
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    continue
                for op in ops:
                    attr = prim.GetAttribute(op["attr_name"])
                    if attr:
                        attr.Set(op["value"])
        except Exception as exc:
            print(f"[IsaacBackend] warning: failed to restore scene xforms for Backspace: {exc!r}")

    def _write_reset_trace(
        self,
        *,
        command: str,
        snapshot_source: str,
        before: dict[str, Any],
        after: dict[str, Any],
        elastic_band_before: bool,
        elastic_band_after: bool,
        object_xform_count: int,
    ) -> None:
        if self._reset_trace_path is None:
            return
        row = {
            "timestamp": f"{time.time():.9f}",
            "sim_time": f"{self._sim_time:.9f}",
            "physics_step": int(round(self._sim_time / self.physics_dt)) if self.physics_dt else 0,
            "command": command,
            "snapshot_source": snapshot_source,
            "root_pose_before": repr(
                {
                    "position": before.get("root_position"),
                    "quaternion": before.get("root_quaternion"),
                }
            ),
            "root_pose_after": repr(
                {
                    "position": after.get("root_position"),
                    "quaternion": after.get("root_quaternion"),
                }
            ),
            "joint_q_before": repr(before.get("joint_position")),
            "joint_q_after": repr(after.get("joint_position")),
            "joint_dq_before": repr(before.get("joint_velocity")),
            "joint_dq_after": repr(after.get("joint_velocity")),
            "root_linear_velocity_before": repr(before.get("root_linear_velocity")),
            "root_linear_velocity_after": repr(after.get("root_linear_velocity")),
            "root_angular_velocity_before": repr(before.get("root_angular_velocity")),
            "root_angular_velocity_after": repr(after.get("root_angular_velocity")),
            "object_xform_count": int(object_xform_count),
            "elastic_band_before": bool(elastic_band_before),
            "elastic_band_after": bool(elastic_band_after),
            "control_process_restarted": False,
            "dds_reconnected": False,
        }
        self._reset_trace_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._reset_trace_path.exists() or not self._reset_trace_header_written
        with self._reset_trace_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._reset_trace_header_written = True
            writer.writerow(row)

    def _apply_elastic_band_force(self) -> None:
        if not self.elastic_band.enabled:
            self._write_elastic_band_trace()
            return
        if self.articulation is None:
            return
        if self._elastic_band_rigid_prim is None:
            self.elastic_band.last_error = "elastic band rigid prim view is not initialized"
            self._write_elastic_band_trace()
            return
        try:
            root_position, root_quaternion = self.articulation.get_world_pose()
            position = np.asarray(root_position, dtype=np.float32).reshape(3)
            orientation = np.asarray(root_quaternion, dtype=np.float32).reshape(4)
            linear_velocity = np.asarray(self.articulation.get_linear_velocity(), dtype=np.float32).reshape(3)
            angular_velocity = np.asarray(self.articulation.get_angular_velocity(), dtype=np.float32).reshape(3)
            force, torque = self.elastic_band.compute(
                position=position,
                orientation_wxyz=orientation,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
            self._elastic_band_rigid_prim.apply_forces_and_torques_at_pos(
                forces=np.expand_dims(force, axis=0),
                torques=np.expand_dims(torque, axis=0),
                positions=np.expand_dims(position, axis=0),
                is_global=True,
            )
            self.elastic_band.last_error = None
        except Exception as exc:
            self.elastic_band.last_error = repr(exc)
        self._write_elastic_band_trace()

    def _write_elastic_band_trace(self) -> None:
        if self._elastic_band_trace_path is None:
            return
        row = {
            "timestamp": f"{time.time():.9f}",
            "sim_time": f"{self._sim_time:.9f}",
            "physics_step": int(round(self._sim_time / self.physics_dt)) if self.physics_dt else 0,
            "elastic_band_enabled": bool(self.elastic_band.enabled),
            "elastic_band_target_link": self.elastic_band.target_link,
            "elastic_band_target_prim_path": self.elastic_band.target_prim_path,
            "elastic_band_anchor": repr([float(x) for x in self.elastic_band.point]),
            "elastic_band_force": repr([float(x) for x in self.elastic_band.force]),
            "elastic_band_torque": repr([float(x) for x in self.elastic_band.torque]),
            "elastic_band_last_error": self.elastic_band.last_error or "",
        }
        self._elastic_band_trace_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self._elastic_band_trace_path.exists()
            or not self._elastic_band_trace_header_written
        )
        with self._elastic_band_trace_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._elastic_band_trace_header_written = True
            writer.writerow(row)

    @property
    def dof_names(self) -> list[str]:
        self._require_started()
        return list(self.articulation.dof_names)

    @property
    def num_dof(self) -> int:
        self._require_started()
        return int(self.articulation.num_dof)

    def _require_started(self) -> None:
        if not self._running or self.world is None or self.articulation is None:
            raise RuntimeError("IsaacSimulationBackend.start() must be called first")

    def _resolve_articulation_root_path(
        self,
        get_articulation_root_api_prim_path,
        get_current_stage,
        usd_physics,
    ) -> str:
        if self.requested_articulation_root_path:
            return self.requested_articulation_root_path

        root_path = get_articulation_root_api_prim_path(self.prim_path)
        if root_path:
            return str(root_path)

        stage = get_current_stage()
        candidates = []
        prefix = self.prim_path.rstrip("/") + "/"
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim_path == self.prim_path or prim_path.startswith(prefix):
                if prim.HasAPI(usd_physics.ArticulationRootAPI):
                    candidates.append(prim_path)

        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise RuntimeError(f"no articulation root found below {self.prim_path}")
        raise RuntimeError(f"multiple articulation roots found below {self.prim_path}: {candidates}")

    def _validate_joint_mapping(self) -> None:
        dof_names = self.dof_names
        duplicate_names = sorted({name for name in dof_names if dof_names.count(name) > 1})
        if duplicate_names:
            raise RuntimeError(f"duplicate Isaac DOF names: {duplicate_names}")

        expected_total = int(self.robot_asset.expected_total_dof)
        if len(dof_names) != expected_total:
            raise RuntimeError(
                f"{self.robot_model} DOF count mismatch: expected_total_dof={expected_total} "
                f"actual={len(dof_names)} names={dof_names}"
            )

        expected_body_names = list(self.robot_asset.body_joint_names)
        body_name_set = set(expected_body_names)
        extra_names = [name for name in dof_names if name not in body_name_set]
        if extra_names:
            raise RuntimeError(
                f"{self.robot_model} exposes DOFs outside configured body joints: {extra_names}"
            )
        missing_names = [name for name in expected_body_names if name not in set(dof_names)]
        if missing_names:
            raise RuntimeError(
                f"{self.robot_model} missing configured body joints: {missing_names}"
            )

        body_source_names = [name for name in dof_names if name in body_name_set]
        body_source_to_sonic = build_joint_mapping(
            source_joint_names=body_source_names,
            target_joint_names=expected_body_names,
        )
        sonic_body_to_isaac_dof = [
            dof_names.index(body_source_names[source_index]) for source_index in body_source_to_sonic
        ]
        self.joint_mapping = IsaacJointMappingResult(
            sonic_body_joint_names=expected_body_names,
            isaac_dof_names=dof_names,
            sonic_body_to_isaac_dof=sonic_body_to_isaac_dof,
        )
        self.joint_mapping_report = {
            "robot_model": self.robot_model,
            "status": "PASS",
            "expected_body_dof": self.robot_asset.expected_body_dof,
            "expected_total_dof": self.robot_asset.expected_total_dof,
            "actual_total_dof": len(dof_names),
            "hand_dof_available": self.robot_asset.hand_dof_available,
            "rows": self.joint_mapping.as_rows(),
        }

    def _validate_configured_prims(self) -> None:
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        checked: list[dict[str, Any]] = []
        for label, paths in (
            ("root_body", (self.robot_asset.root_body_prim_path,)),
            ("foot", self.robot_asset.foot_prim_paths),
            ("imu", self.robot_asset.imu_prim_paths),
        ):
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                exists = bool(prim and prim.IsValid())
                checked.append({"label": label, "path": path, "exists": exists})
                if not exists:
                    raise RuntimeError(
                        f"{self.robot_model} configured {label} prim does not exist: {path}"
                    )
        self.robot_asset_report = {
            **self.robot_asset_report,
            "configured_prims": checked,
        }

    def _print_robot_summary(self) -> None:
        print(f"[IsaacBackend] robot asset: {self.robot_asset_report}")
        print(f"[IsaacBackend] DOF count: {self.num_dof}")
        print(f"[IsaacBackend] joint/dof names ({len(self.dof_names)}):")
        for idx, name in enumerate(self.dof_names):
            print(f"  [{idx:02d}] {name}")

        if self.joint_mapping is not None:
            print("[IsaacBackend] Sonic body29 -> Isaac DOF mapping:")
            for row in self.joint_mapping.as_rows():
                print(
                    "  sonic[{sonic_body_index:02d}] {sonic_body_joint} "
                    "-> isaac[{isaac_dof_index:02d}] {isaac_dof_name}".format(**row)
                )

    @staticmethod
    def _tolist(value: Any) -> list[float]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        return value


def rootless(values: Any) -> Any:
    """Compatibility hook for Isaac joint arrays.

    SingleArticulation joint APIs already return only DOF values, not floating
    root coordinates. The function is intentionally a no-op but keeps the state
    read path explicit.
    """

    return values


def _fit_float_list(values: Any, count: int, *, fill: float = 0.0) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1) if values is not None else np.asarray([])
    result = []
    for index in range(int(count)):
        result.append(float(array[index]) if index < array.size else float(fill))
    return result


def _quat_wxyz_to_rotvec(quaternion: Any) -> np.ndarray:
    """Convert a MuJoCo/Isaac wxyz quaternion to a rotation vector."""

    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:4]
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(xyz_norm, w)
    return np.asarray((angle / xyz_norm) * xyz, dtype=np.float32)


def _get_usd_attr_value(prim: Any, name: str) -> Any:
    attr = prim.GetAttribute(name)
    if not attr:
        return None
    try:
        value = attr.Get()
    except Exception as exc:
        return f"error:{exc!r}"
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class IsaacSimulationServer(JsonZmqSimulationServer):
    """JSON/ZMQ server shell for the Isaac Sim backend."""

    def __init__(
        self,
        *backend_args: Any,
        host: str = "127.0.0.1",
        port: int = 5590,
        **backend_kwargs: Any,
    ):
        backend = IsaacSimulationBackend(*backend_args, **backend_kwargs)
        super().__init__(backend=backend, host=host, port=port)
