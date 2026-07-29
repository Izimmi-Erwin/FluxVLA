"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

import sys
from pathlib import Path
from dataclasses import MISSING, fields
from datetime import datetime
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_str = str(REPO_ROOT)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

try:
    import tyro
except ModuleNotFoundError:
    tyro = None

import gear_sonic
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

ArgsConfig = SimLoopConfig


def _parse_cli_fallback() -> ArgsConfig:
    """Small argparse fallback for Isaac Python environments without tyro."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    config = ArgsConfig()
    for field_info in fields(ArgsConfig):
        name = field_info.name
        default = field_info.default if field_info.default is not MISSING else getattr(config, name)
        arg_name = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg_name, dest=name, action="store_true", default=default)
            parser.add_argument("--no-" + name.replace("_", "-"), dest=name, action="store_false")
        elif isinstance(default, int) and not isinstance(default, bool):
            parser.add_argument(arg_name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(arg_name, type=float, default=default)
        elif default is None:
            parser.add_argument(arg_name, default=default)
        elif isinstance(default, str):
            parser.add_argument(arg_name, type=str, default=default)

    namespace, unknown = parser.parse_known_args()
    if unknown:
        print(f"[run_sim_loop] warning: argparse fallback ignored unknown args: {unknown}")
    for key, value in vars(namespace).items():
        setattr(config, key, value)
    config.__post_init__()
    return config


class SimWrapper:
    def __init__(self, robot_model: Any, env_name: str, config: Dict[str, Any], **kwargs):
        from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel

        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def _sonic_to_isaac_positions(backend: Any, sonic_positions: Any) -> Any:
    import numpy as np

    current_isaac = np.asarray(backend.articulation.get_joint_positions(), dtype=np.float32)
    result = current_isaac.copy()
    for sonic_index, isaac_index in enumerate(backend.joint_mapping.sonic_body_to_isaac_dof):
        result[isaac_index] = float(sonic_positions[sonic_index])
    return result


def _run_isaac_loop(config: ArgsConfig, wbc_config: Dict[str, Any]) -> None:
    """Run Isaac as the simulator terminal in the MuJoCo-compatible stack.

    Port ownership intentionally matches the MuJoCo deployment:

    - 5555: simulator publishes camera msgpack frames.
    - 5557: normally owned by ``g1_deploy_onnx_ref``; Isaac only provides an
      optional fallback publisher when explicitly requested.
    """

    import os
    import numpy as np

    from gear_sonic.robot_interface.isaac_robot_assets import DEFAULT_ROBOT_MODEL, get_robot_asset
    from gear_sonic.robot_interface.isaac_unitree_bridge import IsaacUnitreeBridge
    from gear_sonic.robot_interface.pd_mapping import SONIC_STANDING_POSE
    from gear_sonic.simulation_server.isaac_server import (
        DEFAULT_SONICSTAR_TASK_SCENE_LAYER_PATH,
        IsaacSimulationBackend,
        _ensure_optional_site_packages,
    )

    robot_asset = get_robot_asset(DEFAULT_ROBOT_MODEL)
    log_dir = (
        REPO_ROOT
        / "work_dirs"
        / "isaac_sim_loop"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    headless = not bool(config.enable_onscreen)
    physics_dt = 1.0 / float(config.sim_frequency)
    backend = IsaacSimulationBackend(
        robot_model=DEFAULT_ROBOT_MODEL,
        prim_path=robot_asset.prim_path,
        articulation_root_path=robot_asset.articulation_root,
        headless=headless,
        physics_dt=physics_dt,
        rendering_dt=1.0 / 60.0,
        startup_steps=20,
        scene_layer_path=DEFAULT_SONICSTAR_TASK_SCENE_LAYER_PATH,
        collision_layer_path=robot_asset.collision_layer_path,
    )
    backend.elastic_band.point = np.asarray(
        [0.0, 0.0, float(config.isaac_elastic_band_anchor_z)],
        dtype=np.float32,
    )
    backend.elastic_band.enabled = bool(config.isaac_elastic_band_enabled)
    bridge = IsaacUnitreeBridge(
        domain_id=int(wbc_config.get("DOMAIN_ID", 0)),
        network_interface=wbc_config.get("INTERFACE", None),
        log_dir=log_dir,
        verbose=bool(config.verbose),
        control_mode=os.environ.get("ISAAC_CONTROL_MODE", "position"),
        robot_asset=robot_asset,
    )

    eval_server = None
    try:
        backend.start()
        bridge.articulation = backend.articulation
        bridge.joint_mapping = backend.joint_mapping
        bridge.robot_asset = backend.robot_asset

        standing_pose_sonic = np.asarray(SONIC_STANDING_POSE, dtype=np.float32)
        standing_pose_isaac = _sonic_to_isaac_positions(backend, standing_pose_sonic)
        init_report = backend.initialize_articulation_state_for_control(
            joint_positions=standing_pose_isaac,
            root_position=np.asarray(
                [0.0, 0.0, float(config.isaac_initial_root_height)],
                dtype=np.float32,
            ),
            root_orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            configure_drives=lambda: bridge.configure_joint_drives(force=True),
            target_writer=bridge._apply_isaac_position_targets,
            reset_world=True,
            startup_warmup_steps=backend.startup_steps,
            state_commit_steps=20,
            target_commit_steps=20,
            render=not headless,
        )
        backend.set_elastic_band_enabled(bool(config.isaac_elastic_band_enabled))
        (log_dir / "canonical_control_init.log").write_text(
            f"{init_report}\n"
            f"elastic_band={backend.elastic_band.as_dict()}\n"
            f"isaac_initial_root_height={float(config.isaac_initial_root_height)}\n",
            encoding="utf-8",
        )

        if config.isaac_publish_camera:
            backend.start_image_publish_subprocess(camera_port=config.camera_port)
        if config.isaac_publish_debug_state:
            backend.start_realtime_state_publisher(
                port=config.state_zmq_port,
                topic=config.state_zmq_topic,
            )
            print(
                "[run_sim_loop] warning: Isaac direct 5557 publisher is enabled. "
                "Do not run g1_deploy_onnx_ref on the same 5557 port."
            )

        if config.enable_eval_control:
            _ensure_optional_site_packages("zmq")
            from gear_sonic.simulation_server.protocol import JsonZmqSimulationServer

            eval_server = JsonZmqSimulationServer(
                backend=backend,
                host=config.eval_control_host,
                port=config.eval_control_port,
            )

        print(
            "[run_sim_loop] Isaac backend ready: "
            f"DDS lowstate/lowcmd active after bridge.start(), camera_port={config.camera_port}, "
            f"debug_5557={'direct-isaac' if config.isaac_publish_debug_state else 'owned-by-g1_deploy_onnx_ref'}, "
            f"elastic_band_enabled={backend.elastic_band.enabled}, "
            f"elastic_band_anchor={backend.elastic_band.point.tolist()}, "
            f"initial_root_height={float(config.isaac_initial_root_height)}, "
            f"log_dir={log_dir}",
            flush=True,
        )
        backend.spin(
            frequency_hz=int(config.sim_frequency),
            dds_bridge=bridge,
            log_dir=log_dir,
            control_server=eval_server,
        )
    except KeyboardInterrupt:
        print("+++++Isaac simulator interrupted by user.")
    except Exception:
        import traceback

        print("[run_sim_loop] Isaac backend fatal error:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if eval_server is not None:
            try:
                eval_server.stop()
            except Exception:
                pass
        bridge.stop()
        backend.close()


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    base_sim_path = "not_loaded"
    if config.simulator == "mujoco":
        from gear_sonic.utils.mujoco_sim import base_sim as base_sim_module

        base_sim_path = str(Path(base_sim_module.__file__).resolve())
    print(
        "[run_sim_loop] "
        f"script={Path(__file__).resolve()} "
        f"repo_root={REPO_ROOT} "
        f"gear_sonic={Path(gear_sonic.__file__).resolve()} "
        f"base_sim={base_sim_path} "
        f"backend={config.simulator} "
        f"robot_scene={wbc_config.get('ROBOT_SCENE')}",
        flush=True,
    )
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name

    if config.simulator == "isaac":
        _run_isaac_loop(config=config, wbc_config=wbc_config)
        return

    if config.enable_image_publish:
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

    robot_model = instantiate_g1_robot_model()

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )

    if config.enable_eval_control:
        sim_wrapper.sim.start_eval_control_server(
            host=config.eval_control_host,
            port=config.eval_control_port,
        )
    # Start simulator as independent process
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    if tyro is not None:
        config = tyro.cli(ArgsConfig)
    else:
        print("[run_sim_loop] warning: tyro is not installed; using limited argparse fallback")
        config = _parse_cli_fallback()
    main(config)
