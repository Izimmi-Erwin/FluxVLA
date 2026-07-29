"""Shared JSON/ZMQ protocol for Sonic simulation control servers.

The automated evaluation suite talks to a simulation process through a stable
REQ/REP JSON endpoint, normally tcp://127.0.0.1:5590. Backends must implement
the same commands whether the simulator is MuJoCo or Isaac Sim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Thread
import time
from typing import Any

COMMAND_PING = "ping"
COMMAND_RESET = "reset"
COMMAND_BAND_ON = "band_on"
COMMAND_BAND_OFF = "band_off"
COMMAND_KEYBOARD = "keyboard"
COMMAND_GET_TASK_STATE = "get_task_state"


class SimulationBackend(ABC):
    """Backend interface used by the JSON protocol dispatcher."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return whether the simulation loop should continue serving requests."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation state."""

    @abstractmethod
    def band_on(self) -> bool:
        """Enable robot support band or equivalent initialization constraint."""

    @abstractmethod
    def band_off(self) -> bool:
        """Disable robot support band or equivalent initialization constraint."""

    @abstractmethod
    def keyboard(self, key: str) -> None:
        """Handle an evaluation keyboard command."""

    @abstractmethod
    def get_task_state(self, task_name: str) -> dict[str, Any]:
        """Return task state in the format consumed by EpisodeSuccessJudge."""


class SimulationProtocol:
    """Dispatch JSON command dictionaries to a SimulationBackend."""

    def __init__(self, backend: SimulationBackend):
        self.backend = backend

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")

        if command == COMMAND_PING:
            return {"ok": True, "running": self.backend.is_running()}

        if command == COMMAND_RESET:
            self.backend.reset()
            return {"ok": True, "command": command}

        if command == COMMAND_BAND_ON:
            return {"ok": self.backend.band_on(), "command": command}

        if command == COMMAND_BAND_OFF:
            return {"ok": self.backend.band_off(), "command": command}

        if command == COMMAND_GET_TASK_STATE:
            task_name = request.get("task_name", "cup_to_bin")
            return {
                "ok": True,
                "state": self.backend.get_task_state(task_name=str(task_name)),
            }

        if command == COMMAND_KEYBOARD:
            key = request.get("key")
            if not key:
                return {"ok": False, "error": "missing key"}
            self.backend.keyboard(str(key))
            return {"ok": True, "command": command, "key": key}

        return {"ok": False, "error": f"unknown command: {command}"}


class JsonZmqSimulationServer:
    """Small JSON REQ/REP server compatible with run_sonic_eval_suite.py."""

    def __init__(
        self,
        backend: SimulationBackend,
        host: str = "127.0.0.1",
        port: int = 5590,
        poll_sleep_s: float = 0.005,
    ):
        self.backend = backend
        self.protocol = SimulationProtocol(backend)
        self.endpoint = f"tcp://{host}:{port}"
        self.poll_sleep_s = poll_sleep_s
        self.context: Any | None = None
        self.socket: Any | None = None
        self.thread: Thread | None = None
        self._running = False
        self._zmq: Any | None = None

    @staticmethod
    def _load_zmq() -> Any:
        try:
            import zmq
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyzmq is required only when starting JsonZmqSimulationServer. "
                "Install pyzmq in this Python environment or run the backend directly."
            ) from exc
        return zmq

    def start(self, as_thread: bool = True, serve: bool = True) -> None:
        """Bind the endpoint and begin serving requests."""
        if self._running:
            return

        self._zmq = self._load_zmq()
        self.context = self._zmq.Context()
        self.socket = self.context.socket(self._zmq.REP)
        self.socket.setsockopt(self._zmq.LINGER, 0)
        self.socket.bind(self.endpoint)
        self._running = True

        if not serve:
            print(f"[SimulationServer] listening at {self.endpoint}")
            return

        if as_thread:
            self.thread = Thread(target=self.serve_forever, daemon=True)
            self.thread.start()
        else:
            self.serve_forever()

    def serve_forever(self) -> None:
        print(f"[SimulationServer] listening at {self.endpoint}")
        while self._running and self.backend.is_running():
            if not self.poll_once():
                time.sleep(self.poll_sleep_s)

    def poll_once(self) -> bool:
        """Serve at most one pending request without blocking."""

        if not self._running or not self.backend.is_running():
            return False
        try:
            assert self.socket is not None
            assert self._zmq is not None
            request = self.socket.recv_json(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return False
        except (AttributeError, self._zmq.ZMQError):
            self._running = False
            return False

        try:
            response = self.protocol.dispatch(request)
        except Exception as exc:
            response = {"ok": False, "error": repr(exc)}

        try:
            assert self.socket is not None
            self.socket.send_json(response)
        except (AttributeError, self._zmq.ZMQError):
            self._running = False
            return False
        return True

    def stop(self) -> None:
        self._running = False
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None
