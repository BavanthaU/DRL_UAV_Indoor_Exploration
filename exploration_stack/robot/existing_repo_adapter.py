from __future__ import annotations

from typing import Any

from exploration_stack.core import CollisionState, DroneState, LocalCommand, Pose3D, SensorPacket
from exploration_stack.robot.base import RobotAdapter


class ExistingRepoRobotAdapter(RobotAdapter):
    """Adapter around the current Gym/Isaac Lab environment path.

    This keeps the old IL/SAC environment callable while the new stack is
    validated. It expects an already-created environment or SB3 VecEnv-like
    wrapper; it does not launch Isaac Sim by itself.
    """

    def __init__(self, env: Any):
        self.env = env
        self._last_command = LocalCommand()
        self._last_state = DroneState()

    def reset(self) -> DroneState:
        reset_result = self.env.reset()
        self._last_state = self._extract_state(reset_result)
        return self._last_state

    def step(self, action: LocalCommand) -> DroneState:
        self.apply_local_command(action)
        raw_action = self._command_to_action(action)
        step_result = self.env.step(raw_action)
        self._last_state = self._extract_state(step_result)
        return self._last_state

    def get_state(self) -> DroneState:
        return self._extract_state(None)

    def get_sensors(self) -> SensorPacket:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        metadata = {"source": "existing_repo_adapter"}
        if hasattr(unwrapped, "env_map"):
            metadata["occupancy_grid"] = unwrapped.env_map.environment_map
        return SensorPacket(
            timestamp_s=float(getattr(unwrapped, "common_step_counter", 0)),
            collision=CollisionState(),
            metadata=metadata,
        )

    def apply_local_command(self, cmd: LocalCommand) -> None:
        self._last_command = cmd

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _command_to_action(self, command: LocalCommand) -> list[float]:
        forward = command.velocity_body[0]
        yaw = command.yaw_rate
        return [forward, yaw]

    def _extract_state(self, _: Any) -> DroneState:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        try:
            robot = unwrapped.scene["robot"]
            pose = robot.data.root_pos_w[0].detach().cpu().tolist()
            quat = robot.data.root_quat_w[0].detach().cpu().tolist()
            lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().tolist()
            ang_vel = robot.data.root_ang_vel_w[0].detach().cpu().tolist()
            state = DroneState(
                timestamp_s=float(getattr(unwrapped, "common_step_counter", 0)),
                pose=Pose3D(
                    position=(float(pose[0]), float(pose[1]), float(pose[2])),
                    orientation_xyzw=(float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])),
                    frame_id="world",
                ),
                linear_velocity=tuple(float(v) for v in lin_vel),
                angular_velocity=tuple(float(v) for v in ang_vel),
                metadata={"source": "existing_repo_adapter"},
            )
            self._last_state = state
            return state
        except Exception:
            return self._last_state
