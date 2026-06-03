from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from exploration_stack.control.base import LocalController
from exploration_stack.core import DroneState, LocalCommand, SensorPacket, SlamState, Subgoal


class ExistingILSACLocalController(LocalController):
    """Wrapper around the current BC+SAC local policy.

    Goal conditioning is intentionally not forced in this first architecture
    task. If a policy observation is present in ``sensor_packet.metadata``, this
    wrapper calls the old policy. Otherwise it emits a conservative subgoal-
    pointing command for smoke tests.
    """

    def __init__(self, *, model: Any | None = None, checkpoint_path: str | Path | None = None):
        self.model = model
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    def act(
        self,
        *,
        robot_state: DroneState,
        sensor_packet: SensorPacket,
        slam_state: SlamState,
        subgoal: Subgoal | None,
    ) -> LocalCommand:
        observation = sensor_packet.metadata.get("policy_observation")
        if observation is not None:
            model = self._ensure_model()
            if model is not None:
                action, _ = model.predict(observation, deterministic=True)
                return self._action_to_command(action)

        if subgoal is None:
            return LocalCommand(stop=True, metadata={"source": "existing_il_sac_controller", "reason": "no_subgoal"})
        return self._fallback_goal_command(robot_state, subgoal)

    def _ensure_model(self) -> Any | None:
        if self.model is not None:
            return self.model
        if self.checkpoint_path is None:
            return None
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise RuntimeError(
                "Loading the existing IL-SAC local controller requires "
                "stable-baselines3. Install/activate the training environment, "
                "or pass a model object directly."
            ) from exc
        self.model = SAC.load(str(self.checkpoint_path))
        return self.model

    @staticmethod
    def _action_to_command(action: Any) -> LocalCommand:
        values = action.tolist() if hasattr(action, "tolist") else action
        if values and isinstance(values[0], list):
            values = values[0]
        forward = float(values[0]) if len(values) > 0 else 0.0
        yaw = float(values[1]) if len(values) > 1 else 0.0
        return LocalCommand(
            velocity_body=(forward, 0.0, 0.0),
            yaw_rate=yaw,
            metadata={"source": "existing_il_sac_controller"},
        )

    @staticmethod
    def _fallback_goal_command(robot_state: DroneState, subgoal: Subgoal) -> LocalCommand:
        dx = subgoal.position_xy[0] - robot_state.pose.position[0]
        dy = subgoal.position_xy[1] - robot_state.pose.position[1]
        distance = math.hypot(dx, dy)
        forward = min(0.5, distance)
        yaw = max(-0.8, min(0.8, math.atan2(dy, dx)))
        return LocalCommand(
            velocity_body=(forward, 0.0, 0.0),
            yaw_rate=yaw,
            metadata={
                "source": "existing_il_sac_controller_fallback",
                "todo": "replace with goal-conditioned IL-SAC observation when policy is retrained",
            },
        )
