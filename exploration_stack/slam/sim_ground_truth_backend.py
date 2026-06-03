from __future__ import annotations

import json
import logging
from pathlib import Path

from exploration_stack.core import ExplorationMap, Pose3D, SensorPacket, SlamState, dataclass_from_dict, dataclass_to_dict
from exploration_stack.slam.base import SlamBackend

LOGGER = logging.getLogger(__name__)


class SimGroundTruthSlamBackend(SlamBackend):
    """Ground-truth pose backend for debug and smoke tests only."""

    def __init__(self, *, use_for_training: bool = False):
        self._state = SlamState(tracking_status="debug_ground_truth")
        self._map: ExplorationMap | None = None
        if use_for_training:
            LOGGER.warning(
                "SimGroundTruthSlamBackend selected for training. This uses "
                "simulator truth and should be treated as debug/evaluation only."
            )

    def reset(self) -> None:
        self._state = SlamState(tracking_status="debug_ground_truth")

    def update(self, sensor_packet: SensorPacket) -> SlamState:
        pose = sensor_packet.metadata.get("ground_truth_pose")
        if isinstance(pose, Pose3D):
            self._state = SlamState(
                pose=pose,
                tracking_status="debug_ground_truth",
                timestamp_s=sensor_packet.timestamp_s,
                metadata={"source": "sim_ground_truth_debug"},
            )
        elif isinstance(pose, dict):
            self._state = SlamState(
                pose=dataclass_from_dict(Pose3D, pose),
                tracking_status="debug_ground_truth",
                timestamp_s=sensor_packet.timestamp_s,
                metadata={"source": "sim_ground_truth_debug"},
            )
        else:
            LOGGER.warning("SensorPacket has no ground_truth_pose metadata; keeping previous pose.")
        return self._state

    def get_pose(self) -> Pose3D:
        return self._state.pose

    def get_map(self) -> ExplorationMap | None:
        return self._map

    def save_map(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(dataclass_to_dict(self._state), indent=2), encoding="utf-8")

    def load_map(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._state = dataclass_from_dict(SlamState, data)
