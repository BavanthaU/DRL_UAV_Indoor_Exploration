from __future__ import annotations

import unittest

from exploration_stack.core import (
    DroneState,
    Frontier,
    OccupancyGrid2D,
    Pose3D,
    SensorPacket,
    dataclass_from_dict,
    dataclass_to_dict,
)


class CoreSerializationTest(unittest.TestCase):
    def test_dataclass_round_trip(self):
        state = DroneState(
            timestamp_s=1.0,
            pose=Pose3D(position=(1.0, 2.0, 3.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            linear_velocity=(0.1, 0.0, 0.0),
            metadata={"mode": "test"},
        )
        rebuilt = dataclass_from_dict(DroneState, dataclass_to_dict(state))
        self.assertEqual(rebuilt.pose.position, (1.0, 2.0, 3.0))
        self.assertEqual(rebuilt.linear_velocity, (0.1, 0.0, 0.0))
        self.assertEqual(rebuilt.metadata["mode"], "test")

    def test_nested_grid_and_frontier_round_trip(self):
        packet = SensorPacket(
            metadata={
                "grid": dataclass_to_dict(OccupancyGrid2D(cells=[[0, 1], [2, 1]], resolution_m=0.5)),
                "frontier": dataclass_to_dict(Frontier(frontier_id=2, cells=[(0, 1)], centroid_xy=(0.75, 0.25))),
            }
        )
        rebuilt = dataclass_from_dict(SensorPacket, dataclass_to_dict(packet))
        self.assertEqual(rebuilt.metadata["grid"]["cells"][1][0], 2)
        self.assertEqual(rebuilt.metadata["frontier"]["frontier_id"], 2)


if __name__ == "__main__":
    unittest.main()
