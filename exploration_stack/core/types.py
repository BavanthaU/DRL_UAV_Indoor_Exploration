from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from types import UnionType
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints


def dataclass_to_dict(value: Any) -> Any:
    """Recursively convert dataclasses into plain Python containers."""

    if is_dataclass(value):
        return {key: dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    return value


def dataclass_from_dict(cls: Any, value: Any) -> Any:
    """Rebuild a typed dataclass tree from plain Python containers."""

    if value is None:
        return None
    if cls is Any:
        return value

    origin = get_origin(cls)
    args = get_args(cls)

    if origin in (Union, UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return dataclass_from_dict(non_none[0], value)
        for arg in non_none:
            try:
                return dataclass_from_dict(arg, value)
            except (TypeError, ValueError):
                continue
        return value

    if origin is list:
        item_type = args[0] if args else Any
        return [dataclass_from_dict(item_type, item) for item in value]

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(dataclass_from_dict(args[0], item) for item in value)
        return tuple(
            dataclass_from_dict(args[index], item) if index < len(args) else item
            for index, item in enumerate(value)
        )

    if origin is dict:
        key_type = args[0] if args else Any
        value_type = args[1] if len(args) > 1 else Any
        return {
            dataclass_from_dict(key_type, key): dataclass_from_dict(value_type, item)
            for key, item in value.items()
        }

    if isinstance(cls, type) and is_dataclass(cls):
        hints = get_type_hints(cls)
        kwargs = {}
        for item in fields(cls):
            if not item.init or item.name not in value:
                continue
            kwargs[item.name] = dataclass_from_dict(hints.get(item.name, Any), value[item.name])
        return cls(**kwargs)

    return value


@dataclass
class Pose3D:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    frame_id: str = "map"
    timestamp_s: float = 0.0


@dataclass
class CameraFrame:
    timestamp_s: float = 0.0
    frame_id: str = "camera"
    rgb: Any = None
    depth_m: Any = None
    semantic: Any = None
    intrinsics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImuPacket:
    timestamp_s: float = 0.0
    frame_id: str = "imu"
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class CollisionState:
    in_collision: bool = False
    contact_force_norm: float = 0.0
    closest_distance_m: float | None = None


@dataclass
class SensorPacket:
    timestamp_s: float = 0.0
    frame_id: str = "base_link"
    cameras: dict[str, CameraFrame] = field(default_factory=dict)
    imu: ImuPacket | None = None
    collision: CollisionState = field(default_factory=CollisionState)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DroneState:
    timestamp_s: float = 0.0
    pose: Pose3D = field(default_factory=Pose3D)
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    battery_remaining: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalCommand:
    velocity_body: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rate: float = 0.0
    thrust_body: tuple[float, float, float] | None = None
    position_goal: tuple[float, float, float] | None = None
    stop: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OccupancyGrid2D:
    """2D grid using existing-repo semantics by default.

    Cell values:
    - 0: unknown
    - 1: free/traversable and observed
    - 2: occupied/obstacle
    """

    UNKNOWN: ClassVar[int] = 0
    FREE: ClassVar[int] = 1
    OCCUPIED: ClassVar[int] = 2

    cells: list[list[int]] = field(default_factory=list)
    resolution_m: float = 0.10
    origin_xy: tuple[float, float] = (0.0, 0.0)
    frame_id: str = "map"

    @property
    def height(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def cell_to_xy(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return (
            self.origin_xy[0] + (col + 0.5) * self.resolution_m,
            self.origin_xy[1] + (row + 0.5) * self.resolution_m,
        )

    def xy_to_cell(self, xy: tuple[float, float]) -> tuple[int, int]:
        col = int((xy[0] - self.origin_xy[0]) / self.resolution_m)
        row = int((xy[1] - self.origin_xy[1]) / self.resolution_m)
        return row, col


@dataclass
class EsdfMap:
    distances_m: list[list[float]] = field(default_factory=list)
    resolution_m: float = 0.10
    origin_xy: tuple[float, float] = (0.0, 0.0)
    frame_id: str = "map"


@dataclass
class SemanticGrid:
    labels: list[list[int]] = field(default_factory=list)
    label_names: dict[int, str] = field(default_factory=dict)
    resolution_m: float = 0.10
    origin_xy: tuple[float, float] = (0.0, 0.0)
    frame_id: str = "map"


@dataclass
class Frontier:
    frontier_id: int
    cells: list[tuple[int, int]] = field(default_factory=list)
    centroid_xy: tuple[float, float] = (0.0, 0.0)
    information_gain: float = 0.0
    size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DoorwayHypothesis:
    doorway_id: int
    position_xy: tuple[float, float]
    confidence: float = 0.0
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoomGraph:
    rooms: dict[int, dict[str, Any]] = field(default_factory=dict)
    edges: list[tuple[int, int]] = field(default_factory=list)
    doorways: list[DoorwayHypothesis] = field(default_factory=list)


@dataclass
class ExplorationMap:
    occupancy_2d: OccupancyGrid2D = field(default_factory=OccupancyGrid2D)
    esdf: EsdfMap | None = None
    semantic_grid: SemanticGrid | None = None
    frontiers: list[Frontier] = field(default_factory=list)
    room_graph: RoomGraph = field(default_factory=RoomGraph)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SlamState:
    pose: Pose3D = field(default_factory=Pose3D)
    covariance: list[list[float]] | None = None
    tracking_status: str = "unknown"
    timestamp_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrontierScore:
    frontier_id: int
    score: float
    reason: str = ""
    risk: float = 0.0
    expected_information_gain: float = 0.0
    doorway_likelihood: float = 0.0
    corridor_likelihood: float = 0.0


@dataclass
class SemanticPrior:
    scene_summary: str = ""
    room_type_guess: str = "unknown"
    frontier_scores: list[FrontierScore] = field(default_factory=list)
    recommended_subgoal_id: int | None = None
    uncertainty: float = 1.0
    planner_prior: dict[str, Any] = field(default_factory=dict)

    def score_for_frontier(self, frontier_id: int) -> FrontierScore | None:
        for score in self.frontier_scores:
            if score.frontier_id == frontier_id:
                return score
        return None


@dataclass
class VlmObservationPacket:
    sensor_packet: SensorPacket
    slam_state: SlamState
    exploration_map: ExplorationMap
    frontiers: list[Frontier]
    robot_state: DroneState
    rendered_map: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VlmStructuredOutput:
    scene_summary: str
    room_type_guess: str
    visible_structures: dict[str, Any]
    frontier_scores: list[FrontierScore]
    recommended_subgoal_id: int | None
    uncertainty: float


@dataclass
class PlannerPrior:
    preferred_frontier_ids: list[int] = field(default_factory=list)
    blocked_frontier_ids: list[int] = field(default_factory=list)
    doorway_frontier_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Subgoal:
    subgoal_id: int
    position_xy: tuple[float, float]
    frontier_id: int | None = None
    score: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
