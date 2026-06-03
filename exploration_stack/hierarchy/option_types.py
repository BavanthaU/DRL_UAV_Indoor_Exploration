from __future__ import annotations

from enum import IntEnum


class ExplorationOption(IntEnum):
    EXPLORE_FRONTIER_CLUSTER = 0
    ENTER_DOORWAY = 1
    FOLLOW_CORRIDOR = 2
    SWEEP_OPEN_SPACE = 3
    BACKTRACK_TO_UNVISITED_BRANCH = 4
    ROTATE_SCAN = 5
    AVOID_AND_RECOVER = 6
    STOP_IF_COMPLETE = 7


OPTION_NAMES = [option.name for option in ExplorationOption]
NUM_OPTIONS = len(ExplorationOption)


class CandidateType(IntEnum):
    FRONTIER = 0
    DOORWAY = 1
    CORRIDOR = 2
    OPEN_SPACE_SECTOR = 3
    BACKTRACK_NODE = 4
    ROTATE_SCAN = 5
    AVOID_AND_RECOVER = 6
    STOP = 7


NUM_CANDIDATE_TYPES = len(CandidateType)
