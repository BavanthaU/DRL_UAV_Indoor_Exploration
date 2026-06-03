"""Map-progress training config without privileged room/door reward shortcuts.

This config is for new RL experiments where the agent should learn exploration
from sensor-derived map progress and safety signals, not from hard-coded room
or doorway coordinates.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaac45 import mdp
from isaac45.RL_drone.env_config_training import ActionsCfg, EventCfg, ObservationsCfg, QuadrotorSceneCfg


GENERIC_COVERAGE_TARGET_CELLS = 2600


@configclass
class MapProgressRewardsCfg:
    """Reward terms based only on online map progress and safety."""

    explore = RewTerm(
        func=mdp.rewards.area_coverage_and_loop_penalty,
        weight=0.1,
    )

    collision = RewTerm(
        func=mdp.rewards.check_collision_single_contact_sensor,
        weight=10.0,
        params={"M": -10.0, "N": 0.0, "force_threshold": 0.01},
    )

    idle_behavior = RewTerm(
        func=mdp.rewards.penalize_idle_behavior,
        weight=1.0,
        params={"idle_penalty": -0.01, "motion_threshold": 0.10},
    )


@configclass
class MapProgressTerminationCfg:
    """Termination terms without scene labels or doorway markers."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    area_is_covered = DoneTerm(
        func=mdp.terminations.drone_covers_fixed_area,
        params={"num_cells_to_cover": GENERIC_COVERAGE_TARGET_CELLS},
    )

    drone_flips = DoneTerm(func=mdp.terminations.drone_flips_upsidedown)

    drone_crashes = DoneTerm(
        func=mdp.terminations.drone_crashes_single_contact_sensor,
        params={"force_threshold": 0.01},
    )


@configclass
class DroneMapProgressEnvCfg(ManagerBasedRLEnvCfg):
    """Map-progress exploration MDP for new SAC/no-IL training."""

    scene = QuadrotorSceneCfg(num_envs=1, env_spacing=100)
    observations = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: MapProgressRewardsCfg = MapProgressRewardsCfg()
    terminations: MapProgressTerminationCfg = MapProgressTerminationCfg()

    def __post_init__(self) -> None:
        self.decimation = 25
        self.episode_length_s = 1000
        self.viewer.eye = (0.0, 0.0, 8.0)
        self.sim.dt = 0.01
        self.sim.render_interval = 25
        self.sim.physx.gpu_collision_stack_size = 2**28
