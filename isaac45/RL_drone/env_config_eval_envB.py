import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, ContactSensorCfg

# Custom MDP logic (actions, observations, rewards, terminations)
from isaac45 import mdp
from isaac45.paths import ENVIRONMENT_DIR

# Drone_models directory
from isaac45.drone_models.iris_contact import get_iris_config


@configclass
class QuadrotorSceneCfg(InteractiveSceneCfg):
    # Testing Office environment B
    office_A = AssetBaseCfg(
        prim_path = "{ENV_REGEX_NS}/officeB", 
        spawn=sim_utils.UsdFileCfg(usd_path=str(ENVIRONMENT_DIR / "TestEnvOfficeB.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos = (0.0, 0.0, 0),))

    # Global lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Drone robot config
    robot: ArticulationCfg = get_iris_config().replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Wide lens camera mounted on the robot body for depth + semantic segmentation. Zed X one wide parameters are used. 
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/Isaaclab_camera",
        update_period= 0.0,       # dt / dt * decimation. With 0.0 the update is at every step
        height=48,
        width=64, 
        colorize_semantic_segmentation = False,
        semantic_filter ="class: door; class: wall; class: ceiling; class: floor", # Relevent classes are chosen, also unlabeled and unknown are a class. 6 total classes. 
        data_types=["distance_to_camera", "semantic_segmentation"],
        depth_clipping_behavior= ["max"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.208, focus_distance = 28.0, horizontal_aperture=5.76, vertical_aperture=3.24, clipping_range=(0.1, 4.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0,-0.25), convention="world"),
    )

    # Contact sensor for collision detection
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0, history_length=0, debug_vis=True,track_air_time=False,
    )


@configclass
class ActionsCfg:   
    # drone_drive_vel_setpoint = mdp.actions.custom_actions_cfg.RLDriven2DActionCfg(asset_name="robot")
    drone_drive_vel_setpoint = mdp.actions.custom_actions_cfg.IdealRLDrivenAction2DCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class ObsCfg(ObsGroup):
        # Depth image
        # depth_obs = ObsTerm(
        #    func=mdp.observations.get_depth_images_switched_channel,
        #     params={"depth_threshold": 4.0},
        # )

        # Egocentric occupancy map
        map_obs = ObsTerm(
            func=mdp.observations.get_rotated_egocentric_mapping_ONE_HOT,
            params={"egocentric_map_half_size": 30},
        )
        
        # Semantic image
        # semantic_obs = ObsTerm(
        #     func=mdp.observations.get_id_semantic_images_switched_channel_ONE_HOT,
        #     params={"num_classes": 6},
        # )

        # Depth Line (middle line of depth image)
        oneline_depth = ObsTerm(func=mdp.observations.get_1d_depth, params={"depth_threshold": 4.0})

        # Semantic Line (middle line of semantic image)
        oneline_sems = ObsTerm(func=mdp.observations.get_oneline_semantic, params={"num_classes": 6})
        

        def __post_init__(self) -> None:
            self.enable_corruption = False # Option to add noise or not. 
            self.concatenate_terms = False # Keep observations as dict; feature extractor will handle concatenation

    #Observation groups
    policy: ObsCfg = ObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # reset event
    reset_drone_eval_envB = EventTerm(
        func=mdp.events.reset_drone_eval_envB,  # Using reset_joints_by_offset function
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Reward for area coverage, each new cell found in occupancy map 
    explore = RewTerm(
        func=mdp.rewards.area_coverage,
        weight=0.1,
    )

    # Penalty for collision. 
    collision = RewTerm(
        func= mdp.rewards.check_collision_single_contact_sensor, 
        weight=10.0, 
        params={"M": -10.0, "N": 0.0, "force_threshold": 0.01} # M: Negative reward for crashing. N: positive reward for keeping alive
    )
    
    # Reward for finishing exploration
    exploration_finished = RewTerm(func=mdp.rewards.fixed_area_covered, weight=150.0, params={"num_cells_to_cover":5200})
    
    # Reward for first time entering a new room
    entered_room = RewTerm(func=mdp.rewards.doorway_reward_per_drone, weight=10.0, params={"doorway_radius":1.0},)

    # Penalty for staying idle
    idle_behavior = RewTerm(func=mdp.rewards.penalize_idle_behavior, weight = 1, params={"idle_penalty":-0.01, "motion_threshold":0.10})



@configclass 
class TerminationCfg:
    """Termination terms for the MDP."""

    # Time out termination
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Termination if area is fully covered 
    area_is_covered=DoneTerm(func=mdp.terminations.drone_covers_fixed_area, params={"num_cells_to_cover":5200},)
    
    # Termination if drone flips upside down along x or y axis (used for non-linear controller)
    drone_flips = DoneTerm(func=mdp.terminations.drone_flips_upsidedown)

    # Termination if drone crashes into wall with force threshold N
    drone_crashes = DoneTerm(
        func=mdp.terminations.drone_crashes_single_contact_sensor,
        params={"force_threshold": 0.01},
    )  

@configclass
class DroneEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the drone environment."""

    scene = QuadrotorSceneCfg(num_envs=1, env_spacing=100)
    observations = ObservationsCfg()
    actions: actions = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationCfg = TerminationCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 25
        self.episode_length_s = 2000
        
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 8.0)
        
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = 25 #self.decimation
        self.sim.physx.gpu_collision_stack_size = 2 ** 28
