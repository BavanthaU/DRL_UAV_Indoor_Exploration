from __future__ import annotations

from pathlib import Path

try:
    import isaaclab.sim as sim_utils
    from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from isaaclab_assets import CRAZYFLIE_CFG
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "IsaacVlmPpoUavExplorationEnvCfg requires Isaac Lab. Use the debug "
        "mock training config for CPU/unit tests outside Isaac Lab."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENT_DIR = REPO_ROOT / "assets" / "environments"


@configclass
class VlmPpoActionCfg:
    action_mode: str = "body_xy_yawrate_altitude_hold"
    max_vx_mps: float = 1.0
    max_vy_mps: float = 0.6
    max_yaw_rate_radps: float = 1.0
    target_altitude_m: float = 1.2
    min_altitude_m: float = 0.8
    max_altitude_m: float = 1.8
    kp_xy_velocity: float = 1.8
    kp_z: float = 9.0
    kd_z: float = 4.0
    kp_yaw_rate: float = 0.001
    angular_damping: float = 0.0002
    max_roll_pitch_torque_nm: float = 0.003
    max_yaw_torque_nm: float = 0.003


@configclass
class VlmPpoMapCfg:
    grid_size: int = 64
    crop_size: int = 32
    resolution_m: float = 0.25
    sensor_radius_cells: int = 3
    stuck_steps: int = 80
    frontier_closed_steps: int = 12
    min_mapped_cells_for_completion: int = 64
    return_home_after_s: float = 480.0
    home_reached_radius_m: float = 0.75


@configclass
class VlmPpoRewardCfg:
    beta_count: float = 1.0
    beta_rnd: float = 0.05
    beta_semantic_novelty: float = 0.02
    w_frontier_closure: float = 5.0
    w_map_progress: float = 1.0
    w_return_home_progress: float = 1.0
    w_collision: float = 10.0
    w_near_obstacle: float = 1.0
    w_time: float = 0.002
    w_idle: float = 0.5
    w_oscillation: float = 0.05
    w_action_smoothness: float = 0.02
    w_altitude_error: float = 0.2


@configclass
class IsaacVlmPpoUavExplorationEnvCfg(DirectRLEnvCfg):
    """Direct Isaac Lab quadrotor exploration task with VLM-ready observations."""

    episode_length_s: float = 600.0
    decimation: int = 25
    use_office_asset: bool = True
    office_usd_path: str = str(ENVIRONMENT_DIR / "TrainEnvOffice1.usd")
    office_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    action_space = 3
    observation_space = {
        "camera_rgb": [3, 224, 224],
        "map_crop": [32, 32],
        "frontier_mask": [32, 32],
        "trajectory_mask": [32, 32],
        "depth_line": 64,
        "semantic_line": 64,
        "subgoal_features": 5,
    }
    state_space = 0
    viewer = ViewerCfg(eye=(0.0, -6.0, 4.0), lookat=(0.0, 0.0, 1.2))
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=40.0, replicate_physics=True)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.08, 0.0, 0.02), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.2,
            focus_distance=28.0,
            horizontal_aperture=5.76,
            vertical_aperture=3.24,
            clipping_range=(0.1, 6.0),
        ),
        width=224,
        height=224,
    )
    action_cfg = VlmPpoActionCfg()
    map_cfg = VlmPpoMapCfg()
    reward_cfg = VlmPpoRewardCfg()
