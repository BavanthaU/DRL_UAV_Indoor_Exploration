import argparse
import sys
import os
import time

# --- Isaac Kit app MUST be created first ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay trained SAC agent and livestream viewport.")
parser.add_argument("--task", type=str, default="Drone_eval_envB")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to SB3 .zip model")
parser.add_argument("--vecnorm_path", type=str, default=None, help="Optional VecNormalize stats .pkl")
parser.add_argument("--fps", type=float, default=30.0, help="Target display FPS for pacing")
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--deterministic", action="store_true", default=True)
parser.add_argument("--ray_debug", action="store_true", help="Print ray-caster min distances each step.")
parser.add_argument("--ray_debug_hits", action="store_true", help="Additionally print raw ray hit points (env 0).")
# pass-through common AppLauncher args (device, headless, enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Create Kit app (set --headless 0 when you launch to stream)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]] + [arg for arg in hydra_args if not arg.startswith("--/")]
# # ----(optional) make sure a viewport window exists for streaming
# try:
#     import omni.kit.app
#     import omni.kit.viewport.window as vpw
#     app = omni.kit.app.get_app()
#     mw = app.get_main_window()
#     if mw is None or getattr(mw, "get_window_minimize_event_stream", lambda: None)() is None:
#         vpw.create_viewport_window(width=1280, height=720, x=100, y=100)
#         print("[INFO] Created a viewport window for streaming.")
# except Exception as e:
#     print(f"[WARN] Viewport bootstrap skipped: {e}")

# === rest of imports AFTER Kit boot ===
import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# your project bits (exactly like validation)
import DRL_UAV_Indoor_Exploration.isaac45.RL_drone  # registers custom gym environments
from DRL_UAV_Indoor_Exploration.isaac45.utils.custom_sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg
from DRL_UAV_Indoor_Exploration.isaac45.mdp.common import ensure_ray_caster_initialized
import DRL_UAV_Indoor_Exploration.isaac45.utils.env_mapping_classes as env_mapping
from DRL_UAV_Indoor_Exploration.isaac45.utils.sac import SAC
from stable_baselines3.common.vec_env import VecNormalize


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):

    # mirror validation setup
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # build env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env.unwrapped.raycast_debug_print = args_cli.ray_debug
    env.unwrapped.raycast_debug_print_hits = args_cli.ray_debug_hits
    ensure_ray_caster_initialized(env)
    env_model = env_mapping.EnvironmentModelFOVTraversability(
        env.unwrapped.scene.num_envs, env.unwrapped.sim.device, env.unwrapped.scene.env_origins
    )
    env.unwrapped.env_map = env_model

    # SB3 wrapper: actions in [-1,1] forward/back and [-1,1] yaw rate
    env = Sb3VecEnvWrapper(env, lower_bound=np.array([-1, -1]), upper_bound=np.array([1, 1]))

    # (optional) VecNormalize restore (eval mode)
    if args_cli.vecnorm_path and os.path.isfile(args_cli.vecnorm_path):
        env = VecNormalize(env, training=False, norm_obs=False, norm_reward=False)
        env = VecNormalize.load(args_cli.vecnorm_path, env)
        env.training = False
        env.norm_obs = False
        env.norm_reward = False
        print(f"[INFO] Loaded VecNormalize stats from: {args_cli.vecnorm_path}")

    # load model
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    agent = SAC.load(args_cli.checkpoint, env=env, print_system_info=False)

    # roll deterministic for viewing
    obs = env.reset()
    steps = 0
    dt = 1.0 / max(1e-3, args_cli.fps)
    last_t = time.time()

    print("[INFO] Starting streaming replay...")
    while steps < args_cli.max_steps:
        action, _ = agent.predict(obs, deterministic=args_cli.deterministic)
        obs, reward, done, info = env.step(action)

        if done.any():
            obs = env.reset()

        # pace the GUI so you can watch comfortably
        now = time.time()
        sleep_t = dt - (now - last_t)
        if sleep_t > 0:
            time.sleep(sleep_t)
        last_t = time.time()
        steps += 1

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
