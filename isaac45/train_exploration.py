"""
Training script for the drone exploration SAC policy.

Quick primer on planner modes:

  --planner_mode heuristic   -> Pure classical frontier selector (default). No frontier manager created.
  --planner_mode observe     -> Classical selector commands. Frontier manager logs demonstrations only.
  --planner_mode assist      -> Classical selector commands. Manager learns in the background from demo data.
  --planner_mode rl          -> Learned frontier manager produces subgoals. Requires --manager_rl.

Other useful switches:

  --manager_rl               -> Build the frontier manager (needed for observe/assist/rl).
  --manager_load_path PATH   -> Load a saved frontier manager state (works in training/validation/replay).
  --manager_*                -> Hyperparameters for the frontier manager (lr, gamma, epsilon, max candidates).
  --ray_debug[/_hits]        -> Print ray-caster distances / raw hits for debugging.

Training saves both the SAC model (model.zip) and the frontier manager state (manager_state.pt) in the
run directory. Use --resume to continue SAC, and --manager_load_path to reload the planner.
"""

import argparse
import sys
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a drone for a RL task")
parser.epilog = """Key switches:
  --task                    Gym ID registered in RL_drone/__init__.py (e.g. Drone_SAC_convnextv2).
  --num_envs                Number of Isaac environments simulated in parallel.
  --max_iterations          Override hydra-configured rollout iterations (converted to SB3 timesteps).
  --planner_mode            Frontier manager mode: heuristic | observe | assist | rl.
  --manager_*               Frontier manager hyperparameters (lr/gamma/epsilon/max_candidates).
  --map_snapshot_interval   Log map/frontier overlays every N steps (0 = only on episode end).
  --use_IL / --IL_model_path Use behaviour-cloned weights to warm start the SAC policy.
  --fill_replay_buffer      Ask the IL policy to pre-populate the replay buffer.
  --buffer_path             Load an existing replay buffer pickle.
  --livestream / --enable_cameras Isaac Sim rendering options (forwarded to AppLauncher).
"""
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Drone_SAC_IL", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--use_IL", action="store_true", default=False, help="Use IL as pretrained weights")
parser.add_argument("--IL_model_path", type=str, default=None, help="Path to IL.")
parser.add_argument("--fill_replay_buffer", action="store_true", default=False, help="Fill replay buffer with IL trajectories")
parser.add_argument("--buffer_path", type=str, default=None, help="Path to buffer file..")
parser.add_argument("--resume", type=str, default=None, help="Path to SB3 .zip checkpoint to resume from")
parser.add_argument("--wandb_project", type=str, default="isaac-drone-hierarchical", help="W&B project name")
parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name (optional)")
parser.add_argument("--wandb_mode", type=str, default="online", help="'online'|'offline'|'disabled'")
parser.add_argument("--ray_debug", action="store_true", help="Print ray-caster min distances each step.")
parser.add_argument("--ray_debug_hits", action="store_true", help="Additionally print raw ray hit points (env 0).")
parser.add_argument("--manager_rl", action="store_true", help="Enable RL-based frontier manager.")
parser.add_argument("--manager_max_candidates", type=int, default=8, help="Maximum frontier candidates considered by the manager.")
parser.add_argument("--manager_lr", type=float, default=1e-3, help="Learning rate for the frontier manager policy.")
parser.add_argument("--manager_gamma", type=float, default=0.95, help="Discount factor for frontier manager rewards.")
parser.add_argument("--manager_epsilon", type=float, default=0.1, help="Epsilon-greedy exploration for the frontier manager.")
parser.add_argument("--planner_mode", choices=["heuristic", "observe", "assist", "rl"], default="heuristic", help="Frontier planner mode: 'heuristic' for classical, 'observe' to log heuristics, 'assist' to imitate while heuristics run, 'rl' to learn subgoals.")
parser.add_argument("--manager_load_path", type=str, default=None, help="Path to a saved frontier manager checkpoint.")
parser.add_argument("--map_snapshot_interval", type=int, default=0, help="Log map and frontier overlays every N steps (0 disables periodic snapshots).")
parser.add_argument("--obs_stack", type=int, default=4, help="Number of consecutive observations to stack per modality.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]] + [arg for arg in hydra_args if not arg.startswith("--/")]
# at the top with other imports
import time
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

class TqdmCallback(BaseCallback):
    def __init__(self, total_target: int, start: int = 0, update_interval: int = 1000):
        super().__init__()
        self.total_target = int(total_target)
        self.start = int(start)
        self.update_interval = int(update_interval)
        self._last = self.start
        self._t0 = None
        self.pbar = None

    def _on_training_start(self) -> bool:
        self._t0 = time.time()
        # note: if you set CUDA_VISIBLE_DEVICES=1, tqdm still prints fine in headless
        self.pbar = tqdm(total=self.total_target, initial=self.start,
                         dynamic_ncols=True, unit="step", leave=True)
        return True

    def _on_step(self) -> bool:
        cur = int(self.model.num_timesteps)  # SB3 global step counter
        delta = cur - self._last
        if delta >= self.update_interval:
            self.pbar.update(delta)
            # ETA
            done = cur - self.start
            elapsed = time.time() - self._t0 # type: ignore
            rate = done / max(elapsed, 1e-6)  # steps/sec
            remain = max(0, self.total_target - cur)
            eta_s = remain / max(rate, 1e-6)
            self.pbar.set_postfix_str(f"~{rate:.0f} step/s | ETA {eta_s/3600:.1f}h")
            self._last = cur
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            # flush to 100% if needed and close
            self.pbar.update(max(0, self.total_target - self.pbar.n))
            self.pbar.close()


class WandbMetricsCallback(BaseCallback):
    """Periodic logging of custom exploration metrics and maps to Weights & Biases."""

    def __init__(self, env, log_interval: int = 200, max_map_images: int = 2):
        super().__init__()
        self._env = env
        self._log_interval = max(1, log_interval)
        self._max_map_images = max(0, max_map_images)
        self._ep_returns: deque[float] = deque(maxlen=100)
        self._ep_lengths: deque[float] = deque(maxlen=100)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                if not isinstance(info, dict):
                    continue
                ep_info = info.get("episode")
                if ep_info is None:
                    continue
                if "r" in ep_info:
                    self._ep_returns.append(float(ep_info["r"]))
                if "l" in ep_info:
                    self._ep_lengths.append(float(ep_info["l"]))

        if self.num_timesteps % self._log_interval != 0:
            return True

        env_base = getattr(self._env, "unwrapped", self._env)
        env_map = getattr(env_base, "env_map", None)
        log_payload: dict[str, object] = {}

        rewards = self.locals.get("rewards")
        if rewards is not None:
            log_payload["reward/step_mean"] = float(np.asarray(rewards).mean())
        if self._ep_returns:
            log_payload["rollout/ep_rew_mean"] = float(np.mean(self._ep_returns))
        if self._ep_lengths:
            log_payload["rollout/ep_len_mean"] = float(np.mean(self._ep_lengths))

        if env_map is not None and hasattr(env_map, "environment_map"):
            try:
                coverage = (env_map.environment_map != 0).float().mean(dim=(1, 2))
                log_payload["coverage/mean"] = coverage.mean().item()
                log_payload["coverage/min"] = coverage.min().item()
                log_payload["coverage/max"] = coverage.max().item()
                log_payload["coverage/mean_percent"] = (coverage * 100.0).mean().item()
                total_cells = float(env_map.environment_map.shape[1] * env_map.environment_map.shape[2])
                explored_cells = coverage * total_cells
                log_payload["coverage/mean_cells"] = explored_cells.mean().item()
                log_payload["coverage/min_cells"] = explored_cells.min().item()
            except Exception:
                pass

            if hasattr(env_map, "subgoal_active") and env_map.subgoal_active.numel() > 0:
                active_mask = env_map.subgoal_active.bool()
                log_payload["subgoal/active_ratio"] = active_mask.float().mean().item()
                if active_mask.any():
                    distances = env_map.prev_subgoal_distance[active_mask]
                    log_payload["subgoal/distance_mean"] = distances.mean().item()
                    log_payload["subgoal/distance_min"] = distances.min().item()
                    rewards_accum = env_map.subgoal_reward_accum[active_mask] if hasattr(env_map, "subgoal_reward_accum") else None
                    if rewards_accum is not None:
                        log_payload["subgoal/accum_reward_mean"] = rewards_accum.mean().item()

            if hasattr(env_map, "area_diff") and env_map.area_diff.numel() > 0:
                log_payload["coverage/new_cells_mean"] = env_map.area_diff.float().mean().item()

            if hasattr(env_map, "curiosity_reward") and env_map.curiosity_reward.numel() > 0:
                log_payload["curiosity/mean"] = env_map.curiosity_reward.mean().item()

            map_logs = {}
            if self._max_map_images > 0 and hasattr(env_map, "wandb_environment_map_dict"):
                try:
                    map_dict = env_map.wandb_environment_map_dict
                    traj_dict = getattr(env_map, "wandb_drone_traj_dict", {})
                    status_dict = getattr(env_map, "episode_end_status", {})
                    frontier_cache = None
                    unknown_cache_np = None
                    logged = 0
                    for env_idx, grid in map_dict.items():
                        if grid is None or logged >= self._max_map_images:
                            continue
                        arr = grid.detach().cpu().numpy().astype(np.uint8)
                        traj = traj_dict.get(env_idx)
                        arr = np.clip(arr, 0, 3)
                        if traj is not None and traj.numel() > 0:
                            arr = arr.copy()
                            traj_np = traj.detach().cpu().numpy()
                            for point in traj_np:
                                x, y = int(point[0]), int(point[1])
                                if 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1]:
                                    arr[x, y] = 3
                        vis_np = None
                        unknown_np = None
                        frontier_vis = env_map.frontier_snapshot.get(env_idx) if hasattr(env_map, "frontier_snapshot") else None
                        if frontier_vis is not None:
                            vis_np = frontier_vis.detach().cpu().numpy().astype(np.uint8)
                            if hasattr(env_map, "frontier_unknown_snapshot"):
                                unk_snapshot = env_map.frontier_unknown_snapshot.get(env_idx)
                                if unk_snapshot is not None:
                                    unknown_np = unk_snapshot.detach().cpu().numpy().astype(np.uint8)
                        elif hasattr(env_map, "current_frontier_mask"):
                            if frontier_cache is None:
                                candidate_cache, unknown_cache = env_map.current_frontier_mask(include_unknown=True)
                                frontier_cache = candidate_cache.detach().cpu().numpy().astype(np.uint8)
                                unknown_cache_np = unknown_cache.detach().cpu().numpy().astype(np.uint8)
                            vis_np = frontier_cache[env_idx]
                            if unknown_cache_np is not None:
                                unknown_np = unknown_cache_np[env_idx]
                        if vis_np is not None:
                            arr = arr.copy()
                            arr[vis_np == 1] = 4  # frontier candidates
                            arr[vis_np == 2] = 5  # selected frontier
                            arr[vis_np == 3] = 6  # drone marker
                            if frontier_vis is not None:
                                env_map.frontier_snapshot[env_idx] = None
                                if hasattr(env_map, "frontier_unknown_snapshot"):
                                    env_map.frontier_unknown_snapshot[env_idx] = None
                        status = None
                        if isinstance(status_dict, dict):
                            status = status_dict.get(env_idx)
                        status_to_prefix = {
                            "completed": "maps/completed",
                            "crashed": "maps/crashed",
                            "timeout": "maps/timeout",
                            "snapshot": "maps/snapshot",
                            "terminated": "maps/terminated",
                        }
                        prefix = status_to_prefix.get(status, "maps")
                        key = f"{prefix}/env_{env_idx}" if prefix.startswith("maps/") else f"maps/env_{env_idx}"
                        if prefix == "maps":
                            key = f"{prefix}/env_{env_idx}"
                        caption = f"env_{env_idx} ({status if status is not None else 'unspecified'})"
                        palette = np.array([
                            [20, 20, 20],      # unknown
                            [200, 200, 200],   # free space
                            [240, 80, 80],     # obstacle
                            [80, 160, 255],    # trajectory
                            [255, 215, 0],     # frontier candidate
                            [80, 200, 120],    # selected frontier
                            [170, 80, 255],    # drone marker
                        ], dtype=np.uint8)
                        color_img = palette[arr]
                        map_logs[key] = (color_img, caption)
                        logged += 1
                    if logged > 0:
                        env_map.reset_wandb_dicts_ended_episodes()
                except Exception:
                    map_logs = {}

            if map_logs:
                try:
                    import wandb

                    for key, arr in map_logs.items():
                        if isinstance(arr, tuple):
                            image_arr, caption = arr
                        else:
                            image_arr, caption = arr, key
                        log_payload[key] = wandb.Image(image_arr, caption=caption)
                except Exception:
                    pass

        manager = getattr(env_map, "manager", None) if env_map is not None else None
        if manager is not None:
            if hasattr(manager, "epsilon"):
                log_payload["manager/epsilon"] = float(manager.epsilon)
            if hasattr(manager, "_buffer"):
                log_payload["manager/buffer_size"] = float(len(manager._buffer))

        if log_payload:
            try:
                import wandb

                log_payload["global_step"] = self.num_timesteps
                wandb.log(log_payload)
            except Exception:
                pass
        return True


class ManagerCheckpointCallback(BaseCallback):
    """Periodically save the frontier manager state during training."""

    def __init__(
        self,
        manager,
        save_freq: int,
        save_dir: str,
        name_prefix: str = "manager",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._manager = manager
        self._save_freq = max(1, int(save_freq))
        self._save_dir = save_dir
        self._name_prefix = name_prefix
        self._last_saved_step = -1

    def _init_callback(self) -> None:
        os.makedirs(self._save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self._manager is None:
            return True
        step = int(self.model.num_timesteps)
        if step <= 0 or step == self._last_saved_step or (step % self._save_freq) != 0:
            return True
        filename = f"{self._name_prefix}_{step}_steps.pt"
        path = os.path.join(self._save_dir, filename)
        try:
            torch.save(self._manager.state_dict(), path)
            self._last_saved_step = step
            if self.verbose > 0:
                print(f"[INFO] Saved frontier manager checkpoint to {path}")
        except Exception as err:
            if self.verbose > 0:
                print(f"[WARN] Failed to save frontier manager checkpoint ({err})")
        return True

# Import packages to use gymnasium environments
import gymnasium as gym
import random
from DRL_UAV_Indoor_Exploration.isaac45.RL_drone.custom_feature_extractor import check_custom_feature_extractor_in_sb3_cfg
import DRL_UAV_Indoor_Exploration.isaac45.utils.env_mapping_classes as env_mapping
from DRL_UAV_Indoor_Exploration.isaac45.utils.sac import SAC

# Isaac Lab libraries
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml, dump_pickle
from isaaclab_tasks.utils.hydra import hydra_task_config
from DRL_UAV_Indoor_Exploration.isaac45.utils.custom_sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg
from DRL_UAV_Indoor_Exploration.isaac45.utils.vec_dict_frame_stack import VecDictFrameStack
from DRL_UAV_Indoor_Exploration.isaac45.planner import FrontierRLManager

# Stable-Baselines3 tools
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.logger import configure

import numpy as np
import os
from datetime import datetime
import torch



@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Main training function."""
    
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # Set environment and agent configuration
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    if args_cli.max_iterations is not None:
        if "n_steps" in agent_cfg:
            horizon = agent_cfg["n_steps"] * env_cfg.scene.num_envs
            agent_cfg["n_timesteps"] = args_cli.max_iterations * horizon
        else:
            agent_cfg["n_timesteps"] = args_cli.max_iterations


    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Logging setup
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    
    # Save environment and agent configs for reproducibility
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # --- W&B: optional TB sync ---
    _wandb = False
    try:
        import wandb
        wandb.init(
            project=args_cli.wandb_project,
            name=args_cli.wandb_name or f"{args_cli.task}_{run_info}",
            mode=args_cli.wandb_mode,
            dir=log_dir,
            sync_tensorboard=True,   # <- mirrors SB3 TensorBoard scalars to W&B
            resume="allow" if args_cli.resume else None,
            config={
                "task": args_cli.task,
                "seed": args_cli.seed,
                "num_envs": env_cfg.scene.num_envs,
                "planner_mode": args_cli.planner_mode,
            },
        )
        _wandb = True
        print("[INFO] W&B enabled (syncing TensorBoard).")
    except Exception as e:
        print(f"[WARN] W&B disabled: {e}")

    # Post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    agent_cfg = check_custom_feature_extractor_in_sb3_cfg(agent_cfg)

    # Read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")
    planner_mode = args_cli.planner_mode
    manager_enabled = planner_mode != "heuristic" or args_cli.manager_rl
    if not manager_enabled:
        planner_mode = "heuristic"

    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped
    # base_env.raycast_debug_print = args_cli.ray_debug
    # base_env.raycast_debug_print_hits = args_cli.ray_debug_hits
    # ensure_ray_caster_initialized(env)

    env_model = env_mapping.EnvironmentModelFOVTraversability(env.unwrapped.scene.num_envs, env.unwrapped.sim.device, env.unwrapped.scene.env_origins)
    env.unwrapped.env_map = env_model
    env_model.manager_mode = planner_mode
    if hasattr(env_model, "map_snapshot_interval"):
        env_model.map_snapshot_interval = max(0, int(args_cli.map_snapshot_interval))

    frontier_manager = None
    if manager_enabled and planner_mode != "heuristic":
        device = getattr(base_env, "device", getattr(base_env.sim, "device", "cpu"))
        train_manager = planner_mode in ("assist", "rl")
        frontier_manager = FrontierRLManager(
            device=device,
            max_candidates=args_cli.manager_max_candidates,
            lr=args_cli.manager_lr,
            gamma=args_cli.manager_gamma,
            epsilon=args_cli.manager_epsilon,
            training=train_manager,
            mode=planner_mode,
        )
        env_model.register_manager(frontier_manager, mode=planner_mode)
        if args_cli.manager_load_path is not None:
            try:
                state = torch.load(args_cli.manager_load_path, map_location=device)
                frontier_manager.load_state_dict(state)
                print(f"[INFO] Loaded frontier manager from {args_cli.manager_load_path}")
            except Exception as err:
                print(f"[WARN] Failed to load frontier manager checkpoint ({err})")

    # Wrapper around environment for SB3: actions are normalized forward/back velocity [-1,1] and yaw rate [-1,1]
    env = Sb3VecEnvWrapper(env, lower_bound=np.array([-1, -1]), upper_bound=np.array([1, 1]))
    if args_cli.obs_stack > 1:
        env = VecDictFrameStack(env, n_stack=args_cli.obs_stack)
    
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # Initialize SAC agent from SB3
    start_timesteps = 0
    if args_cli.resume:
        try:
            agent = SAC.load(
                args_cli.resume,
                env=env,
                print_system_info=True,
                tensorboard_log=os.path.join(log_dir, "tensorboard_runs"),
            )
            start_timesteps = int(getattr(agent, "num_timesteps", 0))
            print(f"[INFO] Resumed from {args_cli.resume} at {start_timesteps} timesteps.")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint ({e}); starting fresh.")
            agent = SAC(policy_arch, env, verbose=1, **agent_cfg)
    else:
        agent = SAC(policy_arch, env, verbose=1, **agent_cfg)

    # Optionally load IL weights into SAC agent
    if args_cli.use_IL:
        checkpoint_path = args_cli.IL_model_path
        bc_model = torch.load(checkpoint_path, map_location="cpu")
        state_dict = bc_model["state_dict"] if isinstance(bc_model, dict) and "state_dict" in bc_model else bc_model

        # Gather feature-extractor weights (old checkpoints used "extractors.", newer use "image_extractors"/"line_extractors")
        feature_prefix = "feature_extractor."
        feature_extractor_state = {}
        for key, value in state_dict.items():
            if key.startswith(feature_prefix):
                stripped_key = key[len(feature_prefix):]
                if stripped_key.startswith("extractors."):
                    stripped_key = stripped_key.replace("extractors.", "", 1)
                feature_extractor_state[stripped_key] = value

        with torch.no_grad():
            feature_extractor = agent.policy.actor.features_extractor
            if feature_extractor_state:
                missing_keys, unexpected_keys = feature_extractor.load_state_dict(feature_extractor_state, strict=False)
                if missing_keys:
                    print(f"[WARN] Missing IL feature weights for keys: {missing_keys}")
                if unexpected_keys:
                    print(f"[WARN] Unexpected IL feature weights ignored: {unexpected_keys}")
            else:
                print("[WARN] No feature extractor weights found in IL checkpoint.")

            def _copy_param(target, key: str) -> None:
                if key in state_dict:
                    target.copy_(state_dict[key])
                else:
                    print(f"[WARN] IL checkpoint missing '{key}'; skipped transfer.")

            # First hidden layer (input → first hidden layer)
            _copy_param(agent.policy.actor.latent_pi[0].weight, "actor_fc.0.weight")
            _copy_param(agent.policy.actor.latent_pi[0].bias, "actor_fc.0.bias")

            # Second hidden layer (first hidden → second hidden layer)
            _copy_param(agent.policy.actor.latent_pi[2].weight, "actor_fc.3.weight")
            _copy_param(agent.policy.actor.latent_pi[2].bias, "actor_fc.3.bias")

            # Output layers
            _copy_param(agent.policy.actor.mu.weight, "actor_head.weight")
            _copy_param(agent.policy.actor.mu.bias, "actor_head.bias")
            _copy_param(agent.policy.actor.log_std.weight, "log_std.weight")
            _copy_param(agent.policy.actor.log_std.bias, "log_std.bias")

        # Optionally fill replay buffer
        if args_cli.fill_replay_buffer:
            obs = env.reset()
            n_prefill_steps = 10000  # Set to a reasonable number of steps for warm-up

            for iter in range(n_prefill_steps):
                # Sample action using the pretrained IL policy
                action, _states = agent.predict(obs, deterministic=True)

                # Step environment
                next_obs, reward, done, info = env.step(action)

                done_env_ids = [i for i, d in enumerate(done) if d]

                # Reset only the done environments
                if done_env_ids:
                    reset_obs, _ = env.unwrapped.reset(env_ids=torch.tensor(done_env_ids, dtype=torch.int64, device=env.unwrapped.device))

                # Store transition in replay buffer
                agent.replay_buffer.add(obs, next_obs, action, reward, done, info)

                # Move to next observation
                obs = next_obs
                print(f"Prefill step {iter}/{n_prefill_steps}")
            agent.save_replay_buffer("logs/sb3/replay_buffer_widelens.pkl")
            print("[INFO] Replay buffer prefilled.")

        # Otherwise, load existing replay buffer from disk
        elif args_cli.buffer_path:
            agent.load_replay_buffer(args_cli.buffer_path)

  
    # Option to load agent from checkpoint: 
    # checkpoint_path = "/home/desiree/Documents/Desiree/IsaacLab/logs/sb3/Drone_learn/2025-04-01_15-01-19/model_600000_steps.zip"
    # agent = SAC.load(checkpoint_path, env, print_system_info=True, tensorboard_log=os.path.join(log_dir, f"tensorboard_runs"))

    # configure the logger
    new_logger = configure(log_dir, ["stdout", "csv"])
    agent.set_logger(new_logger)
    # callbacks for agent
    # callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="model", verbose=2)
    tqdm_cb = TqdmCallback(total_target=n_timesteps, start=start_timesteps, update_interval=1000)
    callback_list = [checkpoint_callback, tqdm_cb]
    if frontier_manager is not None:
        manager_ckpt_dir = os.path.join(log_dir, "manager_checkpoints")
        manager_checkpoint = ManagerCheckpointCallback(
            manager=frontier_manager,
            save_freq=checkpoint_callback.save_freq,
            save_dir=manager_ckpt_dir,
            name_prefix="manager",
            verbose=1,
        )
        callback_list.append(manager_checkpoint)
    if _wandb:
        callback_list.append(WandbMetricsCallback(env, log_interval=200))
    callbacks = CallbackList(callback_list)

    # train the agent
    remaining = max(0, int(n_timesteps) - int(start_timesteps))
    print(f"[INFO] Training for {remaining} timesteps (target total: {n_timesteps}).")
    agent.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=not bool(args_cli.resume),
    )

    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    if frontier_manager is not None:
        manager_path = os.path.join(log_dir, "manager_state.pt")
        torch.save(frontier_manager.state_dict(), manager_path)
        print(f"[INFO] Saved frontier manager checkpoint to {manager_path}")
        if _wandb:
            try:
                wandb.save(manager_path)
            except Exception:
                pass

    # close the simulator
    env.close()

    if _wandb:
        try:
            import wandb; wandb.finish()
        except Exception:
            pass

if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
