"""
Script for training a drone explorationpolicy in Isaac Sim using SAC.
Supports initialization from imitation learning (IL) and replay buffer pre-filling. 
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a drone for a RL task")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Drone_SAC_IL", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--use_IL", action="store_true", default=False, help="Use IL as pretrained weights")
parser.add_argument("--IL_model_path", type=str, default=None, help="Path to IL.")
parser.add_argument("--fill_replay_buffer", action="store_true", default=False, help="Fill replay buffer with IL trajectories")
parser.add_argument("--buffer_path", type=str, default=None, help="Path to buffer file..")
parser.add_argument("--resume", type=str, default=None, help="Path to SB3 .zip checkpoint to resume from")
parser.add_argument("--wandb_project", type=str, default="isaac-drone-higherarchical", help="W&B project name")
parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name (optional)")
parser.add_argument("--wandb_mode", type=str, default="online", help="'online'|'offline'|'disabled'")
parser.add_argument("--ray_debug", action="store_true", help="Print ray-caster min distances each step.")
parser.add_argument("--ray_debug_hits", action="store_true", help="Additionally print raw ray hit points (env 0).")
parser.add_argument("--manager_rl", action="store_true", help="Enable RL-based frontier manager.")
parser.add_argument("--manager_max_candidates", type=int, default=8, help="Maximum frontier candidates considered by the manager.")
parser.add_argument("--manager_lr", type=float, default=1e-3, help="Learning rate for the frontier manager policy.")
parser.add_argument("--manager_gamma", type=float, default=0.95, help="Discount factor for frontier manager rewards.")
parser.add_argument("--manager_epsilon", type=float, default=0.1, help="Epsilon-greedy exploration for the frontier manager.")
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
    """Periodic logging of custom exploration metrics to Weights & Biases."""

    def __init__(self, env, log_interval: int = 200):
        super().__init__()
        self._env = env
        self._log_interval = max(1, log_interval)

    def _on_step(self) -> bool:
        if self.num_timesteps % self._log_interval != 0:
            return True

        env_base = getattr(self._env, "unwrapped", self._env)
        env_map = getattr(env_base, "env_map", None)
        metrics: dict[str, float] = {}

        if env_map is not None and hasattr(env_map, "environment_map"):
            try:
                coverage = (env_map.environment_map != 0).float().mean(dim=(1, 2))
                metrics["coverage/mean"] = coverage.mean().item()
                metrics["coverage/min"] = coverage.min().item()
                metrics["coverage/max"] = coverage.max().item()
            except Exception:
                pass

            if hasattr(env_map, "prev_subgoal_distance") and env_map.prev_subgoal_distance.numel() > 0:
                metrics["subgoal/distance_mean"] = env_map.prev_subgoal_distance.mean().item()
                metrics["subgoal/distance_min"] = env_map.prev_subgoal_distance.min().item()

            if hasattr(env_map, "subgoal_reward_accum") and env_map.subgoal_reward_accum.numel() > 0:
                metrics["subgoal/accum_reward_mean"] = env_map.subgoal_reward_accum.mean().item()

            if hasattr(env_map, "area_diff") and env_map.area_diff.numel() > 0:
                metrics["coverage/new_cells_mean"] = env_map.area_diff.float().mean().item()

        manager = getattr(env_map, "manager", None) if env_map is not None else None
        if manager is not None:
            if hasattr(manager, "epsilon"):
                metrics["manager/epsilon"] = float(manager.epsilon)
            if hasattr(manager, "_buffer"):
                metrics["manager/buffer_size"] = float(len(manager._buffer))

        if metrics:
            try:
                import wandb

                wandb.log(metrics, step=self.num_timesteps)
            except Exception:
                pass
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
from DRL_UAV_Indoor_Exploration.isaac45.mdp.common import ensure_ray_caster_initialized
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
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs


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

    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped
    base_env.raycast_debug_print = args_cli.ray_debug
    base_env.raycast_debug_print_hits = args_cli.ray_debug_hits
    ensure_ray_caster_initialized(env)

    frontier_manager = None
    if args_cli.manager_rl and hasattr(base_env, "env_map"):
        device = getattr(base_env, "device", getattr(base_env.sim, "device", "cpu"))
        frontier_manager = FrontierRLManager(
            device=device,
            max_candidates=args_cli.manager_max_candidates,
            lr=args_cli.manager_lr,
            gamma=args_cli.manager_gamma,
            epsilon=args_cli.manager_epsilon,
            training=True,
        )
        base_env.env_map.register_manager(frontier_manager)
    env_model = env_mapping.EnvironmentModelFOVTraversability(env.unwrapped.scene.num_envs, env.unwrapped.sim.device, env.unwrapped.scene.env_origins)
    env.unwrapped.env_map = env_model

    # Wrapper around environment for SB3: actions are normalized forward/back velocity [-1,1] and yaw rate [-1,1]
    env = Sb3VecEnvWrapper(env, lower_bound=np.array([-1, -1]), upper_bound=np.array([1, 1]))
    
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
    new_logger = configure(log_dir, ["stdout", "tensorboard", "csv"])
    agent.set_logger(new_logger)
    # callbacks for agent
    # callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="model", verbose=2)
    tqdm_cb = TqdmCallback(total_target=n_timesteps, start=start_timesteps, update_interval=1000)
    callback_list = [checkpoint_callback, tqdm_cb]
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
