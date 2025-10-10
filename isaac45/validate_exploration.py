"""
Script for evaluating a drone exploration policy in Isaac Sim.
Can be for pure IL, hybrid IL-RL and pure RL. 
Through task name can the testing environment be changed to A, B or C. 
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate a trained IL or RL drone")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Drone_eval_envA", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--val_IL", action="store_true", default=False, help="Turn this on if only IL will be evaluated")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# Filter out AppLauncher-injected args before Hydra sees them
sys.argv = [sys.argv[0]] + [arg for arg in hydra_args if not arg.startswith("--/")]
# Import packages to use gymnasium environments
import gymnasium as gym
import random
from DRL_UAV_Indoor_Exploration.isaac45.RL_drone.custom_feature_extractor import check_custom_feature_extractor_in_sb3_cfg
import DRL_UAV_Indoor_Exploration.isaac45.utils.env_mapping_classes as env_mapping
from DRL_UAV_Indoor_Exploration.isaac45.utils.sac import SAC

# Import used RL Isaac Lab libraries
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml, dump_pickle
from isaaclab_tasks.utils.hydra import hydra_task_config
from DRL_UAV_Indoor_Exploration.isaac45.utils.custom_sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg

# Stable-Baselines3 tools
from stable_baselines3.common.vec_env import VecNormalize


# Import other needed libraries
import numpy as np
import os
from datetime import datetime
import torch.nn as nn
import torch
import csv
import os

# Create csv file for saving evaluation data. 
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename_data = f'DRL_UAV_Indoor_Exploration/evaluation_files/{args_cli.task}_data_{timestamp}.csv'
file_exists = os.path.isfile(filename_data)
if not os.path.isfile(filename_data) or os.path.getsize(filename_data) == 0:
    with open(filename_data, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['reward', 'length', 'occupied_cells', 'explore_reward'])

  

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Main evaluation function."""
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # Set environment and agent configuration
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Directory for logging into
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

    # Post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    agent_cfg = check_custom_feature_extractor_in_sb3_cfg(agent_cfg)

    # Read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")
    print("Using policy:", policy_arch)
    
    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env_model = env_mapping.EnvironmentModelFOVTraversability(env.unwrapped.scene.num_envs, env.unwrapped.sim.device, env.unwrapped.scene.env_origins)   
    env.unwrapped.env_map = env_model
    
    # Wrapper around environment for SB3: actions are forward velocity [0,1] and rotational velocity [-1,1]
    env = Sb3VecEnvWrapper(env, lower_bound=np.array([0, -1]) , upper_bound=np.array([1, 1]))

    # Normalize env
    env = VecNormalize(
        env,
        training=True,
        norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
        norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
        clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        gamma=agent_cfg["gamma"],
        clip_reward=np.inf,
    )
    
    # Create agent from stable baselines
    # agent = SAC(policy_arch, env, verbose=1, **agent_cfg)
    
    if args_cli.val_IL:
        # Load in Behavior Cloning model
        checkpoint_path = args_cli.checkpoint
        bc_model = torch.load(checkpoint_path, map_location="cpu")
        state_dict = bc_model["state_dict"]
        
        # Rename extractor keys (different between IL and RL agents)
        extractor_state_dict = {
            key.replace("feature_extractor.extractors.",""): value
            for key, value in state_dict.items()
            if key.startswith("feature_extractor.extractors.")}
    
        # Transfer pretrained weights into SAC policy
        with torch.no_grad():
            # Transfer feature extractor weights
            agent.policy.actor.features_extractor.extractors.load_state_dict(extractor_state_dict)
            
            # First hidden layer (input → first hidden layer)
            agent.policy.actor.latent_pi[0].weight.copy_(bc_model["state_dict"]["actor_fc.0.weight"])
            agent.policy.actor.latent_pi[0].bias.copy_(bc_model["state_dict"]["actor_fc.0.bias"])
      
            # Second hidden layer (first hidden → second hidden layer)
            agent.policy.actor.latent_pi[2].weight.copy_(bc_model["state_dict"]["actor_fc.3.weight"])
            agent.policy.actor.latent_pi[2].bias.copy_(bc_model["state_dict"]["actor_fc.3.bias"])
      
            # Output layers
            agent.policy.actor.mu.weight.copy_(bc_model["state_dict"]["actor_head.weight"])
            agent.policy.actor.mu.bias.copy_(bc_model["state_dict"]["actor_head.bias"])
            agent.policy.actor.log_std.weight.copy_(bc_model["state_dict"]["log_std.weight"])
            agent.policy.actor.log_std.bias.copy_(bc_model["state_dict"]["log_std.bias"])
    elif not args_cli.val_IL:
        # Load in trained RL model, SAC or IL+SAC
        checkpoint_path = args_cli.checkpoint
        agent = SAC.load(checkpoint_path, env, print_system_info=True, tensorboard_log=os.path.join(log_dir, f"tensorboard_runs"))
        
    # Get environment and policy
    obs = env.reset()
    num_dones = 0
    full_exploration_time = 0
    
    while num_dones < 60:
        action, _ = agent.predict(obs, deterministic=True)

        # Step environment
        next_obs, reward, done, info = env.step(action)
        done_env_ids = [i for i, d in enumerate(done) if d]

        if done_env_ids:
        # Reset only the done environments
            reset_obs, _ = env.unwrapped.reset(env_ids=torch.tensor(done_env_ids, dtype=torch.int64, device=env.unwrapped.device))
            with open(filename_data, 'a', newline='') as file:
                writer = csv.writer(file)
                ep = info[0]['episode']  # assuming 'entry' is your latest episode dict
                writer.writerow([
                    ep['r'],
                    ep['l'],
                    ep['occupied_cells'],
                    ep['Episode_Reward/explore'].item()
                ])
            num_dones += 1
            print(num_dones)
            full_exploration_time = 0


        # Move to next observation
        obs = next_obs
        
    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
