# Register custom Gym environments for drone training:
# - "Drone_SAC_IL" uses imitation learning (IL) weights and configuration.
# - "Drone_SAC_no_IL" uses standard SAC configuration without IL.

import gymnasium as gym

from . import agents
from . import env_config_training
from . import env_config_eval_envA
from . import env_config_eval_envB
from . import env_config_eval_envC

# Training registers
gym.register(
    id="Drone_SAC_IL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_training.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_IL_cfg.yaml", 
    },
)

# Training registers
gym.register(
    id="Drone_SAC_IL_V1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_training.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_IL_v1.yaml", 
    },
)

gym.register(
    id="Drone_SAC_no_IL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_training.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)


gym.register(
    id="Drone_SAC_no_IL_V1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_training.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg_v1.yaml",
    },
)

gym.register(
    id="Drone_SAC_mobilenet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_training.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_mobilenet.yaml",
    },
)

gym.register(
    id="Drone_SAC_convnextv2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_training.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_convnextv2_nano.yaml",
    },
)

# Evaluation registers
gym.register(
    id="Drone_eval_envA",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_eval_envA.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_IL_cfg.yaml", 
    },
)
gym.register(
    id="Drone_eval_envB",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_eval_envB.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_IL_cfg.yaml", 
    },
)
gym.register(
    id="Drone_eval_envC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_eval_envC.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_IL_cfg.yaml", 
    },
)

gym.register(
    id="Drone_eval_envB_V1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_config_eval_envB.DroneEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg_v1.yaml", 
    },
)
