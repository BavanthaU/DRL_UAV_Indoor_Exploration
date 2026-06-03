from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm

from isaaclab.utils.math import matrix_from_quat, euler_xyz_from_quat, quat_from_euler_xyz
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import custom_actions_cfg

from .quadrotor_controller import RLSetpointController
from .quadrotor_controller_2D_action import RL2DSetpointController
from .quadrotor_controller_discrete_action_limited_5vals import RLDiscreteLimited5ValuesSetpointController
from .quadrotor_controller_manual import KeyboardInputActions
from .quadrotor_controller_manual import KeyboardSetpointController

class KeyBoardDrivenAction(ActionTerm):
    
    cfg: custom_actions_cfg.KeyBoardDrivenActionCfg

    def __init__(self, cfg: custom_actions_cfg.KeyBoardDrivenActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # Setup Low-level controller
        self.controller = KeyboardSetpointController(
            num_envs = self.num_envs,            
            Kp=[10.0, 10.0, 10.0],
            Kd=[4.0, 4.0, 4.0],             #  Kd=[8.5, 8.5, 8.5], Kd=[4.0, 4.0, 4.0],
            Ki=[1.5, 1.5, 1.5], 
            Kr=[3.5, 3.5, 3.5],
            Kw=[0.5, 0.5, 0.5]
        )
        print('[INFO] Controller set up')

        self.dt = env.cfg.sim.dt
        self.robot = env.scene["robot"]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)                       # DIMENSION: (#env, 3) -> 2nd DIM: x-axis (forward / backwards) / y-axis (right / left) / Turn in z-axis  (CW/CCW)                         
        self._processed_actions = torch.zeros(self.num_envs, self._asset.num_bodies, 6, device=self.device)       # DIMENSION: (#env, #bodies, 6) -> 3rd DIM: Force [x, y, z] + Torque [x, y, z]

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):

        # NOTE: Due to the structure of IsaacLab, actions must be passed to the process_actions method.
        #       However, when manually controlling the drone with the keyboard, these actions are not utilized.
        #       Instead, the controller class computes setpoints internally by capturing keyboard inputs.
        self._raw_actions[:] = actions 

        # Apply low-level controller
        self.controller.update_state(state= self.robot.data.root_state_w)                                       # root state DIMENSION = (# env, 13) ->  2nd DIM: [pos (x,y,z), quat(q_w,q_x,q_y,q_z), lin_vel(v_x,v_y,v_z), ang_vel (w_x,w_y,w_z)]) in simulation world frame.
        wrench = self.controller.update(dt=self.dt).to(torch.device(self.device))                               # OUTPUT DIMENSION = # envs x 6

        body_num = 1
        self._processed_actions = wrench.unsqueeze(1).repeat(1, body_num, 1)                                    # DIMENSION: (#env, #body_parts_where_force_torque_is_applied, 6)

    def apply_actions(self):
        self._asset.set_external_force_and_torque(forces=self._processed_actions[:,:,0:3], torques= self._processed_actions[:,:, 3:6], body_ids=0)





class RLDrivenAction(ActionTerm):
    
    cfg: custom_actions_cfg.RLDrivenActionCfg

    def __init__(self, cfg: custom_actions_cfg.RLDrivenActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # Setup Low-level controller
        # self.controller = RLSetpointController(
        #     num_envs = self.num_envs,            
        #     Kp=[5.0, 5.0,8.0],
        #     Kd=[2, 2, 5],
        #     Ki=[1.5, 1.5, 1.5], 
        #     Kr=[3.5, 3.5, 2.0],
        #     Kw=[1, 1, 1]
        # )
        self.controller = RLSetpointController(
            num_envs = self.num_envs,            
            Kp=[10.0, 10.0, 10.0],
            Kd=[8.5, 8.5, 8.5],
            Ki=[1.5, 1.5, 1.5], 
            Kr=[3.5, 3.5, 3.5],
            Kw=[1.5, 1.5, 1.5]
        )
        print('[INFO] Controller set up')

        self.dt = env.cfg.sim.dt
        self.robot = env.scene["robot"]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)                       # DIMENSION: (#env, 3) -> 2nd DIM: x-axis (forward / backwards) / y-axis (right / left) / Turn in z-axis  (CW/CCW)                         
        self._processed_actions = torch.zeros(self.num_envs, self._asset.num_bodies, 6, device=self.device)       # DIMENSION: (#env, #bodies, 6) -> 3rd DIM: Force [x, y, z] + Torque [x, y, z]
        self.step_counter = 6
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):
        self.step_counter += 1

        if self.step_counter > 6:
            self._raw_actions = actions
            self.step_counter = 0
        # self._raw_actions[:] = actions 

        # Apply low-level controller
        self.controller.update_state(state= self.robot.data.root_state_w)                                       # root state DIMENSION = (# env, 13) ->  2nd DIM: [pos (x,y,z), quat(q_w,q_x,q_y,q_z), lin_vel(v_x,v_y,v_z), ang_vel (w_x,w_y,w_z)]) in simulation world frame.
        wrench = self.controller.update(dt=self.dt, action = self._raw_actions).to(torch.device(self.device))   # OUTPUT DIMENSION = # envs x 6

        body_num = 1
        self._processed_actions = wrench.unsqueeze(1).repeat(1, body_num, 1)                                    # DIMENSION: (#env, #body_parts_where_force_torque_is_applied, 6)

    def apply_actions(self):
        self._asset.set_external_force_and_torque(forces=self._processed_actions[:,:,0:3], torques= self._processed_actions[:,:, 3:6], body_ids=0)

        
class RLDriven2DAction(ActionTerm):
    
    cfg: custom_actions_cfg.RLDriven2DActionCfg

    def __init__(self, cfg: custom_actions_cfg.RLDriven2DActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # Setup Low-level controller
        # self.controller = RL2DSetpointController(
        #     num_envs = self.num_envs,            
        #     Kp=[5.0, 5.0, 10.0],
        #     Kd=[3.5, 3.5, 8.5],
        #     Ki=[1.5, 1.5, 1.5], 
        #     Kr=[1.5, 1.5, 1.5],
        #     Kw=[1.0, 1.0, 1.0]
        # )
        self.controller = RL2DSetpointController(
            num_envs = self.num_envs,  
        Kp=[10.0, 10.0, 10.0],
            Kd=[8.5, 8.5, 8.5],
            Ki=[1.5, 1.5, 1.5], 
            Kr=[3.5, 3.5, 3.5],
            Kw=[1.5, 1.5, 1.5]
        )
        print('[INFO] Controller set up')

        self.dt = env.cfg.sim.dt
        self.robot = env.scene["robot"]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)                       # DIMENSION: (#env, 3) -> 2nd DIM: x-axis (forward / backwards) / y-axis (right / left) / Turn in z-axis  (CW/CCW)                         
        self._processed_actions = torch.zeros(self.num_envs, self._asset.num_bodies, 6, device=self.device)       # DIMENSION: (#env, #bodies, 6) -> 3rd DIM: Force [x, y, z] + Torque [x, y, z]

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):

        self._raw_actions[:] = actions 

        # Apply low-level controller
        self.controller.update_state(state= self.robot.data.root_state_w)                                       # root state DIMENSION = (# env, 13) ->  2nd DIM: [pos (x,y,z), quat(q_w,q_x,q_y,q_z), lin_vel(v_x,v_y,v_z), ang_vel (w_x,w_y,w_z)]) in simulation world frame.
        wrench = self.controller.update(dt=self.dt, action = self._raw_actions).to(torch.device(self.device))   # OUTPUT DIMENSION = # envs x 6

        body_num = 1
        self._processed_actions = wrench.unsqueeze(1).repeat(1, body_num, 1)                                    # DIMENSION: (#env, #body_parts_where_force_torque_is_applied, 6)

    def apply_actions(self):

        self._asset.set_external_force_and_torque(forces=self._processed_actions[:,:,0:3], torques= self._processed_actions[:,:, 3:6], body_ids=0)



class RLDrivenDiscreteLimited5ValuesAction(ActionTerm):
    
    cfg: custom_actions_cfg.RLDrivenDiscreteLimited5ValuesActionCfg

    def __init__(self, cfg: custom_actions_cfg.RLDrivenDiscreteLimited5ValuesActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # Setup Low-level controller
        self.controller = RLDiscreteLimited5ValuesSetpointController(
            num_envs = self.num_envs,            
            Kp=[10.0, 10.0, 10.0],
            Kd=[8.5, 8.5, 8.5],
            Ki=[1.5, 1.5, 1.5], 
            Kr=[3.5, 3.5, 3.5],
            Kw=[0.5, 0.5, 0.5]
        )
        print('[INFO] Controller set up')

        self.dt = env.cfg.sim.dt
        self.robot = env.scene["robot"]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)                       # DIMENSION: (#env, 3) -> 2nd DIM: x-axis (forward / backwards) / y-axis (right / left) / Turn in z-axis  (CW/CCW)                         
        self._processed_actions = torch.zeros(self.num_envs, self._asset.num_bodies, 6, device=self.device)       # DIMENSION: (#env, #bodies, 6) -> 3rd DIM: Force [x, y, z] + Torque [x, y, z]

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):

        self._raw_actions[:] = actions 

        # Apply low-level controller
        self.controller.update_state(state= self.robot.data.root_state_w)                                       # root state DIMENSION = (# env, 13) ->  2nd DIM: [pos (x,y,z), quat(q_w,q_x,q_y,q_z), lin_vel(v_x,v_y,v_z), ang_vel (w_x,w_y,w_z)]) in simulation world frame.
        wrench = self.controller.update(dt=self.dt, action = self._raw_actions).to(torch.device(self.device))   # OUTPUT DIMENSION = # envs x 6

        body_num = 1
        self._processed_actions = wrench.unsqueeze(1).repeat(1, body_num, 1)                                    # DIMENSION: (#env, #body_parts_where_force_torque_is_applied, 6)

    def apply_actions(self):
        self._asset.set_external_force_and_torque(forces=self._processed_actions[:,:,0:3], torques= self._processed_actions[:,:, 3:6], body_ids=0)



class IdealRLDrivenAction(ActionTerm):

    # NOTE: this Action Term is only useful for decimation=1. 
    
    cfg: custom_actions_cfg.IdealRLDrivenActionCfg

    def __init__(self, cfg: custom_actions_cfg.IdealRLDrivenActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self.dt = env.cfg.sim.dt
        self.robot = env.scene["robot"]
        self.lin_vel = 0.8  # m/s
        self.ang_vel = 0.8#40*math.pi/180
        self.z_pos =  1.8 #1.8 

        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)              # tensor of shape (num_envs, 3) -> 2nd DIM: x-axis (forward / backwards) / y-axis (right / left) / Turn in z-axis  (CW/CCW)                 
        self._processed_actions = torch.zeros(self.num_envs, 13, device=self.device)       # Tensor of shape (num_envs, 13). 2nd DIM: [pos (x,y,z), quat(q_w,q_x,q_y,q_z), lin_vel(v_x,v_y,v_z), ang_vel (w_x,w_y,w_z)]) in simulation world frame.
        self.update_interval_steps = int(0.05 / self.dt)  # Every 12 steps
        self.step_counter = self.update_interval_steps  # Track steps
        # self.prev_actions = torch.tensor([0,0,0], device=self.device).unsqueeze(0)
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):  # actions.shape = (num_envs, 3)

        self._raw_actions = actions
        
        # Scale actions (assuming actions are in range [-1,1])
        drone_lin_vel_body_frame = torch.cat((self._raw_actions[:, :2] * self.lin_vel, torch.zeros(self.num_envs, 1, device=self.device)), dim=1)  # Scale linear velocities
        drone_ang_vel_body_frame = torch.cat(
            (torch.zeros(self.num_envs, 2, device=self.device), self._raw_actions[:, 2].unsqueeze(1) * self.ang_vel),
            dim=1
        )  # Only yaw rotation

        # Transform to world frame
        drone_world_orientations = self.robot.data.root_quat_w
        drone_rotation_matrices = matrix_from_quat(drone_world_orientations)
        drone_lin_vel_world_frame = torch.bmm(drone_rotation_matrices, drone_lin_vel_body_frame.unsqueeze(2)).squeeze(2)

        # Compute reference position
        v_ref = torch.cat((drone_lin_vel_world_frame[:, :2], torch.zeros(self.num_envs, 1, device=self.device)), dim=1)
        ang_vel_ref = torch.bmm(drone_rotation_matrices, drone_ang_vel_body_frame.unsqueeze(2)).squeeze(2)

        drone_world_positions = self.robot.data.root_pos_w
        p_ref = drone_world_positions + self.dt * v_ref
        p_ref[:, 2] = self.z_pos  # Keep fixed height

        yaw_ref = euler_xyz_from_quat(drone_world_orientations)[2] + ang_vel_ref[:, 2] * self.dt

        # Construct processed actions
        self._processed_actions[:, 0:3] = p_ref
        self._processed_actions[:, 3:7] = quat_from_euler_xyz(
            roll=torch.zeros(self.num_envs, device=self.device),
            pitch=torch.zeros(self.num_envs, device=self.device),
            yaw=yaw_ref
        )
    def apply_actions(self):
        self._asset.write_root_state_to_sim(root_state=self._processed_actions)
        
class IdealRLDrivenAction2D(ActionTerm):

    # NOTE: this Action Term is only useful for decimation=1. 
    
    cfg: custom_actions_cfg.IdealRLDrivenActionCfg

    def __init__(self, cfg: custom_actions_cfg.IdealRLDrivenActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self.dt = env.cfg.sim.dt
        self.robot = env.scene["robot"]
        self.lin_vel = 4.0  # m/s
        self.ang_vel = 3.0 #40*math.pi/180
        self.z_pos =  2.0 #1.8 

        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)              # tensor of shape (num_envs, 3) -> 2nd DIM: x-axis (forward / backwards) / y-axis (right / left) / Turn in z-axis  (CW/CCW)                 
        self._processed_actions = torch.zeros(self.num_envs, 13, device=self.device)       # Tensor of shape (num_envs, 13). 2nd DIM: [pos (x,y,z), quat(q_w,q_x,q_y,q_z), lin_vel(v_x,v_y,v_z), ang_vel (w_x,w_y,w_z)]) in simulation world frame.
        self.update_interval_steps = int(0.05 / self.dt)  # Every 12 steps
        self.step_counter = self.update_interval_steps  # Track steps
        # self.prev_actions = torch.tensor([0,0,0], device=self.device).unsqueeze(0)
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):  # actions.shape = (num_envs, 3)

        self._raw_actions = actions
        
        # Scale actions (assuming actions are in range [-1,1])
        drone_lin_vel_body_frame = torch.cat(((self._raw_actions[:, 0] * self.lin_vel).unsqueeze(1), torch.zeros(self.num_envs, 2, device=self.device)), dim=1)  # Scale linear velocities
        drone_ang_vel_body_frame = torch.cat(
            (torch.zeros(self.num_envs, 2, device=self.device), self._raw_actions[:, 1].unsqueeze(1) * self.ang_vel),
            dim=1
        )  # Only yaw rotation

        # Transform to world frame
        drone_world_orientations = self.robot.data.root_quat_w
        drone_rotation_matrices = matrix_from_quat(drone_world_orientations)
        drone_lin_vel_world_frame = torch.bmm(drone_rotation_matrices, drone_lin_vel_body_frame.unsqueeze(2)).squeeze(2)

        # Compute reference position
        v_ref = torch.cat((drone_lin_vel_world_frame[:, :2], torch.zeros(self.num_envs, 1, device=self.device)), dim=1)
        ang_vel_ref = torch.bmm(drone_rotation_matrices, drone_ang_vel_body_frame.unsqueeze(2)).squeeze(2)

        drone_world_positions = self.robot.data.root_pos_w
        p_ref = drone_world_positions + self.dt * v_ref
        p_ref[:, 2] = self.z_pos  # Keep fixed height

        yaw_ref = euler_xyz_from_quat(drone_world_orientations)[2] + ang_vel_ref[:, 2] * self.dt

        # Construct processed actions
        self._processed_actions[:, 0:3] = p_ref
        self._processed_actions[:, 3:7] = quat_from_euler_xyz(
            roll=torch.zeros(self.num_envs, device=self.device),
            pitch=torch.zeros(self.num_envs, device=self.device),
            yaw=yaw_ref
        )
    def apply_actions(self):
        self._asset.write_root_state_to_sim(root_state=self._processed_actions)

