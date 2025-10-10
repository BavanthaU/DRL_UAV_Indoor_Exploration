#!/usr/bin/env python
"""
| File: quadrotor_controller_2D_action.py
| Author: Andrea Bravo i Forn
   NOTE: 
   - Code adapted from Marcelo Jacinto and Joao Pinto (creators of Pegasus Simulator).
   - Receives velocity commands from a RL algorithm and converts them into body-wrench commands for the drone, enabling control in the 2D plane at a fixed height.
   - It needs to be used with the RLDrivenAction action term (set up from the class ActionsCfg with the RLDrivenActionCfg class).
   - The only drone-specific parameters needed are: drone mass (m), height at which the drone flies (self.z_pos), maximum linear velocity of the drone (self.lin_vel), and maximum angular velocity of the drone (self.ang_vel).
   - FLU drone frame assumed (x-axis = Forwards (red), y-axis = Left (green), z-axis = Up (blue)).
   - In the prim tree, the drone body must be the first item (otherwise the body_ids in the RLDrivenAction class needs to be changed).
"""
# Auxiliary scipy, torch and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation
import torch


class RL2DSetpointController():
    """A nonlinear controller class. It implements a nonlinear controller that allows a drone to track
    aggressive trajectories. 
    """

    def __init__(self,  
        num_envs: int = 1,
        Kp=None,
        Kd=None,
        Ki=None,
        Kr=None,
        Kw=None):

        # Number of spawned environments
        self.num_envs = num_envs 

        # Wrench to apply [N, Nm] for all environments
        self.input_ref = torch.zeros((self.num_envs, 6))

        # The drone state in each environment. See details in the update_state() method.
        self.p = np.zeros((self.num_envs, 3))                   # The vehicle position
        identity_quat = np.tile([0, 0, 0, 1], (self.num_envs, 1))
        self.R = Rotation.from_quat(identity_quat)              # The vehicle attitude                 
        self.w = np.zeros((self.num_envs, 3))                   # The angular velocity of the vehicle
        self.v = np.zeros((self.num_envs, 3))                   # The linear velocity of the vehicle in the inertial frame
        self.a = np.zeros((self.num_envs, 3))                   # The linear acceleration of the vehicle in the inertial frame

        # The drone desired state in each environment.
        self.p_ref = np.zeros((self.num_envs, 3))
        self.v_ref = np.zeros((self.num_envs, 3))
        self.a_ref = np.zeros((self.num_envs, 3))
        self.j_ref = np.zeros((self.num_envs, 3))
        self.yaw_ref =  np.zeros(self.num_envs)                        
        self.yaw_rate_ref =  np.zeros(self.num_envs)

        # Define the control gains matrix for the outer-loop
        self.Kp = np.diag(Kp)
        self.Kd = np.diag(Kd)
        self.Ki = np.diag(Ki)
        self.Kr = np.diag(Kr)
        self.Kw = np.diag(Kw)


        self.int = np.array([0.0, 0.0, 0.0])   
   
        # Define the dynamic parameters for the vehicle
        self.m = 1.50                       # Mass in Kg
        self.g = 9.81                       # The gravity acceleration ms^-2
       
        # Define parameters
        self.max_lin_vel = 2.0
        self.max_ang_vel = 1.5
        self.z_pos = 1.8                    # Height at which the drone flies

    def update_state(self, state: torch.Tensor):
        """
        Method that updates the current state of the drone in each environment.

        Args:
            state (torch.Tensor): Shape is (num_envs, 13). In the 2nd dim: [pos (x,y,z), quat(q_w,q_x,q_y,q_z), lin_vel(v_x,v_y,v_z), ang_vel (w_x,w_y,w_z)]) in the simulation world (inertial) frame. 
        
        NOTE:
            - self.p: numpy array of size (num_envs, 3) with the positions [x,y,z] of the drone-attached frame in each environment, w.r.t the common world inertial frame, expressed in the world frame.
            - orientation_quat: numpy array of size (num_envs, 4) with the quaternions [qx, qy, qz, qw] encoding the orientation of the drone-attached frame in each environment, w.r.t the common world frame, expressed in the world frame.
                                The input quaterions have to be re-ordered to fit the scipy.spatial.transform library convention.
            - self.R: Rotation element storing the rotation matrices of the drone-attached frame in each environment, w.r.t the common world frame, expressed in the world frame.
            - self.w: numpy array of size (num_envs, 3) with the [w_x,w_y,w_z] angular velocities of the drone-attached frame in each environment, w.r.t the common world frame, expressed in the drone FLU frame.
            - self.v: numpy array of size (num_envs, 3) with the [v_x,v_y,v_z] linear velocities of the drone-attached frame in each environment, w.r.t the common world frame, expressed in the world frame.
            - self.a: numpy array of size (num_envs, 3) with the [a_x,a_y,a_z] linear accelerations of the drone-attached frame in each environment, w.r.t the world frame, expressed in the world frame.
        """

        if self.num_envs != state.size(0):
            raise ValueError(f"Unexpected state row number: rows in state {state.size(0)}, env_num: {self.num_envs}.")
        
        orientation_quat = np.zeros((self.num_envs, 4))
        for i in range(self.num_envs):
            self.p[i,:] = state[i,:3].cpu().numpy()

            orientation_quat[i,:] = state[i,3:7].cpu().numpy()
            orientation_quat[i,:] = orientation_quat[i, [1, 2, 3, 0]]
            self.R[i] = Rotation.from_quat(orientation_quat[i,:])

            w_world_frame= state[i, 10:13].cpu().numpy()
            self.w[i,:] = self.R[i].as_matrix().T  @ w_world_frame

            self.v[i,:] = state[i, 7:10].cpu().numpy()                      

    def update(self, dt: float, action: torch.Tensor) -> torch.Tensor:
        """Method that updates the target wrench applied to the drone body, based on inputs from a RL algorithm and executing a nonlinear control law. 
        This method will be called by the simulation on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
            action (torch.Tensor): linear and angular velocity commands in the body frame to move the drone in 2D (at a fixed height). Shape is (num_env, 2). In the 2nd dim: [v_x, w_z] -> v_x > 0 (go forward), v_x < 0 (go backwards) / w_z > 0 (turn CCW), w_z < 0 (turn CW)

        """
        self.update_ref(dt, action)

        self.apply_control_law (dt)

        return self.input_ref
    
    def update_ref(self, dt: float, action: torch.Tensor):
        """Method that updates the references for the controller to track. . The action from the RL algorithm lives in a 2D, bouned, box-like continuous action space. 
        The reference signal is: target position [m], target velocity [m/s], target acceleration [m/s^2], target jerk [m/s^3], target yaw-angle [rad], target yaw-rate [rad/s]
        
        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
            action (torch.Tensor): linear and angular velocity commands in the body frame to move the drone in 2D (at a fixed height). Shape is (num_env, 2). In the 2nd dim: [v_x, w_z] -> v_x > 0 (go forward), v_x < 0 (go backwards) / w_z > 0 (turn CCW), w_z < 0 (turn CW)
        """

        action = action.cpu().numpy()

        for i in range(self.num_envs):
            vxy_world_frame =  self.R[i].as_matrix()  @ np.array([action[i, 0]*self.max_lin_vel, 0 , 0])
            self.v_ref[i, 0:2] = vxy_world_frame[0:2] 
            self.yaw_rate_ref[i] = action[i,1] * self.max_ang_vel

            self.p_ref[i,:] = self.p[i,:] + self.v_ref[i,:]*dt
            self.p_ref[i,2] = self.z_pos
            # self.p_ref[i,2] = self.z_pos - 0.1 * np.linalg.norm(self.v_ref[i, :2])


            euler_angles = self.R[i].as_euler('xyz')
            self.yaw_ref[i] = euler_angles[2] + self.yaw_rate_ref[i]*dt

    def apply_control_law(self, dt: float):

        for i in range(self.num_envs):

            # Compute the tracking errors
            ep = self.p[i,:] - self.p_ref[i,:]
            ev = self.v[i,:] - self.v_ref[i,:]
            self.int = self.int +  (ep * dt)
            ei = self.int

            # Compute F_des term
            F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei) + np.array([0.0, 0.0, self.m * self.g]) + (self.m * self.a_ref[i,:])
            
            # Get the current axis Z_B (given by the last column of the rotation matrix)
            Z_B = self.R[i].as_matrix()[:,2]
            
            # Get the desired total thrust in Z_B direction (u_1)
            u_1 = F_des @ Z_B

            self.input_ref [i, 2] = u_1

            # Compute the desired body-frame axis Z_b
            Z_b_des = F_des / np.linalg.norm(F_des)
            
            # Compute X_C_des 
            X_c_des = np.array([np.cos(self.yaw_ref[i]), np.sin(self.yaw_ref[i]), 0.0])
            
            # Compute Y_b_des
            Z_b_cross_X_c = np.cross(Z_b_des, X_c_des)
            Y_b_des = Z_b_cross_X_c / np.linalg.norm(Z_b_cross_X_c)
            
            # Compute X_b_des
            X_b_des = np.cross(Y_b_des, Z_b_des)
            
            # Compute the desired rotation R_des = [X_b_des | Y_b_des | Z_b_des]
            R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
            R = self.R[i].as_matrix()

            # Compute the rotation error
            e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))

            # Compute an approximation of the current vehicle acceleration in the inertial frame (since we cannot measure it directly)
            self.a[i,:] = (u_1 * Z_B) / self.m - np.array([0.0, 0.0, self.g])

            # Compute the desired angular velocity by projecting the angular velocity in the Xb-Yb plane
            # projection of angular velocity on xB − yB plane
            # see eqn (7) from [2].
            hw = (self.m / u_1) * (self.j_ref[i,:] - np.dot(Z_b_des, self.j_ref[i,:]) * Z_b_des) 
            
            # desired angular velocity
            w_des = np.array([-np.dot(hw, Y_b_des), 
                            np.dot(hw, X_b_des), 
                            self.yaw_rate_ref[i] * Z_b_des[2]])
            
            # Compute the angular velocity error
            e_w = self.w[i,:] - w_des

            # Compute the torques to apply on the rigid body
            tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

            self.input_ref [i, 3:6] = torch.from_numpy(tau)


    @staticmethod
    def vee(S):
        """Auxiliary function that computes the 'v' map which takes elements from so(3) to R^3.
            Retrieves the vector corresponding to a skew symmetric matrix.
        Args:
            S (np.array): A matrix in so(3)
        """
        return np.array([-S[1,2], S[0,2], -S[0,1]])