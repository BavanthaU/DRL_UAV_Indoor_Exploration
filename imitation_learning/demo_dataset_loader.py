import os
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader

import h5py


class DemoDatasetLoader(Dataset):
    """
    Custom Dataset for loading Isaac simulation data from HDF5 files.
    """
    def __init__(self, root_dir="DRL_UAV_Indoor_Exploration/trained_model_and_trajectories/expert_trajectories", seq_len=400):
        self.root_dir = root_dir
        self.seq_len = seq_len

        # List all demo folders
        demo_folders = os.listdir(self.root_dir)
        demo_folders.sort()

        self.demo_folders = [os.path.join(self.root_dir, x) for x in demo_folders]

        # Get all HDF5 files
        self.h5_dir = root_dir
        self.h5_files = [f for f in os.listdir(self.h5_dir) if f.endswith(".h5")]

    def __len__(self):
        return len(self.demo_folders)

    def __getitem__(self, idx=0):
        # Randomly select a .h5 data file
        chosen_file = random.choice(self.h5_files)
        h5_path = os.path.join(self.h5_dir, chosen_file)

        # Load data from the HDF5 file
        with h5py.File(h5_path, 'r') as f:
            occ_map_data = f['map_obs'][:]             # Occupancy map
            sem_data = f['semantic_obs'][:]            # Semantic image
            linear_velocity_data = f['root_lin_vel'][:]  # Drone linear velocity
            angular_velocity_data = f['root_ang_vel'][:]  # Drone angular velocity
            depthline_data = f['oneline_depth'][:]     # 1D depth information
            sems_line_data = f['oneline_sem'][:]       # 1D semantic information

        traj_len = len(occ_map_data)

        # Ensure sequence is long enough
        if traj_len < self.seq_len:
            from IPython import embed
            embed()


        start_idx = random.randint(2, traj_len - self.seq_len - 1)

        # Initialize lists
        sems = []
        occ_maps = []
        oneline_depth = []
        oneline_sems = []
        linear_velocities = []
        angular_velocities = []
        actions = []

        flip_data = np.random.choice([0, 1])  # Randomly flips demo data for data augmentation

        for k in range(start_idx, start_idx + self.seq_len + 1):
            
            # Augmentation branch: original (non-flipped)
            if flip_data == 1:
                sem = sem_data[k-1]
                sems.append(sem[0])

                oneline_sems.append(sems_line_data[k - 1][0][0])

                occ_map = occ_map_data[k - 1]
                occ_maps.append(occ_map[0])

                # Process velocities
                linear_velocity = np.array([linear_velocity_data[k - 1][0, 0] * 0.8, 0.0, 0.0])
                angular_velocity = np.array([0.0, 0.0, angular_velocity_data[k - 1][0, 1] * 0.8])
                linear_velocities.append(linear_velocity)
                angular_velocities.append(angular_velocity)

                # Process depth line
                oneline_depth.append(depthline_data[k - 1])

                # Generate noisy action by adding Gaussian noise to linear velocity (Useful for data augmentation and robustness)
                action = 0.8 * linear_velocity_data[k]
                noise = np.random.normal(loc=0.0, scale=0.05, size=action.shape)
                noisy_action = [[0,0]]
                # noisy_action = np.clip(action + noise, 0, 1)   # Clip values to [0, 1] range to avoid invalid actions
                noisy_action[0][0] = np.clip(action[0][0] + noise[0][0], 0, 1)    # Clip so values do not cross the boundaries of 0 and 1
                noisy_action[0][1] = np.clip(action[0][1] + noise[0][1], -1, 1)
                # noisy_action = np.where(noisy_action < 0.03, 0, noisy_action)   # Clip so values close to zero do not contain noise
                actions.append(np.array([noisy_action[0][0],noisy_action[0][1]]))

            # Augmentation branch: Horizontal Flip
            if flip_data  == 0:           
                sem = sem_data[k-1]
                sem = np.flip(sem, axis=-1)
                sems.append((sem)[0])

                oneline_sems.append(sems_line_data[k - 1][0][0][::-1])
                
                occ_map =  occ_map_data[k-1]
                occ_map = np.flip(occ_map, axis=-1)
                occ_maps.append(occ_map[0])

                # Process velocities
                linear_velocity = np.array([linear_velocity_data[k-1][0,0]*0.8, 0.0, 0.0]) 
                angular_velocity = np.array([0.0, 0.0,-angular_velocity_data[k-1][0,1]*0.8]) 
                linear_velocities.append(linear_velocity)
                angular_velocities.append(angular_velocity)

                # Process depth line
                oneline_depth_processed = depthline_data[k-1][:,::-1]       
                oneline_depth.append(oneline_depth_processed)

                # Process action with Gaussian noise
                action = 0.8 * linear_velocity_data[k]
                noise = np.random.normal(loc=0.0, scale=0.05, size=action.shape)
                noisy_action = [[0,0]]
                noisy_action[0][0] = np.clip(action[0][0] + noise[0][0], 0, 1)    # Clip so values do not cross the boundaries of 0 and 1
                noisy_action[0][1] = np.clip(action[0][1] + noise[0][1], -1, 1)
                # print(noisy_action)
                # noisy_action = np.where(noisy_action < 0.03, 0, noisy_action)
                actions.append(np.array([noisy_action[0][0], -noisy_action[0][1]]))



        # Convert lists to NumPy arrays
        sems = np.array(sems)
        occ_maps = np.array(occ_maps)
        linear_velocities = np.array(linear_velocities)
        angular_velocities = np.array(angular_velocities)
        actions = np.array(actions)
        oneline_depth = np.array(oneline_depth)
        oneline_sems = np.array(oneline_sems)

        return occ_maps, sems, linear_velocities, angular_velocities, actions, oneline_depth, oneline_sems


if __name__ == "__main__":
    # Create dataset and dataloader
    human_demo = DemoDatasetLoader()
    dataloader = DataLoader(human_demo, batch_size=1, shuffle=True, num_workers=0)

    # Iterate through the dataloader
    for i, (occ_maps, sems, linear_velocities, angular_velocities, actions, oneline_depth, oneline_sems) in enumerate(dataloader):
        print(i)




