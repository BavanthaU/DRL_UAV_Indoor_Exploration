import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np

from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from demo_dataset_loader import DemoDatasetLoader
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sac_net import SACNet

import torch.distributions as D

def save_model(net_model, global_iter, global_step, model_dir):
    """
    Save the model checkpoint
    """
    ckpt_file = os.path.join(model_dir,
                             'new_ckpt_{:08d}.pth'.format(global_iter))
    print('\r Saving checkpoint: %s' % ckpt_file)
    data_to_save = {
        'global_iter': global_iter,
        'global_step': global_step,
        'state_dict': net_model.state_dict(),
    }
    torch.save(data_to_save, ckpt_file)


def parse_arguments():
    """
    Parse command line arguments with argparse.
    Sets hyperparameters and directory paths with defaults.
    """
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--demo_dir', type=str, default='/workspace/isaaclab/DRL_UAV_Indoor_Exploration/trained_model_and_trajectories/expert_trajectories',
                        help='directory of human demonstration data')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu for rendering')
    parser.add_argument('--traj_len', type=int, default=600,
                        help='truncated trajectory length')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0.000,
                        help='weight_decay')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='save model every n iterations')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='logging every n steps')
    parser.add_argument('--max_iters', type=int, default=1000, 
                        help='maximum number of episodes/iterations')
    parser.add_argument('--fix_cnn', action='store_true',
                        help='fix cnn weights')
    parser.add_argument('--save_dir', type=str, default='./occupancy_depthline_semline')
    return parser.parse_args()


def main():
    """
    Main training loop: parse arguments, setup environment and model,
    load data, and perform training and validation.
    """
    args = parse_arguments()

    print('Program starts at: \033[92m %s '
                '\033[0m' % datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    # Set device to CUDA if available and not disabled, else CPU
    args.device = torch.device("cuda:0" if torch.cuda.is_available()
                                           and not args.disable_cuda
                               else "cpu")
    config = vars(args)
    train_mode = True

    # Setup directories for logs and models
    log_dir = os.path.join(args.save_dir, 'logs')
    model_dir = os.path.join(args.save_dir, 'model')
    os.makedirs(args.save_dir, exist_ok=True)
    shutil.rmtree(args.save_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Make a tensorboard writer and save hyperparameter to json file. 
    writer = SummaryWriter(log_dir=log_dir)
    if train_mode:
        with open(os.path.join(args.save_dir, 'hyperparams.json'), 'w') as f:
            hps = {key: val for key, val in config.items() if key != 'device'}
            json.dump(hps, f, indent=2)

    # Define observation spaces for different inputs
    depth_image_space = spaces.Box(low=0, high=1, shape=(1,48, 64),dtype=np.float32)
    occ_map_space = spaces.Box(low=0, high=4, shape=(4, 61, 61),dtype=np.float32)
    sem_space = spaces.Box(low=0, high=6, shape=(6, 48, 64),dtype=np.float32)
    lin_vel_space = spaces.Box(low=-2.0, high=2.0, shape=(1,3), dtype=np.float32)
    ang_vel_space = spaces.Box(low=-2.0, high=2.0, shape=(1,3), dtype=np.float32)
    oned_depth_space = spaces.Box(low=0, high=1, shape=(1,64),dtype=np.float32)
    oned_sems_space = spaces.Box(low=0, high=6, shape=(1,64),dtype=np.float32)


    # Make dictionary observation space for the model input.
    observation_space = spaces.Dict({
        'depth_obs': depth_image_space,
        'map_obs': occ_map_space,
        'semantic_obs': sem_space,
        'linear_velocity': lin_vel_space,
        'angular_velocity': ang_vel_space,
        'oneline_depth': oned_depth_space,
        'oneline_sems': oned_sems_space,
    })
    
    # Instantiate the SAC model with action dimension=2 and the observation space defined above
    net_model = SACNet(act_dim=2, observation_space=observation_space,
                              device=args.device,
                              fix_cnn=False)
               
    net_model.to(args.device)
    
    # Initialize iteration and step variable for saving to tensorboard
    global_iter = 0
    global_step = 0
    global_step_val = 0
    traj_step = 0
    traj_step_val = 0

    optimizer = optim.Adam(net_model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # Loss function: Smooth L1 (Huber) loss for regression. 
    # Good for continuous actions since small difference is not punished as harsly as with MSE lsos
    loss_criterion = nn.SmoothL1Loss().to(args.device)

    # Load demonstration dataset
    train_dataset = DemoDatasetLoader(root_dir=args.demo_dir,
                                           seq_len=args.traj_len)
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  shuffle=False, num_workers=0, drop_last=False)

    for iter in tqdm(range(global_iter, args.max_iters), desc='iter'):
        global_iter = iter
        net_model.train()
        for i_batch, (occ_map, sems, linear_velocities, angular_velocities, expert_actions, oneline_depth, oneline_sems) in enumerate(train_dataloader):
            global_step += 1

            # Extract episode data, squeeze batch dimension if needed
            ep_occ_maps = occ_map.float().squeeze(0)
            ep_sems = sems.float().squeeze(0)
            ep_linear_velocities = linear_velocities
            ep_angular_velocities = angular_velocities
            ep_oneline_depth = oneline_depth
            ep_oneline_sems = oneline_sems
            ep_expert_actions = expert_actions

            iter_loss = []
            # Process episode trajectory in batches of batch_size steps
            for start in range(0, args.traj_len, args.batch_size):
                traj_step += 1
                end = start + args.batch_size

                # Slice the batch segment for each input and move to device
                it_occ_map = ep_occ_maps[start: end].to(args.device).float()
                it_sems = ep_sems[start: end].to(args.device).float()
                it_linear_velocities = ep_linear_velocities[:,start: end].to(args.device).squeeze(0).float()
                it_angular_velocities = ep_angular_velocities[:,start: end].to(args.device).squeeze(0).float()
                it_oneline_depth = ep_oneline_depth[:,start: end].to(args.device).squeeze(0).float()
                it_oneline_sems = ep_oneline_sems[:,start: end].to(args.device).squeeze(0).float()
                
                it_actions = ep_expert_actions[:,start:end].to(args.device).squeeze(0).float()
                
                # Define observations dictionary (comment/uncomment to test different inputs)
                observations = {
                            # 'depth_obs': it_depths,       # Depth image or observation image
                            'map_obs': it_occ_map,
                            # 'semantic_obs': it_sems,
                            # 'position': it_position,
                            # 'linear_velocity': it_linear_velocities, # 3D position
                            # 'angular_velocity': it_angular_velocities, # 4D orientation (e.g., Euler angles or quaternions)
                            'oneline_depth': it_oneline_depth,
                            'oneline_sems': it_oneline_sems,
                        }

                # Forward pass through the model
                _, pi_mean, pi_std = net_model(observations)

                # Calculate loss between predicted mean action and expert actions
                loss = loss_criterion(pi_mean, it_actions)
                loss = loss.float()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), 10.0)
                optimizer.step()

                # Log training loss for this batch
                writer.add_scalar('train/loss', loss, traj_step)
                iter_loss.append(loss.item())
            
            # Log mean loss for this iteration
            iter_loss = np.mean(iter_loss)
            writer.add_scalar('train/iter_loss', iter_loss, global_step)     

        if global_iter % args.save_interval == 0:
            val_loss = 0
            val_correct = 0
            val_total = 0
            net_model.eval()
            with torch.no_grad():
                for i_batch, (occ_map,sems, linear_velocities, angular_velocities, expert_actions, oneline_depth, oneline_sems) in enumerate(train_dataloader):
                    global_step_val += 1

                    # Extract episode data, squeeze batch dimension if needed
                    ep_occ_maps = occ_map.float().squeeze(0)
                    ep_sems = sems.float().squeeze(0)
                    ep_linear_velocities = linear_velocities
                    ep_angular_velocities = angular_velocities
                    ep_oneline_depth = oneline_depth
                    ep_oneline_sems = oneline_sems
                    ep_expert_actions = expert_actions
                    
                    iter_loss = []
                    # Process episode trajectory in batches of batch_size steps
                    for start in range(0, args.traj_len, 1):
                        traj_step_val += 1
                        end = start + 1
                        
                        # Slice the batch segment for each input and move to device
                        it_occ_map = ep_occ_maps[start: end].to(args.device).float()
                        it_sems = ep_sems[start: end].to(args.device).float()
                        it_linear_velocities = ep_linear_velocities[:,start: end].to(args.device).squeeze(0).float()
                        it_angular_velocities = ep_angular_velocities[:,start: end].to(args.device).squeeze(0).float()
                        it_oneline_depth = ep_oneline_depth[:,start: end].to(args.device).squeeze(0).float()
                        it_oneline_sems = ep_oneline_sems[:,start: end].to(args.device).squeeze(0).float()

                        it_actions = ep_expert_actions[:,start:end].to(args.device).squeeze(0).float()


                        # Same observation dict as training
                        observations = {
                                    # 'depth_obs': it_depths,       # Depth image or observation image
                                    'map_obs': it_occ_map,
                                    # 'semantic_obs': it_sems,
                                    # 'position': it_position,
                                    # 'linear_velocity': it_linear_velocities, # 3D position
                                    # 'angular_velocity': it_angular_velocities, # 4D orientation (e.g., Euler angles or quaternions)
                                    'oneline_depth': it_oneline_depth,
                                    'oneline_sems': it_oneline_sems,
                                }

                        # Forward pass
                        q_value, pi_mean, pi_std = net_model(observations)

                        # Compute "correct" prediction mask within 0.1 tolerance
                        correct_mask = torch.abs(pi_mean - it_actions) < 0.1
                        difference = torch.abs(pi_mean - it_actions)

                        # Accumulate number of correct predictions and total samples
                        val_correct += correct_mask.sum().item()
                        val_total += correct_mask.numel()

                        # Calculate validation accuracy so far and log
                        val_acc = val_correct / val_total
                        writer.add_scalar('test/approx_acc', val_acc, traj_step_val)

                        # Log difference of each action dimension separately
                        writer.add_scalar('test/difference0', difference[:, 0], traj_step_val)
                        writer.add_scalar('test/difference1', difference[:, 1], traj_step_val)

                        # Compute loss and accumulate
                        loss = loss_criterion(pi_mean, it_actions)
                        loss = loss.float()
                        iter_loss.append(loss.item())

                    iter_loss = np.mean(iter_loss)
                    val_loss += iter_loss

            # Log accuracy to tensorboard writer
            val_acc = val_correct / float(val_total)
            writer.add_scalar('test/loss', val_loss, global_step_val)
            writer.add_scalar('test/acc', val_acc, global_step_val)

            # Save model checkpoint every 100 iterations
            if iter % 100 == 0:
                save_model(net_model=net_model, global_iter=global_iter,
                           global_step=global_step, model_dir=model_dir)

    # Save final model after training loop ends
    save_model(net_model=net_model, global_iter=global_iter,
               global_step=global_step,
               model_dir=model_dir)


if __name__ == '__main__':
    main()



