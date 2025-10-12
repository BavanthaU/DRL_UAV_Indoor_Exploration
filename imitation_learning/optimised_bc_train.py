import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from demo_dataset_loader import DemoDatasetLoader
from gymnasium import spaces

# from model.sac_net import SACNet
from model.baseline_conv_sac import ILSACBaseline

# ======= Perf knobs =======
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# ======= Weights & Biases =======
import wandb


def save_model(net_model, global_iter, global_step, model_dir):
    ckpt_file = os.path.join(model_dir, f'new_ckpt_{global_iter:08d}.pth')
    print(f'\r Saving checkpoint: {ckpt_file}')
    data_to_save = {
        'global_iter': global_iter,
        'global_step': global_step,
        'state_dict': net_model.state_dict(),
    }
    torch.save(data_to_save, ckpt_file)
    return ckpt_file


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--demo_dir', type=str, default='/workspace/isaaclab/DRL_UAV_Indoor_Exploration/trained_model_and_trajectories/expert_trajectories',
                        help='directory of human demonstration data')
    parser.add_argument('--num_gpu', type=int, default=1, help='number of gpu for rendering')
    parser.add_argument('--traj_len', type=int, default=600, help='truncated trajectory length')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=200, help='batch size per chunk')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--save_interval', type=int, default=10, help='validate/save every N iters (epochs)')
    parser.add_argument('--log_interval', type=int, default=50, help='log every N chunks')
    parser.add_argument('--max_iters', type=int, default=1000, help='maximum number of epochs')
    parser.add_argument('--fix_cnn', action='store_true', help='fix cnn weights')
    parser.add_argument('--save_dir', type=str, default='./occupancy_depthline_semline')

    # wandb
    parser.add_argument('--wandb_project', type=str, default='bc_uav', help='wandb project name')
    parser.add_argument('--wandb_run', type=str, default=None, help='wandb run name (optional)')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='wandb logging mode')

    # optional compile
    parser.add_argument('--compile', action='store_true', help='use torch.compile if available')

    return parser.parse_args()


def main():
    args = parse_arguments()

    print(f'Program starts at: \033[92m {datetime.now().strftime("%Y-%m-%d %H:%M")} \033[0m')

    # Device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.disable_cuda else "cpu")
    using_cuda = (args.device.type == "cuda")

    # Dirs
    log_dir = os.path.join(args.save_dir, 'logs')
    model_dir = os.path.join(args.save_dir, 'model')
    os.makedirs(args.save_dir, exist_ok=True)
    # Keep this behavior (your original): start clean
    shutil.rmtree(args.save_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Observation spaces (unchanged)
    depth_image_space = spaces.Box(low=0, high=1, shape=(1, 48, 64), dtype=np.float32)
    occ_map_space = spaces.Box(low=0, high=4, shape=(4, 61, 61), dtype=np.float32)
    sem_space = spaces.Box(low=0, high=6, shape=(6, 48, 64), dtype=np.float32)
    lin_vel_space = spaces.Box(low=-2.0, high=2.0, shape=(1, 3), dtype=np.float32)
    ang_vel_space = spaces.Box(low=-2.0, high=2.0, shape=(1, 3), dtype=np.float32)
    oned_depth_space = spaces.Box(low=0, high=1, shape=(1, 64), dtype=np.float32)
    oned_sems_space = spaces.Box(low=0, high=6, shape=(1, 64), dtype=np.float32)

    observation_space = spaces.Dict({
        'depth_obs': depth_image_space,
        'map_obs': occ_map_space,
        'semantic_obs': sem_space,
        'linear_velocity': lin_vel_space,
        'angular_velocity': ang_vel_space,
        'oneline_depth': oned_depth_space,
        'oneline_sems': oned_sems_space,
    })

    # Model
    net_model = ILSACBaseline(
        observation_space=observation_space,
        act_dim=2,
        device=args.device,
        activation_fn=nn.ELU,
        log_std_init=-0.5,
        net_arch=dict(pi=[256, 128], qf=[256, 256]),
        features_extractor_kwargs=dict(
            image_out=128,
            line_out=64,
            features_dim=128,
            ego_base_channels=64,
            ego_stages=(2, 2, 3),
            ego_use_se=True,
            ego_use_aspp=True,
            ordered_keys=('map_obs', 'oneline_depth', 'oneline_sems'),
        ),
        action_low=[0.0, -1.0],
        action_high=[1.0, 1.0],
    ).to(args.device)

    # Optional channels_last (often helps convs)
    try:
        net_model = net_model.to(memory_format=torch.channels_last)
        use_channels_last = True
    except Exception:
        use_channels_last = False

    # Optional torch.compile
    if args.compile and hasattr(torch, "compile"):
        try:
            net_model = torch.compile(net_model, mode="max-autotune")
            print("Compiled model with torch.compile.")
        except Exception as e:
            print(f"torch.compile unavailable or failed: {e}")

    # Optimizer / Loss / AMP
    optimizer = optim.Adam(net_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_criterion = nn.SmoothL1Loss().to(args.device)
    scaler = torch.cuda.amp.GradScaler(enabled=using_cuda)

    # Freeze CNN if requested
    if args.fix_cnn:
        for n, p in net_model.named_parameters():
            if "features_extractor" in n:
                p.requires_grad = False

    # Data
    print("data loading....")
    train_dataset = DemoDatasetLoader(root_dir=args.demo_dir, seq_len=args.traj_len)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    print("data loaded!")

    # wandb init
    config = vars(args).copy()
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        mode=args.wandb_mode,
        config=config,
        save_code=True,
    )
    wandb.watch(net_model, log='gradients', log_freq=200, log_graph=False)

    # Train
    global_iter = 0
    global_step = 0
    global_step_val = 0
    traj_step = 0

    for iter_idx in tqdm(range(global_iter, args.max_iters), desc='epoch'):
        global_iter = iter_idx
        net_model.train()

        epoch_losses = []

        for i_batch, (occ_map, sems, linear_velocities, angular_velocities, expert_actions, oneline_depth, oneline_sems) in enumerate(train_dataloader):
            global_step += 1

            # Squeeze batch=1 (trajectory), move once, set memory_format
            ep_occ_maps = occ_map.squeeze(0).to(args.device, non_blocking=True).float()
            ep_oneline_depth = oneline_depth.to(args.device, non_blocking=True).squeeze(0).float()
            ep_oneline_sems = oneline_sems.to(args.device, non_blocking=True).squeeze(0).float()
            ep_expert_actions = expert_actions.to(args.device, non_blocking=True).squeeze(0).float()

            if use_channels_last:
                ep_occ_maps = ep_occ_maps.to(memory_format=torch.channels_last)

            # Vectorize into chunks of B
            T = args.traj_len
            B = args.batch_size
            trimT = (T // B) * B if T >= B else T
            if trimT == 0:
                continue

            occ = ep_occ_maps[:trimT].view(-1, B, *ep_occ_maps.shape[1:])            # [N, B, 4, 61, 61]
            dln = ep_oneline_depth[:trimT].view(-1, B, *ep_oneline_depth.shape[1:])  # [N, B, 1, 64]
            sln = ep_oneline_sems[:trimT].view(-1, B, *ep_oneline_sems.shape[1:])    # [N, B, 1, 64]
            act = ep_expert_actions[:trimT].view(-1, B, ep_expert_actions.shape[-1]) # [N, B, 2]

            chunk_losses = []

            for n in range(occ.shape[0]):
                traj_step += 1

                observations = {
                    'map_obs':       occ[n],  # [B, 4, 61, 61]
                    'oneline_depth': dln[n],  # [B, 1, 64]
                    'oneline_sems':  sln[n],  # [B, 1, 64]
                }

                with torch.cuda.amp.autocast(enabled=using_cuda):
                    _, pi_mean, _ = net_model(observations)
                    loss = loss_criterion(pi_mean, act[n])

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()

                chunk_losses.append(loss.detach().item())

                # Throttle logging
                if traj_step % args.log_interval == 0:
                    wandb.log({'train/loss': chunk_losses[-1],
                               'train/traj_step': traj_step,
                               'train/global_step': global_step})

            # per-trajectory mean loss
            iter_loss = float(np.mean(chunk_losses)) if chunk_losses else 0.0
            epoch_losses.append(iter_loss)

        # per-epoch mean
        if epoch_losses:
            wandb.log({'train/iter_loss': float(np.mean(epoch_losses)),
                       'epoch': global_iter})

        # ===== Validation & checkpoint =====
        if global_iter % args.save_interval == 0:
            net_model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for i_batch, (occ_map, sems, linear_velocities, angular_velocities, expert_actions, oneline_depth, oneline_sems) in enumerate(train_dataloader):
                    global_step_val += 1

                    ep_occ_maps = occ_map.squeeze(0).to(args.device, non_blocking=True).float()
                    ep_oneline_depth = oneline_depth.to(args.device, non_blocking=True).squeeze(0).float()
                    ep_oneline_sems = oneline_sems.to(args.device, non_blocking=True).squeeze(0).float()
                    ep_expert_actions = expert_actions.to(args.device, non_blocking=True).squeeze(0).float()

                    if use_channels_last:
                        ep_occ_maps = ep_occ_maps.to(memory_format=torch.channels_last)

                    T = args.traj_len
                    B = args.batch_size
                    trimT = (T // B) * B if T >= B else T
                    if trimT == 0:
                        continue

                    occ = ep_occ_maps[:trimT].view(-1, B, *ep_occ_maps.shape[1:])
                    dln = ep_oneline_depth[:trimT].view(-1, B, *ep_oneline_depth.shape[1:])
                    sln = ep_oneline_sems[:trimT].view(-1, B, *ep_oneline_sems.shape[1:])
                    act = ep_expert_actions[:trimT].view(-1, B, ep_expert_actions.shape[-1])

                    for n in range(occ.shape[0]):
                        observations = {
                            'map_obs':       occ[n],
                            'oneline_depth': dln[n],
                            'oneline_sems':  sln[n],
                        }
                        _, pi_mean, _ = net_model(observations)
                        diff = (pi_mean - act[n]).abs()  # [B, 2]

                        batch_loss = loss_criterion(pi_mean, act[n]).item()
                        val_loss_sum += batch_loss
                        val_batches += 1

                        val_correct += (diff < 0.1).sum().item()
                        val_total += diff.numel()

                if val_batches > 0:
                    val_loss = val_loss_sum / val_batches
                    val_acc = (val_correct / val_total) if val_total > 0 else 0.0
                    wandb.log({'val/loss': val_loss,
                               'val/approx_acc': val_acc,
                               'epoch': global_iter,
                               'val/global_step_val': global_step_val})

            # Save (also upload to wandb as artifact)
            ckpt_path = save_model(net_model=net_model, global_iter=global_iter,
                                   global_step=global_step, model_dir=model_dir)
            wandb.save(ckpt_path)

    # Final save
    ckpt_path = save_model(net_model=net_model, global_iter=global_iter,
                           global_step=global_step, model_dir=model_dir)
    wandb.save(ckpt_path)
    wandb.finish()


if __name__ == '__main__':
    main()
