#!/usr/bin/env python3

# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for variable impedance cabinet manipulation task.

This script trains an RL agent to learn variable impedance parameters
for opening cabinet drawers using the Franka robot.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train variable impedance cabinet manipulation agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=2000, help="RL Policy training iterations.")
# Note: --headless and --device are added by AppLauncher.add_app_launcher_args()

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add the data directory to Python path to import singlearm
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(script_dir)
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)

# Import our custom environment - now singlearm should be found
import singlearm  # noqa: F401

# Type ignore for dynamic tensor attributes that pyright can't detect
from typing import Any, Dict, Union
import torch

from isaaclab.utils.dict import print_dict

# Set up basic training configuration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train variable impedance cabinet manipulation agent."""
    
    # === PREFERRED SOLUTION: Use parse_env_cfg helper ===
    try:
        # Import IsaacLab tasks to ensure environments are registered
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        import gymnasium as gym
        
        task = "Isaac-Variable-Impedance-Cabinet-v0"
        
        # Override default number of environments if not specified
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else 2
        
        print("=== PREFERRED SOLUTION: Using parse_env_cfg ===")
        print(f"Task: {task}")
        print(f"Number of environments: {num_envs}")
        print(f"Device: {args_cli.device}")
        print(f"Seed: {args_cli.seed}")
        
        # Parse environment configuration - this fills all defaults automatically
        env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=num_envs)
        
        # Create the environment using gym.make with cfg parameter
        env = gym.make(task, cfg=env_cfg)
        
        print("✓ Environment created successfully using parse_env_cfg!")
        
    except Exception as e:
        print(f"✗ parse_env_cfg approach failed: {e}")
        print("\n=== FALLBACK SOLUTION: Manual configuration ===")
        
        # === MANUAL FALLBACK: Fill missing reward params manually ===
        from singlearm.variable_impedance_cabinet_env_cfg import VariableImpedanceCabinetEnvCfg
        from isaaclab.envs import ManagerBasedRLEnv
        
        # Override default number of environments if not specified
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else 2
        
        print(f"Creating environment with manual configuration")
        print(f"Number of environments: {num_envs}")
        print(f"Device: {args_cli.device}")
        print(f"Seed: {args_cli.seed}")
        
        # Create environment configuration and override settings
        env_cfg = VariableImpedanceCabinetEnvCfg()
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.device = args_cli.device
        
        # === FIX MISSING REWARD PARAMETERS ===
        # 1. Fix approach_gripper_handle.params.offset
        env_cfg.rewards.approach_gripper_handle.params["offset"] = 0.04
        
        # 2. Fix grasp_handle.params.open_joint_pos
        env_cfg.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        
        # 3. Fix grasp_handle.params.asset_cfg.joint_names
        env_cfg.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger_.*"]
        
        print("✓ Fixed missing reward parameters:")
        print(f"  - approach_gripper_handle.offset = {env_cfg.rewards.approach_gripper_handle.params['offset']}")
        print(f"  - grasp_handle.open_joint_pos = {env_cfg.rewards.grasp_handle.params['open_joint_pos']}")
        print(f"  - grasp_handle.asset_cfg.joint_names = {env_cfg.rewards.grasp_handle.params['asset_cfg'].joint_names}")
        
        # Create the environment directly
        env = ManagerBasedRLEnv(cfg=env_cfg)
        
        print("✓ Environment created successfully using manual configuration!")
    
    # Set seed for reproducibility - use gym.Env.reset method with seed parameter instead
    # as env.seed is deprecated in newer gymnasium versions
    
    print("\n" + "="*80)
    print("ENVIRONMENT INFORMATION")
    print("="*80)
    print(f"Environment type: {type(env)}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of environments: {getattr(env, 'num_envs', num_envs)}")
    print(f"Episode length: {getattr(env, 'max_episode_length', 'unknown')}")
    
    # Print action space details
    if hasattr(env.action_space, 'shape'):
        print(f"Action space shape: {env.action_space.shape}")
        if hasattr(env.action_space, 'low'):
            print(f"Action space low: {env.action_space.low}")
        if hasattr(env.action_space, 'high'):
            print(f"Action space high: {env.action_space.high}")
    
    # Reset environment
    print("\n" + "="*80)
    print("TESTING ENVIRONMENT")
    print("="*80)
    
    # Reset with seed
    obs, _ = env.reset(seed=args_cli.seed)
    print(f"Initial observation shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")
    
    # Test a few random actions
    print("\nTesting random actions...")
    for step in range(5):
        actions = env.action_space.sample()
        # Convert numpy array to torch tensor if needed
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(env.unwrapped.device)
        
        obs, reward, terminated, truncated, info = env.step(actions)
        
        print(f"Step {step+1}:")
        
        # Handle reward printing based on type
        if hasattr(reward, 'mean') and hasattr(reward, 'std'):
            # Batch of rewards (tensor)
            print(f"  Reward: {reward.mean():.4f} ± {reward.std():.4f}")
        else:
            # Single reward (scalar)
            print(f"  Reward: {reward:.4f}")
            
        # Handle termination flags safely
        if hasattr(terminated, 'sum'):
            # Batch of terminations (tensor)
            term_count = terminated.sum().item()
            term_total = len(terminated)
            print(f"  Terminated: {term_count}/{term_total}")
        else:
            # Single termination flag
            print(f"  Terminated: {terminated}")
        
        if step == 0:
            # Print action info for first step
            if hasattr(actions, 'shape'):
                print(f"  Action shape: {actions.shape}")
            print(f"  Observation shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")
            
            # Print reward breakdown for first step
            if hasattr(info, 'get') and info.get('episode') is not None:
                episode_info = info['episode']
                if 'reward_terms' in episode_info:
                    print("  Reward breakdown:")
                    for term_name, term_value in episode_info['reward_terms'].items():
                        if hasattr(term_value, 'mean'):
                            print(f"    {term_name}: {term_value.mean():.4f}")
                        else:
                            print(f"    {term_name}: {term_value:.4f}")
    
    print("\n" + "="*80)
    print("ENVIRONMENT TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nTo train the agent with RSL-RL, run:")
    print("./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Variable-Impedance-Cabinet-v0")
    print("\nTo play with a trained policy, run:")
    print("./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Variable-Impedance-Cabinet-v0")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
