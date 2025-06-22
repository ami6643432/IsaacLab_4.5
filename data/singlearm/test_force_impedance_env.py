#!/usr/bin/env python3

# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for force-based variable impedance cabinet manipulation task.

This script tests the new force-based variable impedance environment that uses
NVIDIA's manager workflow approach.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test force-based variable impedance cabinet manipulation.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

# Import IsaacLab tasks to ensure environments are registered
import isaaclab_tasks  # noqa: F401

def main():
    """Test force-based variable impedance cabinet manipulation environment."""
    
    # Create the environment
    task = "Isaac-Open-Drawer-Franka-Force-Impedance-v0"
    env = gym.make(task, num_envs=args_cli.num_envs)
    
    print("=" * 80)
    print("FORCE-BASED VARIABLE IMPEDANCE ENVIRONMENT TEST")
    print("=" * 80)
    print(f"Environment: {task}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Print action space details
    if hasattr(env.action_space, 'shape'):
        print(f"Action space shape: {env.action_space.shape}")
        print(f"Action space dimensionality: {env.action_space.shape[-1]}")
        
        # Expected: pose (6) + impedance (12) + gripper (1) = 19D
        expected_dim = 19
        actual_dim = env.action_space.shape[-1]
        if actual_dim == expected_dim:
            print(f"✓ Action space has expected {expected_dim} dimensions")
            print("  - Pose commands: 6D (position + orientation)")
            print("  - Impedance parameters: 12D (6 stiffness + 6 damping)")
            print("  - Gripper action: 1D (binary open/close)")
        else:
            print(f"⚠ Action space has {actual_dim} dimensions (expected {expected_dim})")
    
    # Reset environment
    print("\nTesting environment reset...")
    obs, _ = env.reset(seed=args_cli.seed)
    print(f"✓ Environment reset successful")
    print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
    
    if isinstance(obs, dict) and 'policy' in obs:
        print(f"Policy observation shape: {obs['policy'].shape}")
        
        # Check for force feedback and impedance observations
        obs_dim = obs['policy'].shape[-1]
        print(f"Total observation dimensions: {obs_dim}")
        
        # Expected observations might include:
        # - Robot state, EE pose, contact forces, current impedance, etc.
        if obs_dim > 30:  # Reasonable check for rich observations
            print("✓ Observations include rich state information")
        
    # Test a few steps
    print("\nTesting random actions...")
    for step in range(3):
        actions = env.action_space.sample()
        
        # Convert numpy to torch if needed
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(env.device)
            
        obs, reward, terminated, truncated, info = env.step(actions)
        
        print(f"Step {step+1}:")
        print(f"  Reward: {reward.mean():.4f} ± {reward.std():.4f}")
        print(f"  Terminated: {terminated.sum().item()}/{len(terminated)}")
        
        # Print available info
        if hasattr(info, 'keys'):
            print(f"  Info keys: {list(info.keys())}")
    
    # Check for contact sensor
    if hasattr(env.scene, 'contact_forces'):
        print(f"\n✓ Contact sensor available: {env.scene.contact_forces}")
        contact_data = env.scene.contact_forces.data
        if hasattr(contact_data, 'net_forces_w_magnitude'):
            force_mag = contact_data.net_forces_w_magnitude
            print(f"  Current contact forces: {force_mag.mean():.3f} ± {force_mag.std():.3f} N")
    else:
        print("\n⚠ Contact sensor not found")
    
    print("\n" + "=" * 80)
    print("ENVIRONMENT TEST COMPLETED!")
    print("=" * 80)
    
    # Training command
    print("\nTo train the force-based variable impedance agent:")
    print(f"./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task {task} --num_envs 512")
    
    print("\nTo play with a trained policy:")
    print(f"./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Open-Drawer-Franka-Force-Impedance-Play-v0")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
