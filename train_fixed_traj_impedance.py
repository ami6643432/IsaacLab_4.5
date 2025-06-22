#!/usr/bin/env python3

# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for fixed trajectory impedance control.

This script trains an RL agent to learn optimal impedance parameters
while following a fixed Cartesian trajectory for cabinet manipulation.
"""

import argparse
import gymnasium as gym
import torch
from datetime import datetime
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train fixed trajectory impedance control.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=1500, help="RL Policy training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import data.singlearm  # noqa: F401  # Import to register environments
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from data.singlearm.fixed_traj_impedance_env_cfg import FixedTrajImpedanceEnvCfg
from data.singlearm.agents.rsl_rl_ppo_cfg import FixedTrajImpedancePPORunnerCfg


def create_trajectory_points(num_envs: int, device: str) -> torch.Tensor:
    """Create predefined trajectory points for drawer opening task."""
    # Define key waypoints for drawer opening
    waypoints = [
        [0.45, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],    # Approach handle
        [0.35, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],    # At handle
        [0.25, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],    # Pull drawer
        [0.15, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],    # Fully open
    ]
    
    # Convert to tensor and repeat for all environments
    trajectory = torch.tensor(waypoints, dtype=torch.float32, device=device)
    trajectory = trajectory.unsqueeze(0).repeat(num_envs, 1, 1)
    
    return trajectory


def main():
    """Main training function."""
    
    print("=" * 80)
    print("FIXED TRAJECTORY IMPEDANCE CONTROL TRAINING")
    print("=" * 80)
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Device: {args_cli.device}")
    print(f"Max iterations: {args_cli.max_iterations}")
    print("=" * 80)
    
    # Create environment configuration
    env_cfg = FixedTrajImpedanceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # Create agent configuration
    agent_cfg = FixedTrajImpedancePPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = args_cli.device
    
    # Create environment
    env = gym.make("Isaac-Open-Drawer-Franka-Fixed-Impedance-v0", cfg=env_cfg)
    
    # Generate trajectory points
    trajectory = create_trajectory_points(args_cli.num_envs, args_cli.device)
    
    # Set trajectory in environment (custom method)
    if hasattr(env.unwrapped, 'set_reference_trajectory'):
        env.unwrapped.set_reference_trajectory(trajectory)
    
    # Wrap environment for RSL-RL
    wrapped_env = RslRlVecEnvWrapper(env)
    
    print("\nEnvironment Details:")
    print(f"Action space: {wrapped_env.action_space}")
    print(f"Expected: 12D (6 stiffness + 6 damping)")
    print(f"Observation space: {wrapped_env.observation_space}")
    
    # Create log directory
    log_dir = os.path.join("logs", "fixed_traj_impedance", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # Create runner
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # Load checkpoint if provided
    if args_cli.checkpoint:
        print(f"Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)
    
    print("\nStarting Training...")
    print("Policy controls: impedance parameters only (12D)")
    print("Trajectory: predefined waypoints for drawer opening")
    print("Focus: learn optimal impedance for different manipulation phases")
    
    try:
        runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
        print(f"\nTraining completed! Results saved to: {log_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
