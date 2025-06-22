#!/usr/bin/env python3

# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for strict contact force penalties approach.

This script trains an RL agent to control both pose and impedance parameters
while enforcing strict penalties for unwanted contact forces.
"""

import argparse
import gymnasium as gym
import torch
from datetime import datetime
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train strict contact force penalties impedance control.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the simulation on.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=2000, help="RL Policy training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument("--phase", type=str, default="full", choices=["phase1", "phase2", "phase3", "full"],
                    help="Training phase: phase1=approach+grasp, phase2=+contact_awareness, phase3=+impedance, full=all")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab_tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import configurations from backup files
from Backups.strict_contact_steps_1_2_backup_20250621_135409.force_variable_impedance_env_cfg import (
    ForceVariableImpedanceCabinetEnvCfg
)


def configure_training_phase(env_cfg, phase: str):
    """Configure environment for specific training phase."""
    
    if phase == "phase1":
        # Phase 1: Focus on approach and grasping only
        print("Phase 1: Basic approach and grasping")
        
        # Boost basic manipulation rewards
        env_cfg.rewards.approach_ee_handle.weight = 10.0
        env_cfg.rewards.approach_gripper_handle.weight = 25.0
        env_cfg.rewards.grasp_handle.weight = 15.0
        env_cfg.rewards.open_drawer_bonus.weight = 30.0
        
        # Disable contact penalties initially
        if hasattr(env_cfg.rewards, 'simple_contact_penalty'):
            env_cfg.rewards.simple_contact_penalty.weight = 0.0
        if hasattr(env_cfg.rewards, 'approach_contact_penalty'):
            env_cfg.rewards.approach_contact_penalty.weight = 0.0
            
    elif phase == "phase2":
        # Phase 2: Introduce contact awareness
        print("Phase 2: Basic manipulation + contact awareness")
        
        # Maintain manipulation rewards
        env_cfg.rewards.approach_ee_handle.weight = 8.0
        env_cfg.rewards.approach_gripper_handle.weight = 20.0
        env_cfg.rewards.grasp_handle.weight = 12.0
        env_cfg.rewards.open_drawer_bonus.weight = 25.0
        
        # Enable basic contact penalties
        if hasattr(env_cfg.rewards, 'simple_contact_penalty'):
            env_cfg.rewards.simple_contact_penalty.weight = -5.0
        if hasattr(env_cfg.rewards, 'approach_contact_penalty'):
            env_cfg.rewards.approach_contact_penalty.weight = -15.0
            
    elif phase == "phase3":
        # Phase 3: Full impedance optimization
        print("Phase 3: Full force-impedance control")
        
        # Balanced rewards
        env_cfg.rewards.approach_ee_handle.weight = 6.0
        env_cfg.rewards.approach_gripper_handle.weight = 15.0
        env_cfg.rewards.grasp_handle.weight = 10.0
        env_cfg.rewards.open_drawer_bonus.weight = 20.0
        
        # Full contact penalties
        if hasattr(env_cfg.rewards, 'simple_contact_penalty'):
            env_cfg.rewards.simple_contact_penalty.weight = -8.0
        if hasattr(env_cfg.rewards, 'approach_contact_penalty'):
            env_cfg.rewards.approach_contact_penalty.weight = -25.0
        if hasattr(env_cfg.rewards, 'grasp_contact_penalty'):
            env_cfg.rewards.grasp_contact_penalty.weight = -15.0
        if hasattr(env_cfg.rewards, 'manipulation_contact_penalty'):
            env_cfg.rewards.manipulation_contact_penalty.weight = -8.0
            
    else:  # full
        # Full training with all systems active
        print("Full training: All reward systems active")
        # Use default weights from configuration


def main():
    """Main training function."""
    
    print("=" * 80)
    print("STRICT CONTACT FORCE PENALTIES TRAINING")
    print("=" * 80)
    print(f"Training phase: {args_cli.phase}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Device: {args_cli.device}")
    print(f"Max iterations: {args_cli.max_iterations}")
    print("=" * 80)
    
    # Create environment configuration
    env_cfg = ForceVariableImpedanceCabinetEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # Configure for specific training phase
    configure_training_phase(env_cfg, args_cli.phase)
    
    # Create agent configuration
    from Backups.force_impedance_system_backup_20250621_140926.configurations.rsl_rl_ppo_cfg import (
        ForceVariableImpedanceCabinetPPORunnerCfg
    )
    
    agent_cfg = ForceVariableImpedanceCabinetPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = args_cli.device
    
    # Adjust training parameters based on phase
    if args_cli.phase == "phase1":
        agent_cfg.algorithm.learning_rate = 5e-4  # Higher LR for faster basic learning
        agent_cfg.algorithm.entropy_coef = 0.02   # More exploration
    elif args_cli.phase == "phase2":
        agent_cfg.algorithm.learning_rate = 3e-4  # Standard LR
        agent_cfg.algorithm.entropy_coef = 0.01   # Moderate exploration
    else:
        agent_cfg.algorithm.learning_rate = 2e-4  # Lower LR for fine-tuning
        agent_cfg.algorithm.entropy_coef = 0.005  # Less exploration
    
    # Create environment
    env = gym.make("Isaac-Open-Drawer-Franka-Force-Variable-Impedance-v0", cfg=env_cfg)
    
    # Wrap environment for RSL-RL
    wrapped_env = RslRlVecEnvWrapper(env)
    
    print("\nEnvironment Details:")
    print(f"Action space: {wrapped_env.action_space}")
    print(f"Expected: 19D (6 pose + 6 stiffness + 6 damping + 1 gripper)")
    print(f"Observation space: {wrapped_env.observation_space}")
    
    # Create log directory
    log_dir = os.path.join("logs", "strict_contact_penalties", args_cli.phase, 
                          datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # Create runner
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # Load checkpoint if provided
    if args_cli.checkpoint:
        print(f"Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)
    
    print(f"\nStarting {args_cli.phase} Training...")
    print("Policy controls: pose + impedance parameters (19D)")
    print("Contact sensors: panda_hand with force feedback")
    print("Key features:")
    
    if args_cli.phase == "phase1":
        print("  - High rewards for approach and grasping")
        print("  - No contact penalties (learning basic manipulation)")
    elif args_cli.phase == "phase2":
        print("  - Maintained manipulation rewards")
        print("  - Basic contact force penalties")
    elif args_cli.phase == "phase3":
        print("  - Full phase-aware contact penalties")
        print("  - Impedance adaptation rewards")
    else:
        print("  - All reward systems active")
        print("  - Progressive curriculum learning")
        print("  - Phase-aware contact penalties")
    
    try:
        runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
        print(f"\nTraining completed! Results saved to: {log_dir}")
        
        # Print next steps
        if args_cli.phase == "phase1":
            print("\nNext step: Train phase2 using this checkpoint")
            print(f"Command: python train_strict_contact_force_penalties.py --phase phase2 --checkpoint {log_dir}/model_*.pt")
        elif args_cli.phase == "phase2":
            print("\nNext step: Train phase3 using this checkpoint")
            print(f"Command: python train_strict_contact_force_penalties.py --phase phase3 --checkpoint {log_dir}/model_*.pt")
        
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
