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
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10, help="RL Policy training iterations.")
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
    
    # Additional required environment configuration
    env_cfg.scene.robot.init_state.default_joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }
    
    # Create agent configuration
    agent_cfg = FixedTrajImpedancePPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = args_cli.device
    
    # Create environment
    print("Creating environment...")
    try:
        env = gym.make("Isaac-Open-Drawer-Franka-Fixed-Impedance-v0", cfg=env_cfg)
        print("✓ Custom Fixed Trajectory Impedance environment created successfully!")
    except Exception as e:
        print(f"Error creating custom environment: {str(e)}")
        # Fallback to standard environment with custom configuration
        print("Falling back to standard environment with fixed trajectory config...")
        from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetEnvCfg
        base_env_cfg = FrankaCabinetEnvCfg()
        base_env_cfg.scene.num_envs = args_cli.num_envs
        base_env_cfg.sim.device = args_cli.device
        env = gym.make("Isaac-Open-Drawer-Franka-v0", cfg=base_env_cfg)
    
    # Generate trajectory points
    print("Generating trajectory points...")
    trajectory = create_trajectory_points(args_cli.num_envs, args_cli.device)
    
    # Log trajectory information
    print(f"Generated trajectory with shape: {trajectory.shape}")
    
    # Set trajectory in the environment action
    print("Setting up trajectory in arm action...")
    if hasattr(env.unwrapped, "action_manager") and hasattr(env.unwrapped.action_manager, "_terms"):
        if "arm_action" in env.unwrapped.action_manager._terms:
            arm_action = env.unwrapped.action_manager._terms["arm_action"]
            if hasattr(arm_action, "set_trajectory"):
                print("Setting trajectory in arm action...")
                arm_action.set_trajectory(trajectory)
                print("✓ Trajectory successfully set.")
            else:
                print("⚠ Warning: arm_action does not have set_trajectory method")
        else:
            print("⚠ Warning: arm_action not found in action manager terms")
    else:
        print("⚠ Warning: action_manager or terms not found in environment")
    
    # Print environment information
    print(f"Environment action space: {env.action_space}")
    print(f"Environment observation space: {env.observation_space}")
    
    # Test environment for force and impedance information
    print("\nTesting environment for force feedback...")
    obs, _ = env.reset()
    if isinstance(obs, dict) and "policy" in obs:
        obs_tensor = obs["policy"]
        print(f"Policy observation shape: {obs_tensor.shape}")
        
        # Check observation manager terms
        if hasattr(env.unwrapped, "observation_manager"):
            print("\nAvailable observation terms:")
            for term_name, term in env.unwrapped.observation_manager._terms.items():
                try:
                    term_data = term(env.unwrapped)
                    if hasattr(term_data, 'shape'):
                        print(f"  • {term_name}: shape {term_data.shape}")
                    else:
                        print(f"  • {term_name}: {type(term_data)}")
                except Exception as e:
                    print(f"  • {term_name}: Error getting data - {str(e)}")
            
            # Check specifically for force-related observations
            force_terms = [name for name in env.unwrapped.observation_manager._terms.keys() 
                          if 'force' in name.lower() or 'contact' in name.lower()]
            if force_terms:
                print(f"\n✓ Force-related observations found: {force_terms}")
            else:
                print("\n⚠ No force-related observations found")
                
            # Check for impedance-related observations  
            impedance_terms = [name for name in env.unwrapped.observation_manager._terms.keys() 
                              if 'impedance' in name.lower() or 'stiffness' in name.lower() or 'damping' in name.lower()]
            if impedance_terms:
                print(f"✓ Impedance-related observations found: {impedance_terms}")
            else:
                print("⚠ No impedance-related observations found")
    
    # Wrap environment for RSL-RL
    vec_env = RslRlVecEnvWrapper(env)

    # Create runner
    print("\nCreating OnPolicyRunner...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"fixed_traj_impedance_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training configuration
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 0.5,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "init_noise_std": 1.0,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256, 256],
            "critic_hidden_dims": [256, 256, 256],
            "activation": 'elu',
        },
        "max_iterations": args_cli.max_iterations,
        "empirical_normalization": False,
        "device": args_cli.device,
        "num_steps_per_env": 24,
        "save_interval": 50,
        "experiment_name": "fixed_traj_impedance",
        "run_name": "",
        "logger": "tensorboard",
        "neptune_project": "isaaclab",
        "wandb_project": "isaaclab",
        "resume": False,
        "load_run": -1,
        "checkpoint": -1,
    }
    
    # Create runner
    ppo_runner = OnPolicyRunner(
        vec_env=vec_env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=args_cli.device,
    )
    
    # Load from checkpoint if provided
    if args_cli.checkpoint is not None and os.path.exists(args_cli.checkpoint):
        print(f"Loading checkpoint from {args_cli.checkpoint}")
        ppo_runner.load(args_cli.checkpoint)
    
    # Train the agent
    print("=" * 80)
    print("Starting training...")
    print("Key Features:")
    print("  • 12D Action Space: 6D stiffness + 6D damping parameters")
    print("  • Fixed Trajectory: 4 predefined waypoints for drawer opening")
    print("  • Force Feedback: Contact forces used for impedance adaptation")
    print("  • Variable Impedance: Real-time stiffness and damping control")
    print("=" * 80)
    
    try:
        ppo_runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
        print("\n🎉 Training completed successfully!")
        print(f"📊 Results saved to: {log_dir}")
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final model
        try:
            final_model_path = os.path.join(log_dir, "model_final.pt")
            ppo_runner.save(final_model_path)
            print(f"Final model saved to {final_model_path}")
        except:
            print("Could not save final model")
        
        # Clean up
        env.close()


# Run main function if script is executed directly
if __name__ == "__main__":
    main()
    simulation_app.close()
