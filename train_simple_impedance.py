#!/usr/bin/env python3

"""Simple fixed trajectory impedance training script."""

import argparse
import gymnasium as gym
import torch
from datetime import datetime
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train fixed trajectory impedance control.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10, help="RL Policy training iterations.")

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
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetEnvCfg
from isaaclab_tasks.direct.franka_cabinet.agents.rsl_rl_ppo_cfg import FrankaCabinetPPORunnerCfg


def create_trajectory_points(num_envs: int, device: str) -> torch.Tensor:
    """Create predefined trajectory points for drawer opening task."""
    waypoints = [
        [0.45, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],    # Approach handle
        [0.35, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],    # At handle
        [0.25, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],    # Pull drawer
        [0.15, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],    # Fully open
    ]
    
    trajectory = torch.tensor(waypoints, dtype=torch.float32, device=device)
    trajectory = trajectory.unsqueeze(0).repeat(num_envs, 1, 1)
    
    return trajectory


def main():
    """Main training function."""
    
    print("=" * 80)
    print("SIMPLE IMPEDANCE CONTROL TRAINING")
    print("=" * 80)
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Device: {args_cli.device}")
    print(f"Max iterations: {args_cli.max_iterations}")
    print("=" * 80)
    
    # Create environment configuration
    env_cfg = FrankaCabinetEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    print("Creating environment...")
    env = gym.make("Isaac-Open-Drawer-Franka-v0", cfg=env_cfg)
    
    # Generate trajectory points
    print("Generating trajectory points...")
    trajectory = create_trajectory_points(args_cli.num_envs, args_cli.device)
    print(f"Generated trajectory with shape: {trajectory.shape}")
    
    # Print environment information
    print(f"Environment action space: {env.action_space}")
    print(f"Environment observation space: {env.observation_space}")
    
    # Wrap environment for RSL-RL
    vec_env = RslRlVecEnvWrapper(env)

    print("Creating OnPolicyRunner...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"simple_impedance_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
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
        "experiment_name": "simple_impedance",
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
        env=vec_env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=args_cli.device,
    )
    
    # Train the agent
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    try:
        ppo_runner.learn(num_learning_iterations=args_cli.max_iterations)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training error: {str(e)}")
    finally:
        # Clean up
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
