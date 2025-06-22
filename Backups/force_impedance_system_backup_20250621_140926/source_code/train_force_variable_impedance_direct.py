#!/usr/bin/env python3
"""
Direct Force Variable Impedance Training Script

This script directly trains the force variable impedance cabinet manipulation task
using the validated configurations.

Usage:
    ./isaaclab.sh -p train_force_variable_impedance_direct.py
"""

import torch
import gymnasium as gym
import os
import argparse
from datetime import datetime

from isaaclab.app import AppLauncher

# Parse arguments and launch app
parser = argparse.ArgumentParser(description="Direct Force Variable Impedance Training")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=1200, help="Maximum training iterations")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--video", action="store_true", help="Record training videos")

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import isaaclab_tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml, dump_pickle

from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.force_variable_impedance_strict_env_cfg import (
    ForceVariableImpedanceStrictCabinetEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.agents.rsl_rl_ppo_cfg import (
    ForceVariableImpedanceCabinetPPORunnerCfg,
)


def main():
    """Train force variable impedance control."""
    
    print("🚀 Direct Force Variable Impedance Training")
    print("=" * 60)
    
    # Create configurations
    env_cfg = ForceVariableImpedanceCabinetEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    agent_cfg = ForceVariableImpedanceCabinetPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    
    print(f"Environment Configuration:")
    print(f"  • Number of environments: {env_cfg.scene.num_envs}")
    print(f"  • Device: {env_cfg.sim.device}")
    print(f"  • Episode length: {env_cfg.episode_length_s}s")
    
    print(f"\nAgent Configuration:")
    print(f"  • Algorithm: PPO")
    print(f"  • Max iterations: {agent_cfg.max_iterations}")
    print(f"  • Steps per env: {agent_cfg.num_steps_per_env}")
    print(f"  • Learning rate: {agent_cfg.algorithm.learning_rate}")
    # Handle different policy types safely
    try:
        if hasattr(agent_cfg.policy, 'actor_hidden_dims'):
            print(f"  • Network: {agent_cfg.policy.actor_hidden_dims}")
        else:
            print(f"  • Policy type: {type(agent_cfg.policy).__name__}")
    except:
        print(f"  • Policy configuration available")
    
    # Create logging directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"\nLogging to: {log_root_path}")
    
    # Create timestamped log directory
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    print("\n" + "=" * 60)
    print("Creating Environment...")
    print("=" * 60)
    
    env = gym.make("Isaac-Open-Drawer-Franka-v0", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap environment for RSL-RL
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    print(f"✅ Environment created successfully")
    print(f"   Action space: {wrapped_env.action_space}")
    print(f"   Observation space: {wrapped_env.observation_space}")
    
    # Create training runner
    print("\n" + "=" * 60)
    print("Creating Training Runner...")
    print("=" * 60)
    
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # Save configurations
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    print(f"✅ Training runner created successfully")
    print(f"   Device: {runner.device}")
    print(f"   Log directory: {log_dir}")
    
    # Load checkpoint if provided
    if args_cli.checkpoint:
        print(f"\n📁 Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)
    
    # Start training
    print("\n" + "=" * 60)
    print("🚀 STARTING FORCE VARIABLE IMPEDANCE TRAINING")
    print("=" * 60)
    print("Key Features:")
    print("  • 19D Action Space: 6D pose + 6D stiffness + 6D damping + 1D gripper")
    print("  • Force Feedback: Contact forces from panda_hand")
    print("  • Variable Impedance: Real-time stiffness and damping adaptation")
    print("  • Curriculum Learning: 4-phase progressive training")
    print("=" * 60)
    
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Results saved to: {log_dir}")
        print("\n🚀 Next steps:")
        print("  1. Evaluate the trained policy")
        print("  2. Compare with baseline joint position control")
        print("  3. Analyze force adaptation behavior")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        print(f"💾 Partial results saved to: {log_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        wrapped_env.close()
    
    return True


if __name__ == "__main__":
    success = main()
    simulation_app.close()
    exit(0 if success else 1)
