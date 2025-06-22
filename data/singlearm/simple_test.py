#!/usr/bin/env python3

# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple test for variable impedance environment.
This script manually registers and creates the environment.

Run with: ./isaaclab.sh -p data/singlearm/simple_test.py
"""

"""Launch Isaac Sim Simulator first."""

import os
import sys
import argparse

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Simple test for variable impedance environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
# Note: Don't add --device as it's already handled by AppLauncher
# AppLauncher adds its own arguments including --headless and --device
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Set device after parsing to use AppLauncher's device parameter
device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

print("="*80)
print("VARIABLE IMPEDANCE ENVIRONMENT TEST")
print("="*80)

# Add the data directory to Python path
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)
    print(f"Added {data_dir} to sys.path")

# Import environment modules
try:
    print("Loading variable impedance modules...")
    from data.singlearm.variable_impedance_cabinet_env_cfg import VariableImpedanceCabinetEnvCfg
    from data.singlearm.variable_impedance_actions import VariableImpedanceAction, VariableImpedanceActionCfg
    import data.singlearm.agents.rsl_rl_ppo_cfg as agents_cfg
    print("✓ Successfully loaded modules")
except ImportError as e:
    print(f"✗ Error loading modules: {e}")
    import traceback
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

# Register the environment
env_name = "Isaac-Variable-Impedance-Cabinet-v0"

print(f"Registering '{env_name}'...")
gym.register(
    id=env_name,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "data.singlearm.variable_impedance_cabinet_env_cfg:VariableImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": "data.singlearm.agents.rsl_rl_ppo_cfg:VariableImpedanceCabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)
print(f"✓ Successfully registered '{env_name}'")

# Create and test the environment
try:
    print(f"\nCreating environment with {args_cli.num_envs} envs on {args_cli.device}...")
    env = gym.make(env_name, num_envs=args_cli.num_envs, device=args_cli.device)
    print("✓ Environment created successfully!")
    
    print(f"\nEnvironment Info:")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space: {env.action_space}")
    if hasattr(env.action_space, 'shape'):
        print(f"  Action Space Shape: {env.action_space.shape}")
    
    print("\nResetting environment...")
    obs, _ = env.reset(seed=args_cli.seed)
    print("✓ Reset successful")
    if isinstance(obs, dict) and 'policy' in obs:
        print(f"  Observation shape: {obs['policy'].shape}")
    
    print("\nTaking a random action...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("✓ Step successful")
    
    # Clean up
    env.close()
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()

# Close the simulation
simulation_app.close()
