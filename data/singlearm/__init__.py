# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Single arm manipulation tasks with variable impedance control.
"""

import gymnasium as gym
from . import agents  # Import the agents module for configs

# Register the environments
gym.register(
    id="Isaac-Variable-Impedance-Cabinet-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "data.singlearm.variable_impedance_cabinet_env_cfg:VariableImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": "data.singlearm.agents.rsl_rl_ppo_cfg:VariableImpedanceCabinetPPORunnerCfg",
    }
)

# Register play version
gym.register(
    id="Isaac-Variable-Impedance-Cabinet-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "data.singlearm.variable_impedance_cabinet_env_cfg:VariableImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": "data.singlearm.agents.rsl_rl_ppo_cfg:VariableImpedanceCabinetPPORunnerCfg",
    }
)

gym.register(
    id="Isaac-Variable-Impedance-Cabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "data.singlearm.variable_impedance_cabinet_env_cfg:VariableImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": "data.singlearm.agents.rsl_rl_ppo_cfg:VariableImpedanceCabinetPPORunnerCfg",
    }
)

# Register fixed trajectory impedance environments
gym.register(
    id="Isaac-Open-Drawer-Franka-Fixed-Impedance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "data.singlearm.fixed_traj_impedance_env_cfg:FixedTrajImpedanceEnvCfg",
        "rsl_rl_cfg_entry_point": "data.singlearm.agents.rsl_rl_ppo_cfg:FixedTrajImpedancePPORunnerCfg",
    }
)

# Register fixed trajectory play version  
gym.register(
    id="Isaac-Open-Drawer-Franka-Fixed-Impedance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, 
    kwargs={
        "env_cfg_entry_point": "data.singlearm.fixed_traj_impedance_env_cfg:FixedTrajImpedanceEnvCfg",
        "rsl_rl_cfg_entry_point": "data.singlearm.agents.rsl_rl_ppo_cfg:FixedTrajImpedancePPORunnerCfg",
    }
)
