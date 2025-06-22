# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Single arm manipulation tasks with variable impedance control.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Variable-Impedance-Cabinet-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.variable_impedance_cabinet_env_cfg:VariableImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VariableImpedanceCabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Variable-Impedance-Cabinet-Play-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.variable_impedance_cabinet_env_cfg:VariableImpedanceCabinetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VariableImpedanceCabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)
