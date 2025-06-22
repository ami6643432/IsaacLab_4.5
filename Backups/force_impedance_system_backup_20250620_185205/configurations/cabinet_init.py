# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manipulation environments to open drawers in a cabinet."""

import gymnasium as gym

from . import mdp
from .cabinet_env_cfg import CabinetEnvCfg, CabinetEnvCfg_PLAY
from .config.franka import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Open-Drawer-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cabinet_env_cfg:CabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cabinet_env_cfg:CabinetEnvCfg_PLAY",
    },
)

# Register force-based variable impedance environment
from .config.franka.force_variable_impedance_env_cfg import (
    ForceVariableImpedanceCabinetEnvCfg,
    ForceVariableImpedanceCabinetEnvCfg_PLAY,
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Force-Variable-Impedance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.force_variable_impedance_env_cfg:ForceVariableImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForceVariableImpedanceCabinetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Force-Variable-Impedance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.force_variable_impedance_env_cfg:ForceVariableImpedanceCabinetEnvCfg_PLAY",
    },
)

# Register auxiliary impedance environment (joint position + force-based impedance adaptation)
from .config.franka.auxiliary_impedance_env_cfg import (
    AuxiliaryImpedanceCabinetEnvCfg,
    AuxiliaryImpedanceCabinetEnvCfg_PLAY,
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Auxiliary-Impedance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.auxiliary_impedance_env_cfg:AuxiliaryImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Auxiliary-Impedance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.auxiliary_impedance_env_cfg:AuxiliaryImpedanceCabinetEnvCfg_PLAY",
    },
)

# Register dual network impedance environment (using fixed version)
from .config.franka.dual_network_impedance_env_cfg_fixed import (
    DualNetworkImpedanceCabinetEnvCfg,
    DualNetworkImpedanceCabinetEnvCfg_PLAY,
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Dual-Network-Impedance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.dual_network_impedance_env_cfg_fixed:DualNetworkImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Dual-Network-Impedance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.dual_network_impedance_env_cfg_fixed:DualNetworkImpedanceCabinetEnvCfg_PLAY",
    },
)
