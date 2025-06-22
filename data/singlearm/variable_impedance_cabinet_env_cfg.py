# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Variable impedance cabinet manipulation environment configuration for Franka robot.

This environment uses an RL agent to generate variable impedance parameters
that are fed to a lower-level operational space controller for drawer opening tasks.
"""

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.cabinet.mdp as cabinet_mdp
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg

# Import our custom variable impedance action
from .variable_impedance_actions import VariableImpedanceAction, VariableImpedanceActionCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class VariableImpedanceCabinetEnvCfg(CabinetEnvCfg):
    """Configuration for variable impedance cabinet manipulation task."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set franka as robot - copy configuration and modify prim_path
        from copy import deepcopy
        self.scene.robot = deepcopy(FRANKA_PANDA_CFG)
        self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        
        # Set stiffness and damping to zero for effort control mode
        # This allows the OSC to have full control over the impedance
        self.scene.robot.actuators["panda_shoulder"].stiffness = 0.0
        self.scene.robot.actuators["panda_shoulder"].damping = 0.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 0.0
        self.scene.robot.actuators["panda_forearm"].damping = 0.0

        # Set Actions for variable impedance operational space control
        # Use our custom variable impedance action
        self.actions.arm_action = VariableImpedanceActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            # Use basic OSC for now since we're extending it with variable impedance
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],  # Relative pose control 
                impedance_mode="fixed",  # We'll override this in our action
                motion_control_axes_task=[1, 1, 1, 1, 1, 1],  # Control all 6 DOF
                contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],  # No force control initially
                motion_damping_ratio_task=1.0,  # Default damping ratio
                motion_stiffness_task=[1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0],  # Default stiffness
            ),
            # Scale factors for different command components
            position_scale=0.1,  # Smaller position commands for safety
            orientation_scale=0.1,  # Smaller orientation commands
        )
        
        # Keep binary gripper action
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Update frame transformer for end-effector tracking
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,  # Enable visualization for debugging
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.107),  # Tool center point offset
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger", 
                    name="tool_rightfinger",
                ),
            ],
        )

        # Enhanced observations for impedance learning
        # Note: Additional observations would need to be added by extending the PolicyCfg class
        # For now, we'll use the standard cabinet observations

        # Reduce action penalty weight since we have more actions now
        if hasattr(self.rewards, 'action_rate_l2'):
            self.rewards.action_rate_l2.weight = -0.001  # Reduced from default

    def _get_current_impedance_params(self, env, asset_cfg):
        """Get current stiffness and damping parameters from the OSC controller."""
        import torch
        
        # Get the robot asset
        robot = env.scene[asset_cfg["asset_name"]]
        
        # Try to get impedance parameters from the action manager
        if hasattr(env.action_manager, "_terms") and "arm_action" in env.action_manager._terms:
            arm_action = env.action_manager._terms["arm_action"]
            if hasattr(arm_action, "get_current_impedance_params"):
                impedance_data = arm_action.get_current_impedance_params()
                # Concatenate stiffness and damping
                return torch.cat([impedance_data["stiffness"], impedance_data["damping"]], dim=-1)
        
        # Fallback: return default impedance parameters
        num_envs = robot.data.root_pos_w.shape[0]
        default_stiffness = torch.tensor([1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0], device=env.device)
        default_damping = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=env.device)
        stiffness_batch = default_stiffness.unsqueeze(0).repeat(num_envs, 1)
        damping_batch = default_damping.unsqueeze(0).repeat(num_envs, 1)
        return torch.cat([stiffness_batch, damping_batch], dim=-1)


@configclass 
class VariableImpedanceCabinetEnvCfg_PLAY(VariableImpedanceCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
