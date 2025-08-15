#!/usr/bin/env python3

# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for fixed trajectory impedance control cabinet manipulation."""

import torch

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg

# Import our custom fixed trajectory action and config
from .fixed_traj_impedance_action import (
    FixedTrajImpedanceAction,
    FixedTrajImpedanceActionCfg,
)

# Import force impedance MDP functions
from .mdp import force_impedance_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class FixedTrajImpedanceEnvCfg(CabinetEnvCfg):
    """Configuration for fixed trajectory impedance control cabinet manipulation."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Set franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True
        
        # Add end-effector frame transformer
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
                ),
            ],
        )
        
        # Add contact sensor for force feedback
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
            update_period=0.0,
            history_length=3,
            track_pose=False,
        )

        # Configure Operational Space Controller
        controller_cfg = OperationalSpaceControllerCfg(
            # Remove command_types as it's not a valid parameter
            target_types=["pose_abs"],   # Only need pose_abs
            impedance_mode="variable",   # Enable variable impedance
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],  # Control all 6 DOF
            contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
            # Base impedance parameters - will be modulated by RL
            motion_stiffness_task=[200.0, 200.0, 200.0, 20.0, 20.0, 20.0],
            motion_damping_ratio_task=1.0,
            # Impedance limits
            motion_stiffness_limits_task=(50.0, 1000.0),
            motion_damping_ratio_limits_task=(0.2, 5.0),
            # Use position control in nullspace
            nullspace_control="position",
        )

        # Set Actions for fixed trajectory impedance control
        action_cfg = FixedTrajImpedanceActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            body_offset=FixedTrajImpedanceActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            controller_cfg=controller_cfg,
            stiffness_scale=0.03,  # Scaling for RL actions
            damping_ratio_scale=0.08,  # Scaling for RL actions
            class_type=FixedTrajImpedanceAction,
        )
        self.actions.arm_action = action_cfg
        
        # Binary gripper action
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Enhanced observations for impedance learning
        self.observations.policy.contact_forces = ObsTerm(
            func=force_impedance_mdp.contact_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )
        
        self.observations.policy.current_impedance = ObsTerm(
            func=force_impedance_mdp.current_impedance_params,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        self.observations.policy.desired_joint_pos = ObsTerm(
            func=lambda env, asset_cfg: env.action_manager._terms["arm_action"].desired_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        self.observations.policy.desired_joint_vel = ObsTerm(
            func=lambda env, asset_cfg: env.action_manager._terms["arm_action"].desired_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Impedance-focused rewards
        impedance_tracking_reward = RewTerm(
            func=self._impedance_tracking_reward,
            weight=2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )
        
        trajectory_following_reward = RewTerm(
            func=self._trajectory_following_reward,
            weight=3.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        # Add new rewards
        setattr(self.rewards, 'impedance_tracking', impedance_tracking_reward)
        setattr(self.rewards, 'trajectory_following', trajectory_following_reward)
        
        # Maintain core manipulation rewards but with lower weights
        self.rewards.approach_ee_handle.weight = 2.0
        self.rewards.approach_gripper_handle.weight = 5.0
        self.rewards.grasp_handle.weight = 3.0
        self.rewards.open_drawer_bonus.weight = 10.0

        # Fix missing parameters
        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger.*"]

    def _impedance_tracking_reward(self, env, sensor_cfg):
        """Reward for appropriate impedance based on contact forces."""
        import torch
        
        contact_sensor = env.scene[sensor_cfg.name]
        
        # Get contact forces
        if contact_sensor.data.net_forces_w is not None:
            force_magnitude = torch.norm(contact_sensor.data.net_forces_w, dim=-1).sum(dim=-1)
        else:
            force_magnitude = torch.zeros(env.num_envs, device=env.device)
        
        # Simple impedance adaptation logic
        # High forces → lower stiffness (compliant)
        # Low forces → higher stiffness (precise)
        
        # Reward for appropriate impedance selection
        # This is a simplified version - full implementation would be in MDP functions
        reward = torch.ones(env.num_envs, device=env.device) * 0.1
        
        return reward

    def _trajectory_following_reward(self, env, asset_cfg):
        """Reward for following the desired trajectory."""
        import torch
        
        # Get desired joint positions from action manager
        arm_action = env.action_manager._terms["arm_action"]
        desired_joint_pos = arm_action.desired_joint_pos
        actual_joint_pos = env.scene[asset_cfg.name].data.joint_pos
        
        # Compute tracking error
        tracking_error = torch.norm(desired_joint_pos - actual_joint_pos, dim=-1)
        
        # Convert to reward (lower error = higher reward)
        reward = torch.exp(-tracking_error)
        
        return reward


class FixedTrajImpedanceEnvImp:
    """Implementation class with additional methods for the environment."""
    
    def set_reference_trajectory(self, trajectory: torch.Tensor):
        """Set the reference trajectory to follow.
        
        Args:
            trajectory: Tensor of shape [num_envs, num_waypoints, 7] with waypoints for each environment.
        """
        if not hasattr(self.action_manager, "_terms") or "arm_action" not in self.action_manager._terms:
            print("Warning: arm_action not found in action manager")
            return
        
        arm_action = self.action_manager._terms["arm_action"]
        if not hasattr(arm_action, "set_trajectory"):
            print("Warning: set_trajectory method not found in arm_action")
            return
            
        arm_action.set_trajectory(trajectory)
        print(f"Set reference trajectory with {trajectory.shape[1]} waypoints.")


@configclass 
class FixedTrajImpedanceEnvCfg_PLAY(FixedTrajImpedanceEnvCfg):
    """Configuration for playing with trained fixed trajectory impedance policy."""
    
    def __post_init__(self):
        super().__post_init__()
        # Make it easier for play mode
        self.scene.num_envs = 1
        self.episode_length_s = 8.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
