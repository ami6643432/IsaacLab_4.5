#!/usr/bin/env python3
# Copyright (c) 2025, Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Action class for fixed trajectory impedance control."""

from __future__ import annotations

import torch
from typing import Sequence, TYPE_CHECKING, Dict

import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class FixedTrajImpedanceActionCfg(OperationalSpaceControllerActionCfg):
    """Configuration for :class:`FixedTrajImpedanceAction`."""

    class_type: type["FixedTrajImpedanceAction"] | None = None  # assigned after class definition
    
    # Scaling factors for impedance parameters
    stiffness_scale: float = 1.0
    """Scale factor for stiffness parameters."""
    
    damping_ratio_scale: float = 1.0
    """Scale factor for damping ratio parameters."""


class FixedTrajImpedanceAction(OperationalSpaceControllerAction):
    """Operational space action that tracks a fixed Cartesian trajectory."""

    cfg: FixedTrajImpedanceActionCfg

    def __init__(self, cfg: FixedTrajImpedanceActionCfg, env: ManagerBasedEnv):
        # Make sure command_types and target_types are set correctly for variable impedance
        cfg.controller_cfg.target_types = ["pose_abs"]
        cfg.controller_cfg.impedance_mode = "variable"
        
        super().__init__(cfg, env)
        
        # Initialize trajectory tracking
        self._trajectory = None
        self._trajectory_timer = torch.zeros(self.num_envs, device=self.device)
        self._current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Initialize pose and command tensors
        self._desired_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self._osc_command = torch.zeros(self.num_envs, 19, device=self.device)  # 7 pose + 6 stiffness + 6 damping
        
        # Command indices for variable impedance control
        self._pose_abs_idx = slice(0, 7)  # Base pose command (7D)
        self._stiffness_idx = slice(7, 13)  # Stiffness parameters (6D)
        self._damping_ratio_idx = slice(13, 19)  # Damping parameters (6D)
        
        # Scaling factors for impedance parameters
        self._stiffness_scale = cfg.stiffness_scale if hasattr(cfg, 'stiffness_scale') else 1.0
        self._damping_ratio_scale = cfg.damping_ratio_scale if hasattr(cfg, 'damping_ratio_scale') else 1.0
        
        # Initialize processed actions tensor first
        self._processed_actions = torch.zeros(self.num_envs, 19, device=self.device)

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        # Only stiffness and damping are controlled by RL
        # 6 (stiffness) + 6 (damping) = 12
        return 12  # Pose is handled by trajectory

    # ---------------------------------------------------------------------
    # Operations
    # ---------------------------------------------------------------------
    def set_desired_pose(self, pose: torch.Tensor):
        """Set the desired end-effector pose for the next control step."""
        if pose.shape != (self.num_envs, 7):
            raise ValueError(
                f"Expected pose shape ({self.num_envs},7) got {pose.shape}"
            )
        self._desired_pose[:] = pose
    
    def set_trajectory(self, trajectory: torch.Tensor):
        """Set the trajectory to follow."""
        # trajectory shape: [num_envs, num_waypoints, 7]
        self._trajectory = trajectory
        self._current_waypoint_idx.zero_()
        self._trajectory_timer.zero_()
        
        # Set initial waypoint
        if self._trajectory is not None:
            self._desired_pose[:] = self._trajectory[:, 0, :]
    
    def update_trajectory_position(self, dt: float):
        """Update trajectory position based on time and progress."""
        if self._trajectory is None:
            return
            
        # Update timer
        self._trajectory_timer += dt
        
        # Check for waypoint transition (every 2 seconds)
        transition_time = 2.0
        waypoints_to_advance = (self._trajectory_timer / transition_time).floor().long()
        
        # Check if we need to advance waypoints
        needs_update = waypoints_to_advance > 0
        if not torch.any(needs_update):
            return
            
        # Get max valid waypoint index
        max_waypoint = self._trajectory.shape[1] - 1
        
        # Update waypoint indices and timer for envs that need update
        env_ids = torch.where(needs_update)[0]
        for env_id in env_ids:
            # Compute how many waypoints to advance
            advance_by = min(waypoints_to_advance[env_id].item(), 
                             max_waypoint - self._current_waypoint_idx[env_id].item())
            
            if advance_by > 0:
                # Advance waypoint and reset timer
                self._current_waypoint_idx[env_id] += advance_by
                self._trajectory_timer[env_id] = 0.0
                
                # Update desired pose for this env
                waypoint_idx = min(int(self._current_waypoint_idx[env_id].item()), int(max_waypoint))
                self._desired_pose[env_id, :] = self._trajectory[env_id, waypoint_idx, :]

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions into OSC commands.
        
        Args:
            actions: Raw RL actions [batch_size, 12] containing:
                    - stiffness parameters (6D)
                    - damping parameters (6D)
        """
        # Update trajectory position if available
        if self._trajectory is not None:
            self.update_trajectory_position(self._sim_dt)
        
        # Update ee pose for IK computation
        self._compute_ee_pose()
        self._compute_task_frame_pose()

        # Store raw actions (12D: stiffness + damping)
        self._raw_actions[:] = actions
        
        # Extract and scale impedance parameters from raw actions
        stiffness = actions[:, :6] * self._stiffness_scale
        damping = actions[:, 6:] * self._damping_ratio_scale
        
        # Clamp to controller limits
        stiffness = torch.clamp(
            stiffness,
            min=self.cfg.controller_cfg.motion_stiffness_limits_task[0],
            max=self.cfg.controller_cfg.motion_stiffness_limits_task[1],
        )
        damping = torch.clamp(
            damping,
            min=self.cfg.controller_cfg.motion_damping_ratio_limits_task[0],
            max=self.cfg.controller_cfg.motion_damping_ratio_limits_task[1],
        )

        # Store current impedance for observations
        self._current_stiffness = stiffness
        self._current_damping = damping

        # Build full OSC command: [pose(7) + stiffness(6) + damping(6)]
        self._osc_command[:, :7] = self._desired_pose  # 7D pose from trajectory
        self._osc_command[:, 7:13] = stiffness  # 6D stiffness from RL
        self._osc_command[:, 13:19] = damping  # 6D damping from RL
        
        # Set processed actions for OSC controller
        self._processed_actions[:] = self._osc_command

        # Compute desired joint positions for observations
        if self._diff_ik is not None:
            self._diff_ik.set_command(self._desired_pose)
            des_q = self._diff_ik.compute(
                ee_pos=self._ee_pose_b[:, :3],
                ee_quat=self._ee_pose_b[:, 3:7],
                jacobian=self._jacobian_b,
                joint_pos=self._joint_pos,
            )
            self.desired_joint_pos[:] = des_q
            self.desired_joint_vel.zero_()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the controller state."""
        super().reset(env_ids)
        
        if env_ids is None:
            # Reset all environments
            self._current_waypoint_idx.zero_()
            self._trajectory_timer.zero_()
            self.desired_joint_pos.zero_()
            self.desired_joint_vel.zero_()
            
            # Reset to initial waypoint if trajectory exists
            if self._trajectory is not None:
                self._desired_pose[:] = self._trajectory[:, 0, :]
        else:
            # Reset specific environments
            for env_id in env_ids:
                self._current_waypoint_idx[env_id] = 0
                self._trajectory_timer[env_id] = 0.0
            
            # Reset joint states for these environments
            self.desired_joint_pos[env_ids] = 0.0
            self.desired_joint_vel[env_ids] = 0.0
            
            # Reset to initial waypoint if trajectory exists
            if self._trajectory is not None:
                for env_id in env_ids:
                    self._desired_pose[env_id] = self._trajectory[env_id, 0, :]

    def get_current_impedance_params(self) -> Dict[str, torch.Tensor]:
        """Get current impedance parameters for RL observations."""
        if hasattr(self, '_current_stiffness') and hasattr(self, '_current_damping'):
            return {
                "stiffness": self._current_stiffness.clone(),
                "damping": self._current_damping.clone()
            }
        
        # Fallback to defaults during initialization
        default_stiffness = torch.tensor(
            [200.0, 200.0, 200.0, 20.0, 20.0, 20.0], 
            device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        
        default_damping = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
            device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        
        return {"stiffness": default_stiffness, "damping": default_damping}


# Set the class type directly - no circular import issues now
from typing import TYPE_CHECKING, cast
if not TYPE_CHECKING:
    FixedTrajImpedanceActionCfg.class_type = FixedTrajImpedanceAction
