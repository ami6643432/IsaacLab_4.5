# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom action terms for variable impedance control.

This module implements action terms that allow an RL agent to generate
impedance parameters (stiffness and damping) that are fed to a lower-level
operational space controller.
"""

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg


@configclass
class VariableImpedanceActionCfg(OperationalSpaceControllerActionCfg):
    """Configuration for variable impedance action term.
    
    This extends the standard OSC action to separate the RL agent's policy
    from the lower-level impedance parameters.
    """
    
    # Task-space policy action dimensions  
    use_delta_pose: bool = True
    """Whether to use delta pose commands (relative to current pose)."""
    
    # Impedance adaptation parameters
    impedance_adaptation_rate: float = 0.1
    """Rate at which impedance parameters are adapted (0-1)."""
    
    contact_threshold: float = 1.0
    """Contact force threshold for impedance adaptation."""
    
    # Phase-based impedance profiles
    free_motion_stiffness: list[float] = [1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0]
    """Default stiffness during free motion phase."""
    
    contact_stiffness: list[float] = [300.0, 300.0, 1000.0, 50.0, 50.0, 50.0]
    """Default stiffness during contact phase."""
    
    free_motion_damping: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    """Default damping ratio during free motion."""
    
    contact_damping: list[float] = [0.7, 0.7, 1.5, 0.7, 0.7, 0.7]
    """Default damping ratio during contact."""


class VariableImpedanceAction(OperationalSpaceControllerAction):
    """Variable impedance action term for RL-based impedance parameter generation.
    
    This action term allows an RL agent to generate high-level task commands
    and impedance parameters that are fed to a lower-level operational space controller.
    
    Action Space:
    - 6 DOF: Desired pose delta (relative to current pose)
    - 6 DOF: Stiffness parameters (0-1 range, scaled to actual values)
    - 6 DOF: Damping ratio parameters (0-1 range, scaled to actual values)
    - 1 DOF: Gripper command
    
    Total: 19 dimensional action space
    """

    cfg: VariableImpedanceActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: VariableImpedanceActionCfg, env: "ManagerBasedEnv"):
        # Initialize parent OSC action
        super().__init__(cfg, env)
        
        # Store previous impedance parameters for smoothing
        self._prev_stiffness = torch.zeros(
            (self.num_envs, 6), device=self.device, dtype=torch.float32
        )
        self._prev_damping = torch.zeros(
            (self.num_envs, 6), device=self.device, dtype=torch.float32
        )
        
        # Initialize with default parameters
        self._prev_stiffness[:] = torch.tensor(
            self.cfg.free_motion_stiffness, device=self.device
        )
        self._prev_damping[:] = torch.tensor(
            self.cfg.free_motion_damping, device=self.device
        )
        
        # Contact detection state
        self._in_contact = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        # 6 (pose delta) + 6 (stiffness) + 6 (damping) + 1 (gripper) = 19
        return 19

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions from RL agent.
        
        Args:
            actions: Raw actions from RL agent [batch_size, 19]
                     [pose_delta(6), stiffness(6), damping(6), gripper(1)]
        """
        # Split actions into components
        pose_delta = actions[:, :6]  # Delta pose commands
        stiffness_raw = actions[:, 6:12]  # Raw stiffness (0-1)
        damping_raw = actions[:, 12:18]  # Raw damping (0-1)
        gripper_cmd = actions[:, 18:19]  # Gripper command
        
        # Scale and smooth impedance parameters
        stiffness_scaled = self._scale_stiffness_parameters(stiffness_raw)
        damping_scaled = self._scale_damping_parameters(damping_raw)
        
        # Apply impedance smoothing
        smooth_rate = self.cfg.impedance_adaptation_rate
        self._prev_stiffness = (1 - smooth_rate) * self._prev_stiffness + smooth_rate * stiffness_scaled
        self._prev_damping = (1 - smooth_rate) * self._prev_damping + smooth_rate * damping_scaled
        
        # Get current end-effector pose for relative commands
        current_ee_pose = self._compute_ee_pose()
        
        # Convert delta pose to absolute pose commands for OSC
        if self.cfg.use_delta_pose and current_ee_pose is not None:
            # Scale pose deltas
            pose_delta[:, :3] *= self.cfg.position_scale  # Position deltas
            pose_delta[:, 3:6] *= self.cfg.orientation_scale  # Orientation deltas (axis-angle)
            
            # Apply delta to current pose
            target_pose = self._apply_pose_delta(current_ee_pose, pose_delta)
        else:
            target_pose = pose_delta
        
        # Construct full OSC command: [pose(7), stiffness(6), damping(6)]
        osc_command = torch.zeros((self.num_envs, 19), device=self.device)
        osc_command[:, :7] = target_pose  # Target pose (pos + quat)
        osc_command[:, 7:13] = self._prev_stiffness  # Stiffness parameters
        osc_command[:, 13:19] = self._prev_damping  # Damping parameters
        
        # Store processed actions
        self._raw_actions = actions
        self._processed_actions = osc_command
        
        # Update contact state for next iteration
        self._update_contact_state()

    def _scale_stiffness_parameters(self, raw_stiffness: torch.Tensor) -> torch.Tensor:
        """Scale raw stiffness actions (0-1) to actual stiffness values."""
        # Get min/max from controller config
        min_stiff = self.cfg.controller_cfg.motion_stiffness_limits_task[0]
        max_stiff = self.cfg.controller_cfg.motion_stiffness_limits_task[1]
        
        # Apply sigmoid to ensure (0,1) range, then scale
        stiffness_norm = torch.sigmoid(raw_stiffness)
        stiffness_scaled = min_stiff + stiffness_norm * (max_stiff - min_stiff)
        
        return stiffness_scaled
    
    def _scale_damping_parameters(self, raw_damping: torch.Tensor) -> torch.Tensor:
        """Scale raw damping actions (0-1) to actual damping ratio values."""
        # Get min/max from controller config
        min_damp = self.cfg.controller_cfg.motion_damping_ratio_limits_task[0]
        max_damp = self.cfg.controller_cfg.motion_damping_ratio_limits_task[1]
        
        # Apply sigmoid to ensure (0,1) range, then scale
        damping_norm = torch.sigmoid(raw_damping)
        damping_scaled = min_damp + damping_norm * (max_damp - min_damp)
        
        return damping_scaled
    
    def _apply_pose_delta(self, current_pose: torch.Tensor, pose_delta: torch.Tensor) -> torch.Tensor:
        """Apply pose delta to current pose.
        
        Args:
            current_pose: Current EE pose [batch_size, 7] (pos + quat)
            pose_delta: Pose delta [batch_size, 6] (pos_delta + axis_angle)
            
        Returns:
            Target pose [batch_size, 7] (pos + quat)
        """
        # Extract position and orientation deltas
        pos_delta = pose_delta[:, :3]
        axis_angle_delta = pose_delta[:, 3:6]
        
        # Apply position delta
        current_pos = current_pose[:, :3]
        target_pos = current_pos + pos_delta
        
        # Apply orientation delta (axis-angle to quaternion)
        current_quat = current_pose[:, 3:7]
        # Convert axis-angle to quaternion using rodrigues formula
        angle = torch.norm(axis_angle_delta, dim=-1, keepdim=True)
        axis = axis_angle_delta / (angle + 1e-8)  # Avoid division by zero
        half_angle = angle * 0.5
        sin_half = torch.sin(half_angle)
        cos_half = torch.cos(half_angle)
        delta_quat = torch.cat([cos_half, sin_half * axis], dim=-1)
        
        # Multiply quaternions: q_result = q_delta * q_current
        target_quat = math_utils.quat_mul(delta_quat, current_quat)
        
        # Combine target pose
        target_pose = torch.cat([target_pos, target_quat], dim=-1)
        
        return target_pose
    
    def _update_contact_state(self):
        """Update contact detection state for impedance adaptation."""
        # This would typically use contact sensors or force feedback
        # For now, use a simple heuristic based on end-effector position
        ee_pose = self._compute_ee_pose()
        if ee_pose is not None:
            ee_pos = ee_pose[:, :3]
            # Simple contact detection: if EE is below certain height
            contact_height_threshold = 0.8  # Adjust based on environment
            self._in_contact[:] = ee_pos[:, 2] < contact_height_threshold
        else:
            # If pose computation fails, assume no contact
            self._in_contact[:] = False
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset action term."""
        super().reset(env_ids)
        
        # Reset impedance parameters
        if env_ids is None:
            # Reset all environments
            self._prev_stiffness[:] = torch.tensor(
                self.cfg.free_motion_stiffness, device=self.device
            )
            self._prev_damping[:] = torch.tensor(
                self.cfg.free_motion_damping, device=self.device
            )
            self._in_contact[:] = False
        else:
            # Reset specific environments
            self._prev_stiffness[env_ids] = torch.tensor(
                self.cfg.free_motion_stiffness, device=self.device
            )
            self._prev_damping[env_ids] = torch.tensor(
                self.cfg.free_motion_damping, device=self.device
            )
            self._in_contact[env_ids] = False

    def get_current_impedance_params(self) -> dict[str, torch.Tensor]:
        """Get current impedance parameters for observations.
        
        Returns:
            Dictionary containing current stiffness and damping parameters.
        """
        return {
            "stiffness": self._prev_stiffness.clone(),
            "damping": self._prev_damping.clone(),
            "in_contact": self._in_contact.clone(),
        }
