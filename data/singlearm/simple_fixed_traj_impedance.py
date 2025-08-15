# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple fixed trajectory impedance controller for testing."""

import torch
import numpy as np
from typing import Dict

from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import ActionCfg
from isaaclab.managers import ActionTerm


class SimpleFixedTrajImpedance(ActionTerm):
    """Simple implementation for fixed trajectory impedance control."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.num_envs = env.num_envs
        self.device = env.device
        
        # Initialize controller
        controller_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="variable",
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],
            contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
            motion_damping_ratio_task=1.0,
            motion_stiffness_task=[200.0, 200.0, 200.0, 20.0, 20.0, 20.0],
            motion_stiffness_limits_task=(50.0, 1000.0),
            motion_damping_ratio_limits_task=(0.2, 5.0),
        )
        self.controller = OperationalSpaceController(
            controller_cfg, self.num_envs, self.device
        )
        
        # Initialize trajectory and state
        self._trajectory = None
        self._current_waypoint_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._desired_pose = torch.zeros(self.num_envs, 7, device=self.device)
        
    def __call__(self, actions):
        """Process impedance control actions."""
        # Split actions into stiffness and damping components
        stiffness = actions[:, :6].clone()
        damping = actions[:, 6:].clone()
        
        # Process stiffness and damping values
        stiffness = torch.clamp(
            stiffness,
            min=50.0,
            max=1000.0,
        )
        damping = torch.clamp(
            damping,
            min=0.2,
            max=5.0,
        )
        
        # Create command tensor for controller
        command = torch.zeros(self.num_envs, 19, device=self.device)
        # Pose (7), stiffness (6), damping (6)
        command[:, :7] = self._desired_pose
        command[:, 7:13] = stiffness
        command[:, 13:19] = damping
        
        # Get current state
        # TODO: Implement this for your specific environment
        
        # Return control action
        return {}
        
    def reset(self, env_ids=None):
        """Reset the controller for the specified environments."""
        if env_ids is None:
            self._current_waypoint_idx.zero_()
            # Reset to first waypoint if trajectory exists
            if self._trajectory is not None:
                self._desired_pose[:] = self._trajectory[:, 0, :]
        else:
            self._current_waypoint_idx[env_ids] = 0
            # Reset to first waypoint if trajectory exists
            if self._trajectory is not None:
                self._desired_pose[env_ids] = self._trajectory[env_ids, 0, :]
                
    def set_trajectory(self, trajectory: torch.Tensor):
        """Set the trajectory to follow.
        
        Args:
            trajectory: Tensor of shape [num_envs, num_waypoints, 7] with waypoints.
        """
        self._trajectory = trajectory
        self._current_waypoint_idx.zero_()
        
        # Set initial waypoint
        if self._trajectory is not None:
            self._desired_pose[:] = self._trajectory[:, 0, :]
            
    def update_trajectory_position(self, dt: float, current_step: int):
        """Update the desired pose based on the trajectory."""
        if self._trajectory is None:
            return
            
        # Simple progression through waypoints based on step count
        steps_per_waypoint = 50  # About 0.8s at dt=0.0166
        
        # Calculate waypoint indices for all environments
        waypoint_indices = (current_step // steps_per_waypoint).clamp(
            0, self._trajectory.shape[1] - 1
        )
        
        # Update waypoints for environments that need to change
        changed_envs = torch.where(waypoint_indices != self._current_waypoint_idx)[0]
        if len(changed_envs) > 0:
            for env_id in changed_envs:
                self._current_waypoint_idx[env_id] = waypoint_indices[env_id]
                self._desired_pose[env_id] = self._trajectory[
                    env_id, waypoint_indices[env_id]
                ]
                

class SimpleFixedTrajImpedanceCfg(ActionCfg):
    """Configuration for SimpleFixedTrajImpedance."""
    
    class_type = SimpleFixedTrajImpedance
