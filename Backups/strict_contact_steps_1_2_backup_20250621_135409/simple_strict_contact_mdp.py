# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple strict contact force penalty functions for force variable impedance control."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def simple_phase_aware_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    phase: str,
    force_threshold: float,
) -> torch.Tensor:
    """
    Simple phase-aware contact force penalty - just get the net force vector and apply penalty.
    
    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration
        phase: Manipulation phase ("approach", "grasp", "manipulation")
        force_threshold: Force threshold for this phase
    """
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    # Get the net force data - use the most direct approach
    if contact_sensor.data.net_forces_w is not None:
        net_forces = contact_sensor.data.net_forces_w
        # Expected shape: (num_envs, num_bodies, 3) or (num_envs, 3)
        
        if len(net_forces.shape) == 3:
            # Sum across bodies: (num_envs, num_bodies, 3) -> (num_envs, 3)
            force_vector = torch.sum(net_forces, dim=1)
        else:
            # Already (num_envs, 3)
            force_vector = net_forces
            
        # Get magnitude: (num_envs, 3) -> (num_envs,)
        force_magnitude = torch.norm(force_vector, dim=1)
        
    else:
        # No contact forces available
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Ensure we have the right number of environments
    if force_magnitude.shape[0] != env.num_envs:
        if force_magnitude.shape[0] < env.num_envs:
            # Pad with zeros
            padding = torch.zeros(env.num_envs - force_magnitude.shape[0], device=env.device)
            force_magnitude = torch.cat([force_magnitude, padding])
        else:
            # Truncate
            force_magnitude = force_magnitude[:env.num_envs]
    
    # Get manipulation phases
    phases = get_simple_manipulation_phases(env)
    
    # Select phase mask
    if phase == "approach":
        phase_mask = phases[:, 0] > 0.5
    elif phase == "grasp":
        phase_mask = phases[:, 1] > 0.5
    elif phase == "manipulation":
        phase_mask = phases[:, 2] > 0.5
    else:
        # Unknown phase - no penalty
        return torch.zeros(env.num_envs, device=env.device)
    
    # Calculate penalty
    excess_force = torch.clamp(force_magnitude - force_threshold, min=0.0)
    penalty = torch.exp(excess_force / force_threshold) - 1.0
    
    # Apply only during the specified phase
    final_penalty = torch.where(phase_mask, penalty, torch.zeros_like(penalty))
    
    return final_penalty


def get_simple_manipulation_phases(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Simple manipulation phase detection.
    
    Returns:
        torch.Tensor: Phase encoding (num_envs, 3) [approach, grasp, manipulate]
    """
    robot: Articulation = env.scene["robot"]
    cabinet = env.scene["cabinet"]
    
    # Get end-effector position
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]
    
    # Get handle position (approximate)
    handle_pos = cabinet.data.root_pos_w + torch.tensor([0.4, 0.0, 0.0], device=env.device).unsqueeze(0)
    
    # Handle broadcasting for single cabinet position
    if handle_pos.shape[0] == 1 and ee_pos.shape[0] > 1:
        handle_pos = handle_pos.expand(ee_pos.shape[0], -1)
    
    # Calculate distance to handle
    distance_to_handle = torch.norm(ee_pos - handle_pos, dim=-1)
    
    # Get gripper state
    gripper_pos = robot.data.joint_pos[:, -2:]  # Last 2 joints are gripper
    gripper_opening = torch.mean(gripper_pos, dim=-1)
    
    # Get drawer position
    drawer_pos = cabinet.data.joint_pos[:, 0]
    
    # Phase detection with simple thresholds
    approach_phase = (distance_to_handle > 0.1).float()
    grasp_phase = ((distance_to_handle <= 0.1) & (distance_to_handle > 0.05) & 
                   (gripper_opening > 0.01) & (drawer_pos < 0.01)).float()
    manipulate_phase = ((distance_to_handle <= 0.05) & (gripper_opening <= 0.01) & 
                       (drawer_pos >= 0.01)).float()
    
    # Stack phases: (num_envs, 3)
    phases = torch.stack([approach_phase, grasp_phase, manipulate_phase], dim=-1)
    
    return phases


def simple_contact_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """
    Simple contact force penalty - just get net force magnitude and penalize.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    if contact_sensor.data.net_forces_w is not None:
        net_forces = contact_sensor.data.net_forces_w
        
        if len(net_forces.shape) == 3:
            # Sum across bodies: (num_envs, num_bodies, 3) -> (num_envs, 3)
            force_vector = torch.sum(net_forces, dim=1)
        else:
            # Already (num_envs, 3)
            force_vector = net_forces
            
        # Get magnitude: (num_envs, 3) -> (num_envs,)
        force_magnitude = torch.norm(force_vector, dim=1)
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Ensure correct number of environments
    if force_magnitude.shape[0] != env.num_envs:
        if force_magnitude.shape[0] < env.num_envs:
            padding = torch.zeros(env.num_envs - force_magnitude.shape[0], device=env.device)
            force_magnitude = torch.cat([force_magnitude, padding])
        else:
            force_magnitude = force_magnitude[:env.num_envs]
    
    # Simple penalty
    excess_force = torch.clamp(force_magnitude - force_threshold, min=0.0)
    penalty = torch.exp(excess_force / force_threshold) - 1.0
    
    return penalty
