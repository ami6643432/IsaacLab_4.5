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


def simple_phase_aware_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    phase: str,
    force_threshold: float,
) -> torch.Tensor:
    """
    Simple phase-aware contact force penalty - just get the net force vector and apply penalty.
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
    
    # Get manipulation phases - simple version
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
    Get simple manipulation phases based on robot state.
    Returns a tensor of shape (num_envs, 3) where each row is [approach, grasp, manipulation]
    """
    robot: Articulation = env.scene["robot"]
    cabinet = env.scene["cabinet"]
    
    # Get end-effector position
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]
    
    # Get handle position (approximate)
    handle_pos = cabinet.data.root_pos_w + torch.tensor([0.4, 0.0, 0.0], device=env.device).unsqueeze(0)
    if handle_pos.shape[0] == 1 and ee_pos.shape[0] > 1:
        handle_pos = handle_pos.expand(ee_pos.shape[0], -1)
    
    # Distance to handle
    distance_to_handle = torch.norm(ee_pos - handle_pos, dim=-1)
    
    # Get gripper state
    gripper_pos = robot.data.joint_pos[:, -2:]  # Last 2 joints are gripper
    gripper_opening = torch.mean(gripper_pos, dim=-1)
    
    # Get drawer opening
    drawer_opening = cabinet.data.joint_pos[:, 0]  # First joint is drawer
    
    # Phase logic
    approach_phase = (distance_to_handle > 0.15) & (gripper_opening > 0.02)
    grasp_phase = (distance_to_handle <= 0.15) & (gripper_opening > 0.01) & (drawer_opening < 0.05)
    manipulation_phase = (gripper_opening <= 0.01) | (drawer_opening >= 0.05)
    
    # Convert to float and stack
    phases = torch.stack([
        approach_phase.float(),
        grasp_phase.float(),
        manipulation_phase.float()
    ], dim=1)
    
    return phases


def non_grasp_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalty for contact forces when not properly grasping the handle.
    """
    # Get contact forces
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    if contact_sensor.data.net_forces_w is not None:
        net_forces = contact_sensor.data.net_forces_w
        if len(net_forces.shape) == 3:
            force_vector = torch.sum(net_forces, dim=1)
        else:
            force_vector = net_forces
        force_magnitude = torch.norm(force_vector, dim=1)
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Get robot state
    robot: Articulation = env.scene[asset_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -2:]  # Last 2 joints are gripper
    gripper_opening = torch.mean(gripper_pos, dim=-1)
    
    # Handle is grasped if gripper is sufficiently closed
    handle_grasped = gripper_opening < 0.015
    
    # Get end-effector position relative to handle
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]
    cabinet = env.scene["cabinet"]
    handle_pos = cabinet.data.root_pos_w + torch.tensor([0.4, 0.0, 0.0], device=env.device).unsqueeze(0)
    
    # Ensure shape compatibility
    if handle_pos.shape[0] == 1 and ee_pos.shape[0] > 1:
        handle_pos = handle_pos.expand(ee_pos.shape[0], -1)
    
    distance_to_handle = torch.norm(ee_pos - handle_pos, dim=-1)
    near_handle = distance_to_handle < 0.08
    
    # Allow contact only when properly grasping handle
    contact_allowed = handle_grasped & near_handle
    
    # Penalty for unwanted contact
    contact_penalty = torch.where(
        contact_allowed,
        torch.zeros_like(force_magnitude),  # No penalty when properly grasping
        force_magnitude * 2.0  # Penalty for contact when not grasping
    )
    
    return contact_penalty


def smooth_approach_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Reward for smooth approach without contact.
    """
    # Get contact forces
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    if contact_sensor.data.net_forces_w is not None:
        net_forces = contact_sensor.data.net_forces_w
        if len(net_forces.shape) == 3:
            force_vector = torch.sum(net_forces, dim=1)
        else:
            force_vector = net_forces
        force_magnitude = torch.norm(force_vector, dim=1)
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Get phases
    phases = get_simple_manipulation_phases(env)
    approach_phase = phases[:, 0] > 0.5
    
    # Reward for approaching without contact
    no_contact = force_magnitude < 1.0
    smooth_approach = approach_phase & no_contact
    
    reward = smooth_approach.float() * 0.1
    
    return reward


def anti_dragging_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalty for dragging behavior (high velocity + high force).
    """
    # Get velocity and force information
    robot: Articulation = env.scene[asset_cfg.name]
    ee_vel = robot.data.body_lin_vel_w[:, robot.body_names.index("panda_hand"), :]
    ee_speed = torch.norm(ee_vel, dim=-1)
    
    # Get contact force
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    if contact_sensor.data.net_forces_w is not None:
        net_forces = contact_sensor.data.net_forces_w
        if len(net_forces.shape) == 3:
            force_vector = torch.sum(net_forces, dim=1)
        else:
            force_vector = net_forces
        force_magnitude = torch.norm(force_vector, dim=1)
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Dragging penalty: high speed + high force
    dragging_penalty = (ee_speed > 0.1) & (force_magnitude > 5.0)
    penalty = dragging_penalty.float() * force_magnitude * 0.5
    
    return penalty
