# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict contact force penalty functions for force variable impedance control."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def phase_aware_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    phase: str,
    force_threshold: float,
) -> torch.Tensor:
    """
    Strict phase-aware contact force penalty with exponential scaling.
    
    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration
        phase: Manipulation phase ("approach", "grasp", "manipulation")
        force_threshold: Force threshold for this phase
    """
    # Get contact forces with proper shape handling
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    print(f"DEBUG: env.num_envs = {env.num_envs}")
    print(f"DEBUG: contact_sensor.num_instances = {contact_sensor.num_instances}")
    print(f"DEBUG: contact_sensor.num_bodies = {contact_sensor.num_bodies}")
    
    if contact_sensor.data.net_forces_w_history is not None:
        net_contact_forces = contact_sensor.data.net_forces_w_history
        print(f"DEBUG: net_forces_w_history shape: {net_contact_forces.shape}")
        print(f"DEBUG: net_forces_w_history actual data: {net_contact_forces}")
        
        # Check if we have the right number of environments
        if net_contact_forces.shape[0] != env.num_envs:
            print(f"ERROR: net_forces_w_history has {net_contact_forces.shape[0]} envs, expected {env.num_envs}")
            # Pad or truncate to match expected size
            if net_contact_forces.shape[0] < env.num_envs:
                # Pad with zeros
                padding_shape = (env.num_envs - net_contact_forces.shape[0],) + net_contact_forces.shape[1:]
                padding = torch.zeros(padding_shape, device=net_contact_forces.device, dtype=net_contact_forces.dtype)
                net_contact_forces = torch.cat([net_contact_forces, padding], dim=0)
            else:
                # Truncate
                net_contact_forces = net_contact_forces[:env.num_envs]
        
        # Sum across bodies first, then compute magnitude: (num_envs, history, num_bodies, 3) -> (num_envs, 3) -> (num_envs,)
        # Take the most recent history (index 0)
        if len(net_contact_forces.shape) == 4:
            net_contact_forces = net_contact_forces[:, 0, :, :]  # Take most recent: (num_envs, num_bodies, 3)
        
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies: (num_envs, 3)
        print(f"DEBUG: total_force shape after sum: {total_force.shape}")
        force_magnitude = torch.norm(total_force, dim=-1)  # Compute magnitude: (num_envs,)
        print(f"DEBUG: force_magnitude shape after norm: {force_magnitude.shape}")
        
    elif contact_sensor.data.net_forces_w is not None:
        net_contact_forces = contact_sensor.data.net_forces_w
        print(f"DEBUG: net_forces_w shape: {net_contact_forces.shape}")
        print(f"DEBUG: net_forces_w actual data: {net_contact_forces}")
        
        # Check if we have the right number of environments
        if net_contact_forces.shape[0] != env.num_envs:
            print(f"ERROR: net_forces_w has {net_contact_forces.shape[0]} envs, expected {env.num_envs}")
            # Pad or truncate to match expected size
            if net_contact_forces.shape[0] < env.num_envs:
                # Pad with zeros
                padding_shape = (env.num_envs - net_contact_forces.shape[0],) + net_contact_forces.shape[1:]
                padding = torch.zeros(padding_shape, device=net_contact_forces.device, dtype=net_contact_forces.dtype)
                net_contact_forces = torch.cat([net_contact_forces, padding], dim=0)
            else:
                # Truncate
                net_contact_forces = net_contact_forces[:env.num_envs]
        
        # Sum across bodies first, then compute magnitude: (num_envs, num_bodies, 3) -> (num_envs, 3) -> (num_envs,)
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies: (num_envs, 3)
        print(f"DEBUG: total_force shape after sum: {total_force.shape}")
        force_magnitude = torch.norm(total_force, dim=-1)  # Compute magnitude: (num_envs,)
        print(f"DEBUG: force_magnitude shape after norm: {force_magnitude.shape}")
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
        print(f"DEBUG: force_magnitude shape (fallback): {force_magnitude.shape}")
    
    # Ensure force_magnitude has exactly the right shape
    if force_magnitude.shape[0] != env.num_envs:
        print(f"WARNING: force_magnitude shape mismatch. Resizing from {force_magnitude.shape} to ({env.num_envs},)")
        if force_magnitude.shape[0] < env.num_envs:
            # Pad with zeros
            padding = torch.zeros(env.num_envs - force_magnitude.shape[0], device=force_magnitude.device)
            force_magnitude = torch.cat([force_magnitude, padding], dim=0)
        else:
            # Truncate
            force_magnitude = force_magnitude[:env.num_envs]
    
    # Determine current manipulation phase - ensure proper shape (num_envs, 3)
    current_phases = get_manipulation_phases(env)
    print(f"DEBUG: current_phases shape: {current_phases.shape}")
    
    # Apply penalty based on phase with strict multipliers
    if phase == "approach":
        phase_mask = current_phases[:, 0] > 0.5  # Approach phase active
        print(f"DEBUG: approach phase_mask shape: {phase_mask.shape}")
        penalty_multiplier = 5.0  # Extra severe during approach
        
    elif phase == "grasp":
        phase_mask = current_phases[:, 1] > 0.5  # Grasp phase active
        print(f"DEBUG: grasp phase_mask shape: {phase_mask.shape}")
        penalty_multiplier = 3.0  # Moderate but strict penalty
        
    elif phase == "manipulation":
        phase_mask = current_phases[:, 2] > 0.5  # Manipulation phase active
        print(f"DEBUG: manipulation phase_mask shape: {phase_mask.shape}")
        penalty_multiplier = 1.5  # Standard penalty with some leniency
        
    else:
        phase_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        penalty_multiplier = 3.0
    
    # Ensure all tensors have consistent shape (num_envs,)
    print(f"DEBUG: Before shape fixes - force_magnitude: {force_magnitude.shape}, phase_mask: {phase_mask.shape}")
    if force_magnitude.dim() > 1:
        force_magnitude = force_magnitude.squeeze()
        print(f"DEBUG: After squeeze - force_magnitude: {force_magnitude.shape}")
    if force_magnitude.shape[0] != env.num_envs:
        print(f"DEBUG: Truncating force_magnitude from {force_magnitude.shape[0]} to {env.num_envs}")
        force_magnitude = force_magnitude[:env.num_envs]
        print(f"DEBUG: After truncation - force_magnitude: {force_magnitude.shape}")
    
    # Compute penalty for forces exceeding threshold
    excess_force = torch.clamp(force_magnitude - force_threshold, min=0.0)
    print(f"DEBUG: excess_force shape: {excess_force.shape}")
    
    # Exponential penalty scaling - very harsh for any contact
    penalty = penalty_multiplier * ((excess_force / force_threshold) ** 2 + 
                                  0.5 * (excess_force / force_threshold))
    print(f"DEBUG: penalty shape after computation: {penalty.shape}")
    
    # Ensure penalty has the correct shape (num_envs,) and matches phase_mask
    if penalty.dim() > 1:
        penalty = penalty.squeeze()
        print(f"DEBUG: penalty shape after squeeze: {penalty.shape}")
    if penalty.shape[0] != phase_mask.shape[0]:
        print(f"DEBUG: Adjusting penalty shape from {penalty.shape[0]} to {phase_mask.shape[0]}")
        penalty = penalty[:phase_mask.shape[0]]
        print(f"DEBUG: penalty shape after adjustment: {penalty.shape}")
    
    print(f"DEBUG: Final shapes before torch.where - phase_mask: {phase_mask.shape}, penalty: {penalty.shape}")
    
    # Apply penalty only during relevant phase - ensure shape compatibility
    final_penalty = torch.where(phase_mask, penalty, torch.zeros_like(penalty))
    
    # Final safety check - ensure output shape is (num_envs,)
    if final_penalty.shape[0] != env.num_envs:
        final_penalty = final_penalty[:env.num_envs]
    
    return final_penalty


def non_grasp_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Severe penalty for contact forces when not properly grasping the handle.
    """
    # Get contact forces with safety checks and proper shape handling
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    if contact_sensor.data.net_forces_w_history is not None:
        net_contact_forces = contact_sensor.data.net_forces_w_history
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    elif contact_sensor.data.net_forces_w is not None:
        net_contact_forces = contact_sensor.data.net_forces_w
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Ensure proper shape
    if force_magnitude.shape[0] != env.num_envs:
        force_magnitude = force_magnitude[:env.num_envs]
    
    # Check if handle is properly grasped
    robot: Articulation = env.scene[asset_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -2:]  # Last 2 joints are gripper
    gripper_opening = torch.mean(gripper_pos, dim=-1)  # (num_envs,)
    
    # Handle is grasped if gripper is sufficiently closed
    handle_grasped = gripper_opening < 0.012  # Very tight grasp requirement
    
    # Get end-effector position relative to handle
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]  # (num_envs, 3)
    cabinet = env.scene["cabinet"]
    handle_pos = cabinet.data.root_pos_w + torch.tensor([0.4, 0.0, 0.0], device=env.device).unsqueeze(0)
    
    # Ensure shape compatibility
    if handle_pos.shape[0] == 1 and ee_pos.shape[0] > 1:
        handle_pos = handle_pos.expand(ee_pos.shape[0], -1)
    
    distance_to_handle = torch.norm(ee_pos - handle_pos, dim=-1)  # (num_envs,)
    
    # Very close to handle position required
    near_handle = distance_to_handle < 0.08
    
    # Allow contact only when properly grasping handle
    contact_allowed = handle_grasped & near_handle
    
    # Severe penalty for any unwanted contact
    contact_penalty = torch.where(
        contact_allowed,
        torch.zeros_like(force_magnitude),  # No penalty when properly grasping
        force_magnitude * 3.0 + (force_magnitude ** 2) * 0.5  # Very severe penalty
    )
    
    # Ensure output shape
    if contact_penalty.shape[0] != env.num_envs:
        contact_penalty = contact_penalty[:env.num_envs]
    
    return contact_penalty


def smooth_approach_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    High reward for approaching handle smoothly without any contact.
    """
    # Get distance to handle
    robot: Articulation = env.scene[asset_cfg.name]
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]  # (num_envs, 3)
    cabinet = env.scene["cabinet"]
    handle_pos = cabinet.data.root_pos_w + torch.tensor([0.4, 0.0, 0.0], device=env.device).unsqueeze(0)
    
    # Ensure shape compatibility
    if handle_pos.shape[0] == 1 and ee_pos.shape[0] > 1:
        handle_pos = handle_pos.expand(ee_pos.shape[0], -1)
    
    distance = torch.norm(ee_pos - handle_pos, dim=-1)  # (num_envs,)
    
    # Get contact force with safety checks and proper shape handling
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    if contact_sensor.data.net_forces_w_history is not None:
        net_contact_forces = contact_sensor.data.net_forces_w_history
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    elif contact_sensor.data.net_forces_w is not None:
        net_contact_forces = contact_sensor.data.net_forces_w
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Ensure proper shape
    if force_magnitude.shape[0] != env.num_envs:
        force_magnitude = force_magnitude[:env.num_envs]
    
    # High reward for getting close without ANY contact
    approach_reward = torch.exp(-8.0 * distance)  # Strong exponential reward for proximity
    contact_free_bonus = torch.exp(-force_magnitude * 5.0)  # Large bonus for zero contact
    
    # Only reward during approach phase
    phases = get_manipulation_phases(env)
    approach_mask = phases[:, 0] > 0.5
    
    total_reward = approach_reward * contact_free_bonus
    
    # Ensure output shape
    if total_reward.shape[0] != env.num_envs:
        total_reward = total_reward[:env.num_envs]
    
    return torch.where(approach_mask, total_reward, torch.zeros_like(total_reward))


def dragging_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Severe penalty for dragging behavior (high velocity + high force).
    """
    # Get velocity and force information
    robot: Articulation = env.scene[asset_cfg.name]
    ee_vel = robot.data.body_lin_vel_w[:, robot.body_names.index("panda_hand"), :]  # (num_envs, 3)
    ee_speed = torch.norm(ee_vel, dim=-1)  # (num_envs,)
    
    # Get contact force with safety checks and proper shape handling
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    if contact_sensor.data.net_forces_w_history is not None:
        net_contact_forces = contact_sensor.data.net_forces_w_history
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    elif contact_sensor.data.net_forces_w is not None:
        net_contact_forces = contact_sensor.data.net_forces_w
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Ensure proper shapes
    if force_magnitude.shape[0] != env.num_envs:
        force_magnitude = force_magnitude[:env.num_envs]
    if ee_speed.shape[0] != env.num_envs:
        ee_speed = ee_speed[:env.num_envs]
    
    # Detect dragging: high speed with significant force
    dragging_indicator = ee_speed * force_magnitude
    
    # Exponential penalty for dragging behavior
    dragging_penalty_value = dragging_indicator * 0.5 + (dragging_indicator ** 2) * 0.1
    
    # Extra penalty during non-manipulation phases
    phases = get_manipulation_phases(env)
    manipulation_phase = phases[:, 2] > 0.5
    
    penalty_multiplier = torch.where(manipulation_phase, 1.0, 3.0)  # 3x penalty outside manipulation
    
    final_penalty = dragging_penalty_value * penalty_multiplier
    
    # Ensure output shape
    if final_penalty.shape[0] != env.num_envs:
        final_penalty = final_penalty[:env.num_envs]
    
    return final_penalty


def get_manipulation_phases(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get current manipulation phase encoding with stricter thresholds.
    
    Returns:
        torch.Tensor: Phase encoding (num_envs, 3) [approach, grasp, manipulate]
    """
    robot: Articulation = env.scene["robot"]
    cabinet = env.scene["cabinet"]
    
    # Get end-effector position - ensure proper shape handling
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]
    handle_pos = cabinet.data.root_pos_w + torch.tensor([0.4, 0.0, 0.0], device=env.device).unsqueeze(0)
    
    # Ensure shapes are compatible (num_envs, 3)
    if handle_pos.shape[0] == 1 and ee_pos.shape[0] > 1:
        handle_pos = handle_pos.expand(ee_pos.shape[0], -1)
    
    distance_to_handle = torch.norm(ee_pos - handle_pos, dim=-1)  # (num_envs,)
    
    # Get gripper state
    gripper_pos = robot.data.joint_pos[:, -2:]  # (num_envs, 2)
    gripper_opening = torch.mean(gripper_pos, dim=-1)  # (num_envs,)
    
    # Get drawer state
    drawer_pos = cabinet.data.joint_pos[:, 0]  # (num_envs,)
    
    # Phase 1: Approach (far from handle) - stricter distance threshold
    approach_phase = (distance_to_handle > 0.10).float()  # (num_envs,)
    
    # Phase 2: Grasp (close to handle, gripper open, drawer not moving) - stricter requirements
    grasp_phase = ((distance_to_handle <= 0.10) & 
                   (distance_to_handle > 0.05) &
                   (gripper_opening > 0.015) & 
                   (drawer_pos < 0.003)).float()  # (num_envs,)
    
    # Phase 3: Manipulate (handle grasped, drawer moving) - stricter grasp requirement
    manipulate_phase = ((distance_to_handle <= 0.05) &
                       (gripper_opening <= 0.015) & 
                       (drawer_pos >= 0.003)).float()  # (num_envs,)
    
    # Stack to create (num_envs, 3) tensor
    phases = torch.stack([approach_phase, grasp_phase, manipulate_phase], dim=-1)
    
    # Ensure correct shape
    if phases.shape[0] != env.num_envs:
        phases = phases[:env.num_envs]
    
    return phases


def progressive_contact_penalty(
    env: ManagerBasedRLEnv,
    term_name: str,
    initial_weight: float,
    final_weight: float,
    initial_threshold: float,
    final_threshold: float,
    num_steps: int,
) -> torch.Tensor:
    """
    Progressive curriculum for stricter contact force penalties.
    """
    current_step = getattr(env, '_curriculum_step', 0)
    
    if current_step < num_steps:
        progress = current_step / num_steps
        
        # Interpolate weight and threshold
        current_weight = initial_weight + progress * (final_weight - initial_weight)
        current_threshold = initial_threshold + progress * (final_threshold - initial_threshold)
        
        # Store current values in environment
        setattr(env, f'_{term_name}_weight', current_weight)
        setattr(env, f'_{term_name}_threshold', current_threshold)
        
    else:
        current_weight = final_weight
        current_threshold = final_threshold
    
    # Apply the penalty with current parameters - with safety checks and proper shape handling
    contact_sensor: ContactSensor = env.scene["contact_forces"]
    if contact_sensor.data.net_forces_w_history is not None:
        net_contact_forces = contact_sensor.data.net_forces_w_history
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    elif contact_sensor.data.net_forces_w is not None:
        net_contact_forces = contact_sensor.data.net_forces_w
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    excess_force = torch.clamp(force_magnitude - current_threshold, min=0.0)
    penalty = current_weight * ((excess_force / current_threshold) ** 2)
    
    return penalty


def progressive_approach_reward(
    env: ManagerBasedRLEnv,
    term_name: str,
    initial_weight: float,
    final_weight: float,
    num_steps: int,
) -> torch.Tensor:
    """
    Progressive curriculum for increasing contact-free approach rewards.
    """
    current_step = getattr(env, '_curriculum_step', 0)
    
    if current_step < num_steps:
        progress = current_step / num_steps
        current_weight = initial_weight + progress * (final_weight - initial_weight)
    else:
        current_weight = final_weight
    
    # Get basic approach reward
    robot: Articulation = env.scene["robot"]
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand")]
    cabinet = env.scene["cabinet"]
    handle_pos = cabinet.data.root_pos_w + torch.tensor([0.4, 0.0, 0.0], device=env.device)
    distance = torch.norm(ee_pos - handle_pos, dim=-1)
    
    # Get contact force with safety checks and proper shape handling
    contact_sensor: ContactSensor = env.scene["contact_forces"]
    if contact_sensor.data.net_forces_w_history is not None:
        net_contact_forces = contact_sensor.data.net_forces_w_history
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    elif contact_sensor.data.net_forces_w is not None:
        net_contact_forces = contact_sensor.data.net_forces_w
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # Sum across bodies
        force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Reward for approach without contact
    approach_reward = torch.exp(-5.0 * distance)
    contact_free_bonus = torch.exp(-force_magnitude * 2.0)
    
    total_reward = current_weight * approach_reward * contact_free_bonus
    return total_reward


def simple_contact_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """
    Simple contact force penalty - the most basic strict contact reward.
    
    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration
        force_threshold: Force threshold for penalty
    """
    # Get contact forces with proper shape handling
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    if contact_sensor.data.net_forces_w_history is not None:
        net_contact_forces = contact_sensor.data.net_forces_w_history
    elif contact_sensor.data.net_forces_w is not None:
        net_contact_forces = contact_sensor.data.net_forces_w
    else:
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
        return force_magnitude
    
    # Debug print to understand the tensor shape
    # print(f"Debug: net_contact_forces shape: {net_contact_forces.shape}")
    
    # Handle different tensor shapes properly
    if len(net_contact_forces.shape) == 3:
        # Shape is (num_envs, num_bodies, 3) - sum across bodies first
        total_force = torch.sum(net_contact_forces[..., :3], dim=-2)  # -> (num_envs, 3)
        force_magnitude = torch.norm(total_force, dim=-1)  # -> (num_envs,)
    elif len(net_contact_forces.shape) == 2:
        # Shape is (num_envs, 3) - already summed across bodies
        force_magnitude = torch.norm(net_contact_forces[..., :3], dim=-1)  # -> (num_envs,)
    else:
        # Unknown shape - create zero tensor
        force_magnitude = torch.zeros(env.num_envs, device=env.device)
    
    # Simple exponential penalty for forces above threshold
    excess_force = torch.clamp(force_magnitude - force_threshold, min=0.0)
    penalty = torch.exp(excess_force / force_threshold) - 1.0
    
    # Ensure proper output shape: (num_envs,)
    if penalty.shape != (env.num_envs,):
        # Try to reshape - if this fails, create zeros
        try:
            penalty = penalty.view(env.num_envs)
        except RuntimeError:
            # If reshape fails, return zeros and debug
            print(f"Warning: simple_contact_force_penalty tensor reshape failed. Shape: {penalty.shape}, Expected: ({env.num_envs},)")
            penalty = torch.zeros(env.num_envs, device=env.device)
    
    return penalty
