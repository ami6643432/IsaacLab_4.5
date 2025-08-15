# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for force-based variable impedance control."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def current_impedance_params(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current impedance parameters from the operational space controller.
    
    Returns:
        Tensor of shape (num_envs, 12) containing [stiffness (6), damping (6)]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Try to get impedance from the action manager
    if hasattr(env.action_manager, "_terms") and "arm_action" in env.action_manager._terms:
        arm_action = env.action_manager._terms["arm_action"]
        if hasattr(arm_action, "get_current_impedance_params"):
            impedance_data = arm_action.get_current_impedance_params()
            # Concatenate stiffness and damping
            return torch.cat([impedance_data["stiffness"], impedance_data["damping"]], dim=-1)
    
    # Fallback: return default impedance parameters
    batch_size = env.num_envs
    device = env.device
    
    # Default impedance values
    default_stiffness = torch.tensor([500.0, 500.0, 500.0, 50.0, 50.0, 50.0], device=device)
    default_damping = torch.tensor([50.0, 50.0, 50.0, 10.0, 10.0, 10.0], device=device)
    
    # Expand to batch size
    stiffness_batch = default_stiffness.unsqueeze(0).repeat(batch_size, 1)
    damping_batch = default_damping.unsqueeze(0).repeat(batch_size, 1)
    
    return torch.cat([stiffness_batch, damping_batch], dim=-1)


def contact_force_penalty(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    max_force: float = 50.0
) -> torch.Tensor:
    """Penalty for excessive contact forces.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.
        max_force: Maximum allowed contact force in Newtons.
        
    Returns:
        Penalty values for each environment (negative rewards).
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    # Get magnitude of contact forces - compute from net_forces_w tensor
    # net_forces_w shape: (N, B, 3) where N=num_envs, B=num_bodies, 3=xyz forces
    force_magnitude = torch.norm(contact_sensor.data.net_forces_w, dim=-1)  # Shape: (N, B)
    
    # Sum over all bodies for each environment
    total_force_per_env = force_magnitude.sum(dim=-1)  # Shape: (N,)
    
    # Penalty increases quadratically with force above threshold
    force_penalty = torch.where(
        total_force_per_env > max_force,
        ((total_force_per_env - max_force) / max_force) ** 2,
        torch.zeros_like(total_force_per_env)
    )
    
    return -force_penalty


def impedance_adaptation_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for appropriate impedance adaptation based on contact forces.
    
    This reward encourages the agent to:
    - Use low stiffness during approach (compliant behavior)
    - Use high stiffness during contact (precise manipulation)
    - Adapt damping appropriately for stability
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Reward values for appropriate impedance adaptation.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    # Get contact forces - compute from net_forces_w tensor
    # net_forces_w shape: (N, B, 3) where N=num_envs, B=num_bodies, 3=xyz forces
    force_magnitude = torch.norm(contact_sensor.data.net_forces_w, dim=-1).sum(dim=-1)  # Shape: (N,)
    
    # Get current impedance parameters from observations
    # This comes from the current_impedance_params function
    current_impedance = current_impedance_params(env, asset_cfg)
    current_stiffness = current_impedance[:, :6].mean(dim=-1)  # Average translational stiffness
    current_damping = current_impedance[:, 6:].mean(dim=-1)   # Average damping
    
    # Define contact phases
    no_contact = force_magnitude < 1.0         # Free space motion
    light_contact = (force_magnitude >= 1.0) & (force_magnitude < 10.0)  # Initial contact
    firm_contact = force_magnitude >= 10.0     # Firm contact for manipulation
    
    # Optimal impedance for each phase
    # No contact: Low stiffness for compliant approach
    optimal_stiffness_no_contact = torch.tensor(100.0, device=env.device)
    
    # Light contact: Medium stiffness for gentle interaction
    optimal_stiffness_light = torch.tensor(300.0, device=env.device)
    
    # Firm contact: High stiffness for precise manipulation
    optimal_stiffness_firm = torch.tensor(800.0, device=env.device)
    
    # Determine optimal stiffness based on contact phase
    optimal_stiffness = torch.where(
        no_contact, 
        optimal_stiffness_no_contact,
        torch.where(
            light_contact,
            optimal_stiffness_light,
            optimal_stiffness_firm
        )
    )
    
    # Reward for matching optimal stiffness
    stiffness_error = torch.abs(current_stiffness - optimal_stiffness)
    stiffness_reward = torch.exp(-stiffness_error / 200.0)
    
    # Reward for appropriate damping (should be proportional to stiffness)
    optimal_damping = optimal_stiffness / 10.0  # Rule of thumb: damping = stiffness/10
    damping_error = torch.abs(current_damping - optimal_damping)
    damping_reward = torch.exp(-damping_error / 20.0)
    
    # Combined adaptation reward
    adaptation_reward = 0.7 * stiffness_reward + 0.3 * damping_reward
    
    return adaptation_reward


def smooth_impedance_change(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for smooth impedance changes to avoid jerky behavior.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Reward values for smooth impedance transitions.
    """
    # This would require tracking impedance history
    # For now, return a small positive reward
    return torch.ones(env.num_envs, device=env.device) * 0.1


def force_tracking_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_force: float = 5.0,
) -> torch.Tensor:
    """Reward for maintaining desired contact force during manipulation.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.
        target_force: Desired contact force in Newtons.
        
    Returns:
        Reward values for maintaining target force.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    # Get current contact force - compute from net_forces_w tensor  
    # net_forces_w shape: (N, B, 3) where N=num_envs, B=num_bodies, 3=xyz forces
    force_magnitude = torch.norm(contact_sensor.data.net_forces_w, dim=-1).sum(dim=-1)  # Shape: (N,)
    
    # Only apply this reward when in contact
    in_contact = force_magnitude > 0.5
    
    # Reward for being close to target force
    force_error = torch.abs(force_magnitude - target_force)
    force_reward = torch.exp(-force_error / target_force)
    
    # Only give reward when in contact
    return torch.where(in_contact, force_reward, torch.zeros_like(force_reward))


def contact_force_magnitude(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact force magnitude for impedance adaptation.
    
    Returns:
        Tensor of shape (num_envs, 1) containing force magnitudes
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    # Get net forces in Newtons - shape: (num_envs, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w
    
    if net_forces is None:
        # Return zero forces if no contact data available
        return torch.zeros(env.num_envs, 1, device=env.device)
    
    # Sum forces across all bodies and compute magnitude
    total_force = torch.sum(net_forces, dim=1)  # Sum across bodies: (num_envs, 3)
    force_magnitude = torch.norm(total_force, dim=-1, keepdim=True)  # (num_envs, 1)
    
    return force_magnitude


def force_adaptive_impedance_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for adaptive impedance based on contact forces.
    
    Encourages:
    - Low stiffness when forces are high (compliant behavior)
    - High stiffness when forces are low (precise positioning)
    - Appropriate damping for different force levels
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    # Get contact forces - shape: (num_envs, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w
    
    if net_forces is None:
        # Return zero reward if no contact data available
        return torch.zeros(env.num_envs, device=env.device)
    
    # Sum forces across all bodies and compute magnitude
    total_force = torch.sum(net_forces, dim=1)  # (num_envs, 3)
    force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    
    # Get current impedance parameters
    current_impedance = current_impedance_params(env, asset_cfg)
    current_stiffness = current_impedance[:, :6]  # First 6 values
    current_damping = current_impedance[:, 6:]    # Last 6 values
    
    # Calculate optimal impedance based on force feedback
    # High forces → lower stiffness (more compliant)
    # Low forces → higher stiffness (more precise)
    force_threshold_high = 10.0  # N
    force_threshold_low = 1.0    # N
    
    # Normalize force magnitude
    force_normalized = torch.clamp(force_magnitude / force_threshold_high, 0.0, 1.0)
    
    # Optimal stiffness decreases with force
    optimal_stiffness_scale = 1.0 - 0.8 * force_normalized.unsqueeze(-1)  # Range: [0.2, 1.0]
    
    # Optimal damping increases slightly with force for stability
    optimal_damping_scale = 1.0 + 0.5 * force_normalized.unsqueeze(-1)    # Range: [1.0, 1.5]
    
    # Calculate current stiffness and damping scales (assuming base values)
    base_stiffness = torch.tensor([150.0, 150.0, 150.0, 15.0, 15.0, 15.0], device=env.device)
    base_damping = torch.tensor([50.0, 50.0, 50.0, 10.0, 10.0, 10.0], device=env.device)
    
    current_stiffness_scale = current_stiffness / base_stiffness.unsqueeze(0)
    current_damping_scale = current_damping / base_damping.unsqueeze(0)
    
    # Reward based on how close current scales are to optimal
    stiffness_error = torch.abs(current_stiffness_scale - optimal_stiffness_scale)
    damping_error = torch.abs(current_damping_scale - optimal_damping_scale)
    
    # Combine errors (lower is better)
    total_error = torch.mean(stiffness_error + damping_error, dim=-1)
    
    # Convert to reward (higher is better)
    reward = torch.exp(-total_error)
    
    return reward


def impedance_smoothness_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for smooth impedance parameter changes over time."""
    
    # Get current impedance parameters
    current_impedance = current_impedance_params(env, asset_cfg)
    
    # Use a class attribute to store previous impedance (safer than setting env attributes)
    if not hasattr(impedance_smoothness_reward, '_prev_impedance_dict'):
        impedance_smoothness_reward._prev_impedance_dict = {}
    
    env_id = id(env)  # Use environment ID as key
    
    if env_id not in impedance_smoothness_reward._prev_impedance_dict:
        impedance_smoothness_reward._prev_impedance_dict[env_id] = current_impedance.clone()
        return torch.zeros(env.num_envs, device=env.device)
    
    # Calculate impedance velocity (rate of change)
    prev_impedance = impedance_smoothness_reward._prev_impedance_dict[env_id]
    impedance_velocity = current_impedance - prev_impedance
    impedance_rate = torch.norm(impedance_velocity, dim=-1)
    
    # Update history
    impedance_smoothness_reward._prev_impedance_dict[env_id] = current_impedance.clone()
    
    # Reward smooth changes (penalize rapid changes)
    smoothness_reward = torch.exp(-impedance_rate)
    
    return smoothness_reward


def excessive_force_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float) -> torch.Tensor:
    """Penalty for excessive contact forces."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    # Get force magnitude - shape: (num_envs, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w
    
    if net_forces is None:
        # Return zero penalty if no contact data available
        return torch.zeros(env.num_envs, device=env.device)
    
    # Sum forces across all bodies and compute magnitude
    total_force = torch.sum(net_forces, dim=1)  # (num_envs, 3)
    force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
    
    # Penalty for forces above threshold
    excess_force = torch.clamp(force_magnitude - force_threshold, min=0.0)
    penalty = excess_force / force_threshold  # Normalize
    
    return penalty


def current_impedance_parameters(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get current impedance parameters for observation."""
    return current_impedance_params(env, asset_cfg)


# Create dummy classes for compatibility
class ForceImpedanceRewardTerm:
    """Dummy reward term class for compatibility."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return torch.zeros(1)


class ForceImpedanceObservationTerm:
    """Dummy observation term class for compatibility."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return torch.zeros(1, 1)
