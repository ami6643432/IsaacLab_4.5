"""Strict contact penalty helpers for force-controlled cabinet tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def non_grasp_contact_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize contact forces when the handle isn't properly grasped."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w
    if net_forces is None:
        force_mag = torch.zeros(env.num_envs, device=env.device)
    else:
        total_force = torch.sum(net_forces, dim=1)
        force_mag = torch.norm(total_force, dim=-1)

    robot: Articulation = env.scene[asset_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_opening = torch.mean(gripper_pos, dim=-1)
    grasped = gripper_opening < 0.012
    penalty = torch.where(grasped, torch.zeros_like(force_mag), force_mag)
    return penalty


def smooth_approach_reward(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward approaching the handle without contact."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]
    cabinet = env.scene["cabinet"]
    handle_pos = cabinet.data.root_pos_w + torch.tensor(
        [0.4, 0.0, 0.0], device=env.device
    )
    distance = torch.norm(ee_pos - handle_pos, dim=-1)

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w
    if net_forces is None:
        force_mag = torch.zeros(env.num_envs, device=env.device)
    else:
        total_force = torch.sum(net_forces, dim=1)
        force_mag = torch.norm(total_force, dim=-1)

    return torch.exp(-8.0 * distance) * torch.exp(-5.0 * force_mag)


def dragging_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty for dragging the gripper while in contact."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_vel = robot.data.body_lin_vel_w[:, robot.body_names.index("panda_hand"), :]
    ee_speed = torch.norm(ee_vel, dim=-1)

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w
    if net_forces is None:
        force_mag = torch.zeros(env.num_envs, device=env.device)
    else:
        total_force = torch.sum(net_forces, dim=1)
        force_mag = torch.norm(total_force, dim=-1)

    return ee_speed * force_mag


# Backwards compatibility alias
anti_dragging_penalty = dragging_penalty
