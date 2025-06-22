"""Simpler contact penalty helpers used during curriculum."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def simple_phase_aware_contact_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    phase: str,
    force_threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w
    if net_forces is None:
        force_mag = torch.zeros(env.num_envs, device=env.device)
    else:
        total_force = torch.sum(net_forces, dim=1)
        force_mag = torch.norm(total_force, dim=-1)

    phases = get_simple_manipulation_phases(env)
    if phase == "approach":
        mask = phases[:, 0] > 0.5
    elif phase == "grasp":
        mask = phases[:, 1] > 0.5
    else:
        mask = phases[:, 2] > 0.5

    excess = torch.clamp(force_mag - force_threshold, min=0.0)
    penalty = torch.exp(excess / force_threshold) - 1.0
    return torch.where(mask, penalty, torch.zeros_like(penalty))


def get_simple_manipulation_phases(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    cabinet = env.scene["cabinet"]
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]
    handle_pos = cabinet.data.root_pos_w + torch.tensor(
        [0.4, 0.0, 0.0], device=env.device
    )
    distance = torch.norm(ee_pos - handle_pos, dim=-1)
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_opening = torch.mean(gripper_pos, dim=-1)
    drawer_pos = cabinet.data.joint_pos[:, 0]

    approach = (distance > 0.1).float()
    grasp = (
        (distance <= 0.1)
        & (distance > 0.05)
        & (gripper_opening > 0.01)
        & (drawer_pos < 0.01)
    ).float()
    manipulate = (
        (distance <= 0.05) & (gripper_opening <= 0.01) & (drawer_pos >= 0.01)
    ).float()
    return torch.stack([approach, grasp, manipulate], dim=-1)


def simple_contact_force_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w
    if net_forces is None:
        force_mag = torch.zeros(env.num_envs, device=env.device)
    else:
        total_force = torch.sum(net_forces, dim=1)
        force_mag = torch.norm(total_force, dim=-1)

    excess = torch.clamp(force_mag - force_threshold, min=0.0)
    return torch.exp(excess / force_threshold) - 1.0


# simple wrappers reused from strict version for convenience
from . import strict_contact_mdp

non_grasp_contact_penalty = strict_contact_mdp.non_grasp_contact_penalty
smooth_approach_reward = strict_contact_mdp.smooth_approach_reward
anti_dragging_penalty = strict_contact_mdp.dragging_penalty
