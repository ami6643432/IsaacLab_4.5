# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationData
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["object"].data

    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

    return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[
        ..., 1:, :
    ] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat


def current_impedance_params(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Current stiffness and damping from the operational space controller."""
    asset: Articulation = env.scene[asset_cfg.name]

    if (
        hasattr(env.action_manager, "_terms")
        and "arm_action" in env.action_manager._terms
    ):
        arm_action = env.action_manager._terms["arm_action"]
        if hasattr(arm_action, "get_current_impedance_params"):
            impedance = arm_action.get_current_impedance_params()
            return torch.cat([impedance["stiffness"], impedance["damping"]], dim=-1)

    default_stiffness = torch.tensor(
        [500.0, 500.0, 500.0, 50.0, 50.0, 50.0], device=env.device
    )
    default_damping = torch.tensor(
        [50.0, 50.0, 50.0, 10.0, 10.0, 10.0], device=env.device
    )
    return torch.cat(
        [
            default_stiffness.unsqueeze(0).repeat(env.num_envs, 1),
            default_damping.unsqueeze(0).repeat(env.num_envs, 1),
        ],
        dim=-1,
    )


def contact_force_magnitude(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Magnitude of net contact force from sensor."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w
    if net_forces is None:
        return torch.zeros(env.num_envs, 1, device=env.device)

    total_force = torch.sum(net_forces, dim=1)
    return torch.norm(total_force, dim=-1, keepdim=True)
