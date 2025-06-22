#!/usr/bin/env python3
# Copyright (c) 2025, Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Action class for fixed trajectory impedance control."""

from __future__ import annotations

import torch
from typing import Sequence, TYPE_CHECKING

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

    class_type: type[FixedTrajImpedanceAction] | None = None  # assigned after class definition
    
    # Scaling factors for impedance parameters
    stiffness_scale: float = 1.0
    """Scale factor for stiffness parameters."""
    
    damping_ratio_scale: float = 1.0
    """Scale factor for damping ratio parameters."""


class FixedTrajImpedanceAction(OperationalSpaceControllerAction):
    """Operational space action that tracks a fixed Cartesian trajectory.

    The policy only outputs impedance parameters. The Cartesian pose is provided
    via :meth:`set_desired_pose` before every environment step.
    """

    cfg: FixedTrajImpedanceActionCfg

    def __init__(self, cfg: FixedTrajImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._desired_pose = torch.zeros(self.num_envs, 7, device=self.device)
        
        # Add missing scaling attributes required by parent class
        self._stiffness_scale = cfg.stiffness_scale if hasattr(cfg, 'stiffness_scale') else 1.0
        self._damping_ratio_scale = cfg.damping_ratio_scale if hasattr(cfg, 'damping_ratio_scale') else 1.0
        
        # buffers to expose desired joint states
        self.desired_joint_pos = torch.zeros(
            self.num_envs, self._num_DoF, device=self.device
        )
        self.desired_joint_vel = torch.zeros_like(self.desired_joint_pos)
        # simple IK controller to compute desired joint pose for observations
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        )
        self._diff_ik = DifferentialIKController(ik_cfg, self.num_envs, self.device)

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        # Only stiffness and damping are commanded by the policy
        return 12

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

    def process_actions(self, actions: torch.Tensor):
        # Update ee pose for IK computation
        self._compute_ee_pose()
        self._compute_task_frame_pose()

        self._raw_actions[:] = actions
        stiffness = actions[:, :6] * self._stiffness_scale
        stiffness = torch.clamp(
            stiffness,
            min=self.cfg.controller_cfg.motion_stiffness_limits_task[0],
            max=self.cfg.controller_cfg.motion_stiffness_limits_task[1],
        )
        damping = actions[:, 6:] * self._damping_ratio_scale
        damping = torch.clamp(
            damping,
            min=self.cfg.controller_cfg.motion_damping_ratio_limits_task[0],
            max=self.cfg.controller_cfg.motion_damping_ratio_limits_task[1],
        )

        # build command tensor
        self._processed_actions.zero_()
        if self._pose_abs_idx is None:
            raise RuntimeError("OSC expected pose_abs command")
        self._processed_actions[:, self._pose_abs_idx : self._pose_abs_idx + 7] = (
            self._desired_pose
        )
        if self._stiffness_idx is not None:
            self._processed_actions[
                :, self._stiffness_idx : self._stiffness_idx + 6
            ] = stiffness
        if self._damping_ratio_idx is not None:
            self._processed_actions[
                :, self._damping_ratio_idx : self._damping_ratio_idx + 6
            ] = damping

        # compute desired joint positions for observations using differential IK
        self._diff_ik.set_command(self._desired_pose)
        des_q = self._diff_ik.compute(
            ee_pos=self._ee_pose_b[:, :3],
            ee_quat=self._ee_pose_b[:, 3:7],
            jacobian=self.jacobian_b,
            joint_pos=self._joint_pos,
        )
        self.desired_joint_pos[:] = des_q
        self.desired_joint_vel[:] = (des_q - self._joint_pos) / self._sim_dt

        # set command in controller
        self._osc.set_command(
            command=self._processed_actions,
            current_ee_pose_b=self._ee_pose_b,
            current_task_frame_pose_b=self._task_frame_pose_b,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._desired_pose.zero_()
        else:
            self._desired_pose[env_ids] = 0.0
        self.desired_joint_pos[env_ids] = 0.0
        self.desired_joint_vel[env_ids] = 0.0


FixedTrajImpedanceActionCfg.class_type = FixedTrajImpedanceAction
