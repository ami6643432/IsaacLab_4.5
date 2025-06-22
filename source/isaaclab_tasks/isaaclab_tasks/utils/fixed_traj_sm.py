"""Simple state machine for opening a cabinet drawer with a fixed trajectory."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.utils.math import combine_frame_transforms


class FixedTrajStateMachine:
    """State machine to compute desired end-effector poses."""

    REST = 0
    APPROACH = 1
    GRASP = 2
    OPEN = 3
    DONE = 4

    WAIT_TIMES = {
        REST: 0.5,
        APPROACH: 1.0,
        GRASP: 1.0,
        OPEN: 3.0,
        DONE: 0.2,
    }

    def __init__(self, dt: float, num_envs: int, device: str):
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wait_time = torch.zeros(num_envs, device=device)
        # offsets
        self.approach_offset = (
            torch.tensor([-0.1, 0, 0, 1, 0, 0, 0], device=device)
            .unsqueeze(0)
            .repeat(num_envs, 1)
        )
        self.grasp_offset = (
            torch.tensor([0.025, 0, 0, 1, 0, 0, 0], device=device)
            .unsqueeze(0)
            .repeat(num_envs, 1)
        )
        self.open_rate = (
            torch.tensor([-0.015, 0, 0, 1, 0, 0, 0], device=device)
            .unsqueeze(0)
            .repeat(num_envs, 1)
        )

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = self.REST
        self.wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, handle_pose: torch.Tensor) -> torch.Tensor:
        des_pose = torch.zeros_like(ee_pose)
        # REST
        mask = self.state == self.REST
        des_pose[mask] = ee_pose[mask]
        next_mask = mask & (self.wait_time >= self.WAIT_TIMES[self.REST])
        self.state[next_mask] = self.APPROACH
        self.wait_time[next_mask] = 0.0

        # APPROACH
        mask = self.state == self.APPROACH
        if mask.any():
            pos, quat = combine_frame_transforms(
                handle_pose[mask, :3],
                handle_pose[mask, 3:],
                self.approach_offset[mask, :3],
                self.approach_offset[mask, 3:],
            )
            des_pose[mask, :3] = pos
            des_pose[mask, 3:] = quat
            reached = torch.norm(ee_pose[mask, :3] - pos, dim=-1) < 0.01
            cond = reached & (self.wait_time[mask] >= self.WAIT_TIMES[self.APPROACH])
            idx = mask.nonzero(as_tuple=False).squeeze(-1)[cond]
            self.state[idx] = self.GRASP
            self.wait_time[idx] = 0.0

        # GRASP
        mask = self.state == self.GRASP
        if mask.any():
            pos, quat = combine_frame_transforms(
                handle_pose[mask, :3],
                handle_pose[mask, 3:],
                self.grasp_offset[mask, :3],
                self.grasp_offset[mask, 3:],
            )
            des_pose[mask, :3] = pos
            des_pose[mask, 3:] = quat
            cond = self.wait_time[mask] >= self.WAIT_TIMES[self.GRASP]
            idx = mask.nonzero(as_tuple=False).squeeze(-1)[cond]
            self.state[idx] = self.OPEN
            self.wait_time[idx] = 0.0

        # OPEN
        mask = self.state == self.OPEN
        if mask.any():
            pos, quat = combine_frame_transforms(
                handle_pose[mask, :3],
                handle_pose[mask, 3:],
                self.open_rate[mask, :3],
                self.open_rate[mask, 3:],
            )
            des_pose[mask, :3] = pos
            des_pose[mask, 3:] = quat
            cond = self.wait_time[mask] >= self.WAIT_TIMES[self.OPEN]
            idx = mask.nonzero(as_tuple=False).squeeze(-1)[cond]
            self.state[idx] = self.DONE
            self.wait_time[idx] = 0.0

        # DONE
        mask = self.state == self.DONE
        des_pose[mask] = ee_pose[mask]

        self.wait_time += self.dt
        return des_pose
