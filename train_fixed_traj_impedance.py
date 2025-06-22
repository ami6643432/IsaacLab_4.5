#!/usr/bin/env python3
"""Train fixed trajectory impedance environment with RSL-RL."""

import argparse
import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed trajectory impedance training")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_iterations", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.agents.rsl_rl_ppo_cfg import (
    ForceVariableImpedanceCabinetPPORunnerCfg,
)
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.fixed_traj_impedance_env_cfg import (
    FixedTrajImpedanceCabinetEnvCfg,
)
from isaaclab_tasks.utils.fixed_traj_sm import FixedTrajStateMachine


def main():
    env_cfg = FixedTrajImpedanceCabinetEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

    agent_cfg = ForceVariableImpedanceCabinetPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = args_cli.device if args_cli.device else agent_cfg.device

    env = gym.make(
        "Isaac-Open-Drawer-Franka-Fixed-Impedance-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.render else None,
    )

    sm = FixedTrajStateMachine(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
    )
    arm_action = env.unwrapped.action_manager._terms["arm_action"]

    class SMWrapper(gym.Wrapper):
        def step(self, action):
            with torch.no_grad():
                ee_tf = env.unwrapped.scene["ee_frame"].data
                cab_tf = env.unwrapped.scene["cabinet_frame"].data
                ee_pose = torch.cat(
                    [
                        ee_tf.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins,
                        ee_tf.target_quat_w[..., 0, :],
                    ],
                    dim=-1,
                )
                handle_pose = torch.cat(
                    [
                        cab_tf.target_pos_w[..., 0, :]
                        - env.unwrapped.scene.env_origins,
                        cab_tf.target_quat_w[..., 0, :],
                    ],
                    dim=-1,
                )
                des = sm.compute(ee_pose, handle_pose)
                arm_action.set_desired_pose(des)
            return self.env.step(action)

    env = SMWrapper(env)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = os.path.join(
        "logs",
        "rsl_rl",
        agent_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    runner = OnPolicyRunner(
        wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
