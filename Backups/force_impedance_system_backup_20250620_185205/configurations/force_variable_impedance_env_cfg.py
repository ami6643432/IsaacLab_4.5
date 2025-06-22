# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for force-based variable impedance cabinet manipulation using Franka robot."""

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.cabinet.mdp as cabinet_mdp

from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg, FRAME_MARKER_SMALL_CFG

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class CurriculumCfg:
    """Curriculum terms configuration for the force variable impedance environment."""
    pass


@configclass
class ForceVariableImpedanceCabinetEnvCfg(CabinetEnvCfg):
    """Configuration for force-based variable impedance cabinet manipulation."""

    def __post_init__(self):
        # Initialize parent configuration
        super().__post_init__()

        # Set franka as robot with contact sensors enabled
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True
        
        # CRITICAL: Add the missing ee_frame configuration
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        
        # Add contact sensor for force feedback at end-effector 
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # Use the hand link instead of link8
            update_period=0.0,  # Update every step for real-time feedback
            history_length=3,   # Keep last 3 timesteps for filtering
            track_pose=False,   # We only need force data
        )

        # Configure variable impedance operational space controller
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="variable",  # KEY: Enable variable impedance
                motion_control_axes_task=[1, 1, 1, 1, 1, 1],
                contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
                motion_damping_ratio_task=1.0,
                # Initial impedance values (will be overridden by RL)
                motion_stiffness_task=[500.0, 500.0, 500.0, 50.0, 50.0, 50.0],
                motion_stiffness_limits_task=(50.0, 2000.0),  # Stiffness bounds
                motion_damping_ratio_limits_task=(0.1, 10.0),     # Damping ratio bounds
            ),
            position_scale=0.1,
            orientation_scale=0.1,
            stiffness_scale=0.1,  # Scale for stiffness parameter changes
            damping_ratio_scale=0.1,  # Scale for damping ratio parameter changes
        )

        # Modify gripper action to use position control
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
            close_command_expr={"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
        )

        # Add force feedback and impedance parameters to observations
        self.observations.policy.contact_forces = ObsTerm(
            func=cabinet_mdp.contact_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )
        
        self.observations.policy.current_impedance = ObsTerm(
            func=cabinet_mdp.current_impedance_params,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # CURRICULUM LEARNING: Progressive skill acquisition
        # Phase 1 (iter 0-300): Master basic approach and grasping
        # Phase 2 (iter 300-500): Learn drawer opening mechanics  
        # Phase 3 (iter 500-700): Introduce force awareness
        # Phase 4 (iter 700-800): Full force/impedance optimization
        
        from isaaclab_tasks.manager_based.manipulation.cabinet.mdp import force_impedance_mdp
        
        # Create force/impedance reward terms with curriculum weighting
        contact_force_penalty = RewTerm(
            func=force_impedance_mdp.contact_force_penalty,
            weight=0.0,  # Will be activated in Phase 3
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "max_force": 50.0,  # Maximum allowed contact force (Newtons)
            }
        )
        
        impedance_adaptation = RewTerm(
            func=force_impedance_mdp.impedance_adaptation_reward,
            weight=0.0,  # Will be activated in Phase 4
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            }
        )
        
        # Add force/impedance rewards to configuration
        setattr(self.rewards, 'contact_force_penalty', contact_force_penalty)
        setattr(self.rewards, 'impedance_adaptation', impedance_adaptation)

        # PHASE 1-2: Dramatically boost basic manipulation rewards
        # Focus on getting the robot to reliably approach, grasp, and open drawer
        self.rewards.approach_ee_handle.weight = 6.0     # Boosted from 2.0 → 6.0  
        self.rewards.approach_gripper_handle.weight = 15.0  # Boosted from 5.0 → 15.0 (CRITICAL)
        self.rewards.grasp_handle.weight = 8.0          # Boosted from 0.5 → 8.0 (CRITICAL) 
        self.rewards.open_drawer_bonus.weight = 20.0    # Boosted from 7.5 → 20.0
        self.rewards.multi_stage_open_drawer.weight = 5.0  # Boosted from 1.0 → 5.0
        
        # Curriculum schedule for progressive reward activation
        grasp_focus_curriculum = CurrTerm(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "grasp_handle",
                "weight": 12.0,  # Further boost grasp learning
                "num_steps": 9600,   # ~100 iterations * 96 steps = boost grasping early
            }
        )
        
        force_awareness_curriculum = CurrTerm(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "contact_force_penalty",
                "weight": -0.005,  # Gentle introduction of force awareness
                "num_steps": 48000,  # ~500 iterations * 96 steps = Phase 3 start
            }
        )
        
        impedance_mastery_curriculum = CurrTerm(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "impedance_adaptation", 
                "weight": 1.0,   # Full impedance optimization
                "num_steps": 67200,  # ~700 iterations * 96 steps = Phase 4 start
            }
        )
        
        # Initialize curriculum manager if needed
        if not hasattr(self, 'curriculum') or self.curriculum is None:
            self.curriculum = CurriculumCfg()
            
        # Add all curriculum terms
        setattr(self.curriculum, 'grasp_focus_curriculum', grasp_focus_curriculum)
        setattr(self.curriculum, 'force_awareness_curriculum', force_awareness_curriculum) 
        setattr(self.curriculum, 'impedance_mastery_curriculum', impedance_mastery_curriculum)

        # Fix missing reward parameters from parent
        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger.*"]


@configclass 
class ForceVariableImpedanceCabinetEnvCfg_PLAY(ForceVariableImpedanceCabinetEnvCfg):
    """Configuration for playing with trained variable impedance policy."""
    
    def __post_init__(self):
        super().__post_init__()
        # Make it easier for play mode
        self.scene.num_envs = 1
        self.episode_length_s = 8.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False

##
# Register Gym environments
##

import gymnasium as gym
from . import agents  # noqa: F401

gym.register(
    id="Isaac-Open-Drawer-Franka-Force-Variable-Impedance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ForceVariableImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForceVariableImpedanceCabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Force-Variable-Impedance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ForceVariableImpedanceCabinetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForceVariableImpedanceCabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)
