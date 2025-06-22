# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for force-based variable impedance cabinet manipulation using Franka robot."""

import isaaclab.envs.mdp as mdp
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.cabinet.mdp as cabinet_mdp
import isaaclab_tasks.manager_based.manipulation.cabinet.mdp.strict_contact_mdp as strict_mdp
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import FRAME_MARKER_SMALL_CFG, CabinetEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class CurriculumCfg:
    """Curriculum terms configuration for the force variable impedance environment."""

    pass


@configclass
class FixedTrajImpedanceCabinetEnvCfg(CabinetEnvCfg):
    """Configuration for force-based variable impedance cabinet manipulation."""

    def __post_init__(self):
        # Initialize parent configuration
        super().__post_init__()
        # ensure observation terms are concatenated
        self.observations.policy.concatenate_terms = True

        # Set franka as robot with contact sensors enabled (correct approach)
        import copy

        from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

        robot_cfg = copy.deepcopy(FRANKA_PANDA_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        # Enable contact sensors by modifying the spawn configuration
        if hasattr(robot_cfg.spawn, "activate_contact_sensors"):
            robot_cfg.spawn.activate_contact_sensors = True
        setattr(self.scene, "robot", robot_cfg)

        # CRITICAL: Add the missing ee_frame configuration
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
                prim_path="/Visuals/EndEffectorFrameTransformer"
            ),
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

        # Add contact sensor for force feedback at end-effector using setattr (as scene doesn't support direct assignment)
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # Use the hand link
            update_period=0.0,  # Update every step for real-time feedback
            history_length=3,  # Keep last 3 timesteps for filtering
            track_pose=False,  # We only need force data
        )
        setattr(self.scene, "contact_forces", contact_sensor_cfg)

        # Enhanced operational space controller with strict force limits
        from data.singlearm.fixed_traj_impedance_action import FixedTrajImpedanceActionCfg

        self.actions.arm_action = FixedTrajImpedanceActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.107)
            ),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="variable",
                motion_control_axes_task=[1, 1, 1, 1, 1, 1],
                contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
                motion_damping_ratio_task=1.5,  # Increased damping for stability
                # Lower stiffness to prevent excessive contact forces
                motion_stiffness_task=[
                    150.0,
                    150.0,
                    150.0,
                    15.0,
                    15.0,
                    15.0,
                ],  # Reduced from default
                motion_stiffness_limits_task=(
                    50.0,
                    1500.0,
                ),  # Reduced upper limit from 2000
                motion_damping_ratio_limits_task=(0.2, 8.0),  # Adjusted damping range
            ),
            position_scale=0.05,  # Further reduced for gentler motion
            orientation_scale=0.05,  # Further reduced for gentler motion
            stiffness_scale=0.03,  # Further reduced scaling
            damping_ratio_scale=0.08,  # Increased damping scaling
        )

        # Modify gripper action to use position control
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            close_command_expr={"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
        )

        # Add force feedback and impedance parameters to observations using setattr (consistent approach)
        contact_forces_obs = ObsTerm(
            func=cabinet_mdp.contact_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )
        setattr(self.observations.policy, "contact_forces", contact_forces_obs)

        current_impedance_obs = ObsTerm(
            func=cabinet_mdp.current_impedance_params,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        setattr(self.observations.policy, "current_impedance", current_impedance_obs)

        # expose current joint states and desired joint states
        self.observations.policy.joint_pos = ObsTerm(func=cabinet_mdp.joint_pos_rel)
        self.observations.policy.joint_vel = ObsTerm(func=cabinet_mdp.joint_vel_rel)

        desired_joint_pos_obs = ObsTerm(
            func=lambda env: env.action_manager._terms["arm_action"].desired_joint_pos,
        )
        desired_joint_vel_obs = ObsTerm(
            func=lambda env: env.action_manager._terms["arm_action"].desired_joint_vel,
        )
        setattr(self.observations.policy, "desired_joint_pos", desired_joint_pos_obs)
        setattr(self.observations.policy, "desired_joint_vel", desired_joint_vel_obs)

        # CURRICULUM LEARNING: Progressive skill acquisition
        # Phase 1 (iter 0-300): Master basic approach and grasping
        # Phase 2 (iter 300-500): Learn drawer opening mechanics
        # Phase 3 (iter 500-700): Introduce force awareness
        # Phase 4 (iter 700-800): Full force/impedance optimization

        from isaaclab_tasks.manager_based.manipulation.cabinet.mdp import force_impedance_mdp

        # === GRADUAL STRICT CONTACT SYSTEM ===
        # Step 1: Add simple contact force penalty to start building strict contact awareness
        # This is the most basic reward - just penalize excessive contact forces

        simple_contact_penalty = RewTerm(
            func=strict_mdp.simple_contact_force_penalty,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "force_threshold": 10.0,  # Conservative threshold to start
            },
            weight=-3.0,  # Moderate penalty to begin with
        )
        setattr(self.rewards, "simple_contact_penalty", simple_contact_penalty)

        # STEP 2: Phase-aware contact penalties - different thresholds for different manipulation phases
        from isaaclab_tasks.manager_based.manipulation.cabinet.mdp import simple_strict_contact_mdp as simple_strict_mdp

        # Approach phase - strictest penalties (should be minimal contact)
        approach_contact_penalty = RewTerm(
            func=simple_strict_mdp.simple_phase_aware_contact_penalty,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "phase": "approach",
                "force_threshold": 5.0,  # Very low threshold for approach phase
            },
            weight=-25.0,  # High penalty for approach phase contact
        )
        setattr(self.rewards, "approach_contact_penalty", approach_contact_penalty)

        # Grasp phase - moderate penalties (some contact expected during grasping)
        grasp_contact_penalty = RewTerm(
            func=simple_strict_mdp.simple_phase_aware_contact_penalty,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "phase": "grasp",
                "force_threshold": 10.0,  # Medium threshold for grasp phase
            },
            weight=-15.0,  # Moderate penalty for grasp phase contact
        )
        setattr(self.rewards, "grasp_contact_penalty", grasp_contact_penalty)

        # Manipulation phase - gentle penalties (contact needed for manipulation)
        manipulation_contact_penalty = RewTerm(
            func=simple_strict_mdp.simple_phase_aware_contact_penalty,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "phase": "manipulation",
                "force_threshold": 15.0,  # Higher threshold for manipulation phase
            },
            weight=-8.0,  # Lower penalty for manipulation phase contact
        )
        setattr(
            self.rewards, "manipulation_contact_penalty", manipulation_contact_penalty
        )

        # STEP 3: Non-grasp contact penalties - prevent contact when not properly grasping

        # Severe penalty for any contact when not grasping handle properly
        non_grasp_contact_penalty = RewTerm(
            func=simple_strict_mdp.non_grasp_contact_penalty,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
            weight=-20.0,  # Very high penalty for unwanted contact
        )
        setattr(self.rewards, "non_grasp_contact_penalty", non_grasp_contact_penalty)

        # STEP 4: Smooth approach rewards - encourage contact-free approach behavior

        # High reward for approaching handle smoothly without contact
        smooth_approach_reward = RewTerm(
            func=simple_strict_mdp.smooth_approach_reward,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
            weight=5.0,  # Positive reward for smooth approach
        )
        setattr(self.rewards, "smooth_approach_reward", smooth_approach_reward)

        # STEP 5: Anti-dragging penalties - prevent high velocity + high force combinations

        # Severe penalty for dragging (high force + high velocity)
        anti_dragging_penalty = RewTerm(
            func=simple_strict_mdp.anti_dragging_penalty,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
            weight=-15.0,  # High penalty for dragging behavior
        )
        setattr(self.rewards, "anti_dragging_penalty", anti_dragging_penalty)

        # === END GRADUAL STRICT CONTACT SYSTEM ===

        # PHASE 1-2: Dramatically boost basic manipulation rewards while enforcing strict contact control
        # Focus on getting the robot to reliably approach, grasp, and open drawer without unwanted contact
        self.rewards.approach_ee_handle.weight = 8.0  # Further boosted from 6.0 → 8.0
        self.rewards.approach_gripper_handle.weight = (
            20.0  # Further boosted from 15.0 → 20.0 (CRITICAL)
        )
        self.rewards.grasp_handle.weight = (
            12.0  # Further boosted from 8.0 → 12.0 (CRITICAL)
        )
        self.rewards.open_drawer_bonus.weight = 25.0  # Further boosted from 20.0 → 25.0
        self.rewards.multi_stage_open_drawer.weight = (
            8.0  # Further boosted from 5.0 → 8.0
        )

        # Enhanced curriculum schedule for progressive reward activation with extended durations
        grasp_focus_curriculum = CurrTerm(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "grasp_handle",
                "weight": 15.0,  # Further boost grasp learning
                "num_steps": 19200,  # Extended: ~200 iterations * 96 steps = longer grasp focus
            },
        )

        # TODO: Re-enable strict contact curriculum after fixing tensor shape issues
        # smooth_approach_curriculum, contact_penalty_curriculum temporarily disabled

        force_awareness_curriculum = CurrTerm(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "contact_force_penalty",
                "weight": -0.005,  # Gentle introduction of force awareness
                "num_steps": 48000,  # Extended: ~500 iterations * 96 steps = Phase 3 start
            },
        )

        impedance_mastery_curriculum = CurrTerm(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "impedance_adaptation",
                "weight": 1.0,  # Full impedance optimization
                "num_steps": 76800,  # Extended: ~800 iterations * 96 steps = Phase 4 start
            },
        )

        # Initialize curriculum manager if needed
        if not hasattr(self, "curriculum") or self.curriculum is None:
            self.curriculum = CurriculumCfg()

        # Add working curriculum terms only
        setattr(self.curriculum, "grasp_focus_curriculum", grasp_focus_curriculum)
        setattr(
            self.curriculum, "force_awareness_curriculum", force_awareness_curriculum
        )
        setattr(
            self.curriculum,
            "impedance_mastery_curriculum",
            impedance_mastery_curriculum,
        )

        # Fix missing reward parameters from parent
        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger.*"]

        # === GRADUAL STRICT CONTACT SYSTEM ===
        # Step 1: Add simple contact force penalty to start building strict contact awareness
        # This is the most basic reward - just penalize excessive contact forces

        simple_contact_penalty = RewTerm(
            func=strict_mdp.simple_contact_force_penalty,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "force_threshold": 10.0,  # Conservative threshold to start
            },
            weight=-3.0,  # Moderate penalty to begin with
        )
        setattr(self.rewards, "simple_contact_penalty", simple_contact_penalty)

        # TODO: Step 2 - Add phase awareness (after testing step 1)
        # TODO: Step 3 - Add non-grasp contact penalties (after testing step 2)
        # TODO: Step 4 - Add smooth approach rewards (after testing step 3)
        # TODO: Step 5 - Add dragging penalty (after testing step 4)

        # === END GRADUAL STRICT CONTACT SYSTEM ===


@configclass
class FixedTrajImpedanceCabinetEnvCfg_PLAY(FixedTrajImpedanceCabinetEnvCfg):
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
    id="Isaac-Open-Drawer-Franka-Fixed-Impedance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:FixedTrajImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForceVariableImpedanceCabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Fixed-Impedance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:FixedTrajImpedanceCabinetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForceVariableImpedanceCabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)
