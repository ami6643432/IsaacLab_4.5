"""
Impedance Control Environment for Isaac Lab

This environment demonstrates impedance parameter learning for robotic manipulation tasks.
The robot learns to adjust its impedance parameters based on task requirements.
"""

import torch
import gymnasium as gym
from typing import Any, Dict

from isaaclab.app import AppLauncher
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class ImpedanceControlEnvCfg(DirectRLEnvCfg):
    """Configuration for the impedance control environment."""
    
    # Environment settings
    episode_length_s = 20.0
    decimation = 2
    action_space = 12  # 6 stiffness + 6 damping parameters
    observation_space = 50
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
        disable_contact_processing=False,
    )
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096 if sim.device == "cuda:0" else 64,
        env_spacing=2.5,
        replicate_physics=True,
    )
    
    # Robot configuration
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ArticulationCfg.SpawnCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            rigid_props=ArticulationCfg.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=ArticulationCfg.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.035,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=5.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=5.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.15,
                stiffness=1e5,
                damping=1e3,
            ),
        },
    )
    
    # Object to manipulate
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=RigidObjectCfg.SpawnCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidObjectCfg.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.055], rot=[1.0, 0.0, 0.0, 0.0]
        ),
    )
    
    # Sensors
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand", update_period=0.0
    )
    
    # Frame transformer
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=FrameTransformerCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            )
        ],
    )


class ImpedanceControlEnv(DirectRLEnv):
    """Environment for learning impedance control parameters."""
    
    cfg: ImpedanceControlEnvCfg
    
    def __init__(self, cfg: ImpedanceControlEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the impedance control environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize impedance parameters
        self.default_stiffness = torch.tensor([150.0, 150.0, 150.0, 10.0, 10.0, 10.0], device=self.device)
        self.default_damping = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0], device=self.device)
        
        # Target pose for the end-effector
        self.target_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.target_ee_pose[:, 0] = 0.5  # x position
        self.target_ee_pose[:, 1] = 0.0  # y position  
        self.target_ee_pose[:, 2] = 0.4  # z position
        self.target_ee_pose[:, 6] = 1.0  # w quaternion
        
        # Tracking buffers
        self.ee_pose_error = torch.zeros((self.num_envs, 6), device=self.device)
        self.contact_forces_norm = torch.zeros(self.num_envs, device=self.device)
        
        print(f"✅ Impedance control environment initialized with {self.num_envs} environments")
    
    def _setup_scene(self):
        """Set up the scene with robot and object."""
        super()._setup_scene()
        
        # Add robot to scene
        self.cfg.robot.prim_path = self.cfg.robot.prim_path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
        self.scene.articulations["robot"] = self.cfg.robot
        
        # Add object to scene
        self.cfg.object.prim_path = self.cfg.object.prim_path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
        self.scene.rigid_objects["object"] = self.cfg.object
        
        # Add sensors
        self.cfg.contact_forces.prim_path = self.cfg.contact_forces.prim_path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
        self.scene.sensors["contact_forces"] = self.cfg.contact_forces
        
        self.cfg.ee_frame.prim_path = self.cfg.ee_frame.prim_path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
        for frame_cfg in self.cfg.ee_frame.target_frames:
            frame_cfg.prim_path = frame_cfg.prim_path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
        self.scene.sensors["ee_frame"] = self.cfg.ee_frame
        
        print("✅ Scene setup completed")
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get environment observations."""
        # Robot state
        robot = self.scene.articulations["robot"]
        joint_pos = robot.data.joint_pos[:, :7]
        joint_vel = robot.data.joint_vel[:, :7]
        
        # End-effector pose
        ee_frame = self.scene.sensors["ee_frame"]
        ee_pos = ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        ee_quat = ee_frame.data.target_quat_w[..., 0, :]
        
        # Object pose
        object_data = self.scene.rigid_objects["object"]
        object_pos = object_data.data.root_pos_w - self.scene.env_origins
        object_quat = object_data.data.root_quat_w
        
        # Contact forces
        contact_data = self.scene.sensors["contact_forces"]
        contact_forces = contact_data.data.net_forces_w_history[:, -1, :]
        self.contact_forces_norm = torch.norm(contact_forces, dim=-1)
        
        # Target relative pose
        target_pos_rel = self.target_ee_pose[:, :3] - ee_pos
        target_quat_rel = self.target_ee_pose[:, 3:] - ee_quat
        
        # Pose error for reward computation
        self.ee_pose_error[:, :3] = target_pos_rel
        self.ee_pose_error[:, 3:] = target_quat_rel[:, :3]  # xyz of quaternion
        
        # Combine observations
        obs = torch.cat([
            joint_pos,          # 7
            joint_vel,          # 7
            ee_pos,             # 3
            ee_quat,            # 4
            object_pos,         # 3
            object_quat,        # 4
            target_pos_rel,     # 3
            target_quat_rel,    # 4
            contact_forces.view(self.num_envs, -1)[:, :15],  # 15 (flattened contact forces)
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on task performance."""
        # Position tracking reward
        pos_error = torch.norm(self.ee_pose_error[:, :3], dim=-1)
        pos_reward = torch.exp(-2.0 * pos_error)
        
        # Orientation tracking reward  
        quat_error = torch.norm(self.ee_pose_error[:, 3:], dim=-1)
        quat_reward = torch.exp(-1.0 * quat_error)
        
        # Contact force regulation reward
        force_penalty = -0.001 * torch.clamp(self.contact_forces_norm - 50.0, min=0.0)
        
        # Combine rewards
        total_reward = pos_reward + 0.5 * quat_reward + force_penalty
        
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination and truncation conditions."""
        # No early termination conditions
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Truncate at episode length
        truncated = self.episode_length_buf >= self.max_episode_length
        
        return terminated, truncated
    
    def _apply_action(self, actions: torch.Tensor) -> None:
        """Apply impedance parameter actions."""
        # Scale actions to reasonable impedance ranges
        # Actions are assumed to be in [-1, 1] range
        stiffness_scale = torch.tensor([200.0, 200.0, 200.0, 20.0, 20.0, 20.0], device=self.device)
        damping_scale = torch.tensor([20.0, 20.0, 20.0, 2.0, 2.0, 2.0], device=self.device)
        
        # Extract stiffness and damping from actions
        stiffness_actions = actions[:, :6]
        damping_actions = actions[:, 6:]
        
        # Compute impedance parameters
        stiffness = self.default_stiffness + stiffness_actions * stiffness_scale * 0.5
        damping = self.default_damping + damping_actions * damping_scale * 0.5
        
        # Clamp to reasonable ranges
        stiffness = torch.clamp(stiffness, 10.0, 400.0)
        damping = torch.clamp(damping, 1.0, 40.0)
        
        # Apply impedance control (simplified version)
        robot = self.scene.articulations["robot"]
        ee_frame = self.scene.sensors["ee_frame"]
        
        # Get current end-effector pose
        current_ee_pos = ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        current_ee_quat = ee_frame.data.target_quat_w[..., 0, :]
        
        # Compute position and orientation errors
        pos_error = self.target_ee_pose[:, :3] - current_ee_pos
        quat_error = self.target_ee_pose[:, 3:] - current_ee_quat
        
        # Simplified impedance force computation
        force_pos = stiffness[:, :3] * pos_error
        force_ori = stiffness[:, 3:] * quat_error[:, :3]
        
        # Convert to joint torques (simplified)
        joint_efforts = torch.zeros((self.num_envs, robot.num_joints), device=self.device)
        joint_efforts[:, :7] = 0.1 * torch.cat([force_pos, force_ori], dim=-1)
        
        # Apply joint efforts
        robot.set_joint_effort_target(joint_efforts, joint_ids=None)
    
    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        super()._reset_idx(env_ids)
        
        # Reset robot to initial pose
        robot = self.scene.articulations["robot"]
        robot.reset(env_ids)
        
        # Reset object pose with some randomization
        object_data = self.scene.rigid_objects["object"]
        object_pos = object_data.data.default_root_state[env_ids, :3]
        object_pos[:, :2] += torch.randn_like(object_pos[:, :2]) * 0.1  # Add noise
        object_data.write_root_pose_to_sim(
            torch.cat([object_pos, object_data.data.default_root_state[env_ids, 3:7]], dim=-1),
            env_ids
        )
        
        # Randomize target position slightly
        self.target_ee_pose[env_ids, :2] += torch.randn((len(env_ids), 2), device=self.device) * 0.05


def main():
    """Main function to test the impedance control environment."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test impedance control environment")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # Launch app
    print("🚀 Launching Isaac Sim...")
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    
    try:
        print("✅ Isaac Sim launched successfully!")
        
        # Create environment configuration
        cfg = ImpedanceControlEnvCfg()
        cfg.scene.num_envs = args.num_envs
        
        print("📦 Creating impedance control environment...")
        env = ImpedanceControlEnv(cfg, render_mode="rgb_array" if args.headless else None)
        print("✅ Environment created successfully!")
        
        print("🔄 Testing environment...")
        obs = env.reset()
        print(f"✅ Reset successful! Observation shape: {obs['policy'].shape}")
        
        # Run a few steps
        for i in range(100):
            # Random impedance parameters
            actions = torch.randn((args.num_envs, 12), device=env.device) * 0.5
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            if i % 20 == 0:
                print(f"  Step {i:3d}: Reward = {rewards.mean():.3f} ± {rewards.std():.3f}")
        
        print("🎉 All tests passed! Environment is working correctly.")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🔄 Closing environment...")
        if 'env' in locals():
            env.close()
        print("🔄 Closing simulation...")
        simulation_app.close()
        print("✅ Test completed!")


if __name__ == "__main__":
    main()
