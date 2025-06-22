import torch
import gymnasium as gym
from typing import Dict, Any

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

# Import your existing state machine
from test import OpenDrawerSm  # Import from your test.py

@configclass 
class ImpedanceRLEnvCfg(DirectRLEnvCfg):
    """Configuration for impedance parameter learning."""
    
    # Environment settings
    episode_length_s = 8.0
    decimation = 2
    action_space = 12  # 6 stiffness + 6 damping parameters
    observation_space = 30  # Force(6) + EE pose(7) + desired pose(7) + handle pose(7) + state(3)
    
    # Simulation configuration
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, 
        env_spacing=3.0, 
        replicate_physics=True
    )
    
    # Robot configuration (Franka Panda)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=True,  # Enable for force sensing
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=12, 
                solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )
    
    # Cabinet configuration
    cabinet: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.4),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    
    # Contact sensor for force feedback
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand",
        update_period=0.0,
        history_length=5,
    )
    
    # Frame transformer for pose tracking
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
            ),
        ],
    )

class ImpedanceRLEnv(DirectRLEnv):
    """RL Environment for learning impedance parameters during cabinet opening."""
    
    cfg: ImpedanceRLEnvCfg
    
    def __init__(self, cfg: ImpedanceRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        # Initialize your existing state machine
        self.open_sm = OpenDrawerSm(
            dt=self.cfg.sim.dt * self.cfg.decimation,
            num_envs=self.num_envs,
            device=self.device
        )
        
        # Impedance parameter bounds
        self.stiffness_bounds = (50.0, 2000.0)   # N/m
        self.damping_bounds = (5.0, 200.0)       # Ns/m
        
    def _setup_scene(self):
        """Set up the scene with robot and cabinet."""
        super()._setup_scene()
        
        # Add robot
        self.scene.articulations["robot"] = self.cfg.robot.replace(prim_path="/World/envs/env_.*/Robot")
        
        # Add cabinet  
        self.scene.articulations["cabinet"] = self.cfg.cabinet.replace(prim_path="/World/envs/env_.*/Cabinet")
        
        # Add sensors
        self.scene.sensors["contact_forces"] = self.cfg.contact_sensor.replace(
            prim_path="/World/envs/env_.*/Robot/panda_hand"
        )
        self.scene.sensors["ee_frame"] = self.cfg.ee_frame.replace(
            prim_path="/World/envs/env_.*/Robot/panda_hand"
        )
        
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations including force feedback and trajectory state."""
        
        # Simplified observations for initial testing
        robot = self.scene["robot"]
        cabinet = self.scene["cabinet"]
        
        # Robot joint positions and velocities
        joint_pos = robot.data.joint_pos[:, :7]  # 7 arm joints
        joint_vel = robot.data.joint_vel[:, :7]  # 7 arm joints
        
        # Cabinet state (drawer position)
        drawer_pos = cabinet.data.joint_pos[:, 0:1]  # Drawer joint
        
        # Simple force approximation (you can enhance this)
        forces = torch.zeros((self.num_envs, 6), device=self.device)
        
        # State machine state (simplified)
        sm_state = torch.zeros((self.num_envs, 6), device=self.device)
        
        obs = torch.cat([
            joint_pos,          # 7D: Joint positions
            joint_vel,          # 7D: Joint velocities  
            drawer_pos,         # 1D: Drawer position
            forces,             # 6D: Contact forces (placeholder)
            sm_state,           # 6D: State machine state
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for impedance parameter learning."""
        
        # Simple reward based on drawer opening
        cabinet = self.scene["cabinet"]
        drawer_pos = cabinet.data.joint_pos[:, 0]  # Drawer position
        
        # Reward opening the drawer
        opening_reward = torch.clamp(drawer_pos * 10.0, 0.0, 5.0)
        
        # Small penalty for high impedance values (energy efficiency)
        if hasattr(self, 'actions'):
            stiffness = self._denormalize_impedance(self.actions[:, :6], self.stiffness_bounds)
            efficiency_penalty = -torch.mean(stiffness, dim=1) * 0.0001
        else:
            efficiency_penalty = torch.zeros(self.num_envs, device=self.device)
        
        return opening_reward + efficiency_penalty
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        
        # Episode timeout
        time_outs = self.episode_length_buf >= self.max_episode_length
        
        # Task completion - drawer opened enough
        cabinet = self.scene["cabinet"]
        drawer_pos = cabinet.data.joint_pos[:, 0]
        task_completed = drawer_pos > 0.35
        
        terminated = task_completed
        truncated = time_outs
        
        return terminated, truncated
    
    def _apply_action(self, actions: torch.Tensor) -> None:
        """Apply impedance parameters and control robot."""
        
        self.actions = actions.clone()
        
        # Denormalize impedance parameters
        stiffness = self._denormalize_impedance(actions[:, :6], self.stiffness_bounds)
        damping = self._denormalize_impedance(actions[:, 6:12], self.damping_bounds)
        
        # Simple joint control (you can enhance with state machine)
        robot = self.scene["robot"]
        
        # Generate simple target positions (move towards drawer)
        current_pos = robot.data.joint_pos[:, :7]
        target_pos = current_pos + torch.randn_like(current_pos) * 0.01  # Small random movements
        
        # Apply to robot
        robot.set_joint_position_target(target_pos)
    
    def _denormalize_impedance(self, normalized_params: torch.Tensor, bounds: tuple) -> torch.Tensor:
        """Convert normalized [-1, 1] actions to impedance parameter ranges."""
        min_val, max_val = bounds
        return min_val + (normalized_params + 1.0) * 0.5 * (max_val - min_val)