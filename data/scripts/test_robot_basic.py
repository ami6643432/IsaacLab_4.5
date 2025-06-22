#!/usr/bin/env python3
"""
Basic robot loading and control test for the converted robot arm.
This script tests Step 1: Loading the USD robot and basic joint control.
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test basic robot loading and control.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

##
# Pre-defined configs
##

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene with the robot and sensors."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot - Load directly from URDF
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/amitabh/IsaacLab/data/arm_urdf/urdf/Arm.urdf",
            activate_contact_sensors=True,
            # Additional URDF options
            fix_base=True,  # Alternative to articulation_props fix_root_link
            merge_fixed_joints=False,  # Keep all joints separate
            default_drive_type="position",  # Set default actuator type
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.38),  # Position robot above ground
            rot=(1.0, 0.0, 0.0, 0.0),  # quaternion (w, x, y, z)
            joint_pos={
                "Servo1": 0.0,
                "Servo2": 0.0,
                "force_sensor_joint": 0.0,  # Ensure the force sensor joint is initialized
            },
        ),
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["Servo1", "Servo2"],
                effort_limit_sim=50.0,
                velocity_limit_sim=10.0,
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )

    # Contact sensor - adjust path based on URDF structure
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Link2",  # Direct link path from URDF
        update_period=0.0,
        history_length=6,
        debug_vis=True
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract robot from scene entities
    robot = scene["robot"]
    
    # DEBUG: Print all available scene entities
    print("\nAVAILABLE SCENE ENTITIES:")
    for key in scene.keys():
        print(f"- {key}")
    print("="*50)
    
    # print robot information
    print("\n" + "="*50)
    print("ROBOT INFORMATION")
    print("="*50)
    print(f"Robot DoF: {robot.num_joints}")
    print(f"Joint names: {robot.joint_names}")
    print(f"Robot body names: {robot.body_names}")
    print("="*50)
    
    # simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Simulation loop
    while simulation_app.is_running():
        
        # reset environment periodically
        if count % 1000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state - offset by origin
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print(f"\n[INFO] Reset at step {count}")
        
        # create simple test motion - sinusoidal joint angles
        # Joint 1: slow sine wave
        # Joint 2: faster sine wave with different phase
        joint1_target = 0.5 * math.sin(sim_time * 0.5)  # ±0.5 rad, 0.5 Hz
        joint2_target = 0.3 * math.sin(sim_time * 1.0 + math.pi/4)  # ±0.3 rad, 1 Hz, phase shift
        
        # create target joint positions tensor
        joint_pos_targets = torch.tensor(
            [[joint1_target, joint2_target]], 
            device=robot.device
        )
        
        # apply joint position targets
        robot.set_joint_position_target(joint_pos_targets)
        # write data to sim
        scene.write_data_to_sim()
        
        # print state every 100 steps
        if count % 100 == 0:
            current_joint_pos = robot.data.joint_pos[0]  # first environment
            current_joint_vel = robot.data.joint_vel[0]  # first environment
            
            # Read contact sensor data if available
            contact_info = ""
            try:
                contact_sensor = scene["contact_forces"]
                contact_data = contact_sensor.data
                if contact_data.force_matrix_w is not None and len(contact_data.force_matrix_w) > 0:
                    contact_forces = contact_data.force_matrix_w[0]  # first environment
                    if torch.any(contact_forces):  # Check if any forces are non-zero
                        force_magnitude = torch.norm(contact_forces).item()
                        contact_info = f" | Contact Force: {force_magnitude:6.3f}N"
                    else:
                        contact_info = " | No contact detected (zero force)"
                else:
                    contact_info = " | No contact data available"
            except KeyError:
                contact_info = " | Contact sensor not found in scene"
            except Exception as e:
                contact_info = f" | Error: {str(e)}"
            
            print(f"Step {count:4d} | Time: {sim_time:6.2f}s | "
                  f"Targets: [{joint1_target:6.3f}, {joint2_target:6.3f}] | "
                  f"Actual: [{current_joint_pos[0]:6.3f}, {current_joint_pos[1]:6.3f}] | "
                  f"Vel: [{current_joint_vel[0]:6.3f}, {current_joint_vel[1]:6.3f}]{contact_info}")
        
        # perform step
        sim.step()
        
        # update robot buffers
        scene.update(sim_dt)
        
        # update counters
        sim_time += sim_dt
        count += 1


def main():
    """Main function."""
    
    # initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # set main camera
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])
    
    # design scene
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # play the simulator
    sim.reset()
    
    print("\n" + "="*60)
    print("STARTING BASIC ROBOT TEST")
    print("="*60)
    print("This test will:")
    print("1. Load the converted robot USD file")
    print("2. Apply sinusoidal joint position targets")
    print("3. Display joint positions and velocities")
    print("4. Reset every 1000 steps")
    print("Press Ctrl+C to stop the simulation")
    print("="*60)
    
    # run the simulation
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user")
    finally:
        # close sim app
        simulation_app.close()
