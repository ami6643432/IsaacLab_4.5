#!/usr/bin/env python3
"""
Basic robot loading and control test for the converted robot arm.
This script tests Step 1: Loading the USD robot and basic joint control.
"""

"""Launch Isaac Sim Simulator first."""

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
import isaacsim.core.utils.prims as prim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # Lights
    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)

    # Create robot origin
    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # Robot arm configuration - Fixed base in the air
    robot_cfg = ArticulationCfg(
        prim_path="/World/Origin1/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/data/arm_urdf/arm_minimal_contact_reporter.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,  # This fixes the base link in place
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.38),  # Position robot 1.5m above ground
            rot=(1.0, 0.0, 0.0, 0.0),  # quaternion (w, x, y, z)
            joint_pos={
                "Servo1": 0.0,
                "Servo2": 0.0,
            },
        ),
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["Servo1", "Servo2"],
                effort_limit=50.0,
                velocity_limit=10.0,
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )
    
    # Create robot articulation
    robot = Articulation(cfg=robot_cfg)

    # return the scene information
    scene_entities = {"robot": robot}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract robot from scene entities
    robot = entities["robot"]
    
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
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
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
        robot.write_data_to_sim()
        
        # print state every 100 steps
        if count % 100 == 0:
            current_joint_pos = robot.data.joint_pos[0]  # first environment
            current_joint_vel = robot.data.joint_vel[0]  # first environment
            print(f"Step {count:4d} | Time: {sim_time:6.2f}s | "
                  f"Targets: [{joint1_target:6.3f}, {joint2_target:6.3f}] | "
                  f"Actual: [{current_joint_pos[0]:6.3f}, {current_joint_pos[1]:6.3f}] | "
                  f"Vel: [{current_joint_vel[0]:6.3f}, {current_joint_vel[1]:6.3f}]")
        
        # perform step
        sim.step()
        
        # update robot buffers
        robot.update(sim_dt)
        
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
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
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
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user")
    finally:
        # close sim app
        simulation_app.close()
