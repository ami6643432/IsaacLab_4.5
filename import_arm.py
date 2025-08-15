from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

usd_path = "/home/amitabh/IsaacLab_4.5/data/arm_urdf/urdf/Arm/Arm.usd"

@configclass
class ArmSceneCfg(InteractiveSceneCfg):
    num_envs = 1
    env_spacing = 2.0
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=1.0,
                velocity_limit=1.0,
                stiffness=0.0,
                damping=0.0,
            )
        }
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Link2",
        history_length=10,
        update_period=0.0,
    )

# Create the simulation context (use dt instead of physics_dt/rendering_dt)
sim = sim_utils.SimulationContext()

# Now create the scene using the config class
scene = InteractiveScene(cfg=ArmSceneCfg())

for _ in range(100):
    scene.step()

simulation_app.close()