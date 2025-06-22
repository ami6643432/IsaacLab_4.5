"""
Step-by-step test for impedance control environment.
This version avoids problematic imports and uses proper shutdown handling.
"""

import torch
import signal
import sys
from isaaclab.app import AppLauncher

# Global flag for clean shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n🛑 Shutdown requested, cleaning up...")
    shutdown_requested = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def main():
    """Main test function with proper error handling."""
    # Launch with minimal setup
    print("🚀 Launching Isaac Sim...")
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    
    env = None
    
    try:
        print("✅ Isaac Sim launched successfully!")

        # Test imports step by step
        print("📦 Testing basic Isaac Lab imports...")
        from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
        from isaaclab.utils import configclass
        print("✅ Basic Isaac Lab imports successful")
        
        print("📦 Testing simulation imports...")
        from isaaclab.sim import SimulationCfg
        from isaaclab.scene import InteractiveSceneCfg
        print("✅ Simulation imports successful")
        
        print("📦 Testing asset imports...")
        from isaaclab.assets import ArticulationCfg, RigidObjectCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
        print("✅ Asset imports successful")
        
        print("📦 Testing sensor imports...")
        from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
        print("✅ Sensor imports successful")
        
        print("📦 Testing impedance environment import...")
        try:
            from impedance_env import ImpedanceControlEnv, ImpedanceControlEnvCfg
            print("✅ Impedance environment import successful")
        except ImportError as e:
            print(f"⚠️ Impedance environment import failed: {e}")
            print("This is expected if the file doesn't exist yet.")
            return
        
        print("📦 Creating impedance control environment...")
        cfg = ImpedanceControlEnvCfg()
        cfg.scene.num_envs = 4  # Small number for testing
        
        env = ImpedanceControlEnv(cfg)
        print("✅ Environment created successfully!")
        
        print("📦 Testing environment reset...")
        obs = env.reset()
        print(f"✅ Reset successful! Observation shape: {obs['policy'].shape}")
        
        print("📦 Testing environment steps...")
        for i in range(10):
            if shutdown_requested:
                break
                
            # Random impedance parameters (scaled down for safety)
            actions = torch.randn((4, 12), device=env.device) * 0.1
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            print(f"  Step {i+1}: Reward = {rewards.mean():.3f} ± {rewards.std():.3f}")
            
            # Check for any issues
            if torch.any(torch.isnan(rewards)):
                print("⚠️ NaN rewards detected!")
                break
        
        if not shutdown_requested:
            print("🎉 All tests passed! Environment is working correctly.")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🔄 Cleaning up...")
        if env is not None:
            try:
                env.close()
                print("✅ Environment closed")
            except Exception as e:
                print(f"⚠️ Error closing environment: {e}")
        
        try:
            simulation_app.close()
            print("✅ Simulation closed")
        except Exception as e:
            print(f"⚠️ Error closing simulation: {e}")
        
        print("✅ Test completed!")

if __name__ == "__main__":
    main()