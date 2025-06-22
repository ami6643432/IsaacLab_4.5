import torch
from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from impedance_rl_env import ImpedanceRLEnv, ImpedanceRLEnvCfg

def main():
    """Train impedance parameter learning."""
    
    # Environment configuration
    env_cfg = ImpedanceRLEnvCfg()
    env_cfg.scene.num_envs = 64
    env_cfg.sim.device = "cuda:0"
    
    # Create environment
    env = ImpedanceRLEnv(cfg=env_cfg)
    
    print(f"Action space: {env.action_managers.action_space}")
    print(f"Observation space: {env.observation_managers.observation_space}")
    
    # Training loop
    env.reset()
    
    for step in range(10000):
        # Random actions for testing (replace with your RL algorithm)
        actions = torch.rand((env.num_envs, 12), device=env.device) * 2.0 - 1.0  # [-1, 1]
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if step % 100 == 0:
            print(f"Step {step}: Mean reward = {rewards.mean():.3f}")
            print(f"Force violations: {info.get('force_violations', 0)}")
            print(f"Success rate: {info.get('success_rate', 0):.3f}")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()