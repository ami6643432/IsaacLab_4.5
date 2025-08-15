import gymnasium as gym
import isaaclab_tasks.direct.cartpole  # This registers the envs

env = gym.make("Isaac-Cartpole-Direct-v0")
print("Observation space:", env.observation_space)

obs, _ = env.reset()
print("First observation shape:", obs.shape)
print("First observation:", obs)
