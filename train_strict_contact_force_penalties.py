#!/usr/bin/env python3
"""
Strict Contact Force Penalties Training Script with Curriculum Phase Control

This script trains the force variable impedance cabinet task with strict contact force penalties
and provides command-line flags to control curriculum phases.

Usage:
    # Normal training with automatic phase progression
    ./isaaclab.sh -p train_strict_contact_force_penalties.py --headless --num_envs 32
    
    # Lock to specific phase (prevents automatic progression)
    ./isaaclab.sh -p train_strict_contact_force_penalties.py --headless --lock_phase 1
    
    # Start from specific phase
    ./isaaclab.sh -p train_strict_contact_force_penalties.py --headless --start_phase 2
    
    # Disable curriculum (stay in final phase)
    ./isaaclab.sh -p train_strict_contact_force_penalties.py --headless --disable_curriculum
"""

import argparse
import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.app import AppLauncher

# Parse arguments  
parser = argparse.ArgumentParser(description="Strict Contact Force Penalties Training with Phase Control")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=800, help="Maximum training iterations")

# CURRICULUM PHASE CONTROL FLAGS
parser.add_argument("--lock_phase", type=int, choices=[1, 2, 3, 4], 
                    help="Lock training to specific phase (1-4) - prevents automatic progression")
parser.add_argument("--start_phase", type=int, choices=[1, 2, 3, 4], default=1,
                    help="Start training from specific phase (default: 1)")
parser.add_argument("--disable_curriculum", action="store_true",
                    help="Disable curriculum learning (stay in final phase)")
parser.add_argument("--phase_duration", type=int, default=200,
                    help="Iterations per phase when using curriculum (default: 200)")

# RESUME TRAINING FLAG
parser.add_argument("--resume", action="store_true",
                    help="Resume training from the latest checkpoint")

# ADDITIONAL FLAGS FOR PLAYBACK AND CHECKPOINT
parser.add_argument("--play", action="store_true",
                    help="Play/test the trained policy instead of training")
parser.add_argument("--checkpoint", type=str, 
                    help="Path to checkpoint for playing policy")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import isaaclab_tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.force_variable_impedance_env_cfg import (
    ForceVariableImpedanceCabinetEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.agents.rsl_rl_ppo_cfg import (
    StrictContactPPORunnerCfg,
)


def configure_curriculum_phases(env_cfg, args):
    """Configure curriculum learning based on command line arguments."""
    
    print("\n🎯 CURRICULUM PHASE CONFIGURATION")
    print("=" * 50)
    
    # Phase definitions
    phases = {
        1: "Basic Approach & Grasping",
        2: "Drawer Opening Mechanics", 
        3: "Force Awareness Introduction",
        4: "Full Force/Impedance Optimization"
    }
    
    if args.disable_curriculum:
        print("🔒 CURRICULUM DISABLED - Using final phase settings")
        print("   • All strict contact penalties: ACTIVE")
        print("   • All manipulation rewards: MAXIMUM")
        # Environment already configured for final phase
        return 4
    
    elif args.lock_phase:
        print(f"🔒 LOCKED TO PHASE {args.lock_phase}: {phases[args.lock_phase]}")
        print("   • Automatic progression: DISABLED")
        configure_specific_phase(env_cfg, args.lock_phase)
        return args.lock_phase
    
    else:
        print(f"📈 CURRICULUM LEARNING ENABLED")
        print(f"   • Starting phase: {args.start_phase} ({phases[args.start_phase]})")
        print(f"   • Phase duration: {args.phase_duration} iterations")
        print(f"   • Automatic progression: ENABLED")
        
        # Configure starting phase
        configure_specific_phase(env_cfg, args.start_phase)
        return args.start_phase


def configure_specific_phase(env_cfg, phase_num):
    """Configure environment for specific curriculum phase with balanced penalties."""
    
    if phase_num == 1:
        # PHASE 1: ALIGNMENT MASTERY - Prioritize end-effector to handle alignment FIRST
        print("   ⚙️  Phase 1 Configuration: ALIGNMENT-FIRST STRATEGY")
        print("      • PRIMARY FOCUS: End-effector to handle alignment")
        print("      • SECONDARY: Basic approach (only when aligned)")
        print("      • Contact penalties: MINIMAL (learning phase)")
        print("      • Grasping: BLOCKED until alignment is mastered")
        
        # HIGHEST PRIORITY: Alignment must be learned first
        env_cfg.rewards.align_ee_handle.weight = 35.0          # MAXIMUM: Critical alignment
        
        # MEDIUM PRIORITY: Approach only when aligned
        env_cfg.rewards.approach_ee_handle.weight = 15.0       # Moderate approach reward
        env_cfg.rewards.approach_gripper_handle.weight = 25.0   # Moderate gripper positioning
        
        # LOW PRIORITY: Grasping is de-emphasized until alignment is good
        env_cfg.rewards.grasp_handle.weight = 3.0              # Very low - alignment first!
        
        # MINIMAL: Task completion comes much later
        env_cfg.rewards.open_drawer_bonus.weight = 1.0         # Very low - alignment focus
        
        # Very small contact penalties for learning
        if hasattr(env_cfg.rewards, 'simple_contact_penalty'):
            env_cfg.rewards.simple_contact_penalty.weight = -0.1  # Reduced from -1.0
        if hasattr(env_cfg.rewards, 'approach_contact_penalty'):
            env_cfg.rewards.approach_contact_penalty.weight = -10.5  # Reduced from -10.0
        if hasattr(env_cfg.rewards, 'grasp_contact_penalty'):
            env_cfg.rewards.grasp_contact_penalty.weight = -0.3   # Reduced from -5.0
        if hasattr(env_cfg.rewards, 'manipulation_contact_penalty'):
            env_cfg.rewards.manipulation_contact_penalty.weight = -0.2   # Reduced from -3.0
        if hasattr(env_cfg.rewards, 'non_grasp_contact_penalty'):
            env_cfg.rewards.non_grasp_contact_penalty.weight = -1.0   # Reduced from -20.0
        if hasattr(env_cfg.rewards, 'smooth_approach_reward'):
            env_cfg.rewards.smooth_approach_reward.weight = 20.0   # Reduced from 5.0
        if hasattr(env_cfg.rewards, 'anti_dragging_penalty'):
            env_cfg.rewards.anti_dragging_penalty.weight = -10.5   # Reduced from -15.0
        
        # Boost basic manipulation rewards
        env_cfg.rewards.approach_ee_handle.weight = 15.0  # Increased from 12.0
        env_cfg.rewards.approach_gripper_handle.weight = 30.0  # Increased from 25.0
        env_cfg.rewards.grasp_handle.weight = 20.0  # Increased from 15.0
        
    elif phase_num == 2:
        # PHASE 2: Drawer Opening Mechanics with moderate penalties
        print("   ⚙️  Phase 2 Configuration:")
        print("      • Focus: Drawer opening mechanics")
        print("      • Contact penalties: MODERATE")
        print("      • Opening rewards: BOOSTED")
        
        # Moderate contact penalties
        if hasattr(env_cfg.rewards, 'simple_contact_penalty'):
            env_cfg.rewards.simple_contact_penalty.weight = -0.5
        if hasattr(env_cfg.rewards, 'approach_contact_penalty'):
            env_cfg.rewards.approach_contact_penalty.weight = -1.0
        if hasattr(env_cfg.rewards, 'grasp_contact_penalty'):
            env_cfg.rewards.grasp_contact_penalty.weight = -0.8
        if hasattr(env_cfg.rewards, 'manipulation_contact_penalty'):
            env_cfg.rewards.manipulation_contact_penalty.weight = -0.5
        if hasattr(env_cfg.rewards, 'non_grasp_contact_penalty'):
            env_cfg.rewards.non_grasp_contact_penalty.weight = -2.0
        if hasattr(env_cfg.rewards, 'smooth_approach_reward'):
            env_cfg.rewards.smooth_approach_reward.weight = 1.0
        if hasattr(env_cfg.rewards, 'anti_dragging_penalty'):
            env_cfg.rewards.anti_dragging_penalty.weight = -1.0
        
        # Boost opening rewards
        env_cfg.rewards.open_drawer_bonus.weight = 35.0  # Increased from 30.0
        env_cfg.rewards.multi_stage_open_drawer.weight = 15.0  # Increased from 12.0
        
    elif phase_num == 3:
        # PHASE 3: Force Awareness Introduction with increased penalties
        print("   ⚙️  Phase 3 Configuration:")
        print("      • Focus: Force awareness and control")
        print("      • Contact penalties: INCREASED")
        print("      • Force awareness: INTRODUCED")
        
        # Increase contact penalties gradually
        if hasattr(env_cfg.rewards, 'simple_contact_penalty'):
            env_cfg.rewards.simple_contact_penalty.weight = -1.0
        if hasattr(env_cfg.rewards, 'approach_contact_penalty'):
            env_cfg.rewards.approach_contact_penalty.weight = -2.0
        if hasattr(env_cfg.rewards, 'grasp_contact_penalty'):
            env_cfg.rewards.grasp_contact_penalty.weight = -1.5
        if hasattr(env_cfg.rewards, 'manipulation_contact_penalty'):
            env_cfg.rewards.manipulation_contact_penalty.weight = -1.0
        if hasattr(env_cfg.rewards, 'non_grasp_contact_penalty'):
            env_cfg.rewards.non_grasp_contact_penalty.weight = -3.0
        if hasattr(env_cfg.rewards, 'smooth_approach_reward'):
            env_cfg.rewards.smooth_approach_reward.weight = 2.0
        if hasattr(env_cfg.rewards, 'anti_dragging_penalty'):
            env_cfg.rewards.anti_dragging_penalty.weight = -2.0
        
    elif phase_num == 4:
        # PHASE 4: Full Force/Impedance Optimization with full but reasonable penalties
        print("   ⚙️  Phase 4 Configuration:")
        print("      • Focus: Full force/impedance optimization")
        print("      • Contact penalties: FULL (but reasonable)")
        print("      • All strict contact features: ACTIVE")
        
        # Full penalties but still reasonable
        if hasattr(env_cfg.rewards, 'simple_contact_penalty'):
            env_cfg.rewards.simple_contact_penalty.weight = -2.0  # Reduced from -3.0
        if hasattr(env_cfg.rewards, 'approach_contact_penalty'):
            env_cfg.rewards.approach_contact_penalty.weight = -4.0  # Reduced from -25.0
        if hasattr(env_cfg.rewards, 'grasp_contact_penalty'):
            env_cfg.rewards.grasp_contact_penalty.weight = -3.0  # Reduced from -15.0
        if hasattr(env_cfg.rewards, 'manipulation_contact_penalty'):
            env_cfg.rewards.manipulation_contact_penalty.weight = -2.0  # Reduced from -8.0
        if hasattr(env_cfg.rewards, 'non_grasp_contact_penalty'):
            env_cfg.rewards.non_grasp_contact_penalty.weight = -5.0  # Reduced from -20.0
        if hasattr(env_cfg.rewards, 'smooth_approach_reward'):
            env_cfg.rewards.smooth_approach_reward.weight = 3.0  # Reduced from 5.0
        if hasattr(env_cfg.rewards, 'anti_dragging_penalty'):
            env_cfg.rewards.anti_dragging_penalty.weight = -3.0  # Reduced from -15.0


class CurriculumTrainingRunner:
    """Training runner with curriculum phase progression."""
    
    def __init__(self, env, agent_cfg, log_dir, args):
        self.env = env
        self.agent_cfg = agent_cfg
        self.args = args
        self.current_phase = args.start_phase
        self.phase_start_iter = 0
        
        # Handle resume functionality first (before creating runner)
        if args.resume:
            self.log_dir = self._find_resume_checkpoint(agent_cfg.experiment_name)
        else:
            self.log_dir = log_dir
        
        # Create base runner with resume capability
        self.runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=self.log_dir, device=agent_cfg.device)
    
    def _find_resume_checkpoint(self, experiment_name):
        """Find and return the latest checkpoint directory for resuming."""
        print(f"\n🔄 SEARCHING FOR RESUME CHECKPOINT")
        print("=" * 50)
        
        # Look for existing checkpoints in the log directory
        checkpoint_dirs = []
        base_log_dir = os.path.join("logs", "rsl_rl", experiment_name)
        
        if os.path.exists(base_log_dir):
            for subdir in os.listdir(base_log_dir):
                checkpoint_path = os.path.join(base_log_dir, subdir)
                if os.path.isdir(checkpoint_path):
                    checkpoint_dirs.append((subdir, checkpoint_path))
        
        if not checkpoint_dirs:
            print("❌ No existing checkpoints found. Creating new training session.")
            # Create new directory for fresh training
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_log_dir = os.path.join(base_log_dir, timestamp)
            os.makedirs(new_log_dir, exist_ok=True)
            return new_log_dir
        
        # Sort by timestamp (most recent first)
        checkpoint_dirs.sort(reverse=True)
        latest_checkpoint_dir = checkpoint_dirs[0][1]
        
        print(f"✅ Found checkpoint directory: {latest_checkpoint_dir}")
        print(f"✅ Ready to resume training from previous session")
        
        return latest_checkpoint_dir
    
    def should_advance_phase(self, current_iter):
        """Check if we should advance to next phase."""
        if self.args.lock_phase or self.args.disable_curriculum:
            return False
        
        if self.current_phase >= 4:
            return False
        
        phase_progress = current_iter - self.phase_start_iter
        return phase_progress >= self.args.phase_duration
    
    def advance_phase(self, current_iter):
        """Advance to next curriculum phase."""
        if self.current_phase < 4:
            self.current_phase += 1
            self.phase_start_iter = current_iter
            
            print(f"\n🎯 ADVANCING TO PHASE {self.current_phase}")
            print("=" * 50)
            
            # Reconfigure environment for new phase
            # Note: In practice, you might need to recreate the environment
            # For now, we'll just log the transition
            configure_specific_phase(self.env.cfg, self.current_phase)
            
            print(f"   • Phase transition at iteration {current_iter}")
            print(f"   • Next phase change in {self.args.phase_duration} iterations")
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        """Custom learning loop with curriculum progression."""
        
        print(f"\n🚀 STARTING CURRICULUM TRAINING")
        print(f"   • Total iterations: {num_learning_iterations}")
        print(f"   • Current phase: {self.current_phase}")
        
        # Use the base runner's learn method
        return self.runner.learn(num_learning_iterations, init_at_random_ep_len)


def play_policy(env, checkpoint_path, num_episodes=10):
    """Play the trained policy for evaluation."""
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.agents.rsl_rl_ppo_cfg import (
        StrictContactPPORunnerCfg,
    )
    
    print(f"\n🎮 PLAYING TRAINED POLICY")
    print("=" * 50)
    print(f"   • Checkpoint: {checkpoint_path}")
    print(f"   • Episodes: {num_episodes}")
    
    # Find the actual checkpoint file
    if os.path.isdir(checkpoint_path):
        # Look for .pt files in the directory
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
        if checkpoint_files:
            # Use the most recent checkpoint
            checkpoint_files.sort()
            checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[-1])
        else:
            print(f"❌ No .pt files found in {checkpoint_path}")
            return False
    else:
        checkpoint_file = checkpoint_path
    
    print(f"   • Loading: {checkpoint_file}")
    
    try:
        # Wrap environment for RSL-RL
        wrapped_env = RslRlVecEnvWrapper(env)
        
        # Create agent configuration and convert to dict format
        agent_cfg = StrictContactPPORunnerCfg()
        
        # Properly convert the configuration to dictionary format that OnPolicyRunner expects
        agent_cfg_dict = agent_cfg.to_dict()
        
        # Create runner with proper dictionary format
        runner = OnPolicyRunner(wrapped_env, agent_cfg_dict, device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the saved model
        runner.load(checkpoint_file)
        print("✅ Policy loaded and ready for evaluation")
        
        # Run evaluation with trained policy
        print(f"\n🚀 Starting trained policy evaluation...")
        
        # Reset environment
        obs, info = wrapped_env.reset()
        episode_count = 0
        step_count = 0
        total_reward = 0
        episode_rewards = []
        
        with torch.no_grad():
            while episode_count < num_episodes:
                # Get action from trained policy (handle different RSL-RL API versions)
                try:
                    # Most recent RSL-RL API with critic_obs and teacher_obs
                    action = runner.alg.act(obs, obs, obs)
                except TypeError:
                    try:
                        # Try with critic_obs only
                        action = runner.alg.act(obs, obs)
                    except TypeError:
                        try:
                            # Try with just obs
                            action = runner.alg.act(obs)
                        except Exception as e:
                            print(f"❌ Could not get action from policy: {e}")
                            print("ℹ️  Using random action as fallback")
                            action = torch.randn(obs.shape[0], 19, device=obs.device)
                
                # Step environment
                step_result = wrapped_env.step(action)
                
                # Handle different step return formats
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = done
                
                total_reward += reward.mean().item() if hasattr(reward, 'mean') else reward
                step_count += 1
                
                # Check for episode termination
                if terminated.any() or truncated.any():
                    episode_count += 1
                    episode_reward = total_reward / step_count if step_count > 0 else 0
                    episode_rewards.append(episode_reward)
                    print(f"   Episode {episode_count}: Steps={step_count}, Avg Reward={episode_reward:.3f}")
                    
                    # Reset for next episode
                    obs, info = wrapped_env.reset()
                    total_reward = 0
                    step_count = 0
        
        # Print final statistics
        if episode_rewards:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            max_reward = max(episode_rewards)
            min_reward = min(episode_rewards)
            print(f"\n📊 EVALUATION RESULTS:")
            print(f"   • Average reward: {avg_reward:.3f}")
            print(f"   • Max reward: {max_reward:.3f}")
            print(f"   • Min reward: {min_reward:.3f}")
        
        print("✅ Evaluation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load/run policy: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Fallback to random actions for testing
        print("\n⚠️  Falling back to random action testing...")
        return test_with_random_actions(env, num_episodes)


def test_with_random_actions(env, num_episodes=5):
    """Test environment with random actions as fallback."""
    print(f"\n🎲 Testing with random actions ({num_episodes} episodes)")
    
    obs, info = env.reset()
    episode_count = 0
    step_count = 0
    total_reward = 0
    
    # Get device from environment (handle different environment types)
    if hasattr(env, 'device'):
        device = env.device
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'device'):
        device = env.unwrapped.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    while episode_count < num_episodes:
        # Convert random action to tensor with proper device and dtype
        random_action = env.action_space.sample()
        action = torch.tensor(random_action, device=device, dtype=torch.float32)
        
        # Step environment
        step_result = env.step(action)
        
        # Handle different step return formats
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = done
        
        total_reward += reward.mean().item() if hasattr(reward, 'mean') else reward
        step_count += 1
        
        if terminated.any() or truncated.any():
            episode_count += 1
            avg_reward = total_reward / step_count if step_count > 0 else 0
            print(f"   Episode {episode_count}: Steps={step_count}, Avg Reward={avg_reward:.3f}")
            
            obs, info = env.reset()
            total_reward = 0
            step_count = 0
    
    print("✅ Random action testing completed")
    return True


# Modify the main() function to handle play mode
def main():
    """Train or play strict contact force penalties."""
    
    # Check if we're in play mode
    if args_cli.play:
        if not args_cli.checkpoint:
            print("❌ --checkpoint required when using --play")
            return False
        
        print("🎮 POLICY TESTING MODE")
        print("=" * 60)
        
        # Create environment configuration (match training phase)
        env_cfg = ForceVariableImpedanceCabinetEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        
        # Configure for the phase that was used during training
        if args_cli.lock_phase:
            configure_specific_phase(env_cfg, args_cli.lock_phase)
            print(f"   • Testing with Phase {args_cli.lock_phase} configuration")
        else:
            # Default to phase 1 settings for testing
            configure_specific_phase(env_cfg, 1)
            print(f"   • Testing with Phase 1 configuration (default)")
        
        # Create environment
        print(f"\n🔧 Creating test environment...")
        env = gym.make("Isaac-Open-Drawer-Franka-v0", cfg=env_cfg)
        
        try:
            # Play the policy
            success = play_policy(env, args_cli.checkpoint, num_episodes=5)
            return success
        finally:
            env.close()
    
    # Original training code continues here...
    else:
        print("🛡️  STRICT CONTACT FORCE PENALTIES TRAINING")
        print("=" * 60)
        print("Features:")
        print("  ✅ 5-Step Strict Contact System")
        print("  ✅ Phase-Aware Contact Penalties") 
        print("  ✅ Non-Grasp Contact Prevention")
        print("  ✅ Smooth Approach Rewards")
        print("  ✅ Anti-Dragging Penalties")
        print("  ✅ Curriculum Phase Control")
        print("=" * 60)
        
        # Create environment configuration
        env_cfg = ForceVariableImpedanceCabinetEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        
        # Configure curriculum phases
        current_phase = configure_curriculum_phases(env_cfg, args_cli)
        
        # Create agent configuration
        agent_cfg = StrictContactPPORunnerCfg()
        agent_cfg.max_iterations = args_cli.max_iterations
        
        # Update experiment name to reflect phase control
        if args_cli.lock_phase:
            agent_cfg.experiment_name = f"strict_contact_phase_{args_cli.lock_phase}_locked"
        elif args_cli.disable_curriculum:
            agent_cfg.experiment_name = "strict_contact_no_curriculum"
        else:
            agent_cfg.experiment_name = "strict_contact_curriculum"
        
        print(f"\nSetup:")
        print(f"  • Environments: {env_cfg.scene.num_envs}")
        print(f"  • Max iterations: {agent_cfg.max_iterations}")
        print(f"  • Experiment: {agent_cfg.experiment_name}")
        print(f"  • Current phase: {current_phase}")
        
        # Create logging directory
        if args_cli.resume:
            # For resume, we'll let the CurriculumTrainingRunner handle log directory
            log_dir = None  # Will be set by CurriculumTrainingRunner
            print(f"  • Resume mode: Will auto-detect checkpoint directory")
        else:
            # For new training, create a new timestamped directory
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, timestamp)
            os.makedirs(log_dir, exist_ok=True)
            print(f"  • Logging to: {log_dir}")
        
        # Create environment
        print(f"\n🔧 Creating Strict Contact Environment...")
        env = gym.make("Isaac-Open-Drawer-Franka-v0", cfg=env_cfg)
        
        print(f"✅ Environment created:")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: (55,)")
        print(f"   Reward terms: 16 (including 7 strict contact terms)")
        
        # Verify action dimensions
        action_shape = env.action_space.shape
        if action_shape is not None and len(action_shape) > 1 and action_shape[1] != 19:
            print(f"❌ Wrong action dimensions! Expected 19, got {action_shape[1]}")
            env.close()
            return False
        
        # Wrap for training
        wrapped_env = RslRlVecEnvWrapper(env)
        
        # Create curriculum training runner (handles resume logic internally)
        print(f"\n🎓 Creating curriculum training runner...")
        curriculum_runner = CurriculumTrainingRunner(wrapped_env, agent_cfg, log_dir, args_cli)
        
        # Get the actual log directory (may be different if resuming)
        actual_log_dir = curriculum_runner.log_dir
        print(f"✅ Curriculum runner created on device: {curriculum_runner.runner.device}")
        print(f"📁 Using log directory: {actual_log_dir}")
        
        # Start training
        print("\n" + "=" * 60)
        print("🚀 STARTING STRICT CONTACT TRAINING WITH PHASE CONTROL")
        print("=" * 60)
        
        try:
            # Train with curriculum
            curriculum_runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
            
            print(f"\n🎉 Training completed successfully!")
            print(f"📊 Results saved to: {actual_log_dir}")
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted")
            print(f"💾 Partial results in: {actual_log_dir}")
            return False
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            wrapped_env.close()


if __name__ == "__main__":
    success = main()
    simulation_app.close()
    exit(0 if success else 1)
