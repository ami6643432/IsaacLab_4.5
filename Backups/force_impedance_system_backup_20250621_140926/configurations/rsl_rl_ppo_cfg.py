# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 400
    save_interval = 50
    experiment_name = "franka_open_drawer"


@configclass 
class ForceVariableImpedanceCabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for force-based variable impedance cabinet manipulation with curriculum learning."""
    
    # Environment and training settings
    num_steps_per_env = 96
    max_iterations = 1200  # Extended for full curriculum learning (4 phases + fine-tuning)
    save_interval = 50
    experiment_name = "franka_force_impedance_curriculum"
    
    # Network architecture optimized for impedance control
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # Higher exploration for early phases
        actor_hidden_dims=[512, 256, 128],  # Larger network for complex impedance control
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    # PPO algorithm settings tuned for force control
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=2e-3,  # Higher entropy for impedance exploration
        num_learning_epochs=8,  # More epochs for stable impedance learning
        num_mini_batches=4,
        learning_rate=3e-4,  # Conservative learning rate for stable force control
        schedule="adaptive",
        gamma=0.995,  # Slightly higher discount for long-term impedance optimization
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
    
    # Advanced settings
    empirical_normalization = True  # Important for force/impedance normalization
    resume = False  # Set to True to resume from checkpoint


@configclass
class TrajectoryNetworkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for trajectory network training (Phase 1 of dual network approach)."""
    
    # Training settings optimized for trajectory learning
    num_steps_per_env = 96
    max_iterations = 800  # Focused training for trajectory network
    save_interval = 50
    experiment_name = "trajectory_network_phase1"
    
    # Network architecture optimized for manipulation learning
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,  # Moderate exploration for precise manipulation
        actor_hidden_dims=[256, 128, 64],  # Smaller network for 7D action space
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    
    # PPO algorithm settings tuned for trajectory control
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Moderate entropy for exploration
        num_learning_epochs=6,  # Sufficient for trajectory learning
        num_mini_batches=4,
        learning_rate=5e-4,  # Slightly higher LR for faster convergence
        schedule="adaptive",
        gamma=0.99,  # Standard discount factor
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
    
    # Training settings
    empirical_normalization = True
    resume = False
