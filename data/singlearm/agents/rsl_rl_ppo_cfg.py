# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class VariableImpedanceCabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO agent training variable impedance cabinet manipulation."""
    
    num_steps_per_env = 96
    max_iterations = 2000  # Longer training for impedance learning
    save_interval = 100
    experiment_name = "variable_impedance_cabinet"
    empirical_normalization = False
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Larger networks for impedance parameter learning
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128], 
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1e-3,  # Encourage exploration of impedance space
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,  # Slightly lower LR for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class FixedTrajImpedancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO agent training fixed trajectory impedance control."""
    
    num_steps_per_env = 96
    max_iterations = 1500  # Focused training on impedance only
    save_interval = 100
    experiment_name = "fixed_traj_impedance"
    empirical_normalization = True
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,  # Lower noise for precise impedance control
        # Smaller networks since action space is only 12D
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64], 
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Moderate exploration for impedance parameters
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class StrictContactPenaltiesPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO agent training strict contact force penalties approach."""
    
    num_steps_per_env = 96
    max_iterations = 2000  # Extended for progressive training
    save_interval = 100
    experiment_name = "strict_contact_penalties"
    empirical_normalization = True
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # Higher exploration for contact awareness
        # Larger networks for complex force-impedance control
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128], 
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=2e-3,  # Higher entropy for force exploration
        num_learning_epochs=8,  # More epochs for stable force control
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.995,  # Slightly higher discount for long-term force optimization
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
