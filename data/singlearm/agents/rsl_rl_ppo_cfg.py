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
