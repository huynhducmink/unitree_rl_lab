# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from isaaclab.utils import configclass
# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


# @configclass
# class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 50000
#     save_interval = 100
#     experiment_name = ""  # same as task name
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )

# --- PATCH: rsl_rl_ppo_cfg.py ---

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32           # a bit longer rollouts
    max_iterations = 60000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False  # keep off for raw pixels unless you normalize externally

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[384, 256, 128],
        critic_hidden_dims=[384, 256, 128],
        activation="elu",
        # If available in your fork, uncomment a tiny CNN:
        # actor_cnn=[("conv", 16, 8, 4), ("conv", 32, 4, 2), ("flatten",), ("mlp", [256])],
        # critic_cnn=[("conv", 16, 8, 4), ("conv", 32, 4, 2), ("flatten",), ("mlp", [256])],
        # pixel_keys=["rgb_head"],   # match obs term name if encoder supported
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,          # slightly lower
        num_learning_epochs=6,       # a touch more optimization per update
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,                 # a bit longer horizon helps sparse contact
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
