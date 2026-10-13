# TACO Brush paper parameter registry

## AUTHOR_PUBLISHED

- `retarget_backend`: `"MINK"`
- `retarget_tasks`: `"five fingertip positions/orientations plus wrist orientation"`
- `retarget_constraints`: `"joint limits and self collision"`
- `robot_base_scene_offset_magnitude_m`: `0.6`
- `table_height_m`: `0.72`
- `taco_base_alignment`: `"fixed-offset heuristic; no AprilTag"`
- `action_abstraction`: `"floating Cartesian wrist/base plus XHand"`
- `chunk_control_steps`: `20`
- `lookahead_chunks`: `2`
- `solver_order`: `["Replay", "MPC", "RL"]`
- `rl_policy`: `"residual added to reference/base action; optimized with PPO"`
- `domain_randomization`: `false`
- `human_mimic_reward`: `false`
- `action_smoothness_reward`: `false`
- `evaluation`: `"object tracking only"`
- `contact_bonus_semantics`: `"thumb plus at least one non-thumb finger on manipulated object"`
- `lifting_reward_semantics`: `"only when vertical lifting is needed"`
- `brush_rotation_threshold_rad`: `1.2`

## UNPUBLISHED_OR_LOCAL

- `brush_position_threshold_m`: `null`
- `object_position_weight`: `null`
- `object_rotation_weight`: `null`
- `taco_contact_bonus`: `null`
- `residual_noise_schedule`: `null`
- `episode_length`: `null`
- `ppo_learning_rate`: `null`
- `ppo_epochs`: `null`
- `ppo_minibatches`: `null`
- `ppo_clip`: `null`
- `ppo_gamma`: `null`
- `ppo_gae_lambda`: `null`
- `ppo_entropy_coefficient`: `null`
- `ppo_network`: `null`
- `mpc_samples`: `null`
- `mpc_iterations`: `null`
- `mink_numeric_task_weights`: `null`

## EXTERNAL_REFERENCE_ONLY

All library/reproduction values are `REFERENCE_ONLY_NOT_AUTHORIZED`.
