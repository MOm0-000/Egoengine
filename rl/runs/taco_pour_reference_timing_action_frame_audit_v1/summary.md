# Pour reference timing × recurrent context × action frame audit v1

Read-only local semantic audit; no optimizer step or chunk commit occurred.

## Timing forward counterfactual

- shift -1: summed positive μ_y excess = 3.724216.
- shift +0: summed positive μ_y excess = 4.210286.
- shift +1: summed positive μ_y excess = 4.686592.

## Hidden isolation

- source 40: ppo_path_pre_forward=0.4981, replay_path_pre_forward=0.3082, zero_hidden=0.5373, previous_source_ppo_pre_forward=0.4007.
- source 41: ppo_path_pre_forward=0.7407, replay_path_pre_forward=0.5831, zero_hidden=0.7987, previous_source_ppo_pre_forward=0.6813.
- source 42: ppo_path_pre_forward=0.9475, replay_path_pre_forward=0.8103, zero_hidden=0.9733, previous_source_ppo_pre_forward=0.8781.
- source 43: ppo_path_pre_forward=1.2942, replay_path_pre_forward=1.2402, zero_hidden=1.4762, previous_source_ppo_pre_forward=1.2742.
- source 44: ppo_path_pre_forward=1.6861, replay_path_pre_forward=1.7456, zero_hidden=1.9129, previous_source_ppo_pre_forward=1.6269.
- source 45: ppo_path_pre_forward=1.9866, replay_path_pre_forward=2.0441, zero_hidden=2.1821, previous_source_ppo_pre_forward=1.9318.
- source 46: ppo_path_pre_forward=2.1051, replay_path_pre_forward=2.2028, zero_hidden=2.2454, previous_source_ppo_pre_forward=2.0765.
- source 47: ppo_path_pre_forward=2.1384, replay_path_pre_forward=2.2084, zero_hidden=2.2483, previous_source_ppo_pre_forward=2.1470.

## Frame geometry

- Mean +y projection (43–46): {'world_or_robot_base': -0.889421194796358, 'current_palm_site': -0.3268491948050574, 'reference_palm_site': -0.31941198765574175}.
- Best local frame: `reference_palm_site`; support-safe: False.

## Diagnostic closed loop

- `diagnostic_actor_reference_shift_-1`: 30/40, first failure={'control_interval': 50, 'endpoint': 51, 'object_roles': ['tool'], 'reason': 'tracking_boundary'}.

## Limits

- Reference shifts modify actor input only; base command and reward keep the corrected t→t+1 contract.
- Frame projections are geometric diagnostics, not contact-dynamics gradients.
- A frame candidate is not run if transforming the frozen action would leave the existing normalized support.
