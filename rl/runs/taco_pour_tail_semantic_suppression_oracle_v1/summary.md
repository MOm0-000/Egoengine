# Tail semantic suppression oracle

- selected modes at sources 44--55: ['zero_complete_right_wrist', 'zero_complete_right_wrist', 'zero_right_wrist_translation', 'zero_right_wrist_translation', 'zero_full_36d_residual', 'zero_right_wrist_translation', 'zero_right_wrist_translation', 'zero_right_wrist_translation', 'zero_full_36d_residual', 'zero_entire_right_hand', 'zero_right_wrist_translation', 'complete_PPO']
- source-56 passing candidates: ['complete_PPO', 'zero_right_wrist_translation', 'zero_right_wrist_rotation', 'zero_right_fingers', 'zero_complete_right_wrist', 'zero_entire_right_hand', 'zero_full_36d_residual']
- successful intervals: 37/40
- first failure endpoint: 58
- next blocker: tail_semantic_suppression_creates_recoverable_source56_state_requires_mode_characterization

The selector uses only next-endpoint termination/score plus the predeclared least-intervention tie break.
No contact signal, source-index rule, scale sweep, training or chunk commit is used.
