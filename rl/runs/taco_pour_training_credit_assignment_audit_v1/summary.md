# Pour corrected PPO training-credit assignment audit v1

Read-only local audit; no optimizer step, training resume or chunk commit occurred.

## Timing -1 versus Replay

- Replay: 30/40.
- timing -1: 30/40.
- suppression candidate: True.
- refinement candidate: False.

## Full one-step counterfactuals

- source 43: formal +1 ends at 48; best y=-1.00 ends at 50.
- source 44: formal +1 ends at 48; best y=-1.00 ends at 51.
- source 45: formal +1 ends at 48; best y=-1.00 ends at 56.
- source 46: formal +1 ends at 48; best y=-0.75 ends at 51.

## Historical credit availability

- Exact historical advantage/value/return available: False.
- Reason: v5 visitation omitted rollout reward, critic observation/value, return, advantage, raw observation and recurrent hidden; the final checkpoint contains only final networks/optimizers/environment state, not historical rollout buffers or per-epoch critics.
