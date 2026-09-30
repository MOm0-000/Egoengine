# Candidate G static training signal and loss path

This is a source-only audit. No checkpoint, model forward, gradient, optimizer, or simulator was invoked.

| Path | Frozen behavior in Candidate G |
|---|---|
| Rollout value | `PpoAgent.get_action_values()` replaces the actor model value with the independent asymmetric critic value when `has_asymmetric_critic`; GAE therefore starts from the fresh privileged critic. |
| GAE | `discount_values()` uses gamma=0.998 and tau=0.95. Stored visits contain the denormalized rollout value, return, and raw advantage before dataset value normalization. |
| External critic | A separate `[1024,512]` privileged-state MLP, fresh per seed, trained for 4 mini-epochs against the same GAE return with clipped value loss. |
| Actor optimizer loss | `actor_loss + 0.5 * critic_coef * internal_value_loss - entropy_coef * entropy + bounds_coef * bounds_loss`; frozen coefficients are critic_coef=4, entropy=0, bounds=0. Thus internal actor value loss has coefficient 2.0. |
| Actor internal value head | The actor has `separate_value_mlp=false`; its value head and policy heads share the 512-MLP, LSTM-1024, layer norm, and concatenated representation. Internal value loss can update shared policy representation even though rollout GAE uses an external critic. |
| Inheritance distinction | `actor-only inheritance` describes checkpoint state provenance. It does **not** mean actor optimizer uses policy-surrogate-only loss. |

Static source hashes are recorded in `input_manifest.json`. This establishes an influence path, not historical per-loss gradient magnitude. Missing historical per-loss gradients, natural closed-loop performance immediately after every update, and isolated RMS causal effects remain unknown.
