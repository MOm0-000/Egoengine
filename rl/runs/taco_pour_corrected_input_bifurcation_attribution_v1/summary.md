# Corrected frozen-policy input/bifurcation attribution v1

This is a read-only local diagnosis, not a paper-recovered policy contract.

## Path forward comparison

| source | PPO μ_y | Replay μ_y (own hidden) | Replay obs + PPO hidden μ_y |
|---:|---:|---:|---:|
| 40 | 0.498084 | 0.284152 | 0.494606 |
| 41 | 0.740690 | 0.552547 | 0.738063 |
| 42 | 0.947483 | 0.904391 | 1.086210 |
| 43 | 1.294169 | 1.439576 | 1.562811 |
| 44 | 1.686058 | 1.815947 | 1.839898 |
| 45 | 1.986601 | 2.159229 | 2.184128 |
| 46 | 2.105058 | 2.226626 | 2.202583 |
| 47 | 2.138400 | 2.223717 | 2.217579 |

## One-step rescue follow-up

- source 44: later μ_y = [45:1.9886, 46:2.1235, 47:2.1626], endpoint48 score=0.955944.
- source 45: later μ_y = [46:2.1002, 47:2.1304], endpoint48 score=0.917122.
- source 46: later μ_y = [47:2.1160], endpoint48 score=0.984747.

## Largest single-group substitutions

- source 42: `current_object_anchors` changes μ_y by +0.151432.
- source 43: `current_object_anchors` changes μ_y by +0.166071.
- source 44: `current_object_anchors` changes μ_y by +0.166775.
- source 45: `current_object_anchors` changes μ_y by +0.132903.
- source 46: `current_object_anchors` changes μ_y by +0.131053.

## Frozen conclusion

- PPO-path μ_y exceeds support at sources [43, 44, 45, 46, 47].
- Replay-path μ_y also exceeds support at sources [43, 44, 45, 46, 47].
- Every successful one-step rescue keeps all later μ_y values above support; the actor does not self-correct.
- No recorded PPO/Replay input dimension reaches the normalization ±5 clamp.

## Limits

- Full historical training observations were not logged, so running mean/std is not treated as empirical training support.
- Finite differences are actor input sensitivities with fixed recurrent memory, not causal physics gradients.
- Counterfactual group substitutions may form observations that did not occur physically; they localize the actor input dependency only.
