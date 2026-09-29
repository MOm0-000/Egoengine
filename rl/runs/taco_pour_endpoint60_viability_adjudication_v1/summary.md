# Endpoint-60 viability adjudication v1

- Decision: `C3 / ARRIVAL_STATE_VIABILITY_BOTTLENECK`
- Continuous Candidate-D carryover: `20/40`, first failure endpoint `61`.
- Endpoint 21-60 historical reproduction: bitwise exact for every saved array.

| Boundary | Feasible / 7168 | Best score | Best score - 1 |
|---|---:|---:|---:|
| A | 0 | 1.013592601 | +0.013592601 |
| B | 0 | 1.020016551 | +0.020016551 |
| P | 727 | 0.989890456 | -0.010109544 |

No PPO training, reward/observation/action/physics/LR change, support
expansion, reference/GT state injection, or chunk commit was performed.
A finite negative search is not a mathematical infeasibility proof.

Verification: `818 passed, 19 warnings, 57 subtests passed`.
