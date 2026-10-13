# Projection accounting erratum

The archived 93.75% field conflated true support clipping with float32 storage roundoff. No parent bytes were changed.

| condition | slots | support clip | roundoff only | final action on bound |
|---|---|---:|---:|---:|
| A_PLAN | all_slots | 3.121828259% | 90.628171741% | 3.319811699% |
| A_PLAN | noisy_slots | 3.329950142% | 96.670049858% | 3.329950142% |
| L_PLAN | all_slots | 2.792968750% | 90.957031250% | 3.000300481% |
| L_PLAN | noisy_slots | 2.979166667% | 97.020833333% | 2.979166667% |

All 1,024 archived candidate sequences and all archived winner choices were reproduced exactly from the frozen noise/support inputs.
