# Evaluation output schema (`results/eval.csv`)

One row = one (policy, condition, hazard source) evaluated over `n_ep` episodes.

| column | meaning |
|---|---|
| `planet` | corrosion / jam / quake / combined |
| `mission` | search / haul / escort |
| `arm` | default / silent / blackboard / solo / protected / heuristic |
| `seed` | training seed (int); -1 for heuristic |
| `tempo` | 0.3 / 0.8 / 1.3 |
| `n` | swarm size: 2,3,4,6,8,12 |
| `mode` | `nature` (random-walk hazards), `adv` (a learned adversary from the cell's pool), `br` (best-response adversary trained against this frozen policy) |
| `adv_name` | run name whose adversary was used (`""` for nature) |
| `remove_t` | -1, or step at which one robot was force-removed (0/50/100; only arm=default, n=6) |
| `n_ep` | episodes |
| `J`, `J_sem` | mean mission metric at T and its standard error |
| `deaths_per_robot` | mean deaths / n (forced removal not counted) |
| `coupled_frac` | mean over time of fraction of live robots in a coupled state |
| `exp_first`, `exp_mid`, `exp_last` | mean hazard exposure of live robots in steps [0,50), [50,100), [100,150) |
| `speed` | mean realised speed of live robots (arena units / step) |
| `health` | mean health of live robots, time-averaged |
| `tau_stale` | mean first-passage time (steps, capped at 150) of an active blob centre to displacement > SIGMA=0.09 |

Derived scores (see PREREGISTRATION.md): nature score = `J` of the `nature`
row; adversarial score = min over `adv` rows (same policy, tempo, n,
remove_t) of `J`.

Training curves: `results/raw/runs/<planet>__<mission>__<arm>__s<seed>.json`
(config + per-iteration curve).

`n_ep` is not constant across rows: the default arm's `n = 6` rows (full swarm
and the three removal conditions) are evaluated with `--n-ep-worth` episodes
(more than `--n-ep`), because the worth trajectory `W(t)` is a small difference
of two means.
