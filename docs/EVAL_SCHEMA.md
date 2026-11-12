# Evaluation output schema (`results/eval.csv`)

One row = one (policy, condition, hazard source) evaluated over `n_ep` episodes.

| column | meaning |
|---|---|
| `planet` | corrosion / jam / quake / combined |
| `mission` | search / haul / escort |
| `arm` | default / silent / blackboard / solo / protected / heuristic |
| `seed` | training seed (int); -1 for heuristic |
| `tempo` | 0.1 / 0.3 / 0.6 |
| `n` | swarm size: 2,3,4,6,8,12 |
| `mode` | `nature` (random-walk hazards), `adv` (a learned adversary from the cell's pool), `br` (best-response adversary trained against this frozen policy) |
| `adv_name` | run name whose adversary was used (`""` for nature) |
| `remove_t` | -1 (standard condition); step at which one robot was force-removed (0/50/100) or -2 (full swarm) in the worth block (only arm=default, n=6) |
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

Worth block. For the default arm there is a second block of rows evaluated with
`--n-ep-worth` episodes (512 in the stored results, against 128 elsewhere),
because the worth trajectory `W(t)` is a small difference of two means: the
full swarm at `n = 6`, stored with `remove_t = -2`, and the three removal
conditions (`remove_t` in {0, 50, 100}). Rows with `remove_t = -1` are the
standard conditions and use the same episodes for every arm.

The stored `results/eval.csv` came from a first evaluation that lacked the
common episodes between the default arm and the others, plus `scripts/patch_eval_default.py`,
which fixed that; the script's docstring explains the bias this removed.
