# Hostile Planets

A simulated suite of hostile "planets" for ground-robot swarms, and one
pre-registered experiment run on it: do a few shared principles (how fast
information goes stale, how a robot's worth to the mission changes over time)
let results from one hostile world predict another?

**The findings are in [`results/REPORT.md`](results/REPORT.md).** This file
describes what was built and how to rerun it.

## The study in one page

* **Planets.** `corrosion` degrades bodies, `jam` disrupts sensing,
  communication and navigation, `quake` destabilises the ground (slowing,
  slipping, trapping), `combined` has all three. Hazards are moving blobs whose
  speed (`tempo`) is the dial for how fast information goes stale.
* **Missions.** `search` an area for targets, `haul` cargo from a depot to a
  goal, `escort` a larger scripted robot across the arena. Each has one stated
  metric `J` in [0, 1].
* **Adversary.** Hazard blobs are steered by a learned adversary (one shared
  policy, one agent per blob) co-trained against the swarm; half of the
  training environments use random-walk "nature" hazards instead.
* **Swarm.** Shared-parameter PPO policies with egocentric observations, trained
  over random swarm sizes (2-12) and tempos.
* **Questions and arms.** Five separately trained arms per (planet, mission)
  cell, one factor changed at a time:

  | question | arm | change from `default` |
  |---|---|---|
  | how to coordinate | `silent`, `blackboard` | no communication / a global hub instead of local gossip |
  | when to couple tightly | `solo` | all co-presence mechanics disabled (shielding, beacon, rescue, joint carry) |
  | how large | (all) | every policy is evaluated at N = 2, 3, 4, 6, 8, 12 |
  | when to sacrifice | `protected` | a death penalty in training |

* **Grid.** 4 planets x 3 missions x 5 arms x 2 seeds = 120 training runs,
  evaluated at 3 tempos x 6 swarm sizes under nature and under every learned
  adversary of the cell (the adversarial score is the minimum over that pool).
* **Test.** [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) was committed
  before any grid result existed. It fixes the outcomes, the two principle
  variables (staleness ratio `S`, worth trajectory `W(t)`), the transfer test
  (fit on three single-hazard planets, predict `combined`; leave-one-planet-out)
  and the bar a "supported" verdict has to clear.

Everything is an abstract 2-D kinematic simulator written in JAX and was run
on 4 CPU cores. It is not a physics simulation and makes no claim about real
robots; see the limitations in the report.

## Layout

| path | what |
|---|---|
| `docs/DESIGN.md` | simulator spec: constants, hazards, missions, information model, API |
| `docs/PREREGISTRATION.md` | outcomes, test, verdict rule, and the amendments made before the grid ran |
| `docs/EVAL_SCHEMA.md` | columns of `results/eval.csv` |
| `hostile_planets/env.py` | the environment (`HostileEnv`, pure `reset`/`step`, `vmap`-able) |
| `hostile_planets/heuristics.py` | scripted swarm policies and a scripted chasing adversary |
| `hostile_planets/ppo.py`, `train.py` | independent PPO for swarm and adversary, training CLI |
| `hostile_planets/evaluate.py` | evaluation of one cell under nature and the adversary pool |
| `hostile_planets/analysis.py` | outcomes, principle variables, transfer test, validity checks |
| `scripts/` | grid, best-response, evaluation and report runners |
| `results/` | `eval.csv`, `tables/`, `figures/`, `summary.json`, `REPORT.md`; raw runs under `results/raw/` |

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
python3 -m pytest -q tests
```

```bash
python3 scripts/run_grid.py --iters 150 --n-envs 128 --seeds 0,1 --jobs 4
```

```bash
python3 scripts/run_br.py
```

```bash
python3 scripts/run_eval.py --n-ep 128 --n-ep-worth 512
```

```bash
python3 scripts/make_report.py --eval results/eval.csv --runs results/raw/runs --out results/
```

```bash
python3 scripts/exploratory.py
```

On 4 cores the grid took about 4.4 hours, the best responses 10 minutes and
the evaluation about 3.5 hours. Training is seeded, but results are not
bit-reproducible across machines.

`scripts/patch_eval_default.py` is a one-off that repaired the first evaluation
and is not needed for a fresh run. The exploratory budget check in the report
was produced with:

```bash
python3 scripts/run_extended.py --iters 300 --cells corrosion:search,jam:search,quake:search,combined:search,quake:haul,quake:escort,combined:escort
```

```bash
python3 scripts/run_eval.py --runs results/raw/extended --br results/raw/none --out results/raw/eval_extended --merged results/eval_extended.csv --cells corrosion:search,jam:search,quake:search,combined:search,quake:haul,quake:escort,combined:escort
```
