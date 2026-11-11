"""Write a schema-valid *synthetic* eval.csv (+ training-curve jsons) for demoing the report.

The numbers are made up. They carry a planted structure so that the analysis has something
to find: every outcome is generated from mission intercepts + log S + R_late (+ some planet
effects the principle model cannot see, + noise), so model P should do well on V_comm etc.

    python scripts/make_synthetic_eval.py --out /tmp/demo/eval.csv --runs /tmp/demo/raw/runs
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

PLANETS = ["corrosion", "jam", "quake", "combined"]
MISSIONS = ["search", "haul", "escort"]
ARMS = ["default", "silent", "blackboard", "solo", "protected"]
TEMPOS = [0.1, 0.3, 0.6]
NS = [2, 3, 4, 6, 8, 12]
MODES = ["nature", "adv"]

# ---- planted structure ---------------------------------------------------------------
SPEED_P = {"corrosion": 0.80, "jam": 0.90, "quake": 0.60, "combined": 0.70}
SPEED_M = {"search": 1.00, "haul": 0.75, "escort": 0.55}
TAU_P = {"corrosion": 1.00, "jam": 1.30, "quake": 0.80, "combined": 1.10}
RL_M = {"search": 0.5, "haul": 0.9, "escort": 1.6}
RL_P = {"corrosion": 0.3, "jam": -0.1, "quake": 0.1, "combined": 0.4}
A_M = {"search": 0.45, "haul": 0.30, "escort": 0.40}
A_P = {"corrosion": 0.80, "jam": 0.90, "quake": 0.85, "combined": 0.70}
LOSS_P = {"corrosion": 0.25, "jam": 0.02, "quake": 0.08, "combined": 0.20}
LOSS_M = {"search": 1.0, "haul": 0.8, "escort": 1.3}

DEFAULT_STRUCTURE = {
    # outcome: (mission intercepts, slope on log S, slope on R_late, planet effects)
    "V_comm": ({"search": 0.25, "haul": 0.15, "escort": 0.30}, -0.20, 0.0, {}),
    "V_gossip": ({"search": 0.15, "haul": 0.10, "escort": 0.20}, -0.12, 0.0, {}),
    "V_couple": ({"search": 0.02, "haul": 0.25, "escort": 0.40}, -0.08, 0.10, {"quake": 0.05}),
    "V_sac": ({"search": 0.00, "haul": 0.05, "escort": 0.08}, 0.0, 0.10, {"corrosion": 0.10, "combined": 0.10}),
    "beta": ({"search": 0.70, "haul": 0.80, "escort": 0.35}, -0.05, 0.0, {}),
}


def speed_of(p, m, tempo, mode):
    return 0.022 * SPEED_P[p] * SPEED_M[m] * (1 - 0.05 * tempo) * (0.95 if mode == "adv" else 1.0)


def tau_of(p, tempo, mode):
    return min(150.0, 4.5 / tempo * TAU_P[p] * (0.7 if mode == "adv" else 1.0))


def log_s(p, m, tempo, mode):
    return float(np.log((0.5 / speed_of(p, m, tempo, mode)) / tau_of(p, tempo, mode)))


def r_late(p, m, tempo, mode):
    return RL_M[m] + RL_P[p] - 0.1 * tempo + (0.1 if mode == "adv" else 0.0)


def solve_a(b, v):
    """a such that rel(a, b) = v."""
    return (b * (1 + 0.5 * v) + 0.02 * v) / (1 - 0.5 * v)


def solve_b(a, v):
    """b such that rel(a, b) = v."""
    return (a * (1 - 0.5 * v) - 0.02 * v) / (1 + 0.5 * v)


def planted_outcome(name, p, m, tempo, mode, structure=DEFAULT_STRUCTURE):
    mi, a, b, pe = structure[name]
    return mi[m] + a * (log_s(p, m, tempo, mode) - 1.5) + b * r_late(p, m, tempo, mode) + pe.get(p, 0.0)


def make_synthetic(rng_seed=0, seeds=(0, 1), pool=4, cond_noise=0.03, seed_noise=0.01, eval_noise=0.005,
                   sem=0.01, structure=DEFAULT_STRUCTURE, heuristic=True, br=True) -> pd.DataFrame:
    """Synthetic eval table. With all noises 0 the planted outcomes are recovered exactly by the analysis."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for p in PLANETS:
        for m in MISSIONS:
            runs = [f"{p}__{m}__{a}__s{s}" for a in ARMS for s in seeds]
            for tempo in TEMPOS:
                for mode in MODES:
                    V = {k: planted_outcome(k, p, m, tempo, mode, structure) + cond_noise * rng.standard_normal()
                         for k in structure}
                    A = A_M[m] * A_P[p] * (1 - 0.1 * tempo) * (1.15 if mode == "nature" else 1.0)
                    rl = r_late(p, m, tempo, mode)
                    W = {0: rl + 0.8, 50: rl + 0.4, 100: rl}
                    spd, tau = speed_of(p, m, tempo, mode), tau_of(p, tempo, mode)
                    destroy = p in ("corrosion", "combined")
                    for n in NS:
                        jd = A * (n / 6) ** V["beta"]
                        js = solve_b(jd, V["V_gossip"])
                        J = {"default": jd, "silent": js, "blackboard": solve_a(js, V["V_comm"]),
                             "solo": solve_b(jd, V["V_couple"]), "protected": solve_b(jd, V["V_sac"])}
                        if heuristic:
                            bad_cell = (p, m) in {("jam", "escort"), ("quake", "haul")}
                            J["heuristic"] = jd * (1.08 if bad_cell else 0.9)
                        for arm, jarm in J.items():
                            arm_seeds = [-1] if arm == "heuristic" else list(seeds)
                            for s in arm_seeds:
                                jseed = max(0.005, jarm + seed_noise * rng.standard_normal())
                                loss = LOSS_P[p] * LOSS_M[m] * (1 + 0.03 * n) * (0.4 if arm == "protected" else 1.0)
                                loss *= 1.2 if mode == "adv" else 1.0
                                cpl = 0.0 if arm == "solo" else {"search": 0.02, "haul": 0.3, "escort": 0.25}[m]
                                prot = arm == "protected"
                                removes = [-1] + ([0, 50, 100] if arm == "default" and n == 6 else [])
                                for rt in removes:
                                    jr = jseed if rt < 0 else jseed * (1 - W[rt] / 6)
                                    e = {t: 0.12 + (0.04 * (t / 50) if destroy and not prot else 0.0)
                                         - 0.02 * W[t] + 0.01 * rng.standard_normal() for t in (0, 50, 100)}
                                    base = dict(planet=p, mission=m, arm=arm, seed=s, tempo=tempo, n=n,
                                                remove_t=rt, n_ep=64, J_sem=sem,
                                                deaths_per_robot=max(0.0, loss + 0.01 * rng.standard_normal()),
                                                coupled_frac=cpl, exp_first=e[0], exp_mid=e[50], exp_last=e[100],
                                                speed=spd, health=0.8 - 0.5 * loss, tau_stale=tau)
                                    if mode == "nature":
                                        rows.append({**base, "mode": "nature", "adv_name": "",
                                                     "J": jr + eval_noise * rng.standard_normal()})
                                        continue
                                    own = f"{p}__{m}__{arm}__s{s}" if arm != "heuristic" else runs[0]
                                    others = [r for r in runs if r != own]
                                    names = [own] + list(rng.choice(others, size=min(pool - 1, len(others)),
                                                                    replace=False))
                                    k_min = int(rng.integers(len(names)))
                                    for k, nm in enumerate(names):
                                        off = 0.0 if k == k_min else float(rng.uniform(0.01, 0.08))
                                        rows.append({**base, "mode": "adv", "adv_name": nm,
                                                     "J": jr + off + eval_noise * rng.standard_normal()})
                                    if br and arm == "default" and s == seeds[0] and rt == -1 and n == 6:
                                        rows.append({**base, "mode": "br", "adv_name": f"br::{own}",
                                                     "J": jr - float(rng.uniform(0.0, 0.04))})
    df = pd.DataFrame(rows)
    cols = ["planet", "mission", "arm", "seed", "tempo", "n", "mode", "adv_name", "remove_t", "n_ep", "J", "J_sem",
            "deaths_per_robot", "coupled_frac", "exp_first", "exp_mid", "exp_last", "speed", "health", "tau_stale"]
    return df[cols]


def write_runs(runs_dir, seeds=(0, 1), iters=60, rng_seed=0):
    """Fake training-curve jsons; a few use alternative key names to exercise the defensive loader."""
    rng = np.random.default_rng(rng_seed)
    os.makedirs(runs_dir, exist_ok=True)
    for i, (p, m, a, s) in enumerate((p, m, a, s) for p in PLANETS for m in MISSIONS for a in ARMS for s in seeds):
        top = A_M[m] * A_P[p] * (1.1 if a == "blackboard" else 1.0)
        it = np.arange(iters)
        jn = top * 1.15 * (1 - np.exp(-it / 15)) + 0.01 * rng.standard_normal(iters)
        ja = top * (1 - np.exp(-it / 20)) + 0.015 * rng.standard_normal(iters)
        alt = i % 7 == 3
        curve = [{("iteration" if alt else "iter"): int(k), ("j_nat" if alt else "J_nature"): float(jn[k]),
                  ("j_adversarial" if alt else "J_adv"): float(ja[k]), "ret": float(jn[k] * 10),
                  "deaths": float(LOSS_P[p]), "time": float(k * 3.0)} for k in it]
        cfg = {"planet": p, "mission": m, "arm": a, "seed": s}
        with open(os.path.join(runs_dir, f"{p}__{m}__{a}__s{s}.json"), "w") as f:
            json.dump({("cfg" if alt else "config"): cfg, ("history" if alt else "curve"): curve}, f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("results", "synthetic", "eval.csv"))
    ap.add_argument("--runs", default=None, help="also write fake training-curve jsons here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    df = make_synthetic(rng_seed=args.seed, seeds=seeds)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")
    if args.runs:
        write_runs(args.runs, seeds=seeds, rng_seed=args.seed)
        print(f"wrote training curves -> {args.runs}")


if __name__ == "__main__":
    main()
