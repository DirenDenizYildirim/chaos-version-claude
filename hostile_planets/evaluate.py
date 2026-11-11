"""Evaluate every trained policy of one (planet, mission) cell.

Writes rows in the schema of docs/EVAL_SCHEMA.md. Each policy is rolled out
under nature hazards and under every learned adversary of the cell's pool
(all adversaries co-trained in that cell), with common random numbers across
policies and adversaries.

    python -m hostile_planets.evaluate --planet corrosion --mission haul \
        --runs results/raw/runs --out results/raw/eval
"""
import argparse
import glob
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import constants as C
from .env import HostileEnv, rollout
from .heuristics import HEURISTICS
from .ppo import load_params, make_policy_fns

ARMS = {  # arm -> (comm, coupling, protect)
    "default": ("gossip", True, False),
    "silent": ("silent", True, False),
    "blackboard": ("blackboard", True, False),
    "solo": ("gossip", False, False),
    "protected": ("gossip", True, True),
    "heuristic": ("gossip", True, False),
}
TEMPOS = (0.1, 0.3, 0.6)
NS = (2, 3, 4, 6, 8, 12)
REMOVE_TS = (0, 50, 100)
STALE_STARTS = (0, 30, 60)
STALE_WINDOW = 90


def _tau_stale(blob_pos, amp):
    """Mean first-passage time of an active blob centre to displacement > SIGMA,
    over a few start times, capped at STALE_WINDOW."""
    taus = []
    for t0 in STALE_STARTS:
        seg = blob_pos[t0:t0 + STALE_WINDOW]                    # [W, NB, 2]
        far = jnp.linalg.norm(seg - seg[0], axis=-1) > C.SIGMA  # [W, NB]
        first = jnp.where(far.any(0), jnp.argmax(far, 0), STALE_WINDOW).astype(jnp.float32)
        taus.append(first)
    tau = jnp.stack(taus).mean(0)
    act = (amp > 0).astype(jnp.float32)
    return jnp.sum(tau * act) / jnp.maximum(act.sum(), 1.0)


def _summarise(info, p):
    live = (info["n_alive"] > 0).astype(jnp.float32)
    wmean = lambda x: jnp.sum(x * live) / jnp.maximum(live.sum(), 1.0)
    third = lambda x, a, b: jnp.sum((x * live)[a:b]) / jnp.maximum(live[a:b].sum(), 1.0)
    e = info["exposure"]
    return {
        "J": info["J"][-1],
        "deaths": info["deaths"][-1],
        "coupled_frac": wmean(info["coupled_frac"]),
        "exp_first": third(e, 0, 50), "exp_mid": third(e, 50, 100), "exp_last": third(e, 100, 150),
        "speed": wmean(info["mean_speed"]),
        "health": wmean(info["mean_health"]),
        "tau_stale": _tau_stale(info["blob_pos"], p.amp),
    }


def make_eval_fns(env):
    @jax.jit
    def learned(robot_params, adv_params, params, keys):
        pol, adv = make_policy_fns(robot_params, adv_params, deterministic=False)
        return jax.vmap(lambda k, p: _summarise(rollout(env, k, p, pol, adv)[1], p))(keys, params)

    @jax.jit
    def scripted(adv_params, params, keys):
        _, adv = make_policy_fns(adv_params, adv_params, deterministic=False)
        pol = HEURISTICS[env.mission]
        return jax.vmap(lambda k, p: _summarise(rollout(env, k, p, pol, adv)[1], p))(keys, params)

    return learned, scripted


def condition_batch(env, planet, arm, adversarial, conds, n_ep):
    """conds: list of (tempo, n, remove_t). Returns batched params [len(conds)*n_ep]."""
    comm, coupling, protect = ARMS[arm]
    plist = [env.make_params(planet=planet, n_robots=n, tempo=tempo, comm=comm, coupling=coupling,
                             adversarial=adversarial, protect=protect, remove_t=rt)
             for tempo, n, rt in conds]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *plist)
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x, n_ep, axis=0), stacked)


def rows_from(out, conds, n_ep, meta):
    rows = []
    out = {k: np.asarray(v).reshape(len(conds), n_ep) for k, v in out.items()}
    for i, (tempo, n, rt) in enumerate(conds):
        r = dict(meta, tempo=tempo, n=n, remove_t=rt, n_ep=n_ep)
        r["J"] = out["J"][i].mean()
        r["J_sem"] = out["J"][i].std(ddof=1) / np.sqrt(n_ep)
        r["deaths_per_robot"] = out["deaths"][i].mean() / n
        for k in ("coupled_frac", "exp_first", "exp_mid", "exp_last", "speed", "health", "tau_stale"):
            r[k] = out[k][i].mean()
        rows.append(r)
    return rows


def parse_run(path):
    planet, mission, arm, seed = os.path.basename(path)[:-4].split("__")
    return planet, mission, arm, int(seed[1:])


def evaluate_cell(planet, mission, runs_dir, n_ep=128, seed=1234, br_dir=None, max_pool=None):
    env = HostileEnv(mission)
    learned, scripted = make_eval_fns(env)
    paths = sorted(glob.glob(os.path.join(runs_dir, f"{planet}__{mission}__*.npz")))
    runs = {os.path.basename(p)[:-4]: load_params(p) for p in paths}
    if not runs:
        return pd.DataFrame()
    pool = list(runs)[:max_pool] if max_pool else list(runs)
    any_adv = runs[pool[0]][1]

    conds = [(t, n, -1) for t in TEMPOS for n in NS]
    conds_rm = [(t, 6, rt) for t in TEMPOS for rt in REMOVE_TS]
    rows = []

    def run_policy(name, arm, seed_id, robot_params):
        cs = conds + (conds_rm if arm == "default" else [])
        keys = jax.random.split(jax.random.PRNGKey(seed), len(cs) * n_ep)  # common random numbers
        sources = [("nature", "", any_adv)] + [("adv", a, runs[a][1]) for a in pool]
        if br_dir and name is not None:
            bp = os.path.join(br_dir, f"br__{name}.npz")
            if os.path.exists(bp):
                sources.append(("br", f"br__{name}", load_params(bp)[1]))
        for mode, adv_name, adv_params in sources:
            params = condition_batch(env, planet, arm, mode != "nature", cs, n_ep)
            out = (scripted(adv_params, params, keys) if robot_params is None
                   else learned(robot_params, adv_params, params, keys))
            rows.extend(rows_from(out, cs, n_ep, dict(
                planet=planet, mission=mission, arm=arm, seed=seed_id, mode=mode, adv_name=adv_name)))

    t0 = time.time()
    for name, (rp, _) in runs.items():
        _, _, arm, s = parse_run(name + ".npz")
        run_policy(name, arm, s, rp)
        print(f"  {name} done ({time.time()-t0:.0f}s)", flush=True)
    run_policy(None, "heuristic", -1, None)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planet", required=True)
    ap.add_argument("--mission", required=True)
    ap.add_argument("--runs", default="results/raw/runs")
    ap.add_argument("--br", default=None)
    ap.add_argument("--out", default="results/raw/eval")
    ap.add_argument("--n-ep", type=int, default=128)
    ap.add_argument("--max-pool", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = evaluate_cell(args.planet, args.mission, args.runs, args.n_ep, br_dir=args.br, max_pool=args.max_pool)
    path = os.path.join(args.out, f"{args.planet}__{args.mission}.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
