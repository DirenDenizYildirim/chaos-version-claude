"""Sanity table: scripted heuristics on a benign setting vs the hostile planets, and J(N).

Usage: PYTHONPATH=. python3 scripts/sanity_heuristics.py [episodes]
"""
import sys

import jax
import numpy as np

from hostile_planets import constants as C
from hostile_planets.env import HostileEnv, rollout, stack_params
from hostile_planets.heuristics import (HEURISTICS, chase_adversary, haul_pair_policy,
                                        idle_policy, random_policy)

EP = int(sys.argv[1]) if len(sys.argv) > 1 else 128
PLANETS = ("corrosion", "jam", "quake", "combined")
SETTINGS = [("benign", dict(planet="corrosion", amp_scale=0.3, tempo=0.2, ambient=0.0))] + \
    [(pl, dict(planet=pl, tempo=0.6)) for pl in PLANETS] + \
    [(pl + "/chase", dict(planet=pl, tempo=0.6, adversarial=True)) for pl in PLANETS]
NS = (2, 4, 8, 12)


def evaluate(env, policy, plist):
    """Mean final J / deaths / coupled fraction for each params in plist (EP episodes each)."""
    ps = stack_params([p for p in plist for _ in range(EP)])
    keys = jax.random.split(jax.random.PRNGKey(0), len(plist) * EP)
    info = jax.jit(jax.vmap(lambda k, p: rollout(env, k, p, policy, chase_adversary)[1]))(keys, ps)
    f = lambda x: np.asarray(x).reshape(len(plist), EP, -1)
    return f(info["J"])[:, :, -1].mean(1), f(info["deaths"])[:, :, -1].mean(1), \
        f(info["coupled_frac"]).mean((1, 2))


def main():
    for mission in C.MISSIONS:
        env = HostileEnv(mission)
        pols = {"heuristic": HEURISTICS[mission], "random": random_policy, "idle": idle_policy}
        if mission == "haul":
            pols["heur-pair"] = haul_pair_policy
        print(f"\n=== {mission}: mean final J over {EP} episodes (gossip, coupling on; '/chase' = scripted chasing adversary, else nature)")
        print(f"{'policy':10s} {'setting':15s} " + " ".join(f"N={n:<2d}  " for n in NS)
              + " deaths(N=8) coupled(N=8)")
        for pname, pol in pols.items():
            plist = [env.make_params(n_robots=n, **kw) for _, kw in SETTINGS for n in NS]
            J, deaths, cpl = (x.reshape(len(SETTINGS), len(NS)) for x in evaluate(env, pol, plist))
            for i, (sname, _) in enumerate(SETTINGS):
                print(f"{pname:10s} {sname:15s} " + " ".join(f"{v:.3f} " for v in J[i])
                      + f"  {deaths[i, 2]:6.2f}     {cpl[i, 2]:.3f}")


if __name__ == "__main__":
    main()
