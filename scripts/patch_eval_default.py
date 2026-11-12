"""One-off patch of the first full evaluation (kept for the record; `hostile_planets.evaluate`
now produces the patched layout directly).

In the first evaluation the default arm's standard conditions were run on different episodes from
the other arms, and its n=6 full-swarm rows only existed inside the 512-episode worth block. Two
consequences for the arm comparisons (V_gossip, V_couple, V_sac):

* no common random numbers between `default` and the other arms (extra noise), and
* at n=6 a bias: the adversarial score is a minimum over a pool of adversaries, a minimum over
  noisier means is biased lower, so a 512-episode default row compared with 128-episode rows of
  another arm favoured `default`. It showed up as V_sac > 0 on the jam planet, where `protected`
  and `default` are the same policy.

This script (1) relabels the 512-episode full-swarm rows as worth-block rows (remove_t = -2) and
(2) replaces every standard row of the default arm by one evaluated on the same 128 episodes as
all other arms. No other row is touched.
"""
import glob
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL, STD = "results/raw/eval", "results/raw/eval_default_std"
PLANETS = ["corrosion", "jam", "quake", "combined"]
MISSIONS = ["search", "haul", "escort"]


def main():
    os.makedirs(os.path.join(ROOT, STD), exist_ok=True)
    cores = Queue()
    for c in range(4):
        cores.put(c)

    def run(cell):
        planet, mission = cell
        if os.path.exists(os.path.join(ROOT, STD, f"{planet}__{mission}.csv")):
            return planet, mission, 0
        core = cores.get()
        cmd = ["taskset", "-c", str(core), sys.executable, "-m", "hostile_planets.evaluate",
               "--planet", planet, "--mission", mission, "--runs", "results/raw/runs",
               "--br", "results/raw/br", "--out", STD, "--n-ep", "128", "--only-default-standard"]
        with open(os.path.join(ROOT, STD, f"{planet}__{mission}.log"), "w") as log:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        cores.put(core)
        return planet, mission, rc

    with ThreadPoolExecutor(4) as ex:
        for planet, mission, rc in ex.map(run, itertools.product(PLANETS, MISSIONS)):
            print(f"{planet}/{mission} rc={rc}", flush=True)
            assert rc == 0

    parts = []
    for f in sorted(glob.glob(os.path.join(ROOT, EVAL, "*.csv"))):
        d = pd.read_csv(f)
        is_def = d["arm"] == "default"
        d.loc[is_def & (d["remove_t"] == -1) & (d["n_ep"] > 128), "remove_t"] = -2
        d = d[~(is_def & (d["remove_t"] == -1))]
        new = pd.read_csv(os.path.join(ROOT, STD, os.path.basename(f)))
        assert set(new["arm"]) == {"default"} and (new["remove_t"] == -1).all() and (new["n_ep"] == 128).all()
        d = pd.concat([d, new], ignore_index=True)
        d.to_csv(f, index=False)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["adv_name"] = df["adv_name"].fillna("")
    df.to_csv(os.path.join(ROOT, "results/eval.csv"), index=False)
    print(f"merged {len(df)} rows -> results/eval.csv")


if __name__ == "__main__":
    main()
