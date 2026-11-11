"""Evaluate all cells (4 single-core processes side by side) and merge into results/eval.csv."""
import argparse
import glob
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANETS = ["corrosion", "jam", "quake", "combined"]
MISSIONS = ["search", "haul", "escort"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/raw/runs")
    ap.add_argument("--br", default="results/raw/br")
    ap.add_argument("--out", default="results/raw/eval")
    ap.add_argument("--merged", default="results/eval.csv")
    ap.add_argument("--n-ep", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cells", default=None, help="comma list of planet:mission (default all)")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    cells = ([tuple(c.split(":")) for c in args.cells.split(",")] if args.cells
             else list(itertools.product(PLANETS, MISSIONS)))
    cores = Queue()
    for c in range(args.jobs):
        cores.put(c)

    def run(cell):
        planet, mission = cell
        core = cores.get()
        t0 = time.time()
        cmd = ["taskset", "-c", str(core), sys.executable, "-m", "hostile_planets.evaluate",
               "--planet", planet, "--mission", mission, "--runs", args.runs, "--br", args.br,
               "--out", args.out, "--n-ep", str(args.n_ep)]
        with open(os.path.join(ROOT, args.out, f"{planet}__{mission}.log"), "w") as log:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        cores.put(core)
        return planet, mission, rc, time.time() - t0

    with ThreadPoolExecutor(args.jobs) as ex:
        for planet, mission, rc, dt in ex.map(run, cells):
            print(f"{planet}/{mission} rc={rc} {dt:.0f}s", flush=True)

    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, args.out, "*.csv")))
             if os.path.getsize(f) > 10]
    df = pd.concat(parts, ignore_index=True)
    df["adv_name"] = df["adv_name"].fillna("")
    df.to_csv(os.path.join(ROOT, args.merged), index=False)
    print(f"merged {len(df)} rows -> {args.merged}")


if __name__ == "__main__":
    main()
