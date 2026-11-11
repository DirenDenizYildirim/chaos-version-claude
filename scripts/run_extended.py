"""Exploratory budget check: continue training the `default` swarm (and its adversary) of some
cells for more iterations, to see whether the gap to the scripted heuristic is a training-budget
effect. Outputs keep the run name but live in results/raw/extended/ (optimiser state restarts)."""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="comma list of planet:mission")
    ap.add_argument("--runs", default="results/raw/runs")
    ap.add_argument("--out", default="results/raw/extended")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--n-envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)

    def run(cell):
        planet, mission = cell.split(":")
        name = f"{planet}__{mission}__default__s{args.seed}"
        src = os.path.join(ROOT, args.runs, name + ".npz")
        cmd = [sys.executable, "-m", "hostile_planets.train", "--mission", mission, "--planet", planet,
               "--comm", "gossip", "--coupling", "1", "--protect", "0", "--seed", str(200 + args.seed),
               "--iters", str(args.iters), "--n-envs", str(args.n_envs), "--init-from", src,
               "--out", args.out, "--name", name]
        with open(os.path.join(ROOT, args.out, name + ".log"), "w") as log:
            return name, subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    with ThreadPoolExecutor(args.jobs) as ex:
        for name, rc in ex.map(run, args.cells.split(",")):
            print(name, "ok" if rc == 0 else f"FAILED {rc}", flush=True)


if __name__ == "__main__":
    main()
