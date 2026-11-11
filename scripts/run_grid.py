"""Run the training grid: 4 planets x 3 missions x 5 arms x seeds.

Each run is one single-threaded `hostile_planets.train` process; `--jobs`
of them run side by side. Finished runs (json present) are skipped, so the
script is resumable.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLANETS = ["corrosion", "jam", "quake", "combined"]
MISSIONS = ["search", "haul", "escort"]
# arm -> (comm, coupling, protect)
ARMS = {
    "default": ("gossip", 1, 0),
    "silent": ("silent", 1, 0),
    "blackboard": ("blackboard", 1, 0),
    "solo": ("gossip", 0, 0),
    "protected": ("gossip", 1, 1),
}


def run_name(planet, mission, arm, seed):
    return f"{planet}__{mission}__{arm}__s{seed}"


def launch(job, args):
    planet, mission, arm, seed = job
    name = run_name(*job)
    out_json = os.path.join(args.out, name + ".json")
    if os.path.exists(out_json):
        return name, 0, 0.0
    comm, coupling, protect = ARMS[arm]
    cmd = [
        sys.executable, "-m", "hostile_planets.train",
        "--mission", mission, "--planet", planet, "--comm", comm,
        "--coupling", str(coupling), "--protect", str(protect),
        "--seed", str(seed), "--iters", str(args.iters),
        "--n-envs", str(args.n_envs), "--out", args.out, "--name", name,
    ]
    t0 = time.time()
    with open(os.path.join(args.out, name + ".log"), "w") as log:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return name, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "raw", "runs"))
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--n-envs", type=int, default=128)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--planets", default=",".join(PLANETS))
    ap.add_argument("--missions", default=",".join(MISSIONS))
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    # seed-major order: a full single-seed grid finishes before the next seed starts
    jobs = [
        (p, m, a, s)
        for s, m, p, a in itertools.product(
            seeds, args.missions.split(","), args.planets.split(","), args.arms.split(",")
        )
    ]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(args.jobs) as ex:
        for name, rc, dt in ex.map(lambda j: launch(j, args), jobs):
            done += 1
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"[{done}/{len(jobs)}] {name} {status} {dt:.0f}s  (elapsed {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
