"""Best-response adversaries: fine-tune the co-trained adversary against each frozen
`default` swarm (all envs adversarial), as an exploitability check on the pool-minimum score."""
import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/raw/runs")
    ap.add_argument("--out", default="results/raw/br")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--n-envs", type=int, default=128)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    paths = []
    for s in args.seeds.split(","):
        paths += sorted(glob.glob(os.path.join(ROOT, args.runs, f"*__default__s{s}.npz")))

    def run(path):
        name = os.path.basename(path)[:-4]
        planet, mission, _, seed = name.split("__")
        out = f"br__{name}"
        if os.path.exists(os.path.join(ROOT, args.out, out + ".json")):
            return name, 0
        cmd = [sys.executable, "-m", "hostile_planets.train", "--mission", mission, "--planet", planet,
               "--comm", "gossip", "--coupling", "1", "--protect", "0", "--seed", str(100 + int(seed[1:])),
               "--iters", str(args.iters), "--n-envs", str(args.n_envs), "--adv-frac", "1.0",
               "--freeze-robot", "1", "--init-from", path, "--out", args.out, "--name", out]
        with open(os.path.join(ROOT, args.out, out + ".log"), "w") as log:
            return name, subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    with ThreadPoolExecutor(args.jobs) as ex:
        for name, rc in ex.map(run, paths):
            print(name, "ok" if rc == 0 else f"FAILED {rc}", flush=True)


if __name__ == "__main__":
    main()
