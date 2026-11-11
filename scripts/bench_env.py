"""Benchmark: full-episode (T=150) rollouts of a vmapped batch with a random policy.

Usage: PYTHONPATH=. python3 scripts/bench_env.py [batch] [mission ...]
Single-thread mode:
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      taskset -c 0 python3 scripts/bench_env.py
"""
import os
import sys
import time

import jax

from hostile_planets import constants as C
from hostile_planets.env import HostileEnv, rollout, stack_params
from hostile_planets.heuristics import random_policy

B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
MISSIONS = sys.argv[2:] or C.MISSIONS


def main():
    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}  batch={B}  T={C.T}  N_MAX={C.N_MAX}")
    for mission in MISSIONS:
        env = HostileEnv(mission)
        # worst case for cost: all slots live, every hazard type, random adversary, cycle comm modes
        ps = stack_params([env.make_params(planet="combined", n_robots=C.N_MAX, comm=i % 3,
                                           adversarial=bool(i % 2)) for i in range(B)])
        adv = lambda o, k: jax.random.uniform(k, (C.NB, 2), minval=-1.0, maxval=1.0)
        fn = jax.jit(jax.vmap(lambda k, p: rollout(env, k, p, random_policy, adv)[1]["J"][-1]))
        keys = jax.random.split(jax.random.PRNGKey(0), B)
        t0 = time.time()
        fn(keys, ps).block_until_ready()
        t_first = time.time() - t0
        reps = 3
        t0 = time.time()
        for i in range(reps):
            fn(jax.random.split(jax.random.PRNGKey(i + 1), B), ps).block_until_ready()
        dt = (time.time() - t0) / reps
        print(f"{mission:7s} compile {t_first - dt:5.1f}s | {dt:6.3f}s per {B}-env episode batch | "
              f"{B / dt:7.0f} episodes/s | {B * C.T / dt / 1e3:7.1f}k env-steps/s | "
              f"{B * C.T * C.N_MAX / dt / 1e6:5.2f}M robot-steps/s")


if __name__ == "__main__":
    main()
