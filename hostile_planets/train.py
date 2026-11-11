"""PPO training CLI / library entry point.

python -m hostile_planets.train --mission search --planet corrosion --comm gossip \
    --coupling 1 --protect 0 --seed 0 --iters 300 --n-envs 256 --out results/raw
"""
import os

if os.environ.get("HP_THREADS") != "multi":
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                               " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1").strip()

    def _pin_to_free_core():
        # The XLA flags above do not stop JAX>=0.5 CPU matmuls from using every core, so
        # also pin this process to one core. HP_CORE picks it explicitly; otherwise grab
        # the first core whose lock file is free (locks die with the process).
        import fcntl
        import tempfile
        cores = sorted(os.sched_getaffinity(0))
        if "HP_CORE" in os.environ:
            os.sched_setaffinity(0, {int(os.environ["HP_CORE"])})
            return
        for c in cores:
            f = open(os.path.join(tempfile.gettempdir(), f"hp_core_{c}.lock"), "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            globals()["_CORE_LOCK"] = f
            os.sched_setaffinity(0, {c})
            return
        os.sched_setaffinity(0, {cores[os.getpid() % len(cores)]})

    if hasattr(os, "sched_setaffinity"):
        _pin_to_free_core()

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from hostile_planets import ppo  # noqa: E402

DEFAULTS = dict(
    mission="search", planet="corrosion", comm="gossip", coupling=1, protect=0, seed=0,
    iters=300, n_envs=256, out="results/raw", name=None, adv_frac=0.5,
    tempo_choices=(0.3, 0.8, 1.3), n_choices=(2, 3, 4, 6, 8, 12),
    lr=5e-4, ent_coef=0.003, n_epochs=3, n_minibatches=4,
    freeze_robot=False, freeze_adv=False, init_from=None,
    env_module="hostile_planets.env", save_every=25, log_every=10, verbose=True,
)


def run_name(cfg):
    return cfg["name"] or (f"{cfg['mission']}_{cfg['planet']}_{cfg['comm']}"
                           f"_c{int(cfg['coupling'])}_p{int(cfg['protect'])}_s{cfg['seed']}")


def make_sample_params(env, cfg):
    n_choices = jnp.array(cfg["n_choices"], jnp.int32)
    tempos = jnp.array(cfg["tempo_choices"], jnp.float32)

    def sample_params(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return env.make_params(planet=cfg["planet"], n_robots=jax.random.choice(k1, n_choices),
                               tempo=jax.random.choice(k2, tempos), comm=cfg["comm"],
                               coupling=bool(cfg["coupling"]),
                               adversarial=jax.random.bernoulli(k3, cfg["adv_frac"]),
                               protect=bool(cfg["protect"]))
    return sample_params


def _clean(x):
    x = float(x)
    return None if math.isnan(x) else x


def train(config: dict, env=None) -> dict:
    """Train; returns {"config", "run_name", "curve", "robot_params", "adv_params", "paths"}.
    `env` overrides construction from config["env_module"] (e.g. a stub)."""
    cfg = {**DEFAULTS, **config}
    if env is None:
        env = importlib.import_module(cfg["env_module"]).HostileEnv(mission=cfg["mission"])
    pcfg = ppo.PPOConfig(n_envs=cfg["n_envs"], lr=cfg["lr"], ent_coef=cfg["ent_coef"],
                         n_epochs=cfg["n_epochs"], n_minibatches=cfg["n_minibatches"],
                         total_iters=cfg["iters"], freeze_robot=bool(cfg["freeze_robot"]),
                         freeze_adv=bool(cfg["freeze_adv"]))
    key = jax.random.PRNGKey(cfg["seed"])
    key, k_init = jax.random.split(key)
    init = ppo.load_params(cfg["init_from"]) if cfg["init_from"] else (None, None)
    ts = ppo.init_train_state(k_init, env, pcfg, *init)
    iteration = ppo.make_iteration(env, make_sample_params(env, cfg), pcfg)

    name = run_name(cfg)
    if cfg["out"]:
        os.makedirs(cfg["out"], exist_ok=True)
    paths = {"npz": os.path.join(cfg["out"], name + ".npz"),
             "json": os.path.join(cfg["out"], name + ".json")} if cfg["out"] else {}
    json_cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    curve, t0 = [], time.time()

    def save():
        if not paths:
            return
        ppo.save_params(paths["npz"], ts.robot, ts.adv)
        with open(paths["json"], "w") as f:
            json.dump({"config": json_cfg, "run_name": name, "curve": curve}, f, indent=1)

    for it in range(cfg["iters"]):
        key, k = jax.random.split(key)
        t_it = time.time()
        ts, m = iteration(ts, k)
        m = jax.device_get(m)
        now = time.time()
        row = {"iter": it, **{k_: _clean(v) for k_, v in m.items()},
               "iter_time": now - t_it, "wall_time": now - t0}
        curve.append(row)
        if cfg["verbose"] and (it % cfg["log_every"] == 0 or it == cfg["iters"] - 1):
            print(f"[{name}] it {it:4d}  J nat {row['J_nature']}  J adv {row['J_adv']}  "
                  f"R_rob {row['robot_return']:.3f}  R_adv {row['adv_return']}  "
                  f"deaths {row['deaths']:.2f}  {row['iter_time']:.2f}s", flush=True)
        if cfg["save_every"] and (it + 1) % cfg["save_every"] == 0:
            save()
    save()
    return {"config": json_cfg, "run_name": name, "curve": curve,
            "robot_params": ts.robot, "adv_params": ts.adv, "paths": paths}


def _floats(s):
    return tuple(float(x) for x in s.split(","))


def _ints(s):
    return tuple(int(x) for x in s.split(","))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mission", default=DEFAULTS["mission"], choices=["search", "haul", "escort"])
    p.add_argument("--planet", default=DEFAULTS["planet"])
    p.add_argument("--comm", default=DEFAULTS["comm"], choices=["silent", "gossip", "blackboard"])
    p.add_argument("--coupling", type=int, choices=[0, 1], default=1)
    p.add_argument("--protect", type=int, choices=[0, 1], default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=DEFAULTS["iters"])
    p.add_argument("--n-envs", type=int, default=DEFAULTS["n_envs"])
    p.add_argument("--out", default=DEFAULTS["out"])
    p.add_argument("--name", default=None, help="run name (default: auto from config)")
    p.add_argument("--adv-frac", type=float, default=0.5)
    p.add_argument("--tempo-choices", type=_floats, default=DEFAULTS["tempo_choices"])
    p.add_argument("--n-choices", type=_ints, default=DEFAULTS["n_choices"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--ent-coef", type=float, default=DEFAULTS["ent_coef"])
    p.add_argument("--init-from", default=None, help="npz to initialise both teams from")
    p.add_argument("--freeze-robot", type=int, choices=[0, 1], default=0)
    p.add_argument("--freeze-adv", type=int, choices=[0, 1], default=0)
    p.add_argument("--env-module", default=DEFAULTS["env_module"])
    p.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    p.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    train(vars(p.parse_args(argv)))


if __name__ == "__main__":
    main()
