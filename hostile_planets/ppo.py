"""Independent PPO with parameter sharing for the robot team and the hazard adversary.

Networks are plain pytrees (nested dicts) so they flatten with a stable key order.
One PPO iteration = rollout of T steps over B fresh envs + one update per team,
all inside a single jitted function (see `make_iteration`).
"""
import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

LOG2PI = math.log(2 * math.pi)


class PPOConfig(NamedTuple):
    n_envs: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    lr: float = 5e-4
    max_grad_norm: float = 0.5
    n_epochs: int = 3
    n_minibatches: int = 4
    total_iters: int = 1000          # for the linear lr decay
    robot_hidden: int = 64
    adv_hidden: int = 32
    log_std_init: float = -0.5
    freeze_robot: bool = False
    freeze_adv: bool = False
    adv_rew_scale: float = 10.0  # adversary reward = -scale*dJ + potential-based proximity shaping
    adv_shape: float = 1.0


# ----------------------------------------------------------------------------- networks

def init_mlp(key, sizes, out_scale):
    keys = jax.random.split(key, len(sizes) - 1)
    layers = {}
    for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = out_scale if i == len(sizes) - 2 else math.sqrt(2)
        layers[f"l{i}"] = {"w": jax.nn.initializers.orthogonal(scale)(k, (a, b)),
                           "b": jnp.zeros(b)}
    return layers


def mlp(layers, x):
    n = len(layers)
    for i in range(n):
        x = x @ layers[f"l{i}"]["w"] + layers[f"l{i}"]["b"]
        if i < n - 1:
            x = jnp.tanh(x)
    return x


def init_actor_critic(key, obs_dim, act_dim, hidden, log_std_init=-0.5):
    """Separate policy and value trunks (2 x hidden tanh), state-independent log_std."""
    kp, kv = jax.random.split(key)
    return {"pi": init_mlp(kp, [obs_dim, hidden, hidden, act_dim], 0.01),
            "v": init_mlp(kv, [obs_dim, hidden, hidden, 1], 1.0),
            "log_std": jnp.full(act_dim, log_std_init, jnp.float32)}


def forward(params, obs):
    """obs [..., obs_dim] -> (mean [..., act_dim], log_std [act_dim], value [...])."""
    return mlp(params["pi"], obs), params["log_std"], mlp(params["v"], obs)[..., 0]


def gauss_logp(x, mean, log_std):
    return (-0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * LOG2PI).sum(-1)


def gauss_entropy(log_std):
    return (log_std + 0.5 * (1 + LOG2PI)).sum()


def sample_action(params, obs, key):
    mean, log_std, value = forward(params, obs)
    act = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    return act, gauss_logp(act, mean, log_std), value


def make_policy_fns(robot_params, adv_params, deterministic: bool):
    """Returns policy_fn(obs_robot [N_MAX,obs_dim], key) -> [N_MAX,3] and
    adv_fn(obs_adv [NB,adv_obs_dim], key) -> [NB,2] (unclipped; the env clips).
    Both also work with extra leading batch dims."""
    def make(params):
        def fn(obs, key):
            mean, log_std, _ = forward(params, obs)
            if deterministic:
                return mean
            return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
        return fn
    return make(robot_params), make(adv_params)


# ----------------------------------------------------------------------------- params io

def flatten_params(tree, prefix=""):
    out = {}
    for k in sorted(tree):
        v = tree[k]
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_params(v, name + "/"))
        else:
            out[name] = np.asarray(v)
    return out


def unflatten_params(flat):
    tree = {}
    for name, v in flat.items():
        node = tree
        *path, leaf = name.split("/")
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = jnp.asarray(v)
    return tree


def save_params(path, robot_params, adv_params):
    flat = {**flatten_params(robot_params, "robot/"), **flatten_params(adv_params, "adv/")}
    np.savez(path, **flat)


def load_params(path):
    """-> (robot_params, adv_params) as nested dicts of jnp arrays."""
    with np.load(path) as z:
        tree = unflatten_params({k: z[k] for k in z.files})
    return tree["robot"], tree["adv"]


# ----------------------------------------------------------------------------- PPO core

def gae(rew, val, mask, gamma, lam):
    """All arrays [T, ...]. mask[t]=1 if the agent is alive at t. Bootstrap 0 at T;
    an agent dead at t+1 is terminal at t."""
    rew, val = jnp.where(mask > 0, rew, 0.), jnp.where(mask > 0, val, 0.)  # 0*nan guard
    next_val = jnp.concatenate([val[1:], jnp.zeros_like(val[:1])])
    next_mask = jnp.concatenate([mask[1:], jnp.zeros_like(mask[:1])])
    delta = rew + gamma * next_val * next_mask - val

    def body(carry, x):
        d, nm = x
        a = d + gamma * lam * nm * carry
        return a, a
    _, adv = jax.lax.scan(body, jnp.zeros_like(val[0]), (delta, next_mask), reverse=True)
    return adv, adv + val


def masked_mean(x, mask):
    return jnp.where(mask, x, 0.).sum() / jnp.maximum(mask.sum(), 1)


def ppo_loss(params, b, cfg: PPOConfig):
    """b: dict obs, act, logp, val, adv, ret, mask with a common leading sample axis.
    Samples with mask=0 contribute exactly nothing (value and gradient)."""
    m = b["mask"]
    mean, log_std, v = forward(params, b["obs"])
    logp = gauss_logp(b["act"], mean, log_std)
    ratio = jnp.exp(jnp.where(m, logp - b["logp"], 0.))
    pg = -jnp.minimum(ratio * b["adv"],
                      jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b["adv"])
    v_clip = b["val"] + jnp.clip(v - b["val"], -cfg.clip_eps, cfg.clip_eps)
    vl = 0.5 * jnp.maximum((v - b["ret"]) ** 2, (v_clip - b["ret"]) ** 2)
    ent = gauss_entropy(log_std) * jnp.ones_like(logp)
    pg_l, v_l, ent_m = masked_mean(pg, m), masked_mean(vl, m), masked_mean(ent, m)
    loss = pg_l + cfg.vf_coef * v_l - cfg.ent_coef * ent_m
    kl = masked_mean((ratio - 1) - jnp.log(ratio), m)
    return loss, {"pg_loss": pg_l, "v_loss": v_l, "entropy": gauss_entropy(log_std), "kl": kl}


def prepare_batch(traj, cfg: PPOConfig):
    """traj: dict of [T,B,N,...] arrays (obs, act, logp, val, rew, mask) -> flat [T*B*N,...]
    with GAE, normalised advantages and invalid samples sanitised to zeros."""
    m = traj["mask"].astype(jnp.float32)
    adv, ret = gae(traj["rew"], traj["val"], m, cfg.gamma, cfg.lam)
    mask = traj["mask"]
    mu = masked_mean(adv, mask)
    sd = jnp.sqrt(masked_mean((adv - mu) ** 2, mask))
    adv = (adv - mu) / (sd + 1e-8)
    b = {"obs": traj["obs"], "act": traj["act"], "logp": traj["logp"], "val": traj["val"],
         "adv": adv, "ret": ret}
    b = jax.tree.map(lambda x: jnp.where(mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim)), x, 0), b)
    b["mask"] = mask
    lead = int(np.prod(mask.shape))
    return jax.tree.map(lambda x: x.reshape((lead,) + x.shape[mask.ndim:]), b)


def ppo_update(params, opt_state, opt, batch, key, cfg: PPOConfig):
    M = batch["mask"].shape[0]
    mb = M // cfg.n_minibatches
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    def epoch(carry, k):
        perm = jax.random.permutation(k, M)[: mb * cfg.n_minibatches]
        perm = perm.reshape(cfg.n_minibatches, mb)

        def step(carry, idx):
            params, opt_state = carry
            (_, aux), grads = grad_fn(params, jax.tree.map(lambda x: x[idx], batch), cfg)
            updates, opt_state = opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), aux
        return jax.lax.scan(step, carry, perm)

    (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state),
                                            jax.random.split(key, cfg.n_epochs))
    return params, opt_state, jax.tree.map(jnp.mean, aux)


# ----------------------------------------------------------------------------- training

class TrainState(NamedTuple):
    robot: dict
    adv: dict
    robot_opt: optax.OptState
    adv_opt: optax.OptState


def make_optimizer(cfg: PPOConfig):
    steps = max(cfg.total_iters * cfg.n_epochs * cfg.n_minibatches, 1)
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                       optax.adam(optax.linear_schedule(cfg.lr, 0.0, steps), eps=1e-5))


def init_train_state(key, env, cfg: PPOConfig, robot_params=None, adv_params=None):
    kr, ka = jax.random.split(key)
    if robot_params is None:
        robot_params = init_actor_critic(kr, env.obs_dim, env.act_dim, cfg.robot_hidden, cfg.log_std_init)
    if adv_params is None:
        adv_params = init_actor_critic(ka, env.adv_obs_dim, env.adv_act_dim, cfg.adv_hidden, cfg.log_std_init)
    opt = make_optimizer(cfg)
    return TrainState(robot_params, adv_params, opt.init(robot_params), opt.init(adv_params))


def _adv_target_dist(adv_obs):
    """Distance from each blob to its natural target (the VIP if there is one, else the
    nearest live robot), read from the adversary obs. Used for potential-based shaping."""
    vip, rob = adv_obs[..., 15:17], adv_obs[..., 5:7]
    has_vip = jnp.abs(vip).sum(-1, keepdims=True) > 0
    return jnp.linalg.norm(jnp.where(has_vip, vip, rob), axis=-1)


def _env_mean(x, sel):
    n = sel.sum()
    return jnp.where(n > 0, jnp.where(sel, x, 0.).sum() / jnp.maximum(n, 1), jnp.nan)


def make_iteration(env, sample_params: Callable, cfg: PPOConfig):
    """Returns jitted iteration(train_state, key) -> (train_state, metrics dict of scalars).

    sample_params(key) -> env params NamedTuple (vmapped over n_envs inside)."""
    opt = make_optimizer(cfg)
    B = cfg.n_envs
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    def rollout(ts: TrainState, key):
        kp, kr, ks = jax.random.split(key, 3)
        params = jax.vmap(lambda k: jax.tree.map(jnp.asarray, sample_params(k)))(jax.random.split(kp, B))
        adv_on = params.adversarial.astype(bool)
        state, obs = v_reset(jax.random.split(kr, B), params)

        def step(carry, k):
            state, obs = carry
            k_r, k_a, k_e = jax.random.split(k, 3)
            r_act, r_logp, r_val = sample_action(ts.robot, obs["robot"], k_r)
            a_act, a_logp, a_val = sample_action(ts.adv, obs["adv"], k_a)
            a_env = jnp.where(adv_on[:, None, None], a_act, 0.)
            state, obs2, rew, done, info = v_step(jax.random.split(k_e, B), state, r_act, a_env, params)
            out = {"robot": {"obs": obs["robot"], "act": r_act, "logp": r_logp, "val": r_val,
                             "rew": rew["robot"], "mask": obs["robot_mask"].astype(bool)},
                   "adv": {"obs": obs["adv"], "act": a_act, "logp": a_logp, "val": a_val,
                           "rew": cfg.adv_rew_scale * rew["adv"][:, None]
                           + cfg.adv_shape * (_adv_target_dist(obs["adv"]) - _adv_target_dist(obs2["adv"])),
                           "mask": obs["adv_mask"].astype(bool) & adv_on[:, None]},
                   "J": info["J"], "deaths": info["deaths"]}
            return (state, obs2), out

        _, traj = jax.lax.scan(step, (state, obs), jax.random.split(ks, env_T(env)))
        return traj, adv_on

    def iteration(ts: TrainState, key):
        k_roll, k_r, k_a = jax.random.split(key, 3)
        traj, adv_on = rollout(ts, k_roll)
        r, a = traj["robot"], traj["adv"]
        J = traj["J"][-1]
        r_ret = jnp.where(r["mask"], r["rew"], 0.).sum(0)          # [B,N]
        a_ret = a["rew"][:, :, 0].sum(0)                           # [B]
        metrics = {"J_nature": _env_mean(J, ~adv_on), "J_adv": _env_mean(J, adv_on),
                   "J_all": J.mean(),
                   "robot_return": masked_mean(r_ret, r["mask"][0]),
                   "adv_return": _env_mean(a_ret, adv_on),
                   "deaths": traj["deaths"][-1].mean(),
                   "n_valid_robot": r["mask"].sum(), "n_valid_adv": a["mask"].sum()}
        robot, adv, robot_opt, adv_opt = ts
        if not cfg.freeze_robot:
            robot, robot_opt, aux = ppo_update(robot, robot_opt, opt, prepare_batch(r, cfg), k_r, cfg)
            metrics.update({f"robot_{k}": v for k, v in aux.items()})
        if not cfg.freeze_adv:
            adv, adv_opt, aux = ppo_update(adv, adv_opt, opt, prepare_batch(a, cfg), k_a, cfg)
            metrics.update({f"adv_{k}": v for k, v in aux.items()})
        return TrainState(robot, adv, robot_opt, adv_opt), metrics

    return jax.jit(iteration, donate_argnums=0)


def env_T(env):
    """Episode length: env.horizon / env.T if exposed, else DESIGN.md's T=150."""
    return int(getattr(env, "horizon", getattr(env, "T", 150)))
