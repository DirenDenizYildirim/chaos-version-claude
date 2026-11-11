"""Scripted potential-field baselines, one per mission.

Each policy maps the robot observation batch ``[N_MAX, obs_dim]`` (layout in
``env.py``) to actions ``[N_MAX, 3]`` and has the ``policy_fn(obs, key)``
signature expected by ``env.rollout``. They are sanity baselines, not optimal.
"""

import jax
import jax.numpy as jnp

from . import constants as C
from .env import OBS, zero_adv_action

N = C.N_MAX
_M = OBS["mission"].start


def _unit(v):
    return v / jnp.maximum(jnp.sqrt(jnp.sum(v * v, -1, keepdims=True) + 1e-12), 1e-6)


def _avoid(obs, w_rep=1.0, w_grad=1.5):
    """Move away from believed blobs and down the in-situ field gradients (all hazard types)."""
    rep = obs[:, OBS["blob_repulse"]].reshape(N, 3, 2).sum(1)
    grad = obs[:, OBS["grad"]].reshape(N, 3, 2).sum(1)
    return w_rep * rep - w_grad * grad


def _separate(obs, radius=0.15):
    out = jnp.zeros((N, 2))
    for sl in (OBS["mate1"], OBS["mate2"]):
        m = obs[:, sl]
        d = jnp.sqrt(jnp.sum(m[:, :2] ** 2, -1) + 1e-6)
        out = out - _unit(m[:, :2]) * (m[:, 4] * jnp.clip(1 - d / radius, 0.0, 1.0))[:, None]
    return out


def _act(v, engage):
    v = v / jnp.maximum(jnp.sqrt(jnp.sum(v * v, -1, keepdims=True) + 1e-12), 1.0)
    return jnp.concatenate([v, jnp.broadcast_to(jnp.asarray(engage, jnp.float32), (N,))[:, None]], -1)


def search_policy(obs, key=None):
    """Head for the nearest cell not known to be visited; spread out; avoid hazards."""
    mis = obs[:, OBS["mission"]]
    t = obs[:, OBS["time"]]
    ang = 2 * jnp.pi * jnp.arange(N) / N
    fan = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1) * jnp.clip(1 - t / 0.15, 0.0, 1.0)
    v = _unit(mis[:, 0:2]) * mis[:, 3:4] + 0.8 * _separate(obs) + 0.8 * fan + _avoid(obs)
    return _act(_unit(v), -1.0)


def haul_policy(obs, key=None, pair=False):
    """Shuttle depot -> goal. ``pair=True`` waits at the depot for a partner (heavy crates)."""
    mis = obs[:, OBS["mission"]]
    carrying = (mis[:, 4] + mis[:, 5]) > 0
    target = jnp.where(carrying[:, None], mis[:, 2:4], mis[:, 0:2])
    near = jnp.sqrt(jnp.sum(target ** 2, -1, keepdims=True)) < 0.12
    v = _unit(target) + jnp.where(near, 0.0, 0.7) * _avoid(obs)
    return _act(_unit(v), 1.0 if pair else -1.0)


def haul_pair_policy(obs, key=None):
    return haul_policy(obs, key, pair=True)


def escort_policy(obs, key=None):
    """Every third robot hugs the VIP as shield/rescuer; the rest scout ahead on its route."""
    mis = obs[:, OBS["mission"]]
    slot = jnp.arange(N)
    shield = (slot % 3) == 0
    known = mis[:, 3:4]
    fwd = _unit(mis[:, 9:11])
    side = jnp.stack([-fwd[:, 1], fwd[:, 0]], -1)
    lat = jnp.where(slot % 2 == 0, 1.0, -1.0) * (0.04 + 0.03 * (slot // 3))
    scout_target = mis[:, 0:2] + 0.16 * fwd + lat[:, None] * side
    target = jnp.where(shield[:, None], mis[:, 0:2], scout_target)
    target = jnp.where(known > 0, target, mis[:, 11:13])   # VIP lost: head for the destination
    gain = jnp.clip(jnp.sqrt(jnp.sum(target ** 2, -1, keepdims=True)) / 0.03, 0.0, 1.0)
    v = _unit(target) * gain + jnp.where(shield[:, None], 0.0, 0.6) * _avoid(obs)
    return _act(v, jnp.where(shield, 1.0, -1.0))


def idle_policy(obs, key=None):
    return jnp.zeros((N, 3)).at[:, 2].set(-1.0)


def random_policy(obs, key):
    return jax.random.uniform(key, (N, 3), minval=-1.0, maxval=1.0)


HEURISTICS = {"search": search_policy, "haul": haul_policy, "escort": escort_policy}
zero_adversary = zero_adv_action


def chase_adversary(obs_adv, key=None):
    """Scripted adversary: every blob chases the VIP (escort) or its nearest live robot."""
    vip, robot = obs_adv[:, 15:17], obs_adv[:, 5:7]
    has_vip = jnp.sum(jnp.abs(vip), -1, keepdims=True) > 0
    return _unit(jnp.where(has_vip, vip, robot))
