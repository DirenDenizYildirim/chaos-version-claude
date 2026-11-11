"""Stub environment following docs/DESIGN.md section 7 exactly (API only).

Task: robots start near a left anchor and are rewarded for approaching a goal at
the right anchor; 9 hazard blobs drain health and reward of nearby robots.
In adversarial envs the blobs are steered by `adv_act`, otherwise they follow a
heading random walk. J = mean over robots of alive * progress (cumulative-ish),
`rew["adv"] = -dJ`.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

T, N_MAX, NB, V0, SIGMA = 150, 12, 9, 0.03, 0.09
COMMS = {"silent": 0, "gossip": 1, "blackboard": 2}


class Params(NamedTuple):
    n_robots: jnp.ndarray
    tempo: jnp.ndarray
    comm: jnp.ndarray
    coupling: jnp.ndarray
    adversarial: jnp.ndarray
    protect: jnp.ndarray
    amp: jnp.ndarray
    remove_t: jnp.ndarray


class State(NamedTuple):
    t: jnp.ndarray
    pos: jnp.ndarray
    h: jnp.ndarray
    present: jnp.ndarray
    goal: jnp.ndarray
    d0: jnp.ndarray
    blob: jnp.ndarray
    heading: jnp.ndarray
    J: jnp.ndarray
    deaths: jnp.ndarray


class HostileEnv:
    def __init__(self, mission="search", obs_dim=70):
        self.mission = mission
        self.obs_dim, self.act_dim = obs_dim, 3
        self.adv_obs_dim, self.adv_act_dim = 14, 2

    def make_params(self, planet="corrosion", n_robots=8, tempo=0.6, comm="gossip",
                    coupling=True, adversarial=False, protect=False, remove_t=-1):
        amp = jnp.ones(NB) if planet != "combined" else jnp.array([.8, .8, 0.] * 3)
        return Params(jnp.asarray(n_robots, jnp.int32), jnp.asarray(tempo, jnp.float32),
                      jnp.asarray(COMMS[comm], jnp.int32), jnp.asarray(coupling, bool),
                      jnp.asarray(adversarial, bool), jnp.asarray(protect, bool), amp,
                      jnp.asarray(remove_t, jnp.int32))

    def _field(self, s, p, pos):
        d2 = ((pos[:, None, :] - s.blob[None]) ** 2).sum(-1)
        g = p.amp * jnp.exp(-d2 / (2 * SIGMA ** 2))  # [N,NB]
        grad = (g[..., None] * (s.blob[None] - pos[:, None, :])).sum(1) / SIGMA ** 2
        return jnp.clip(g.sum(-1), 0, 1), grad

    def _alive(self, s):
        return s.present & (s.h > 0)

    def _obs(self, s, p):
        alive = self._alive(s)
        f, grad = self._field(s, p, s.pos)
        rel_goal = s.goal - s.pos
        rel_blob = (s.blob[None] - s.pos[:, None]).reshape(N_MAX, -1)
        tt = jnp.full((N_MAX, 1), s.t / T)
        feats = jnp.concatenate([s.pos * 2 - 1, s.h[:, None], rel_goal * 2, rel_blob * 2,
                                 f[:, None], jnp.tanh(grad * 0.1), tt,
                                 jnp.full((N_MAX, 1), p.n_robots / N_MAX)], -1)
        robot = jnp.zeros((N_MAX, self.obs_dim)).at[:, :feats.shape[1]].set(feats)
        robot = robot * alive[:, None]
        # adversary obs
        d2 = ((s.blob[:, None] - s.pos[None]) ** 2).sum(-1) + jnp.where(alive, 0., 1e3)[None]
        idx = jnp.argsort(d2, axis=1)[:, :2]
        near = (s.pos[idx] - s.blob[:, None]).reshape(NB, 4)
        near = jnp.where(alive.any(), near, 0.)
        cen = (s.pos * alive[:, None]).sum(0) / jnp.maximum(alive.sum(), 1)
        onehot = jnp.repeat(jnp.eye(3), 3, axis=0)
        adv = jnp.concatenate([s.blob * 2 - 1, onehot, near * 2, (cen - s.blob) * 2,
                               (s.goal - s.blob) * 2, jnp.full((NB, 1), s.t / T)], -1)
        return {"robot": robot, "robot_mask": alive, "adv": adv, "adv_mask": p.amp > 0}

    def _J(self, s):
        prog = jnp.clip(1 - jnp.linalg.norm(s.goal - s.pos, axis=-1) / s.d0, 0, 1)
        return (prog * self._alive(s)).sum() / jnp.maximum(s.present.sum(), 1)

    def reset(self, key, p):
        k = jax.random.split(key, 5)
        left = jnp.array([jax.random.uniform(k[0], (), minval=.1, maxval=.2),
                          jax.random.uniform(k[1], (), minval=.2, maxval=.8)])
        goal = jnp.array([.85, jax.random.uniform(k[2], (), minval=.2, maxval=.8)])
        pos = left + 0.03 * jax.random.normal(k[3], (N_MAX, 2))
        present = jnp.arange(N_MAX) < p.n_robots
        blob = jax.random.uniform(k[4], (NB, 2), minval=.3, maxval=.7)
        d0 = jnp.linalg.norm(goal - pos, axis=-1)
        s = State(jnp.int32(0), pos, jnp.ones(N_MAX), present, goal, d0, blob,
                  jax.random.uniform(k[4], (NB,), maxval=2 * jnp.pi), jnp.float32(0), jnp.float32(0))
        return s, self._obs(s, p)

    def step(self, key, s, robot_act, adv_act, p):
        k1, k2 = jax.random.split(key)
        alive = self._alive(s)
        a = jnp.clip(robot_act, -1, 1)
        v = a[:, :2] / jnp.maximum(1, jnp.linalg.norm(a[:, :2], axis=-1, keepdims=True))
        pos = jnp.clip(s.pos + V0 * (0.4 + 0.6 * s.h[:, None]) * v * alive[:, None], 0, 1)
        heading = s.heading + 0.3 * jax.random.normal(k1, (NB,))
        u_nat = jnp.stack([jnp.cos(heading), jnp.sin(heading)], -1)
        ua = jnp.clip(adv_act, -1, 1)
        ua = ua / jnp.maximum(1, jnp.linalg.norm(ua, axis=-1, keepdims=True))
        u = jnp.where(p.adversarial, ua, u_nat)
        blob = jnp.clip(s.blob + p.tempo * V0 * u, .05, .95)
        s2 = s._replace(pos=pos, blob=blob, heading=heading)
        f, _ = self._field(s2, p, pos)
        h = jnp.where(alive, s.h - 0.04 * f, s.h)
        h = jnp.where((jnp.arange(N_MAX) == jnp.argmax(alive)) & (s.t == p.remove_t) & alive, 0., h)
        s2 = s2._replace(h=h, t=s.t + 1)
        died = alive & ~self._alive(s2)
        dprev = jnp.linalg.norm(s.goal - s.pos, axis=-1)
        dnew = jnp.linalg.norm(s.goal - pos, axis=-1)
        indiv = 10 * (dprev - dnew) * alive - 0.2 * f * alive - jnp.where(p.protect, 1., 0.) * died
        n = jnp.maximum(s.present.sum(), 1)
        rew_r = (0.5 * indiv + 0.5 * indiv.sum() / n) * alive
        J = self._J(s2)
        s2 = s2._replace(J=J, deaths=s.deaths + died.sum())
        info = {"J": J, "n_alive": self._alive(s2).sum(), "deaths": s2.deaths,
                "coupled_frac": jnp.float32(0), "exposure": (f * alive).sum() / n,
                "mean_speed": jnp.float32(0), "mean_health": (h * s.present).sum() / n}
        return s2, self._obs(s2, p), {"robot": rew_r, "adv": -(J - s.J)}, s2.t >= T, info
