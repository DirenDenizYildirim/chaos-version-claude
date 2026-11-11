"""Hostile Planets simulator (docs/DESIGN.md sections 2-7).

Pure-JAX, functional: ``reset``/``step`` are jit/vmap/scan-able over
``(key, state, actions, params)``.  The mission is static (one compiled function
per mission); everything else lives in ``EnvParams`` and can differ per env.

Robot observation layout (``obs_dim = 62``, identical size for every mission;
all relative positions are ``other - self`` in arena units, so within [-1,1])::

    0        health
    1:3      own position * 2 - 1
    3        t / T
    4        stuck flag
    5        (believed-alive teammates + self) / N_MAX
    6        coupled flag (was in a coupled state last step)
    7:10     in-situ field  f_corrosion, f_jam, f_quake
    10:16    in-situ field gradient * SIGMA  (corrosion xy, jam xy, quake xy);
             points towards higher field
    16:28    nearest believed blob per type, 4 each (type-major):
             rel pos (2), clip(age/50,0,1), known flag     [unknown: 0,0,1,0]
    28:34    repulsion vector per type (2 each): sum over believed blobs of
             exp(-age/30) * exp(-d^2 / (2 (2 SIGMA)^2)) * unit(self - blob)
    34:39    nearest known live teammate: rel pos (2), age, health, valid flag
    39:44    second nearest known live teammate (same 5)
    44:46    centroid of known live teammates, rel pos (0 if none)
    46:62    mission block (16):
      search  46:48 rel pos of nearest unvisited cell centre (own belief)
              48    its distance          49 "any unvisited" flag
              50:52 rel pos of the centroid of unvisited cells
              52    fraction of cells unvisited
              53:62 visited flags of the 3x3 cell neighbourhood (row-major in
                    (dx,dy), dx outer; off-arena cells read 1)
      haul    46:48 rel pos depot         48:50 rel pos goal
              50    carrying solo crate   51 carrying heavy crate (paired)
              52    inside depot zone     53 inside goal zone
              54    (# known live teammates within R_LINK) / 3, clipped to 1
              55    coupling available    56:62 zero
      escort  46:48 rel pos of the VIP (own belief)   48 age   49 known flag
              50    VIP health   51 VIP stuck flag    52:55 VIP in-situ fields
              55:57 VIP -> destination vector (from believed VIP position)
              57:59 rel pos of the destination
              59    VIP remaining distance / initial distance (believed)
              60    self within R_LINK of the believed VIP position
              61    coupling available

Adversary observation per blob (``adv_obs_dim = 20``; the adversary is omniscient)::

    0:2 own centre * 2 - 1      2:5 type one-hot
    5:7 rel pos nearest live robot     7:9 rel pos second nearest live robot
    9:11 rel pos swarm centroid        11:13 rel pos left anchor
    13:15 rel pos right anchor         15:17 rel pos VIP (escort, else 0)
    17 t / T     18 own amplitude     19 tempo

Episodes have fixed length ``T``; callers must ``reset`` after ``done``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import constants as C
from .planets import planet_ambient, planet_amp

F = jnp.float32
NEG = -1e9
INF = 1e9

OBS_DIM = 62
ADV_OBS_DIM = 20
OBS = {  # index ranges into the robot observation
    "health": slice(0, 1), "pos": slice(1, 3), "time": slice(3, 4), "stuck": slice(4, 5),
    "n_known": slice(5, 6), "coupled": slice(6, 7), "field": slice(7, 10),
    "grad": slice(10, 16), "blob_nearest": slice(16, 28), "blob_repulse": slice(28, 34),
    "mate1": slice(34, 39), "mate2": slice(39, 44), "mate_centroid": slice(44, 46),
    "mission": slice(46, 62),
}
VIP_DIM = 7  # VIP belief entry: pos (2), health, stuck flag, in-situ fields (3)

_g = (np.arange(C.GRID) + 0.5) / C.GRID
CELL_CENTRES = np.stack(np.meshgrid(_g, _g, indexing="ij"), -1).reshape(-1, 2).astype(np.float32)
G2 = C.GRID * C.GRID


class EnvParams(NamedTuple):
    n_robots: jnp.ndarray     # int32, robots present at reset (<= N_MAX)
    amp: jnp.ndarray          # [NB] blob amplitudes (0 = slot unused)
    tempo: jnp.ndarray        # blob speed in units of V0
    comm: jnp.ndarray         # int32: 0 silent, 1 gossip, 2 blackboard
    coupling: jnp.ndarray     # bool: co-presence synergies available
    adversarial: jnp.ndarray  # bool: blobs driven by adv_act (else nature)
    protect: jnp.ndarray      # bool: death penalty in the training reward
    ambient: jnp.ndarray      # ambient health loss / step
    remove_t: jnp.ndarray     # int32: step at which one robot is removed (-1 never)


class EnvState(NamedTuple):
    t: jnp.ndarray
    # robots
    pos: jnp.ndarray; h: jnp.ndarray; present: jnp.ndarray; alive: jnp.ndarray
    stuck: jnp.ndarray; fld: jnp.ndarray; grad: jnp.ndarray
    carry: jnp.ndarray; partner: jnp.ndarray; coupled: jnp.ndarray
    # hazards / layout
    bpos: jnp.ndarray; bdir: jnp.ndarray; anchors: jnp.ndarray
    # per-robot beliefs (timestamp -1 = never observed)
    b_pos: jnp.ndarray; b_t: jnp.ndarray          # blobs      [N,NB,2], [N,NB]
    m_val: jnp.ndarray; m_t: jnp.ndarray          # teammates  [N,N,3] (pos, health), [N,N]
    v_val: jnp.ndarray; v_t: jnp.ndarray          # VIP        [N,VIP_DIM], [N]
    vis: jnp.ndarray                              # visited    [N,G2]
    # blackboard hub
    hub_b_pos: jnp.ndarray; hub_b_t: jnp.ndarray
    hub_m_val: jnp.ndarray; hub_m_t: jnp.ndarray
    hub_v_val: jnp.ndarray; hub_v_t: jnp.ndarray
    hub_vis: jnp.ndarray
    # mission
    targets: jnp.ndarray; found: jnp.ndarray
    delivered: jnp.ndarray
    vip_pos: jnp.ndarray; vip_h: jnp.ndarray; vip_stuck: jnp.ndarray
    vip_arrived: jnp.ndarray; vip_fld: jnp.ndarray
    vip_b_pos: jnp.ndarray; vip_b_t: jnp.ndarray
    J: jnp.ndarray; deaths: jnp.ndarray


def _norm(x, axis=-1, keepdims=False):
    return jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=keepdims) + 1e-12)


def _unit_cap(v):
    """Scale vectors down to norm <= 1."""
    return v / jnp.maximum(_norm(v, keepdims=True), 1.0)


def _unit(v):
    return v / jnp.maximum(_norm(v, keepdims=True), 1e-6)


def _rotate(v, theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def _pairwise(x):
    return _norm(x[:, None, :] - x[None, :, :])


def fields(x, bpos, amp):
    """Hazard fields at points x[...,2]: clipped value [...,3] and gradient [...,3,2]."""
    d = bpos - x[..., None, :]
    g = amp * jnp.exp(-jnp.sum(d * d, -1) / (2 * C.SIGMA ** 2))
    f = g.reshape(g.shape[:-1] + (C.N_TYPES, C.K)).sum(-1)
    grad = (g[..., None] * d / C.SIGMA ** 2).reshape(g.shape[:-1] + (C.N_TYPES, C.K, 2)).sum(-2)
    return jnp.clip(f, 0.0, 1.0), grad


def _merge(recv_t, recv_v, send_t, send_v, link):
    """Newest-timestamp merge. recv_t [R,E], recv_v [R,E,D], send_* [S,E(,D)], link [R,S]."""
    cand = jnp.where(link[:, :, None], send_t[None], NEG)
    best, arg = cand.max(1), cand.argmax(1)
    bv = send_v[arg, jnp.arange(recv_t.shape[1])[None, :]]
    take = best > recv_t
    return jnp.where(take, best, recv_t), jnp.where(take[..., None], bv, recv_v)


def _project_safe(c, anchors):
    """Push blob centres out of R_SAFE around each anchor, staying inside the blob walls."""
    for a in anchors:
        d = c - a
        r = _norm(d)
        d = jnp.where((r < 1e-5)[:, None], jnp.array([0.0, 1.0], F), d)
        c = jnp.where((r < C.R_SAFE)[:, None], a + _unit(d) * C.R_SAFE, c)
    c = jnp.clip(c, C.BLOB_LO, C.BLOB_HI)
    # a centre clipped back onto a wall can land inside the disk again: slide it along the wall
    for a in anchors:
        d = c - a
        inside = _norm(d) < C.R_SAFE - 1e-5
        dy = jnp.sqrt(jnp.maximum(C.R_SAFE ** 2 - d[:, 0] ** 2, 0.0)) + 1e-5
        sgn = jnp.where(d[:, 1] >= 0, 1.0, -1.0)
        y1, y2 = a[1] + sgn * dy, a[1] - sgn * dy
        y = jnp.where((y1 >= C.BLOB_LO) & (y1 <= C.BLOB_HI), y1, y2)
        c = jnp.where(inside[:, None], jnp.stack([c[:, 0], y], -1), c)
    return c


class HostileEnv:
    """One mission on any planet. See the module docstring for the obs layout."""

    def __init__(self, mission: str = "search"):
        assert mission in C.MISSIONS, mission
        self.mission = mission
        self.n_max = C.N_MAX
        self.obs_dim = OBS_DIM
        self.act_dim = 3
        self.adv_obs_dim = ADV_OBS_DIM
        self.adv_act_dim = 2
        self.horizon = C.T
        self.obs_layout = OBS

    # ------------------------------------------------------------------ params
    def make_params(self, planet="corrosion", n_robots=8, tempo=0.6, comm="gossip",
                    coupling=True, adversarial=False, protect=False, remove_t=-1,
                    amp_scale=1.0, amp=None, ambient=None) -> EnvParams:
        """Build one (unbatched) EnvParams. ``amp``/``ambient`` override the planet preset."""
        amp = planet_amp(planet) * amp_scale if amp is None else np.asarray(amp, np.float32)
        ambient = planet_ambient(planet) if ambient is None else ambient
        comm = C.COMM_MODES.index(comm) if isinstance(comm, str) else int(comm)
        return EnvParams(
            n_robots=jnp.asarray(n_robots, jnp.int32), amp=jnp.asarray(amp, F),
            tempo=jnp.asarray(tempo, F), comm=jnp.asarray(comm, jnp.int32),
            coupling=jnp.asarray(coupling, bool), adversarial=jnp.asarray(adversarial, bool),
            protect=jnp.asarray(protect, bool), ambient=jnp.asarray(ambient, F),
            remove_t=jnp.asarray(remove_t, jnp.int32))

    def _safe_anchors(self, anchors):
        return [anchors[0]] if self.mission == "search" else [anchors[0], anchors[1]]

    # ------------------------------------------------------------------- reset
    def reset(self, key, p: EnvParams):
        N, NB = C.N_MAX, C.NB
        ks = jax.random.split(key, 8)
        left = jnp.stack([jax.random.uniform(ks[0], (), F, 0.1, 0.2),
                          jax.random.uniform(ks[1], (), F, 0.2, 0.8)])
        right = jnp.stack([jax.random.uniform(ks[2], (), F, 0.8, 0.9),
                           jax.random.uniform(ks[3], (), F, 0.2, 0.8)])
        anchors = jnp.stack([left, right])
        present = jnp.arange(N) < p.n_robots
        u = jax.random.uniform(ks[4], (N, 2), F)
        ang, rad = 2 * jnp.pi * u[:, 0], 0.05 * jnp.sqrt(u[:, 1])
        pos = jnp.clip(left + rad[:, None] * jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1), 0.0, 1.0)
        bpos = _project_safe(jax.random.uniform(ks[5], (NB, 2), F, C.BLOB_LO, C.BLOB_HI),
                             self._safe_anchors(anchors))
        bang = jax.random.uniform(ks[6], (NB,), F, 0.0, 2 * jnp.pi)
        k_t, k_c = jax.random.split(ks[7])
        fld, grad = fields(pos, bpos, p.amp)
        vip_fld, _ = fields(left, bpos, p.amp)
        z = jnp.zeros
        s = EnvState(
            t=z((), jnp.int32),
            pos=pos, h=jnp.ones(N, F), present=present, alive=present,
            stuck=z(N, jnp.int32), fld=fld, grad=grad,
            carry=z(N, F), partner=-jnp.ones(N, jnp.int32), coupled=z(N, bool),
            bpos=bpos, bdir=jnp.stack([jnp.cos(bang), jnp.sin(bang)], -1), anchors=anchors,
            b_pos=z((N, NB, 2), F), b_t=-jnp.ones((N, NB), F),
            m_val=z((N, N, 3), F), m_t=-jnp.ones((N, N), F),
            v_val=z((N, VIP_DIM), F), v_t=-jnp.ones(N, F),
            vis=z((N, G2), bool),
            hub_b_pos=z((NB, 2), F), hub_b_t=-jnp.ones(NB, F),
            hub_m_val=z((N, 3), F), hub_m_t=-jnp.ones(N, F),
            hub_v_val=z(VIP_DIM, F), hub_v_t=-jnp.ones((), F),
            hub_vis=z(G2, bool),
            targets=jax.random.uniform(k_t, (C.M_TARGETS, 2), F, 0.03, 0.97),
            found=z(C.M_TARGETS, bool), delivered=z((), F),
            vip_pos=left, vip_h=jnp.ones((), F), vip_stuck=z((), jnp.int32),
            vip_arrived=z((), bool), vip_fld=vip_fld,
            vip_b_pos=z((NB, 2), F), vip_b_t=-jnp.ones(NB, F),
            J=z((), F), deaths=z((), F))
        s = self._sense_comm(s, k_c, p)
        return s, self._obs(s, p)

    # -------------------------------------------------------------------- step
    def step(self, key, s: EnvState, robot_act, adv_act, p: EnvParams):
        N = C.N_MAX
        mission = self.mission
        k_rot, k_slip, k_trap, k_blob, k_comm, k_vip = jax.random.split(key, 6)
        idx = jnp.arange(N)
        eye = jnp.eye(N, dtype=bool)
        t1 = s.t + 1
        tf = t1.astype(F)
        depot, goal = s.anchors[0], s.anchors[1]

        # counterfactual removal of the lowest-index live robot (not a death)
        rm = (s.t == p.remove_t) & s.alive & (idx == jnp.argmax(s.alive))
        present = s.present & ~rm
        alive0 = s.alive & ~rm

        a = jnp.clip(jnp.nan_to_num(robot_act), -1.0, 1.0)
        cmd = _unit_cap(a[:, :2])
        eng = a[:, 2] > 0
        f_j, f_q = s.fld[:, 1], s.fld[:, 2]

        carry, partner = s.carry, s.partner
        shaping = jnp.zeros(N, F)
        haul_len = _norm(goal - depot)

        def drop_penalty(x):
            # potential jump when a heavy crate is lost: undo the progress paid so far
            # and pre-pay the walk back, so drop-and-return loops earn no shaping
            return -C.HAUL_SHAPE * ((haul_len - _norm(x - goal)) + _norm(x - depot))

        if mission == "haul":
            pidx = jnp.maximum(partner, 0)
            brk = (partner >= 0) & ~(eng & eng[pidx] & alive0 & alive0[pidx])
            shaping = shaping + drop_penalty(s.pos) * (brk & alive0)
            carry = jnp.where(brk, 0.0, carry)
            partner = jnp.where(brk, -1, partner)
        paired = partner >= 0
        pidx = jnp.maximum(partner, 0)

        # ---- robot motion (effects use the in-situ fields the robots just observed)
        theta = jax.random.uniform(k_rot, (N,), F, -jnp.pi, jnp.pi) * f_j
        dirv = _rotate(cmd, theta)
        close = (_pairwise(s.pos) < C.R_CROWD) & alive0[None, :] & ~eye \
            & (partner[:, None] != idx[None, :])
        crowd = 1.0 / (1.0 + 0.5 * close.sum(1))
        load = jnp.where(carry > 0, C.LOAD_SLOW, 1.0)
        cap = C.V0 * (0.4 + 0.6 * s.h) * (1 - C.QUAKE_SLOW * f_q) * load * crowd
        slip = jax.random.normal(k_slip, (N, 2), F) * (C.QUAKE_SLIP * C.V0 * f_q)[:, None]
        if mission == "haul":  # a pair is one rigid body
            dirv = jnp.where(paired[:, None], 0.5 * (dirv + dirv[pidx]), dirv)
            cap = jnp.where(paired, jnp.minimum(cap, cap[pidx]), cap)
            slip = jnp.where(paired[:, None], 0.5 * (slip + slip[pidx]), slip)
        mobile = alive0 & (s.stuck == 0)
        pos1 = jnp.clip(s.pos + jnp.where(mobile[:, None], dirv * cap[:, None] + slip, 0.0), 0.0, 1.0)
        speed = _norm(pos1 - s.pos)

        # ---- blob motion
        kb1, kb2 = jax.random.split(k_blob)
        bdir = _rotate(s.bdir, jax.random.normal(kb1, (C.NB,), F) * C.NATURE_TURN_STD)
        u_adv = _unit_cap(jnp.clip(jnp.nan_to_num(adv_act), -1.0, 1.0))
        c = s.bpos + p.tempo * C.V0 * jnp.where(p.adversarial, u_adv, bdir)
        lo, hi = c < C.BLOB_LO, c > C.BLOB_HI
        c = jnp.where(lo, 2 * C.BLOB_LO - c, c)
        c = jnp.where(hi, 2 * C.BLOB_HI - c, c)
        bdir = jnp.where(lo | hi, -bdir, bdir)
        bpos1 = _project_safe(c, self._safe_anchors(s.anchors))

        fld1, grad1 = fields(pos1, bpos1, p.amp)
        d1 = _pairwise(pos1)
        coupled = jnp.zeros(N, bool)
        extra_dose = jnp.zeros(N, F)

        # ---- VIP (escort)
        vip_pos1, vip_h1, vip_stuck1 = s.vip_pos, s.vip_h, s.vip_stuck
        vip_arrived1, vip_fld1 = s.vip_arrived, s.vip_fld
        if mission == "escort":
            kv1, kv2, kv3 = jax.random.split(k_vip, 3)
            active = (s.vip_h > 0) & ~s.vip_arrived
            g = _unit(goal - s.vip_pos)
            dvec = s.vip_pos - s.vip_b_pos
            r = _norm(dvec)
            w = (s.vip_b_t >= 0) * jnp.exp(-(s.t - s.vip_b_t) / C.VIP_MEM) \
                * jnp.exp(-r ** 2 / (2 * (1.8 * C.SIGMA) ** 2))
            rad = dvec / jnp.maximum(r, 1e-6)[:, None]
            perp = jnp.stack([-rad[:, 1], rad[:, 0]], -1)
            perp = perp * jnp.where(perp @ g >= 0, 1.0, -1.0)[:, None]
            ahead = (rad @ g) < 0
            steer = g + jnp.sum(w[:, None] * (C.VIP_REP * rad + C.VIP_TAN * ahead[:, None] * perp), 0)
            steer = jnp.where(_norm(steer) < 1e-4, g, _unit(steer))
            steer = _rotate(steer, jax.random.uniform(kv1, (), F, -jnp.pi, jnp.pi) * s.vip_fld[1])
            vcap = C.VIP_SPEED * C.V0 * (0.4 + 0.6 * s.vip_h) * (1 - C.QUAKE_SLOW * s.vip_fld[2])
            vslip = jax.random.normal(kv2, (2,), F) * C.QUAKE_SLIP * C.V0 * s.vip_fld[2]
            vmob = active & (s.vip_stuck == 0)
            vip_pos1 = jnp.clip(s.vip_pos + jnp.where(vmob, steer * vcap + vslip, 0.0), 0.0, 1.0)
            vip_fld1, _ = fields(vip_pos1, bpos1, p.amp)
            near_v = alive0 & (_norm(pos1 - vip_pos1) < C.R_LINK) & p.coupling & active
            # shielding: up to two engaged escorts each halve the VIP's blob dose and absorb it
            shield = near_v & eng
            shield = shield & (jnp.cumsum(shield) <= 2)
            n_sh = shield.sum()
            blob_dose = C.RHO_C * vip_fld1[0]
            keep = 0.5 ** n_sh
            extra_dose = shield * blob_dose * (1 - keep) / jnp.maximum(n_sh, 1)
            vtrap = vmob & (jax.random.uniform(kv3, (), F) < C.P_TRAP * vip_fld1[2] ** 2)
            rescuer_v = near_v & (s.stuck == 0) & (s.vip_stuck > 0)
            vip_stuck1 = jnp.maximum(s.vip_stuck - 1 - C.RESCUE_RATE * rescuer_v.any(), 0)
            vip_stuck1 = jnp.where(vtrap, C.D_STUCK_VIP, vip_stuck1)
            vip_h1 = jnp.where(active, jnp.clip(
                s.vip_h - p.ambient - blob_dose * keep - C.TRAP_DAMAGE * vtrap, 0.0, 1.0), s.vip_h)
            vip_arrived1 = s.vip_arrived | (active & (vip_h1 > 0)
                                            & (_norm(vip_pos1 - goal) < C.R_ZONE))
            coupled = coupled | shield | rescuer_v

        # ---- hazard effects on robots
        dose = p.ambient + C.RHO_C * fld1[:, 0] + extra_dose
        trap = mobile & (jax.random.uniform(k_trap, (N,), F) < C.P_TRAP * fld1[:, 2] ** 2)
        free = mobile & ~trap
        near = (d1 < C.R_LINK) & ~eye
        was_stuck = alive0 & (s.stuck > 0)
        helper = (near & free[None, :]).any(1) & p.coupling
        stuck1 = jnp.maximum(s.stuck - 1 - C.RESCUE_RATE * helper, 0)
        stuck1 = jnp.where(trap, C.D_STUCK, stuck1)
        h1 = jnp.clip(jnp.where(alive0, s.h - dose - C.TRAP_DAMAGE * trap, s.h), 0.0, 1.0)
        alive1 = alive0 & (h1 > 0)
        died = alive0 & ~alive1
        stuck1 = jnp.where(alive1, stuck1, 0)
        coupled = coupled | (was_stuck & helper) \
            | (free & p.coupling & (near & was_stuck[None, :]).any(1))

        # ---- mission
        ind = jnp.zeros(N, F)
        found1, delivered1 = s.found, s.delivered
        if mission == "search":
            r_det = C.R_DET * (1 - C.JAM_SENSE * fld1[:, 1])
            dt = _norm(pos1[:, None, :] - s.targets[None, :, :])
            hit = alive1[:, None] & (dt < r_det[:, None]) & ~s.found[None, :]
            newly = hit.any(0)
            w_t = 1.0 - tf / C.T
            credit = (jnp.argmax(hit, 0)[None, :] == idx[:, None]) & newly[None, :]
            ind = C.SEARCH_REW * w_t * credit.sum(1)
            J1 = s.J + w_t * newly.sum() / C.M_TARGETS
            found1 = s.found | newly
        elif mission == "haul":
            sub = jnp.where((carry > 0)[:, None], goal, depot)
            shaping = shaping + C.HAUL_SHAPE * (_norm(s.pos - sub) - _norm(pos1 - sub)) * alive0
            brk = paired & ~(alive1 & alive1[pidx] & (stuck1 == 0) & (stuck1[pidx] == 0))
            shaping = shaping + drop_penalty(pos1) * (brk & alive1)
            lost = brk | ~alive1
            carry = jnp.where(lost, 0.0, carry)
            partner = jnp.where(lost, -1, partner)
            deliv = (carry > 0) & (_norm(pos1 - goal) < C.R_ZONE)
            ind = jnp.where(deliv, carry, 0.0)
            carry = jnp.where(deliv, 0.0, carry)
            partner = jnp.where(deliv, -1, partner)
            elig = alive1 & (stuck1 == 0) & (carry == 0) & ~deliv & (_norm(pos1 - depot) < C.R_ZONE)
            want = elig & eng & p.coupling
            dw = jnp.where(want[:, None] & want[None, :] & near, d1, INF)
            nn = jnp.argmin(dw, 1)
            has = dw.min(1) < INF
            mutual = has & has[nn] & (nn[nn] == idx)
            carry = jnp.where(elig & ~want, 1.0, carry)
            carry = jnp.where(mutual, 1.5, carry)
            partner = jnp.where(mutual, nn, partner)
            pos1 = jnp.where(mutual[:, None], 0.5 * (pos1 + pos1[nn]), pos1)
            delivered1 = s.delivered + ind.sum()
            J1 = delivered1 / C.HAUL_NORM
            coupled = coupled | (partner >= 0)
        else:  # escort
            prog = jnp.where(vip_arrived1, 1.0,
                             jnp.clip(1.0 - _norm(vip_pos1 - goal) / haul_len, 0.0, 1.0))
            J1 = prog * (0.3 + 0.7 * vip_h1)
        dJ = J1 - s.J

        # ---- training reward
        n0 = jnp.maximum(alive0.sum(), 1)
        if mission == "escort":
            rew = jnp.full(N, C.ESCORT_REW, F) * dJ
        else:
            rew = 0.5 * ind + 0.5 * ind.sum() / n0 + shaping
        n_died = died.sum()
        rew = rew - p.protect * C.DEATH_PEN * (died + 0.5 * (n_died - died))
        rew = jnp.where(alive0, rew, 0.0)

        s1 = s._replace(
            t=t1, pos=pos1, h=h1, present=present, alive=alive1, stuck=stuck1,
            fld=fld1, grad=grad1, carry=carry, partner=partner, coupled=coupled & alive1,
            bpos=bpos1, bdir=bdir, found=found1, delivered=delivered1,
            vip_pos=vip_pos1, vip_h=vip_h1, vip_stuck=vip_stuck1, vip_arrived=vip_arrived1,
            vip_fld=vip_fld1, J=J1, deaths=s.deaths + n_died)
        s1 = self._sense_comm(s1, k_comm, p)

        n1 = alive1.sum()
        den = jnp.maximum(n1, 1)
        am = alive1.astype(F)
        info = {
            "J": J1, "n_alive": n1.astype(F), "deaths": s1.deaths,
            "coupled_frac": (s1.coupled.sum() / den).astype(F),
            "exposure": jnp.sum(am * (fld1[:, 0] + fld1[:, 2])) / den,
            "exposure_c": jnp.sum(am * fld1[:, 0]) / den,
            "exposure_j": jnp.sum(am * fld1[:, 1]) / den,
            "exposure_q": jnp.sum(am * fld1[:, 2]) / den,
            "mean_speed": jnp.sum(am * speed) / den,
            "mean_health": jnp.sum(am * h1) / den,
            "stuck_frac": jnp.sum(am * (stuck1 > 0)) / den,
            "blob_pos": bpos1,
            "found": found1.sum().astype(F), "delivered": delivered1,
            "vip_h": vip_h1, "vip_arrived": vip_arrived1.astype(F),
            "vip_dist": _norm(vip_pos1 - goal) / haul_len,
        }
        rew_out = {"robot": rew, "adv": -dJ, "team": dJ}
        return s1, self._obs(s1, p), rew_out, t1 >= C.T, info

    # ---------------------------------------------------- sensing and comms
    def _sense_comm(self, s: EnvState, key, p: EnvParams) -> EnvState:
        N = C.N_MAX
        escort, search = self.mission == "escort", self.mission == "search"
        tf = s.t.astype(F)
        eye = jnp.eye(N, dtype=bool)
        f_j = s.fld[:, 1]
        r_s = C.R_SENSE * (1 - C.JAM_SENSE * f_j)

        # own sensing
        see_b = s.alive[:, None] & (_norm(s.pos[:, None] - s.bpos[None]) < r_s[:, None]) \
            & (p.amp > 0)[None, :]
        b_pos = jnp.where(see_b[..., None], s.bpos[None], s.b_pos)
        b_t = jnp.where(see_b, tf, s.b_t)
        dm = _pairwise(s.pos)
        see_m = s.alive[:, None] & s.present[None, :] & ((dm < r_s[:, None]) | eye)
        cur = jnp.concatenate([s.pos, (s.h * s.alive)[:, None]], -1)
        m_val = jnp.where(see_m[..., None], cur[None], s.m_val)
        m_t = jnp.where(see_m, tf, s.m_t)
        v_val, v_t, vip_b_pos, vip_b_t = s.v_val, s.v_t, s.vip_b_pos, s.vip_b_t
        vis = s.vis
        if escort:
            vip_alive = s.vip_h > 0
            vip_cur = jnp.concatenate([s.vip_pos, s.vip_h[None], (s.vip_stuck > 0)[None].astype(F),
                                       s.vip_fld])
            see_v = s.alive & (_norm(s.pos - s.vip_pos) < r_s)
            v_val = jnp.where(see_v[:, None], vip_cur[None], v_val)
            v_t = jnp.where(see_v, tf, v_t)
            r_v = C.R_SENSE_VIP * (1 - C.JAM_SENSE * s.vip_fld[1])
            see_vb = vip_alive & (_norm(s.vip_pos - s.bpos) < r_v) & (p.amp > 0)
            vip_b_pos = jnp.where(see_vb[:, None], s.bpos, vip_b_pos)
            vip_b_t = jnp.where(see_vb, tf, vip_b_t)
        if search:
            cells = jnp.asarray(CELL_CENTRES)
            vis = vis | (s.alive[:, None] & (_norm(s.pos[:, None] - cells[None]) < C.R_VISIT))

        # links: nodes are the robots (+ the VIP in escort); the hub is a virtual sender
        if escort:
            n_pos = jnp.concatenate([s.pos, s.vip_pos[None]])
            n_alive = jnp.concatenate([s.alive, vip_alive[None]])
            clear = 1 - jnp.concatenate([f_j, s.vip_fld[1][None]])
            dn = _pairwise(n_pos)
        else:
            n_alive, clear, dn = s.alive, 1 - f_j, dm
        nn = n_alive.shape[0]
        u = jax.random.uniform(key, (nn, nn + 1), F)
        ok = 1 - C.P_LOSS_AMB
        link_g = (p.comm == 1) & n_alive[:, None] & n_alive[None, :] & (dn < C.R_COMM) \
            & ~jnp.eye(nn, dtype=bool) & (u[:, :nn] < ok * clear[:, None] * clear[None, :])
        up = (p.comm == 2) & n_alive & (u[:, nn] < ok * clear)
        link = jnp.concatenate([link_g, up[:, None]], 1)   # receivers x (nodes + hub)

        # blob table (robots + VIP)
        nb_t, nb_v = b_t, b_pos
        if escort:
            nb_t = jnp.concatenate([b_t, vip_b_t[None]])
            nb_v = jnp.concatenate([b_pos, vip_b_pos[None]])
        hub_b_t, hub_b_pos = _merge(s.hub_b_t[None], s.hub_b_pos[None], nb_t, nb_v, up[None, :])
        nb_t2, nb_v2 = _merge(nb_t, nb_v, jnp.concatenate([nb_t, hub_b_t]),
                              jnp.concatenate([nb_v, hub_b_pos]), link)
        b_t, b_pos = nb_t2[:N], nb_v2[:N]
        if escort:
            vip_b_t, vip_b_pos = nb_t2[N], nb_v2[N]

        # teammate table (robots only)
        hub_m_t, hub_m_val = _merge(s.hub_m_t[None], s.hub_m_val[None], m_t, m_val, up[None, :N])
        link_r = jnp.concatenate([link_g[:N, :N], up[:N, None]], 1)
        m_t, m_val = _merge(m_t, m_val, jnp.concatenate([m_t, hub_m_t]),
                            jnp.concatenate([m_val, hub_m_val]), link_r)

        hub_v_t, hub_v_val, hub_vis = s.hub_v_t, s.hub_v_val, s.hub_vis
        if escort:  # VIP state table: one entry; the VIP node always knows itself
            sv_t = jnp.concatenate([v_t, tf[None]])[:, None]
            sv_v = jnp.concatenate([v_val, vip_cur[None]])[:, None, :]
            hv_t, hv_v = _merge(s.hub_v_t[None, None], s.hub_v_val[None, None], sv_t, sv_v, up[None, :])
            rv_t, rv_v = _merge(sv_t[:N], sv_v[:N], jnp.concatenate([sv_t, hv_t]),
                                jnp.concatenate([sv_v, hv_v]), link[:N])
            v_t, v_val = rv_t[:, 0], rv_v[:, 0]
            hub_v_t, hub_v_val = hv_t[0, 0], hv_v[0, 0]
        if search:
            heard = (link_g.astype(F) @ vis.astype(F)) > 0
            hub_vis = s.hub_vis | (up[:, None] & vis).any(0)
            vis = vis | heard | (up[:, None] & hub_vis[None, :])

        return s._replace(
            b_pos=b_pos, b_t=b_t, m_val=m_val, m_t=m_t, v_val=v_val, v_t=v_t, vis=vis,
            hub_b_pos=hub_b_pos[0], hub_b_t=hub_b_t[0], hub_m_val=hub_m_val[0],
            hub_m_t=hub_m_t[0], hub_v_val=hub_v_val, hub_v_t=hub_v_t, hub_vis=hub_vis,
            vip_b_pos=vip_b_pos, vip_b_t=vip_b_t)

    # ------------------------------------------------------------ observations
    def _obs(self, s: EnvState, p: EnvParams):
        return {"robot": self._robot_obs(s, p), "robot_mask": s.alive,
                "adv": self._adv_obs(s, p), "adv_mask": p.amp > 0}

    def _robot_obs(self, s: EnvState, p: EnvParams):
        N = C.N_MAX
        tf = s.t.astype(F)
        idx = jnp.arange(N)
        eye = jnp.eye(N, dtype=bool)
        age01 = lambda age: jnp.clip(age / C.AGE_SCALE, 0.0, 1.0)

        # teammates
        mrel = s.m_val[..., :2] - s.pos[:, None, :]
        mage = tf - s.m_t
        mk = (s.m_t >= 0) & (s.m_val[..., 2] > 0) & (mage <= C.MATE_MAX_AGE) & ~eye
        md = jnp.where(mk, _norm(mrel), INF)
        j1 = jnp.argmin(md, 1)
        j2 = jnp.argmin(jnp.where(idx[None, :] == j1[:, None], INF, md), 1)

        def mate(j):
            ok = (md[idx, j] < INF).astype(F)
            return jnp.concatenate([mrel[idx, j], age01(mage[idx, j])[:, None],
                                    s.m_val[idx, j, 2:3], jnp.ones((N, 1), F)], -1) * ok[:, None]

        n_mk = mk.sum(1)
        centroid = jnp.sum(mk[..., None] * mrel, 1) / jnp.maximum(n_mk, 1)[:, None]

        self_blk = jnp.concatenate([
            s.h[:, None], s.pos * 2 - 1, jnp.full((N, 1), tf / C.T),
            (s.stuck > 0)[:, None].astype(F), ((1 + n_mk) / N)[:, None].astype(F),
            s.coupled[:, None].astype(F), s.fld, (s.grad * C.SIGMA).reshape(N, 6)], -1)

        # blobs
        brel = s.b_pos - s.pos[:, None, :]
        known = s.b_t >= 0
        bage = tf - s.b_t
        bd = _norm(brel)
        bdm = jnp.where(known, bd, INF).reshape(N, C.N_TYPES, C.K)
        kk = jnp.argmin(bdm, -1)                                    # [N,3]
        flat = kk + jnp.arange(C.N_TYPES)[None, :] * C.K
        any_known = (bdm.min(-1) < INF).astype(F)
        near_rel = jnp.take_along_axis(brel, flat[..., None], 1) * any_known[..., None]
        near_age = jnp.where(any_known > 0, age01(jnp.take_along_axis(bage, flat, 1)), 1.0)
        blob_near = jnp.concatenate([near_rel, near_age[..., None], any_known[..., None]], -1)
        w = known * jnp.exp(-bage / C.REP_MEM) * jnp.exp(-bd ** 2 / (2 * (2 * C.SIGMA) ** 2))
        rep = (w[..., None] * (-brel) / jnp.maximum(bd, 1e-3)[..., None]) \
            .reshape(N, C.N_TYPES, C.K, 2).sum(2)

        mate_blk = jnp.concatenate([mate(j1), mate(j2), centroid], -1)

        # mission block
        depot, goal = s.anchors[0], s.anchors[1]
        pad = lambda x: jnp.concatenate([x, jnp.zeros((N, 16 - x.shape[1]), F)], -1)
        cpl = jnp.broadcast_to(p.coupling.astype(F), (N, 1))
        if self.mission == "search":
            cells = jnp.asarray(CELL_CENTRES)
            unv = ~s.vis
            crel = cells[None] - s.pos[:, None, :]
            cd = _norm(crel)
            jn = jnp.argmin(jnp.where(unv, cd, INF), 1)
            has = unv.any(1).astype(F)
            n_unv = unv.sum(1)
            cent = jnp.sum(unv[..., None] * crel, 1) / jnp.maximum(n_unv, 1)[:, None]
            ci = jnp.clip(jnp.floor(s.pos * C.GRID).astype(jnp.int32), 0, C.GRID - 1)
            off = jnp.asarray([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], jnp.int32)
            nc = ci[:, None, :] + off[None]                          # [N,9,2]
            inb = jnp.all((nc >= 0) & (nc < C.GRID), -1)
            flat_c = jnp.clip(nc[..., 0], 0, C.GRID - 1) * C.GRID + jnp.clip(nc[..., 1], 0, C.GRID - 1)
            nbr = jnp.where(inb, jnp.take_along_axis(s.vis, flat_c, 1), True).astype(F)
            mis = jnp.concatenate([crel[idx, jn] * has[:, None], (cd[idx, jn] * has)[:, None],
                                   has[:, None], cent, (n_unv / G2)[:, None].astype(F), nbr], -1)
        elif self.mission == "haul":
            d_dep, d_goal = _norm(depot - s.pos), _norm(goal - s.pos)
            heavy = s.partner >= 0
            n_link = jnp.clip((md < C.R_LINK).sum(1) / 3.0, 0.0, 1.0)
            mis = pad(jnp.concatenate([
                depot - s.pos, goal - s.pos,
                ((s.carry > 0) & ~heavy)[:, None].astype(F), heavy[:, None].astype(F),
                (d_dep < C.R_ZONE)[:, None].astype(F), (d_goal < C.R_ZONE)[:, None].astype(F),
                n_link[:, None], cpl], -1))
        else:
            vk = (s.v_t >= 0).astype(F)[:, None]
            vpos = s.v_val[:, :2]
            vrel = (vpos - s.pos) * vk
            to_dest = (goal - vpos) * vk
            remain = jnp.clip(_norm(goal - vpos) / _norm(goal - depot), 0.0, 1.0)[:, None] * vk
            mis = jnp.concatenate([
                vrel, jnp.where(vk > 0, age01(tf - s.v_t)[:, None], 1.0), vk,
                s.v_val[:, 2:] * vk, to_dest, goal - s.pos, remain,
                ((_norm(vpos - s.pos) < C.R_LINK)[:, None] & (vk > 0)).astype(F), cpl], -1)

        obs = jnp.concatenate([self_blk, blob_near.reshape(N, 12), rep.reshape(N, 6),
                               mate_blk, mis], -1)
        return jnp.where(s.alive[:, None], obs, 0.0)

    def _adv_obs(self, s: EnvState, p: EnvParams):
        NB = C.NB
        ib = jnp.arange(NB)
        rel = s.pos[None, :, :] - s.bpos[:, None, :]                 # [NB,N,2]
        d = jnp.where(s.alive[None, :], _norm(rel), INF)
        j1 = jnp.argmin(d, 1)
        j2 = jnp.argmin(jnp.where(jnp.arange(C.N_MAX)[None, :] == j1[:, None], INF, d), 1)
        pick = lambda j: rel[ib, j] * (d[ib, j] < INF)[:, None]
        n = jnp.maximum(s.alive.sum(), 1)
        cen = (jnp.sum(s.alive[:, None] * s.pos, 0) / n)[None] - s.bpos
        cen = cen * (s.alive.sum() > 0)
        vip = (s.vip_pos[None] - s.bpos) if self.mission == "escort" else jnp.zeros((NB, 2), F)
        types = jax.nn.one_hot(ib // C.K, C.N_TYPES, dtype=F)
        return jnp.concatenate([
            s.bpos * 2 - 1, types, pick(j1), pick(j2), cen,
            s.anchors[0][None] - s.bpos, s.anchors[1][None] - s.bpos, vip,
            jnp.full((NB, 1), s.t / C.T, F), p.amp[:, None],
            jnp.broadcast_to(p.tempo, (NB, 1))], -1)


# ----------------------------------------------------------------- helpers
def stack_params(plist):
    """Stack a list of EnvParams into one batched EnvParams (leading axis = env)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *plist)


def zero_adv_action(obs_adv=None, key=None):
    """Adversary action placeholder (ignored in nature envs, parks blobs in adversarial ones)."""
    return jnp.zeros((C.NB, 2), F)


def rollout(env: HostileEnv, key, params: EnvParams, policy_fn, adv_fn=None):
    """Run one full episode. ``policy_fn(obs_robot, key) -> [N_MAX,3]``,
    ``adv_fn(obs_adv, key) -> [NB,2]`` (default: zeros). Returns the final state and
    the per-step info dict stacked over T (plus ``rew_robot`` [T,N_MAX] and ``rew_team`` [T]).
    vmap over ``(key, params)`` for batches."""
    adv_fn = zero_adv_action if adv_fn is None else adv_fn
    k0, k1 = jax.random.split(key)
    state, obs = env.reset(k0, params)

    def body(carry, k):
        state, obs = carry
        ka, kb, ks = jax.random.split(k, 3)
        state, obs, rew, _, info = env.step(
            ks, state, policy_fn(obs["robot"], ka), adv_fn(obs["adv"], kb), params)
        return (state, obs), dict(info, rew_robot=rew["robot"], rew_team=rew["team"])

    (state, _), infos = lax.scan(body, (state, obs), jax.random.split(k1, C.T))
    return state, infos
