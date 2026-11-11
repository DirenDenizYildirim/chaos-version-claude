"""Simulator tests: API contract, determinism and one unit check per mechanic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hostile_planets import constants as C
from hostile_planets.env import (ADV_OBS_DIM, OBS, OBS_DIM, HostileEnv, fields, rollout,
                                 stack_params)
from hostile_planets.heuristics import (HEURISTICS, haul_pair_policy, idle_policy,
                                        random_policy)
from hostile_planets.planets import PLANETS

N = C.N_MAX
CENTRE = jnp.array([0.5, 0.5])
FAR = jnp.array([0.5, 0.93])
ENVS = {m: HostileEnv(m) for m in C.MISSIONS}


def rand_adv(obs, key):
    return jax.random.uniform(key, (C.NB, 2), minval=-1.0, maxval=1.0)


def place(env, s, p, pos=None, bpos=None, **kw):
    """Overwrite robot / blob positions in a state and refresh the in-situ fields."""
    pos = s.pos if pos is None else jnp.asarray(pos, jnp.float32)
    bpos = s.bpos if bpos is None else jnp.asarray(bpos, jnp.float32)
    fld, grad = fields(pos, bpos, p.amp)
    return s._replace(pos=pos, bpos=bpos, fld=fld, grad=grad, **kw)


def all_blobs(at):
    return jnp.tile(jnp.asarray(at, jnp.float32)[None], (C.NB, 1))


def run_steps(env, s, p, act, n, seed=0, adv=None):
    """Step n times with a constant robot action; returns final state and stacked (state, rew, info)."""
    adv = jnp.zeros((C.NB, 2)) if adv is None else adv

    def body(s, k):
        s, _, rew, _, info = env.step(k, s, act, adv, p)
        return s, (s, rew, info)

    return jax.jit(lambda s: jax.lax.scan(body, s, jax.random.split(jax.random.PRNGKey(seed), n)))(s)


IDLE = jnp.zeros((N, 3)).at[:, 2].set(-1.0)
ENGAGE = jnp.zeros((N, 3)).at[:, 2].set(1.0)


# ------------------------------------------------------------------ contract
@pytest.mark.parametrize("mission", C.MISSIONS)
def test_shapes_and_finite_all_configs(mission):
    env = ENVS[mission]
    plist = [env.make_params(planet=pl, comm=cm, n_robots=n, adversarial=adv, tempo=1.0,
                             coupling=bool(n % 4), protect=adv)
             for pl in PLANETS for cm in C.COMM_MODES for n, adv in ((3, False), (12, True))]
    ps = stack_params(plist)
    B = len(plist)
    keys = jax.random.split(jax.random.PRNGKey(1), B)

    def episode(k, p):
        k0, k1 = jax.random.split(k)
        s, obs = env.reset(k0, p)

        def body(c, kk):
            s, obs = c
            ka, kb, ks = jax.random.split(kk, 3)
            s, obs, rew, done, info = env.step(ks, s, random_policy(obs["robot"], ka),
                                               rand_adv(obs["adv"], kb), p)
            return (s, obs), (obs, rew, done, info)

        return jax.lax.scan(body, (s, obs), jax.random.split(k1, C.T))[1]

    obs, rew, done, info = jax.jit(jax.vmap(episode))(keys, ps)
    assert obs["robot"].shape == (B, C.T, N, OBS_DIM) == (B, C.T, N, env.obs_dim)
    assert obs["adv"].shape == (B, C.T, C.NB, ADV_OBS_DIM) == (B, C.T, C.NB, env.adv_obs_dim)
    assert obs["robot_mask"].shape == (B, C.T, N) and obs["adv_mask"].shape == (B, C.T, C.NB)
    assert rew["robot"].shape == (B, C.T, N) and rew["adv"].shape == (B, C.T)
    for tree in (obs, rew, info):
        for leaf in jax.tree_util.tree_leaves(tree):
            assert bool(jnp.all(jnp.isfinite(leaf.astype(jnp.float32))))
    assert float(jnp.abs(obs["robot"]).max()) < 5.0
    assert float(jnp.abs(obs["adv"]).max()) < 5.0
    assert bool(jnp.all(done[:, -1])) and not bool(jnp.any(done[:, :-1]))
    np.testing.assert_allclose(np.asarray(rew["adv"]), -np.asarray(rew["team"]))
    J = np.asarray(info["J"])
    assert J.min() >= 0.0 and J.max() <= 1.0 + 1e-5
    # the metric is the running sum of the adversary's (negated) reward
    np.testing.assert_allclose(-np.asarray(rew["adv"]).sum(1), J[:, -1], atol=1e-4)
    # dead / absent slots carry zero obs and zero reward
    dead = ~np.asarray(obs["robot_mask"])
    assert np.all(np.asarray(obs["robot"])[dead] == 0.0)
    assert np.all(np.asarray(info["n_alive"]) <= np.asarray(ps.n_robots)[:, None])


@pytest.mark.parametrize("mission", C.MISSIONS)
def test_determinism(mission):
    env = ENVS[mission]
    p = env.make_params(planet="combined", adversarial=True)
    f = jax.jit(lambda k: rollout(env, k, p, random_policy, rand_adv)[1])
    a, b, c = f(jax.random.PRNGKey(3)), f(jax.random.PRNGKey(3)), f(jax.random.PRNGKey(4))
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
    assert not np.array_equal(np.asarray(a["blob_pos"]), np.asarray(c["blob_pos"]))


@pytest.mark.parametrize("mission", C.MISSIONS)
def test_absent_slots_never_contribute(mission):
    env = ENVS[mission]
    p = env.make_params(planet="combined", n_robots=3, comm="gossip")

    def pol(sign):
        def f(obs, key):
            act = HEURISTICS[mission](obs, key)
            return jnp.where((jnp.arange(N) < 3)[:, None], act, sign)
        return f

    f = jax.jit(lambda k, sign: rollout(env, k, p, pol(sign)))
    (sa, ia), (sb, ib) = f(jax.random.PRNGKey(0), 1.0), f(jax.random.PRNGKey(0), -1.0)
    for k in ia:
        np.testing.assert_array_equal(np.asarray(ia[k]), np.asarray(ib[k]))
    assert np.all(np.asarray(ia["rew_robot"])[:, 3:] == 0.0)
    assert np.all(np.asarray(sa.m_t)[:3, 3:] == -1.0)       # nobody ever "knows" an absent slot
    assert float(ia["n_alive"].max()) <= 3.0


@pytest.mark.parametrize("mission", C.MISSIONS)
def test_blobs_stay_out_of_safe_zones(mission):
    env = ENVS[mission]
    ps = stack_params([env.make_params(planet="combined", tempo=1.5, adversarial=bool(i % 2))
                       for i in range(32)])

    def chase_anchor(obs, key):   # adversary pushes every blob at an anchor
        return jnp.where((jnp.arange(C.NB) % 2 == 0)[:, None], obs[:, 11:13], obs[:, 13:15]) * 100

    s, info = jax.jit(jax.vmap(lambda k, p: rollout(env, k, p, random_policy, chase_anchor)))(
        jax.random.split(jax.random.PRNGKey(5), 32), ps)
    bp = np.asarray(info["blob_pos"])                       # [B,T,NB,2]
    assert bp.min() >= C.BLOB_LO - 1e-6 and bp.max() <= C.BLOB_HI + 1e-6
    anchors = np.asarray(s.anchors)                         # [B,2,2]
    for a in range(1 if mission == "search" else 2):
        d = np.linalg.norm(bp - anchors[:, None, None, a, :], axis=-1)
        assert d.min() >= C.R_SAFE - 1e-4


# ------------------------------------------------------------------ hazards
def test_corrosion_kills_in_blob_core():
    env = ENVS["search"]
    p = env.make_params(planet="corrosion", n_robots=4, tempo=0.0)
    s, _ = env.reset(jax.random.PRNGKey(0), p)
    s = place(env, s, p, pos=jnp.tile(CENTRE[None], (N, 1)), bpos=all_blobs(CENTRE))
    s1, (traj, rew, info) = run_steps(env, s, p, IDLE, 60)
    assert not bool(s1.alive.any()) and float(s1.deaths) == 4.0
    h = np.asarray(traj.h)[:, :4]
    assert np.all(np.diff(h, axis=0) <= 1e-7)               # health never recovers
    assert float(info["n_alive"][-1]) == 0.0
    # outside any blob only the ambient rate applies
    q = p._replace(amp=jnp.zeros(C.NB))
    s2, _ = run_steps(env, place(env, s, q), q, IDLE, 40)
    np.testing.assert_allclose(np.asarray(s2.h)[:4], 1 - 40 * C.RHO_AMB, atol=1e-5)


def test_death_penalty_only_in_protect_arm():
    env = ENVS["search"]
    out = {}
    for protect in (False, True):
        p = env.make_params(planet="corrosion", n_robots=4, tempo=0.0, protect=protect)
        s, _ = env.reset(jax.random.PRNGKey(0), p)
        s = place(env, s, p, pos=jnp.tile(CENTRE[None], (N, 1)), bpos=all_blobs(CENTRE),
                  targets=jnp.zeros((C.M_TARGETS, 2)))
        _, (_, rew, _) = run_steps(env, s, p, IDLE, 60)
        out[protect] = float(rew["robot"].sum())
    assert out[False] == 0.0
    # all four die on the same step: each pays DEATH_PEN + 0.5*DEATH_PEN per other death
    np.testing.assert_allclose(out[True], -4 * C.DEATH_PEN * (1 + 0.5 * 3), rtol=1e-5)


def test_jam_shrinks_sensing():
    env = ENVS["search"]
    p = env.make_params(planet="combined", n_robots=1, tempo=0.0, comm="silent")
    s, _ = env.reset(jax.random.PRNGKey(0), p)
    pos = jnp.tile(CENTRE[None], (N, 1))
    corr = CENTRE + jnp.array([0.12, 0.0])                  # a corrosion blob 0.12 away
    for jam_at, seen in ((FAR, True), (CENTRE, False)):
        bpos = all_blobs(FAR).at[0].set(corr).at[C.K].set(jam_at)
        s1 = place(env, s, p, pos=pos, bpos=bpos, b_t=-jnp.ones((N, C.NB)))
        s1 = env._sense_comm(s1, jax.random.PRNGKey(1), p)
        assert bool(s1.b_t[0, 0] >= 0) == seen
    # target detection shrinks too: a target 0.03 away is found only without the jam
    for jam_at, found in ((FAR, 1.0), (CENTRE, 0.0)):
        bpos = all_blobs(FAR).at[C.K].set(jam_at)
        tg = jnp.zeros((C.M_TARGETS, 2)).at[0].set(CENTRE + jnp.array([0.0, 0.03]))
        s1 = place(env, s, p._replace(amp=p.amp.at[:C.K].set(0.0)), pos=pos, bpos=bpos, targets=tg)
        _, (_, _, info) = run_steps(env, s1, p._replace(amp=p.amp.at[:C.K].set(0.0)), IDLE, 1)
        assert float(info["found"][-1]) == found


@pytest.mark.parametrize("comm", ["gossip", "blackboard"])
def test_jam_drops_packets(comm):
    """Robot 1 can only learn about robot 2 through robot 0; a jam on robot 1 blocks that."""
    env = ENVS["search"]
    p = env.make_params(planet="jam", n_robots=3, tempo=0.0, comm=comm)
    s, _ = env.reset(jax.random.PRNGKey(0), p)
    pos = jnp.tile(jnp.array([[0.3, 0.3]]), (N, 1)).at[1].set(jnp.array([0.55, 0.3])) \
        .at[2].set(jnp.array([0.2, 0.3]))                   # 0-1: 0.25 apart, 1-2: 0.35 apart
    for jam_at, heard in ((FAR, True), (pos[1], False)):
        s1 = place(env, s, p, pos=pos, bpos=all_blobs(FAR).at[C.K].set(jam_at),
                   m_t=-jnp.ones((N, N)), hub_m_t=-jnp.ones(N))
        s1, _ = run_steps(env, s1, p, IDLE, 6)
        assert bool(s1.m_t[0, 2] >= 0)                      # robot 0 senses robot 2 directly
        assert bool(s1.m_t[1, 2] >= 0) == heard


def test_silent_mode_only_own_sensing():
    """In silent mode a belief entry is fresh only if its owner sensed it; gossip breaks this."""
    env = ENVS["escort"]

    def fresh_violations(comm):
        p = env.make_params(planet="combined", n_robots=8, comm=comm)
        k0, k1 = jax.random.split(jax.random.PRNGKey(2))
        s, obs = env.reset(k0, p)

        def body(c, k):
            s, obs = c
            ka, ks = jax.random.split(k)
            s, obs, *_ = env.step(ks, s, random_policy(obs["robot"], ka), jnp.zeros((C.NB, 2)), p)
            t = s.t.astype(jnp.float32)
            db = jnp.linalg.norm(s.pos[:, None] - s.bpos[None], axis=-1)
            dm = jnp.linalg.norm(s.pos[:, None] - s.pos[None], axis=-1)
            dv = jnp.linalg.norm(s.pos - s.vip_pos, axis=-1)
            bad = jnp.sum((s.b_t == t) & (db >= C.R_SENSE)) \
                + jnp.sum((s.m_t == t) & (dm >= C.R_SENSE) & ~jnp.eye(N, dtype=bool)) \
                + jnp.sum((s.v_t == t) & (dv >= C.R_SENSE)) \
                + jnp.sum((s.vip_b_t == t) & (jnp.linalg.norm(s.vip_pos - s.bpos, axis=-1)
                                              >= C.R_SENSE_VIP))
            return (s, obs), (bad, s.hub_b_t.max(), s.hub_m_t.max(), s.hub_v_t)

        return jax.jit(lambda: jax.lax.scan(body, (s, obs), jax.random.split(k1, C.T))[1])()

    bad, hb, hm, hv = fresh_violations("silent")
    assert int(bad.sum()) == 0
    assert float(hb.max()) == -1.0 and float(hm.max()) == -1.0 and float(hv.max()) == -1.0
    assert int(fresh_violations("gossip")[0].sum()) > 0
    bad, hb, hm, hv = fresh_violations("blackboard")
    assert int(bad.sum()) > 0 and float(hb.max()) >= 0 and float(hm.max()) >= 0


def test_search_visited_grid_only_shared_by_comm():
    env = ENVS["search"]
    for comm, shared in (("silent", False), ("gossip", True), ("blackboard", True)):
        p = env.make_params(planet="jam", amp_scale=0.0, n_robots=2, comm=comm)
        s, _ = env.reset(jax.random.PRNGKey(0), p)
        pos = jnp.tile(jnp.array([[0.3125, 0.3125]]), (N, 1)).at[1].set(jnp.array([0.5625, 0.3125]))
        s = place(env, s, p, pos=pos, vis=jnp.zeros_like(s.vis), hub_vis=jnp.zeros_like(s.hub_vis))
        s, _ = run_steps(env, s, p, IDLE, 5)
        own0, own1 = 2 * C.GRID + 2, 4 * C.GRID + 2
        assert bool(s.vis[0, own0]) and bool(s.vis[1, own1])
        assert bool(s.vis[0, own1]) == shared and bool(s.vis[1, own0]) == shared


def test_quake_traps_and_rescue_needs_coupling():
    env = ENVS["search"]
    p = env.make_params(planet="quake", n_robots=1, tempo=0.0)
    s, _ = env.reset(jax.random.PRNGKey(0), p)
    s = place(env, s, p, pos=jnp.tile(CENTRE[None], (N, 1)), bpos=all_blobs(CENTRE))
    s1, (traj, _, info) = run_steps(env, s, p, IDLE, 60)
    stuck = np.asarray(traj.stuck)[:, 0]
    assert stuck.max() == C.D_STUCK and float(info["stuck_frac"].max()) == 1.0
    first = int(np.argmax(stuck > 0))
    assert np.isclose(float(traj.h[first, 0]), 1 - C.TRAP_DAMAGE)
    held = (stuck[1:] > 0) & (stuck[1:] < stuck[:-1])       # still inside the same trap episode
    x = np.asarray(traj.pos)[:, 0]
    assert held.sum() > 10 and np.all(x[1:][held] == x[:-1][held])

    # rescue: a stuck robot with a free teammate 0.03 away, no hazard around
    steps_alone = C.D_STUCK
    steps_helped = -(-C.D_STUCK // (1 + C.RESCUE_RATE))
    for coupling, expect in ((True, steps_helped), (False, steps_alone)):
        q = env.make_params(planet="quake", amp_scale=0.0, n_robots=2, coupling=coupling)
        s, _ = env.reset(jax.random.PRNGKey(0), q)
        pos = jnp.tile(CENTRE[None], (N, 1)).at[1].set(CENTRE + jnp.array([0.03, 0.0]))
        s = place(env, s, q, pos=pos, stuck=s.stuck.at[0].set(C.D_STUCK))
        _, (traj, _, info) = run_steps(env, s, q, IDLE, C.D_STUCK + 2)
        freed_after = int(np.argmax(np.asarray(traj.stuck)[:, 0] == 0)) + 1
        assert freed_after == expect
        assert (float(info["coupled_frac"][0]) == 1.0) == coupling


# ------------------------------------------------------------------ missions
def test_search_metric():
    env = ENVS["search"]
    p = env.make_params(planet="jam", amp_scale=0.0, n_robots=1)
    s, _ = env.reset(jax.random.PRNGKey(0), p)
    tg = jnp.zeros((C.M_TARGETS, 2)).at[0].set(CENTRE + jnp.array([0.02, 0.0])) \
        .at[1].set(CENTRE + jnp.array([0.2, 0.0]))
    s = place(env, s, p, pos=jnp.tile(CENTRE[None], (N, 1)), targets=tg)
    s1, (_, rew, info) = run_steps(env, s, p, IDLE, 3)
    w = 1 - 1 / C.T
    np.testing.assert_allclose(float(info["J"][-1]), w / C.M_TARGETS, rtol=1e-5)
    assert float(info["found"][-1]) == 1.0
    np.testing.assert_allclose(float(rew["robot"][0, 0]), C.SEARCH_REW * w, rtol=1e-5)


def test_haul_solo_and_heavy_pair():
    env = ENVS["haul"]
    for coupling, policy, unit, frac in ((True, haul_pair_policy, 3.0, 0.5),
                                         (True, HEURISTICS["haul"], 1.0, 0.0),
                                         (False, haul_pair_policy, 1.0, 0.0)):
        p = env.make_params(planet="jam", amp_scale=0.0, n_robots=2, coupling=coupling)
        _, info = jax.jit(lambda k: rollout(env, k, p, policy))(jax.random.PRNGKey(0))
        d = np.asarray(info["delivered"])
        jumps = np.diff(d)[np.diff(d) > 0]
        assert len(jumps) >= 1
        if unit == 3.0:
            np.testing.assert_allclose(jumps, 3.0)          # a pair delivers 3 units at once
        else:
            assert np.all(np.isin(jumps, [1.0, 2.0])) and d[-1] >= 2.0
        np.testing.assert_allclose(float(info["J"][-1]), d[-1] / C.HAUL_NORM, rtol=1e-6)
        assert (float(info["coupled_frac"].mean()) > frac) == (unit == 3.0)


def test_haul_pair_breaks_on_disengage():
    env = ENVS["haul"]
    p = env.make_params(planet="jam", amp_scale=0.0, n_robots=2)
    s, _ = env.reset(jax.random.PRNGKey(0), p)
    s, _ = run_steps(env, s, p, ENGAGE, 2)
    assert np.all(np.asarray(s.partner)[:2] == [1, 0]) and np.all(np.asarray(s.carry)[:2] == 1.5)
    np.testing.assert_array_equal(np.asarray(s.pos[0]), np.asarray(s.pos[1]))
    go = ENGAGE.at[:, 0].set(1.0)
    s, _ = run_steps(env, s, p, go, 5)                      # rigid body: positions stay identical
    np.testing.assert_array_equal(np.asarray(s.pos[0]), np.asarray(s.pos[1]))
    assert float(s.pos[0, 0]) > float(s.anchors[0, 0]) + 0.05
    s2, (_, rew, _) = run_steps(env, s, p, go.at[0, 2].set(-1.0), 1)
    assert np.all(np.asarray(s2.partner)[:2] == -1) and np.all(np.asarray(s2.carry)[:2] == 0.0)
    assert np.all(np.asarray(rew["robot"])[0, :2] < 0)      # the drop refunds the shaping


def test_escort_vip_arrives_and_shield_halves_dose():
    env = ENVS["escort"]
    p = env.make_params(planet="jam", amp_scale=0.0, n_robots=2)
    _, info = jax.jit(lambda k: rollout(env, k, p, idle_policy))(jax.random.PRNGKey(0))
    assert 0.7 < float(info["J"][-1]) < 0.9 and float(info["vip_arrived"][-1]) == 1.0  # time-averaged progress
    assert np.all(np.diff(np.asarray(info["J"])) >= -1e-6)

    p = env.make_params(planet="corrosion", n_robots=2, tempo=0.0)
    s, _ = env.reset(jax.random.PRNGKey(0), p)
    pos = jnp.tile(FAR[None], (N, 1))
    loss = {}
    for n_sh in (0, 1, 2):
        pos_n = pos.at[:n_sh].set(CENTRE)
        s0 = place(env, s, p, pos=pos_n, bpos=all_blobs(FAR).at[0].set(CENTRE), vip_pos=CENTRE)
        s1, _ = run_steps(env, s0, p, ENGAGE, 1)
        blob = C.RHO_C * float(s1.vip_fld[0])
        loss[n_sh] = 1.0 - float(s1.vip_h) - float(p.ambient)
        np.testing.assert_allclose(loss[n_sh], blob * 0.5 ** n_sh, rtol=1e-3)
        if n_sh:
            # shielders absorb the diverted dose on top of their own
            own = C.RHO_C * np.asarray(s1.fld)[:n_sh, 0] + float(p.ambient)
            np.testing.assert_allclose(1 - np.asarray(s1.h)[:n_sh],
                                       own + blob * (1 - 0.5 ** n_sh) / n_sh, rtol=1e-3)
    # no shielding without coupling or without engage
    s0 = place(env, s, p, pos=pos.at[:2].set(CENTRE), bpos=all_blobs(FAR).at[0].set(CENTRE),
               vip_pos=CENTRE)
    for q, act in ((p._replace(coupling=jnp.asarray(False)), ENGAGE), (p, IDLE)):
        s1, _ = run_steps(env, s0, q, act, 1)
        np.testing.assert_allclose(1.0 - float(s1.vip_h) - float(p.ambient), loss[0], rtol=1e-3)


def test_escort_vip_rescue():
    env = ENVS["escort"]
    freed = {}
    for coupling in (True, False):
        p = env.make_params(planet="jam", amp_scale=0.0, n_robots=1, coupling=coupling)
        s, _ = env.reset(jax.random.PRNGKey(0), p)
        s = place(env, s, p, pos=jnp.tile(s.vip_pos[None], (N, 1)),
                  vip_stuck=jnp.asarray(C.D_STUCK_VIP, jnp.int32))
        _, (traj, _, _) = run_steps(env, s, p, IDLE, C.D_STUCK_VIP + 2)
        freed[coupling] = int(np.argmax(np.asarray(traj.vip_stuck) == 0)) + 1
    assert freed[False] == C.D_STUCK_VIP
    assert freed[True] == -(-C.D_STUCK_VIP // (1 + C.RESCUE_RATE))


@pytest.mark.parametrize("mission", C.MISSIONS)
def test_remove_t_removes_one_robot(mission):
    env = ENVS[mission]
    p = env.make_params(planet="jam", amp_scale=0.0, n_robots=5, remove_t=20)
    s, info = jax.jit(lambda k: rollout(env, k, p, HEURISTICS[mission]))(jax.random.PRNGKey(0))
    n = np.asarray(info["n_alive"])
    assert np.all(n[:20] == 5) and np.all(n[20:] == 4)      # step index 20 is the 21st step
    assert float(info["deaths"][-1]) == 0.0
    np.testing.assert_array_equal(np.asarray(s.alive)[:5], [False, True, True, True, True])
    assert not bool(s.present[0])


def test_obs_layout_fields():
    env = ENVS["haul"]
    p = env.make_params(planet="corrosion", n_robots=4, tempo=0.0)
    s, obs = env.reset(jax.random.PRNGKey(0), p)
    o = np.asarray(obs["robot"])
    np.testing.assert_allclose(o[:4, OBS["health"]], 1.0)
    np.testing.assert_allclose(o[:4, OBS["pos"]], np.asarray(s.pos)[:4] * 2 - 1, atol=1e-6)
    m = o[:4, OBS["mission"]]
    np.testing.assert_allclose(m[:, 0:2], np.asarray(s.anchors[0] - s.pos[:4]), atol=1e-6)
    np.testing.assert_allclose(m[:, 2:4], np.asarray(s.anchors[1] - s.pos[:4]), atol=1e-6)
    assert np.all(m[:, 6] == 1.0)                           # robots start inside the depot zone
    assert np.all(o[:4, OBS["mate1"]][:, 4] == 1.0)         # clustered start: teammates known
    assert np.all(o[4:] == 0.0) and not obs["robot_mask"][4:].any()
    assert int(obs["adv_mask"].sum()) == 3
