import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hostile_planets import ppo  # noqa: E402

CFG = ppo.PPOConfig()
T, B, N, D, A = 7, 3, 4, 5, 3


def _traj(key, n=N):
    ks = jax.random.split(key, 6)
    mask = jnp.ones((T, B, n), bool)
    mask = mask.at[4:, 0, 1].set(False)        # agent dies at t=4
    return {"obs": jax.random.normal(ks[0], (T, B, n, D)),
            "act": jax.random.normal(ks[1], (T, B, n, A)),
            "logp": jax.random.normal(ks[2], (T, B, n)) - 3,
            "val": jax.random.normal(ks[3], (T, B, n)),
            "rew": jax.random.normal(ks[4], (T, B, n)),
            "mask": mask}


def _loss_and_grad(params, traj):
    batch = ppo.prepare_batch(traj, CFG)
    return jax.value_and_grad(ppo.ppo_loss, has_aux=True)(params, batch, CFG)


def _with_garbage_agent(traj, key):
    """Append an absent agent (mask 0 everywhere) whose data is huge / NaN."""
    g = {k: jnp.concatenate([v, jnp.full(v[:, :, :1].shape, 1e6 if k != "rew" else jnp.nan, v.dtype)], 2)
         for k, v in traj.items() if k != "mask"}
    g["act"] = g["act"].at[:, :, -1].set(jax.random.normal(key, (T, B, A)) * 1e4)
    g["mask"] = jnp.concatenate([traj["mask"], jnp.zeros((T, B, 1), bool)], 2)
    # also corrupt the dead agent's post-death samples
    for k in ("obs", "act"):
        g[k] = g[k].at[4:, 0, 1].set(1e6)
    g["val"] = g["val"].at[4:, 0, 1].set(jnp.nan)
    return g


def test_masked_agents_contribute_nothing():
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    params = ppo.init_actor_critic(k1, D, A, 16)
    traj = _traj(k2)
    (l0, aux0), g0 = _loss_and_grad(params, traj)
    clean = dict(traj)
    for k in ("obs", "act"):
        clean[k] = clean[k].at[4:, 0, 1].set(0.)
    (l_clean, _), _ = _loss_and_grad(params, clean)
    (l1, aux1), g1 = _loss_and_grad(params, _with_garbage_agent(traj, k3))
    assert np.isfinite(l1)
    np.testing.assert_allclose(l0, l_clean, rtol=1e-6)
    np.testing.assert_allclose(l0, l1, rtol=1e-6)
    for a, b in zip(jax.tree.leaves(g0), jax.tree.leaves(g1)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


def test_no_valid_samples_gives_zero_grad():
    params = ppo.init_actor_critic(jax.random.PRNGKey(1), D, A, 16)
    traj = _traj(jax.random.PRNGKey(2))
    traj["mask"] = jnp.zeros_like(traj["mask"])
    (loss, _), g = _loss_and_grad(params, traj)
    assert loss == 0
    assert all(float(jnp.abs(x).max()) == 0 for x in jax.tree.leaves(g))


def test_gae_matches_reference_with_death():
    rng = np.random.default_rng(0)
    rew, val = rng.normal(size=(T,)), rng.normal(size=(T,))
    mask = np.array([1, 1, 1, 1, 0, 0, 0], np.float32)
    adv, ret = ppo.gae(jnp.array(rew), jnp.array(val), jnp.array(mask), 0.99, 0.95)
    ref, last = np.zeros(T), 0.0
    for t in reversed(range(4)):            # last valid step t=3 is terminal
        nv = val[t + 1] if t < 3 else 0.0
        delta = rew[t] + 0.99 * nv - val[t]
        last = delta + 0.99 * 0.95 * (last if t < 3 else 0.0)
        ref[t] = last
    np.testing.assert_allclose(np.asarray(adv)[:4], ref[:4], rtol=1e-5)
    np.testing.assert_allclose(np.asarray(ret)[:4], ref[:4] + val[:4], rtol=1e-5)


def test_save_load_roundtrip(tmp_path):
    r = ppo.init_actor_critic(jax.random.PRNGKey(0), 70, 3, 64)
    a = ppo.init_actor_critic(jax.random.PRNGKey(1), 14, 2, 32)
    ppo.save_params(tmp_path / "p.npz", r, a)
    r2, a2 = ppo.load_params(tmp_path / "p.npz")
    assert jax.tree.structure(r) == jax.tree.structure(r2)
    for x, y in zip(jax.tree.leaves((r, a)), jax.tree.leaves((r2, a2))):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("det", [True, False])
def test_policy_fns(det):
    r = ppo.init_actor_critic(jax.random.PRNGKey(0), 70, 3, 64)
    a = ppo.init_actor_critic(jax.random.PRNGKey(1), 14, 2, 32)
    pf, af = ppo.make_policy_fns(r, a, deterministic=det)
    k = jax.random.PRNGKey(3)
    assert pf(jnp.zeros((12, 70)), k).shape == (12, 3)
    assert af(jnp.zeros((9, 14)), k).shape == (9, 2)
    same = np.allclose(pf(jnp.ones((12, 70)), k), pf(jnp.ones((12, 70)), jax.random.PRNGKey(4)))
    assert same == det


def test_iteration_runs_on_stub():
    from tests.stub_env import HostileEnv
    env = HostileEnv(obs_dim=32)
    cfg = ppo.PPOConfig(n_envs=4, n_minibatches=2, n_epochs=1, total_iters=2)
    sp = lambda k: env.make_params(n_robots=jax.random.choice(k, jnp.array([2, 3])),
                                   adversarial=jax.random.bernoulli(k, 0.5))
    ts = ppo.init_train_state(jax.random.PRNGKey(0), env, cfg)
    it = ppo.make_iteration(env, sp, cfg)
    ts, m = it(ts, jax.random.PRNGKey(1))
    assert np.isfinite(float(m["robot_return"])) and np.isfinite(float(m["robot_pg_loss"]))
