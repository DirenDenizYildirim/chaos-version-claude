"""Tests for hostile_planets.analysis on synthetic eval tables with planted structure."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hostile_planets import analysis as A  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SYN = _load_script("make_synthetic_eval")

# V_comm = mission offset - 0.2 * log S (no R_late term); other outcomes as in the default structure
PLANTED = dict(SYN.DEFAULT_STRUCTURE)
PLANTED["V_comm"] = ({"search": 0.10, "haul": -0.05, "escort": 0.20}, -0.20, 0.0, {})


@pytest.fixture(scope="module")
def exact_df():
    return SYN.make_synthetic(rng_seed=1, cond_noise=0, seed_noise=0, eval_noise=0, structure=PLANTED)


@pytest.fixture(scope="module")
def exact(exact_df):
    sc = A.scores(exact_df)
    out = A.outcomes(sc, "adv")
    pr = A.principles(sc, "adv")
    return sc, out, pr, A.transfer_table(out.table, pr)


# ------------------------------------------------------------------ formulas

def test_rel_formula():
    assert A.rel(0.5, 0.3) == pytest.approx(0.2 / (0.4 + 0.02))
    assert A.rel(0.3, 0.3) == 0
    assert A.rel(0.0, 0.0) == 0
    assert A.rel(0.3, 0.5) == pytest.approx(-A.rel(0.5, 0.3))
    # the generator's inverses are consistent with rel
    assert A.rel(SYN.solve_a(0.3, 0.25), 0.3) == pytest.approx(0.25)
    assert A.rel(0.3, SYN.solve_b(0.3, -0.4)) == pytest.approx(-0.4)


def _row(**kw):
    base = dict(planet="jam", mission="search", arm="default", seed=0, tempo=0.8, n=6, mode="adv", adv_name="",
                remove_t=-1, n_ep=10, J=0.5, J_sem=0.01, deaths_per_robot=0.0, coupled_frac=0.0, exp_first=0.1,
                exp_mid=0.1, exp_last=0.1, speed=0.02, health=1.0, tau_stale=10.0)
    base.update(kw)
    return base


def test_min_over_adversaries_and_carried_columns():
    df = pd.DataFrame([
        _row(mode="nature", J=0.70, deaths_per_robot=0.01),
        _row(adv_name="a", J=0.50, deaths_per_robot=0.10, tau_stale=11),
        _row(adv_name="b", J=0.40, deaths_per_robot=0.20, tau_stale=22),   # minimiser
        _row(adv_name="c", J=0.45, deaths_per_robot=0.30, tau_stale=33),
        _row(adv_name="b", J=np.nan),                                       # NaN ignored
        _row(adv_name="x", J=0.60, tempo=1.3),                             # other condition
        _row(mode="br", adv_name="br1", J=0.30),
    ])
    sc = A.scores(df)
    r = sc[(sc["tempo"] == 0.8)].iloc[0]
    assert r["J_nature"] == 0.70
    assert r["J_adv"] == 0.40
    assert r["adv_name_adv"] == "b"
    assert r["deaths_per_robot_adv"] == 0.20 and r["tau_stale_adv"] == 22
    assert r["deaths_per_robot_nature"] == 0.01
    assert r["pool_size_adv"] == 3
    assert r["J_br"] == 0.30
    r2 = sc[(sc["tempo"] == 1.3)].iloc[0]
    assert r2["J_adv"] == 0.60 and np.isnan(r2["J_nature"])


def test_seed_mean_before_rel_then_mean_over_n():
    rows = []
    # n=4: default seeds 0.2/0.4 (mean .3), solo seeds .1/.3 (mean .2); n=6: .5 vs .5; n=8 missing; n=2 ignored
    for s, jd, js in [(0, 0.2, 0.1), (1, 0.4, 0.3)]:
        rows += [_row(arm="default", seed=s, n=4, J=jd), _row(arm="solo", seed=s, n=4, J=js)]
    for s in [0, 1]:
        rows += [_row(arm="default", seed=s, n=6, J=0.5), _row(arm="solo", seed=s, n=6, J=0.5),
                 _row(arm="default", seed=s, n=2, J=0.9), _row(arm="solo", seed=s, n=2, J=0.1)]
    out = A.outcomes(A.scores(pd.DataFrame(rows)), "adv").table
    expect = np.mean([A.rel(0.3, 0.2), 0.0])
    assert out["V_couple"].iloc[0] == pytest.approx(expect)
    assert out["n_cov_V_couple"].iloc[0] == 2
    assert np.isnan(out["V_comm"].iloc[0])  # arms missing -> NaN


def test_nstar_and_beta_formula():
    n = np.array(A.NS_ALL, float)
    j = pd.Series(0.1 * n ** 0.5, index=A.NS_ALL)
    res = A._size_outcomes(j)
    assert res["beta"] == pytest.approx(0.5)
    for lam, col in zip(A.LAMBDAS, A.NSTAR_COLS):
        obj = j.values - lam * n * j.loc[2] / 2
        assert res[col] == n[np.argmax(obj)]
    assert np.isnan(A._size_outcomes(j.drop(2))["Nstar_0.5"])


# ------------------------------------------------------------------ planted structure

def test_outcomes_recover_planted_values(exact):
    sc, out, pr, tab = exact
    for r in tab.itertuples():
        for y in ["V_comm", "V_gossip", "V_couple", "V_sac", "beta"]:
            want = SYN.planted_outcome(y, r.planet, r.mission, r.tempo, "adv", PLANTED)
            assert getattr(r, y) == pytest.approx(want, abs=1e-9), (y, r.planet, r.mission, r.tempo)
        assert r.log_S == pytest.approx(SYN.log_s(r.planet, r.mission, r.tempo, "adv"))
        assert r.R_late == pytest.approx(SYN.r_late(r.planet, r.mission, r.tempo, "adv"))
        assert r.R_early == pytest.approx(r.R_late + 0.8)


def test_transfer_recovers_planted_slope_and_passes(exact):
    tt = A.transfer_test(exact[3])
    c = tt["coefs"]
    c = c[(c.outcome == "V_comm") & (c.split == "headline") & (c.model == "P")].set_index("param")["value"]
    assert c["log_S"] == pytest.approx(-0.2, abs=1e-8)
    assert c["R_late"] == pytest.approx(0.0, abs=1e-8)
    assert c["mission[search]"] == pytest.approx(0.10 + 0.2 * 1.5)
    m = tt["metrics"].set_index(["outcome", "split", "model"])
    assert m.loc[("V_comm", "lopo", "P"), "mae"] < 1e-8
    assert m.loc[("V_comm", "lopo", "P"), "n"] == 36
    assert m.loc[("V_comm", "headline", "P"), "n"] == 9
    v = tt["verdicts"].set_index("outcome")
    assert v.loc["V_comm", "supported"] is True or v.loc["V_comm", "supported"] == True  # noqa: E712
    assert set(tt["predictions"]["split"]) == {"headline", "lopo"}


def test_transfer_recovers_slope_with_noise():
    df = SYN.make_synthetic(rng_seed=3, cond_noise=0.02, seed_noise=0.01, eval_noise=0.005, structure=PLANTED)
    sc = A.scores(df)
    tab = A.transfer_table(A.outcomes(sc, "adv").table, A.principles(sc, "adv"))
    tt = A.transfer_test(tab, "V_comm")
    c = tt["coefs"]
    a = c[(c.split == "headline") & (c.model == "P") & (c.param == "log_S")]["value"].iloc[0]
    assert a == pytest.approx(-0.2, abs=0.05)
    assert tt["verdicts"]["supported"].iloc[0] is True


def test_transfer_fails_when_no_structure():
    rng = np.random.default_rng(0)
    rows = [dict(planet=p, mission=m, tempo=t, log_S=rng.normal(), R_late=rng.normal(), y=rng.normal())
            for p in A.PLANETS for m in A.MISSIONS for t in A.TEMPOS]
    tt = A.transfer_test(pd.DataFrame(rows), "y")
    assert tt["verdicts"]["supported"].iloc[0] is False
    assert tt["overall"]["label"] == "not supported"


# ------------------------------------------------------------------ verdict logic

def test_prereg_verdict_thresholds():
    assert A.prereg_verdict(0.090, 0.1, 0.1, 0.5) is True       # exactly 10% better, rho exactly 0.5
    assert A.prereg_verdict(0.0901, 0.1, 0.1, 0.9) is False     # < 10% better than B1
    assert A.prereg_verdict(0.05, 0.1, 0.0555, 0.9) is False    # < 10% better than B2
    assert A.prereg_verdict(0.05, 0.1, 0.1, 0.49) is False      # rho too low
    assert A.prereg_verdict(np.nan, 0.1, 0.1, 0.9) is None


def test_overall_verdict_counts():
    v = {"a": True, "b": True, "c": True, "d": False, "e": None, "f": False}
    o = A.overall_verdict(v)
    assert o["n_supported"] == 3 and not o["supported_overall"] and o["label"] == "partially supported"
    assert o["untestable_outcomes"] == ["e"]
    v["d"] = True
    assert A.overall_verdict(v)["supported_overall"] and A.overall_verdict(v)["label"] == "supported"
    assert A.overall_verdict({k: False for k in v})["label"] == "not supported"


def test_verdict_uses_lopo_split(exact):
    tt = A.transfer_test(exact[3])
    v = tt["verdicts"]
    for r in v.itertuples():
        assert r.supported == r.lopo_supported


# ------------------------------------------------------------------ other pieces

def test_seed_spread_and_unresolved():
    df = SYN.make_synthetic(rng_seed=2, cond_noise=0.0, seed_noise=0.2, eval_noise=0.0)
    out = A.outcomes(A.scores(df), "adv")
    sp = out.spread.set_index("outcome")
    assert sp.loc["V_comm", "n_seeds"] == 2
    assert sp.loc["V_comm", "unresolved"] is True  # huge seed noise swamps the planted structure
    assert len(out.per_seed) == 2 * 36
    df0 = SYN.make_synthetic(rng_seed=2, cond_noise=0.0, seed_noise=0.0, eval_noise=0.0)
    sp0 = A.outcomes(A.scores(df0), "adv").spread.set_index("outcome")
    assert sp0.loc["V_comm", "unresolved"] is False


def test_bootstrap_collapses_without_sem(exact_df, exact):
    df = exact_df.assign(J_sem=0.0)
    b = A.bootstrap_outcomes(df, "adv", n_boot=20).merge(exact[3][A.COND + ["V_comm"]], on=A.COND)
    assert np.allclose(b["V_comm_lo"], b["V_comm"]) and np.allclose(b["V_comm_hi"], b["V_comm"])
    b2 = A.bootstrap_outcomes(exact_df.assign(J_sem=0.02), "adv", n_boot=50)
    assert (b2["V_comm_hi"] - b2["V_comm_lo"] > 0).all()


def test_secondary_and_validity_shapes(exact):
    sc, out, pr, _ = exact
    sec = A.secondary_checks(sc, "adv", out, pr)
    assert {"1a", "1b", "1c", "1d", "2a", "2b", "2c", "3a", "3b", "4a", "4b", "4c", "4d"} <= set(sec["id"])
    assert set(sec["verdict"]) <= {"pass", "fail", "mixed", "na"}
    assert sec.set_index("id").loc["2a", "verdict"] == "pass"   # planted escort > haul > search
    assert sec.set_index("id").loc["3b", "verdict"] == "pass"   # planted escort beta smallest
    val = A.validity_checks(sc)
    assert len(val["heuristic"]) == 12 and len(val["adv_drop"]) == 12
    assert val["summary"]["heuristic_failing_cells"] == ["jam/escort", "quake/haul"]
    assert val["summary"]["adv_gate_pass"]
    assert val["summary"]["br_cells_covered"] == 12


# ------------------------------------------------------------------ partial data

def _run_everything(df, tmp_path=None):
    sc = A.scores(df)
    for which in ["adv", "nature"]:
        out = A.outcomes(sc, which)
        pr = A.principles(sc, which)
        tab = A.transfer_table(out.table, pr)
        A.transfer_test(tab)
        A.secondary_checks(sc, which, out, pr)
        A.answers(sc, out.table, which)
        A.bootstrap_outcomes(df, which, n_boot=10)
    A.validity_checks(sc)
    return sc


@pytest.mark.parametrize("variant", ["drop_arm", "one_seed", "random_rows", "no_combined", "nature_only",
                                     "single_cell", "no_remove_rows"])
def test_partial_tables_do_not_crash(exact_df, variant):
    df = exact_df
    rng = np.random.default_rng(0)
    if variant == "drop_arm":
        df = df[df["arm"] != "blackboard"]
    elif variant == "one_seed":
        df = df[df["seed"] <= 0]
    elif variant == "random_rows":
        df = df[rng.random(len(df)) > 0.6]
    elif variant == "no_combined":
        df = df[df["planet"] != "combined"]
    elif variant == "nature_only":
        df = df[df["mode"] == "nature"]
    elif variant == "single_cell":
        df = df[(df["planet"] == "jam") & (df["mission"] == "haul") & (df["arm"].isin(["default", "solo"]))]
    elif variant == "no_remove_rows":
        df = df[df["remove_t"] == -1]
    _run_everything(df.reset_index(drop=True))
    if variant == "drop_arm":
        out = A.outcomes(A.scores(df), "adv").table
        assert out["V_comm"].isna().all() and out["V_couple"].notna().all()


def test_report_end_to_end_on_partial_data(exact_df, tmp_path):
    rep = _load_script("make_report")
    df = exact_df[(exact_df["arm"] != "protected") & (exact_df["seed"] <= 0)]
    df = df[np.random.default_rng(1).random(len(df)) > 0.3]
    ev = tmp_path / "eval.csv"
    df.to_csv(ev, index=False)
    runs = tmp_path / "runs"
    SYN.write_runs(str(runs), seeds=(0,), iters=5)
    (runs / "broken.json").write_text("{not json")
    (runs / "weird__name.json").write_text('{"log": {"step": [0, 1], "eval_J_nature": [0.1, 0.2]}}')
    s = rep.build(str(ev), str(runs), str(tmp_path / "out"), n_boot=10, figures=True, log=lambda *a: None)
    assert s["figure_errors"] == {}
    assert len(s["figures"]) == 7
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "tables" / "answers_adv.csv").exists()
    assert any("broken.json" in p for p in s["coverage"]["training_run_problems"])
