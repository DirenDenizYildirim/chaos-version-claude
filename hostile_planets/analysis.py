"""Pre-registered analysis of the Hostile Planets grid (docs/PREREGISTRATION.md).

Pure functions on a pandas DataFrame in the eval schema (docs/EVAL_SCHEMA.md).
No simulator / JAX dependency: this module only needs numpy, pandas and scipy.

Pipeline::

    sc   = scores(df)                         # one row per policy x (tempo, n, remove_t)
    out  = outcomes(sc, "adv")                # OutcomeResult(table, per_seed, spread)
    pr   = principles(sc, "adv")              # S, tau_stale, tau_act, W(0/50/100)
    tab  = transfer_table(out.table, pr)      # merge on (planet, mission, tempo)
    tt   = transfer_test(tab)                 # P vs B0/B1/B2, headline + LOPO, verdicts
    sec  = secondary_checks(sc, "adv")
    val  = validity_checks(sc)
    ci   = bootstrap_outcomes(df, "adv")      # 90% intervals of the V outcomes

Conventions that resolve ambiguities in the prereg text are documented inline
and collected in ``AMBIGUITIES`` (also written to the report's summary.json).
Missing arms / seeds / cells never raise: they propagate as NaN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- constants

PLANETS = ["corrosion", "jam", "quake", "combined"]
SINGLE_PLANETS = ["corrosion", "jam", "quake"]
MISSIONS = ["search", "haul", "escort"]
ARMS = ["default", "silent", "blackboard", "solo", "protected"]
TEMPOS = [0.3, 0.8, 1.3]
NS_ALL = [2, 3, 4, 6, 8, 12]
NS_AVG = [4, 6, 8]  # outcomes are averaged over these swarm sizes
N_W = 6  # worth trajectory measured at N = 6
REMOVE_TS = [0, 50, 100]
LAMBDAS = [0.25, 0.5, 0.75]
T_HORIZON = 150

KEY = ["planet", "mission", "arm", "seed", "tempo", "n", "remove_t"]
COND = ["planet", "mission", "tempo"]
CELL = ["planet", "mission"]

V_PAIRS = {  # outcome -> (arm a, arm b) for rel(J[a], J[b])
    "V_comm": ("blackboard", "silent"),
    "V_gossip": ("default", "silent"),
    "V_couple": ("default", "solo"),
    "V_sac": ("default", "protected"),
}
V_OUTCOMES = list(V_PAIRS)
PRIMARY_OUTCOMES = ["V_comm", "V_gossip", "V_couple", "V_sac", "beta", "loss_default"]
NSTAR_COLS = [f"Nstar_{lam:g}" for lam in LAMBDAS]
OUTCOME_COLS = V_OUTCOMES + ["beta"] + NSTAR_COLS + ["loss_default", "loss_protected", "coupled_frac_default"]
MODELS = ["P", "B0", "B1", "B2"]

AMBIGUITIES = {
    "n_average": "Outcomes average over n in {4,6,8} with nanmean: if some n are missing the mean is over the n present "
                 "(column n_cov_<outcome> counts them). Seed-mean J first, then rel(), then mean over n.",
    "beta": "beta = OLS slope of log(seed-mean J of default) on log N over all evaluated N in {2,3,4,6,8,12} with J>0; "
            "needs >=3 points, else NaN.",
    "nstar": "N*(lambda) = argmax over evaluated N in {2,...,12} of Jbar(N) - lambda*N*Jbar(2)/2 (default arm, seed-mean J); "
             "NaN if N=2 is missing. Ties -> smallest N.",
    "loss_rate": "loss rate = seed-mean deaths_per_robot, mean over n in {4,6,8}, of the arm's rows in the chosen score "
                 "(for 'adv' the row of the minimising adversary).",
    "adv_row_columns": "For the adversarial score every non-J column (deaths, exposure, speed, tau_stale, ...) comes from the "
                       "row of the adversary that minimises J for that policy/tempo/n/remove_t.",
    "principles": "S = tau_act / tau_stale with tau_act = 0.5 / mean(speed) and tau_stale = mean(tau_stale), both means over "
                  "default-arm rows (seeds x n in {4,6,8}, remove_t=-1) of the chosen score. Means are taken before the ratio.",
    "worth": "W(t) = (Jfull - Jremoved) * 6 / Jfull with Jfull, Jremoved the seed-mean J of default at n=6 (remove_t=-1 / t). "
             "Under the adversarial score both are pool-minimum scores. NaN if Jfull <= 1e-9.",
    "per_seed": "Per-seed outcomes pair training seed s of every arm with seed s of the other arms; seed spread = std (ddof=1) "
                "over seeds per condition. An outcome is 'unresolved' when the pooled seed spread sqrt(mean_c sd_c^2) exceeds "
                "the between-condition spread (std over conditions of the seed-mean outcome).",
    "transfer_rows": "All four models are fitted and scored on the same rows: conditions with finite y, log S, R_late and tempo.",
    "transfer_verdict": "'P beats B1 and B2 in held-out MAE by >=10%' means MAE_P <= 0.9*MAE_B1 and MAE_P <= 0.9*MAE_B2. "
                        "The prereg sentence ends 'on the leave-one-planet-out predictions'; we read that as qualifying both the "
                        "MAE and the Spearman criterion, so the verdict uses the 36 LOPO predictions. The same criterion on the "
                        "headline (train single-hazard -> predict combined) split is reported alongside but does not gate.",
    "transfer_overall": ">=4 of 6 outcomes supported -> 'supported'; 1-3 -> 'partially supported'; 0 -> 'not supported'. "
                        "Untestable outcomes (NaN metrics) count as not supported and are listed.",
    "sign_agreement": "Fraction of held-out conditions with sign(pred) == sign(obs) (beta and loss rates are positive, so this "
                      "metric is trivially ~1 for them).",
    "bootstrap": "Only per-row mean J and J_sem are available, so uncertainty is propagated parametrically: every eval row's J is "
                 "redrawn as Normal(J, J_sem) (J_sem NaN -> 0), the pool minimum is re-taken per draw, and the V outcomes are "
                 "recomputed. This covers evaluation-episode noise only, not seed-to-seed training variation (see seed spread).",
    "heuristic_match": "'match or beat the heuristic' = seed-mean J_default >= J_heuristic - 1.645*SE, SE combining the J_sem of "
                       "both (one-sided 5% non-inferiority). Pass overall if >= 9 of 12 cells pass (cells without data count as fail).",
    "adv_drop": "Per cell: mean over default rows (all seeds, tempi, n; remove_t=-1) of J_nature - J_adv. 'Most cells' = more "
                "than half of the 12 cells show a positive drop.",
    "secondary_score": "Secondary expectations are evaluated under the same score as the primary test (adversarial) by default; "
                       "the report also writes the nature-score version as a sensitivity table.",
}


# --------------------------------------------------------------------------- basics

def rel(a, b):
    """rel(a, b) = (a - b) / (0.5*(a + b) + 0.02). Works on scalars, arrays and aligned pandas objects."""
    return (a - b) / (0.5 * (a + b) + 0.02)


def normalize_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an eval table to canonical dtypes (missing optional columns are added as NaN)."""
    d = df.copy()
    for c in ["planet", "mission", "arm", "mode"]:
        d[c] = d[c].astype(str).str.strip().str.lower()
    d["mode"] = d["mode"].replace({"adversarial": "adv", "best_response": "br", "nat": "nature"})
    d["seed"] = pd.to_numeric(d.get("seed", -1), errors="coerce").fillna(-1).astype(int)
    d["n"] = pd.to_numeric(d["n"], errors="coerce").astype("Int64")
    d = d[d["n"].notna()].copy()
    d["n"] = d["n"].astype(int)
    d["tempo"] = pd.to_numeric(d["tempo"], errors="coerce").round(3)
    rt = d["remove_t"] if "remove_t" in d.columns else pd.Series(-1, index=d.index)
    d["remove_t"] = pd.to_numeric(rt, errors="coerce").fillna(-1).astype(int)
    if "adv_name" not in d.columns:
        d["adv_name"] = ""
    d["adv_name"] = d["adv_name"].fillna("").astype(str)
    for c in ["J", "J_sem", "deaths_per_robot", "coupled_frac", "exp_first", "exp_mid", "exp_last",
              "speed", "health", "tau_stale", "n_ep"]:
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    return d


def _pick_min(rows: pd.DataFrame) -> pd.DataFrame:
    """For each KEY group keep the row with the minimum J (rows with NaN J dropped); add pool size."""
    rows = rows[rows["J"].notna()]
    if rows.empty:
        return rows.assign(pool_size=pd.Series(dtype=float))
    idx = rows.groupby(KEY, sort=False)["J"].idxmin()
    out = rows.loc[idx.values].copy()
    size = rows.groupby(KEY, sort=False)["J"].size().rename("pool_size")
    return out.merge(size.reset_index(), on=KEY, how="left")


def scores(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (planet, mission, arm, seed, tempo, n, remove_t).

    Columns: ``J_nature`` (nature row), ``J_adv`` (minimum J over the ``adv`` rows: the pool-minimum
    adversarial score), ``J_br`` (minimum over ``br`` rows, if any) and, for each score ``m``, every other
    eval column ``<col>_<m>`` taken from the same row (for adv/br: the minimising adversary's row,
    so ``adv_name_adv`` names it), plus ``pool_size_<m>``.
    """
    d = normalize_eval(df)
    carry = [c for c in d.columns if c not in KEY + ["mode"]]
    out = None
    for mode in ["nature", "adv", "br"]:
        rows = d[d["mode"] == mode]
        if mode == "nature":
            rows = rows[rows["J"].notna()].drop_duplicates(KEY, keep="last").assign(pool_size=1)
        else:
            rows = _pick_min(rows)
        cols = carry + ["pool_size"]
        rows = rows[KEY + cols].rename(columns={c: f"{c}_{mode}" for c in cols})
        out = rows if out is None else out.merge(rows, on=KEY, how="outer")
    for mode in ["nature", "adv", "br"]:
        if f"J_{mode}" not in out.columns:
            out[f"J_{mode}"] = np.nan
    return out.sort_values(KEY).reset_index(drop=True)


def score_view(sc: pd.DataFrame, which: str = "adv") -> pd.DataFrame:
    """Rows of ``scores`` under one score ('adv', 'nature' or 'br') with the suffix stripped (J, deaths_per_robot, ...)."""
    if which not in ("adv", "nature", "br"):
        raise ValueError(f"which must be adv/nature/br, got {which!r}")
    suf = f"_{which}"
    cols = [c for c in sc.columns if c.endswith(suf)]
    v = sc[KEY + cols].rename(columns={c: c[: -len(suf)] for c in cols})
    return v[v["J"].notna()].reset_index(drop=True)


def _conditions(sc: pd.DataFrame) -> pd.MultiIndex:
    """All (planet, mission, tempo) conditions present anywhere in the table, in canonical order."""
    c = sc[COND].drop_duplicates()
    c = c[c["tempo"].notna()]
    c["_p"] = c["planet"].map({p: i for i, p in enumerate(PLANETS)}).fillna(99)
    c["_m"] = c["mission"].map({m: i for i, m in enumerate(MISSIONS)}).fillna(99)
    c = c.sort_values(["_p", "planet", "_m", "mission", "tempo"])
    return pd.MultiIndex.from_frame(c[COND])


# --------------------------------------------------------------------------- outcomes

def _v_from_jbar(jbar: pd.DataFrame, cond_index: pd.MultiIndex) -> dict[str, pd.DataFrame]:
    """V outcomes from seed-mean J.

    ``jbar`` is indexed by (planet, mission, tempo, arm, n) with any number of columns (1 for the point
    estimate, B for bootstrap draws). Returns {V name: frame indexed by condition, same columns}.
    """
    arms = set(jbar.index.get_level_values("arm")) if len(jbar) else set()
    out = {}
    empty = pd.DataFrame(np.nan, index=cond_index, columns=jbar.columns)
    for name, (a, b) in V_PAIRS.items():
        if a not in arms or b not in arms:
            out[name] = empty.copy()
            continue
        A = jbar.xs(a, level="arm")
        B = jbar.xs(b, level="arm")
        r = rel(A, B)  # aligned on (planet, mission, tempo, n); unmatched -> NaN
        r = r[r.index.get_level_values("n").isin(NS_AVG)]
        out[name] = r.groupby(level=COND).mean().reindex(cond_index)
    return out


def _n_mean(series: pd.Series, cond_index) -> pd.Series:
    s = series[series.index.get_level_values("n").isin(NS_AVG)]
    return s.groupby(level=COND).mean().reindex(cond_index)


def _size_outcomes(jdef: pd.Series) -> dict:
    """beta and N*(lambda) from a seed-mean J(N) curve of the default arm (index n)."""
    res = {"beta": np.nan, **{c: np.nan for c in NSTAR_COLS}}
    j = jdef.dropna()
    j = j[j.index.isin(NS_ALL)]
    pos = j[j > 0]
    if len(pos) >= 3:
        res["beta"] = float(np.polyfit(np.log(pos.index.values.astype(float)), np.log(pos.values), 1)[0])
    if 2 in j.index and len(j) >= 2:
        j2 = j.loc[2]
        ns = j.index.values.astype(float)
        for lam, col in zip(LAMBDAS, NSTAR_COLS):
            obj = j.values - lam * ns * j2 / 2.0
            res[col] = float(ns[int(np.argmax(obj))])
    return res


def _outcomes_core(view: pd.DataFrame, cond_index: pd.MultiIndex) -> pd.DataFrame:
    """Outcome table from rows already restricted to one seed set (seed-mean taken inside)."""
    full = view[(view["remove_t"] == -1) & view["arm"].isin(ARMS)]
    g = full.groupby(COND + ["arm", "n"])
    jbar = g[["J"]].mean()
    tab = pd.DataFrame(index=cond_index)
    for name, fr in _v_from_jbar(jbar, cond_index).items():
        tab[name] = fr["J"]
    # coverage: number of n in {4,6,8} that contributed to each V
    for name, (a, b) in V_PAIRS.items():
        if len(jbar):
            ja = jbar["J"].xs(a, level="arm") if a in jbar.index.get_level_values("arm") else pd.Series(dtype=float)
            jb = jbar["J"].xs(b, level="arm") if b in jbar.index.get_level_values("arm") else pd.Series(dtype=float)
            both = rel(ja, jb).dropna()
            both = both[both.index.get_level_values("n").isin(NS_AVG)] if len(both) else both
            cov = both.groupby(level=COND).size().reindex(cond_index) if len(both) else pd.Series(0, index=cond_index)
            tab[f"n_cov_{name}"] = cov.fillna(0).astype(int).values
        else:
            tab[f"n_cov_{name}"] = 0
    # size outcomes from the default arm
    size_rows = []
    jd = jbar["J"].xs("default", level="arm") if len(jbar) and "default" in jbar.index.get_level_values("arm") else None
    for cond in cond_index:
        if jd is not None:
            try:
                curve = jd.loc[cond]
            except KeyError:
                curve = pd.Series(dtype=float)
        else:
            curve = pd.Series(dtype=float)
        size_rows.append(_size_outcomes(curve))
    size = pd.DataFrame(size_rows, index=cond_index)
    for c in size.columns:
        tab[c] = size[c]
    # loss rates and coupling usage
    for arm, col, val in [("default", "loss_default", "deaths_per_robot"),
                          ("protected", "loss_protected", "deaths_per_robot"),
                          ("default", "coupled_frac_default", "coupled_frac")]:
        rows = full[full["arm"] == arm]
        if rows.empty:
            tab[col] = np.nan
        else:
            m = rows.groupby(COND + ["n"])[val].mean()
            tab[col] = _n_mean(m, cond_index)
    return tab


@dataclass
class OutcomeResult:
    table: pd.DataFrame      # per condition: outcomes, sd_<y>, unresolved_<y>, n_seeds
    per_seed: pd.DataFrame   # per condition x seed: outcomes with seed s paired across arms
    spread: pd.DataFrame     # per outcome: seed_spread, between_spread, unresolved


def outcomes(sc: pd.DataFrame, which: str = "adv") -> OutcomeResult:
    """Pre-registered outcomes per condition (planet, mission, tempo) under the given score."""
    view = score_view(sc, which)
    cond_index = _conditions(sc)
    tab = _outcomes_core(view, cond_index)

    seeds = sorted(s for s in view.loc[view["arm"].isin(ARMS), "seed"].unique() if s >= 0)
    per = []
    for s in seeds:
        t = _outcomes_core(view[view["seed"] == s], cond_index)
        t = t[[c for c in OUTCOME_COLS]].copy()
        t["seed"] = s
        per.append(t.reset_index())
    per_seed = (pd.concat(per, ignore_index=True) if per
                else pd.DataFrame(columns=COND + OUTCOME_COLS + ["seed"]))

    spread_rows = []
    for y in OUTCOME_COLS:
        if per_seed.empty:
            sd = pd.Series(np.nan, index=cond_index)
        else:
            sd = per_seed.groupby(COND)[y].std(ddof=1).reindex(cond_index)
        tab[f"sd_{y}"] = sd.values
        seed_spread = float(np.sqrt(np.nanmean(sd.values ** 2))) if np.isfinite(sd.values).any() else np.nan
        vals = tab[y].values.astype(float)
        between = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() >= 2 else np.nan
        unres = (bool(seed_spread > between) if np.isfinite(seed_spread) and np.isfinite(between) else None)
        spread_rows.append({"outcome": y, "seed_spread": seed_spread, "between_spread": between,
                            "unresolved": unres, "n_conditions": int(np.isfinite(vals).sum()),
                            "n_seeds": len(seeds)})
        tab[f"unresolved_{y}"] = unres
    spread = pd.DataFrame(spread_rows)
    tab["n_seeds"] = len(seeds)
    return OutcomeResult(tab.reset_index(), per_seed, spread)


# --------------------------------------------------------------------------- principles

def principles(sc: pd.DataFrame, which: str = "adv") -> pd.DataFrame:
    """Principle variables per condition from default-arm rows: S, tau_stale, tau_act, R_early/mid/late (= W(0/50/100))."""
    view = score_view(sc, which)
    cond_index = _conditions(sc)
    d = view[view["arm"] == "default"]
    full = d[(d["remove_t"] == -1) & d["n"].isin(NS_AVG)]
    tab = pd.DataFrame(index=cond_index)
    if full.empty:
        tab["speed"] = np.nan
        tab["tau_stale"] = np.nan
    else:
        g = full.groupby(COND)
        tab["speed"] = g["speed"].mean().reindex(cond_index)
        tab["tau_stale"] = g["tau_stale"].mean().reindex(cond_index)
    with np.errstate(divide="ignore", invalid="ignore"):
        tab["tau_act"] = np.where(tab["speed"] > 0, 0.5 / tab["speed"], np.nan)
        tab["S"] = np.where(tab["tau_stale"] > 0, tab["tau_act"] / tab["tau_stale"], np.nan)
        tab["log_S"] = np.where(tab["S"] > 0, np.log(tab["S"]), np.nan)
    w6 = d[d["n"] == N_W]
    jm = w6.groupby(COND + ["remove_t"])["J"].mean() if not w6.empty else pd.Series(dtype=float)

    def _get(t):
        if jm.empty:
            return pd.Series(np.nan, index=cond_index)
        s = jm[jm.index.get_level_values("remove_t") == t]
        return s.droplevel("remove_t").reindex(cond_index) if len(s) else pd.Series(np.nan, index=cond_index)

    jfull = _get(-1)
    tab["J_full_n6"] = jfull.values
    for t, name in zip(REMOVE_TS, ["R_early", "R_mid", "R_late"]):
        jr = _get(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(jfull.values > 1e-9, (jfull.values - jr.values) * N_W / jfull.values, np.nan)
        tab[name] = w
    return tab.reset_index()


def exposure(sc: pd.DataFrame, which: str = "adv", arms: Sequence[str] = ("default", "protected")) -> pd.DataFrame:
    """Hazard exposure of live robots per condition x arm, by mission third.

    ``exp_first/mid/last``: seed-mean, mean over n in {4,6,8}; ``exp6_*``: seed-mean at n=6 (the size where W(t) is measured).
    """
    view = score_view(sc, which)
    cond_index = _conditions(sc)
    rows = []
    for arm in arms:
        d = view[(view["arm"] == arm) & (view["remove_t"] == -1)]
        t = pd.DataFrame(index=cond_index)
        for c in ["exp_first", "exp_mid", "exp_last"]:
            if d.empty:
                t[c] = np.nan
                t[c.replace("exp_", "exp6_")] = np.nan
                continue
            m = d.groupby(COND + ["n"])[c].mean()
            t[c] = _n_mean(m, cond_index)
            m6 = d[d["n"] == N_W].groupby(COND)[c].mean()
            t[c.replace("exp_", "exp6_")] = m6.reindex(cond_index)
        t["arm"] = arm
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


def transfer_table(out_table: pd.DataFrame, prin: pd.DataFrame) -> pd.DataFrame:
    """Merge outcomes and principle variables on (planet, mission, tempo)."""
    keep = [c for c in prin.columns if c not in out_table.columns or c in COND]
    return out_table.merge(prin[keep], on=COND, how="outer")


# --------------------------------------------------------------------------- transfer test

def _design(d: pd.DataFrame, model: str, missions: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    n = len(d)
    if model == "B0":
        return np.ones((n, 1)), ["intercept"]
    cols = [(d["mission"].values == m).astype(float) for m in missions]
    names = [f"mission[{m}]" for m in missions]
    if model == "B2":
        cols.append(d["tempo"].values.astype(float))
        names.append("tempo")
    elif model == "P":
        cols += [d["log_S"].values.astype(float), d["R_late"].values.astype(float)]
        names += ["log_S", "R_late"]
    return np.column_stack(cols) if cols else np.zeros((n, 0)), names


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, y: str, model: str):
    """OLS fit on train, predict test. Returns (pred array, coef dict). NaN if not identifiable."""
    missions = [m for m in MISSIONS if m in set(train["mission"])] + \
               sorted(set(train["mission"]) - set(MISSIONS))
    Xtr, names = _design(train, model, missions)
    Xte, _ = _design(test, model, missions)
    nan_pred = np.full(len(test), np.nan)
    if len(train) < Xtr.shape[1] or Xtr.shape[1] == 0 or np.linalg.matrix_rank(Xtr) < Xtr.shape[1]:
        return nan_pred, {k: np.nan for k in names}
    coef, *_ = np.linalg.lstsq(Xtr, train[y].values.astype(float), rcond=None)
    pred = Xte @ coef
    if model != "B0":  # missions unseen in training have no intercept
        pred = np.where(test["mission"].isin(missions).values, pred, np.nan)
    return pred, dict(zip(names, coef))


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def _metrics(obs, pred) -> dict:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    if ok.sum() == 0:
        return {"mae": np.nan, "spearman": np.nan, "sign_agree": np.nan, "n": 0}
    return {"mae": float(np.mean(np.abs(obs[ok] - pred[ok]))),
            "spearman": _spearman(obs[ok], pred[ok]),
            "sign_agree": float(np.mean(np.sign(obs[ok]) == np.sign(pred[ok]))),
            "n": int(ok.sum())}


def prereg_verdict(mae_P, mae_B1, mae_B2, rho_P) -> bool | None:
    """The per-outcome pre-registered criterion. None if any input is NaN (untestable)."""
    vals = [mae_P, mae_B1, mae_B2, rho_P]
    if any(v is None or not np.isfinite(v) for v in vals):
        return None
    return bool(mae_P <= 0.9 * mae_B1 and mae_P <= 0.9 * mae_B2 and rho_P >= 0.5)


def overall_verdict(per_outcome: dict) -> dict:
    """Aggregate per-outcome verdicts (True/False/None) into the prereg's overall statement."""
    supported = [k for k, v in per_outcome.items() if v is True]
    untestable = [k for k, v in per_outcome.items() if v is None]
    k = len(supported)
    label = "supported" if k >= 4 else ("partially supported" if k >= 1 else "not supported")
    return {"n_supported": k, "n_outcomes": len(per_outcome), "supported_outcomes": supported,
            "untestable_outcomes": untestable, "supported_overall": k >= 4, "label": label}


def transfer_test(table: pd.DataFrame, outcome: str | Sequence[str] = tuple(PRIMARY_OUTCOMES),
                  train_planets: Sequence[str] = tuple(SINGLE_PLANETS), test_planet: str = "combined") -> dict:
    """Cross-world transfer test (model P vs baselines B0/B1/B2).

    ``table`` has one row per condition with columns planet, mission, tempo, log_S (or S), R_late and the
    outcome column(s). Returns a dict with DataFrames ``metrics`` (outcome, split, model, mae, spearman,
    sign_agree, n), ``coefs`` (outcome, split, fold, model, param, value), ``predictions`` (one row per
    outcome x split x held-out condition with obs and pred_<model>), ``verdicts`` (per outcome: the
    prereg booleans and the numbers behind them) and ``overall`` (dict).
    """
    outs = [outcome] if isinstance(outcome, str) else list(outcome)
    t = table.copy()
    if "log_S" not in t.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            t["log_S"] = np.log(t["S"].astype(float))
    met, coefs, preds, verd = [], [], [], []
    for y in outs:
        if y not in t.columns:
            t[y] = np.nan
        ok = np.isfinite(t[[y, "log_S", "R_late", "tempo"]].astype(float)).all(axis=1)
        d = t[ok].reset_index(drop=True)
        splits = {"headline": [(f"train={'+'.join(train_planets)}", d[d["planet"].isin(train_planets)],
                                d[d["planet"] == test_planet])],
                  "lopo": [(f"out={p}", d[d["planet"] != p], d[d["planet"] == p]) for p in PLANETS]}
        for split, folds in splits.items():
            pieces = []
            for fold, tr, te in folds:
                p = te[COND].copy()
                p["obs"] = te[y].values
                for m in MODELS:
                    pr, cf = _fit_predict(tr, te, y, m)
                    p[f"pred_{m}"] = pr
                    for k, v in cf.items():
                        coefs.append({"outcome": y, "split": split, "fold": fold, "model": m, "param": k,
                                      "value": v, "n_train": len(tr)})
                p["fold"] = fold
                pieces.append(p)
            P = pd.concat(pieces, ignore_index=True)
            P.insert(0, "split", split)
            P.insert(0, "outcome", y)
            preds.append(P)
            for m in MODELS:
                met.append({"outcome": y, "split": split, "model": m, **_metrics(P["obs"], P[f"pred_{m}"])})
        M = pd.DataFrame([r for r in met if r["outcome"] == y]).set_index(["split", "model"])
        row = {"outcome": y}
        for split in ["lopo", "headline"]:
            g = lambda m, k: M.loc[(split, m), k] if (split, m) in M.index else np.nan  # noqa: E731
            v = prereg_verdict(g("P", "mae"), g("B1", "mae"), g("B2", "mae"), g("P", "spearman"))
            row.update({f"{split}_mae_P": g("P", "mae"), f"{split}_mae_B1": g("B1", "mae"),
                        f"{split}_mae_B2": g("B2", "mae"), f"{split}_rho_P": g("P", "spearman"),
                        f"{split}_supported": v})
        row["supported"] = row["lopo_supported"]  # the gating verdict
        verd.append(row)
    verdicts = pd.DataFrame(verd)
    overall = overall_verdict({r["outcome"]: r["supported"] for r in verd})
    return {"metrics": pd.DataFrame(met), "coefs": pd.DataFrame(coefs),
            "predictions": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
            "verdicts": verdicts, "overall": overall}


# --------------------------------------------------------------------------- secondary expectations

def _dir_verdict(pooled_ok: bool | None, share_ok: float | None, thresh=0.5) -> str:
    """pass if pooled direction holds and > thresh of units agree; fail if neither; mixed otherwise."""
    if pooled_ok is None or share_ok is None or not np.isfinite(share_ok):
        return "na"
    if pooled_ok and share_ok > thresh:
        return "pass"
    if not pooled_ok and share_ok <= thresh:
        return "fail"
    return "mixed"


def _corr_check(t: pd.DataFrame, y: str, x: str = "log_S", expect_sign: int = -1) -> dict:
    """Spearman of y vs x pooled and within each (planet, mission) cell (3 tempi each)."""
    pooled = _spearman(t[x], t[y])
    per = {}
    for (p, m), g in t.groupby(CELL, sort=False):
        per[f"{p}/{m}"] = _spearman(g[x], g[y])
    vals = np.array([v for v in per.values() if np.isfinite(v)])
    share = float(np.mean(np.sign(vals) == expect_sign)) if len(vals) else np.nan
    pooled_ok = None if not np.isfinite(pooled) else bool(np.sign(pooled) == expect_sign)
    return {"value": pooled, "share_cells_expected_sign": share, "n_cells": len(vals),
            "per_cell": per, "verdict": _dir_verdict(pooled_ok, share)}


def _order_check(t: pd.DataFrame, y: str, order: Sequence[str], by: str, within: str) -> dict:
    """Does mean y follow `order` (descending) across levels of `by`, pooled and within each level of `within`?"""
    def holds(g):
        m = g.groupby(by)[y].mean().reindex(order)
        if m.isna().any():
            return None, m
        return bool(np.all(np.diff(m.values) < 0)), m
    pooled, means = holds(t)
    per = {}
    for w, g in t.groupby(within, sort=False):
        per[w] = holds(g)[0]
    vals = [v for v in per.values() if v is not None]
    share = float(np.mean(vals)) if vals else np.nan
    return {"value": {k: float(v) for k, v in means.items()}, "pooled_holds": pooled,
            "share_within": share, "per_within": per, "verdict": _dir_verdict(pooled, share)}


def _argext_check(t: pd.DataFrame, y: str, target: str, by: str, within: str, largest=True) -> dict:
    def ext(g):
        m = g.groupby(by)[y].mean().dropna()
        if len(m) < 2:
            return None, m
        return bool((m.idxmax() if largest else m.idxmin()) == target), m
    pooled, means = ext(t)
    per = {w: ext(g)[0] for w, g in t.groupby(within, sort=False)}
    vals = [v for v in per.values() if v is not None]
    share = float(np.mean(vals)) if vals else np.nan
    return {"value": {k: float(v) for k, v in means.items()}, "pooled_holds": pooled,
            "share_within": share, "per_within": per, "verdict": _dir_verdict(pooled, share)}


def _clean(o):
    """Make nested stats JSON-friendly."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def secondary_checks(sc: pd.DataFrame, which: str = "adv", out: OutcomeResult | None = None,
                     prin: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each numbered secondary expectation of the prereg as a computed statistic + verdict.

    Returns a DataFrame with columns id, expectation, statistic, value, verdict (pass/fail/mixed/na), detail (dict).
    Directional verdict rule (unless stated otherwise in the row): 'pass' when the pooled statistic has
    the expected direction AND more than half of the sub-units (cells / planets / missions) agree;
    'fail' when neither; 'mixed' otherwise.
    """
    out = out if out is not None else outcomes(sc, which)
    prin = prin if prin is not None else principles(sc, which)
    t = transfer_table(out.table, prin)
    rows = []

    def add(id_, expectation, statistic, res, value=None):
        rows.append({"id": id_, "expectation": expectation, "statistic": statistic,
                     "value": _clean(res.get("value") if value is None else value),
                     "verdict": res["verdict"], "detail": _clean({k: v for k, v in res.items()
                                                                 if k not in ("value", "verdict")})})

    # 1. coordination ---------------------------------------------------------
    v = t["V_comm"].dropna()
    share = float((v >= 0).mean()) if len(v) else np.nan
    add("1a", "V_comm >= 0 in most conditions", "share of conditions with V_comm >= 0",
        {"value": share, "n": len(v), "mean": float(v.mean()) if len(v) else np.nan,
         "verdict": "na" if not len(v) else ("pass" if share > 0.5 else "fail")})

    w = t.pivot_table(index=CELL, columns="tempo", values="V_comm")
    if {0.8, 1.3} <= set(w.columns):
        diff = (w[1.3] - w[0.8]).dropna()
        sh = float((diff < 0).mean()) if len(diff) else np.nan
        res = {"value": float(diff.mean()) if len(diff) else np.nan, "share_cells_decline": sh,
               "n_cells": len(diff), "per_cell": {f"{p}/{m}": d for (p, m), d in diff.items()}}
        res["verdict"] = _dir_verdict(None if not len(diff) else bool(diff.mean() < 0), sh)
    else:
        res = {"value": np.nan, "verdict": "na"}
    add("1b", "V_comm declines at the highest tempo", "mean over cells of V_comm(tempo 1.3) - V_comm(tempo 0.8)", res)

    add("1c", "V_comm decreases as S grows (weak, non-monotone would not surprise)",
        "Spearman rho(V_comm, log S): pooled; per-cell share negative", _corr_check(t, "V_comm"))

    # 1d: jam vs corrosion/quake at matched S -> residual from mission + log S fit on corrosion/quake
    ref = t[t["planet"].isin(["corrosion", "quake"])].dropna(subset=["V_comm", "log_S"])
    jam = t[t["planet"] == "jam"].dropna(subset=["V_comm", "log_S"])
    if len(ref) >= 4 and len(jam):
        ms = [m for m in MISSIONS if m in set(ref["mission"])]
        Xr = np.column_stack([(ref["mission"] == m).astype(float) for m in ms] + [ref["log_S"]])
        cf, *_ = np.linalg.lstsq(Xr, ref["V_comm"].values, rcond=None)
        jam = jam[jam["mission"].isin(ms)]
        Xj = np.column_stack([(jam["mission"] == m).astype(float) for m in ms] + [jam["log_S"]])
        resid = jam["V_comm"].values - Xj @ cf
        sh = float(np.mean(resid < 0)) if len(resid) else np.nan
        res = {"value": float(np.mean(resid)) if len(resid) else np.nan, "share_jam_below": sh,
               "n_jam": len(resid), "ref_slope_logS": float(cf[-1]),
               "verdict": _dir_verdict(bool(np.mean(resid) < 0) if len(resid) else None, sh)}
        # raw contrast at matched (mission, tempo) as a cross-check
        piv = t.pivot_table(index=["mission", "tempo"], columns="planet", values="V_comm")
        if "jam" in piv and any(c in piv for c in ["corrosion", "quake"]):
            other = piv[[c for c in ["corrosion", "quake"] if c in piv]].mean(axis=1)
            res["raw_mean_jam_minus_corr_quake_same_mission_tempo"] = float((piv["jam"] - other).mean())
    else:
        res = {"value": np.nan, "verdict": "na"}
    add("1d", "V_comm smaller on jam than corrosion/quake at matched S",
        "mean residual of jam V_comm from (mission + log S) OLS fitted on corrosion+quake", res)

    # 2. coupling -------------------------------------------------------------
    add("2a", "V_couple ordered escort > haul > search", "mission means of V_couple (pooled; per-planet share)",
        _order_check(t, "V_couple", ["escort", "haul", "search"], by="mission", within="planet"))
    add("2b", "V_couple largest on quake", "planet means of V_couple (pooled; per-mission share quake is max)",
        _argext_check(t, "V_couple", "quake", by="planet", within="mission", largest=True))
    add("2c", "V_couple decreases with S", "Spearman rho(V_couple, log S): pooled; per-cell share negative",
        _corr_check(t, "V_couple"))

    # 3. size ---------------------------------------------------------------
    b = t["beta"].dropna()
    sh = float((b < 1).mean()) if len(b) else np.nan
    add("3a", "beta < 1 everywhere", "share of conditions with beta < 1",
        {"value": sh, "n": len(b), "max_beta": float(b.max()) if len(b) else np.nan,
         "verdict": "na" if not len(b) else ("pass" if sh == 1 else ("mixed" if sh > 0.5 else "fail"))})
    add("3b", "beta smallest for escort", "mission means of beta (pooled; per-planet share escort is min)",
        _argext_check(t, "beta", "escort", by="mission", within="planet", largest=False))
    for lam, col in zip(LAMBDAS, NSTAR_COLS):
        piv = t.pivot_table(index=["mission", "tempo"], columns="planet", values=col)
        diffs = []
        for p in ["corrosion", "combined"]:
            if p in piv and "jam" in piv:
                diffs.append((piv[p] - piv["jam"]).dropna())
        dd = pd.concat(diffs) if diffs else pd.Series(dtype=float)
        if len(dd):
            up, down = int((dd > 0).sum()), int((dd < 0).sum())
            res = {"value": float(dd.mean()), "n_pairs": len(dd), "n_higher": up, "n_lower": down,
                   "n_equal": int((dd == 0).sum())}
            res["verdict"] = ("pass" if dd.mean() > 0 and up > down else
                              "fail" if dd.mean() <= 0 and up <= down else "mixed")
        else:
            res = {"value": np.nan, "verdict": "na"}
        add(f"3c[{lam:g}]", f"corrosion/combined raise N*({lam:g}) relative to jam",
            "mean N*(corrosion or combined) - N*(jam) over matched (mission, tempo) pairs; "
            "pass if mean > 0 and more pairs higher than lower", res)

    # 4. sacrifice ------------------------------------------------------------
    cc = t[t["planet"].isin(["corrosion", "combined"])]["V_sac"].dropna()
    sh = float((cc > 0).mean()) if len(cc) else np.nan
    add("4a", "V_sac > 0 on corrosion/combined", "mean V_sac on corrosion+combined (share > 0)",
        {"value": float(cc.mean()) if len(cc) else np.nan, "share_pos": sh, "n": len(cc),
         "verdict": _dir_verdict(bool(cc.mean() > 0) if len(cc) else None, sh)})
    jm = t[t["planet"] == "jam"]["V_sac"].dropna()
    if len(jm) >= 2 and len(cc):
        se = jm.std(ddof=1) / np.sqrt(len(jm))
        tq = stats.t.ppf(0.95, len(jm) - 1)
        lo, hi = jm.mean() - tq * se, jm.mean() + tq * se
        has0 = bool(lo <= 0 <= hi)
        smaller = bool(abs(jm.mean()) < abs(cc.mean()))
        res = {"value": float(jm.mean()), "ci90": [float(lo), float(hi)], "ci_contains_0": has0,
               "mean_corrosion_combined": float(cc.mean()), "abs_smaller_than_corr_comb": smaller, "n": len(jm),
               "verdict": "pass" if has0 and smaller else ("fail" if not has0 and not smaller else "mixed")}
    else:
        res = {"value": float(jm.mean()) if len(jm) else np.nan, "verdict": "na"}
    add("4b", "V_sac ~ 0 on jam (and smaller than on corrosion/combined)",
        "mean jam V_sac; pass if its 90% t-interval over conditions contains 0 and |mean| < corrosion/combined mean",
        res)

    ex = exposure(sc, which, arms=("default",))
    exc = ex[ex["planet"].isin(["corrosion", "combined"])]
    diff = (exc["exp_last"] - exc["exp_first"]).dropna()
    sh = float((diff > 0).mean()) if len(diff) else np.nan
    per_planet = {p: float((g["exp_last"] - g["exp_first"]).mean()) for p, g in exc.groupby("planet")}
    add("4c", "default: exposure last third > first third on corrosion/combined",
        "mean over conditions of exp_last - exp_first (default arm, n in {4,6,8})",
        {"value": float(diff.mean()) if len(diff) else np.nan, "share_pos": sh, "n": len(diff),
         "per_planet": per_planet,
         "all_planets_mean": float((ex["exp_last"] - ex["exp_first"]).mean()),
         "verdict": _dir_verdict(bool(diff.mean() > 0) if len(diff) else None, sh)})

    pts = exposure_worth_points(sc, which)
    res = {}
    for scope, sub in [("corrosion+combined", pts[pts["planet"].isin(["corrosion", "combined"])]),
                       ("all planets", pts)]:
        sub = sub.dropna(subset=["exposure", "W"])
        if len(sub) >= 3:
            r = stats.spearmanr(sub["exposure"], sub["W"])
            res[scope] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(sub),
                          "pearson": float(np.corrcoef(sub["exposure"], sub["W"])[0, 1])}
        else:
            res[scope] = {"rho": np.nan, "p": np.nan, "n": len(sub)}
    main = res["corrosion+combined"]
    rho, p = main["rho"], main["p"]
    verdict = ("na" if not np.isfinite(rho) else
               "pass" if rho < 0 and p < 0.1 else "mixed" if rho < 0 else "fail")
    add("4d", "exposure correlates negatively with W(t) across condition x removal time (corrosion/combined)",
        "Spearman rho(exposure in the third starting at t, W(t)), default arm n=6; pass if rho<0 and p<0.1, "
        "mixed if rho<0 and p>=0.1", {"value": rho, "verdict": verdict, **res})
    return pd.DataFrame(rows)


def exposure_worth_points(sc: pd.DataFrame, which: str = "adv") -> pd.DataFrame:
    """(condition x removal time) points: exposure in the third starting at t (default, n=6, no removal) and W(t)."""
    pr = principles(sc, which)
    ex = exposure(sc, which, arms=("default",))
    m = pr.merge(ex, on=COND, how="left")
    rows = []
    for t, wcol, ecol in zip(REMOVE_TS, ["R_early", "R_mid", "R_late"], ["exp6_first", "exp6_mid", "exp6_last"]):
        r = m[COND].copy()
        r["remove_t"] = t
        r["W"] = m[wcol].values
        r["exposure"] = m[ecol].values
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- validity

def validity_checks(sc: pd.DataFrame) -> dict:
    """Validity checks that gate interpretation.

    Returns {"heuristic": per-cell DataFrame, "adv_drop": per-cell DataFrame, "br": per-row/per-cell DataFrame,
    "summary": dict}.
    """
    nat = score_view(sc, "nature")
    cells = pd.MultiIndex.from_product([PLANETS, MISSIONS], names=CELL)
    # (1) learned default vs scripted heuristic: nature, n=6, tempo 0.8
    sel = nat[(nat["n"] == 6) & np.isclose(nat["tempo"], 0.8) & (nat["remove_t"] == -1)]

    def agg(arm):
        d = sel[sel["arm"] == arm]
        if d.empty:
            return pd.DataFrame(index=cells, columns=["J", "se", "k"], dtype=float)
        g = d.groupby(CELL)
        return pd.DataFrame({"J": g["J"].mean(),
                             "se": g["J_sem"].apply(lambda s: np.sqrt(np.nansum(s.values ** 2)) / len(s)),
                             "k": g["J"].size()}).reindex(cells)
    dfl, heu = agg("default"), agg("heuristic")
    h = pd.DataFrame({"J_default": dfl["J"], "J_heuristic": heu["J"], "n_seeds_default": dfl["k"]}, index=cells)
    se = np.sqrt(dfl["se"].astype(float).fillna(0) ** 2 + heu["se"].astype(float).fillna(0) ** 2)
    h["diff"] = h["J_default"] - h["J_heuristic"]
    h["margin_1.645se"] = 1.645 * se
    h["beats"] = h["diff"] >= 0
    h["passes"] = (h["diff"] >= -h["margin_1.645se"]) & h["diff"].notna()
    h = h.reset_index()
    n_pass = int(h["passes"].sum())
    n_eval = int(h["diff"].notna().sum())

    # (2) adversary vs nature drop for default
    sc_def = sc[(sc["arm"] == "default") & (sc["remove_t"] == -1)].copy()
    sc_def["drop"] = sc_def["J_nature"] - sc_def["J_adv"]
    sc_def["rel_drop"] = rel(sc_def["J_nature"], sc_def["J_adv"])
    g = sc_def.groupby(CELL)
    a = pd.DataFrame({"J_nature": g["J_nature"].mean(), "J_adv": g["J_adv"].mean(),
                      "mean_drop": g["drop"].mean(), "mean_rel_drop": g["rel_drop"].mean(),
                      "share_rows_adv_lower": g["drop"].apply(lambda s: float((s.dropna() > 0).mean())
                                                              if s.notna().any() else np.nan)}).reindex(cells)
    a["reduces"] = a["mean_drop"] > 0
    a = a.reset_index()
    n_red = int(a["reduces"].sum())
    n_red_eval = int(a["mean_drop"].notna().sum())

    # (3) best-response gap
    br = sc[sc["J_br"].notna()].copy()
    br["gap"] = br["J_adv"] - br["J_br"]
    br["rel_gap"] = rel(br["J_adv"], br["J_br"])
    if len(br):
        gb = br.groupby(CELL)
        brc = pd.DataFrame({"n_rows": gb.size(), "n_seeds": gb["seed"].nunique(),
                            "mean_gap": gb["gap"].mean(), "max_gap": gb["gap"].max(),
                            "mean_rel_gap": gb["rel_gap"].mean()}).reindex(cells).reset_index()
    else:
        brc = pd.DataFrame(index=cells, columns=["n_rows", "n_seeds", "mean_gap", "max_gap", "mean_rel_gap"],
                           dtype=float).reset_index()
    brc["n_rows"] = brc["n_rows"].fillna(0).astype(int)
    summary = {
        "heuristic_cells_pass": n_pass, "heuristic_cells_evaluated": n_eval,
        "heuristic_gate_pass": n_pass >= 9,
        "heuristic_failing_cells": [f"{r.planet}/{r.mission}" for r in h.itertuples() if not r.passes],
        "adv_reduces_cells": n_red, "adv_cells_evaluated": n_red_eval,
        "adv_gate_pass": n_red > 6,
        "br_cells_covered": int((brc["n_rows"] > 0).sum()),
        "br_mean_gap": float(br["gap"].mean()) if len(br) else None,
        "br_mean_rel_gap": float(br["rel_gap"].mean()) if len(br) else None,
        "br_share_rows_pool_overestimates": float((br["gap"] > 0).mean()) if len(br) else None,
    }
    return {"heuristic": h, "adv_drop": a, "br": brc,
            "br_rows": br[KEY + ["J_adv", "J_br", "gap", "rel_gap"]], "summary": summary}


# --------------------------------------------------------------------------- bootstrap

def bootstrap_outcomes(df: pd.DataFrame, which: str = "adv", n_boot: int = 200, level: float = 0.9,
                       seed: int = 0) -> pd.DataFrame:
    """Parametric-resampling intervals for the V outcomes per condition.

    Each eval row's J is redrawn as Normal(J, J_sem) (J_sem NaN -> 0); for the adversarial score the pool
    minimum is re-taken per draw; then seed-mean J -> rel() -> mean over n in {4,6,8} as in ``outcomes``.
    Only episode-sampling noise is propagated (not training-seed variation). Returns one row per condition
    with <V>_lo, <V>_hi (central ``level`` interval) and <V>_bsd (bootstrap sd).
    """
    d = normalize_eval(df)
    cond_index = _conditions(d)
    mode = {"adv": "adv", "nature": "nature", "br": "br"}[which]
    r = d[(d["mode"] == mode) & (d["remove_t"] == -1) & d["arm"].isin(ARMS) & d["n"].isin(NS_AVG)
          & d["J"].notna()].reset_index(drop=True)
    res = pd.DataFrame(index=cond_index)
    if r.empty:
        for v in V_OUTCOMES:
            res[f"{v}_lo"] = res[f"{v}_hi"] = res[f"{v}_bsd"] = np.nan
        return res.reset_index()
    rng = np.random.default_rng(seed)
    sem = r["J_sem"].fillna(0).clip(lower=0).values
    draws = r["J"].values[:, None] + sem[:, None] * rng.standard_normal((len(r), n_boot))
    D = pd.DataFrame(draws, index=pd.MultiIndex.from_frame(r[KEY]))
    D = D.groupby(level=KEY).min() if mode != "nature" else D.groupby(level=KEY).mean()
    jbar = D.groupby(level=COND + ["arm", "n"]).mean()
    q = (1 - level) / 2
    for v, fr in _v_from_jbar(jbar, cond_index).items():
        arr = fr.values
        with np.errstate(all="ignore"):
            ok = np.isfinite(arr).any(axis=1)
            lo = np.full(len(arr), np.nan)
            hi = np.full(len(arr), np.nan)
            sd = np.full(len(arr), np.nan)
            if ok.any():
                lo[ok] = np.nanquantile(arr[ok], q, axis=1)
                hi[ok] = np.nanquantile(arr[ok], 1 - q, axis=1)
                sd[ok] = np.nanstd(arr[ok], axis=1, ddof=1)
        res[f"{v}_lo"], res[f"{v}_hi"], res[f"{v}_bsd"] = lo, hi, sd
    return res.reset_index()


# --------------------------------------------------------------------------- cell-level answers

def cell_curves(sc: pd.DataFrame, which: str = "adv") -> pd.DataFrame:
    """Tempo-averaged J(N) per (planet, mission, arm, seed, n) (remove_t=-1)."""
    v = score_view(sc, which)
    v = v[v["remove_t"] == -1]
    return v.groupby(CELL + ["arm", "seed", "n"], as_index=False)["J"].mean()


def answers(sc: pd.DataFrame, out_table: pd.DataFrame, which: str = "adv") -> pd.DataFrame:
    """Per cell (planet, mission): the practical answers to the four design questions."""
    v = score_view(sc, which)
    full = v[(v["remove_t"] == -1) & v["n"].isin(NS_AVG)]
    cells = pd.MultiIndex.from_product([PLANETS, MISSIONS], names=CELL)
    res = pd.DataFrame(index=cells)
    comm = {"silent": "silent", "default": "gossip", "blackboard": "blackboard"}
    jc = (full[full["arm"].isin(comm)].groupby(CELL + ["arm", "seed"])["J"].mean()
          .groupby(level=CELL + ["arm"]).mean())
    for arm, lab in comm.items():
        res[f"J_{lab}"] = (jc.xs(arm, level="arm").reindex(cells) if len(jc) and
                           arm in jc.index.get_level_values("arm") else np.nan)
    jcols = [f"J_{lab}" for lab in comm.values()]
    best = res[jcols].astype(float)
    res["best_comm"] = best.apply(lambda r: r.idxmax()[2:] if r.notna().any() else None, axis=1)
    res["best_comm_margin"] = best.apply(lambda r: (np.sort(r.dropna().values)[-1] - np.sort(r.dropna().values)[-2])
                                         if r.notna().sum() >= 2 else np.nan, axis=1)
    o = out_table.groupby(CELL)[["V_comm", "V_gossip", "V_couple", "V_sac", "loss_default", "loss_protected",
                                "coupled_frac_default", "beta"]].mean().reindex(cells)
    for c in o.columns:
        res[c] = o[c]
    # N* from the tempo-averaged, seed-mean J(N) curve of default
    cc = cell_curves(sc, which)
    cd = cc[cc["arm"] == "default"].groupby(CELL + ["n"])["J"].mean()
    for col in NSTAR_COLS:
        res[col] = np.nan
    for cell in cells:
        try:
            curve = cd.loc[cell]
        except KeyError:
            continue
        s = _size_outcomes(curve)
        for col in NSTAR_COLS:
            res.loc[cell, col] = s[col]
    return res.reset_index()
