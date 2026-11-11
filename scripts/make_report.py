"""Build the pre-registered report: tables, figures and summary.json.

    python scripts/make_report.py --eval results/eval.csv --runs results/raw/runs --out results/

Runs on partial data: missing arms / seeds / cells / runs become NaN or empty panels, never a crash.
Writes results/tables/*.csv, results/figures/*.png and results/summary.json.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from hostile_planets import analysis as A  # noqa: E402

# ----------------------------------------------------------------------------- style
# Validated (dataviz skill validator) categorical slots, light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ARM_COLOR = {"default": "#2a78d6", "silent": "#eb6834", "blackboard": "#1baf7a",
             "solo": "#eda100", "protected": "#e87ba4", "heuristic": MUTED}
ARM_LABEL = {"default": "default (gossip)", "silent": "silent", "blackboard": "blackboard",
             "solo": "solo (no coupling)", "protected": "protected", "heuristic": "scripted heuristic"}
# 4 planets: blue / orange / aqua / deep violet passes all-pairs CVD + normal-vision floors.
PLANET_COLOR = {"corrosion": "#2a78d6", "jam": "#eb6834", "quake": "#1baf7a", "combined": "#4a3aa7"}
MISSION_MARKER = {"search": "o", "haul": "s", "escort": "^"}
MODEL_COLOR = {"P": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a"}
MODEL_MARKER = {"P": "o", "B1": "s", "B2": "D"}
MODEL_LABEL = {"P": "P  (mission + log S + R_late)", "B1": "B1 (mission means)", "B2": "B2 (mission + tempo)"}
DIVERGING = LinearSegmentedColormap.from_list(
    "bgr", ["#a32b2b", "#e34948", "#f0a5a0", "#f0efec", "#9ec5f4", "#3987e5", "#104281"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
OUTCOME_LABEL = {"V_comm": "V_comm  (blackboard vs silent)", "V_gossip": "V_gossip  (gossip vs silent)",
                 "V_couple": "V_couple  (coupling on vs off)", "V_sac": "V_sac  (price of safety)",
                 "beta": "beta  (size scaling exponent)", "loss_default": "loss rate, default (deaths/robot)",
                 "Nstar_0.5": "N*(0.5)"}
DPI = 200


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8.5, "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True, "lines.linewidth": 2,
        "legend.frameon": False, "xtick.major.size": 0, "ytick.major.size": 0,
    })


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _empty(ax, msg="no data"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, transform=ax.transAxes, fontsize=8)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.grid(False)


def _row_label(ax, text):
    ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction", xytext=(-ax.yaxis.labelpad - 34, 0),
                textcoords="offset points", ha="right", va="center", fontsize=10, color=INK, rotation=90)


# ----------------------------------------------------------------------------- figures

def fig_jn_curves(sc, path, which="adv"):
    """(a) J(N) per cell, one line per arm, tempo-averaged, seed spread band (min-max over seeds)."""
    cc = A.cell_curves(sc, which)
    fig, axes = plt.subplots(4, 3, figsize=(10, 11), sharex=True, sharey="col")
    for i, p in enumerate(A.PLANETS):
        for j, m in enumerate(A.MISSIONS):
            ax = axes[i, j]
            d = cc[(cc["planet"] == p) & (cc["mission"] == m)]
            if d.empty:
                _empty(ax)
            for arm in A.ARMS + ["heuristic"]:
                da = d[d["arm"] == arm]
                if da.empty:
                    continue
                g = da.groupby("n")["J"]
                mean, lo, hi = g.mean(), g.min(), g.max()
                lw = 1.2 if arm == "heuristic" else 1.8
                ax.plot(mean.index, mean.values, color=ARM_COLOR[arm], lw=lw, marker="o", ms=3,
                        zorder=3 if arm == "default" else 2)
                if da["seed"].nunique() > 1:
                    ax.fill_between(mean.index, lo.values, hi.values, color=ARM_COLOR[arm], alpha=0.15, lw=0)
            ax.set_xscale("log")
            ax.set_xticks(A.NS_ALL)
            ax.set_xticklabels([str(n) for n in A.NS_ALL])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(m, fontsize=11)
            if i == 3:
                ax.set_xlabel("swarm size N")
            if j == 0:
                ax.set_ylabel(f"{p}\n\nJ ({'adversarial' if which == 'adv' else which})")
    handles = [Line2D([], [], color=ARM_COLOR[a], lw=2 if a != "heuristic" else 1.2, marker="o", ms=3,
                      label=ARM_LABEL[a]) for a in A.ARMS + ["heuristic"]]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), fontsize=8.5,
               handlelength=1.6, columnspacing=1.2)
    fig.suptitle(f"Mission metric J vs swarm size ({'pool-minimum adversarial' if which == 'adv' else which} score, "
                 "mean over tempo; band = range over seeds)", y=1.025, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return _save(fig, path)


def fig_heatmaps(ans, path):
    """(b) planet x mission heatmaps of V_comm, V_couple, V_sac (diverging, centred 0) and N*(0.5)."""
    panels = ["V_comm", "V_couple", "V_sac", "Nstar_0.5"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, col in zip(axes, panels):
        mat = (ans.set_index(["planet", "mission"])[col].astype(float)
               .unstack("mission").reindex(index=A.PLANETS, columns=A.MISSIONS))
        vals = mat.values
        ax.grid(False)
        if col.startswith("V_"):
            vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
            vmax = max(vmax, 1e-3)
            im = ax.imshow(vals, cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        else:
            im = ax.imshow(vals, cmap=SEQUENTIAL, vmin=2, vmax=12, aspect="auto")
        for (r, c), v in np.ndenumerate(vals):
            if not np.isfinite(v):
                ax.text(c, r, "n/a", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(c, r, f"{v:.0f}" if col.startswith("N") else f"{v:+.2f}", ha="center", va="center",
                    fontsize=9, color="#ffffff" if lum < 0.5 else INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(A.MISSIONS)
        ax.set_yticks(range(4))
        ax.set_yticklabels(A.PLANETS if ax is axes[0] else [])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(OUTCOME_LABEL.get(col, col).replace("  ", "\n", 1), fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle("Answers per cell (adversarial score; V's averaged over tempo; N* from the tempo-averaged J(N))",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def fig_collapse(table, tt, path):
    """(c) each outcome vs log S; colour = planet, marker = mission; model-P fit (single-hazard planets)."""
    outs = A.PRIMARY_OUTCOMES
    coefs = tt["coefs"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    for ax, y in zip(axes.flat, outs):
        d = table.dropna(subset=[y, "log_S"])
        if d.empty:
            _empty(ax)
        for p in A.PLANETS:
            for m in A.MISSIONS:
                dd = d[(d["planet"] == p) & (d["mission"] == m)]
                if dd.empty:
                    continue
                held = p == "combined"
                ax.scatter(dd["log_S"], dd[y], color=PLANET_COLOR[p], marker=MISSION_MARKER[m],
                           s=58 if held else 34, edgecolor=INK if held else SURFACE, linewidth=1.3 if held else 0.8,
                           zorder=4 if held else 3)
        c = coefs[(coefs["outcome"] == y) & (coefs["split"] == "headline") & (coefs["model"] == "P")]
        c = dict(zip(c["param"], c["value"]))
        train = d[d["planet"].isin(A.SINGLE_PLANETS)]
        if c and np.isfinite(c.get("log_S", np.nan)) and len(train):
            for m in A.MISSIONS:
                k = f"mission[{m}]"
                dm = d[d["mission"] == m]
                rl = train.loc[train["mission"] == m, "R_late"].mean()
                if k not in c or dm.empty or not np.isfinite(rl):
                    continue
                xs = np.linspace(dm["log_S"].min(), dm["log_S"].max(), 20)
                ax.plot(xs, c[k] + c["log_S"] * xs + c["R_late"] * rl, color=INK2, lw=1.1, zorder=2)
                ax.text(xs[-1], c[k] + c["log_S"] * xs[-1] + c["R_late"] * rl, f" {m}", fontsize=7,
                        color=INK2, va="center", ha="left")
            fit_txt = f"\nP fit: a (log S) = {c['log_S']:+.3f},  b (R_late) = {c['R_late']:+.3f}"
        else:
            fit_txt = "\nP fit: n/a"
        if y.startswith("V_"):
            ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_title(OUTCOME_LABEL.get(y, y) + fit_txt, fontsize=9.5, linespacing=1.5)
        ax.set_xlabel("log S  (staleness ratio, measured)")
        ax.margins(x=0.12)
    ph = [Line2D([], [], ls="", marker="o", ms=7, color=PLANET_COLOR[p],
                 markeredgecolor=INK if p == "combined" else SURFACE,
                 label=f"{p} (held out)" if p == "combined" else p) for p in A.PLANETS]
    mh = [Line2D([], [], ls="", marker=MISSION_MARKER[m], ms=7, color=MUTED, label=m) for m in A.MISSIONS]
    lh = [Line2D([], [], color=INK2, lw=1.1, label="model P fit on corrosion+jam+quake\n(one line per mission, at its mean R_late)")]
    fig.legend(handles=ph + mh + lh, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.035), fontsize=8,
               handletextpad=0.3, columnspacing=1.0)
    fig.suptitle("Data collapse: outcomes vs measured staleness (adversarial score, one point per planet x mission x tempo)",
                 y=1.07, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig_pred_obs(tt, path):
    """(d) predicted vs observed on the held-out combined planet (headline split), P vs B1 vs B2."""
    pr = tt["predictions"]
    met = tt["metrics"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    for ax, y in zip(axes.flat, A.PRIMARY_OUTCOMES):
        d = pr[(pr["outcome"] == y) & (pr["split"] == "headline")] if len(pr) else pr
        if d is None or d.empty or d["obs"].notna().sum() == 0:
            _empty(ax)
            ax.set_title(OUTCOME_LABEL.get(y, y), fontsize=9.5)
            continue
        allv = np.concatenate([d["obs"].values] + [d[f"pred_{m}"].values for m in MODEL_COLOR])
        allv = allv[np.isfinite(allv)]
        lo, hi = allv.min(), allv.max()
        pad = 0.08 * (hi - lo if hi > lo else 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=AXIS, lw=1, zorder=1)
        handles = []
        for m in MODEL_COLOR:
            ax.scatter(d["obs"], d[f"pred_{m}"], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], s=34,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
            r = met[(met["outcome"] == y) & (met["split"] == "headline") & (met["model"] == m)]
            mae = r["mae"].iloc[0] if len(r) else np.nan
            rho = r["spearman"].iloc[0] if len(r) else np.nan
            handles.append(Line2D([], [], ls="", marker=MODEL_MARKER[m], color=MODEL_COLOR[m], ms=6,
                                  label=f"{m}: MAE {mae:.3f}, rho {rho:.2f}"))
        ax.legend(handles=handles, loc="upper left", fontsize=7.5, handletextpad=0.2, borderaxespad=0.2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(OUTCOME_LABEL.get(y, y), fontsize=9.5)
        ax.set_xlabel("observed (combined planet)")
        ax.set_ylabel("predicted")
    fh = [Line2D([], [], ls="", marker=MODEL_MARKER[m], color=MODEL_COLOR[m], ms=7, label=MODEL_LABEL[m])
          for m in MODEL_COLOR]
    fig.legend(handles=fh, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.suptitle("Held-out prediction of the combined planet from the three single-hazard planets "
                 "(9 conditions; grey line = perfect prediction)", y=1.065, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig_worth(sc, path):
    """(e) worth trajectory W(t) per mission, one line per planet; rows = nature / adversarial score."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True, sharey=True)
    for i, which in enumerate(["nature", "adv"]):
        pr = A.principles(sc, which)
        for j, m in enumerate(A.MISSIONS):
            ax = axes[i, j]
            any_ = False
            for p in A.PLANETS:
                d = pr[(pr["planet"] == p) & (pr["mission"] == m)]
                w = d[["R_early", "R_mid", "R_late"]].astype(float)
                if w.notna().sum().sum() == 0:
                    continue
                any_ = True
                mean, lo, hi = w.mean().values, w.min().values, w.max().values
                ax.fill_between(A.REMOVE_TS, lo, hi, color=PLANET_COLOR[p], alpha=0.12, lw=0)
                ax.plot(A.REMOVE_TS, mean, color=PLANET_COLOR[p], marker="o", ms=4)
            if not any_:
                _empty(ax)
            ax.axhline(1, color=AXIS, lw=0.8, zorder=1)
            ax.set_xticks(A.REMOVE_TS)
            if i == 0:
                ax.set_title(m, fontsize=11)
            if i == 1:
                ax.set_xlabel("removal step t")
            if j == 0:
                ax.set_ylabel(f"{'nature' if which == 'nature' else 'adversarial'} score\n\nW(t)  (x J/N)")
    h = [Line2D([], [], color=PLANET_COLOR[p], marker="o", ms=4, label=p) for p in A.PLANETS]
    fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Worth trajectory: loss in final J from removing one robot at step t (default arm, N=6; "
                 "line = mean over tempo, band = range; W=1 is an average robot)", y=1.06, fontsize=10.5)
    fig.tight_layout()
    return _save(fig, path)


def fig_exposure(sc, path, which="adv"):
    """(f) hazard exposure per live robot by mission third, default vs protected, per planet x mission."""
    ex = A.exposure(sc, which, arms=("default", "protected"))
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 7.6), sharex=True, sharey=True)
    x = [0, 1, 2]
    for i, m in enumerate(A.MISSIONS):
        for j, p in enumerate(A.PLANETS):
            ax = axes[i, j]
            any_ = False
            for arm in ["default", "protected"]:
                d = ex[(ex["planet"] == p) & (ex["mission"] == m) & (ex["arm"] == arm)]
                v = d[["exp_first", "exp_mid", "exp_last"]].astype(float)
                if v.notna().sum().sum() == 0:
                    continue
                any_ = True
                ax.plot(x, v.mean().values, color=ARM_COLOR[arm], marker="o", ms=4)
            if not any_:
                _empty(ax)
            ax.set_xticks(x)
            ax.set_xticklabels(["0-50", "50-100", "100-150"])
            if i == 0:
                ax.set_title(p, fontsize=11)
            if i == 2:
                ax.set_xlabel("mission third (steps)")
            if j == 0:
                ax.set_ylabel(f"{m}\n\nexposure / live robot")
    h = [Line2D([], [], color=ARM_COLOR[a], marker="o", ms=4, label=ARM_LABEL[a]) for a in ["default", "protected"]]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Hazard exposure (corrosion + quake field) by mission third "
                 f"({'adversarial' if which == 'adv' else which} score; mean over tempo, N in {{4,6,8}}, seeds)",
                 y=1.055, fontsize=10.5)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------- training curves

_CFG_KEYS = ["config", "cfg", "args", "params", "hparams", "meta", "settings"]
_CURVE_KEYS = ["curve", "curves", "history", "log", "logs", "metrics", "train_curve", "training", "records"]
_ITER_KEYS = ["iter", "iteration", "iters", "it", "step", "update", "epoch", "i"]
_RUN_RE = re.compile(r"^(?P<planet>[a-z]+)__(?P<mission>[a-z]+)__(?P<arm>[a-z]+)__s(?P<seed>-?\d+)$")
_COMM_TO_ARM = {"silent": "silent", "blackboard": "blackboard"}


def _norm(k):
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _find_col(cols, exact, contains_all=(), contains_any=(), exclude=()):
    nm = {_norm(c): c for c in cols}
    for e in exact:
        if _norm(e) in nm:
            return nm[_norm(e)]
    for n, c in nm.items():
        if all(s in n for s in contains_all) and (not contains_any or any(s in n for s in contains_any)) \
                and not any(s in n for s in exclude):
            return c
    return None


def _curve_frame(obj):
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        lists = {k: v for k, v in obj.items() if isinstance(v, list)}
        if lists and len({len(v) for v in lists.values()}) == 1:
            return pd.DataFrame(lists)
    return None


def load_runs(runs_dir):
    """Read training-curve jsons defensively -> long DataFrame (planet, mission, arm, seed, iter, J_nature, J_adv)."""
    frames, problems = [], []
    for f in sorted(glob.glob(os.path.join(runs_dir or "", "*.json"))):
        try:
            with open(f) as fh:
                js = json.load(fh)
            cfg = next((js[k] for k in _CFG_KEYS if isinstance(js, dict) and isinstance(js.get(k), dict)), {})
            cur = None
            if isinstance(js, dict):
                for k in _CURVE_KEYS:
                    if k in js:
                        cur = _curve_frame(js[k])
                        if cur is not None:
                            break
                if cur is None:  # any list-of-dicts value
                    for v in js.values():
                        cur = _curve_frame(v) if isinstance(v, list) else None
                        if cur is not None:
                            break
            elif isinstance(js, list):
                cur = _curve_frame(js)
            if cur is None or cur.empty:
                problems.append(f"{os.path.basename(f)}: no curve found")
                continue
            ic = _find_col(cur.columns, _ITER_KEYS)
            it = pd.to_numeric(cur[ic], errors="coerce") if ic else pd.Series(np.arange(len(cur)))
            jn = _find_col(cur.columns, ["J_nature", "j_nat", "J_nat", "nature_J", "eval_J_nature"],
                           contains_any=("nat",), exclude=("sem", "std", "adv"))
            ja = _find_col(cur.columns, ["J_adv", "j_adv", "adv_J", "J_adversarial", "eval_J_adv"],
                           contains_any=("adv",), exclude=("sem", "std", "nat", "loss", "ret", "reward"))
            if jn is None and ja is None:
                jn = _find_col(cur.columns, ["J", "metric", "score"])
            name = os.path.splitext(os.path.basename(f))[0]
            mt = _RUN_RE.match(name)
            meta = mt.groupdict() if mt else {}
            planet = meta.get("planet") or cfg.get("planet")
            mission = meta.get("mission") or cfg.get("mission")
            arm = meta.get("arm") or cfg.get("arm") or cfg.get("name", "").split("__")[2:3] or None
            if isinstance(arm, list):
                arm = arm[0] if arm else None
            if arm is None:
                comm, coup, prot = cfg.get("comm"), cfg.get("coupling"), cfg.get("protect")
                arm = (_COMM_TO_ARM.get(comm) or ("solo" if str(coup) in ("0", "False") else
                       "protected" if str(prot) in ("1", "True") else "default"))
            seed = meta.get("seed", cfg.get("seed", -1))
            frames.append(pd.DataFrame({
                "planet": planet, "mission": mission, "arm": arm, "seed": int(seed), "iter": it.values,
                "J_nature": pd.to_numeric(cur[jn], errors="coerce").values if jn else np.nan,
                "J_adv": pd.to_numeric(cur[ja], errors="coerce").values if ja else np.nan}))
        except Exception as e:  # noqa: BLE001 - never let one bad file kill the report
            problems.append(f"{os.path.basename(f)}: {type(e).__name__}: {e}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["planet", "mission", "arm", "seed", "iter", "J_nature", "J_adv"])
    return df, problems


def fig_training(runs, path):
    """(g) training curves: J vs iteration per cell; colour = arm; solid = nature, dashed = adversarial; seed mean."""
    fig, axes = plt.subplots(4, 3, figsize=(10, 11), sharex=True, sharey="col")
    for i, p in enumerate(A.PLANETS):
        for j, m in enumerate(A.MISSIONS):
            ax = axes[i, j]
            d = runs[(runs["planet"] == p) & (runs["mission"] == m)]
            if d.empty:
                _empty(ax, "no runs yet")
            for arm in A.ARMS:
                da = d[d["arm"] == arm]
                if da.empty:
                    continue
                g = da.groupby("iter")[["J_nature", "J_adv"]].mean()
                if g["J_nature"].notna().any():
                    ax.plot(g.index, g["J_nature"], color=ARM_COLOR[arm], lw=1.5)
                if g["J_adv"].notna().any():
                    ax.plot(g.index, g["J_adv"], color=ARM_COLOR[arm], lw=1.2, ls=(0, (3, 2)))
            if i == 0:
                ax.set_title(m, fontsize=11)
            if i == 3:
                ax.set_xlabel("training iteration")
            if j == 0:
                ax.set_ylabel(f"{p}\n\nJ")
    h = [Line2D([], [], color=ARM_COLOR[a], lw=2, label=ARM_LABEL[a]) for a in A.ARMS]
    h += [Line2D([], [], color=INK2, lw=1.5, label="nature hazards"),
          Line2D([], [], color=INK2, lw=1.2, ls=(0, (3, 2)), label="co-trained adversary")]
    fig.legend(handles=h, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.0), fontsize=8, columnspacing=1.0,
               handlelength=2.2)
    fig.suptitle("Training curves (mean over seeds)", y=1.02, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return _save(fig, path)


# ----------------------------------------------------------------------------- main

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict(orient="records"))
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is pd.NA or o is pd.NaT:
        return None
    return o


def _csv(df, path):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == object and d[c].map(lambda v: isinstance(v, (dict, list))).any():
            d[c] = d[c].map(lambda v: json.dumps(_jsonable(v)) if isinstance(v, (dict, list)) else v)
    d.to_csv(path, index=False)


def build(eval_path, runs_dir, out_dir, n_boot=200, figures=True, log=print):
    tdir, fdir = os.path.join(out_dir, "tables"), os.path.join(out_dir, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    df = pd.read_csv(eval_path, keep_default_na=True)
    if "adv_name" in df.columns:
        df["adv_name"] = df["adv_name"].fillna("")
    sc = A.scores(df)
    _csv(sc, os.path.join(tdir, "scores.csv"))

    res = {}
    for which in ["adv", "nature"]:
        o = A.outcomes(sc, which)
        pr = A.principles(sc, which)
        tab = A.transfer_table(o.table, pr)
        boot = A.bootstrap_outcomes(df, which, n_boot=n_boot)
        tab = tab.merge(boot, on=A.COND, how="left")
        _csv(tab, os.path.join(tdir, f"outcomes_{which}.csv"))
        _csv(o.per_seed, os.path.join(tdir, f"outcomes_per_seed_{which}.csv"))
        _csv(o.spread, os.path.join(tdir, f"seed_spread_{which}.csv"))
        _csv(pr, os.path.join(tdir, f"principles_{which}.csv"))
        _csv(A.exposure(sc, which), os.path.join(tdir, f"exposure_{which}.csv"))
        _csv(A.exposure_worth_points(sc, which), os.path.join(tdir, f"exposure_vs_worth_{which}.csv"))
        tt = A.transfer_test(tab)
        for k in ["metrics", "coefs", "predictions", "verdicts"]:
            _csv(tt[k], os.path.join(tdir, f"transfer_{k}_{which}.csv"))
        sec = A.secondary_checks(sc, which, o, pr)
        _csv(sec, os.path.join(tdir, f"secondary_checks_{which}.csv"))
        ans = A.answers(sc, o.table, which)
        _csv(ans, os.path.join(tdir, f"answers_{which}.csv"))
        res[which] = dict(out=o, prin=pr, table=tab, tt=tt, sec=sec, ans=ans)

    val = A.validity_checks(sc)
    _csv(val["heuristic"], os.path.join(tdir, "validity_heuristic.csv"))
    _csv(val["adv_drop"], os.path.join(tdir, "validity_adv_drop.csv"))
    _csv(val["br"], os.path.join(tdir, "validity_best_response.csv"))
    _csv(val["br_rows"], os.path.join(tdir, "validity_best_response_rows.csv"))

    # sensitivity: drop cells failing the heuristic gate
    failing = [tuple(c.split("/")) for c in val["summary"]["heuristic_failing_cells"]]
    tab = res["adv"]["table"]
    mask = ~tab.set_index(["planet", "mission"]).index.isin(failing)
    tt_sens = A.transfer_test(tab[mask])
    _csv(tt_sens["verdicts"], os.path.join(tdir, "transfer_verdicts_adv_excl_failed_heuristic_cells.csv"))
    _csv(tt_sens["metrics"], os.path.join(tdir, "transfer_metrics_adv_excl_failed_heuristic_cells.csv"))

    runs, problems = load_runs(runs_dir) if runs_dir else (pd.DataFrame(), ["no --runs given"])
    if len(runs):
        _csv(runs, os.path.join(tdir, "training_curves.csv"))

    figs, fig_errors = {}, {}
    if figures:
        set_style()
        tasks = {
            "a_J_vs_N": lambda p: fig_jn_curves(sc, p, "adv"),
            "b_heatmaps": lambda p: fig_heatmaps(res["adv"]["ans"], p),
            "c_collapse": lambda p: fig_collapse(res["adv"]["table"], res["adv"]["tt"], p),
            "d_pred_vs_obs_combined": lambda p: fig_pred_obs(res["adv"]["tt"], p),
            "e_worth_trajectory": lambda p: fig_worth(sc, p),
            "f_exposure_thirds": lambda p: fig_exposure(sc, p, "adv"),
            "g_training_curves": lambda p: fig_training(runs if len(runs) else load_runs(None)[0], p),
        }
        for name, fn in tasks.items():
            path = os.path.join(fdir, f"{name}.png")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    figs[name] = fn(path)
            except Exception as e:  # noqa: BLE001 - a broken figure must not kill the report
                plt.close("all")
                fig_errors[name] = f"{type(e).__name__}: {e}"
                log(f"[warn] figure {name} failed:\n{traceback.format_exc()}")

    d = A.normalize_eval(df)
    coverage = {
        "n_rows": int(len(df)),
        "rows_by_mode": d["mode"].value_counts().to_dict(),
        "seeds_by_arm": {a: sorted(int(s) for s in g["seed"].unique()) for a, g in d.groupby("arm")},
        "cells_with_rows": int(d[["planet", "mission"]].drop_duplicates().shape[0]),
        "conditions_with_all_V": int(res["adv"]["table"][A.V_OUTCOMES].notna().all(axis=1).sum()),
        "training_runs_loaded": int(runs[["planet", "mission", "arm", "seed"]].drop_duplicates().shape[0])
        if len(runs) else 0,
        "training_run_problems": problems[:50],
    }
    adv = res["adv"]
    headline_coefs = adv["tt"]["coefs"]
    headline_coefs = headline_coefs[headline_coefs["split"] == "headline"]
    summary = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"eval": os.path.abspath(eval_path), "runs": os.path.abspath(runs_dir) if runs_dir else None},
        "coverage": coverage,
        "primary_transfer_test": {
            "score": "adversarial (pool minimum)",
            "overall": adv["tt"]["overall"],
            "per_outcome": adv["tt"]["verdicts"],
            "metrics": adv["tt"]["metrics"],
            "headline_coefficients": headline_coefs[["outcome", "model", "param", "value"]],
        },
        "sensitivity_excluding_failed_heuristic_cells": {
            "excluded_cells": val["summary"]["heuristic_failing_cells"],
            "overall": tt_sens["overall"], "per_outcome": tt_sens["verdicts"]},
        "exploratory_transfer_nature_score": {"overall": res["nature"]["tt"]["overall"],
                                              "per_outcome": res["nature"]["tt"]["verdicts"]},
        "secondary_expectations": {w: res[w]["sec"][["id", "expectation", "statistic", "value", "verdict"]]
                                   for w in ["adv", "nature"]},
        "validity": val["summary"],
        "seed_spread": {w: res[w]["out"].spread for w in ["adv", "nature"]},
        "answers_per_cell_adv": adv["ans"],
        "outcome_means_adv": {y: float(adv["table"][y].mean()) if adv["table"][y].notna().any() else None
                              for y in A.OUTCOME_COLS},
        "ambiguity_resolutions": A.AMBIGUITIES,
        "figures": {k: os.path.relpath(v, out_dir) for k, v in figs.items()},
        "figure_errors": fig_errors,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(_jsonable(summary), f, indent=1)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default=os.path.join(ROOT, "results", "eval.csv"))
    ap.add_argument("--runs", default=os.path.join(ROOT, "results", "raw", "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    if not os.path.exists(args.eval):
        sys.exit(f"eval table not found: {args.eval}")
    s = build(args.eval, args.runs, args.out, n_boot=args.n_boot, figures=not args.no_figures)
    ov = s["primary_transfer_test"]["overall"]
    print(f"rows={s['coverage']['n_rows']}  transfer claim: {ov['label']} "
          f"({ov['n_supported']}/{ov['n_outcomes']}: {', '.join(ov['supported_outcomes']) or '-'})")
    for r in s["secondary_expectations"]["adv"].itertuples():
        print(f"  secondary {r.id:9s} {r.verdict:6s} {r.expectation}")
    v = s["validity"]
    print(f"validity: heuristic {v['heuristic_cells_pass']}/12 (gate {'pass' if v['heuristic_gate_pass'] else 'FAIL'}), "
          f"adversary reduces J in {v['adv_reduces_cells']}/12 cells, br cells covered {v['br_cells_covered']}")
    if s["figure_errors"]:
        print(f"figure errors: {s['figure_errors']}")
    print(f"wrote {args.out}/tables, {args.out}/figures, {args.out}/summary.json")


if __name__ == "__main__":
    main()
