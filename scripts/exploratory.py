"""Exploratory analyses that were NOT pre-registered. Run after make_report.py.

Writes results/tables/exploratory_*.csv and results/exploratory.json:

1. learned default swarm vs scripted heuristic under the adversarial score (the pre-registered
   gate uses nature hazards only);
2. a "matched mean" transfer predictor: the combined planet's outcome at (mission, tempo) predicted
   by the mean of the three single-hazard planets at the same (mission, tempo), next to the
   pre-registered models;
3. the budget check: default swarms trained 3x longer vs the scripted heuristic (if evaluated);
4. which arms are the same policy as `default` (structural zeros in the outcomes).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")
CELL = ["planet", "mission"]
OUTCOMES = ["V_comm", "V_gossip", "V_couple", "V_sac", "beta", "loss_default"]
SINGLE = ["corrosion", "jam", "quake"]


def heuristic_vs_learned():
    sc = pd.read_csv(os.path.join(TAB, "scores.csv"))
    sc = sc[(sc["remove_t"] == -1) & sc["arm"].isin(["default", "heuristic"])]
    rows = []
    for which in ["nature", "adv"]:
        j = sc.groupby(CELL + ["arm"])[f"J_{which}"].mean().unstack("arm")
        g = sc[(sc["n"] == 6) & np.isclose(sc["tempo"], 0.3)].groupby(CELL + ["arm"])[f"J_{which}"].mean().unstack("arm")
        t = pd.DataFrame({"score": which, "J_default_all": j["default"], "J_heuristic_all": j["heuristic"],
                          "J_default_n6_t03": g["default"], "J_heuristic_n6_t03": g["heuristic"]})
        rows.append(t.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["learned_ge_heuristic_all"] = out["J_default_all"] >= out["J_heuristic_all"]
    return out


def matched_mean_transfer():
    o = pd.read_csv(os.path.join(TAB, "outcomes_adv.csv"))
    tm = pd.read_csv(os.path.join(TAB, "transfer_metrics_adv.csv"))
    rows = []
    for y in OUTCOMES:
        single = o[o["planet"].isin(SINGLE)].groupby(["mission", "tempo"])[y].mean().rename("pred")
        comb = o[o["planet"] == "combined"].set_index(["mission", "tempo"])[y].rename("obs")
        d = pd.concat([single, comb], axis=1).dropna()
        r = {"outcome": y, "n": len(d), "mae_matched_mean": float((d["pred"] - d["obs"]).abs().mean()),
             "rho_matched_mean": float(spearmanr(d["pred"], d["obs"])[0])}
        for m in ["P", "B0", "B1", "B2"]:
            sel = tm[(tm["outcome"] == y) & (tm["split"] == "headline") & (tm["model"] == m)]
            r[f"mae_{m}"] = float(sel["mae"].iloc[0])
            r[f"rho_{m}"] = float(sel["spearman"].iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)


def extended():
    path = os.path.join(ROOT, "results", "eval_extended.csv")
    if not os.path.exists(path):
        return None
    ext = pd.read_csv(path)
    ext = ext[(ext["mode"] == "nature") & (ext["remove_t"] == -1)]
    base = pd.read_csv(os.path.join(ROOT, "results", "eval.csv"))
    base = base[(base["mode"] == "nature") & (base["remove_t"] == -1) & (base["arm"] == "default") & (base["seed"] == 0)]
    rows = []
    for (planet, mission), e in ext.groupby(CELL):
        b = base[(base["planet"] == planet) & (base["mission"] == mission)]
        f = lambda d, arm: d[d["arm"] == arm]
        n6 = lambda d: d[(d["n"] == 6) & np.isclose(d["tempo"], 0.3)]["J"].mean()
        rows.append({"planet": planet, "mission": mission,
                     "J_150it_n6_t03": n6(b), "J_450it_n6_t03": n6(f(e, "default")),
                     "J_heuristic_n6_t03": n6(f(e, "heuristic")),
                     "J_150it_all": b["J"].mean(), "J_450it_all": f(e, "default")["J"].mean(),
                     "J_heuristic_all": f(e, "heuristic")["J"].mean()})
    out = pd.DataFrame(rows)
    out["gap_150it_all"] = out["J_150it_all"] - out["J_heuristic_all"]
    out["gap_450it_all"] = out["J_450it_all"] - out["J_heuristic_all"]
    return out


def identical_arms():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "runs", "*__default__s*.npz"))):
        a = np.load(f)
        for arm in ["silent", "blackboard", "solo", "protected"]:
            b = np.load(f.replace("__default__", f"__{arm}__"))
            if all(np.array_equal(a[k], b[k]) for k in a.files):
                planet, mission, _, seed = os.path.basename(f)[:-4].split("__")
                rows.append({"planet": planet, "mission": mission, "seed": int(seed[1:]), "arm_identical_to_default": arm})
    return pd.DataFrame(rows)


def main():
    res = {}
    for name, fn in [("heuristic_vs_learned", heuristic_vs_learned), ("matched_mean_transfer", matched_mean_transfer),
                     ("extended_training", extended), ("identical_arms", identical_arms)]:
        df = fn()
        if df is None:
            continue
        df.to_csv(os.path.join(TAB, f"exploratory_{name}.csv"), index=False)
        res[name] = json.loads(df.to_json(orient="records"))
        print("==", name)
        print(df.round(3).to_string(index=False))
    with open(os.path.join(ROOT, "results", "exploratory.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
