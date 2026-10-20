"""Audit hardening demo: fold stability, permutation nulls, data forensics.

Part 1: specification audit with cohort groups. An effect that exists
in the pool but reverses inside one cohort is named by the fold block.

Part 2: shortcut diff with permutation nulls. The reliance share of a
label echo is tested against refits on shuffled outcomes.

Part 3: data forensics on the real NFL cohort when the files are
present: selection missingness, drill drift across draft years, join
coverage between drill features and game pressure.

Run: python examples/audit_hardening_demo.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from intervense.analysis.data_forensics import data_forensics
from intervense.analysis.shortcut_diff import shortcut_diff
from intervense.analysis.specification_audit import specification_audit
from intervense.drivers import detect_driver


def part1_fold_stability():
    print("=" * 64)
    print("Part 1: specification audit with cohort fold stability")
    print("=" * 64)
    rng = np.random.default_rng(9)
    n = 600
    x = rng.normal(size=n)
    z = 0.5 * x + rng.normal(scale=0.5, size=n)
    cohorts = np.array(["2023"] * 200 + ["2024"] * 200 + ["2025"] * 200)
    # the effect reverses in the 2025 cohort, the pool cancels over it
    y = np.where(cohorts == "2025", -2.0 * x, x) + 0.2 * rng.normal(size=n)
    X = np.column_stack([x, z])
    model = LinearRegression().fit(X, y)
    drv = detect_driver(model, ["x", "z"], X, name="folds")

    out = specification_audit(drv, "x", y, groups=cohorts)
    print("verdict :", out["verdict"])
    for f in out["fold_stability"]["folds"]:
        print(f"  cohort {f['group']}: lean effect {f['effect_lean']:+.4f} "
              f"(n={f['n_rows']})")


def part2_permutation_null():
    print()
    print("=" * 64)
    print("Part 2: shortcut diff with permutation nulls")
    print("=" * 64)
    rng = np.random.default_rng(7)
    n = 800
    c = rng.normal(size=n)
    x = 0.8 * c + rng.normal(scale=0.6, size=n)
    y = 0.5 * x + rng.normal(scale=0.5, size=n)
    p = y + rng.normal(scale=0.3, size=n)  # label echo
    noise = rng.normal(size=n)
    X = np.column_stack([x, p, noise])
    gbm = GradientBoostingRegressor(n_estimators=60, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, ["x", "p", "noise"], X, name="perm")

    out = shortcut_diff(drv, y, edges=[("x", "outcome"), ("outcome", "p")],
                        n_perm=20)
    print("verdict :", out["verdict"])
    for r in out["reliance"]:
        extra = (f" null median {r['null_share_median']:.0%},"
                 f" perm p={r['perm_p']:.2f}") if "perm_p" in r else ""
        print(f"  {r['feature']:6s} share {r['reliance_share']:.1%}  "
              f"{r['classification']}{extra}")


def part3_nfl_forensics():
    nfl = os.environ.get(
        "KAGGLE_NFL_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl"))
    ds_path = os.path.join(nfl, "pass_rush_dataset.csv")
    gp_path = os.path.join(nfl, "game_pressure.csv")
    if not (os.path.exists(ds_path) and os.path.exists(gp_path)):
        print()
        print("Part 3 skipped: NFL files not found under", nfl)
        return

    print()
    print("=" * 64)
    print("Part 3: data forensics on the real NFL cohort")
    print("=" * 64)
    ds = pd.read_csv(ds_path)
    gp = pd.read_csv(gp_path)
    year_cols = [c for c in ds.columns if c.startswith("draft_year")]
    if year_cols and "draft_year" not in ds.columns:
        ds["draft_year"] = ds[year_cols[0]]
        ds = ds.drop(columns=year_cols)
    if "draft_year" not in ds.columns:
        years = pd.read_csv(os.path.join(nfl, "combine_results.csv"))[["nfl_id", "draft_year"]]
        ds = ds.merge(years, on="nfl_id", how="left")
    out = data_forensics(ds, groups="draft_year", other_df=gp, key="nfl_id")
    print(out["verdict"])
    print()
    print("MAR suspects:", out["mechanism"]["mar_suspects"][:3])
    print("drift flags :", [(d["column"], d["p_value"]) for d in out["drift"][:5]])
    j = out["join_audit"]
    print("join        :", f"left {j['left_keys_matched']:.0%} / "
          f"right {j['right_keys_matched']:.0%} matched on nfl_id")


if __name__ == "__main__":
    part1_fold_stability()
    part2_permutation_null()
    part3_nfl_forensics()
