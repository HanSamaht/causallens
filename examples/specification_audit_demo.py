"""Specification audit demo: synthetic theory check and the NFL replay.

Part 1 plants a suppressor structure where the lean sign is wrong and
the closed form must agree with textbook algebra.

Part 2 replays the NFL Big Data Bowl 2027 finding: the 16-feature ridge
model hid the drill-speed effect that the lean model showed. The audit
now names the minimal control sets responsible for the flip.

The NFL data files are local and gitignored; point KAGGLE_NFL_DIR at
the directory holding full_cohort.csv (default examples/nfl).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from intervense.analysis.specification_audit import specification_audit
from intervense.drivers import detect_driver

TRACK = ["total_time", "max_speed", "first_step_yds_0p5s", "accel_first_1s",
         "peak_accel", "path_efficiency", "bend_rate_deg_per_s", "arc_speed",
         "speed_at_4yd", "speed_at_5yd"]
TRAD = ["forty", "ten_yd_split", "vertical", "broad_jump", "bench_reps",
        "combine_weight", "ngs_athleticism_score", "ngs_college_production_score"]


def part1_synthetic():
    print("=" * 64)
    print("Part 1: planted suppressor, closed form vs textbook algebra")
    print("=" * 64)
    rng = np.random.default_rng(21)
    n = 800
    c = rng.normal(size=n)
    x = 0.8 * c + rng.normal(scale=0.6, size=n)
    z = 0.8 * c + rng.normal(scale=0.6, size=n)
    n1 = rng.normal(size=n)
    n2 = rng.normal(size=n)
    y = 0.3 * x - 1.0 * z + 0.2 * n1 + rng.normal(scale=0.3, size=n)
    X = np.column_stack([x, z, n1, n2])
    model = LinearRegression().fit(X, y)
    drv = detect_driver(model, ["x", "z", "n1", "n2"], X, name="synthetic")

    out = specification_audit(drv, "x", y)
    # analytic check: cov(x, y)/var(x) should match the lean effect scaled
    # by the contrast width, up to sampling noise
    rho = np.corrcoef(x, z)[0, 1]
    beta_lean_theory = 0.3 - 1.0 * rho
    width = out["contrast"]["to_value"] - out["contrast"]["from_value"]
    theory = beta_lean_theory * width

    print(f"verdict        : {out['verdict']}")
    print(f"engine         : {out['engine']} ({out['n_specs']} specs, "
          f"{out['lattice']})")
    print(f"lean effect    : {out['effect_lean']:+.4f} "
          f"(theory {theory:+.4f})")
    print(f"full effect    : {out['effect_full']:+.4f}")
    print(f"minimal flip   : {[h['controls'] for h in out['minimal_flip_sets']]}")
    print(f"gap diagnostic : {out['attribution_gap']}")


def part2_nfl():
    data_dir = os.environ.get("KAGGLE_NFL_DIR",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "nfl"))
    cohort_path = os.path.join(data_dir, "full_cohort.csv")
    if not os.path.exists(cohort_path):
        print(f"\nPart 2 skipped: {cohort_path} not found "
              "(run examples/nfl/nfl_analysis.py steps 1 to 3 first)")
        return

    print()
    print("=" * 64)
    print("Part 2: NFL replay, drill top speed under specification search")
    print("=" * 64)
    m = pd.read_csv(cohort_path)
    # match the production protocol: tracking features standardized
    # within draft year (protocol drift 2023-2025)
    years = pd.read_csv(os.path.join(data_dir, "combine_results.csv"))[["nfl_id", "draft_year"]]
    m = m.merge(years, on="nfl_id", how="left")
    for c in TRACK:
        m[c] = m.groupby("draft_year")[c].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 1e-12 else s * 0.0)
    core = m[m.n_opportunities >= 50].copy()
    core["edge"] = core.nfl_position.isin(["DE", "OLB"]).astype(float)
    core = core.dropna(subset=["pressure_rate"])
    y = core.pressure_rate.values
    cols = ["edge", "draft_overall_pick"] + TRAD + TRACK[:6]

    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=[0.1, 1, 10, 100]))
    pipe.fit(core[cols], y, ridgecv__sample_weight=core.n_opportunities.values)
    drv = detect_driver(pipe, cols, core[cols].to_numpy(), name="nfl_replay")

    out = specification_audit(drv, "max_speed", y, max_specs=200,
                              sample_weight=core.n_opportunities.values,
                              seed=0)
    print(f"cohort         : n={len(core)} players, "
          f"{len(cols)} candidate features")
    print(f"verdict        : {out['verdict']}")
    print(f"engine         : {out['engine']} ({out['n_specs']} specs, "
          f"{out['lattice']})")
    print(f"lean effect    : {out['effect_lean']:+.4f}")
    print(f"full effect    : {out['effect_full']:+.4f}")
    print(f"flip fraction  : {out['frac_flip']:.0%} "
          f"(ci95 {out.get('flip_ci95')})")
    print(f"minimal flips  :")
    for h in out["minimal_flip_sets"]:
        print(f"                 {h['controls']} -> {h['effect']:+.4f}")
    print(f"gap diagnostic : {out['attribution_gap']}")


if __name__ == "__main__":
    part1_synthetic()
    part2_nfl()
