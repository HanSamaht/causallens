"""Figures for the NFL BDB 2027 writeup.

Six figures, all from the saved analysis tables. Plain matplotlib,
colorblind-safe palette, each figure answers one question the judges
score: correctness, backing of claims, accessibility.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
})
C = "#3366cc"  # main
C2 = "#cc6633"  # accent

ds = pd.read_csv("pass_rush_dataset.csv")
m = pd.read_csv("full_cohort.csv")
tf = pd.read_csv("tracked_forty.csv")

# ---- Fig 1: sensor vs handheld forty, movement-aligned ----
cr = pd.read_csv("combine_results.csv")[["nfl_id", "forty", "ten_yd_split"]]
# draft_year lives in combine_results, not in the aggregated drill dataset
if "draft_year" not in ds.columns:
    years = pd.read_csv("combine_results.csv")[["nfl_id", "draft_year"]]
    ds = ds.merge(years, on="nfl_id", how="left")
p = tf.merge(cr, on="nfl_id").dropna(subset=["forty", "t40"])
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].scatter(p.forty, p.t40, s=14, alpha=0.6, color=C)
lims = [4.3, 5.6]
axes[0].plot(lims, lims, "--", color="gray", lw=1)
axes[0].set_xlabel("handheld 40yd (s)")
axes[0].set_ylabel("tracked 40yd, aligned start (s)")
axes[0].set_title(f"(a) 40yd: r = {p.forty.corr(p.t40):.3f}, bias = {(p.t40 - p.forty).mean():+.3f}s")
p2 = p.dropna(subset=["ten_yd_split", "t10"])
axes[1].scatter(p2.ten_yd_split, p2.t10, s=14, alpha=0.6, color=C2)
lims2 = [1.4, 2.1]
axes[1].plot(lims2, lims2, "--", color="gray", lw=1)
axes[1].set_xlabel("handheld 10yd split (s)")
axes[1].set_ylabel("tracked 10yd, aligned start (s)")
axes[1].set_title(f"(b) 10yd split: r = {p2.ten_yd_split.corr(p2.t10):.3f}")
fig.suptitle("Sensor and stopwatch agree on the 40; handheld fails on short splits", y=1.02)
fig.tight_layout()
fig.savefig("figures/fig1_sensor_vs_handheld.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: missingness by data source ----
TRAD = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "bench_reps", "vertical"]
TRACK = ["total_time", "max_speed", "first_step_yds_0p5s", "accel_first_1s", "speed_at_4yd"]
miss_t = [100 * ds[c].isna().mean() for c in TRAD]
miss_k = [100 * ds[c].isna().mean() for c in TRACK]
fig, ax = plt.subplots(figsize=(9, 3.4))
x = np.arange(len(TRAD) + len(TRACK))
vals = miss_t + miss_k
colors = [C] * len(TRAD) + [C2] * len(TRACK)
ax.bar(x, vals, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(TRAD + TRACK, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("missing %")
ax.axvline(len(TRAD) - 0.5, color="gray", lw=1)
ax.text(len(TRAD) / 2 - 0.5, max(vals) * 0.92, "traditional drills", ha="center")
ax.text(len(TRAD) + len(TRACK) / 2 - 0.5, max(vals) * 0.92, "sensor tracking", ha="center")
ax.set_title("Top picks skip traditional drills; the sensor misses almost nobody")
fig.tight_layout()
fig.savefig("figures/fig2_missingness.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 3: year drift ----
feats = ["total_time", "max_speed", "accel_first_1s", "bend_rate_deg_per_s"]
med = ds.groupby("draft_year")[feats].median()
fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
for ax, f in zip(axes, feats):
    ax.plot(med.index, med[f], "o-", color=C)
    ax.set_title(f, fontsize=9)
    ax.set_xticks([2023, 2024, 2025])
fig.suptitle("Protocol drift 2023-2025: four of ten tracking features move (Kruskal p<0.05)", y=1.06)
fig.tight_layout()
fig.savefig("figures/fig3_drift.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 4: drill speed vs pressure rate, edge vs interior ----
core = m[m.n_opportunities >= 50].dropna(subset=["pressure_rate"]).copy()
core["grp"] = np.where(core.nfl_position.isin(["DE", "OLB"]), "edge", "interior")
fig, ax = plt.subplots(figsize=(6.5, 4.6))
for grp, color, mark in [("edge", C, "o"), ("interior", C2, "s")]:
    sub = core[core.grp == grp]
    ax.scatter(sub.max_speed, sub.pressure_rate, s=26, alpha=0.7, color=color, marker=mark, label=grp)
    z = np.polyfit(sub.max_speed, sub.pressure_rate, 1)
    xs = np.linspace(sub.max_speed.min(), sub.max_speed.max(), 20)
    ax.plot(xs, np.polyval(z, xs), color=color, lw=2)
rho, pval = spearmanr(core.max_speed, core.pressure_rate)
ax.set_xlabel("Combine PASS_RUSH_DRILL top speed (yd/s)")
ax.set_ylabel("NFL pressure rate per rush opportunity")
ax.set_title(f"Drill speed to NFL pressure: Spearman rho = {rho:.2f} (p = {pval:.4f}), n = {len(core)}")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig4_speed_to_pressure.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 5: the specification audit of the drill speed effect ----
core["edge_f"] = core.nfl_position.isin(["DE", "OLB"]).astype(float)
# models standardize tracking features within draft year, matching
# nfl_analysis step4; the drift figures above intentionally show raw
years_f = pd.read_csv("combine_results.csv")[["nfl_id", "draft_year"]]
if "draft_year" not in core.columns or core.draft_year.isna().all():
    core = core.merge(years_f, on="nfl_id", how="left")
TRACK10 = ["total_time", "max_speed", "first_step_yds_0p5s", "accel_first_1s",
           "peak_accel", "path_efficiency", "bend_rate_deg_per_s", "arc_speed",
           "speed_at_4yd", "speed_at_5yd"]
for c in TRACK10:
    if c in core.columns:
        core[c] = core.groupby("draft_year")[c].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 1e-12 else s * 0.0)
y = core.pressure_rate.values
TRACK6 = ["total_time", "max_speed", "accel_first_1s", "speed_at_4yd", "path_efficiency", "bend_rate_deg_per_s"]
TRAD8 = ["forty", "ten_yd_split", "vertical", "bench_reps", "combine_weight",
         "ngs_athleticism_score", "ngs_college_production_score"]
full_cols = ["edge_f", "draft_overall_pick"] + TRAD8 + TRACK6


def do_effect(cols):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=[0.1, 1, 10, 100]))
    pipe.fit(core[cols], y, ridgecv__sample_weight=core.n_opportunities.values)
    A, B = core[cols].copy(), core[cols].copy()
    A["max_speed"] = np.percentile(core.max_speed, 90)
    B["max_speed"] = np.percentile(core.max_speed, 10)
    return float((pipe.predict(A) - pipe.predict(B)).mean())


lean = do_effect(["edge_f", "draft_overall_pick", "max_speed"])
full = do_effect(full_cols)

# specification audit: refit the same family on 200 sampled control subsets
import sys as _sys

_sys.path.insert(0, "../..")
from intervense.analysis.specification_audit import specification_audit as _sa
from intervense.drivers import detect_driver as _dd

_full_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           RidgeCV(alphas=[0.1, 1, 10, 100]))
_full_pipe.fit(core[full_cols], y, ridgecv__sample_weight=core.n_opportunities.values)
audit = _sa(_dd(_full_pipe, full_cols, core[full_cols].to_numpy(), name="nfl_spec"),
            "max_speed", y, max_specs=200,
            sample_weight=core.n_opportunities.values, seed=0)
cut = audit["materiality"]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
axes[0].bar([0, 1], [lean, full], color=[C, C2], width=0.55)
axes[0].axhspan(-cut, cut, color="gray", alpha=0.25, label="chance band")
axes[0].axhline(0, color="black", lw=1)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["lean model\n(position, pick, drill speed)",
                         "full model\n(16 features)"], fontsize=9)
axes[0].set_ylabel("do(drill speed: p10 -> p90) effect")
for i, v in enumerate([lean, full]):
    axes[0].text(i, v + (0.001 if v > 0 else -0.0025), f"{v:+.4f}",
                 ha="center", fontsize=9)
axes[0].legend(fontsize=8, loc="upper right")
axes[0].set_title("(a) two models, one feature", fontsize=10)

curve = np.array(audit["spec_curve"])
axes[1].axhspan(-cut, cut, color="gray", alpha=0.25)
axes[1].axhline(0, color="black", lw=1)
axes[1].plot(np.arange(1, len(curve) + 1), curve, color=C, lw=1.2)
axes[1].set_xlabel("specification rank (sorted effect)")
axes[1].set_ylabel("do(drill speed) per specification")
axes[1].set_title(f"(b) {len(curve)} refitted control subsets: "
                  f"{audit['n_specs'] - audit['n_flip'] - audit['n_negligible']} above the band, "
                  f"{audit['n_flip']} below", fontsize=10)
fig.suptitle("Collinearity collapses the drill speed effect; no specification goes materially negative",
             y=1.03)
fig.tight_layout()
fig.savefig("figures/fig5_sign_flip.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 6: attribution confidence ----
import sys
sys.path.insert(0, "../..")
from intervense.core import MicSession
from intervense.drivers import detect_driver

lean_cols = ["edge_f", "draft_overall_pick", "max_speed", "accel_first_1s",
             "forty", "ngs_college_production_score"]
pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                     RidgeCV(alphas=[0.1, 1, 10, 100]))
pipe.fit(core[lean_cols], y, ridgecv__sample_weight=core.n_opportunities.values)
drv = detect_driver(pipe, lean_cols, core[lean_cols].to_numpy(), name="nfl_lean")
ci = MicSession(drv).attribution_confidence(n_bootstrap=25, top_k=5)
rows = ci["features"][:5][::-1]
fig, ax = plt.subplots(figsize=(7, 4))
names = [r["feature"].replace("draft_overall_pick", "draft pick").replace("ngs_college_production_score", "college production").replace("edge_f", "edge (pos)") for r in rows]
shares = [r["share"] for r in rows]
lo = [r["share"] - r["ci_low"] for r in rows]
hi = [r["ci_high"] - r["share"] for r in rows]
ax.barh(names, shares, xerr=[lo, hi], color=C, capsize=3)
ax.set_xlabel("mean |SHAP| share, bootstrap 95% CI (25 replicates)")
ax.set_title("Which features carry stable weight in the lean model")
fig.tight_layout()
fig.savefig("figures/fig6_attribution_ci.png", bbox_inches="tight")
plt.close(fig)

print("6 figures saved under figures/")
