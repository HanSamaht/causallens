"""Build the public Kaggle notebook nfl_bdb2027.ipynb.

Embeds the pipeline scripts verbatim as appendix cells (%%writefile),
so the notebook runs on Kaggle from the competition attachment alone
plus intervense installed from the repository tag.

Usage: python build_notebook.py [tag]   (default v0.12.0)
"""
from __future__ import annotations

import sys

import nbformat as nbf

TAG = sys.argv[1] if len(sys.argv) > 1 else "v0.12.0"

SRC = {
    "extract": "extract_pass_rush_features.py",
    "game": "extract_game_pressure.py",
    "analysis": "nfl_analysis.py",
    "figures": "make_figures.py",
    "trajectories": "extract_game_trajectories.py",
}


def src(key: str) -> str:
    with open(SRC[key], encoding="utf-8") as fh:
        text = fh.read()
    return text.rstrip("\n")


nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(f"""# Stopwatch Against Sensor: What Combine Tracking Really Adds to Pass Rush Evaluation

NFL Big Data Bowl 2027. This notebook is the code companion to the writeup: every
number quoted there is reproduced here from the competition attachment alone.

**Requirement 1.** Link Combine sensor tracking (PASS_RUSH_DRILL and
FORTY_YARD_DASH movement) to regular-season NFL performance: pressure rate per
pass-rush opportunity and career snap volume, for the 118 front-seven prospects
of the 2023 to 2025 cohorts.

**Headline findings.**

1. Movement-aligned sensor forty vs handheld stopwatch: r = 0.969, bias
   -0.025s. Handheld 10-yard splits only reach r = 0.718. Short-area timing
   belongs to the sensor, not the stopwatch.
2. Traditional drills are 20 to 64 percent missing (high picks skip them);
   tracking features are 0 to 4 percent missing. A traditional-only analysis
   is conditioned on a sample selected on draft capital.
3. PASS_RUSH_DRILL top speed predicts NFL pressure rate: Spearman 0.528 raw,
   +0.240 partial controlling position group and draft pick (within-year
   standardized tracking features). The backdoor adjusted intervention
   estimate on the lean model is +0.026 pressure rate for moving a
   prospect from the 10th to the 90th within-cohort percentile.
4. A 16-feature ridge regression collapses the drill-speed intervention
   effect from +0.026 to essentially zero (-0.0002), inside the chance
   band for this sample, while its importance ranking puts drill speed
   dead last. A specification audit over 200 refitted control subsets
   confirms the effect is never materially negative: the honest
   description is collapse, not flip. Importance is bookkeeping, not
   behavior.

The causal machinery (interventions, backdoor identification, bootstrap
attribution intervals) comes from
[Intervense](https://github.com/HanSamaht/intervense), the open-source causal
interrogation framework this analysis was built with.

**How to run.** Attach the competition data, switch Internet on in the notebook
settings, then Run All. Step 3 scans the 147MB player_play file and takes a few
minutes. Sections marked Appendix carry the full extraction code.
""")

code(f"""import importlib.util, os, subprocess, sys

# Kaggle: install from the release tag. Inside the repository: skip, the
# analysis cells add the repository root to sys.path themselves.
local_repo = os.path.exists("../../pyproject.toml")
if importlib.util.find_spec("intervense") is None and not local_repo:
    # the full dependency set pulls optional extras (dowhy, tigramite,
    # dice-ml) this notebook never imports and that can fail to build
    # on some images; install the package without deps plus shap
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "shap", "dowhy"],
        check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "git+https://github.com/HanSamaht/intervense.git@{TAG}"],
        check=True)
try:
    import intervense
    print("intervense", intervense.__version__)
except ImportError:
    print("intervense resolves from the local repository in later cells")
""")

code("""import glob, os

# discover the competition files wherever the platform mounted them:
# Kaggle nests or prefixes the filenames depending on dataset packaging
needed = ["combine_tracking.csv", "combine_results.csv", "players.csv",
          "player_career_successes.csv", "player_play.csv",
          "game_tracking_2023.csv", "game_tracking_2024.csv",
          "game_tracking_2025.csv"]
found = 0
for name in needed:
    if os.path.exists(name):
        found += 1
        continue
    hit = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hit:
        hit = [p for p in glob.glob(f"/kaggle/input/**/*{name}", recursive=True)]
    if hit:
        os.symlink(hit[0], name)
        found += 1
print(f"linked {found}/{len(needed)} competition files")
if found == 0 and os.path.isdir("examples/nfl"):
    os.chdir("examples/nfl")  # running from the repository root
print("working dir:", os.getcwd())
""")

md("""## Pipeline overview

Four steps, each with its evidence rule stated up front:

| Step | Question | Discipline |
|---|---|---|
| 1 | What did the sensor measure in the drills? | Phases anchored on the host's published drill geometry, never guessed |
| 2 | Does the sensor agree with the stopwatch? | Compare only where both exist; align the sensor clock to real movement first |
| 3 | Who produced pressure in games? | The host's NA semantics: a missing quick_pressure with a blitz flag is an observed non-event |
| 4 | Does drill movement predict NFL pressure? | Partial correlation, backdoor adjustment, explicit interventions; importances never quoted as effects |

The three appendix cells below write the pipeline source files verbatim, so the
notebook is self-contained on Kaggle.""")

md("""### Appendix A: drill features from published geometry

The host documented the PASS_RUSH_DRILL layout: dummies 4 yards from center,
finish cone 7 yards deep, timing from snap to cone, LEFT attempts use the rip
move and RIGHT attempts the swim move. Phase segmentation uses those anchors.
Each attempt is rotated into the player's travel frame, then reduced to ten
features: total time, top speed, speed at 4 and 5 yards of depth, first-step
displacement at 0.5s, mean acceleration over the first second, peak
acceleration, path efficiency (net displacement over path length, recomputed
from frame positions because the dis column changes units across years), bend
rate, and arc speed.""")

code(f"""%%writefile {SRC['extract']}
{src("extract")}""")

md("""### Appendix B: in-game pressure with the host's NA semantics

Pressure events are the union of time_to_pressure, time_to_qb_hurry and sacks,
which also absorbs the documented hurry-without-pressure anomalies.
Opportunity definition: defensive linemen are credited on every non-nullified
pass play; linebackers only when blitzing, since a coverage linebacker did not
rush. quick_pressure missing while blitzing is an observed non-event and stays
in the denominator; player_get_off missing (never crossed the line) leaves the
get-off average but not the denominator. The BALL carrier rows carry an nfl_id
and are removed via entity_type.""")

code(f"""%%writefile {SRC['game']}
{src("game")}""")

md("""### Appendix C: the four-step driver

Step 1 merges drill aggregates with traditional results and career outcomes.
Step 2 aligns the tracked forty to genuine movement (speed above 2 yd/s) before
comparing to the handheld clock. Step 3 builds per-player pressure rates. Step
4 is the causal core: partial correlation with position and draft capital
controls, lean vs full model interventions, bootstrap attribution intervals,
and backdoor identification of the drill-speed effect.""")

code(f"""%%writefile {SRC['analysis']}
{src("analysis")}""")

md("""## Steps 1 and 2: drill features and the stopwatch comparison

The raw recording start inflates the tracked forty by about 0.4s, because
recording begins before the athlete moves. Aligning the clock to genuine
motion (speed above 2 yd/s) closes the bias to -0.025s and lifts the
correlation to 0.969. The 10-yard split is where the handheld timer fails:
human timers are precise over forty yards and noisy over ten.""")

code("""import nfl_analysis as na

na.step1_drill_features()
na.step2_sensor_vs_handheld()""")

md("""## Step 3: in-game pressure

Per-player opportunity counts and pressure rates under the NA semantics above.
Players with at least 50 pass-rush opportunities (median 199 in cohort) form
the analysis core; the rest are excluded for rate stability, not dropped
silently.""")

code("""na.step3_game_pressure()""")

md("""## Step 4: the main finding and the sign-flip trap

Four checks run in sequence:

1. Partial correlation of drill top speed with pressure rate, controlling
   position group and draft pick: the confounders absorb most of the raw 0.528
   but the residue stays positive.
2. do(max_speed p10 to p90) on the lean three-feature model vs the full
   sixteen-feature model: collinearity collapses the full model's effect
   from +0.026 into the chance band at essentially zero.
3. Mediation chain accel to get_off to pressure: two strong links, but
   the total effect is too small for a stable mediated share.
4. Bootstrap attribution intervals on the lean model: which features carry
   stable weight across resamples.
5. Backdoor identification of the drill-speed effect with the adjustment set
   {position, draft pick}: the estimate a team can act on.""")

code("""na.step4_analysis()""")

md("""## Step 5: specification audit of the drill speed effect

The lean vs full contrast above is two points. The specification audit
refits the same ridge family on 200 sampled control subsets (weighted by
opportunity count, matching the production protocol) and draws the full
curve: in which specifications the effect rises above the chance band,
and whether any specification turns it materially negative. This is the
Intervense method the writeup leads with.""")

code("""import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from intervense.analysis.specification_audit import specification_audit
from intervense.drivers import detect_driver

m = pd.read_csv("full_cohort.csv")
core = m[m.n_opportunities >= 50].copy()
core["edge"] = core.nfl_position.isin(["DE", "OLB"]).astype(float)
core = core.dropna(subset=["pressure_rate"])
# tracking features are standardized within draft year, matching step 4
years = pd.read_csv("combine_results.csv")[["nfl_id", "draft_year"]]
core = core.merge(years, on="nfl_id", how="left")
for c in na.TRACK:
    core[c] = core.groupby("draft_year")[c].transform(
        lambda s: (s - s.mean()) / s.std() if s.std() > 1e-12 else s * 0.0)
y = core.pressure_rate.values
cols = ["edge", "draft_overall_pick"] + na.TRAD + na.TRACK[:6]

pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                     RidgeCV(alphas=[0.1, 1, 10, 100]))
pipe.fit(core[cols], y, ridgecv__sample_weight=core.n_opportunities.values)
drv = detect_driver(pipe, cols, core[cols].to_numpy(), name="nfl_spec_audit")

out = specification_audit(drv, "max_speed", y, max_specs=200,
                          sample_weight=core.n_opportunities.values, seed=0)
print(out["verdict"])
print("lean (speed only):", out["effect_lean"], " full (16 features):", out["effect_full"])
print("minimal flip sets:", out["minimal_flip_sets"])""")

md("""## Step 6: the 2GB game tracking, one rep at a time

The game tracking files cover the Combine cohort only (156 tracked
players in 2023, median one tracked player per play), so passers and
blockers are absent and all mechanics are computed in the rusher's own
travel frame. The extractor below anchors each rep on ball_snap,
censors the window at pass release plus 1.5 seconds so post-throw
jogging cannot leak into the features, and produces one row per pass
rush opportunity: get-off, speed checkpoints, path geometry,
travel-frame bend, and the release timestamp as pocket time.""")

code(f"""%%writefile {SRC['trajectories']}
{src("trajectories")}""")

code("""import extract_game_trajectories as egt
egt.main()""")

code(f"""import numpy as np
import pandas as pd

tp = pd.read_csv("game_trajectories_players.csv")
pdf = pd.read_csv("game_trajectories_plays.csv")
print(f"per-rep rows: {{len(pdf):,}} across {{pdf.nfl_id.nunique()}} rushers, "
      f"pressure rate {{pdf.pressure_event.mean():.1%}}")

feats = ["get_off_time", "speed_at_1s", "top_speed_window",
         "path_efficiency", "heading_change_rate", "bend_travel"]
d = pdf.dropna(subset=["get_off_time", "speed_at_1s"]).copy()
d = d[d.pocket_time > 0.5]
for c in ["pocket_time"] + feats:
    d[c] = d[c].fillna(d[c].median())

from intervense.analysis.panel_fixed_effects import panel_fixed_effects
fe_out = panel_fixed_effects(d[["nfl_id", "pressure_event", "pocket_time"]
                               + feats],
                             outcome="pressure_event",
                             features=["pocket_time"] + feats,
                             group="nfl_id")
print("within-player fixed effects (Intervense panel_fixed_effects):")
for r in fe_out["features"][1:]:  # skip pocket_time control
    star = " ***" if r["p_value"] < 0.001 else " *" if r["p_value"] < 0.05 else ""
    print(f"  {{r['feature']:22s}} {{r['coef_within_per_sd']:+.4f}} "
          f"(se {{r['se_cluster']:.4f}}, p {{r['p_value']:.4f}}){{star}}")

# mediation: framework call replaces 30 lines of manual bootstrap
ds = pd.read_csv("pass_rush_dataset.csv")
gp = pd.read_csv("game_pressure.csv")[["nfl_id", "pressure_rate"]]
years = pd.read_csv("combine_results.csv")[["nfl_id", "draft_year"]]
mm = ds.merge(gp, on="nfl_id").merge(tp, on="nfl_id").merge(years, on="nfl_id", how="left")
mm = mm[mm.n_plays >= 50]
mm["max_speed"] = mm.groupby("draft_year").max_speed.transform(
    lambda s: (s - s.mean()) / s.std() if s.std() > 1e-12 else s * 0)

from intervense.analysis.mediation_chain import mediation_chain
med_df = mm[["max_speed", "speed_at_1s_med", "pressure_rate"]].dropna()
med_out = mediation_chain(med_df, "max_speed", "speed_at_1s_med", "pressure_rate",
                          n_boot=2000, seed=0)
print(f"\\nmediation (Intervense mediation_chain):")
print(f"  {{med_out['verdict'][:120]}}")

# named players: speed transfer leaders and laggards
p_players = pd.read_csv("players.csv")[["nfl_id", "display_name"]]
m_named = mm.merge(p_players, on="nfl_id", how="left")
m_named["game_z"] = ((m_named.speed_at_1s_med - m_named.speed_at_1s_med.mean())
                     / m_named.speed_at_1s_med.std())
m_named["transfer"] = m_named.game_z - m_named.max_speed
hi_drill = m_named[m_named.max_speed > 0.5].dropna(
    subset=["display_name", "transfer"]).sort_values("transfer", ascending=False)
print()
print("speed transfer leaders (drill speed becomes game speed):")
for _, r in hi_drill.head(5).iterrows():
    print(f"  {{str(r.display_name):25s}} {{r.nfl_position:4s}} "
          f"drill {{r.max_speed:+.2f}} game {{r.game_z:+.2f}} "
          f"pressure {{r.pressure_rate:.1%}}")
print("speed transfer laggards (fast in drills, not in games):")
for _, r in hi_drill.tail(5).sort_values("transfer").iterrows():
    print(f"  {{str(r.display_name):25s}} {{r.nfl_position:4s}} "
          f"drill {{r.max_speed:+.2f}} game {{r.game_z:+.2f}} "
          f"pressure {{r.pressure_rate:.1%}}")

# mechanics archetype matrix: speed transfer x efficiency transfer
m_named["game_eff_z"] = ((m_named.path_efficiency_med - m_named.path_efficiency_med.mean())
                          / m_named.path_efficiency_med.std())
m_named["path_eff_z"] = m_named.groupby("draft_year").path_efficiency.transform(
    lambda s: (s - s.mean()) / s.std() if s.std() > 1e-12 else s * 0)
m_named["eff_transfer"] = m_named.game_eff_z - m_named.path_eff_z
conds = [
    (m_named.transfer > 0) & (m_named.eff_transfer > 0),
    (m_named.transfer > 0) & (m_named.eff_transfer <= 0),
    (m_named.transfer <= 0) & (m_named.eff_transfer > 0),
    (m_named.transfer <= 0) & (m_named.eff_transfer <= 0),
]
arch = ["Complete Transcoder", "Speed-Only", "Technique Salvage", "System Mismatch"]
m_named["archetype"] = np.select(conds, arch, default="Unknown")
print()
print("=== mechanics archetype matrix (speed_T x efficiency_T) ===")
for label in arch:
    sub = m_named[m_named.archetype == label]
    print(f"  {{label:22s}} n={{len(sub):3d}}  avg pressure {{sub.pressure_rate.mean():.1%}}")
hi_arch = m_named[m_named.max_speed > 0].nlargest(3, "pressure_rate")
for _, r in hi_arch.iterrows():
    print(f"    {{str(r.display_name):25s}} {{r.nfl_position:4s}} "
          f"archetype {{r.archetype}}  pressure {{r.pressure_rate:.1%}}")

# CATE: honest null from the framework
from sklearn.ensemble import GradientBoostingRegressor
from intervense.analysis.cate_heterogeneity import cate_heterogeneity as _cate
mm["edge"] = mm.nfl_position.isin(["DE", "OLB"]).astype(float)
d_c = mm[["pressure_rate", "edge", "draft_overall_pick", "max_speed"]].dropna()
gbm_c = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0).fit(
    d_c[["edge", "draft_overall_pick", "max_speed"]], d_c.pressure_rate)
drv_c = detect_driver(gbm_c, ["edge", "draft_overall_pick", "max_speed"],
                      d_c[["edge", "draft_overall_pick", "max_speed"]].to_numpy(),
                      name="cate")
cat_out = _cate(drv_c, "max_speed", d_c.pressure_rate, n_bootstrap=15, seed=0)
print()
print(f"CATE heterogeneity: {{cat_out['verdict'][:130]}}")

# THE ONE-SECOND TEST
from scipy.stats import spearmanr
by_season = pdf.groupby(["nfl_id", "season"]).agg(
    n=("play_id", "count"), pr=("pressure_event", "mean"),
    speed=("speed_at_1s", "median")).reset_index()
multi_s = by_season[by_season.n >= 30]
print()
print("=== THE ONE-SECOND TEST ===")
print("year-to-year stability (2024 -> 2025):")
for metric, label in [("speed", "speed@1s"), ("pr", "pressure rate")]:
    a = multi_s[multi_s.season == 2024].set_index("nfl_id")[metric]
    b = multi_s[multi_s.season == 2025].set_index("nfl_id")[metric]
    common = a.index.intersection(b.index)
    if len(common) >= 10:
        r, p = spearmanr(a[common], b[common])
        print(f"  {{label:18s}} r = {{r:+.3f}} (n={{len(common)}})")

m_test = mm.copy()
m_test["edge"] = m_test.nfl_position.isin(["DE", "OLB"]).astype(float)
def partial_r(df, x, y, controls):
    dd = df[[x, y] + controls].dropna()
    Z = np.column_stack([dd[controls].values, np.ones(len(dd))])
    resid = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return np.corrcoef(resid(dd[x].values), resid(dd[y].values))[0, 1]
print()
print("predictive power (partial r with pressure, ctrl edge+pick):")
for label, col in [("speed@1s (game)", "speed_at_1s_med"),
                    ("forty (stopwatch)", "forty"),
                    ("max_speed (drill)", "max_speed"),
                    ("path_eff (game)", "path_efficiency_med")]:
    if col in m_test.columns:
        r = partial_r(m_test, col, "pressure_rate", ["edge", "draft_overall_pick"])
        print(f"  {{label:22s}} r = {{r:+.3f}}")

# game FE: two-way fixed effects
print()
print("game fixed effects (absorbing OL, QB, game script):")
import statsmodels.api as sm
all_cols_fe = ["pressure_event", "pocket_time"] + feats
d_fe = d[["nfl_id", "game_id"] + all_cols_fe].copy()
for c in ["pocket_time"] + feats:
    d_fe[c] = (d_fe[c] - d_fe[c].mean()) / d_fe[c].std()
for c in all_cols_fe:
    d_fe[c + "_p"] = d_fe[c] - d_fe.groupby("nfl_id")[c].transform("mean")
for c in all_cols_fe:
    d_fe[c + "_pg"] = d_fe[c + "_p"] - d_fe.groupby("game_id")[c + "_p"].transform("mean")
y_fe = d_fe["pressure_event_pg"].to_numpy()
X_fe = d_fe[[c + "_pg" for c in ["pocket_time"] + feats]].to_numpy()
m_fe = sm.OLS(y_fe, X_fe).fit(cov_type="cluster",
                               cov_kwds={{"groups": d_fe.nfl_id}})
p_arr = np.asarray(m_fe.pvalues)
b_arr = np.asarray(m_fe.params)
for i, f in enumerate(feats):
    survives = "SURVIVES" if p_arr[i + 1] < 0.05 else "drops"
    print(f"  {{f:22s}} {{b_arr[i+1]:+.4f}} p={{p_arr[i+1]:.4f}} {{survives}}")
print(f"  (n={{len(d_fe):,}}, {{d_fe.nfl_id.nunique()}} players, {{d_fe.game_id.nunique()}} games)")

# holdout: framework call replaces the manual LODO loop
from intervense.analysis.holdout_evaluation import holdout_evaluation
d_hold = d[["season", "pressure_event", "pocket_time"] + feats].copy()
for c in ["pocket_time"] + feats:
    d_hold[c] = (d_hold[c] - d_hold[c].mean()) / d_hold[c].std()
h1 = holdout_evaluation(d_hold, "pressure_rate" if False else "pressure_event",
                        ["pocket_time"], group="season", holdout_groups=[2025])
print(f"\\nholdout (Intervense holdout_evaluation):")
print(f"  pocket only AUC = {{h1['auc']:.3f}}")
h2 = holdout_evaluation(d_hold, "pressure_event",
                        ["pocket_time"] + feats, group="season",
                        holdout_groups=[2025])
print(f"  + mechanics  AUC = {{h2['auc']:.3f}} ({{h2['n_test']:,}} reps, 2025 season)")""")

md("""## Figures

The nine figures of the writeup, regenerated here. Fig 6 calls Intervense for
the bootstrap attribution intervals; figs 7 to 9 come from the game tracking
mining.""")

code(f"""%%writefile {SRC['figures']}
{src("figures")}""")

code("""import os
os.makedirs("figures", exist_ok=True)
%run make_figures.py""")

code("""%%writefile make_mining_figures.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from intervense.analysis.panel_fixed_effects import panel_fixed_effects

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})
NAVY = "#1a365d"
CORAL = "#e8532f"
SLATE = "#64748b"
GOLD = "#d69e2e"
LIGHT = "#e2e8f0"

# ============ fig 7: travel-frame trajectory plot ============
pdf = pd.read_csv("game_trajectories_plays.csv")
sample = pdf[pdf.season == 2025].sample(160, random_state=3)
keys = set(zip(sample.game_id.astype(int), sample.play_id.astype(int),
               sample.nfl_id.astype(int)))
pres = dict(zip(zip(sample.game_id.astype(int), sample.play_id.astype(int),
                    sample.nfl_id.astype(int)), sample.pressure_event))
rows = []
for chunk in pd.read_csv("game_tracking_2025.csv", chunksize=2_000_000,
                         low_memory=False,
                         dtype={"x": np.float32, "y": np.float32, "s": np.float32}):
    ids = list(zip(chunk.game_id.astype(int), chunk.play_id.astype(int),
                   chunk.nfl_id.astype(int)))
    rows.append(chunk[[k in keys for k in ids]])
trj = pd.concat(rows, ignore_index=True)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                               gridspec_kw={"width_ratios": [3, 1]})
plotted = 0
n_pres = 0
for (g, p, n), g_rush in trj.groupby(["game_id", "play_id", "nfl_id"]):
    key = (int(g), int(p), int(n))
    if key not in keys:
        continue
    g_rush = g_rush.sort_values("time")
    t0r = g_rush[g_rush.event == "ball_snap"]
    if not len(t0r):
        continue
    t0 = pd.Timestamp(t0r.iloc[0].time)
    t = (pd.to_datetime(g_rush.time) - t0).dt.total_seconds().values
    mk = (t >= 0) & (t <= 3.0)
    if mk.sum() < 15:
        continue
    x = g_rush.x.values[mk].astype(float)
    y = g_rush.y.values[mk].astype(float)
    s = g_rush.s.values[mk].astype(float)
    bearing = np.arctan2(y[-1] - y[0], x[-1] - x[0])
    c_, s_ = np.cos(-bearing), np.sin(-bearing)
    fx = (x - x[0]) * c_ - (y - y[0]) * s_
    fy = (x - x[0]) * s_ + (y - y[0]) * c_
    is_pressure = pres.get(key)
    if is_pressure:
        n_pres += 1
        ax.plot(fx, fy, color=CORAL, lw=1.8, alpha=0.85, zorder=2)
    else:
        ax.plot(fx, fy, color=NAVY, lw=0.5, alpha=0.25, zorder=1)
    plotted += 1
ax.set_xlabel("yards along the rush bearing", fontweight="bold")
ax.set_ylabel("lateral yards (travel frame)", fontweight="bold")
ax.axhline(0, color=SLATE, lw=0.5, ls="--", alpha=0.5)
ax.set_xlim(-1, 12)
patch_p = mpatches.Patch(color=CORAL, label=f"pressure ({n_pres} reps)")
patch_n = mpatches.Patch(color=NAVY, alpha=0.3, label=f"no pressure ({plotted - n_pres} reps)")
ax.legend(handles=[patch_p, patch_n], loc="upper right", fontsize=10, frameon=True)

# inset: speed profile for one pressure rep
speeds_p = []
for (g, p, n), g_rush in trj.groupby(["game_id", "play_id", "nfl_id"]):
    key = (int(g), int(p), int(n))
    if key in keys and pres.get(key):
        g_rush = g_rush.sort_values("time")
        t0r = g_rush[g_rush.event == "ball_snap"]
        if len(t0r):
            t0 = pd.Timestamp(t0r.iloc[0].time)
            t = (pd.to_datetime(g_rush.time) - t0).dt.total_seconds().values
            mk = (t >= 0) & (t <= 3.0)
            if mk.sum() > 15:
                speeds_p.append((t[mk], g_rush.s.values[mk].astype(float)))
                break
if speeds_p:
    t_s, s_s = speeds_p[0]
    ax2.plot(t_s, s_s, color=CORAL, lw=2.2)
    ax2.fill_between(t_s, s_s, alpha=0.15, color=CORAL)
    ax2.set_xlabel("seconds from snap", fontweight="bold")
    ax2.set_ylabel("speed (yd/s)", fontweight="bold")
    ax2.set_title("one pressure rep", fontsize=11, style="italic")
fig.suptitle(f"{plotted} pass rush paths, 2025 season, first 3 seconds",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("figures/fig7_trajectory_spaghetti.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)

# ============ fig 8: mediation path diagram ============
zm = lambda v: (v - v.mean()) / (v.std() + 1e-12)
players_agg = pd.read_csv("game_trajectories_players.csv")
ds2 = pd.read_csv("pass_rush_dataset.csv")
gp2 = pd.read_csv("game_pressure.csv")[["nfl_id", "pressure_rate"]]
yr2 = pd.read_csv("combine_results.csv")[["nfl_id", "draft_year"]]
m2 = ds2.merge(gp2, on="nfl_id").merge(players_agg, on="nfl_id").merge(
    yr2, on="nfl_id", how="left")
m2 = m2[m2.n_plays >= 50]
m2["max_speed"] = m2.groupby("draft_year").max_speed.transform(
    lambda s: (s - s.mean()) / s.std() if s.std() > 1e-12 else s * 0)
d3 = m2.dropna(subset=["max_speed", "speed_at_1s_med", "pressure_rate"])
X3 = zm(d3.max_speed).values; M3 = zm(d3.speed_at_1s_med).values
Y3 = zm(d3.pressure_rate).values
a3 = np.polyfit(X3, M3, 1)[0]
co3 = np.linalg.lstsq(np.column_stack([X3, M3, np.ones(len(X3))]), Y3,
                      rcond=None)[0]
tot3 = np.polyfit(X3, Y3, 1)[0]

from intervense.analysis.mediation_chain import mediation_chain
mc = mediation_chain(d3[["max_speed", "speed_at_1s_med", "pressure_rate"]],
                     "max_speed", "speed_at_1s_med", "pressure_rate",
                     n_boot=2000, seed=0)

fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
def box(x, y, text, fc, ec, fs=11):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=ec,
            bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=2))
def arrow(x1, y1, x2, y2, color, lw=2, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                shrinkA=10, shrinkB=10))
box(0.10, 0.65, "Combine drill\\ntop speed", "#eef2ff", NAVY)
box(0.50, 0.65, "in-game speed\\nat 1 second", "#eef2ff", NAVY)
box(0.90, 0.65, "pressure\\nrate", "#fff5f5", CORAL)
arrow(0.17, 0.65, 0.43, 0.65, NAVY)
arrow(0.57, 0.65, 0.83, 0.65, NAVY)
arrow(0.10, 0.55, 0.90, 0.25, SLATE, lw=1.5)
ax.text(0.27, 0.70, f"a = {a3:+.2f}***", fontsize=12, color=NAVY,
        fontweight="bold", ha="center")
ax.text(0.70, 0.70, f"b = {co3[1]:+.2f}***", fontsize=12, color=NAVY,
        fontweight="bold", ha="center")
ax.text(0.50, 0.13, f"direct = {mc['direct_path']:+.2f}", fontsize=11,
        color=SLATE, ha="center")
ax.text(0.50, 0.03, f"total = {tot3:+.2f}", fontsize=11, color=SLATE,
        ha="center")
share_pct = mc.get("mediated_share")
share_str = f"{share_pct:.0%}" if share_pct is not None else "unstable"
lo, hi = mc["indirect_ci95"]
summary = (f"indirect a*b = {mc['indirect_ab']:+.2f}  "
           f"95% CI [{lo:+.2f}, {hi:+.2f}]  |  "
           f"{share_str} of total effect  |  n = {mc['n_rows']}")
ax.text(0.50, 0.92, summary, ha="center", fontsize=11,
        fontweight="bold", color=GOLD if lo > 0 else SLATE,
        transform=ax.transAxes)
ax.text(0.50, -0.02, "All variables standardized; CI from 2000 bootstrap resamples",
        ha="center", fontsize=9, color=SLATE, style="italic",
        transform=ax.transAxes)
fig.tight_layout()
fig.savefig("figures/fig8_mediation_path.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============ fig 9: professional forest plot ============
feats = ["get_off_time", "speed_at_1s", "top_speed_window",
         "path_efficiency", "heading_change_rate", "bend_travel"]
d4 = pd.read_csv("game_trajectories_plays.csv")
d4 = d4.dropna(subset=["get_off_time", "speed_at_1s"])
d4 = d4[d4.pocket_time > 0.5]
for c in ["pocket_time"] + feats:
    d4[c] = d4[c].fillna(d4[c].median())
fe_out = panel_fixed_effects(
    d4[["nfl_id", "pressure_event", "pocket_time"] + feats],
    outcome="pressure_event", features=["pocket_time"] + feats,
    group="nfl_id")
names = {"get_off_time": "get-off time\\n(slower = less pressure)",
         "speed_at_1s": "speed at 1 second",
         "top_speed_window": "top speed\\nbefore release",
         "path_efficiency": "path efficiency",
         "heading_change_rate": "heading change\\nrate",
         "bend_travel": "travel-frame bend"}
coefs, los, his, labels, pvals = [], [], [], [], []
for i, f in enumerate(feats):
    r = fe_out["features"][i + 1]  # skip pocket_time
    coefs.append(r["coef_within_per_sd"])
    los.append(r["ci95"][0]); his.append(r["ci95"][1])
    labels.append(names[f]); pvals.append(r["p_value"])
order = np.argsort(coefs)
fig, ax = plt.subplots(figsize=(8.5, 5))
y_pos = np.arange(len(coefs))
for idx, i in enumerate(order):
    color = CORAL if pvals[i] < 0.001 else (GOLD if pvals[i] < 0.01 else NAVY)
    star = "***" if pvals[i] < 0.001 else "**" if pvals[i] < 0.01 else "*" if pvals[i] < 0.05 else ""
    ax.plot([los[i], his[i]], [idx, idx], color=color, lw=2.5, solid_capstyle="round")
    ax.scatter(coefs[i], idx, color=color, s=80, zorder=3, marker="o")
    ax.text(his[i] + 0.003, idx, f"{coefs[i]:+.3f}{star}",
            va="center", fontsize=10, color=color, fontweight="bold")
ax.axvline(0, color=SLATE, lw=1, ls="--", alpha=0.7, label="no effect")
ax.set_yticks(y_pos)
ax.set_yticklabels([labels[i] for i in order], fontsize=11)
ax.set_xlabel("pressure probability points per +1 SD\\n(within player, pocket time controlled)",
              fontweight="bold", fontsize=11)
ax.set_title("Same player, different rep: mechanics execution and pressure odds",
              fontsize=13, fontweight="bold")
leg_el = [plt.Line2D([0], [0], color=CORAL, lw=2.5, label="p < 0.001"),
          plt.Line2D([0], [0], color=GOLD, lw=2.5, label="p < 0.01"),
          plt.Line2D([0], [0], color=NAVY, lw=2.5, label="p < 0.05")]
ax.legend(handles=leg_el + [plt.Line2D([0], [0], color=SLATE, lw=1, ls="--",
           label="zero")], loc="lower right", fontsize=9)
ax.text(0.98, 0.02, f"n = {fe_out['n_rows']:,} reps, {fe_out['n_groups']} players, 2023-2025",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        color=SLATE, style="italic")
fig.tight_layout()
fig.savefig("figures/fig9_fe_forest.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig7-9 saved")
""")

code("""%run make_mining_figures.py""")

code("""from IPython.display import Image, display

for name in ["fig1_sensor_vs_handheld", "fig2_missingness", "fig3_drift",
             "fig4_speed_to_pressure", "fig5_sign_flip", "fig6_attribution_ci",
             "fig7_trajectory_spaghetti", "fig8_mediation_path",
             "fig9_fe_forest"]:
    print("=", name)
    display(Image(filename=f"figures/{name}.png"))""")

md("""## Conclusions

1. Quote sensor splits, not handheld splits, for short-area quickness: the
   handheld 10-yard split agrees with the sensor at only r = 0.718.
2. Among same-position prospects in the same draft band, PASS_RUSH_DRILL top
   speed carries incremental validity over pick and stopwatch for pressure
   production.
3. Never stack the collinear drill family into one regression and read its
   importances: the sixteen-feature model collapses the drill-speed effect
   into the chance band and its importance ranking does not flag it.
4. Movement predicts pressure conversion, production scores predict snap
   volume; they answer different questions and belong in separate reports.

Limits: one drill, one position family, three cohorts, n = 100 with adequate
opportunity counts; the mediation path through in-game get-off is suggestive
but underpowered (bootstrap interval crosses zero).

Code: [github.com/HanSamaht/intervense](https://github.com/HanSamaht/intervense),
examples/nfl. The causal checks shown here are framework methods, not custom
code written per plot.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
nbf.write(nb, "nfl_bdb2027.ipynb")
print(f"wrote nfl_bdb2027.ipynb with {len(cells)} cells, pinned to {TAG}")
