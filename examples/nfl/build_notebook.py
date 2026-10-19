"""Build the public Kaggle notebook nfl_bdb2027.ipynb.

Embeds the pipeline scripts verbatim as appendix cells (%%writefile),
so the notebook runs on Kaggle from the competition attachment alone
plus causallens installed from the repository tag.

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
[CausalLens](https://github.com/HanSamaht/causallens), the open-source causal
interrogation framework this analysis was built with.

**How to run.** Attach the competition data, switch Internet on in the notebook
settings, then Run All. Step 3 scans the 147MB player_play file and takes a few
minutes. Sections marked Appendix carry the full extraction code.
""")

code(f"""import importlib.util, os, subprocess, sys

# Kaggle: install from the release tag. Inside the repository: skip, the
# analysis cells add the repository root to sys.path themselves.
local_repo = os.path.exists("../../pyproject.toml")
if importlib.util.find_spec("causallens") is None and not local_repo:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/HanSamaht/causallens.git@{TAG}"],
        check=True)
try:
    import causallens
    print("causallens", causallens.__version__)
except ImportError:
    print("causallens resolves from the local repository in later cells")
""")

code("""import glob, os

INPUT = "/kaggle/input/nfl-big-data-bowl-2027"
if os.path.isdir(INPUT):
    needed = ["combine_tracking.csv", "combine_results.csv", "players.csv",
              "player_career_successes.csv", "player_play.csv"]
    for name in needed:
        if not os.path.exists(name):
            hit = glob.glob(os.path.join(INPUT, "**", name), recursive=True)
            assert hit, f"competition file not found: {name}"
            os.symlink(hit[0], name)
    print("linked", len(needed), "competition files from", INPUT)
elif os.path.isdir("examples/nfl"):
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
CausalLens method the writeup leads with.""")

code("""import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causallens.analysis.specification_audit import specification_audit
from causallens.drivers import detect_driver

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

md("""## Figures

The six figures of the writeup, regenerated here. Fig 6 calls CausalLens for
the bootstrap attribution intervals.""")

code(f"""%%writefile {SRC['figures']}
{src("figures")}""")

code("""import os
os.makedirs("figures", exist_ok=True)
%run make_figures.py""")

code("""from IPython.display import Image, display

for name in ["fig1_sensor_vs_handheld", "fig2_missingness", "fig3_drift",
             "fig4_speed_to_pressure", "fig5_sign_flip", "fig6_attribution_ci"]:
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

Code: [github.com/HanSamaht/causallens](https://github.com/HanSamaht/causallens),
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
