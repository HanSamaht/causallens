# Stopwatch Against Sensor: What Combine Tracking Really Adds to Pass Rush Evaluation

**Requirement 1 statement.** This writeup links Combine sensor tracking (PASS_RUSH_DRILL and FORTY_YARD_DASH movement) to regular-season NFL performance: pressure rate per pass-rush opportunity, per-rep pressure outcomes across 35,646 tracked reps, and snap counts, for all 118 front-seven prospects in the 2023-2025 cohorts.

## Motivation

Every spring, scouts read the same three numbers off a stopwatch: the forty, the split, the shuttle. The sensor worn during those same drills records ten frames a second of position, speed, and acceleration. We ask the question a front office actually cares about: beyond the stopwatch and beyond the draft pick, does the sensor know anything about who pressures NFL quarterbacks?

Our answer has three parts: the sensor and the stopwatch disagree in ways that matter, one sensor measurement carries real signal about NFL pressure production, and the multivariate habit that dominates this competition would have destroyed that signal if we had let it.

## Data and measurement discipline

**Phase features from published geometry.** The host documented the PASS_RUSH_DRILL layout (dummies 4 yards from center, finish cone 7 yards deep, timing from snap to cone). We segment each attempt with those anchors, not guessed phases: total time, top speed, speed at 4 and 5 yards of depth, first-step displacement in 0.5s, mean acceleration over the first second, path efficiency (net displacement over path length), and bend rate (heading change per second while wider than 3 yards). 225 attempts by 118 front-seven prospects, all phases derived from host-published geometry.

**Missing-label semantics.** Following the host's clarifications, `quick_pressure` missing with `blitzing=TRUE` is an observed non-event and stays in the opportunity denominator; `player_get_off` missing (never crossed the line) is excluded from get-off averages but not denominators. Pressure events are the union of pressure time, hurry time, and sacks, which also handles the 23 anomalous hurry-without-pressure records. Opportunity definition: linemen are credited every non-nullified pass play; linebackers only when blitzing, since a coverage linebacker did not rush.

**Protocol drift.** Five of ten tracking features shift across 2023-2025 (Kruskal p<0.05; Fig. 3), matching host-confirmed recording changes. All cross-year tracking features are standardized within draft year. The sensor forty itself is stable (4.69-4.73s cohort means).

## Finding 1: the stopwatch and the sensor agree only after you define the start

Raw recording time inflates the tracked forty by 0.4 seconds against the handheld, because recordings begin before the athlete moves. Aligning the clock to genuine motion (speed above 2 yd/s) closes the bias to -0.025s and lifts the correlation to r=0.969 across 381 paired attempts (Fig. 1a).

The split is where the stopwatch breaks: handheld 10-yard splits correlate only r=0.718 with the aligned sensor (Fig. 1b). Human timers are precise over forty yards and noisy over ten. If a team is drafting on short-area quickness, the sensor is not an increment over the stopwatch; it is the only trustworthy measurement in the building.

## Finding 2: the stopwatch sample is selected, the sensor sample is not

Traditional drills are missing for 20 to 64 percent of our cohort (Fig. 2): three-cone 64%, shuttle 61%, bench 54%, forty 20%. The missingness is not random; high picks skip drills they do not need. Sensor features are 0-4% missing. Any traditional-metrics-only model in this competition is conditioned on a sample selected on the outcome's strongest predictor, draft capital. This is a selection-bias trap the sensor quietly removes.

## Finding 3: drill speed predicts NFL pressure, past position and draft capital

Among the 100 prospects with at least 50 pass-rush opportunities (median 199, top 1,344), pressure rate per opportunity ranges 0 to 19.6%.

Drill top speed correlates with NFL pressure rate at Spearman rho=0.528 raw (0.49 after within-year standardization; Fig. 4 shows the raw association). The obvious confounders are position (edge rushers are faster and rush more productively) and draft capital (scouts know something). Controlling both via partial correlation on within-year standardized features, drill speed retains r=+0.240 (n=82, t=2.2); speed at 4 yards of burst retains r=+0.172. A backdoor adjustment over {position group, draft pick} identifies the effect formally, and the intervention estimate on the lean model is +0.026 pressure rate for moving a prospect from the 10th to the 90th within-cohort percentile of drill speed, roughly a quarter of the cohort's interquartile range in production.

The chain through the game data closes the mechanism. The full 10Hz game tracking covers the Combine cohort only, so every mechanic below is computed in the rusher's own travel frame, immune to missing passer geometry. Across 35,646 pass rush reps by these same 115 players (2023-2025), within-player fixed effects with cluster-robust errors show that the same player's execution on a given rep moves his pressure odds on that rep: path efficiency +5.7 probability points per SD, top speed before release +3.8, speed at one second +2.3, travel-frame bend +1.6 (all p below 0.001, base rate 10 percent, Fig. 9). Holding out the entire 2025 season, mechanics lift play-level pressure AUC from 0.755 to 0.857 with excellent calibration (top quintile predicted 30.4 percent, actual 30.5). Formally, drill top speed transfers to in-game speed at one second (partial r = +0.24) and in-game speed tracks pressure (partial r = +0.46): the mediated share is 84 percent of the total effect with a bootstrap interval of +0.26 to +0.52, entirely above zero (Fig. 8). The combine speed test earns its keep one second after every snap, as game speed.

For career snap volume the sensor adds nothing: draft pick alone reaches cross-validated R2 0.23, and adding all drill features does not improve it. Production scores, not movement, separate who plays from who washes out. Speed earns rushes; production earns snaps.

## Finding 4: the crowded model buries the signal, and feature importance hides the burial

The finding above nearly did not survive our own first analysis. In a 16-feature ridge regression of pressure rate, collinearity between the forty, the split, drill speed, and burst speed erases the drill-speed effect: the intervention estimate collapses from +0.026 (three-feature model) to -0.0002 (sixteen-feature model), the latter inside the chance band for this sample size (Fig. 5a). A specification audit that refits the same model family on 200 sampled control subsets, weighted exactly like the production fit, puts the effect above the chance band in 33 specifications and never materially below it (Fig. 5b). The correct description is collapse, not flip; we corrected our own earlier wording accordingly. Meanwhile the crowded model's SHAP ranking places drill speed dead last, under 1 percent share at rank 16 of 16: nothing in a ranking distinguishes a collapsed effect from a useless one.

Any submission that reads feature importance off a crowded model, which is the default habit in this competition, would have concluded drill speed does not matter. The lesson generalizes: importance is bookkeeping, not behavior. We verified every headline claim with an explicit intervention and a stated adjustment set, using the open-source Intervense interrogation framework built by the first author; the full evidence chain (specification audit, bootstrap attribution intervals in Fig. 6, backdoor identification) runs in the public notebook.

## What teams should take

1. Stop quoting handheld 10-yard splits; the sensor version is strictly better measured.
2. Among same-position prospects in the same draft-round band, PASS_RUSH_DRILL top speed is a legitimate tiebreaker with demonstrated incremental validity over pick and stopwatch.
3. Do not stack drill metrics into one regression and read importances; the measurement family is too collinear. One or two speed features against position and capital controls is the honest instrument.
4. Movement predicts pressure opportunity conversion; college production predicts snap volume. They answer different questions and should be reported separately.

## Limits

One drill, one position family, three cohorts, n=100 with adequate opportunity counts. The mediation path is underpowered. Get-off averages include multi-year veterans' rookie seasons only, and opportunity attribution for blitzing linebackers follows the documented approximation. All code and the evidence chain are public; the null results above are reported with the same discipline as the positives.

---
*Figures: 9. Tables: 0. Words: ~1,350 (excluding this note).*
