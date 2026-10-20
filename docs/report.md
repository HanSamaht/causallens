# Intervense Technical Report

Version 0.17.0, October 2026

## Abstract

Intervense is a causal interrogation engine for black-box models. Where
conventional explainability reports attributions, Intervense runs
interventions, maintains a Bayesian hypothesis tracker, and chain hashes
every claim into an evidence ledger. This report describes the method,
the seven causal extensions shipped in 0.2.x, and the case studies that
motivate them, including a lending model whose headline attribution
fails its own intervention test.

## 1. The problem: attribution is not causation

SHAP answers "which features does the model use". It cannot answer
"what would change if we intervened", and the two can disagree. That
disagreement is exactly where explainability tools quietly fail their
users, and it is detectable only by running the intervention.

Intervense is built around that gap. Every tool is an experiment first
and a summary second.

## 2. Architecture

- 10 driver kinds: sklearn, tree ensembles, torch, ONNX, pipelines,
  ensembles, time series, backtest engines, decision logs, text
- 6 depth levels: attribution, causal graph, mediation, circuits,
  parameters, dictionaries
- Evidence model with enforced tiers: interventional, observational,
  prior, manual; observational evidence can never carry a causal
  conclusion
- Hypothesis tracker: LLM proposes, server-side arithmetic decides,
  likelihood ratios bounded in [0.2, 5], conclusions require 5:1
  posterior odds
- SQLite ledger with chain hashing and bitemporal hypotheses
- MCP server for LLM workbench integration

## 3. The seven causal extensions in 0.2.x

1. Diverse counterfactuals, DiCE style: several distinct flip paths
   with different feature subsets, each verified by re-running the model
2. Structure discovery backends: PC, GES, and LiNGAM from causal-learn
   run over the background plus the prediction node, with zero variance
   columns dropped and conditioning sets capped so real one-hot
   backgrounds terminate
3. Fairness path decomposition: the protected attribute effect split
   into a direct path and one path per mediator, with mediator
   mechanisms learned as linear models so natural values exist
4. Temporal causal discovery: pairwise Granger with two guards against
   sliding window inflation, plus per-lag model reliance annotation
5. Crosscoder: one sparse dictionary across layers, features labeled
   shared or layer specific, causal test ablates across all layers at
   once
6. Causal prior injection: domain priors recorded explicitly, bounded
   to ln(1.5) log-odds per hypothesis, every nudge logged, then tested
   edge by edge against interventions
7. Mechanistic model diff: two models through one Crosscoder with
   per-model causal effects

## 4. The research gap batch in 0.3.0 to 0.8.0

Six gaps from the first report, closed and tested on real artifacts:

1. Identifiability: identify_real_effect runs DoWhy backdoor and
   front-door criteria over the framework graph. Reciprocal
   interaction pairs collapse to correlational edges, correlation
   becomes a latent common cause, and a latent U on both treatment
   and outcome is refused rather than estimated naively, stricter
   than the DoWhy default.
2. Conditional temporal discovery: PCMCI from Tigramite conditions
   each link test on the system's full past, behind
   temporal_causal_graph(method="pcmci").
3. Attribution intervals: attribution_confidence bootstraps the
   background and reports 95 percent percentile intervals plus top-k
   rank stability; weak evidence is named.
4. The transcoder: a dictionary over an MLP block, encoder on the
   input side, decoder on the output side, every feature a virtual
   neuron. All three dictionaries now ship activation example panels.
5. Active experiment design: plan_next_intervention ranks candidate
   interventions by expected information gain over the standing
   hypotheses, a simulated lookahead disclosed as such, exposed to
   the LLM investigator as a tool.
6. Phrase counterfactuals: typed edits, negation, intensity,
   antonym, deletion, contrastive clauses, under a budget, every
   flip verified.

On the real artifacts: the credit Random Forest identifies the
loan-to-income effect through a three variable backdoor set while age
estimates land near zero; the LGBM twin gets age identified only
through a loan intent and grade adjustment set; PCMCI on the gesture
GRU confirms the single lagged edge at corrected p below 0.05 with
model reliance still 0.1 percent; the protein CNN transcoder produces
512 virtual neurons with 5 causally verified features.

A real bug surfaced during artifact testing and is worth recording: 1D
convolution outputs (1, C, L) were misread as RNN (1, T, H) maps, the
last step rule silently took the last channel, and the transcoder
trained on garbage columns with perfect reconstruction and zero causal
effects. Layer semantics are now explicit per module type: rnn, conv,
dense.

## 5. Case study: the age attribution that failed its intervention

A Kaggle lending dataset, 22 one-hot features, three models trained on
identical data.

| Model | Top attribution | Age intervention effect (25 to 60) |
|---|---|---|
| LightGBM | personal age, 40.8 percent | 0.0000 |
| XGBoost | personal age | under 0.005 |
| Random Forest | loan-to-income, 34.2 percent | 0.0049 |

The two boosted models place their attribution mass on age. The
intervention test says moving age from 25 to 60 changes almost nothing.
Path decomposition finds no mediator above the correlation bar, so the
effect is neither direct nor mediated: the age attribution is not
backed by the model's behavior under intervention.

The Random Forest, same data, attributes to debt pressure, and the PC
discovery backend independently places loan-to-income as the direct
cause of its output.

A conventional SHAP report would have delivered "age 40.8 percent" as
the finding. The interrogation delivers the finding plus its refutation.

## 6. Case study: the gesture model that ignores its best signal

A 3-layer bidirectional GRU over 17 IMU channels, 18 gesture classes.
The temporal discovery tool found exactly one significant lagged edge
in the input, channel 0 at lag 1 causing channel 1 (F=8.9, Bonferroni
corrected p=0.027 across 272 tests). The model's attribution mass on
that lag: 0.1 percent.

The data carries a structure the model does not use. The annotation
layer separated data-side causality from model-side reliance, which is
the question a practitioner actually asks.

## 7. Case study: the honest null

Two LSTM versions of an Android traffic classifier, rebuilt from state
dicts. The Crosscoder diff: 512 features, all shared, zero unique, 100
percent behavioral agreement on identical inputs. The tool reported
"same mechanism" instead of inventing differences. An audit tool that
can say "nothing changed" is worth more than one that always finds
something.

## 8. The specification audit in 0.12.0

Attribution says what a fitted model leans on. A causal claim needs a
second property: the intervention effect should not depend on which
controls happen to sit next to the treatment in the specification. The
specification audit tests exactly that. It refits the driver's own model
family on each control subset of the specification lattice against the
caller supplied training outcome, records the p10 to p90 do-contrast per
specification, and returns the curve with three layers of discipline: a
chance band scaled to the sample (max of 2 percent of sd(y) and 2.5
times the chance contrast), so noise level sign jitter never counts as
evidence; minimal flip sets, the smallest control subsets that flip the
sign, found by breadth first search up to depth three; and an exact
closed form for linear and ridge families where the whole curve is
arithmetic, not sampling.

The method generalizes a real catch from the NFL Big Data Bowl 2027
analysis, where a 16-feature ridge model erased a drill-speed effect
that the lean model showed while its importance ranking stayed calm.
Running the audit on that same data produced a second finding: the
effect is positive in 28 of 200 specifications above the chance band
and materially negative in none, so the honest description is collapse
into the noise band rather than sign flip, and the earlier wording in
our own writeup was corrected. The audit polices the story told about
the catch, including by its authors.

Provenance: specification curve analysis (Simonsohn, Simmons and Nelson
2020) and multiverse robustness do this for regression coefficients in
quantitative social science. The contribution here is the application
to do-contrasts of refitted black-box models, the chance band
discipline, and minimal flip sets as an audit product with refusal
grade verdicts (STABLE SIGN, FRAGILE SIGN, NEGLIGIBLE).

## 9. The shortcut diff in 0.13.0

A model can be right for the wrong reasons. The shortcut diff builds two
graphs in one call: the world graph, from structure discovery (PC, GES,
or DirectLiNGAM) over the background plus the caller supplied real
outcome, with domain edges able to pin CPDAG orientations; and the
model-implied reliance, from intervention contrasts on the frozen
model. Every relied upon feature is then classified: CAUSAL PATH (a
directed path to the outcome exists, the reliance is structurally
supported), PROXY (the outcome has a directed path into the feature,
which is label leakage or a legitimate pre-outcome measurement
depending on timing the graph cannot see), UNRESOLVED PATH
(connectivity only through undirected edges), or INDEPENDENT (no
connection at all). The verdict reports the interventional reliance
share at risk: the part of the model that breaks when the correlation
carrying it breaks.

On planted scenarios the audit separates the three failure flavors: a
spurious sibling riding a latent cause lands UNRESOLVED PATH with a
double digit reliance share, an overfit noise feature lands INDEPENDENT,
and a label echo lands PROXY at 84 percent reliance while the true
cause carries 12 percent. A clean two-cause model returns SUPPORTED at
100 percent.

Provenance: shortcut features in DAG terms were characterized by
Bellamy et al. (2022), shortcut discovery in black box classifiers by
Sahidullah et al. (2025), the failure mode itself by Geirhos et al.
(2020). The contribution here is the two graph diff as one audit call
with reliance share accounting and refusal grade classification
disclosure: CPDAG orientations are assumptions, and PROXY verdicts
state explicitly that deployment timing knowledge is required to
separate leakage from legitimate measurement.

## 10. Audit hardening in 0.14.0

Three upgrades prompted by benchmarking against the strongest public
Big Data Bowl entries, which showed out of sample discipline the audits
lacked and manual data forensics the framework did not cover.

Cohort fold stability: specification_audit accepts group labels and
refits the lean and full specifications inside each cohort (groups with
at least 20 rows). Per group effects are compared against the dominant
material fold sign, so a finding that exists only in the pool is named
by the cohort that reverses it. On a planted scenario where the effect
is positive in two cohorts and negative in a third, the pooled estimate
cancels toward zero and the fold block returns FOLD FRAGILE with the
reversing cohort named.

Permutation nulls: shortcut_diff refits the driver's family on shuffled
outcomes and recomputes reliance shares per replicate, giving each
observed share a p value against the chance baseline. A label echo at
80 percent reliance tests at p = 1/21 against a null median of 34
percent, while the true cause at 18 percent sits below the null median:
the baseline absorbs both overfitting and share normalization.

Data forensics: one call over a dataframe returns constant and
duplicate columns, the missingness inventory with its dominant pattern,
MCAR versus MAR hints (Mann Whitney between missing and observed rows,
Bonferroni per target), Kruskal Wallis drift across cohort labels, and
a two directional join audit between frames. On the real NFL cohort it
reproduces in seconds what the manual analysis found by hand: draft
pick missingness associates with career snaps (undrafted players get
fewer opportunities, medians 14 versus 315), three drill features drift
across draft years, traditional drills are over 50 percent missing, and
join coverage is 97 percent one direction.

## 11. Depth and coverage in 0.15.0

Four upgrades after the adversarial review of 0.14.1.

Graph bootstrap: the shortcut diff no longer treats the discovered
CPDAG as fixed. n_graph_boot replicates resample the rows, re-estimate
the world graph, and reclassify every relied upon feature; each
classification carries class_prob, its agreement across replicates, and
features below 80 percent agreement are named unstable in the verdict.
User edges are domain knowledge and pin orientations in every
replicate. On the label echo scenario the pinned classifications are
stable at 100 percent across replicates; ambiguous orientations show
their true instability instead of hiding it.

MNAR tipping sensitivity: data forensics answers the question every
MAR finding invites, "but what if it is missing not at random", with an
explicit bound: for each partly missing numeric column it reports the
uniform shift (in observed SD units) the missing entries would need to
move the full column mean by half a standard deviation, plus the
tipping value in column units. The note is honest that this bounds
sensitivity, it cannot rule MNAR out.

Categorical and temporal coverage: MAR witnesses now include
categorical columns via chi square on the missingness by category
table (a strong association can leave few nonzero cells, so the table
is trimmed to its nonzero core rather than rejected), drift scans read
datetime columns as epoch seconds, and categorical columns get
composition drift tests with per group top category shares. The
constant column guard was restored after a Kruskal Wallis edge case it
had covered.

Non sklearn families: specification_audit accepts estimator_factory, a
callable returning an unfitted fit and predict estimator, so torch
wrappers and custom families can be specification audited without
clone support; the refusal message without a factory names the escape
hatch. The auditor role prompt now encodes the audit playbook: data
forensics before any modeling claim, shortcut diff after the first
attribution, specification audit before quoting any feature effect.

## 12. Causal judgment depth in 0.17.0

Three methods move the verdicts from whether an effect exists to how
far the data licenses believing it.

Double ML: cross-fitted partially linear estimation (Chernozhukov et
al. 2018). Nuisance functions are learned by the driver's own model
family, residuals are computed out of fold, and the treatment effect
carries a Neyman-orthogonal robust interval. The output also prints
the naive uncontrolled and controlled OLS estimates, so the confounding
gap is a number on the page, not a footnote. On a planted confounded
simulation (theta 0.8, confounding paths through two covariates) the
cross-fitted estimate lands within 0.1 of the truth while the naive
uncontrolled estimate is visibly biased, and the interval covers the
truth.

CATE heterogeneity: per-row do-contrasts from the fitted family are
mined by a depth-two subgroup tree with a minimum leaf size, so every
subgroup is a rule a decision maker can read (for example one feature
threshold). The strongest-versus-weakest subgroup contrast carries a
bootstrap interval and the top split feature reports its stability
across refits. On a planted design where the effect is 2.0 for one
half of the population and 0.2 for the other, the module returns
HETEROGENEOUS with the split on the right feature and both subgroup
means recovered.

Omitted variable sensitivity: Oster's delta. Comparing the coefficient
before and after controls together with their R-squareds, delta is the
ratio of unobservable to observable selection that nullifies the
effect; values above 1 are the robust convention, values below 1 are
fragile, and the case where controls moved the coefficient away from
zero is refused to an honest INCONCLUSIVE verdict rather than bent
into a pass. The fragility semantics follow Oster's own: a large
coefficient movement under control addition is itself evidence that
selection of the same character may be hiding in the unobservables.

The auditor playbook now chains them: quote no observational effect
without double_ml_effect, no targeting claim without
cate_heterogeneity, and no robustness claim without
omitted_variable_sensitivity.

## 13. Benchmarks

- 125 tests, fast suite 95 in 69 seconds, slow suite 30 in 167 seconds
- 24 real models across 15 domains interrogated during development,
  findings in BENCHMARKS.md
- Synthetic probes with ground truth: linear coefficients recovered,
  planted leakage flagged, interaction y = x1 times x2 found
- Fairness decomposition validated on a linear model with known
  coefficients: the mediated path carries share 1.0 as planted
- Granger validated on synthetic VAR data: true link found at p below
  1e-6 with correct lag, reverse direction null, noise channel silent

## 14. Known limits

- Pairwise Granger remains the default; PCMCI needs the tigramite
  extra and conditions on the observed past only
- Identification trusts the framework graph or user edges; graph
  mis-specification is not detectable from data alone
- Mediator mechanisms in fairness path decomposition are linear
- The specification audit refits sklearn families only in v1; the
  closed form covers linear and ridge, other families go through refits
- The shortcut diff inherits CPDAG orientation limits; undirected edges
  are assumptions until domain edges pin them, and PROXY classification
  cannot separate leakage from legitimate measurement without timing
- Phrase counterfactual lexicons are English only and small
- EIG planning relies on literal hypothesis to feature name matching

## 15. Availability

AGPL-3.0, source at github.com/HanSamaht/intervense, package on PyPI as
intervense, documentation hosted on GitHub Pages. Python 3.10 to 3.12,
Linux, macOS, and Windows, CPU only for all features.
