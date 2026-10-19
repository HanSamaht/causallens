# CausalLens Technical Report

Version 0.12.0, October 2026

## Abstract

CausalLens is a causal interrogation engine for black-box models. Where
conventional explainability reports attributions, CausalLens runs
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

CausalLens is built around that gap. Every tool is an experiment first
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

## 9. Benchmarks

- 125 tests, fast suite 95 in 69 seconds, slow suite 30 in 167 seconds
- 24 real models across 15 domains interrogated during development,
  findings in BENCHMARKS.md
- Synthetic probes with ground truth: linear coefficients recovered,
  planted leakage flagged, interaction y = x1 times x2 found
- Fairness decomposition validated on a linear model with known
  coefficients: the mediated path carries share 1.0 as planted
- Granger validated on synthetic VAR data: true link found at p below
  1e-6 with correct lag, reverse direction null, noise channel silent

## 10. Known limits

- Pairwise Granger remains the default; PCMCI needs the tigramite
  extra and conditions on the observed past only
- Identification trusts the framework graph or user edges; graph
  mis-specification is not detectable from data alone
- Mediator mechanisms in fairness path decomposition are linear
- The specification audit refits sklearn families only in v1; the
  closed form covers linear and ridge, other families go through refits
- Phrase counterfactual lexicons are English only and small
- EIG planning relies on literal hypothesis to feature name matching

## 11. Availability

AGPL-3.0, source at github.com/HanSamaht/causallens, package on PyPI as
causallens, documentation hosted on GitHub Pages. Python 3.10 to 3.12,
Linux, macOS, and Windows, CPU only for all features.
