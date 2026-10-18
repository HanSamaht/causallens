# CausalLens Technical Report

Version 0.2.1, October 2026

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

## 4. Case study: the age attribution that failed its intervention

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

## 5. Case study: the gesture model that ignores its best signal

A 3-layer bidirectional GRU over 17 IMU channels, 18 gesture classes.
The temporal discovery tool found exactly one significant lagged edge
in the input, channel 0 at lag 1 causing channel 1 (F=8.9, Bonferroni
corrected p=0.027 across 272 tests). The model's attribution mass on
that lag: 0.1 percent.

The data carries a structure the model does not use. The annotation
layer separated data-side causality from model-side reliance, which is
the question a practitioner actually asks.

## 6. Case study: the honest null

Two LSTM versions of an Android traffic classifier, rebuilt from state
dicts. The Crosscoder diff: 512 features, all shared, zero unique, 100
percent behavioral agreement on identical inputs. The tool reported
"same mechanism" instead of inventing differences. An audit tool that
can say "nothing changed" is worth more than one that always finds
something.

## 7. Benchmarks

- 125 tests, fast suite 95 in 69 seconds, slow suite 30 in 167 seconds
- 24 real models across 15 domains interrogated during development,
  findings in BENCHMARKS.md
- Synthetic probes with ground truth: linear coefficients recovered,
  planted leakage flagged, interaction y = x1 times x2 found
- Fairness decomposition validated on a linear model with known
  coefficients: the mediated path carries share 1.0 as planted
- Granger validated on synthetic VAR data: true link found at p below
  1e-6 with correct lag, reverse direction null, noise channel silent

## 8. Known limits

- Pairwise Granger, not conditional (PCMCI is future work)
- No identifiability calculus: backdoor or front-door adjustment is not
  implemented; all interventions are model-input level
- Attribution point estimates lack confidence intervals, except
  circuits which have bootstrap intervals
- The Crosscoder has no transcoder variant or activation visualizer
- Mediator mechanisms in fairness path decomposition are linear

## 9. Availability

AGPL-3.0, source at github.com/HanSamaht/causallens, package on PyPI as
causallens, documentation hosted on GitHub Pages. Python 3.10 to 3.12,
Linux, macOS, and Windows, CPU only for all features.
