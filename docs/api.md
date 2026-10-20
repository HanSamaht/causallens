# API Reference

The main entry point is `MicSession`. Everything runs through it; the
analysis modules underneath are importable for advanced use.

```python
from intervense.core import MicSession, ConceptPack
```

## MicSession

```python
MicSession(driver, pack=None)
```

- `driver`: from `detect_driver`, any model with predict
- `pack`: optional `ConceptPack` for domain grounding

### Session and model

| Method | What it does |
|---|---|
| `describe_model()` | kind, features, semantics, capabilities |
| `list_samples(n=5)` | preview background rows |
| `set_target_class(i)` | for multiclass probability models |
| `register_data(name, data)` | named dataset for tools like leak scans |
| `list_datasets()` | what is registered |
| `update_background(X)` | swap the reference distribution |

### Level 1: attribution

| Method | What it does |
|---|---|
| `get_attribution(i)` | SHAP attribution with concept grounding |
| `fit_fastshap(n_samples=150, epochs=200)` | train the amortized explainer |
| `fastshap_attribution(i)` | one-pass attribution after fitting |
| `ts_attribution(i, granularity="auto")` | timeseries: channel, segment, full, gradient |
| `attribution_confidence(n_bootstrap=50)` | bootstrap percentile intervals plus top-k rank stability |
| `simulate_score(i)` | simulate the model score on a sample |

### Level 2: causal graph and discovery

| Method | What it does |
|---|---|
| `get_causal_graph(i=None, discovery=None)` | intervention graph; `discovery="pc"`, `"ges"`, or `"lingam"` adds structure discovery |
| `temporal_causal_graph(max_lag=4, method="granger")` | Granger or PCMCI lagged channel causes with model reliance annotation |
| `identify_real_effect(treatment, outcome="prediction")` | DoWhy backdoor and front-door identification of the real world effect, latent confounding refused |
| `double_ml_effect(treatment, y, n_folds=5)` | cross-fitted partially linear effect, robust SE, naive comparison |
| `cate_heterogeneity(treatment, y, min_leaf=15)` | per-row effect contrasts, depth-2 subgroup rules, stability |
| `omitted_variable_sensitivity(treatment, y)` | Oster delta, bias-adjusted bound, robust or fragile verdict |
| `specification_audit(treatment, y, max_specs=200, sample_weight=None, groups=None, estimator_factory=None)` | refit the model family across control subsets; sign stability, minimal flip sets, chance band verdict, per cohort effect heterogeneity, non sklearn families via factory |
| `shortcut_diff(y, method="pc", edges=None, n_perm=20, n_graph_boot=10)` | classify every relied upon feature against the world graph; reliance at risk, permutation p values, classification stability under graph bootstrap |
| `data_forensics(df, groups=None, other_df=None, key=None)` | pre modeling audit: mechanism hints across numeric, categorical, and datetime witnesses, MNAR tipping, numeric and composition drift, join coverage |

### Level 3: mediation and fairness

| Method | What it does |
|---|---|
| `mediation_decomposition(i, ref_i, layer)` | NDE and NIE at an activation layer |
| `counterfactual_fairness(i, feature, contrasts)` | Kusner style flip test on a protected attribute |
| `fairness_path_decomposition(feature, lo, hi)` | direct versus mediated paths, with verdict |

### Level 4: circuits and activations

| Method | What it does |
|---|---|
| `get_activations(i)` | all hooked layer activations |
| `patch_activation_test(i, ref_i, layer)` | single patch, recovery measured |
| `activation_graph(i, ref_i)` | unit level intervention graph |
| `neuron_ranking(i, layer=None)` | units by contribution |
| `discover_circuit(tolerance=0.02)` | ACDC style greedy circuit search |
| `circuit_with_confidence(n_bootstrap=20)` | circuit with 95 percent bootstrap intervals |

### Level 5: parameters

| Method | What it does |
|---|---|
| `param_ablation(groups, metric="pred_shift")` | reversible weight lesions |

### Level 6: dictionaries

| Method | What it does |
|---|---|
| `sae_features()` | single layer sparse dictionary, tagged and causally checked |
| `crosscoder_features(layers=None)` | one dictionary across layers, shared versus specific, activation examples per feature |
| `transcoder_features(block=None)` | dictionary over one MLP block, input side to output side, virtual neurons |
| `mechanism_diff(other_session, layer=None)` | Crosscoder diff of two models |

### Counterfactuals

| Method | What it does |
|---|---|
| `run_counterfactual(i)` | minimal flip search |
| `phrase_counterfactual(text, max_edits=2)` | typed phrase edits for text drivers, every flip verified |
| `diverse_counterfactuals(i, n_paths=3)` | DiCE style distinct flip paths |

### Trust checks

| Method | What it does |
|---|---|
| `run_faithfulness()` | comprehensiveness, sufficiency, random baseline |
| `run_faithfulness_profile()` | the same across many samples, pass rate |
| `check_drift(new_data)` | KS, PSI, prediction drift |
| `check_contamination(...)` | train versus test overlap |
| `leak_scan(y)` | single feature leakage scan |
| `symbolic_distill()` | model to formula audit |

### Repair

| Method | What it does |
|---|---|
| `steering_repair(i, feature, layer=None)` | learn a steering direction, verify with follow-up |

### Investigator loop

| Method | What it does |
|---|---|
| `investigate(n_samples=10)` | offline investigator pass over findings |
| `plan_next_intervention(n_candidates=16)` | rank candidate interventions by expected information gain |
| `form_hypotheses(statements)` | open a tracked investigation |
| `set_causal_prior(edges)` | record domain prior, bounded nudge |
| `test_causal_prior()` | check each prior edge against interventions |
| `get_hypotheses()` | snapshot with posteriors |
| `update_hypotheses(updates)` | bounded likelihood ratio updates |
| `conclude_investigation()` | verdict, threshold enforced |

### Ledger

| Method | What it does |
|---|---|
| `attach_ledger(ledger, model_version="v1")` | start chain hashed recording |
| `case_briefing()` | chain integrity plus priors from history |

### Reports

| Method | What it does |
|---|---|
| `offline_report(i=0)` | deterministic full text report |
| `audit(i=None)` | findings list with codes |
| `visualize(i=0, output_path=...)` | Neuronpedia style HTML |

### Backtest engines

| Method | What it does |
|---|---|
| `slippage_stress()` | PF survival curve over execution costs |
| `assumption_gap()` | ladder versus market fill inflation |
| `live_faithfulness(backtest, live)` | did the backtest keep its promise |

## Drivers

```python
from intervense.drivers import detect_driver
```

`detect_driver(model, feature_names, background, name="model")` returns
the right driver for: sklearn estimators, tree ensembles, torch modules
(`torch` extra), ONNX (`onnx` extra), pipelines, ensembles, time series
(`TimeSeriesDriver`), backtest engines (`BacktestDriver`), decision
logs, and text classifiers (`TextDriver`).

## Analysis modules

For advanced use the layers are importable directly:

```python
from intervense.analysis import build_causal_graph, discovery_edges, explain_sample
from intervense.analysis.temporal_causal import granger_pairwise
from intervense.analysis.crosscoder import train_crosscoder, crosscoder_model_diff
```

## Agents and server

```python
from intervense.agents import LLMClient
from intervense.agents.interrogator import Interrogator

interrogator = Interrogator(session, llm=LLMClient(), role="auditor")
print(interrogator.ask("which feature dominates the decisions"))
```

MCP server: `python -m intervense serve --model model.pkl --background bg.csv`
