# Tools

All interrogation tools, grouped by what they answer. Method names link
mentally to the [API reference](api.md).

## Before any modeling

| Tool | Kind | Gives you |
|---|---|---|
| `data_forensics` | descriptive plus rank tests | constant and duplicate columns, missingness inventory, MCAR versus MAR hints (numeric, categorical, datetime witnesses), MNAR tipping sensitivity, numeric and composition drift, two directional join audit |

## What drives the decision

| Tool | Kind | Gives you |
|---|---|---|
| `get_attribution` | observational plus interventional check | SHAP shares with concept grounding |
| `attribution_confidence` | bootstrap | 95 percent intervals per feature plus top-k rank stability |
| `fit_fastshap` + `fastshap_attribution` | amortized | full attribution in one forward pass |
| `intervene` | interventional | the actual do() response |
| `ts_attribution` | timeseries | channel, segment, full window, gradient granularity |

## How do the features relate

| Tool | Kind | Gives you |
|---|---|---|
| `get_causal_graph` | interventional | direct, interaction, correlational edges |
| `get_causal_graph(discovery="pc")` | structure discovery | PC, GES, or LiNGAM edges over features plus prediction |
| `identify_real_effect` | DoWhy identification | backdoor or front-door verdict on the real world effect, latent confounding refused |
| `specification_audit` | specification curve | sign stability across control subsets, minimal flip sets, cohort fold stability, estimator_factory for non sklearn families |
| `shortcut_diff` | two graph diff | reliance classified against the discovered world graph, permutation nulls per share, classification stability under graph bootstrap |
| `temporal_causal_graph` | interventional plus discovery | Granger or PCMCI lagged channel causes with model reliance per lag |

## Is it fair

| Tool | Kind | Gives you |
|---|---|---|
| `counterfactual_fairness` | interventional | flip rate when only the protected attribute changes |
| `fairness_path_decomposition` | interventional plus learned mechanisms | direct versus mediated paths, named mediator, verdict |

## Can I trust the explanation

| Tool | Kind | Gives you |
|---|---|---|
| `run_faithfulness` | interventional | comprehensiveness, sufficiency, random baseline |
| `run_faithfulness_profile` | interventional | pass rate across samples |
| `check_contamination` | observational | train test overlap |
| `leak_scan` | observational | features whose solo AUC rivals the model |
| `check_drift` | observational | KS, PSI, prediction drift |

## What would change the decision

| Tool | Kind | Gives you |
|---|---|---|
| `run_counterfactual` | interventional | minimal flip path |
| `phrase_counterfactual` | interventional | typed phrase edits on text, verified flips |
| `diverse_counterfactuals` | interventional | several distinct flip paths |

## What happens inside

| Tool | Kind | Gives you |
|---|---|---|
| `get_activations` | observational | every hooked layer |
| `patch_activation_test` | interventional | single patch recovery |
| `activation_graph` | interventional | unit level edges |
| `mediation_decomposition` | interventional | NDE and NIE at a layer |
| `neuron_ranking` | mixed | units by contribution |
| `discover_circuit` | interventional | minimal circuit, greedy or exhaustive |
| `circuit_with_confidence` | interventional | circuit with bootstrap intervals |
| `param_ablation` | interventional | reversible weight lesions |
| `sae_features` | interventional check | tagged dictionary with causal verdicts |
| `crosscoder_features` | interventional check | shared versus layer specific concepts, activation examples |
| `transcoder_features` | interventional check | virtual neurons over an MLP block, input to output rules |
| `mechanism_diff` | interventional check | two models through one dictionary |
| `steering_repair` | interventional | verified steering direction repair |

## Is the backtest real

| Tool | Kind | Gives you |
|---|---|---|
| `slippage_stress` | interventional | PF survival under costs |
| `assumption_gap` | interventional | fill semantics inflation ratio |
| `live_faithfulness` | observational | backtest versus live promise |
| deflated Sharpe | statistical | does the edge survive selection bias |

## The investigator

The LLM loop: `form_hypotheses`, `set_causal_prior`, `test_causal_prior`,
`update_hypotheses`, `conclude_investigation`, plus the active designer
`plan_next_intervention` which ranks candidate experiments by expected
information gain. The LLM proposes, the tracker owns the arithmetic,
the threshold enforces honesty. Fully available offline through
`investigate()` and `offline_report()`.

## Evidence integrity

`attach_ledger` chain hashes every record. `case_briefing` verifies the
chain and loads priors from history.
