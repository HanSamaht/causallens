# CausalLens: Make All Black-Box Models Transparent

CausalLens is a framework for interrogating black-box models. It accepts any model with a `predict` function and runs intervention experiments, faithfulness verification, and (with LLM configured) Bayesian hypothesis testing.

Given a model file and background data, it produces:

- Feature attribution with role annotation from concept packs
- Counterfactual analysis: minimal input changes that flip predictions
- Internal analysis for torch models: activation patching, SAE. Circuit discovery available for non-RNN architectures.
- Faithfulness verification of its own explanations
- Leakage detection: train/test contamination, label leakage, look-ahead bias
- Distribution drift monitoring

## Example: credit risk model

A LightGBM model from Kaggle, 22 features, trained on a lending dataset.

Attribution result: `person_age` accounts for 40.8% of total attribution share.

Additional checks SHAP does not perform:

| Check | Result |
|---|---|
| Faithfulness (mask top features, compare vs random baseline) | Failed. Top-k masking effect not stronger than random baseline. |
| Intervention (set age to different values, measure prediction shift) | Age has a causal effect on predictions. This confirms the feature is used, though the overall explanation quality remains poor. |
| Auditor (concept pack role tagging) | Flagged age as fairness risk |
| Evidence ledger | All findings recorded with chain-hash |

A Random Forest from a different Kaggle author, trained on the same dataset, showed a different profile:

| | Random Forest | LightGBM |
|---|---|---|
| Top factor | Loan-to-income ratio 34% | person_age 40.8% |
| Faithfulness | 75% pass rate | Failed (faithfulness and simulate_score checks) |

## Comparison with SHAP

CausalLens uses SHAP for feature attribution. SHAP is one component. The following capabilities are not available in SHAP:

- Direct intervention: set feature to specific value and measure prediction change (SHAP's interventional mode marginalizes over a background distribution instead)
- Counterfactual search: minimal feature changes that flip predictions
- Activation patching, circuit discovery, SAE (torch models)
- Faithfulness verification: masking tests, cross-sample profiles, explain-simulate-score
- Leakage detection: contamination, label leak, look-ahead
- Drift monitoring: background distribution staleness
- Backtest engine auditing: fill assumption analysis, slippage stress
- Decision log interrogation: surrogate model with fidelity bounds
- LLM investigator: hypothesis-driven, Bayesian updates, refuses conclusions below odds ratio threshold

## Explanation depth

```
1  Attribution       SHAP + concept grounding                   All models
2  Causal graph      Direct / interaction / correlated edges    All models
3  Layer mediation   Activation patching                       Torch
4  Unit circuits     h_T units + ACDC-lite                     Torch
5  Parameter role    Reversible ablation                       Torch
6  Semantic dict     SAE-lite + causal falsification           Torch
```

Levels 2-6 use intervention experiments. Level 1 uses SHAP. Conclusions carry faithfulness bounds.

## Test results

21 models successfully interrogated across 15 domains (24 attempted, 3 failed due to pickle incompatibility).

| Domain | Model | Finding |
|---|---|---|
| Credit | credit_lgbm (Kaggle) | Age 40.8% (fairness risk) |
| Credit | credit_rf (Kaggle) | Loan-to-income 34% |
| Agriculture | crop_rec (Kaggle) | Phosphorus 29%, potassium 16% |
| Healthcare | heart_risk_rf (Kaggle) | ST depression 19%, max HR 11% |
| Food science | wine_quality (Kaggle) | Volatile acidity 26%, alcohol 24% |
| Bioinformatics | protein_bilstm (Kaggle) | Architecture from state_dict |
| Anti-spam | spam_sms (Kaggle) | Remove-word freq 9%, $ sign 8% |
| Geology | earthquake_reg (Kaggle) | f1 73% |
| Environment | water_quality (Kaggle) | pH 23%, chloramines 19% |
| Crypto | v3d_prod (production) | 42/50 features drifted |
| Crypto | m4h_long/short (production) | ATR family 32-40% |

## Case studies

### Credit risk: age bias

**Model:** credit_lgbm (LightGBM, 22 one-hot encoded features, Kaggle)

Two models loaded on the same dataset. Attribution profiles:

| | credit_rf | credit_lgbm |
|---|---|---|
| Top factor | Loan-to-income 34% | person_age 40.8% |
| Second | Personal income 16% | Loan purpose-Education 14.3% |
| Faithfulness | 75% pass | Failed (faithfulness and simulate_score checks) |

Age at 40.8% of attribution share in lending decisions raises fairness concerns under regulations such as the US Equal Credit Opportunity Act. The auditor flagged this via concept pack role tagging.

### Heart disease: matches clinical consensus

**Model:** heart_risk_rf (Random Forest, 18 features, Kaggle)

| Rank | Factor | Attribution |
|---|---|---|
| 1 | ST depression max | 19% |
| 2 | ST slope up | 14% |
| 3 | Max heart rate | 11% |
| 4 | Exercise angina | 9% |

These match cardiology exercise test indicators. Concept pack was drafted by GLM from feature names with no medical input.

### Protein BiLSTM: no source code

**Model:** protein_bilstm (2-layer BiLSTM, embedding 50001 to 128, Kaggle)

No source code, only a `.pt` weights file.

1. Architecture manually inferred from state_dict keys using CausalLens's inspection utilities
2. Model class manually reconstructed: `Embedding(50001, 128)` + `LSTM(128 to 256, bidirectional)` + `FC(512 to 10)`
3. Weights loaded with `strict=True`
4. Neuron ranking identified decision units

## Quick start

```bash
pip install -e .
python -m causallens hero
```

```bash
python -m causallens ask --demo geo --concepts concepts/geo.json \
    "if precipitation increases by 50%, how does landslide risk change?"
```

```
[Intervention] do(precipitation := 1853)   1235 → 1853 (+50%)
[Answer] System probability: 0.685 → 0.725 (+0.040)
[Boundary] In-model reasoning, not real-world prediction
```

```bash
python -m causallens attach --model your_model.pkl --background data.csv --domain "credit risk"
python -m causallens visualize --demo timeseries
```

## Architecture

```
Entry     CLI / MCP Server (stdio+HTTP) / LLM investigator
Core      LensSession unified interrogation API
Analysis  SHAP, do-interventions, counterfactuals, causal graphs,
          symbolic distillation, 4-tier attribution, FastSHAP
Verify    Single-sample faithfulness, cross-sample profiles,
          explain-simulate-score
Deep      Activation patching, unit circuits, SAE-lite,
          parameter ablation, steering
Leakage   Contamination, label leak, look-ahead, deflated Sharpe
Drivers   tree, sklearn, torch, timeseries, ensemble, onnx,
          pipeline, backtest, decision_log
Domains   Concept packs: finance, medical, timeseries, geo, credit, backtest
Archive   SQLite chain-hash, bi-temporal hypotheses, cross-version priors
```

## Investigator

The LLM investigator forms competing hypotheses, designs experiments, and performs Bayesian updates server-side. Conclusions require odds ratio above 5.0.

Test result: GLM output was "odds ratio 1.9, below 5.0, insufficient evidence."

## Evidence ledger

- Chain-hash: modifications to any evidence entry break verification
- Bi-temporal hypotheses: "finding X held for v3, invalidated at v4"
- Cross-version priors: new investigations load historical outcomes

## Leakage detection

| Tool | Detects |
|---|---|
| duplicate_overlap | Row-level train/test overlap |
| single_feature_leak_scan | Single feature with near-perfect AUC |
| lookahead_scan | Information loss when features lag by one period |
| deflated_sharpe | Sharpe ratio after removing selection bias |

## LLM support

| Provider | Variable |
|---|---|
| Zhipu GLM | ZHIPUAI_API_KEY |
| OpenAI | OPENAI_API_KEY |
| DeepSeek | DEEPSEEK_API_KEY |
| Qwen | DASHSCOPE_API_KEY |
| Moonshot | MOONSHOT_API_KEY |
| Custom | MIC_LLM_BASE_URL + MIC_LLM_API_KEY |

## Self-validation

Synthetic probes with known ground truth:

| Probe | Test | Result |
|---|---|---|
| linear | Attribution ranking vs true coefficients | Spearman = 1.000 |
| leak | Injected leak feature | Located at 84.9% attribution |
| interact | y = x1·x2 | 3 interventional edges detected |

## References

MAIA (MIT CSAIL), Anthropic Circuit Tracing, OpenAI auto-interp, LLM-as-an-Investigator, Causal-INSIGHT, PBO and Deflated Sharpe (Bailey, Lopez de Prado), SHAP (Lundberg, Lee), TimeSHAP (Feedzai), LSTMVis, Karpathy et al. 2015, FastSHAP (ICLR 2022), Captum, GS-SHAP (2026)

## License

AGPL-3.0
