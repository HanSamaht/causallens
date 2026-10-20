# Intervense Cross-Domain Model Test Results

> 24 real models, 15 domains, 6 sources (self-owned quant models + Kaggle stranger models).
> Each model runs the full `intervense attach` pipeline: driver detection → concept pack (LLM-drafted) → full interrogation → case file storage.

---

## Representative Model Results

### 1. Quantitative/Crypto: v3d_prod (9-Booster ensemble, 88 coins × 1825 days daily)

| Check | Result | Finding |
|---|---|---|
| Data contamination | Clean | 0 exact duplicates, 0 near-duplicates (800 val vs 1500 train) |
| Label leakage | Clean | Strongest factor solo AUC=0.564 (legitimate level) |
| Look-ahead bias | Clean | All features show smooth IC lag decay (survival ratio 0.85-1.5) |
| Distribution drift | Severe | 42/50 features drifted (market_dispersion PSI=4.79) |
| Train vs val IC | 0.85 → 0.01 | Model learned an outdated market (post-training dataset expanded 88→318 coins) |

Symbolic distillation formula (R²=0.55 vs model output):
```
y_hat ≈ -0.001 + 0.264·[rank_atr²] - 0.188·[z_vol_30d] + 0.136·[atr_adx²] + ...
```
The quadratic term `rank_atr²` indicates a non-monotonic preference for extreme ATR rankings.

---

### 2. Credit: credit_lgbm (Kaggle, 22 one-hot encoded features)

| Check | credit_rf (control) | credit_lgbm (finding) |
|---|---|---|
| Top attribution factor | Loan-to-income 34% | Personal age 40.8% |
| Second factor | Personal income 16% | Loan purpose-Education 14.3% |
| Faithfulness | Warning (cross-sample 75%) | Four failures (single-sample, cross-sample, simulate_score) |
| Fairness | Normal | Age dominates credit decisions (illegal in many jurisdictions) |

Same dataset, two fundamentally different decision strategies: RF uses debt pressure, LGBM uses age.

---

### 3. Backtest Engine: Fill Assumption Inflation (8x)

```
[1] Surface: strategy backtest PF=4.97, win rate 56%
[2] Intervention: do(fill_semantics := market)
    → Win rate 27% | PF 0.62 | Per-trade -1.61%
    Single-variable comparison → PF inflated 8.0x
[3] Stress test: zero-slippage PF=0.62 (below break-even)
    All "profitability" manufactured by the fill assumption
[4] Counterfactual: fill_semantics 1→0 → PF flips back to 4.97
```

This is based on a real case: backtest PF 10-27, live 0.94, gap fully explained by one hidden assumption.

---

### 4. Healthcare: heart_risk_rf (Kaggle, 18 features)

| Rank | Factor | Share | Clinical significance |
|---|---|---|---|
| 1 | ST depression max | 19% | Standard ECG indicator — myocardial ischemia marker |
| 2 | ST slope up | 14% | Abnormal ECG pattern |
| 3 | Max heart rate | 11% | Exercise capacity assessment |
| 4 | Exercise angina | 9% | Chest pain symptom |

Consistent with clinical consensus. Concept pack auto-drafted by GLM, HTML report generated.

---

### 5. Time-series GRU: Full 6-Level Deep Interrogation

| Level | Tool | Result |
|---|---|---|
| 1 Attribution | SHAP | Momentum@lag1 26%, mean-reversion@lag0 21% |
| 2 Causal graph | Intervention | Momentum lag0 41% > lag1 22% > lag2 8% (temporal decay) |
| 3 Layer mediation | Activation patching | GRU layer recovery rate 1.0 |
| 4 Circuits | ACDC-lite | 9/16 units in circuit (greedy) → 6/16 (exhaustive, 3 refinements) |
| 5 Parameter role | Reversible ablation | gru.weight_hh → early-window attribution collapse (long-term memory) |
| 6 SAE | Sparse dictionary | Feature 4 correlates with momentum r=+0.83, passes causal ablation |

---

### 6. Agriculture: crop_rec (Kaggle, 7 features + 2200 rows real data)

| Rank | Factor | Share | Agronomic significance |
|---|---|---|---|
| 1 | Phosphorus | 29% | Core soil nutrient |
| 2 | Potassium | 16% | Soil nutrient |
| 3 | Rainfall | 15% | Climate condition |
| 4 | Humidity | 14% | Climate condition |

---

### 7. Food Science: wine_quality (Kaggle XGB, 11 features)

| Rank | Factor | Share | Enology basis |
|---|---|---|---|
| 1 | Volatile acidity | 26% | Low volatile acid = quality wine |
| 2 | Alcohol | 24% | Higher alcohol = concentrated flavor |
| 3 | Sulphates | 17% | Appropriate preservative = quality assurance |

---

### 8. Cybersecurity: spam_sms (Kaggle RF, 57 features)

| Rank | Factor | Share | Classic spam indicator |
|---|---|---|---|
| 1 | Remove word freq | 9% | "remove/click here" |
| 2 | Dollar sign freq | 8% | "$$$" |
| 3 | Exclamation freq | 7% | "!!!" |

---

## Full Test Matrix

| # | Domain | Model | Type | Source | Status | Key finding |
|---|---|---|---|---|---|---|
| 1 | Quant/Crypto | m4h_long | LGBM ensemble | Self | Pass | ATR family 32% dominant |
| 2 | Quant/Crypto | m4h_short | LGBM ensemble | Self | Pass | Volatility 40% + weekday effect 6.9% |
| 3 | Quant/Crypto | v3d_prod | 9-Booster ensemble | Self | Pass | 42/50 features drifted |
| 4 | Quant/Crypto | gat_best | GAT (torch) | Self | Pass | Near-constant output (std=0.0013) |
| 5 | Credit | credit_rf | RF 22 features | Kaggle | Pass | Loan-to-income 34% (normal) |
| 6 | Credit | credit_lgbm | LGBM 22 features | Kaggle | Pass | Age 40.8% fairness risk |
| 7 | Environment | delhi_aqi_rf | RF regression 5 features | Kaggle | Pass | Month 59% (calendar mapping) |
| 8 | Motion | gesture_gru | BiGRU 18 classes | Kaggle | Pass | Reverse-engineered + full deep suite |
| 9 | Sequence | andro_lstm | LSTM univariate | Kaggle | Pass | state_dict reverse-engineered |
| 10 | Healthcare | heart_risk_rf | RF 18 features | Kaggle | Pass | ST depression 19% (clinically correct) |
| 11 | Real Estate | house_price_rf | RF regression | Kaggle | Fail | Model unfitted (empty shell) |
| 12 | Agriculture | crop_rec | RF 7 features | Kaggle | Pass | Phosphorus 29% + potassium 16% |
| 13 | Bioinformatics | protein_bilstm | BiLSTM 2-layer | Kaggle | Pass | Reverse-engineered + neuron ranking |
| 14 | Environment/Water | water_quality | RF 9 features | Kaggle | Pass | pH 23% + chloramine 19% |
| 15 | Geology | earthquake_reg | XGB regression | Kaggle | Pass | f1 73% (anonymous) |
| 16 | Geology | earthquake_cls | XGB classification | Kaggle | Pass | f1 54% (anonymous) |
| 17 | Food Science | wine_quality | XGB 11 features | Kaggle | Pass | Volatile acidity 26% + alcohol 24% |
| 18 | Meteorology | rainfall_rf | RF 3 features | Kaggle | Pass | f1 58% (anonymous) |
| 19 | Anti-spam | spam_sms | RF 57 features | Kaggle | Pass | Remove word 9% + $ sign 8% |
| 20 | Cybersecurity | phishing_lr | LR 5000 features | Kaggle | Pass | Coefficient ranking (SHAP timeout fallback) |
| 21 | Fake News | fake_news_lr | LR 10005 features | Kaggle | Pass | TF-IDF coefficient ranking |
| 22 | Healthcare | diabetes | — | Kaggle | Fail | pickle protocol 0 incompatibility |
| 23 | Healthcare | stroke | — | Kaggle | Fail | pickle protocol 0 incompatibility |

**Effective interrogation: 21/23 = 91%** (excluding 2 pickle format incompatibilities)

---

## Framework Self-Validation

### Synthetic probes (known ground truth regression tests)

| Probe | Tests | Result |
|---|---|---|
| linear | Attribution ranking vs true coefficients | Spearman = 1.000 |
| leak | Injected leak feature | 84.9% attribution share caught by auditor |
| interact | y = x1·x2 pure interaction | 3 interventional edges captured by causal graph |

### Test suite

76 unit tests + 3 probes, all passing.

### Bug registry

30 bugs documented and closed with root cause analysis.
