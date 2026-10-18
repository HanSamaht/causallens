# Seven causal upgrades, tested on real Kaggle models

October 2026

Version 0.2.x of CausalLens shipped seven new causal tools. Each one
was validated on synthetic data with ground truth, then run against
real Kaggle artifacts on disk: a lending trio (Random Forest, LightGBM,
XGBoost), a 3-layer bidirectional gesture GRU over 17 IMU channels, and
two LSTM versions of an Android traffic classifier. This post is what
happened.

## 1. Diverse counterfactuals

One flip path is an answer. Several distinct flip paths are options.

On a denied applicant in the lending data, the DiCE style search found
two single-change paths: lower the loan-to-income ratio to 0.09
(prediction 0.414), or clear the rent flag (0.437). Different levers,
same outcome, each verified by re-running the model.

## 2. Structure discovery, hardened for real data

PC from causal-learn now runs over the background plus the prediction
node. The first attempt on the real 22-feature file hung forever: two
one-hot columns were constant over the sample window, the correlation
matrix went singular, and the Fisher-Z tests degenerated. Zero variance
columns are now dropped and conditioning sets capped at three.

After the fix, on real data: PC finishes in 0.3 seconds and places
loan-to-income as the direct cause of the Random Forest's output, GES
picks a loan grade in 1.8 seconds, LiNGAM returns income and loan
amount with weights in 1.6 seconds.

## 3. Fairness path decomposition

The protected attribute effect, split into direct and mediated paths.
On synthetic lending data with a planted age to income to label
mechanism plus a direct age edge, the decomposition attributed the
effect to income at share 1.0 with the direct edge at 0.15, matching
the planted coefficients.

On the real Kaggle models it returned the headline of our other post:
the age effect is negligible under intervention on all three models,
attribution notwithstanding.

## 4. Temporal causal discovery

Pairwise Granger over the channels of a time series driver, with two
guards against the sliding window duplication that inflates F
statistics, and one annotation that makes it useful: per-lag model
reliance, the share of the model's attribution mass on that exact
channel and lag.

On the gesture GRU: exactly one significant lagged edge in the input,
channel 0 at lag 1 into channel 1 (F=8.9, corrected p=0.027 across 272
tests). Model reliance on it: 0.1 percent. The data has a structure the
model ignores. The tool separated the two claims instead of blending
them.

## 5. The Crosscoder

One sparse dictionary across multiple layers, each feature labeled
shared or layer specific, causal test ablating across all layers at
once. On a two-hidden-layer MLP: 160 features tagged, 155 shared, 5
layer specific, strongest causal effect 0.085.

## 6. Priors with bounds

The LLM investigator can now declare its causal prior as explicit
edges. Hypotheses formed after a prior get nudged in log-odds by at
most ln(1.5), every nudge logged, then every prior edge is tested
against the intervention graph and rated supported, refuted, reversed,
violated, or untested. On the lending Random Forest, a three edge prior
came back two supported, zero violated.

## 7. Mechanistic model diff

Two models, one dictionary, per-model causal effects. The honest
headline from the real LSTM pair: 512 features, all shared, zero
unique, 100 percent behavioral agreement. Same mechanism, retrained.
The tool reported nothing changed, which is the correct answer, and
which a diff tool afraid of null results would never say.

## The pattern across all seven

Every tool follows the same discipline: synthetic ground truth first,
real artifacts second, honest nulls reported as findings. The suite
that ran all of this, examples/kaggle_causal_suite.py, ships in the
repository and runs anywhere the artifacts exist via the
KAGGLE_MODELS_DIR environment variable.
