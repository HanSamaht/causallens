"""Run the seven new causal tools against real Kaggle models.

Models on disk (examples/kaggle_models):
  - credit Random Forest, LGBM, XGBoost over the 22 feature lending set
  - gesture GRU: 3-layer bidirectional GRU, 17 IMU channels, 18 classes
  - andro LSTM v1 and v2: 2-layer LSTM, univariate input sequence

The script prints one section per tool with the findings.
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.core import MicSession
from intervense.drivers import detect_driver
from intervense.drivers.timeseries import TimeSeriesDriver

HERE = os.path.dirname(os.path.abspath(__file__))
KAG = os.environ.get("KAGGLE_MODELS_DIR", os.path.join(HERE, "kaggle_models"))


def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def require_artifacts():
    """Exit cleanly when the Kaggle artifacts are not on this machine."""
    if not os.path.isdir(os.path.join(KAG, "credit_rf_dir")):
        print(f"[kaggle-smoke] artifacts not found under {KAG}")
        print("[kaggle-smoke] set KAGGLE_MODELS_DIR to the directory holding")
        print("[kaggle-smoke] credit_rf_dir, credit_ensemble_dir, gesture_gru_dir,")
        print("[kaggle-smoke] andro_lstm_dir; skipping")
        sys.exit(0)


# ================================================================
# 1. Credit Random Forest: discovery, fairness paths, prior, diverse CF
# ================================================================

def load_credit():
    bg_df = pd.read_csv(os.path.join(KAG, "credit_risk_bg.csv"))
    feats = list(bg_df.columns)
    bg = bg_df.to_numpy(dtype=float)

    with open(os.path.join(KAG, "credit_rf_dir", "credit_random_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    lgbm = joblib.load(os.path.join(KAG, "credit_ensemble_dir", "lgbm_model.pkl"))
    xgb = joblib.load(os.path.join(KAG, "credit_ensemble_dir", "xgb_model.pkl"))
    return feats, bg, rf, lgbm, xgb


def section_credit():
    feats, bg, rf, lgbm, xgb = load_credit()
    session = MicSession(detect_driver(rf, feats, bg, name="kaggle_credit_rf"))

    line("1. causal-learn PC structure discovery on the credit Random Forest")
    out = session.get_causal_graph(discovery="pc")
    disc = out["discovered_edges"]
    print(f"discovered edges: {len(disc)}")
    for e in disc[:10]:
        print(f"  {e['source']:>28} -> {e['target']:<28} {e['detail']}")
    direct = sorted(out["direct_edges"], key=lambda e: -e["effect"])[:5]
    print("top direct attribution shares:")
    for e in direct:
        print(f"  {e['source']:>28}    {e['detail']}")

    line("2. fairness path decomposition: person_age 25 vs 60")
    out = session.fairness_path_decomposition("person_age", 25.0, 60.0, n_samples=40)
    print(f"mediators auto-detected: {out['mediators']}")
    print(f"mean total effect: {out['mean_total_effect']:+.4f}")
    for name, p in out["paths"].items():
        print(f"  {name:>22}  effect {p['mean_effect']:+.4f}  share {p['share']:+.2f}")
    print(f"verdict: {out['verdict']}")

    print("\ncross-check on the LGBM twin, where age is the top driver:")
    lgbm_session = MicSession(detect_driver(
        joblib.load(os.path.join(KAG, "credit_ensemble_dir", "lgbm_model.pkl")),
        feats, bg, name="kaggle_credit_lgbm"))
    out = lgbm_session.fairness_path_decomposition("person_age", 25.0, 60.0, n_samples=40)
    print(f"mediators auto-detected: {out['mediators']}")
    print(f"mean total effect: {out['mean_total_effect']:+.4f}")
    for name, p in out["paths"].items():
        print(f"  {name:>22}  effect {p['mean_effect']:+.4f}  share {p['share']:+.2f}")
    print(f"verdict: {out['verdict']}")

    line("3. causal prior injection and intervention test (credit RF)")
    prior = [
        {"source": "loan_percent_income", "target": "prediction", "strength": "strong"},
        {"source": "loan_grade_D", "target": "prediction", "strength": "medium"},
        {"source": "person_age", "target": "prediction", "strength": "weak",
         "expected": False},
    ]
    session.form_hypotheses([
        "loan_percent_income pressure drives the lending decision",
        "person_age alone moves the lending decision",
        "loan_grade_D dominates every other feature",
    ])
    out = session.set_causal_prior(prior)
    print(f"prior stored: {out['n_edges']} edges; hypotheses nudged: "
          f"{out['hypothesis_injection']['n_nudged']}")
    out = session.test_causal_prior()
    for r in out["rows"]:
        print(f"  {r['edge']:>44}  {r['verdict']:<9}  {r['detail']}")
    print(f"verdict: {out['verdict']}")
    snap = session.get_hypotheses()
    for hid, h in snap["hypotheses"].items():
        nudge = [e for e in h["evidence"] if e["tool"] == "prior_injection"]
        tag = f" (prior nudge {nudge[0]['lr_step']})" if nudge else ""
        print(f"  {hid} posterior {h['posterior']}{tag}: {h['statement']}")

    line("4. DiCE diverse counterfactuals on a denied applicant (credit RF)")
    drv = session.driver
    found = None
    for i in range(40):
        if drv.predict_one(drv.sample_at(i)) > 0.5:
            found = i
            break
    if found is None:
        print("no positive sample in the first 40 rows")
    else:
        paths = session.diverse_counterfactuals(found, n_paths=3)
        print(f"sample {found}: {len(paths)} distinct flip paths")
        for p in paths:
            changes = ", ".join(f"{k} -> {v:.2f}" for k, v in p["changed_features"].items())
            print(f"  [{p['method']}] {changes}  => p = {p['prediction_after']}")

    line("5. PC discovery on the LGBM and XGBoost twins for cross-check")
    for m, nm in ((lgbm, "kaggle_credit_lgbm"), (xgb, "kaggle_credit_xgb")):
        s2 = MicSession(detect_driver(m, feats, bg, name=nm))
        o2 = s2.get_causal_graph(discovery="pc")
        print(f"{nm}: {len(o2['discovered_edges'])} discovered edges, "
              f"top direct: {max(o2['direct_edges'], key=lambda e: e['effect'])['source']}")


# ================================================================
# 6. Gesture GRU: temporal causal discovery with model reliance
# ================================================================

def section_gesture():
    import importlib.util
    import torch

    spec = importlib.util.spec_from_file_location(
        "gru_mod", os.path.join(KAG, "gesture_gru_dir", "gru_model_module.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    net = mod.GRUMultiClassModel(input_size=17, hidden_size=116, output_size=18,
                                 num_layers=3, bidirectional=True)
    sd = torch.load(os.path.join(KAG, "gesture_gru_dir", "imu_gru_model_gesture.pt"),
                    map_location="cpu", weights_only=False)
    net.load_state_dict(sd)

    class GestureClass0(torch.nn.Module):
        """Single-class view of the 18 class GRU for scalar interrogation."""

        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, x):
            lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long)
            return self.core(x, lengths)[:, 0:1]

    model = GestureClass0(net).eval()

    # structured windows: accel ch0 leads, gyro ch1 follows at lag 1, rest noise
    rng = np.random.default_rng(3)
    n, T, F = 160, 40, 17
    ch0 = np.cumsum(rng.normal(0, 0.3, size=(n, T)), axis=1)
    ch1 = 0.8 * np.roll(ch0, 1, axis=1) + rng.normal(0, 0.3, size=(n, T))
    ch1[:, 0] = 0.0
    series = rng.normal(0, 0.5, size=(n, T, F))
    series[:, :, 0] = ch0
    series[:, :, 1] = ch1

    names = [f"ch{i}" for i in range(F)]
    drv = TimeSeriesDriver(model, names, series, name="kaggle_gesture_gru")
    session = MicSession(drv)

    line("6. temporal causal discovery on the Kaggle gesture GRU")
    out = session.temporal_causal_graph(max_lag=4, alpha=0.05, annotate_model=True)
    print(f"channels: {out['channels'][:4]} ... ({len(out['channels'])} total)")
    print(f"significant lagged edges: {out['n_significant']}")
    for e in out["significant_edges"]:
        rel = e.get("model_reliance")
        rel_s = f"{rel:.1%} of model attribution mass" if rel is not None else "n/a"
        print(f"  {e['edge']}  F={e['f_stat']}  p_corr={e['p_corrected']}  model uses: {rel_s}")
    print(f"verdict: {out['verdict']}")


# ================================================================
# 7. Andro LSTM v1 vs v2: mechanistic model diff
# ================================================================

def section_andro():
    import torch

    class AndroLSTM(torch.nn.Module):
        """Rebuilt from the state dict: LSTM(1, 64, 2 layers) + Linear(64, 1)."""

        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, 64, num_layers=2, batch_first=True)
            self.fc = torch.nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    def load(version):
        m = AndroLSTM()
        path = os.path.join(KAG, "andro_lstm_dir", f"andro_lstm_rnn_model{version}.pth")
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        return m.eval()

    # bursty univariate traffic windows
    rng = np.random.default_rng(0)
    n, T = 120, 30
    windows = []
    for _ in range(n):
        w = rng.normal(0, 0.2, size=T)
        for _ in range(rng.integers(1, 5)):
            start = rng.integers(0, T - 6)
            w[start:start + rng.integers(3, 7)] += rng.normal(3, 0.5)
        windows.append(w)
    series = np.asarray(windows, dtype=float)[:, :, None]

    a = MicSession(TimeSeriesDriver(load(""), ["traffic"], series, name="andro_v1"))
    b = MicSession(TimeSeriesDriver(load("_v2"), ["traffic"], series, name="andro_v2"))

    line("7. Crosscoder mechanism diff: andro LSTM v1 vs v2")
    out = a.mechanism_diff(b, layer="lstm", epochs=150, n=100, top_k=6)
    print(f"layer: {out['layer']}  inputs compared: {out['n_inputs']}")
    print(f"recon mse: {out['recon_mse']}")
    print(f"verdict: {out['verdict']}")
    for f in out["features"]:
        print(f"  feature {f['feature']:>3}  {f['kind']:<18}"
              f"  effect v1 {f['causal_effect_a']:.4f}  v2 {f['causal_effect_b']:.4f}"
              f"  shares {f['decoder_shares']}")
    # behavior cross-check
    pa = a.driver.predict(series.reshape(len(series), -1))
    pb = b.driver.predict(series.reshape(len(series), -1))
    agree = float(np.mean((pa > 0.5) == (pb > 0.5)))
    print(f"behavioral agreement on same inputs: {agree:.1%}")


if __name__ == "__main__":
    require_artifacts()
    section_credit()
    section_gesture()
    section_andro()
