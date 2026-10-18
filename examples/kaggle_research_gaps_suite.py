"""Run the six gap-closing tools against real Kaggle artifacts.

Companion to kaggle_causal_suite.py: that file tested the seven causal
upgrades, this one tests the research gap batch, identifiability,
PCMCI, attribution intervals, the transcoder, and the EIG planner,
across model types: tree ensembles, a bidirectional GRU, and a protein
1D CNN. The models stay local, they are gitignored.
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

from causallens.core import MicSession
from causallens.drivers import detect_driver

HERE = os.path.dirname(os.path.abspath(__file__))
KAG = os.environ.get("KAGGLE_MODELS_DIR", os.path.join(HERE, "kaggle_models"))


def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def require_artifacts():
    if not os.path.isdir(os.path.join(KAG, "credit_rf_dir")):
        print(f"[gap-suite] artifacts not found under {KAG}, set KAGGLE_MODELS_DIR; skipping")
        sys.exit(0)


def load_credit():
    df = pd.read_csv(os.path.join(KAG, "credit_risk_bg.csv"))
    feats, bg = list(df.columns), df.to_numpy(float)
    with open(os.path.join(KAG, "credit_rf_dir", "credit_random_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    lgbm = joblib.load(os.path.join(KAG, "credit_ensemble_dir", "lgbm_model.pkl"))
    return feats, bg, rf, lgbm


def section_identifiability():
    feats, bg, rf, lgbm = load_credit()
    line("1. real effect identification on the credit Random Forest (DoWhy)")
    session = MicSession(detect_driver(rf, feats, bg, name="kaggle_credit_rf"))
    for treatment in ("loan_percent_income", "person_income", "person_age"):
        out = session.identify_real_effect(treatment)
        bd = out["backdoor_set"]
        print(f"  {treatment:>22}: identified={out['identified']} "
              f"backdoor={bd} estimate={out.get('estimate')} ")
        print(f"  {'':>22}  {out['verdict'][:88]}")

    print("\n  same question on the LGBM twin, where age carries attribution mass:")
    session_l = MicSession(detect_driver(lgbm, feats, bg, name="kaggle_credit_lgbm"))
    out = session_l.identify_real_effect("person_age")
    print(f"  person_age: identified={out['identified']} backdoor={out['backdoor_set']}")
    print(f"  {out['verdict'][:96]}")


def section_attribution_ci():
    feats, bg, rf, _ = load_credit()
    line("2. attribution confidence intervals on the credit Random Forest")
    session = MicSession(detect_driver(rf, feats, bg, name="kaggle_credit_rf"))
    out = session.attribution_confidence(n_bootstrap=20, top_k=5)
    print(f"  {out['verdict'][:96]}")
    for r in out["features"][:5]:
        print(f"  {r['feature']:>28}  share {r['share']:.3f}  "
              f"CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  top5 stability {r['top5_stability']:.0%}")


def section_eig_planner():
    feats, bg, rf, _ = load_credit()
    line("3. EIG planner on the credit Random Forest")
    session = MicSession(detect_driver(rf, feats, bg, name="kaggle_credit_rf"))
    session.form_hypotheses([
        "loan_percent_income pressure drives the lending decision",
        "person_age alone moves the lending decision",
        "loan_grade_D dominates every other feature",
    ])
    out = session.plan_next_intervention(n_candidates=8, top_k=3)
    print(f"  {out['verdict'][:96]}")
    for c in out["candidates"]:
        print(f"  {c['feature']:>28}  EIG {c['eig_nats']:.4f} nats  "
              f"preview delta {c['preview_mean_delta']:.4f}  discriminates {c['discriminates']}")


def section_pcmci():
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
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, x):
            lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long)
            return self.core(x, lengths)[:, 0:1]

    from causallens.drivers.timeseries import TimeSeriesDriver

    rng = np.random.default_rng(3)
    n, T, F = 120, 40, 17
    ch0 = np.cumsum(rng.normal(0, 0.3, size=(n, T)), axis=1)
    ch1 = 0.8 * np.roll(ch0, 1, axis=1) + rng.normal(0, 0.3, size=(n, T))
    ch1[:, 0] = 0.0
    series = rng.normal(0, 0.5, size=(n, T, F))
    series[:, :, 0] = ch0
    series[:, :, 1] = ch1

    line("4. PCMCI conditional temporal graph on the Kaggle gesture GRU")
    session = MicSession(TimeSeriesDriver(GestureClass0(net).eval(),
                                          [f"ch{i}" for i in range(F)], series,
                                          name="kaggle_gesture_gru"))
    out = session.temporal_causal_graph(max_lag=4, method="pcmci",
                                        annotate_model=True, n_windows=100)
    print(f"  method: {out['method']}  significant: {out['n_significant']}")
    for e in out["significant_edges"]:
        rel = e.get("model_reliance")
        rel_s = f"{rel:.1%} model reliance" if rel is not None else "n/a"
        print(f"  {e['edge']}  p_corr={e['p_corrected']}  {rel_s}")
    print(f"  verdict: {out['verdict'][:110]}")


def section_transcoder():
    import torch

    from causallens.drivers import detect_driver as dd

    class ProteinCNNSimple(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(50001, 128)
            self.conv1 = torch.nn.Conv1d(128, 128, 5, padding=2)
            self.conv2 = torch.nn.Conv1d(128, 256, 5, padding=2)
            self.fc = torch.nn.Linear(256, 10)

        def forward(self, x):
            if x.dim() == 2:
                x = x.reshape(-1, 64)
            e = self.embedding(x.long()).permute(0, 2, 1)
            c1 = torch.relu(self.conv1(e))
            c2 = torch.relu(self.conv2(c1))
            pooled = torch.max(c2, dim=2).values  # (B, 256)
            return self.fc(pooled)[:, 0:1]  # single class view for scalar interrogation

    net = ProteinCNNSimple()
    sd = torch.load(os.path.join(KAG, "protein_dl_dir", "best_CNN.pt"),
                    map_location="cpu", weights_only=False)
    missing = set(k for k in sd if k not in net.state_dict())
    assert not missing, f"arch mismatch: {missing}"
    net.load_state_dict(sd)
    net.eval()

    # structured sequences: random background plus a planted motif of a
    # repeated token, so conv responses have a stable dominant position
    # and the global max pool is not blind to channel changes
    rng = np.random.default_rng(0)
    X = rng.integers(1, 5000, size=(120, 64)).astype(float)
    for i in range(len(X)):
        pos = int(rng.integers(4, 50))
        X[i, pos:pos + 10] = 42.0
    drv = dd(net, [f"tok{i}" for i in range(64)], X, name="kaggle_protein_cnn")

    line("5. transcoder over the conv1 -> conv2 block of the protein CNN")
    session = MicSession(drv)
    out = session.transcoder_features(block=("conv1", "conv2"),
                                      epochs=120, n=100, top_k=4)
    print(f"  block {out['block']}  recon mse {out['recon_mse']}")
    print(f"  {out['verdict'][:96]}")
    for f in out["features"]:
        ex = f["activation_examples"][0] if f["activation_examples"] else {}
        print(f"  feat {f['feature']:>3}  causal {f['causal_effect']:.5f}  "
              f"top sample #{ex.get('sample_index')} @ {ex.get('activation')}")


if __name__ == "__main__":
    require_artifacts()
    section_identifiability()
    section_attribution_ci()
    section_eig_planner()
    section_pcmci()
    section_transcoder()
