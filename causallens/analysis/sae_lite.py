"""SAE-lite: a domain dictionary with tagged features and causal checks.

Lineage: Olshausen and Field 1996 (V1 sparse coding) through Anthropic
Scaling Monosemanticity (2024), sized to train on CPU for layers of
roughly 16 to 256 units.

:
  1. traininglayer h_T activation(expand H->4H,L1 )
  2. tagged:every SAE featureandinput base featurecorrelation -> "featurek ↔ @lag0 (r=0.82)"
     (domainmodelhas:top-activating sampleisinput)
  3. causal(NeurIPS 2025 ):thefeature(SAE ->->model)
     predictionsignificantisfeature,itsas"passedcausal"
"""
from __future__ import annotations

import numpy as np


def collect_hidden(drv, layer: str | None = None, n: int = 300) -> np.ndarray:
    """thelayer h_T activation (n, H)."""
    target = layer or (drv.list_layers()[-2] if len(drv.list_layers()) >= 2 else drv.list_layers()[0])
    rows = [drv.get_activations(drv.sample_at(i))[target] for i in range(min(n, len(drv.background)))]
    arr = np.stack(rows)
    if arr.ndim == 4: # RNN: (n, 1, T, H) -> h_T
        return arr[:, 0, -1, :]
    if arr.ndim == 3: # MLP: (n, 1, H)
        return arr[:, 0, :]
    return arr.reshape(arr.shape[0], -1)


def train_sae(hidden: np.ndarray, expand: int = 4, l1: float = 0.01, epochs: int = 300,
              seed: int = 0) -> dict:
    """Train sae."""
    t = __import__("torch")
    t.manual_seed(seed)
    X = t.tensor(hidden, dtype=t.float32)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    H = hidden.shape[1]
    D = H * expand
    enc_w = t.nn.Parameter(t.randn(H, D) * 0.1)
    dec_w = t.nn.Parameter(t.randn(D, H) * 0.1)
    opt = t.optim.Adam([enc_w, dec_w], lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        z = t.relu(Xn @ enc_w)
        recon = z @ dec_w
        loss = ((recon - Xn) ** 2).mean() + l1 * z.abs().mean()
        loss.backward()
        opt.step()
    with t.no_grad():
        Z = t.relu(Xn @ enc_w).numpy()
    return {"enc_w": enc_w.detach(), "dec_w": dec_w.detach(), "mu": mu, "sd": sd,
            "Z": Z, "n_features": D, "recon_mse": float(((recon - Xn) ** 2).mean().item()) if hasattr((recon - Xn) ** 2, "item") else float(((recon - Xn) ** 2).mean()),
            "mean_active_frac": float((Z > 0.01).mean())}


def sae_features(drv, layer: str | None = None, n: int = 300, expand: int = 4,
                 top_k: int = 8, causal_threshold: float = 0.005,
                 l1: float = 0.01, epochs: int = 300) -> dict:
    """expand sets the dictionary size multiple of the layer width,
    4 by default, 8 or 16 for finer concepts at higher cost."""
    target = layer or (drv.list_layers()[-2] if len(drv.list_layers()) >= 2 else drv.list_layers()[0])
    hidden = collect_hidden(drv, target, n)
    sae = train_sae(hidden, expand=expand, l1=l1, epochs=epochs)
    Z, t = sae["Z"], drv.torch

    # ---- tagged:SAE feature × base inputfeature correlation ----
    X_inputs = drv.background[: len(Z)]
    base_names = getattr(drv, "base_feature_names", drv.feature_names)
    input_view = X_inputs
    if drv.kind == "timeseries": # every base featurevaluetagged
        T, F = drv.seq_len, drv.n_features_raw
        input_view = X_inputs.reshape(len(X_inputs), T, F).mean(axis=1)

    rows = []
    for k in range(sae["n_features"]):
        zk = Z[:, k]
        if (zk > 0.01).sum() < 5:
            continue
        cors = {}
        for j, bname in enumerate(base_names):
            if input_view.shape[1] != len(base_names):
                break
            r = float(np.corrcoef(zk, input_view[:, j])[0, 1]) if np.std(zk) > 1e-9 else 0.0
            if abs(r) > 0.3:
                cors[bname] = round(r, 2)
        if not cors:
            continue
        top_idx = np.argsort(-zk)[:5]
        rows.append({"feature": k, "active_frac": round(float((zk > 0.01).mean()), 3),
                     "correlates_with": dict(sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:3]),
                     "activation_examples": [
                         {"sample_index": int(i), "activation": round(float(zk[i]), 4)}
                         for i in top_idx if zk[i] > 0
                     ]})

    # ---- causal:feature -> model -> prediction ----
    verified = []
    for row in rows[: top_k * 2]:
        k = row["feature"]
        top_idx = int(np.argsort(-Z[:, k])[0])
        x = drv.sample_at(top_idx)
        ref_act = drv.get_activations(x)[target]
        flat_act = ref_act[0, -1, :] if ref_act.ndim == 3 else ref_act.reshape(-1)
        zt = t.relu((t.tensor(flat_act, dtype=t.float32)
                     - sae["mu"].squeeze()) / sae["sd"].squeeze() @ sae["enc_w"])
        zt[k] = 0.0
        patched = (zt @ sae["dec_w"]).detach().numpy() * sae["sd"].squeeze().numpy() + sae["mu"].squeeze().numpy()
        shape = ref_act.shape
        patch_tensor = ref_act.copy()
        if ref_act.ndim == 3:
            patch_tensor[0, -1, :] = patched
        else:
            patch_tensor[0, :] = patched
        drv._patch_plan[target] = patch_tensor
        try:
            p_abl = drv.predict_one(x)
        finally:
            drv._patch_plan.pop(target, None)
        p_base = drv.predict_one(x)
        row["causal_effect"] = round(float(abs(p_abl - p_base)), 5)
        row["verdict"] = "passedcausal" if row["causal_effect"] > causal_threshold else "passed(correlationcausal)"
        verified.append(row)

    verified.sort(key=lambda r: -r.get("causal_effect", 0))
    n_pass = sum(1 for r in verified if r["verdict"].startswith("passed"))
    return {
        "layer": target,
        "n_features": sae["n_features"],
        "recon_mse": round(sae["recon_mse"], 4),
        "active_fraction": round(sae["mean_active_frac"], 3),
        "features": verified[:top_k],
        "verdict": f"{n_pass}/{len(verified)} taggedfeaturepassedcausal(itsascorrelation)",
        "note": "tagged=SAEfeatureandinputbasefeaturecorrelation();causal=featureprediction(intervention)",
    }
