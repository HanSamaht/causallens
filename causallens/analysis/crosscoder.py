"""Crosscoder: one sparse dictionary across multiple layers or models.

A single-layer SAE can only describe one layer at a time, so it cannot
say whether a concept is shared between layers or born in one of them.
The Crosscoder (Lindsey et al. 2025) trains one encoder over the
normalized activations of several sources and gives each source its own
decoder. A feature that both decoders use is a shared concept; a
feature only one decoder uses is source specific.

Sources are layers for the within-model view and models for the
model-diff view (see crosscoder_model_diff). Causal verification zeroes
the feature in z and patches the reconstruction into every source at
once, so the measured effect spans the layers the feature lives in.
"""
from __future__ import annotations

import numpy as np


def collect_multilayer_hidden(drv, layers: list[str], n: int = 300) -> dict[str, np.ndarray]:
    """Per-layer activation matrices {layer: (n, H_l)} at the h_T step."""
    out: dict[str, np.ndarray] = {}
    for i in range(min(n, len(drv.background))):
        acts = drv.get_activations(drv.sample_at(i))
        for layer in layers:
            a = acts[layer]
            if a.ndim == 4:  # RNN: (1, 1, T, H) -> last step
                a = a[0, -1, :]
            elif a.ndim == 3:  # conv/linear: (1, T, H) -> last step
                a = a[0, -1, :]
            elif a.ndim == 2:  # linear: (1, H)
                a = a[0]
            out.setdefault(layer, []).append(np.asarray(a, dtype=float).ravel())
    return {layer: np.stack(rows) for layer, rows in out.items()}


def train_crosscoder(hidden: dict[str, np.ndarray], expand: int = 4,
                     l1: float = 0.01, epochs: int = 300, seed: int = 0) -> dict:
    """Train one sparse dictionary that reconstructs every source.

    Encoder input is the concatenation of per-source normalized
    activations. Each source has its own decoder, so shared features
    learn shared decoder directions and source-specific features get
    near-zero decoder weight in the other sources.
    """
    t = __import__("torch")
    t.manual_seed(seed)
    names = list(hidden.keys())
    Xs = {}
    mus, sds = {}, {}
    for name in names:
        X = t.tensor(np.asarray(hidden[name], dtype=np.float32))
        mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
        Xs[name] = (X - mu) / sd
        mus[name], sds[name] = mu, sd

    total_dim = sum(Xs[n].shape[1] for n in names)
    D = int(total_dim * expand)
    enc_w = t.nn.Parameter(t.randn(total_dim, D) * 0.1)
    dec_ws = {n: t.nn.Parameter(t.randn(D, Xs[n].shape[1]) * 0.1) for n in names}
    opt = t.optim.Adam([enc_w] + list(dec_ws.values()), lr=1e-2)

    X_cat = t.cat([Xs[n] for n in names], dim=1)
    for _ in range(epochs):
        opt.zero_grad()
        z = t.relu(X_cat @ enc_w)
        loss = sum(((z @ dec_ws[n] - Xs[n]) ** 2).mean() for n in names) / len(names)
        loss = loss + l1 * z.abs().mean()
        loss.backward()
        opt.step()

    with t.no_grad():
        Z = t.relu(X_cat @ enc_w).numpy()
        dec_norms = {
            n: dec_ws[n].norm(dim=1).numpy() for n in names
        }
        recon_mse = {
            n: float(((z @ dec_ws[n] - Xs[n]) ** 2).mean().item()) for n in names
        }
    return {
        "names": names,
        "enc_w": enc_w.detach(),
        "dec_ws": {n: dec_ws[n].detach() for n in names},
        "mu": mus,
        "sd": sds,
        "Z": Z,
        "dec_norms": dec_norms,
        "n_features": D,
        "recon_mse": recon_mse,
        "mean_active_frac": float((Z > 0.01).mean()),
    }


def _feature_source_profile(dec_norms: dict[str, np.ndarray], k: int) -> dict:
    """Per-source decoder norm profile of one feature, normalized."""
    raw = {n: float(dec_norms[n][k]) for n in dec_norms}
    top = max(raw.values()) or 1.0
    shares = {n: round(v / top, 3) for n, v in raw.items()}
    strongest = max(raw, key=lambda n: raw[n])
    weakest = min(shares.values())
    return {
        "decoder_shares": shares,
        "strongest_source": strongest,
        "kind": "shared" if weakest >= 0.5 else "specific",
    }


def crosscoder_features(drv, layers: list[str] | None = None, n: int = 300,
                        expand: int = 4, top_k: int = 8,
                        l1: float = 0.01, epochs: int = 300,
                        causal_threshold: float = 0.005) -> dict:
    """Multi-layer dictionary with shared versus layer-specific features.

    Trains a Crosscoder across the requested layers (default: the
    hidden layers before the head), tags each feature by decoder share
    per layer and by correlation with input base features, then verifies
    causality by ablating the feature across all layers at once.
    """
    if layers is None:
        # parametrized modules only, dropping the final one (the head)
        parametrized = [
            name for name, mod in drv.model.named_modules()
            if name != "" and any(p.numel() > 1 for p in mod.parameters(recurse=False))
        ]
        layers = parametrized[:-1][-3:] if len(parametrized) > 1 else parametrized[:1]
    layers = [l for l in layers if l in drv.list_layers()]
    if len(layers) < 2:
        raise ValueError(
            "crosscoder_features needs at least 2 hidden layers with parameters; "
            "a single-recurrent-layer model has one, use sae_features there"
        )

    hidden = collect_multilayer_hidden(drv, layers, n=n)
    cc = train_crosscoder(hidden, expand=expand, l1=l1, epochs=epochs)
    Z = cc["Z"]

    # input tagging, averaged over the sequence for timeseries
    X_inputs = drv.background[: len(Z)]
    base_names = getattr(drv, "base_feature_names", drv.feature_names)
    if drv.kind == "timeseries":
        input_view = X_inputs.reshape(len(X_inputs), drv.seq_len, drv.n_features_raw).mean(axis=1)
    else:
        input_view = X_inputs

    rows = []
    for k in range(cc["n_features"]):
        zk = Z[:, k]
        if (zk > 0.01).sum() < 5:
            continue
        profile = _feature_source_profile(cc["dec_norms"], k)
        cors = {}
        if input_view.shape[1] == len(base_names) and np.std(zk) > 1e-9:
            for j, bname in enumerate(base_names):
                r = float(np.corrcoef(zk, input_view[:, j])[0, 1])
                if abs(r) > 0.3:
                    cors[bname] = round(r, 2)
        rows.append({
            "feature": k,
            "active_frac": round(float((zk > 0.01).mean()), 3),
            **profile,
            "correlates_with": dict(sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:3]),
        })

    # causal verification: ablate the feature across every layer at once
    t = drv.torch
    for row in rows[: top_k * 2]:
        k = row["feature"]
        top_idx = int(np.argsort(-Z[:, k])[0])
        x = drv.sample_at(top_idx)
        acts = drv.get_activations(x)
        z_parts = []
        for layer in layers:
            a = acts[layer]
            if a.ndim == 4:
                a = a[0, -1, :]
            elif a.ndim == 3:
                a = a[0, -1, :]
            elif a.ndim == 2:
                a = a[0]
            z_parts.append((t.tensor(np.asarray(a, dtype=np.float32).ravel())
                            - cc["mu"][layer].squeeze()) / cc["sd"][layer].squeeze())
        zt = t.relu(t.cat(z_parts) @ cc["enc_w"])
        zt[k] = 0.0
        try:
            for layer in layers:
                patched = (zt @ cc["dec_ws"][layer]).numpy() * cc["sd"][layer].squeeze().numpy() \
                    + cc["mu"][layer].squeeze().numpy()
                ref = acts[layer]
                tensor = ref.copy()
                if ref.ndim == 4:
                    tensor[0, 0, -1, :] = patched
                elif ref.ndim == 3:
                    tensor[0, -1, :] = patched
                else:
                    tensor[0, :] = patched
                drv._patch_plan[layer] = tensor
            p_abl = drv.predict_one(x)
        finally:
            for layer in layers:
                drv._patch_plan.pop(layer, None)
        p_base = drv.predict_one(x)
        row["causal_effect"] = round(float(abs(p_abl - p_base)), 5)
        row["verdict"] = (
            "passed causal" if row["causal_effect"] > causal_threshold
            else "correlation only"
        )

    rows.sort(key=lambda r: -r.get("causal_effect", 0))
    n_shared = sum(1 for r in rows if r["kind"] == "shared")
    return {
        "layers": layers,
        "n_features": cc["n_features"],
        "recon_mse": {k2: round(v, 4) for k2, v in cc["recon_mse"].items()},
        "active_fraction": round(cc["mean_active_frac"], 3),
        "n_tagged": len(rows),
        "n_shared": n_shared,
        "n_specific": len(rows) - n_shared,
        "features": rows[:top_k],
        "verdict": (
            f"{len(rows)} features tagged across {len(layers)} layers: "
            f"{n_shared} shared, {len(rows) - n_shared} layer specific"
        ),
        "method": "crosscoder_multilayer",
        "reference": "Lindsey et al. (2025) Crosscoders: A Sparsely Additive Decomposition",
        "note": (
            "shared means every layer decoder carries at least half the "
            "norm of the strongest; causal effect ablates the feature in "
            "all layers simultaneously"
        ),
    }
