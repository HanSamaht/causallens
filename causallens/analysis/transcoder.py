"""Transcoder: a dictionary over an MLP block, input side to output side.

An SAE describes what activations look like. A Crosscoder describes
them across layers or models. A transcoder describes what a block
DOES: its encoder reads the block input, its decoder predicts the
block output, so every feature is a virtual neuron, an input pattern
to output pattern rule the block actually implements.

Reference lineage: Pearce et al., transcoders as weight based
interpretability; Paulo et al. 2025, Transcoders Beat Sparse
Autoencoders for Interpretability.

Causal verification follows the house rule: zero feature k in z,
reconstruct the block output without it, patch that into the model,
measure the prediction shift. Every feature row also carries
activation_examples, the top activating samples with strengths, the
Neuronpedia style panel data.
"""
from __future__ import annotations

import numpy as np

from .crosscoder import activation_examples, flat_last_activation


def _parametrized_modules(drv):
    return [
        name for name, mod in drv.model.named_modules()
        if name != "" and any(p.numel() > 1 for p in mod.parameters(recurse=False))
    ]


def train_transcoder(hidden_in: np.ndarray, hidden_out: np.ndarray, expand: int = 4,
                     l1: float = 0.01, epochs: int = 300, seed: int = 0) -> dict:
    """Train one dictionary from block input to block output."""
    t = __import__("torch")
    t.manual_seed(seed)
    Xin = t.tensor(np.asarray(hidden_in, dtype=np.float32))
    Xout = t.tensor(np.asarray(hidden_out, dtype=np.float32))
    mu_i, sd_i = Xin.mean(0, keepdim=True), Xin.std(0, keepdim=True) + 1e-6
    mu_o, sd_o = Xout.mean(0, keepdim=True), Xout.std(0, keepdim=True) + 1e-6
    A, B = (Xin - mu_i) / sd_i, (Xout - mu_o) / sd_o

    D = A.shape[1] * expand
    enc_w = t.nn.Parameter(t.randn(A.shape[1], D) * 0.1)
    dec_w = t.nn.Parameter(t.randn(D, B.shape[1]) * 0.1)
    opt = t.optim.Adam([enc_w, dec_w], lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        z = t.relu(A @ enc_w)
        recon = z @ dec_w
        loss = ((recon - B) ** 2).mean() + l1 * z.abs().mean()
        loss.backward()
        opt.step()
    with t.no_grad():
        Z = t.relu(A @ enc_w).numpy()
        recon_mse = float(((recon - B) ** 2).mean().item())
    return {"enc_w": enc_w.detach(), "dec_w": dec_w.detach(),
            "mu_i": mu_i, "sd_i": sd_i, "mu_o": mu_o, "sd_o": sd_o,
            "Z": Z, "n_features": D, "recon_mse": recon_mse,
            "mean_active_frac": float((Z > 0.01).mean())}


def _patch_flat(tensor: np.ndarray, patched: np.ndarray) -> np.ndarray:
    """Write a flat per-unit vector back into an activation tensor."""
    out = tensor.copy()
    if out.ndim == 4 and not (out.shape[0] == 1 and out.shape[1] == 1):
        out[0] = out[0] + (patched - out[0].mean(axis=(1, 2)))[:, None, None]
    elif out.ndim == 4:
        out[0, 0, -1, :] = patched
    elif out.ndim == 3:
        out[0, -1, :] = patched
    else:
        out[0, :] = patched
    return out


def transcoder_features(drv, block: tuple[str, str] | None = None, n: int = 200,
                        expand: int = 4, top_k: int = 8, l1: float = 0.01,
                        epochs: int = 300,
                        causal_threshold: float = 0.005) -> dict:
    """Dictionary over one MLP block with causal checks and panel data.

    Args:
        block: (input_layer, output_layer) names around the block. The
            default picks the first two parametrized modules.
    """
    if block is None:
        mods = _parametrized_modules(drv)
        if len(mods) < 2:
            raise ValueError(
                "transcoder needs a block with an input and an output layer; "
                "a single-module model has none, use sae_features there"
            )
        block = (mods[0], mods[1])
    in_layer, out_layer = block
    for name in (in_layer, out_layer):
        if name not in drv.list_layers():
            raise ValueError(f"layer {name!r} not found in the model")

    n_use = min(n, len(drv.background))
    rows_in, rows_out = [], []
    for i in range(n_use):
        acts = drv.get_activations(drv.sample_at(i))
        rows_in.append(flat_last_activation(acts[in_layer]))
        rows_out.append(flat_last_activation(acts[out_layer]))
    hidden_in, hidden_out = np.stack(rows_in), np.stack(rows_out)

    tc = train_transcoder(hidden_in, hidden_out, expand=expand, l1=l1, epochs=epochs)
    Z = tc["Z"]

    base_names = getattr(drv, "base_feature_names", drv.feature_names)
    if drv.kind == "timeseries":
        input_view = drv.background[:n_use].reshape(
            n_use, drv.seq_len, drv.n_features_raw).mean(axis=1)
    else:
        input_view = drv.background[:n_use]

    feature_rows = []
    for k in range(tc["n_features"]):
        zk = Z[:, k]
        if (zk > 0.01).sum() < 5:
            continue
        cors = {}
        if input_view.shape[1] == len(base_names) and np.std(zk) > 1e-9:
            for j, bname in enumerate(base_names):
                r = float(np.corrcoef(zk, input_view[:, j])[0, 1])
                if abs(r) > 0.3:
                    cors[bname] = round(r, 2)
        feature_rows.append({
            "feature": k,
            "active_frac": round(float((zk > 0.01).mean()), 3),
            "correlates_with": dict(sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:3]),
            "activation_examples": activation_examples(Z, k),
        })

    # causal verification: run the block without feature k
    t = drv.torch
    for row in feature_rows[: top_k * 2]:
        k = row["feature"]
        top_idx = int(np.argsort(-Z[:, k])[0])
        x = drv.sample_at(top_idx)
        acts = drv.get_activations(x)
        a_in = flat_last_activation(acts[in_layer])
        zn = (t.tensor(a_in.astype(np.float32)) - tc["mu_i"].squeeze()) / tc["sd_i"].squeeze()
        zt = t.relu(zn @ tc["enc_w"])
        zt[k] = 0.0
        patched = (zt @ tc["dec_w"]).numpy() * tc["sd_o"].squeeze().numpy() \
            + tc["mu_o"].squeeze().numpy()
        drv._patch_plan[out_layer] = _patch_flat(acts[out_layer], patched)
        try:
            p_abl = drv.predict_one(x)
        finally:
            drv._patch_plan.pop(out_layer, None)
        p_base = drv.predict_one(x)
        row["causal_effect"] = round(abs(float(p_abl - p_base)), 5)
        row["verdict"] = (
            "passed causal" if row["causal_effect"] > causal_threshold
            else "correlation only"
        )

    feature_rows.sort(key=lambda r: -r.get("causal_effect", 0))
    n_pass = sum(1 for r in feature_rows if r.get("verdict", "").startswith("passed"))
    return {
        "block": {"input": in_layer, "output": out_layer},
        "n_features": tc["n_features"],
        "recon_mse": round(tc["recon_mse"], 4),
        "active_fraction": round(tc["mean_active_frac"], 3),
        "n_tagged": len(feature_rows),
        "features": feature_rows[:top_k],
        "verdict": (
            f"{len(feature_rows)} virtual neurons tagged for block "
            f"{in_layer} -> {out_layer}, {n_pass} passed the causal test"
        ),
        "method": "transcoder_block",
        "reference": "Paulo et al. (2025) Transcoders Beat Sparse Autoencoders",
        "note": (
            "a transcoder feature is an input pattern to output pattern rule "
            "of the block; activation_examples lists the top activating "
            "samples for panel rendering"
        ),
    }
