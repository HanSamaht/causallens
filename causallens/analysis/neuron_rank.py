"""ranking(optional Captum ,fallback,not).

ranking:layerunitcontributionmost.
   captum -> NeuronGradient(Meta ,unit)
         -> fallback:h_T activation × ∂output/∂h_T(×input,)
:activation_graph unit(unit>32),
'this GRU inthis'.
"""
from __future__ import annotations

import numpy as np


def _forward_input(drv, x):
    """typeinput."""
    t = drv.torch
    if drv.kind == "timeseries":
        flat = np.asarray(x, dtype=float).ravel()
        return t.tensor(flat.reshape(1, drv.seq_len, drv.n_features_raw), dtype=t.float32, requires_grad=True)
    return t.tensor(np.asarray(x, dtype=float).reshape(1, -1), dtype=t.float32, requires_grad=True)


def neuron_ranking(drv, sample_index: int, layer: str | None = None, top_k: int = 10) -> dict:
    """drv: TorchDriver / TimeSeriesDriver.returnlayer(defaultfinallylayer)unitranking."""
    assert drv.kind in ("torch", "timeseries"), "rankingrequire torch "
    t = drv.torch
    x = drv.sample_at(sample_index)
    layers = drv.list_layers()
    layer = layer or (layers[-2] if len(layers) >= 2 else layers[0]) # finallylayer
    if layer not in layers:
        raise KeyError(f"layer {layer} notin;: {layers}")

    xt = _forward_input(drv, x)
    engine = "fallback(grad×activation)"

    # ---- :captum NeuronGradient() ----
    last_err = None
    try:
        from captum.attr import NeuronGradient

        module = dict(drv.model.named_modules())[layer]
        width = _layer_width(drv, xt, layer)
        ng = NeuronGradient(drv.model, module)

        def _selector(u):
            # layeroutput (output,hidden) : h_T u unit;layerdirect
            return (lambda out: out[0][0, -1, u]) if drv.kind == "timeseries" else (lambda out: out.reshape(-1)[u])

        contribs = [float(ng.attribute(xt, neuron_selector=_selector(u)).abs().sum()) for u in range(width)]
        contrib = np.array(contribs)
        engine = "captum(NeuronGradient)"
        act_h, grad_h = _act_grad(drv, xt, layer) #
    except ImportError:
        act_h, grad_h = _act_grad(drv, xt, layer)
        contrib = act_h * grad_h
    except Exception as e: # mustbelow(BUG-009/025 )
        last_err = f"{type(e).__name__}: {e}"
        act_h, grad_h = _act_grad(drv, xt, layer)
        contrib = act_h * grad_h

    order = np.argsort(-np.abs(contrib))[:top_k]
    rows = [{"unit": int(u), "activation": round(float(act_h[u]), 4),
             "contribution": round(float(contrib[u]), 5)} for u in order]
    return {
        "layer": layer,
        "n_units": int(act_h.shape[0]),
        "engine": engine,
        "top_units": rows,
        "note": "ranking=contribution; activation_graph unitcausalconfirm(rankingiscorrelationevidence,isintervention)",
        **({"captum_fallback_reason": last_err} if last_err else {}),
    }


def _layer_width(drv, xt, layer) -> int:
    acts = drv.get_activations(drv.background[0])
    a = acts[layer]
    return int(a.shape[-1])


def _act_grad(drv, xt, layer):
    """+,thelayer h_T activationand."""
    t = drv.torch
    captured = {}

    def _hook(_m, _i, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        out.retain_grad() # : backward .grad
        captured["act"] = out

    h = dict(drv.model.named_modules())[layer].register_forward_hook(_hook)
    try:
        out = drv.model(xt)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() > 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        out.squeeze().backward()
        act = captured["act"]
        if isinstance(act, (tuple, list)):
            act = act[0] # RNN: (output, hidden) output
        act = act.detach().cpu().numpy()
        src = captured["act"]
        if isinstance(src, (tuple, list)):
            src = src[0]
        grad = src.grad
        grad = grad.detach().cpu().numpy() if grad is not None else np.zeros_like(act)
    finally:
        h.remove()
    if act.ndim == 3: # layer h_T
        return act[0, -1, :], grad[0, -1, :]
    return act.reshape(-1), grad.reshape(-1)
