"""activationcausal graph(circuit tracing lite, Anthropic attribution).

"featurelayercausal graph"layer:"modellayer,unitinthis".

method(allintervention experiment,notcorrelation)——causalin:
  clean = p(sample) sampleprediction
  corrupt = p(reference) sampleprediction
  patched = p(reference | Llayeractivation ← sampleactivation) prediction
  layerineffect = patched - corrupt( clean recovery)

unit:ineffectmostlayer,unit reference theunit sample value,
unit.
"""
from __future__ import annotations

import numpy as np

from ..drivers.torch_model import TorchDriver


def build_activation_graph(
    drv: TorchDriver,
    sample_index: int,
    reference_index: int,
    top_units: int = 6,
) -> dict:
    assert drv.kind in ("torch", "timeseries"), "activationcausal graphsupport torch (MLP/)"
    x = drv.sample_at(sample_index)
    x_ref = drv.sample_at(reference_index)

    p_clean = drv.predict_one(x)
    p_ref = drv.predict_one(x_ref)
    total_gap = p_clean - p_ref

    clean_acts = drv.get_activations(x) # samplelayeractivation()
    ref_acts = drv.get_activations(x_ref)

    nodes = ["input(sample)", "input(reference)", "prediction"]
    edges = []

    def _patched_pred(layer: str, replacement: np.ndarray) -> float:
        drv._patch_plan[layer] = replacement
        try:
            return drv.predict_one(x_ref)
        finally:
            drv._patch_plan.pop(layer, None)

    layer_effects = {}
    for layer in drv.list_layers():
        if layer not in clean_acts or clean_acts[layer].shape != ref_acts[layer].shape:
            continue
        effect = _patched_pred(layer, clean_acts[layer]) - p_ref
        layer_effects[layer] = float(effect)
        nodes.append(layer)
        edges.append(
            {
                "source": layer,
                "target": "prediction",
                "effect": round(float(effect), 4),
                "recovery": round(float(effect / total_gap), 4) if abs(total_gap) > 1e-6 else None,
                "kind": "layer_mediation",
            }
        )

    if not layer_effects:
        return {"nodes": nodes, "edges": edges, "note": "tolayeractivation"}

    best_layer = max(layer_effects, key=lambda k: abs(layer_effects[k]))
    unit_edges = []
    base_act, clean_act = ref_acts.get(best_layer), clean_acts.get(best_layer)
    if base_act is not None and base_act.ndim == 2 and clean_act.shape == base_act.shape:
        for u in range(base_act.shape[1]):
            if abs(clean_act[0, u] - base_act[0, u]) < 1e-6:
                continue
            mod = np.copy(base_act)
            mod[0, u] = clean_act[0, u]
            eff = _patched_pred(best_layer, mod) - p_ref
            if abs(eff) > 1e-4:
                node = f"{best_layer}#u{u}"
                nodes.append(node)
                unit_edges.append(
                    {
                        "source": node,
                        "target": "prediction",
                        "effect": round(float(eff), 4),
                        "recovery": round(float(eff / total_gap), 4) if abs(total_gap) > 1e-6 else None,
                        "kind": "unit_mediation",
                    }
                )
        unit_edges.sort(key=lambda e: -abs(e["effect"]))

    # ---- (B, T, H):GRU/LSTM unit ----
    elif base_act is not None and base_act.ndim == 3 and clean_act.shape == base_act.shape:
        H = base_act.shape[2]
        units = list(range(H)) if H <= 32 else _pre_rank_units(base_act, clean_act, 32)

        def _seq_eff(u, persistent):
            mod = np.copy(base_act)
            if persistent:
                mod[0, :, u] = clean_act[0, :, u] # :unit
            else:
                mod[0, -1, u] = clean_act[0, -1, u] # h_T :finally
            return _patched_pred(best_layer, mod) - p_ref

        for u in units:
            eff_ht = _seq_eff(u, persistent=False)
            if abs(eff_ht) > 1e-4:
                node = f"{best_layer}#u{u}@h_T"
                nodes.append(node)
                unit_edges.append({
                    "source": node, "target": "prediction",
                    "effect": round(float(eff_ht), 4),
                    "recovery": round(float(eff_ht / total_gap), 4) if abs(total_gap) > 1e-6 else None,
                    "kind": "unit_ht_mediation",
                    "detail": "finallyunit(causal)",
                })
            eff_ps = _seq_eff(u, persistent=True)
            if abs(eff_ps) > 1e-4 and abs(eff_ps - eff_ht) > 1e-4:
                node = f"{best_layer}#u{u}@persistent"
                nodes.append(node)
                unit_edges.append({
                    "source": node, "target": "prediction",
                    "effect": round(float(eff_ps), 4),
                    "recovery": round(float(eff_ps / total_gap), 4) if abs(total_gap) > 1e-6 else None,
                    "kind": "unit_persistent_mediation",
                    "detail": "unit(Karpathy''intervention)",
                })
        unit_edges.sort(key=lambda e: -abs(e["effect"]))

    return {
        "nodes": nodes,
        "edges": edges + unit_edges[:top_units],
        "p_sample": round(p_clean, 4),
        "p_reference": round(p_ref, 4),
        "total_gap": round(float(total_gap), 4),
        "best_layer": best_layer,
        "method": "layer/unitactivation patching(intervention experiment,correlation);model h_T unitandunit",
        "note": "lite circuit tracing:Anthropic attributionfeature(SAE),layer/unitin;"
                " LSTMVis/Karpathy unitanalysis,methodascausal()",
    }


def _pre_rank_units(base_act, clean_act, k: int) -> list[int]:
    """unit: |clean-reference| activation(proxy) k."""
    diff = np.abs(clean_act - base_act).sum(axis=(0, 1)) if base_act.ndim == 3 else np.abs(clean_act - base_act).sum(axis=0)
    return list(np.argsort(-diff)[:k])
