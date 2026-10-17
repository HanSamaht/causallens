"""parameter(reversible ablation):parameter.

:(lesion study)——,.
MIC upgrade:reversible + (->->->recovery->confirm),
recoverynot"finding".

 = behaviortest:
  - "pred_shift":N samplepredictiondistribution
  - timeseries has "early_window_share": lag attribution
    ( = theparameter"")
"""
from __future__ import annotations

import numpy as np


def _capability(drv, metric: str, n_samples: int) -> float:
    """Capability."""
    X = drv.background[:n_samples]
    if metric == "early_window_share" and drv.kind == "timeseries":
        shares = []
        for i in range(min(n_samples, 20)):
            heat, _ = drv.window_attribution(drv.sample_at(i))
            half = heat.shape[0] // 2
            tot = np.abs(heat).sum() + 1e-12
            shares.append(np.abs(heat[:half]).sum() / tot)
        return float(np.mean(shares))
    p = drv.predict(X)
    return float(np.mean(p)) # pred_center:predictionindrift


def param_ablation(drv, param_groups: list[str], metric: str = "pred_center",
                   n_samples: int = 50, scale: float = 0.0) -> dict:
    """param_groups: parameter( ["gru.weight_hh"] alllayer).

    returnevery:,effect,reversiblevalidate.
    """
    assert drv.kind in ("torch", "timeseries"), "parameter ablationrequire torch "
    t = drv.torch
    base = _capability(drv, metric, n_samples)

    results = []
    for prefix in param_groups:
        matched = {n: p for n, p in drv.model.named_parameters() if n.startswith(prefix)}
        if not matched:
            results.append({"param_group": prefix, "error": "toparameter"})
            continue
        backup = {n: p.detach().clone() for n, p in matched.items()}

        for p in matched.values(): #
            with t.no_grad():
                p.mul_(scale)
        ablated = _capability(drv, metric, n_samples)

        for n, p in matched.items(): # recovery
            with t.no_grad():
                p.copy_(backup[n])
        restored = _capability(drv, metric, n_samples)

        effect = ablated - base
        reversible = abs(restored - base) < 1e-6
        results.append(
            {
                "param_group": prefix,
                "n_matched_params": len(matched),
                "n_weights": int(sum(p.numel() for p in matched.values())),
                "metric": metric,
                "baseline": round(base, 6),
                "ablated": round(ablated, 6),
                "effect": round(float(effect), 6),
                "restored": round(restored, 6),
                "reversible": reversible,
                "verdict": (
                    f"{'validfinding' if abs(effect) > 1e-4 else 'unchanged'}: {prefix} {metric} "
                    f"{base:.4g}->{ablated:.4g}(delta={effect:+.4g})"
                    + ("" if reversible else " ⚠recovery,result")
                ),
            }
        )
    return {"results": results, "note": "reversible ablation:->->->recovery->;recoveryresult"}


def discover_circuit(drv, capability_samples: int = 20, tolerance: float = 0.02,
                     layer: str | None = None, mode: str = "greedy",
                     refinement_rounds: int = 3) -> dict:
    """ACDC-lite:whenbehaviormostunitcircuit.

    mode:
      "greedy" — (default,):unit,behaviorunchanged.
                    O(N) ,modelresult.
      "exhaustive" — search + (compute):,
                    then refinement_rounds :unit,
                    circuitunit,acceptcircuitbehavior.
                    O(N² × rounds) ,N≤64 .

    layer(defaultfinallylayer)everyunitpersistent ablation:
    behavior < tolerance -> theunitnotcircuit();
    significant -> circuit(recovery).
    model(≤256 unit)also——notrequire EAP.
    """
    assert drv.kind in ("torch", "timeseries"), "circuit discoveryrequire torch "
    layers = drv.list_layers()
    target = layer or (layers[-2] if len(layers) >= 2 else layers[0])

    X = drv.background[:capability_samples]
    p0 = drv.predict(X)

    def _behavior_delta() -> float:
        """Behavior delta."""
        return float(np.mean(np.abs(drv.predict(X) - p0)))

    width = drv.get_activations(drv.background[0])[target].shape[-1]
    if width > 256 and mode == "exhaustive":
        return {"error": f"layer {target} width {width} > 128,exhaustive modenot( greedy + neuron_ranking )"}
    if width > 256:
        return {"error": f"layer {target} width {width} > 256, neuron_ranking "}

    circuit, pruned = [], []
    drv._unit_mask.setdefault(target, set())

    # ---- : ----
    for u in range(width):
        drv._unit_mask[target].add(u)
        delta = _behavior_delta()
        if delta < tolerance:
            pruned.append(u)
        else:
            drv._unit_mask[target].discard(u)
            circuit.append({"unit": u, "behavior_delta_when_ablated": round(delta, 4)})

    # ---- (exhaustive mode): ----
    improvements = 0
    if mode == "exhaustive":
        for _round in range(refinement_rounds):
            changed = False
            # unit(by)
            for u in list(pruned):
                drv._unit_mask[target].discard(u) # recovery,is
                delta = _behavior_delta()
                if delta < tolerance:
                    # notrequire -> canrecovery(more)
                    pass # recovery,allnot
                else:
                    # itsrequire -> circuit
                    drv._unit_mask[target].add(u)
                    pruned.remove(u)
                    circuit.append({"unit": u, "behavior_delta_when_ablated": round(delta, 4)})
                    changed = True
                    improvements += 1

            # circuitunit(by)
            for c in list(circuit):
                u = c["unit"]
                was_masked = u in drv._unit_mask[target]
                drv._unit_mask[target].add(u)
                # initscircuitunitalsobelowthisunitisalso
                delta = _behavior_delta()
                if delta < tolerance:
                    # notrequire ->
                    circuit.remove(c)
                    pruned.append(u)
                    changed = True
                    improvements += 1
                elif not was_masked:
                    drv._unit_mask[target].discard(u)
            if not changed:
                break

    # :circuitbehaviorto
    p_end = drv.predict(X)
    residual = float(np.mean(np.abs(p_end - p0)))
    drv._unit_mask.pop(target, None)

    circuit.sort(key=lambda c: -c["behavior_delta_when_ablated"])
    mode_desc = ("" if mode == "greedy"
                 else f"search+{refinement_rounds}({improvements} )")
    return {
        "layer": target,
        "mode": mode,
        "n_units_total": width,
        "circuit_units": circuit,
        "n_circuit": len(circuit),
        "n_pruned": len(pruned),
        "sparsity": round(len(pruned) / width, 3),
        "improvements_from_refinement": improvements if mode == "exhaustive" else 0,
        "residual_after_restore": round(residual, 6),
        "verdict": (
            f"[{mode_desc}] mostcircuit = {len(circuit)}/{width} unit"
            f"({len(pruned)} , {len(pruned)/width:.0%});"
            f"allcircuitunitbehavior {residual:.4f}"
            + ("(≈0,circuit)" if residual < tolerance else " ⚠,circuitnot")
        ),
        "note": "(ACDC ,model);everyaspersistent ablationintervention",
    }
