"""Causal mediation analysis with NDE/NIE decomposition.

Decomposes the total effect of an input change into:
  - Natural Direct Effect (NDE): change in Y from changing X, holding M at baseline
  - Natural Indirect Effect (NIE): change in Y from changing M, holding X at baseline
  - Total Effect (TE) = NDE + NIE + interaction

Reference: Mueller et al. 2026 "The Quest for the Right Mediator",
Pearl (2001) "Direct and Indirect Effects".

Uses CausalLens's existing patch_activation() infrastructure.
"""
from __future__ import annotations

import numpy as np


def mediation_decomposition(driver, sample_index: int, reference_index: int,
                           layer: str) -> dict:
    """Decompose prediction change into direct and indirect effects.

    The experiment:
      x0 = reference sample (clean)
      x1 = target sample (the one being interrogated)
      M0 = layer activations from x0
      M1 = layer activations from x1

    Measurements:
      TE  = Y(x1, M1) - Y(x0, M0)           total effect
      NDE = Y(x1, M0) - Y(x0, M0)           direct (change input, keep M at ref)
      NIE = Y(x0, M1) - Y(x0, M0)           indirect (keep input at ref, change M)

    Args:
        driver: TorchDriver or TimeSeriesDriver.
        sample_index: Target sample (x1).
        reference_index: Reference sample (x0).
        layer: Layer name to use as mediator.

    Returns:
        Dict with TE, NDE, NIE, and proportion mediated.
    """
    x0 = driver.sample_at(reference_index)
    x1 = driver.sample_at(sample_index)

    # Y(x0, M0) - baseline
    p_00 = driver.predict_one(x0)

    # Y(x1, M1) - total
    p_11 = driver.predict_one(x1)

    # Y(x1, M0) - direct effect: run x1 but patch M to M0
    ref_acts = driver.get_activations(x0)
    drv_plan = ref_acts.get(layer)
    if drv_plan is None:
        raise ValueError(f"Layer {layer} not found")
    driver._patch_plan[layer] = drv_plan
    try:
        p_10 = driver.predict_one(x1)
    finally:
        driver._patch_plan.pop(layer, None)

    # Y(x0, M1) - indirect effect: run x0 but patch M to M1
    tgt_acts = driver.get_activations(x1)
    driver._patch_plan[layer] = tgt_acts[layer]
    try:
        p_01 = driver.predict_one(x0)
    finally:
        driver._patch_plan.pop(layer, None)

    TE = p_11 - p_00
    NDE = p_10 - p_00
    NIE = p_01 - p_00
    interaction = TE - NDE - NIE

    prop_mediated = abs(NIE) / (abs(TE) + 1e-12)

    if abs(TE) < 1e-6:
        interpretation = "No meaningful total effect between samples"
    elif prop_mediated > 0.7:
        interpretation = f"Primarily mediated: {prop_mediated:.0%} of effect goes through {layer}"
    elif prop_mediated < 0.3:
        interpretation = f"Primarily direct: only {prop_mediated:.0%} goes through {layer}"
    else:
        interpretation = f"Mixed: {prop_mediated:.0%} mediated, {(1-prop_mediated):.0%} direct"

    return {
        "layer": layer,
        "sample_index": sample_index,
        "reference_index": reference_index,
        "p_baseline": round(float(p_00), 6),
        "p_total": round(float(p_11), 6),
        "total_effect": round(float(TE), 6),
        "natural_direct_effect": round(float(NDE), 6),
        "natural_indirect_effect": round(float(NIE), 6),
        "interaction": round(float(interaction), 6),
        "proportion_mediated": round(float(prop_mediated), 4),
        "interpretation": interpretation,
        "method": "causal_mediation_decomposition",
        "reference": "Pearl (2001); Mueller et al. (2026)",
    }
