"""Bootstrap confidence intervals for circuit discovery."""
from __future__ import annotations

import numpy as np


def circuit_with_confidence(driver, n_bootstrap=20, tolerance=0.02, layer=None):
    """Circuit discovery with bootstrap confidence intervals.

    Runs greedy circuit discovery n_bootstrap times on random subsets.
    Reports per-unit probability of circuit membership.
    """
    assert driver.kind in ("torch", "timeseries"), "Circuit discovery requires torch driver"
    layers = driver.list_layers()
    target = layer or (layers[-2] if len(layers) >= 2 else layers[0])

    rng = np.random.default_rng(42)
    n_total = len(driver.background)
    width = driver.get_activations(driver.background[0])[target].shape[-1]

    inclusion_counts = {}
    circuit_sizes = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_total, size=max(5, n_total // 2), replace=False)
        X_sub = driver.background[indices]
        p0 = driver.predict(X_sub)
        driver._unit_mask.setdefault(target, set())
        circuit = set()

        for u in range(width):
            driver._unit_mask[target].add(u)
            p_mod = driver.predict(X_sub)
            delta = float(np.mean(np.abs(p_mod - p0)))
            if delta < tolerance:
                pass
            else:
                driver._unit_mask[target].discard(u)
                circuit.add(u)

        driver._unit_mask.pop(target, None)
        circuit_sizes.append(len(circuit))
        for u in circuit:
            inclusion_counts[u] = inclusion_counts.get(u, 0) + 1

    confidences = {u: inclusion_counts.get(u, 0) / n_bootstrap for u in range(width)}
    ranked = sorted(confidences.items(), key=lambda x: -x[1])
    ci_lo = int(np.percentile(circuit_sizes, 2.5))
    ci_hi = int(np.percentile(circuit_sizes, 97.5))
    median_size = int(np.median(circuit_sizes))
    n_high = len([c for _, c in ranked if c >= 0.8])

    return {
        "layer": target,
        "n_bootstrap": n_bootstrap,
        "circuit_size_median": median_size,
        "circuit_size_95ci": [ci_lo, ci_hi],
        "units_by_confidence": [
            {"unit": u, "confidence": round(c, 3)} for u, c in ranked if c > 0
        ],
        "n_high_confidence": n_high,
        "verdict": (
            f"Circuit size: {median_size} units (95% CI: {ci_lo}-{ci_hi}). "
            f"{n_high} units with confidence >= 0.8"
        ),
        "method": "bootstrap_circuit_discovery",
    }
