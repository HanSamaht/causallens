"""Counterfactual search: which features flip the prediction.

Engines:
  - Few features (F at most 15): full percentile scan over every
    feature, exhaustive for the top features
  - Many features (F above 15): rank by |SHAP| then search, a DiCE
    style budgeted search over the ranked features

Every reported path is an intervention experiment, consistent with the
framework rule that the LLM only proposes and tools decide.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..core.types import CounterfactualResult
from ..drivers.base import MicDriver
from .attribution import explain_sample

_ALL_QUANTILES = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)


def default_threshold(driver: MicDriver) -> float:
    """probabilitydefault 0.5;scoredefault 0();backtest PF default 1.0()."""
    if driver.kind == "backtest":
        return 1.0
    return 0.5 if driver.prediction_semantics == "probability" else 0.0


def _normalized_l1(driver: MicDriver, x: np.ndarray, x2: np.ndarray) -> float:
    """Normalized l1."""
    rng = np.maximum(driver.background.max(axis=0) - driver.background.min(axis=0), 1e-9)
    return float(np.abs(x2 - x).sum() / len(x) / np.mean(rng))


def _flipped(p: float, p_base: float, threshold: float) -> bool:
    """Flipped."""
    return (p - threshold) * (p_base - threshold) < 0


def _base_state(driver: MicDriver, x: np.ndarray) -> tuple[float, int, float]:
    """(probability, argmax, threshold)."""
    full = driver.predict_full(np.asarray(x, dtype=float).reshape(1, -1))[0]
    p_base = float(full[min(driver.target_class, len(full) - 1)]) if len(full) > 1 else float(full[0])
    base_label = int(np.argmax(full))
    return p_base, base_label, default_threshold(driver)


def _eval_flip(driver: MicDriver, x: np.ndarray, assignments: dict,
               p_base: float, base_label: int, threshold: float):
    """executeinterventionverdict:=argmax ;its=threshold."""
    x2 = driver.intervene(x, assignments)
    full = driver.predict_full(x2.reshape(1, -1))[0]
    p = float(full[min(driver.target_class, len(full) - 1)]) if len(full) > 1 else float(full[0])
    if getattr(driver, "n_classes", 2) > 2:
        flipped = int(np.argmax(full)) != base_label
    else:
        flipped = _flipped(p, p_base, threshold)
    return flipped, p, x2


def _make_result(driver, x, p_base, assignments, p_cf, method) -> CounterfactualResult:
    """Make result."""
    x2 = driver.intervene(x, assignments)
    return CounterfactualResult(
        base_prediction=p_base,
        cf_prediction=float(p_cf),
        changed_features={
            f: (float(x[driver.feature_names.index(f)]), float(v)) for f, v in assignments.items()
        },
        distance=_normalized_l1(driver, x, x2),
        flipped=True,
        method=method,
    )


def _sweep_small(driver: MicDriver, x, p_base, threshold, max_features: int) -> CounterfactualResult | None:
    """percentilescan:feature -> Top-2 -> Top-3 ."""
    ranked = [a.feature for a in explain_sample(driver, x)]
    top = ranked[: 2 * max_features]
    for k in range(1, max_features + 1):
        combos = list(itertools.combinations(top[: max(2 * k, k)], k))[: 60]
        plans = []
        for combo in combos:
            for qs in itertools.product(_ALL_QUANTILES, repeat=k):
                plans.append({f: driver.quantile_of(f, q) for f, q in zip(combo, qs)})
        best = None
        for assignments in plans:
            p = driver.predict_one(driver.intervene(x, assignments))
            if _flipped(p, p_base, threshold):
                cand = (len(assignments), _normalized_l1(driver, x, driver.intervene(x, assignments)))
                if best is None or cand < best[0]:
                    best = (cand, assignments, p)
        if best:
            _, assignments, p = best
            return _make_result(driver, x, p_base, assignments, p, "quantile_sweep")
    return None


def _search_large(
    driver: MicDriver,
    x,
    p_base: float,
    threshold: float,
    ranked: list[str],
    n_paths: int,
    n_restarts: int,
    seed: int,
) -> list[CounterfactualResult]:
    """|SHAP| + percentilevalue;feature,return."""
    rng = np.random.default_rng(seed)
    weights = np.array([1.0 / (i + 1) for i in range(len(ranked))]) # byin
    weights /= weights.sum()
    F = len(ranked)
    found: list[CounterfactualResult] = []
    seen_sets: set[frozenset] = set()

    for _ in range(n_restarts):
        k = int(rng.integers(1, min(4, F) + 1))
        chosen = rng.choice(ranked, size=k, replace=False, p=weights) if F > k else ranked[:k]
        assignments = {f: driver.quantile_of(f, float(rng.choice(_ALL_QUANTILES))) for f in chosen}
        p = driver.predict_one(driver.intervene(x, assignments))
        if _flipped(p, p_base, threshold):
            key = frozenset(assignments)
            if key in seen_sets:
                continue
            seen_sets.add(key)
            found.append(_make_result(driver, x, p_base, assignments, p, "weighted_random"))
            if len(found) >= n_paths * 4:
                break

    found.sort(key=lambda r: (len(r.changed_features), r.distance))
    # :notfeature n_paths
    diverse: list[CounterfactualResult] = []
    used: set[frozenset] = set()
    for r in found:
        key = frozenset(r.changed_features)
        if key not in used:
            diverse.append(r)
            used.add(key)
        if len(diverse) >= n_paths:
            break
    return diverse


def search_counterfactual(
    driver: MicDriver,
    x: np.ndarray,
    pack=None,
    threshold: float | None = None,
    max_features: int = 3,
) -> CounterfactualResult:
    """mostcounterfactual()."""
    r = search_counterfactuals(driver, x, pack, threshold, max_features, n_paths=1, seed=0)
    if r:
        return r[0]
    p_base = driver.predict_one(np.asarray(x, dtype=float).ravel())
    return CounterfactualResult(
        base_prediction=p_base,
        cf_prediction=p_base,
        changed_features={},
        distance=0.0,
        flipped=False,
        method=f"not_found(F={len(driver.feature_names)})",
    )


def search_counterfactuals(
    driver: MicDriver,
    x: np.ndarray,
    pack=None,
    threshold: float | None = None,
    max_features: int = 3,
    n_paths: int = 3,
    n_restarts: int = 60,
    seed: int = 0,
) -> list[CounterfactualResult]:
    """counterfactual:returnmost n_paths notfeature."""
    x = np.asarray(x, dtype=float).ravel()
    threshold = default_threshold(driver) if threshold is None else threshold
    p_base = driver.predict_one(x)
    ranked = [a.feature for a in explain_sample(driver, x)]
    F = len(ranked)

    results: list[CounterfactualResult] = []

    if F <= 15:
        best = _sweep_small(driver, x, p_base, threshold, max_features)
        if best:
            results.append(best)
        # increases
        results += _search_large(driver, x, p_base, threshold, ranked, n_paths - 1, n_restarts // 2, seed)
    else:
        results = _search_large(driver, x, p_base, threshold, ranked, n_paths, n_restarts, seed)

    seen: set[frozenset] = set()
    diverse: list[CounterfactualResult] = []
    for r in sorted(results, key=lambda r: (len(r.changed_features), r.distance)):
        key = frozenset(r.changed_features)
        if key and key not in seen:
            diverse.append(r)
            seen.add(key)
        if len(diverse) >= n_paths:
            break
    return diverse
