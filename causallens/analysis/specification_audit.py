"""Specification audit: sign stability of intervention effects.

Feature importance answers "what does this fitted model lean on". A
causal claim needs more: if the model were re-specified with a different
set of controls, would the intervention effect keep its sign? In
collinear families (speed metrics, credit ratios, clinical panels) the
answer is often no, and nothing in the importance ranking warns about
it. This module generalizes that failure into an audit.

For one treatment feature it walks the lattice of control subsets,
refits the same model family per specification against the training
outcome, computes the do-contrast (p10 to p90) per specification, and
reports the sign distribution, the specifications that flip the sign,
and the minimal control sets responsible for the flip. When the fitted
family is linear or ridge the whole curve has a closed form and the
audit becomes exact.

The provenance is specification curve analysis and multiverse
robustness from quantitative social science, applied here to
intervention contrasts of black-box models; see the reference field.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _wilson_interval(k: int, n: int) -> tuple[float, float]:
    """Wilson 95 percent interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def _linear_family(model) -> tuple[str, float] | None:
    """Return ("ols"|"ridge", penalty) when the final estimator is linear."""
    est = model
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            est = model.steps[-1][1]
    except ImportError:
        return None
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

    if isinstance(est, RidgeCV):
        return ("ridge", float(getattr(est, "alpha_", 1.0)))
    if isinstance(est, Ridge):
        return ("ridge", float(est.alpha))
    if isinstance(est, LinearRegression):
        return ("ols", 0.0)
    return None


def _subset_effect_linear(X, y, cols: tuple[int, ...], penalty: float,
                          sample_weight=None) -> float:
    """Coefficient of the treatment in the closed-form subset fit.

    cols indexes X; the treatment is cols[0]. An unpenalized intercept
    column reproduces sklearn's default fit_intercept=True. Sample
    weights enter through the sqrt-weight transform, which turns the
    weighted least squares objective into an ordinary one.
    """
    M = np.column_stack([X[:, list(cols)], np.ones(len(X))])
    yy = y
    if sample_weight is not None:
        w = np.sqrt(np.asarray(sample_weight, dtype=float))
        M = M * w[:, None]
        yy = y * w
    G = M.T @ M
    if penalty > 0:
        pen = np.eye(G.shape[0]) * penalty
        pen[-1, -1] = 0.0
        G = G + pen
    beta = np.linalg.solve(G, M.T @ yy)
    return float(beta[0])


def _fit_with_weights(model, X, y, sample_weight):
    """Fit supporting plain estimators and pipelines alike."""
    if sample_weight is None:
        model.fit(X, y)
        return model
    from sklearn.pipeline import Pipeline

    try:
        if isinstance(model, Pipeline):
            step = model.steps[-1][0]
            model.fit(X, y, **{f"{step}__sample_weight": sample_weight})
        else:
            model.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "the driver's model family does not accept sample_weight "
            f"({type(e).__name__}: {e}); pass sample_weight=None or a "
            "weighted family"
        ) from e
    return model


def specification_audit(
    driver,
    treatment: str,
    y,
    max_specs: int = 200,
    contrast: tuple[float, float] = (0.10, 0.90),
    n_anchor: int = 64,
    exact_threshold: int = 4096,
    max_flip_depth: int = 3,
    materiality: float | None = None,
    sample_weight=None,
    groups=None,
    seed: int = 0,
) -> dict:
    """Sign stability of the do-contrast for one treatment across specifications.

    A specification is the treatment plus a subset of the remaining
    features. Each specification is refit with the driver's own model
    family against y (the training outcome, supplied by the caller:
    specification search must fit to the real outcome, a distilled
    teacher would erase the noise that produces sign flips), then the
    audit measures the mean prediction change of moving the treatment
    from its p10 to its p90 value. Linear and ridge families take a
    closed-form path, which makes the curve exact.

    Args:
        driver: a CausalLens driver whose model is an sklearn estimator
            or pipeline (must support clone, fit, predict).
        treatment: feature to audit.
        y: outcome vector aligned with driver.background.
        max_specs: cap on sampled specifications when the lattice is
            too large for exact enumeration.
        contrast: quantile pair defining the intervention contrast.
        n_anchor: rows over which the contrast is averaged (refit path
            only; the linear path is exact).
        exact_threshold: enumerate the full lattice while 2^(p-1) is
            at or below this, otherwise sample.
        max_flip_depth: largest control subset size searched for the
            minimal flip set.
        materiality: effects within this absolute value count as
            negligible. Default max(0.02 * std(y), 2.5 times the chance
            contrast scale).
        sample_weight: per-row weights matching the production fit
            protocol; refits and the closed form both honor them.
        groups: labels per row (for example draft year). When given, the
            lean and full specifications are refit inside each group
            (effect heterogeneity: groups with at least 20 rows) and
            the per group effects are compared to the pooled signs, so
            a finding that only exists in the pooled sample is named
            by the cohort that reverses it.
        seed: reproducibility.

    Returns:
        Dict with the specification curve, flip counts, minimal flip
        sets, attribution share of the frozen model, optional fold
        stability block, and one of three verdicts: STABLE SIGN,
        FRAGILE SIGN, or NEGLIGIBLE.
    """
    feats = list(driver.feature_names)
    if treatment not in feats:
        raise ValueError(f"unknown treatment: {treatment}")
    bg = np.asarray(driver.background, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if len(y) != len(bg):
        raise ValueError(
            f"y has {len(y)} rows but the background has {len(bg)}; pass the "
            "training outcome aligned with driver.background"
        )
    if np.isnan(y).any():
        raise ValueError("y contains NaN")

    t_idx = feats.index(treatment)
    others = [i for i in range(len(feats)) if i != t_idx]
    if np.std(bg[:, t_idx]) < 1e-12:
        raise ValueError(f"treatment column {treatment} is constant")

    lo_q, hi_q = contrast
    lo = float(np.nanpercentile(bg[:, t_idx], lo_q * 100))
    hi = float(np.nanpercentile(bg[:, t_idx], hi_q * 100))
    if hi - lo < 1e-12:
        raise ValueError("treatment contrast is degenerate at these quantiles")

    # a pure noise feature still shows a chance contrast of about
    # sd_y/sd_t * (hi - lo) / sqrt(n); effects within 2.5 of that scale
    # (or 2 percent of sd_y, whichever is larger) are negligible, so
    # sign jitter from sampling noise never counts as a flip
    sd_y = float(np.std(y))
    sd_t = float(np.std(bg[:, t_idx]))
    chance = sd_y / max(sd_t, 1e-12) * (hi - lo) / np.sqrt(len(y))
    cutoff = (float(materiality) if materiality is not None
              else max(0.02 * sd_y, 2.5 * chance))
    rng = np.random.default_rng(seed)
    names = {i: feats[i] for i in range(len(feats))}

    # ---- effect engine: closed form for linear families, refit otherwise
    linear = _linear_family(driver.model)
    sub_X = bg[:, [t_idx] + others]
    effects: dict[frozenset, float] = {}

    if linear is not None and not np.isnan(sub_X).any():
        kind, penalty = linear
        # position of each original index inside sub_X = [t_idx] + others
        pos_of = {t_idx: 0}
        for j, i in enumerate(others):
            pos_of[i] = j + 1

        # for a linear model the do-contrast equals the coefficient times
        # the contrast width on every row, so the curve is the coefficient
        # distribution across subset refits, computed exactly
        def effect_of(orig_set: frozenset) -> float:
            if orig_set not in effects:
                cols = (0,) + tuple(sorted(pos_of[i] for i in orig_set))
                effects[orig_set] = _subset_effect_linear(
                    sub_X, y, cols, penalty, sample_weight
                ) * (hi - lo)
            return effects[orig_set]
    else:
        try:
            from sklearn.base import clone

            base = clone(driver.model)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                "specification_audit refits the driver's model family and "
                f"sklearn could not clone it ({type(e).__name__}: {e}); V1 "
                "supports sklearn estimators and pipelines"
            ) from e

        anchor_idx = rng.choice(len(bg), size=min(n_anchor, len(bg)),
                                replace=False)
        A = bg[anchor_idx].copy()
        B = bg[anchor_idx].copy()
        A[:, t_idx] = hi
        B[:, t_idx] = lo

        def effect_of(orig_set: frozenset) -> float:
            if orig_set not in effects:
                spec_cols = [t_idx] + sorted(orig_set)
                m = _fit_with_weights(clone(base), bg[:, spec_cols], y,
                                      sample_weight)
                effects[orig_set] = float(np.mean(m.predict(A[:, spec_cols])
                                                 - m.predict(B[:, spec_cols])))
            return effects[orig_set]

    # ---- walk the lattice
    n_lattice = 2 ** len(others)
    exact = n_lattice <= exact_threshold
    if exact:
        specs = [frozenset(c) for d in range(len(others) + 1)
                 for c in combinations(others, d)]
        lattice_mode = "exact"
    else:
        seen = {frozenset(), frozenset(others)}
        while len(seen) < min(max_specs, n_lattice):
            d = int(rng.integers(0, len(others) + 1))
            seen.add(frozenset(rng.choice(others, size=d, replace=False).tolist()))
        specs = list(seen)
        lattice_mode = "sampled"

    for s in specs:
        effect_of(s)

    curve = np.array([effects[s] for s in specs])
    lean = effects[frozenset()]
    full = effects[frozenset(others)]

    def sign_of(v: float) -> int:
        if v > cutoff:
            return 1
        if v < -cutoff:
            return -1
        return 0

    ref = sign_of(lean) or sign_of(full)
    signs = np.array([sign_of(v) for v in curve])
    n_flip = int(np.sum(signs == -ref)) if ref != 0 else 0
    n_neg = int(np.sum(signs == 0))
    frac_flip = n_flip / len(specs) if ref != 0 else 0.0

    # ---- minimal flip sets: smallest controls whose addition flips the sign
    minimal_flips: list[dict] = []
    if ref != 0:
        for d in range(1, min(max_flip_depth, len(others)) + 1):
            hits = []
            for c in combinations(others, d):
                v = effect_of(frozenset(c))
                if sign_of(v) == -ref:
                    hits.append({
                        "controls": sorted(names[i] for i in c),
                        "effect": round(v, 6),
                    })
                if len(hits) >= 5:
                    break
            if hits:
                minimal_flips = hits
                break

    # ---- attribution of the frozen model for the gap diagnostic
    attribution_share = None
    gap_flag = None
    try:
        from .attribution import shap_matrix

        sm = shap_matrix(driver, bg[:256])
        mean_abs = np.abs(sm).mean(axis=0)
        shares = mean_abs / (mean_abs.sum() + 1e-12)
        attribution_share = round(float(shares[t_idx]), 4)
    except Exception:  # noqa: BLE001 - attribution is optional context here
        pass
    collapse = sign_of(lean) != 0 and sign_of(full) == 0
    if attribution_share is not None and (n_flip > 0 or collapse) \
            and attribution_share >= 0.1:
        what = (f"sign flips in {n_flip} of {len(specs)} specifications"
                if n_flip > 0 else
                "effect collapses into the chance band under dense controls")
        gap_flag = (
            f"the frozen model still gives {treatment} a "
            f"{attribution_share:.0%} attribution share while its "
            f"intervention {what}: importance is bookkeeping, not behavior"
        )

    # ---- leave-one-group-out stability of the lean and full effects
    fold_stability = None
    if groups is not None:
        from sklearn.base import clone as _clone

        groups = np.asarray(groups)
        if len(groups) != len(bg):
            raise ValueError(
                f"groups has {len(groups)} rows but the background has "
                f"{len(bg)}"
            )
        sw = (None if sample_weight is None
              else np.asarray(sample_weight, dtype=float))
        folds = []
        for g in np.unique(groups):
            rows = groups == g
            if rows.sum() < 20:
                continue
            entry = {"group": str(g), "n_rows": int(rows.sum())}
            for name, cols in (("lean", ()), ("full", tuple(others))):
                spec_cols = [t_idx] + sorted(cols)
                m = _fit_with_weights(_clone(driver.model),
                                      bg[np.ix_(rows, spec_cols)], y[rows],
                                      None if sw is None else sw[rows])
                xg = bg[np.ix_(rows, spec_cols)]
                a_hi = xg.copy()
                a_hi[:, 0] = hi
                a_lo = xg.copy()
                a_lo[:, 0] = lo
                entry[f"effect_{name}"] = round(
                    float(np.mean(m.predict(a_hi) - m.predict(a_lo))), 6)
            folds.append(entry)
        ref_lean, ref_full = sign_of(lean), sign_of(full)
        # pooled signs can cancel when groups disagree, so the flip
        # reference is the dominant material fold sign, not the pool
        def _fold_flips(key):
            material = [f for f in folds if sign_of(f[key]) != 0]
            if not material:
                return [], 1.0
            pos = [f for f in material if sign_of(f[key]) == 1]
            neg = [f for f in material if sign_of(f[key]) == -1]
            if len(pos) != len(neg):
                # majority rules; ties fall through to magnitude
                dom_sign = 1 if len(pos) > len(neg) else -1
            else:
                dom_sign = sign_of(max(material, key=lambda f: abs(f[key]))[key])
            flips = [f["group"] for f in material
                     if sign_of(f[key]) == -dom_sign]
            kept = len(material) - len(flips)
            return flips, round(kept / max(len(material), 1), 3)

        lean_flips, lean_agree = _fold_flips("effect_lean")
        full_flips, full_agree = _fold_flips("effect_full")
        fold_stability = {
            "n_folds": len(folds),
            "folds": folds,
            "pooled_sign_lean": ref_lean,
            "pooled_sign_full": ref_full,
            "lean_sign_agreement": lean_agree,
            "full_sign_agreement": full_agree,
            "lean_flip_folds": lean_flips,
            "full_flip_folds": full_flips,
        }

    # ---- verdict
    if ref == 0 or n_neg >= 0.9 * len(specs):
        verdict = (
            f"NEGLIGIBLE: |effect| stays within +/-{cutoff:.4g} "
            f"(the chance scale for this n) in {n_neg} of {len(specs)} specifications"
        )
    elif sign_of(lean) != sign_of(full) and sign_of(lean) != 0 and sign_of(full) != 0:
        verdict = (
            f"FRAGILE SIGN: lean specification {lean:+.4f} vs full "
            f"specification {full:+.4f}; the sign flips with the control set"
        )
    elif frac_flip >= 0.05:
        verdict = (
            f"FRAGILE SIGN: sign flips in {n_flip} of {len(specs)} "
            f"specifications ({frac_flip:.0%})"
        )
    else:
        verdict = (
            f"STABLE SIGN: {ref:+d} in {len(specs) - n_flip - n_neg} of "
            f"{len(specs)} specifications above the chance band, none "
            f"opposite; lean {lean:+.4f}, full {full:+.4f}"
        )
        if collapse:
            verdict += (
                "; the effect survives only in sparse specifications and "
                "collapses into the chance band as controls enter"
            )
    if fold_stability and fold_stability["n_folds"] >= 2:
        flipped = (fold_stability["lean_flip_folds"]
                   or fold_stability["full_flip_folds"])
        if flipped:
            verdict += (f"; FOLD FRAGILE: the sign reverses inside "
                        f"group(s) {', '.join(map(str, flipped))} while "
                        "the pool averages over it")
        else:
            verdict += (f"; fold stable across {fold_stability['n_folds']} "
                        "groups")

    flip_specs = [
        {"controls": sorted(names[i] for i in s), "effect": round(effects[s], 6)}
        for s in specs if sign_of(effects[s]) == -ref and ref != 0
    ][:20]

    result = {
        "treatment": treatment,
        "n_specs": len(specs),
        "lattice": lattice_mode,
        "contrast": {"from_quantile": lo_q, "to_quantile": hi_q,
                     "from_value": round(lo, 6), "to_value": round(hi, 6)},
        "materiality": round(cutoff, 6),
        "effect_lean": round(lean, 6),
        "effect_full": round(full, 6),
        "median_effect": round(float(np.median(curve)), 6),
        "n_flip": n_flip,
        "frac_flip": round(frac_flip, 4),
        "n_negligible": n_neg,
        "reference_sign": ref,
        "engine": "closed_form_linear" if (linear is not None
                                           and not np.isnan(sub_X).any()) else "refit_clone",
        "sample_weighted": sample_weight is not None,
        "minimal_flip_sets": minimal_flips,
        "flip_specifications": flip_specs,
        "attribution_share": attribution_share,
        "attribution_gap": gap_flag,
        "spec_curve": [round(float(v), 6) for v in np.sort(curve)],
        "verdict": verdict,
        "method": "specification_curve_do_contrast",
        "reference": (
            "Simonsohn, Simmons and Nelson (2020) specification curve "
            "analysis; Steegen et al. (2016) multiverse analysis; "
            "Oster (2019) coefficient stability; here applied to do-contrasts "
            "of refitted black-box models"
        ),
        "note": (
            "each specification refits the driver's model family on y "
            "(caller supplied training outcome) and averages the p10 to p90 "
            "treatment contrast; linear families use the closed form so the "
            "curve is exact; refit specifications carry no sample weights; "
            "the negligible band is max(0.02 sd of y, 2.5 times the chance "
            "contrast sd_y/sd_t * width / sqrt(n))"
        ),
    }
    if lattice_mode == "sampled":
        ci = _wilson_interval(n_flip, len(specs))
        result["flip_ci95"] = [round(ci[0], 4), round(ci[1], 4)]
    if fold_stability is not None:
        result["fold_stability"] = fold_stability
    return result
