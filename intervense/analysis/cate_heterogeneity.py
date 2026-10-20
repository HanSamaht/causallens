"""CATE heterogeneity: who does the effect work for.

An average effect answers whether a treatment matters. Deployment
decisions need the second question: does it matter equally for
everyone. This module answers it with the framework's contrast
machinery: it fits the driver's own model family, computes the
per-row do-contrast of the treatment (p10 to p90), and mines the
distribution of row effects with a shallow subgroup tree.

Verdicts:

  HETEROGENEOUS: the strongest and weakest subgroups differ by a
  margin whose bootstrap interval excludes zero and exceeds the
  chance scale
  HOMOGENEOUS: the spread is indistinguishable from resampling noise

The subgroup tree is restricted to depth two and a minimum leaf size,
so every reported subgroup is a rule a decision maker can read.
"""
from __future__ import annotations

import numpy as np


def _fit_model(driver, sample_weight):
    from sklearn.base import clone

    try:
        base = clone(driver.model)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            "cate_heterogeneity refits the driver's model family and "
            f"sklearn could not clone it ({type(e).__name__}: {e}); pass a "
            "cloneable sklearn family or use estimator_factory"
        ) from e
    X = np.asarray(driver.background, dtype=float)
    y_model = None
    return base, X


def cate_heterogeneity(
    driver,
    treatment: str,
    y,
    sample_weight=None,
    quantiles: tuple[float, float] = (0.10, 0.90),
    max_depth: int = 2,
    min_leaf: int = 15,
    n_bootstrap: int = 30,
    seed: int = 0,
) -> dict:
    """Per-row treatment contrasts and the subgroups they separate.

    Args:
        driver: any CausalLens driver whose model is a cloneable
            sklearn estimator or pipeline.
        treatment: feature whose effect heterogeneity is mapped.
        y: outcome vector aligned with driver.background.
        sample_weight: per-row weights honored in every refit.
        quantiles: contrast pair defining each row's effect.
        max_depth: subgroup tree depth, two keeps rules readable.
        min_leaf: minimum rows per subgroup.
        n_bootstrap: resamples for subgroup stability, 0 disables.
        seed: reproducibility.

    Returns:
        Dict with per-row effect statistics, the subgroup table,
        the strongest-vs-weakest contrast with its bootstrap
        interval, and one of two verdicts.
    """
    from sklearn.base import clone
    from sklearn.tree import DecisionTreeRegressor

    feats = list(driver.feature_names)
    if treatment not in feats:
        raise ValueError(f"unknown treatment: {treatment}")
    X = np.asarray(driver.background, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if len(y) != len(X):
        raise ValueError(
            f"y has {len(y)} rows but the background has {len(X)}")
    t_idx = feats.index(treatment)
    lo = float(np.nanpercentile(X[:, t_idx], quantiles[0] * 100))
    hi = float(np.nanpercentile(X[:, t_idx], quantiles[1] * 100))
    if hi - lo < 1e-12:
        raise ValueError("treatment contrast is degenerate")

    sw = (None if sample_weight is None
          else np.asarray(sample_weight, dtype=float))
    rng = np.random.default_rng(seed)

    def row_effects(idx):
        from sklearn.base import clone as _clone
        m = _clone(driver.model)
        if sw is None:
            m.fit(X[idx], y[idx])
        else:
            m.fit(X[idx], y[idx], sample_weight=sw[idx])
        A, B = X[idx].copy(), X[idx].copy()
        A[:, t_idx] = hi
        B[:, t_idx] = lo
        return m.predict(A) - m.predict(B), m

    def subgroup_table(effects, idx):
        tree = DecisionTreeRegressor(max_depth=max_depth,
                                     min_samples_leaf=min_leaf,
                                     random_state=0)
        tree.fit(X[idx], effects)
        leaves = np.where(tree.tree_.children_left == -1)[0]
        leaf_id = tree.apply(X[idx])
        rows = []
        for leaf in leaves:
            mask = leaf_id == leaf
            if mask.sum() < min_leaf:
                continue
            rows.append({
                "rule": _leaf_rule(tree, leaf, feats),
                "n": int(mask.sum()),
                "mean_effect": float(np.mean(effects[mask])),
            })
        rows.sort(key=lambda r: r["mean_effect"], reverse=True)
        return rows, tree

    effects_all, model = row_effects(np.arange(len(X)))
    spread = float(np.percentile(effects_all, 90) - np.percentile(effects_all, 10))
    subgroups, tree = subgroup_table(effects_all, np.arange(len(X)))

    top_split = feats[tree.tree_.feature[0]] if len(subgroups) > 1 else None
    contrast = None
    contrast_ci = None
    split_stability = None
    if len(subgroups) >= 2:
        strong, weak = subgroups[0], subgroups[-1]
        contrast = strong["mean_effect"] - weak["mean_effect"]
        if n_bootstrap and n_bootstrap > 0:
            boots = []
            top_features = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(X), len(X))
                try:
                    e_b, _m = row_effects(idx)
                    rows_b, tree_b = subgroup_table(e_b, idx)
                    if len(rows_b) >= 2:
                        boots.append(rows_b[0]["mean_effect"]
                                     - rows_b[-1]["mean_effect"])
                    if tree_b.tree_.node_count > 1:
                        top_features.append(
                            feats[tree_b.tree_.feature[0]])
                except Exception:  # noqa: BLE001 - degenerate resample
                    continue
            if boots:
                contrast_ci = [float(np.percentile(boots, 2.5)),
                               float(np.percentile(boots, 97.5))]
            if top_features:
                from collections import Counter
                counts = Counter(top_features)
                split_stability = round(
                    counts.most_common(1)[0][1] / len(top_features), 2)

    # chance scale for the spread, same spirit as the specification audit
    sd_y = float(np.std(y))
    chance = 2.5 * sd_y * (hi - lo) / max(float(np.std(X[:, t_idx])), 1e-12) \
        / np.sqrt(len(X))

    heterogeneous = (
        contrast_ci is not None
        and contrast_ci[0] > max(chance, 0)
        and spread > 2 * max(chance, 0)
    )
    if heterogeneous:
        verdict = (
            f"HETEROGENEOUS: strongest subgroup "
            f"{subgroups[0]['mean_effect']:+.4f} ({subgroups[0]['rule']}) "
            f"vs weakest {subgroups[-1]['mean_effect']:+.4f} "
            f"({subgroups[-1]['rule']}), contrast {contrast:+.4f} "
            f"[{contrast_ci[0]:+.4f}, {contrast_ci[1]:+.4f}]"
        )
    else:
        verdict = (
            f"HOMOGENEOUS: 10-90 spread {spread:.4f} is within "
            f"{2:.0f}x the chance scale ({chance:.4f}); treat the "
            "average effect as the effect"
        )

    return {
        "treatment": treatment,
        "n_rows": len(X),
        "contrast": {"from": round(lo, 6), "to": round(hi, 6)},
        "mean_effect": float(np.mean(effects_all)),
        "spread_p10_p90": round(spread, 6),
        "subgroups": subgroups,
        "top_split_feature": top_split,
        "top_split_stability": split_stability,
        "subgroup_contrast": round(contrast, 6) if contrast is not None else None,
        "subgroup_contrast_ci95": ([round(v, 6) for v in contrast_ci]
                                   if contrast_ci else None),
        "chance_scale": round(float(chance), 6),
        "verdict": verdict,
        "method": "per_row_contrast_subgroup_tree",
        "reference": (
            "Athey and Imbens (2016) recursive partitioning for "
            "heterogeneous effects; here with do-contrasts from the "
            "driver's own family and bootstrap subgroup stability"
        ),
        "note": (
            "row effects are p10 to p90 contrasts from one fitted "
            "model; subgroups are depth-limited tree rules with a "
            "minimum leaf size; stability is the fraction of bootstrap "
            "refits that pick the same top split feature"
        ),
    }


def _leaf_rule(tree, leaf, feats) -> str:
    """Human readable rule for one leaf of a fitted tree."""
    def find(node, rules):
        if node == leaf:
            return True
        f = tree.tree_.feature[node]
        thr = tree.tree_.threshold[node]
        for child, op in ((tree.tree_.children_left[node], "<="),
                          (tree.tree_.children_right[node], ">")):
            if child == -1:
                continue
            rules.append(f"{feats[f]} {op} {thr:.3f}")
            if find(child, rules):
                return True
            rules.pop()
        return False

    rules = []
    find(0, rules)
    return " AND ".join(rules) if rules else "all rows"
