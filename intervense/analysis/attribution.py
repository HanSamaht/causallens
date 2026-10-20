"""attributionlayer: SHAP attributionevidence,conceptgrounding."""
from __future__ import annotations

import numpy as np

from ..core.concepts import ConceptPack
from ..core.types import Attribution
from ..drivers.base import MicDriver

def _get_explainer(driver: MicDriver):
    """modeltype SHAP .

    in driver on(not id ):id inobjectby,
    model driver.
    """
    cached = getattr(driver, "_mic_explainer", None)
    if cached is not None:
        return cached

    import shap
    import warnings

    model = driver.model
    explainer = None
    if driver.kind == "ensemble-tree":
        from ..drivers.ensemble import _EnsembleExplainer

        return _EnsembleExplainer(driver.members, driver.weights)
    if driver.kind == "tree" or getattr(driver, "tree_compatible", False):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*shap values output has changed.*")
                explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None
    if explainer is None:
        bg = driver.background[: min(len(driver.background), 100)]
        n_feat = len(driver.feature_names)
        try:
            explainer = shap.Explainer(
                lambda X: driver.predict(X), bg, feature_names=driver.feature_names,
                max_evals=min(2 * n_feat + 1, 5000),
            )
        except TypeError: # ExactExplainer etcnot max_evals
            explainer = shap.Explainer(
                lambda X: driver.predict(X), bg, feature_names=driver.feature_names
            )
    driver._mic_explainer = explainer
    return explainer


def shap_matrix(driver: MicDriver, X: np.ndarray) -> np.ndarray:
    """return (n, F) SHAP (direction)."""
    explainer = _get_explainer(driver)
    X = np.asarray(X, dtype=float)
    if driver.kind in ("tree", "ensemble-tree"):
        sv = explainer.shap_values(X)
        if isinstance(sv, list): # :when
            sv = sv[min(driver.target_class, len(sv) - 1)]
        elif np.ndim(sv) == 3: # shap: (n, F, C)
            sv = sv[:, :, min(driver.target_class, sv.shape[2] - 1)]
        return np.asarray(sv, dtype=float)
    ev = explainer(X)
    return np.asarray(ev.values, dtype=float)


def base_value(driver: MicDriver, x: np.ndarray) -> float:
    """thesample SHAP value(backgroundprediction),report."""
    explainer = _get_explainer(driver)
    if driver.kind in ("tree", "ensemble-tree") or getattr(driver, "tree_compatible", False):
        try:
            return float(np.ravel(explainer.expected_value)[0])
        except Exception:
            return float(driver.predict(driver.background[:100]).mean())
    return float(driver.predict(driver.background[:100]).mean())


def explain_sample(
    driver: MicDriver,
    x: np.ndarray,
    pack: ConceptPack | None = None,
) -> list[Attribution]:
    """sampleattribution: |SHAP| ,conceptgrounding."""
    sv = shap_matrix(driver, np.asarray(x, dtype=float).reshape(1, -1))[0]
    x = np.asarray(x, dtype=float).ravel()
    items = []
    for j, f in enumerate(driver.feature_names):
        base = f.split("@")[0] # "momentum@lag3" -> groundingto "momentum"
        entry = pack.ground(base) if pack else None
        items.append(
            Attribution(
                feature=f,
                shap_value=float(sv[j]),
                feature_value=float(x[j]),
                concept=entry.concept if entry else f,
                role=entry.role if entry else "normal",
            )
        )
    items.sort(key=lambda a: abs(a.shap_value), reverse=True)
    return items
