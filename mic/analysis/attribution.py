"""归因层：把 SHAP 适配成统一的归因证据，附带概念接地。"""
from __future__ import annotations

import numpy as np

from ..core.concepts import ConceptPack
from ..core.types import Attribution
from ..drivers.base import MicDriver

def _get_explainer(driver: MicDriver):
    """按模型类型构造 SHAP 解释器并缓存。

    缓存挂在 driver 实例上（而不是模块级 id 映射）：id 在对象回收后会被复用，
    模块级缓存会把别的模型的解释器错配给新 driver。
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
        except TypeError:  # ExactExplainer 等不认 max_evals
            explainer = shap.Explainer(
                lambda X: driver.predict(X), bg, feature_names=driver.feature_names
            )
    driver._mic_explainer = explainer
    return explainer


def shap_matrix(driver: MicDriver, X: np.ndarray) -> np.ndarray:
    """返回 (n, F) 的 SHAP 矩阵（正类方向）。"""
    explainer = _get_explainer(driver)
    X = np.asarray(X, dtype=float)
    if driver.kind in ("tree", "ensemble-tree"):
        sv = explainer.shap_values(X)
        if isinstance(sv, list):          # 多分类：取当前目标类
            sv = sv[min(driver.target_class, len(sv) - 1)]
        elif np.ndim(sv) == 3:            # 新版 shap: (n, F, C)
            sv = sv[:, :, min(driver.target_class, sv.shape[2] - 1)]
        return np.asarray(sv, dtype=float)
    ev = explainer(X)
    return np.asarray(ev.values, dtype=float)


def base_value(driver: MicDriver, x: np.ndarray) -> float:
    """该样本的 SHAP 基准值（背景平均预测），用于报告配平。"""
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
    """单样本归因：按 |SHAP| 降序，概念接地。"""
    sv = shap_matrix(driver, np.asarray(x, dtype=float).reshape(1, -1))[0]
    x = np.asarray(x, dtype=float).ravel()
    items = []
    for j, f in enumerate(driver.feature_names):
        base = f.split("@")[0]  # 时序展平名 "momentum@lag3" → 接地到 "momentum"
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
