"""驱动注册与自动识别：给 mic 一个模型 + 特征名 + 背景数据，它自己挑驱动器。"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import MicDriver
from .pipeline import PipelineDriver
from .sklearn_generic import GenericSklearnDriver
from .tree import TreeDriver


def _looks_like_tree_model(model: Any) -> bool:
    """LightGBM / XGBoost / sklearn 树集成 / 单棵树都算。"""
    if hasattr(model, "booster_") and hasattr(model, "feature_importances_"):
        return True  # LGBMClassifier
    if type(model).__name__ in {"Booster"}:
        return True  # xgboost.Booster
    if hasattr(model, "estimators_") and hasattr(model, "feature_importances_"):
        return True  # RandomForest / GradientBoosting
    if hasattr(model, "get_booster"):
        return True  # XGBClassifier
    if hasattr(model, "tree_") and hasattr(model, "predict_proba"):
        return True  # DecisionTree
    return False


def detect_driver(
    model: Any,
    feature_names: list[str],
    background: np.ndarray,
    name: str = "model",
    semantics: str | None = None,
    model_path: str | None = None,
) -> MicDriver:
    """自动选择合适的驱动器。识别顺序：onnx 路径 → torch → tree → generic sklearn。"""
    if model_path and str(model_path).lower().endswith(".onnx"):
        try:
            from .onnx_model import OnnxDriver

            return OnnxDriver(model_path, feature_names, background, name=name, semantics=semantics)
        except ImportError as e:
            raise ImportError("加载 ONNX 需要 onnxruntime：pip install onnxruntime") from e

    # torch.nn.Module
    mod = type(model).__module__ or ""
    if mod.startswith("torch"):
        from .torch_model import TorchDriver

        return TorchDriver(model, feature_names, background, name=name)

    # Pipeline 对象：专用驱动（内含预处理器，predict 自动走完整管线）
    if hasattr(model, "steps") and hasattr(model, "predict"):
        from .pipeline import PipelineDriver

        return PipelineDriver(model, feature_names, background, name=name, semantics=semantics)

    if _looks_like_tree_model(model):
        d = TreeDriver(model, feature_names, background, name=name)
    else:
        d = GenericSklearnDriver(model, feature_names, background, name=name, semantics=semantics)
    if semantics is not None:
        d.prediction_semantics = semantics
    return d


__all__ = ["MicDriver", "TreeDriver", "GenericSklearnDriver", "detect_driver"]
