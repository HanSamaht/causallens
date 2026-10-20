"""andidentify: mic model + feature + background,driver."""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import MicDriver
from .pipeline import PipelineDriver
from .sklearn_generic import GenericSklearnDriver
from .text import TextDriver
from .tree import TreeDriver


def _is_torch_module(model: Any) -> bool:
    """True for any nn.Module, including subclasses defined in user code."""
    try:
        import torch
    except ImportError:
        return False
    return isinstance(model, torch.nn.Module)


def _looks_like_tree_model(model: Any) -> bool:
    """LightGBM / XGBoost / sklearn / all."""
    if hasattr(model, "booster_") and hasattr(model, "feature_importances_"):
        return True # LGBMClassifier
    if type(model).__name__ in {"Booster"}:
        return True # xgboost.Booster
    if hasattr(model, "estimators_") and hasattr(model, "feature_importances_"):
        return True # RandomForest / GradientBoosting
    if hasattr(model, "get_booster"):
        return True # XGBClassifier
    if hasattr(model, "tree_") and hasattr(model, "predict_proba"):
        return True # DecisionTree
    return False


def detect_driver(
    model: Any,
    feature_names: list[str],
    background: np.ndarray,
    name: str = "model",
    semantics: str | None = None,
    model_path: str | None = None,
) -> MicDriver:
    """driver.identify:onnx -> torch -> tree -> generic sklearn."""
    if model_path and str(model_path).lower().endswith(".onnx"):
        try:
            from .onnx_model import OnnxDriver

            return OnnxDriver(model_path, feature_names, background, name=name, semantics=semantics)
        except ImportError as e:
            raise ImportError("loading ONNX require onnxruntime:pip install onnxruntime") from e

    # torch.nn.Module, including user defined subclasses: the class
    # module name is not always torch.*, so check the type hierarchy
    if _is_torch_module(model):
        from .torch_model import TorchDriver

        return TorchDriver(model, feature_names, background, name=name)

    # Pipeline object:(,predict )
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


__all__ = ["MicDriver", "TreeDriver", "GenericSklearnDriver", "TextDriver", "detect_driver"]
