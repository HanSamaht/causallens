"""模型加载兼容层：处理第三方 pkl 的版本不匹配、格式差异。

三类问题的解决方案：
  1. pickle protocol 不兼容 → joblib 兜底 / 指定 encoding
  2. sklearn 版本差异 → 版本警告忽略 + 模块重映射
  3. torch 架构推断失败 → 自动试多个 head 层名变体

使用方式：
    model = load_model_anyhow(path)   # 自动选择最佳加载策略
"""
from __future__ import annotations

import pickle
import warnings


def load_model_anyhow(path: str):
    """尝试多种策略加载模型文件，返回 (model, method) 或抛出最后的异常。"""
    errors = []

    # 策略 1: 标准 pickle
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
            return pickle.load(open(path, "rb")), "pickle"
    except Exception as e:
        errors.append(("pickle", type(e).__name__, str(e)[:80]))

    # 策略 2: joblib（不同序列化格式，兼容部分 pickle 不认的 numpy 数组嵌入）
    try:
        import joblib

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
            return joblib.load(path), "joblib"
    except Exception as e:
        errors.append(("joblib", type(e).__name__, str(e)[:80]))

    # 策略 3: pickle + latin1 encoding（跨 Python 2/3 的旧文件）
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return pickle.load(open(path, "rb"), encoding="latin1"), "pickle_latin1"
    except Exception as e:
        errors.append(("pickle_latin1", type(e).__name__, str(e)[:80]))

    # 策略 4: torch.load（如果路径以 .pt/.pth 结尾）
    if path.endswith((".pt", ".pth", ".ckpt")):
        try:
            import torch

            obj = torch.load(path, map_location="cpu", weights_only=False)
            sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
            if hasattr(sd, "state_dict"):
                return sd, "torch"
            return obj, "torch_raw"
        except Exception as e:
            errors.append(("torch", type(e).__name__, str(e)[:80]))

    # 全部失败——报告每种策略的报错
    lines = [f"  {method}: {err_type}: {msg}" for method, err_type, msg in errors]
    raise RuntimeError(
        f"模型加载全部失败 ({len(errors)} 种策略尝试):\n" + "\n".join(lines)
    )


def load_pipeline_parts(path: str):
    """尝试加载 Pipeline 并拆解出 (pipeline, estimator, preprocessor_steps)。"""
    model, method = load_model_anyhow(path)

    if hasattr(model, "steps"):
        # 是 Pipeline
        estimator = model.steps[-1][1]
        preprocessors = [(name, step) for name, step in model.steps[:-1]]
        return model, estimator, preprocessors
    elif hasattr(model, "predict"):
        # 裸模型（可能是 Pipeline 里的 estimator 单独存了）
        return model, model, []
    else:
        raise ValueError(f"加载的对象既不是 Pipeline 也没有 predict 方法: {type(model)}")


def infer_torch_head_name(state_dict: dict) -> str:
    """从 state_dict 推断输出头层的属性名。

    常见变体: fc, head, classifier, output, output_layer, linear, mlp, dense
    """
    candidates = ["fc", "head", "classifier", "output", "output_layer",
                  "linear", "mlp", "dense", "prediction_head", "projection"]
    all_keys = list(state_dict.keys())

    for name in candidates:
        # 直接匹配
        if f"{name}.weight" in all_keys:
            return name
        # 嵌套匹配（如 model.fc.weight）
        for key in all_keys:
            if key.endswith(f".{name}.weight") and "weight" in key:
                # 返回完整路径（去掉 .weight）
                return key.rsplit(".weight", 1)[0]

    # 兜底：找最后一个带 .weight 的键（通常是输出层）
    weight_keys = [k for k in all_keys if k.endswith(".weight")]
    if weight_keys:
        return weight_keys[-1].rsplit(".weight", 1)[0]

    return None
