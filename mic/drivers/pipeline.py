"""Pipeline 驱动器：加载 sklearn Pipeline 对象（预处理器 + 模型的组合）。

真实项目中模型常被包在 Pipeline 里（StandardScaler + PCA + RandomForest 等），
Pipeline 对外只有 predict/predict_proba——和我们 GenericSklearnDriver 同构。
关键问题：Pipeline 没有 n_features_in_（那是末端 estimator 的属性）。

方案：从 Pipeline.steps 的第一个预处理器推断输入宽度；如果没有预处理器，
从最后一步的 estimator 拿。背景数据直接喂 Pipeline 即可（Pipeline 自己做预处理）。
"""
from __future__ import annotations

import numpy as np

from .sklearn_generic import GenericSklearnDriver


class PipelineDriver(GenericSklearnDriver):
    """sklearn Pipeline / make_pipeline 产物的驱动器。"""
    kind = "pipeline"

    def __init__(self, pipeline, feature_names, background, name="pipeline_model",
                 semantics=None):
        self.pipeline = pipeline
        # 推断输入宽度：优先第一步预处理器，否则试末端 estimator
        n_feat = None
        for step_name, step in getattr(pipeline, "steps", []):
            n = getattr(step, "n_features_in_", None)
            if n is not None:
                n_feat = int(n)
                break
        if n_feat is None:
            # 兜底：试一次 predict 看报错信息里的维度
            n_feat = len(feature_names) if feature_names else 10

        self.pipeline_input_width = n_feat
        super().__init__(pipeline, feature_names, background, name=name, semantics=semantics)

    def describe(self) -> dict:
        d = super().describe()
        steps = []
        for step_name, step in getattr(self.pipeline, "steps", []):
            steps.append(f"{step_name}: {type(step).__name__}")
        d["pipeline_steps"] = steps
        d["pipeline_input_width"] = self.pipeline_input_width
        d["note"] = f"Pipeline 驱动：{len(steps)} 步 ({' → '.join(steps)})"
        return d
