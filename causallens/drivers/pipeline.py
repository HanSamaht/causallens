"""Pipeline driver:loading sklearn Pipeline object( + model).

projectinmodelbyin Pipeline (StandardScaler + PCA + RandomForest etc),
Pipeline onlyhas predict/predict_proba——and GenericSklearnDriver .
problem:Pipeline no n_features_in_(thatis estimator ).

:from Pipeline.steps inferinputwidth;ifno,
fromfinally estimator .backgrounddirect Pipeline (Pipeline ).
"""
from __future__ import annotations

import numpy as np

from .sklearn_generic import GenericSklearnDriver


class PipelineDriver(GenericSklearnDriver):
    """sklearn Pipeline / make_pipeline driver."""
    kind = "pipeline"

    def __init__(self, pipeline, feature_names, background, name="pipeline_model",
                 semantics=None):
        """Init."""
        self.pipeline = pipeline
        # inferinputwidth:, estimator
        n_feat = None
        for step_name, step in getattr(pipeline, "steps", []):
            n = getattr(step, "n_features_in_", None)
            if n is not None:
                n_feat = int(n)
                break
        if n_feat is None:
            # : predict
            n_feat = len(feature_names) if feature_names else 10

        self.pipeline_input_width = n_feat
        super().__init__(pipeline, feature_names, background, name=name, semantics=semantics)

    def describe(self) -> dict:
        """Describe."""
        d = super().describe()
        steps = []
        for step_name, step in getattr(self.pipeline, "steps", []):
            steps.append(f"{step_name}: {type(step).__name__}")
        d["pipeline_steps"] = steps
        d["pipeline_input_width"] = self.pipeline_input_width
        d["note"] = f"Pipeline :{len(steps)} ({' -> '.join(steps)})"
        return d
