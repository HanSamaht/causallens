"""ONNX driver:safeloading.

pickle loadingmodeletcexecute(not .pkl isexecute);
ONNX iscompute,loadingnotexecute——model:

    : skl2onnx.convert_sklearn(model) -> model.onnx
    interrogation: python -m mic report --model model.onnx --background bg.csv
"""
from __future__ import annotations

import numpy as np

from .base import MicDriver


class OnnxDriver(MicDriver):
    kind = "onnx"
    prediction_semantics = "probability"

    def __init__(self, onnx_path: str, feature_names, background, name="onnx_model", semantics=None):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_meta = self.sess.get_outputs()[0]
        # skl2onnx output label+probabilities probabilities
        self.output_names = [o.name for o in self.sess.get_outputs()]
        has_proba = any("probab" in n.lower() for n in self.output_names)
        if semantics is None:
            semantics = "probability" if has_proba else "score"
        self.prediction_semantics = semantics
        self._proba_name = next((n for n in self.output_names if "probab" in n.lower()), None)
        super().__init__(self.sess, feature_names, background, name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        flat = X.reshape(len(X), -1)
        if self._proba_name is not None:
            outs = self.sess.run([self._proba_name], {self.input_name: flat})
            proba = outs[0]
            if isinstance(proba, list) or (hasattr(proba, "dtype") == False):
                # skl2onnx ZipMap output:[{0: p0, 1: p1}, ...]
                return np.array([float(d.get(1, d.get("1", 0.0))) for d in proba])
            proba = np.asarray(proba)
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()
        outs = self.sess.run([self.output_names[0]], {self.input_name: flat})
        return np.asarray(outs[0], dtype=float).ravel()
