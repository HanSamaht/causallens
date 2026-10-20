"""driver:model(prediction,prediction……).

:"".background (n, T, F) byas (n, T*F),
featureas `{feature}@lag{k}`(lag0 = most).thisattribution,intervention,
counterfactuals, causal graphs, and the rest of the analysis stack;
background stays (B, T, F).

 TorchDriver:activation,activation patching(unit),
Steering and attribution are input level only.

intervention:
  - intervene({"momentum": v}) intervenes on momentum at every lag
  - intervene({"momentum@lag0": v}) intervenes only on the newest step
  - intervene_window(lag_slice, ...) intervenes on a chosen lag range
"""
from __future__ import annotations

import numpy as np

from .torch_model import TorchDriver


def make_flat_names(feature_names: list[str], seq_len: int) -> list[str]:
    """Make flat names."""
    return [f"{f}@lag{seq_len - 1 - t}" for t in range(seq_len) for f in feature_names]


class TimeSeriesDriver(TorchDriver):
    """Timeseriesdriver."""
    kind = "timeseries"
    prediction_semantics = "score" # default(prediction);

    def __init__(self, model, feature_names, background, name="ts_model", seq_len: int | None = None):
        """Init."""
        bg = np.asarray(background, dtype=float)
        if bg.ndim == 3:
            seq_len = seq_len or bg.shape[1]
            self.n_features_raw = bg.shape[2]
            bg = bg.reshape(len(bg), -1)
        else:
            assert seq_len is not None, "backgroundas 2D must seq_len"
            self.n_features_raw = len(feature_names)
        self.seq_len = seq_len
        self.base_feature_names = list(feature_names)
        try:
            import torch

            self.torch = torch
        except ImportError as e:
            raise ImportError("TimeSeriesDriver require torch:pip install torch") from e
        super().__init__(model, make_flat_names(feature_names, seq_len), bg, name=name)

    # ---- input:flat -> (B, T, F) ------------------------------------
    def _to_tensor(self, X: np.ndarray):
        """To tensor."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = len(X)
        return self.torch.tensor(
            X.reshape(n, self.seq_len, self.n_features_raw), dtype=self.torch.float32
        )

    # ---- intervention -----------------------------------------------------
    def intervene(self, x: np.ndarray, assignments: dict[str, float]) -> np.ndarray:
        """feature(not @lag)-> intervention; @lag -> intervention."""
        expanded: dict[str, float] = {}
        for f, v in assignments.items():
            if "@" in f:
                expanded[f] = v
            else:
                for lag in range(self.seq_len):
                    expanded[f"{f}@lag{lag}"] = v
        return super().intervene(x, expanded)

    def intervene_window(self, x: np.ndarray, lag_from: int, lag_to: int, assignments: dict[str, float]) -> np.ndarray:
        """onlyintervention [lag_from, lag_to] (;lag0=most)."""
        expanded = {}
        for f, v in assignments.items():
            for lag in range(lag_from, lag_to + 1):
                expanded[f"{f}@lag{lag}"] = v
        return self.intervene(x, expanded)

    def window_attribution(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """×input -> (T, F) attribution + feature."""
        t = self.torch
        x_flat = np.asarray(x, dtype=float).ravel()
        xt = t.tensor(x_flat.reshape(1, self.seq_len, self.n_features_raw), dtype=t.float32)
        xt.requires_grad_(True)
        out = self.model(xt).squeeze()
        out.backward()
        grad = xt.grad.detach().cpu().numpy()
        heat = grad[0] * xt.detach().cpu().numpy()[0]
        return heat, self.feature_names

    # ---- ------------------------------------------------------------
    def describe(self) -> dict:
        """Describe."""
        d = super().describe()
        d["seq_len"] = self.seq_len
        d["base_features"] = self.base_feature_names
        d["note"] = "featureas {feature}@lag{k},lag0=most;activation patching h_T/unit"
        return d
