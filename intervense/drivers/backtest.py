"""BacktestDriver: the interrogation target is the backtest engine, not an ML model.

"feature" means execution parameters and hypotheses (fill semantics,
slippage, position sizing and friends) and "prediction" means the
profit factor, with 1.0 as the natural counterfactual threshold.

Interrogation still applies:
  - attribution: SHAP over execution parameters of the replay
  - intervention: do(fill semantics := market) changes commission to
    spread to slippage in one switch
  - counterfactual: which parameters push PF back above 1
  - causal graph: execution parameters times fill semantics
  - : attribution vs intervention(notonenginebehavior)
"""
from __future__ import annotations

import numpy as np

from ..backtest.engine import BTParams, run_backtest
from .base import MicDriver

PARAM_NAMES = ["fill_market", "stop_pct", "tp_pct", "fee_bps", "slippage_bps", "entry_every"]


class BacktestDriver(MicDriver):
    """Backtestdriver."""
    kind = "backtest"
    prediction_semantics = "score" # PF:1.0 =

    def __init__(self, ohlc: np.ndarray, feature_names=None, background=None, name="backtest_engine"):
        """Init."""
        self.ohlc = np.asarray(ohlc, dtype=float)
        feature_names = list(feature_names or PARAM_NAMES)
        if background is None:
            # background:parameterexecutehypothesis(fill semantics)
            rng = np.random.default_rng(0)
            n = 40
            bg = np.column_stack(
                [
                    rng.choice([0.0, 1.0], size=n), # fill semantics
                    rng.uniform(0.02, 0.06, size=n), #
                    rng.uniform(0.08, 0.20, size=n), #
                    rng.uniform(0.0, 8.0, size=n), # bps
                    rng.uniform(0.0, 30.0, size=n), # bps
                    rng.choice([6, 12, 24, 48], size=n).astype(float), #
                ]
            )
            bg[:2] = [0.0, 0.03, 0.15, 2.0, 0.0, 12.0] # :
            bg[2:4] = [1.0, 0.03, 0.15, 2.0, 0.0, 12.0] # :
            background = bg
        super().__init__(self.ohlc, feature_names, background, name=name)

    # ---- "":parameter -> PF -------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.empty(len(X))
        for i, row in enumerate(X):
            params = BTParams(**dict(zip(PARAM_NAMES, row)))
            out[i] = run_backtest(self.ohlc, params)["pf"]
        return out

    def run(self, **params) -> dict:
        """parameterbacktest,returnall()."""
        base = dict(zip(PARAM_NAMES, self.background[0]))
        merged = {**{k: base[k] for k in PARAM_NAMES}, **params}
        return run_backtest(self.ohlc, BTParams(**merged))

    def describe(self) -> dict:
        """Describe."""
        d = super().describe()
        d["note"] = (
            "interrogationobjectisbacktestengine:feature=executeparameterhypothesis,prediction=PF(1.0as)."
            "fill_market: 0=ladder price accounting() 1=market fill()"
        )
        d["n_bars"] = int(len(self.ohlc))
        return d
