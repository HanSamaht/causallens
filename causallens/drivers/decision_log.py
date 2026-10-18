"""DecisionLogDriver:interrogationno predict (log/).

layer():
  - (in agent/)-> directintervention = causal(,)
  - not(/historylog/)-> :behaviormodel
    Plus the full analysis stack, attribution, intervention, counterfactual,
    causal graph, all run on the log, so conclusions carry evidence
    "modelconsistent XX%".

causalevidenceisaboutmodel;isandbehavior,
iseveryreasoningconclusionon.
"""
from __future__ import annotations

import numpy as np

from .tree import TreeDriver


class DecisionLogDriver(TreeDriver):
    """Decisionlogdriver."""
    kind = "decision-log"
    tree_compatible = True # attribution layerthis TreeExplainer

    def __init__(
        self,
        logs_X,
        logs_y,
        feature_names: list[str],
        name: str = "decision_system",
        holdout_frac: float = 0.2,
        seed: int = 0,
    ):
        """from (, ) logtrainingmodel."""
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        X = np.asarray(logs_X, dtype=float)
        y = np.asarray(logs_y)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=holdout_frac, random_state=seed, stratify=y)

        try: # LightGBM
            import lightgbm as lgb

            surrogate = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.06, num_leaves=31,
                                           random_state=seed, verbose=-1)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            surrogate = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed)
        surrogate.fit(X_tr, y_tr)

        acc = float(accuracy_score(y_te, surrogate.predict(X_te)))
        try:
            auc = float(roc_auc_score(y_te, surrogate.predict_proba(X_te)[:, 1]))
        except Exception:
            auc = None
        self.fidelity = {"action_agreement": round(acc, 4), "auc": None if auc is None else round(auc, 4),
                         "n_logs": int(len(X)), "n_holdout": int(len(X_te)),
                         "note": "modelinlogonconsistent = conclusionon"}

        super().__init__(surrogate, feature_names, X_te, name=name)

    def describe(self) -> dict:
        """Describe."""
        d = super().describe()
        d["surrogate_fidelity"] = self.fidelity
        d["note"] = (
            "interrogationobjectislogmodel(behavior).allconclusion:"
            f"consistent {self.fidelity['action_agreement']:.1%}"
        )
        return d
