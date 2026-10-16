"""DecisionLogDriver：审讯没有 predict 函数的系统（决策日志/智能体轨迹）。

认识论分层（诚实纪律的落点）：
  - 可重跑系统（在线 agent/模拟器）→ 直接干预 = 真因果（另一路，待接）
  - 不可重跑系统（人类政策/历史日志/他人黑盒）→ 本驱动：行为克隆替代模型
    + 保真度标签。归因/干预/反事实/因果图全部照常运行——但所有结论
    自动携带"替代模型动作一致率 XX%"的边界声明。

因果证据是关于替代模型的；保真度是它与原系统行为的吻合度，
是每条推演结论可信度的上限。
"""
from __future__ import annotations

import numpy as np

from .tree import TreeDriver


class DecisionLogDriver(TreeDriver):
    kind = "decision-log"
    tree_compatible = True  # attribution 层据此走 TreeExplainer 快路径

    def __init__(
        self,
        logs_X,
        logs_y,
        feature_names: list[str],
        name: str = "decision_system",
        holdout_frac: float = 0.2,
        seed: int = 0,
    ):
        """从 (情境, 动作) 日志训练替代模型并自评保真度。"""
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        X = np.asarray(logs_X, dtype=float)
        y = np.asarray(logs_y)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=holdout_frac, random_state=seed, stratify=y)

        try:  # 首选 LightGBM
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
                         "note": "替代模型在留出日志上的动作一致率 = 结论可信度上限"}

        super().__init__(surrogate, feature_names, X_te, name=name)

    def describe(self) -> dict:
        d = super().describe()
        d["surrogate_fidelity"] = self.fidelity
        d["note"] = (
            "审讯对象是决策日志的替代模型（行为克隆）。所有结论附带保真度边界："
            f"动作一致率 {self.fidelity['action_agreement']:.1%}"
        )
        return d
