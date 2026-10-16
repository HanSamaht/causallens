"""偷看检测（look-ahead）：哪个特征在用"决策时还不存在"的信息。

方法（黑盒、干预式，对齐社区 lookahead-bias 检测器的三要素：
哪个列、看多远、值多少）：

  IC 滞后对照 —— 对每个特征比较两种对齐下的信息系数：
    IC_now  = corr( f_t,   未来收益 y_t )   特征"现在"就有的信息
    IC_prev = corr( f_{t-k}, 未来收益 y_t )  特征晚 k 期才拿到时的信息
  合法特征：IC 平滑衰减（晚一期没那么准，但不塌方）
  偷看特征：IC_now 高、IC_prev 断崖 → 它的信息只在"同 bar"存在，
  即特征构造用到了决策时点之后（或同 bar 未收盘）的数据

  值多少：泄漏嫌疑特征的信息占比 = |IC_now| 排名位次（供策略侧换算 Sharpe 代价）。
"""
from __future__ import annotations

import numpy as np


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 20:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def lookahead_scan(y, X_now, X_prev, feature_names, collapse_ratio: float = 0.35) -> dict:
    """X_now: 决策时点 t 的特征矩阵；X_prev: 同一行样本、特征取 t-1（或 t-k）值。
    同组（同币/同资产）内滞后，由调用方构造好 X_prev（防止跨序列错位）。
    """
    y = np.asarray(y, dtype=float)
    A, B = np.asarray(X_now, dtype=float), np.asarray(X_prev, dtype=float)
    rows = []
    for j, f in enumerate(feature_names):
        ic_now = _rank_ic(A[:, j], y)
        ic_prev = _rank_ic(B[:, j], y)
        if np.isnan(ic_now):
            continue
        ratio = ic_prev / ic_now if abs(ic_now) > 1e-6 else 1.0
        rows.append({
            "feature": f,
            "ic_now": round(ic_now, 4),
            "ic_lagged": round(ic_prev, 4) if not np.isnan(ic_prev) else None,
            "survival_ratio": round(float(ratio), 3),
        })
    rows.sort(key=lambda r: -abs(r["ic_now"]))
    suspects = [
        r for r in rows
        if abs(r["ic_now"]) >= 0.03 and r["survival_ratio"] is not None
        and r["survival_ratio"] < collapse_ratio
    ]
    return {
        "top_features": rows[:8],
        "leak_suspects": suspects,
        "verdict": (
            f"发现 {len(suspects)} 个偷看嫌疑特征（IC 在滞后一期后断崖）——核查其构造是否用了决策时点之后的数据"
            if suspects
            else "未见偷看特征：所有特征的 IC 滞后衰减平缓（合法信息衰减的正常形态）"
        ),
        "note": "黑盒检验：只需特征两期对齐矩阵与未来收益，不依赖模型；配合 leak_scan（单特征solo AUC）食用更佳",
    }
