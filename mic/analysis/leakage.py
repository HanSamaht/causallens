"""泄漏与污染检测：模型有没有"偷看未来"、训练集和测试集有没有混。

三个工具（均零依赖、确定性）：
  1. duplicate_overlap    训练/测试行级重叠（精确重复 + 近重复）
     —— 测试样本整行出现在训练集 = 污染实锤
  2. single_feature_leak_scan 单特征泄漏扫描
     —— 某个特征单独一个就有 AUC≈模型整体 = 标签泄漏嫌疑
     （原理：合法特征弱且需组合；泄漏特征（如含未来信息的收盘价）单独即近乎全知）
  3. label_distribution_check 标签口径对账（训练/测试标签分布漂移）

时间泄漏（未来函数）：表格驱动器的背景数据无时间轴，滞后敏感度检验
（do(特征 := 滞后1bar) 看效应存活）需要时间戳元数据 —— 见 GAP-002，
时序驱动器的 window 语义天然支持，表格版待背景数据携带时间列后接入。
"""
from __future__ import annotations

import numpy as np


def duplicate_overlap(X_train, X_test, near_tol: float = 0.01, chunk: int = 256) -> dict:
    """行级重叠：精确重复 + 近重复（归一化 min 距离 < near_tol）。"""
    A = np.asarray(X_train, dtype=float)
    B = np.asarray(X_test, dtype=float)
    rng_vec = np.maximum(A.max(axis=0) - A.min(axis=0), 1e-12)
    An, Bn = A / rng_vec, B / rng_vec

    exact = 0
    near = 0
    suspect_idx: list[int] = []
    for i in range(0, len(Bn), chunk):
        blk = Bn[i : i + chunk]                     # (c, F)
        d = np.abs(blk[:, None, :] - An[None, :, :]).sum(axis=2)  # (c, n_train)
        dmin = d.min(axis=1)
        exact += int((dmin < 1e-9).sum())
        near += int((dmin < near_tol).sum())
        suspect_idx += (i + np.where(dmin < near_tol)[0]).tolist()
    n = len(Bn)
    near_rate = (exact + near) / n if n else 0.0
    verdict = (
        "污染实锤" if exact / max(n, 1) > 0.02
        else ("高嫌疑：大量近重复，检查切分逻辑" if near_rate > 0.15
              else ("轻度重叠：边界期样本可能跨切分" if near_rate > 0.02 else "干净：无有效重叠"))
    )
    return {
        "n_test": n,
        "exact_duplicates": exact,
        "near_duplicates_total": near,
        "overlap_rate": round(float(near_rate), 4),
        "suspect_test_indices": suspect_idx[:20],
        "verdict": verdict,
        "note": "近重复=归一化L1距离<%.3f；时间序列数据中相邻bar的窗口天然相近，需结合切分方式解读" % near_tol,
    }


def single_feature_leak_scan(y, X, feature_names, model_auc: float | None = None) -> dict:
    """单特征 solo AUC：泄漏特征单独近乎全知。"""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    X = np.asarray(X, dtype=float)
    rows = []
    for j, f in enumerate(feature_names):
        col = X[:, j]
        m = ~np.isnan(col)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            continue
        try:
            auc = max(
                roc_auc_score(y[m], col[m]),
                roc_auc_score(y[m], -col[m]),
            )
        except Exception:
            continue
        rows.append({"feature": f, "solo_auc": round(float(auc), 4)})
    rows.sort(key=lambda r: -r["solo_auc"])
    flagged = [r for r in rows if r["solo_auc"] >= 0.75]
    borderline = [r for r in rows if 0.65 <= r["solo_auc"] < 0.75]
    return {
        "top_features": rows[:8],
        "leak_suspects": flagged,
        "borderline": borderline,
        "model_auc_reference": model_auc,
        "verdict": (
            f"发现 {len(flagged)} 个高嫌疑泄漏特征（solo AUC≥0.75）——核查其构造是否含未来信息"
            if flagged
            else f"无高嫌疑（{len(borderline)} 个边缘项 0.65-0.75，多为强合法因子）"
        ),
        "note": "合法强因子（如动量）solo AUC 0.6-0.7 正常；>0.75 且业务上'不该这么准'的特征按泄漏核查",
    }


def label_distribution_check(y_train, y_test) -> dict:
    a, b = np.asarray(y_train, dtype=float), np.asarray(y_test, dtype=float)
    pos_tr, pos_te = float((a > 0).mean()), float((b > 0).mean())
    gap = abs(pos_te - pos_tr)
    return {
        "train_positive_rate": round(pos_tr, 4),
        "test_positive_rate": round(pos_te, 4),
        "gap": round(gap, 4),
        "verdict": "标签口径一致" if gap < 0.1 else "标签口径漂移（切分方式或时期差异，比较结论需谨慎）",
    }
