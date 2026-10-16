"""忠实性验证层：归因必须"配得上"模型的真实决策。

借用 ERASER 的两个标准指标 + 随机基线对照：
  - 完备性 comprehensiveness：抹掉 Top-k（置为背景中位数）后预测应大变
  - 充分性 sufficiency：只保留 Top-k、其余抹掉后预测应基本不变
  - 随机基线：随机选 k 个特征抹掉——SHAP 排名若不显著强于随机，解释不可信
"""
from __future__ import annotations

import numpy as np

from ..core.types import FaithfulnessReport
from ..drivers.base import MicDriver
from .attribution import explain_sample


def _masked_predict(driver: MicDriver, x: np.ndarray, keep: set[str]) -> float:
    """只保留 keep 中的特征，其余抹为背景中位数。"""
    med = np.median(driver.background, axis=0)
    x2 = x.copy()
    for j, f in enumerate(driver.feature_names):
        if f not in keep:
            x2[j] = med[j]
    return driver.predict_one(x2)


def evaluate_faithfulness(
    driver: MicDriver,
    x: np.ndarray,
    attributions=None,
    k: int | None = None,
    n_random: int = 20,
    seed: int = 0,
) -> FaithfulnessReport:
    x = np.asarray(x, dtype=float).ravel()
    if attributions is None:
        attributions = explain_sample(driver, x)
    F = len(driver.feature_names)
    k = k or max(2, int(np.ceil(F * 0.2)))

    p_full = driver.predict_one(x)
    ranked = [a.feature for a in attributions]
    top_k = set(ranked[:k])

    p_comp = _masked_predict(driver, x, keep=set(driver.feature_names) - top_k)  # 抹掉 top-k
    p_suff = _masked_predict(driver, x, keep=top_k)                             # 只留 top-k

    comprehensiveness = abs(p_full - p_comp)
    sufficiency = abs(p_full - p_suff)

    rng = np.random.default_rng(seed)
    random_scores = []
    for _ in range(n_random):
        rand_k = set(rng.choice(ranked, size=k, replace=False).tolist())
        p_rand = _masked_predict(driver, x, keep=set(driver.feature_names) - rand_k)
        random_scores.append(abs(p_full - p_rand))
    random_baseline = float(np.mean(random_scores))

    notes = []
    if comprehensiveness <= random_baseline * 1.2:
        notes.append("警告：SHAP Top-k 抹除效应不优于随机基线，归因可能不可信")
    if sufficiency > comprehensiveness:
        notes.append("警告：充分性差于完备性——Top-k 特征不足以解释该决策")
    passed = len(notes) == 0

    return FaithfulnessReport(
        top_k=k,
        comprehensiveness=float(comprehensiveness),
        sufficiency=float(sufficiency),
        random_baseline=random_baseline,
        passed=passed,
        notes=notes,
    )


# ================= 跨样本聚合：模型的"忠实性档案" =================

def aggregate_faithfulness(
    driver: MicDriver,
    n_samples: int = 20,
    k: int | None = None,
    seed: int = 0,
) -> dict:
    """在背景分布抽 n 个样本逐一评测，输出聚合档案。

    单样本指标可能是运气；通过率与分位数才能说明"这个模型的解释整体可不可信"。
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(driver.background), size=min(n_samples, len(driver.background)), replace=False)
    comps, suffs, bases, n_pass = [], [], [], 0
    for i in idxs:
        rep = evaluate_faithfulness(driver, driver.sample_at(int(i)), k=k)
        comps.append(rep.comprehensiveness)
        suffs.append(rep.sufficiency)
        bases.append(rep.random_baseline)
        n_pass += int(rep.passed)

    def q(a: list[float], p: float) -> float:
        return float(np.quantile(a, p))

    profile = {
        "n_samples": len(idxs),
        "pass_rate": n_pass / len(idxs),
        "comprehensiveness": {"mean": float(np.mean(comps)), "p25": q(comps, 0.25), "p75": q(comps, 0.75)},
        "sufficiency": {"mean": float(np.mean(suffs)), "p25": q(suffs, 0.25), "p75": q(suffs, 0.75)},
        "random_baseline": {"mean": float(np.mean(bases))},
        "verdict": "可信" if n_pass / len(idxs) >= 0.8 else "存疑",
    }
    profile["notes"] = (
        []
        if profile["verdict"] == "可信"
        else [f"忠实性通过率仅 {profile['pass_rate']:.0%}，低于 80% 阈值——该模型的归因解释需谨慎使用"]
    )
    return profile


# ================= explain → simulate → score（确定性版） =================

def explain_simulate_score(
    driver: MicDriver,
    x: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> dict:
    """OpenAI auto-interp 协议的领域模型版。

    explain:   模型给出的"解释" = SHAP 归因排名（声称的特征重要性）
    simulate:  用干预实验独立测量每个特征的真实因果敏感度
               （do(f:=q) 扫背景分位，取预测平均漂移）
    score:     两套排名的 Spearman 等级相关。

    相关高 → 归因忠实于模型真实依赖；相关低 → 解释失真。
    与掩码类指标互补：不制造分布外输入，每一步都是合法 do 干预。
    """
    x = np.asarray(x, dtype=float).ravel()
    p_base = driver.predict_one(x)
    attrs = explain_sample(driver, x)
    claimed = {a.feature: abs(a.shap_value) for a in attrs}

    measured: dict[str, float] = {}
    for f in driver.feature_names:
        drifts = []
        for qv in quantiles:
            p = driver.predict_one(driver.intervene(x, {f: driver.quantile_of(f, qv)}))
            drifts.append(abs(p - p_base))
        measured[f] = float(np.mean(drifts))

    feats = driver.feature_names
    a = np.array([claimed[f] for f in feats])
    b = np.array([measured[f] for f in feats])
    score = _spearman(a, b)
    return {
        "method": "SHAP排名 vs 干预敏感度排名 的 Spearman 相关",
        "spearman": None if score is None else round(score, 4),
        "verdict": (
            "忠实" if (score is not None and score >= 0.6)
            else "部分失真" if (score is not None and score >= 0.3)
            else "失真/无法判定"
        ),
        "per_feature": {
            f: {"claimed_importance": round(claimed[f], 4), "measured_sensitivity": round(measured[f], 4)}
            for f in feats
        },
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """零依赖 Spearman 等级相关（避免为一个小指标引入 scipy）。"""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, len(v) + 1)
        return r

    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])
