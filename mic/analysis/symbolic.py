"""符号蒸馏：让模型"说数学语言"——把黑箱行为压缩成一条可审计的公式。

两级引擎：
  - PySR（若安装，pip install pysr）：进化搜索的解析表达式，最优解释形态
  - 兜底（零依赖）：Top-k 特征的 Lasso 基展开（线性项 + x² + |x| + Top 交互项），
    输出带系数的公式与对模型输出的 R²——足够回答"模型大体上在算什么"

对科学/地理/金融因子合成模型，公式比自然语言更压缩、更可审计。
"""
from __future__ import annotations

import numpy as np

from ..drivers.base import MicDriver
from .attribution import shap_matrix


def _fmt(v: float) -> str:
    return f"{v:+.3f}"


def symbolic_distill(driver: MicDriver, n_samples: int = 400, top_k: int = 6) -> dict:
    X = driver.background[: min(n_samples, len(driver.background))]
    y = driver.predict(X)

    # ---- 首选 PySR（可选依赖） ----
    try:
        from pysr import PySRRegressor

        est = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*"],
            unary_operators=[],
            extra_sympy_mappings={},
            verbosity=0,
            progress=False,
            random_state=0,
        )
        est.fit(X, y)
        best = est.get_best()
        return {
            "engine": "pysr",
            "formula": str(best.equation),
            "complexity": int(best.complexity),
            "r2_vs_model": round(float(best.score), 4),
            "note": "PySR 进化搜索得到的解析表达式（对模型行为的近似，非模型本体）",
        }
    except Exception:
        pass

    # ---- 兜底：Lasso 基展开 ----
    from sklearn.linear_model import LassoCV

    mean_abs = np.abs(shap_matrix(driver, X)).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:top_k]
    feats = [driver.feature_names[i] for i in top_idx]

    Xt = X[:, top_idx]
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0) + 1e-9
    Z = (Xt - mu) / sd

    cols, names = [], []
    for j, f in enumerate(feats):  # 线性项（标准化后）
        cols.append(Z[:, j])
        names.append(f)
        cols.append(Z[:, j] ** 2)  # 二次项
        names.append(f"{f}²")
        cols.append(np.abs(Z[:, j]))  # 绝对值项
        names.append(f"|{f}|")
    for a in range(min(3, len(feats))):  # Top-3 特征两两交互
        for b in range(a + 1, min(4, len(feats))):
            cols.append(Z[:, a] * Z[:, b])
            names.append(f"{feats[a]}·{feats[b]}")

    B = np.stack(cols, axis=1)
    yz = (y - y.mean()) / (y.std() + 1e-9)
    lasso = LassoCV(cv=5, random_state=0).fit(B, yz)
    r2 = float(lasso.score(B, yz))

    terms = []
    for name, c in zip(names, lasso.coef_):
        if abs(c) > 0.02:
            terms.append((name, float(c)))
    terms.sort(key=lambda t: -abs(t[1]))
    formula = " ".join(f"{_fmt(c)}·[{n}]" for n, c in terms[:10]) or "(无可辨识结构)"

    return {
        "engine": "lasso-basis",
        "formula": f"ŷ ≈ {y.mean():+.3f} {formula}",
        "top_terms": [{"term": n, "coef": round(c, 4)} for n, c in terms[:10]],
        "r2_vs_model": round(r2, 4),
        "note": (
            "Lasso 基展开蒸馏（Top 特征 + 二次/绝对/交互项）。这是模型行为的线性近似描述，"
            "R² 越高公式越可信；R²<0.5 时模型存在显著非线性/高阶结构，公式仅作参考。"
            f"安装 PySR 可获得进化搜索的解析式：pip install pysr"
        ),
    }
