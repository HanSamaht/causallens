"""symbolic distillation:model""——behavioraudit.

engine:
  - PySR(,pip install pysr):search,most
  - ():Top-k feature Lasso ( + x² + |x| + Top ),
    outputandmodeloutput R²——"modelonin"

//model,more,moreaudit.
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

    # ---- PySR(optional) ----
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
            "note": "PySR searchto(modelbehavior,model)",
        }
    except Exception:
        pass

    # ---- :Lasso ----
    from sklearn.linear_model import LassoCV

    mean_abs = np.abs(shap_matrix(driver, X)).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:top_k]
    feats = [driver.feature_names[i] for i in top_idx]

    Xt = X[:, top_idx]
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0) + 1e-9
    Z = (Xt - mu) / sd

    cols, names = [], []
    for j, f in enumerate(feats): # ()
        cols.append(Z[:, j])
        names.append(f)
        cols.append(Z[:, j] ** 2) #
        names.append(f"{f}²")
        cols.append(np.abs(Z[:, j])) # value
        names.append(f"|{f}|")
    for a in range(min(3, len(feats))): # Top-3 feature
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
    formula = " ".join(f"{_fmt(c)}·[{n}]" for n, c in terms[:10]) or "()"

    return {
        "engine": "lasso-basis",
        "formula": f"ŷ ≈ {y.mean():+.3f} {formula}",
        "top_terms": [{"term": n, "coef": round(c, 4)} for n, c in terms[:10]],
        "r2_vs_model": round(r2, 4),
        "note": (
            "Lasso (Top feature + //).thisismodelbehavior,"
            "R² ;R²<0.5 modelinsignificant/,."
            f" PySR search:pip install pysr"
        ),
    }
