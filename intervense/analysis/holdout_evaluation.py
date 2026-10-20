"""Holdout evaluation at group level: the honest out-of-sample harness.

Random row splits flatter models that memorize units. The pattern our
NFL claim rests on holds out ENTIRE groups (a full season) and asks
the model to rank and calibrate on units it has never seen. This
module is that harness as one call: split by group levels, fit on the
rest, and report AUC, Brier, log loss, and a calibration table on the
holdout, next to the class-prior baseline so the lift is a number.

Works for binary outcomes; the verdict names the AUC lift over the
prior-only baseline and reads the calibration slope of the top
quintile, because a model that ranks but miscalibrates is a ranking,
not a forecaster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def holdout_evaluation(
    df: pd.DataFrame,
    outcome: str,
    features: list[str],
    group: str,
    holdout_groups: list | None = None,
    model_factory=None,
    seed: int = 0,
) -> dict:
    """Group-level holdout evaluation for a binary outcome.

    Args:
        df: one row per observation.
        outcome: binary 0/1 column.
        features: predictor columns.
        group: column whose LEVELS define the holdout (for example
            season or cohort).
        holdout_groups: levels to hold out; default is the single
            largest level (leave-one-group-out style).
        model_factory: callable returning an unfitted sklearn
            pipeline with predict_proba; default is a scaled
            logistic regression.

    Returns:
        Dict with holdout AUC, Brier, log loss, calibration by
        quintile, prior-only baselines, and a verdict reading the
        lift and the top-quintile calibration.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    missing = [c for c in [outcome, group] + features if c not in df.columns]
    if missing:
        raise ValueError(f"unknown columns: {missing}")
    d = df[[outcome, group] + features].dropna().copy()
    y_all = d[outcome].astype(float)
    if not set(y_all.unique()) <= {0.0, 1.0}:
        raise ValueError("outcome must be binary 0/1")

    if holdout_groups is None:
        counts = d[group].value_counts()
        holdout_groups = [counts.index[0]]
    te_mask = d[group].isin(holdout_groups)
    if te_mask.sum() < 20:
        raise ValueError(
            f"holdout has {int(te_mask.sum())} rows, need at least 20")
    train, test = d[~te_mask], d[te_mask]
    if len(train) < 50:
        raise ValueError(f"train has {len(train)} rows, need at least 50")

    if model_factory is None:
        model_factory = lambda: make_pipeline(  # noqa: E731
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    model = model_factory()
    model.fit(train[features], train[outcome])
    p = model.predict_proba(test[features])[:, 1]
    y = test[outcome].to_numpy()

    auc = float(roc_auc_score(y, p))
    brier = float(brier_score_loss(y, p))
    ll = float(log_loss(y, p, labels=[0, 1]))
    prior = float(train[outcome].mean())
    brier_prior = float(np.mean((prior - y) ** 2))

    q = pd.qcut(p, 5, duplicates="drop")
    cal = (pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True)
           .mean().round(4))
    calibration = [{"predicted": float(r.p), "actual": float(r.y)}
                   for r in cal.itertuples()]
    top = calibration[-1] if calibration else None

    lift = auc - 0.5
    if top and abs(top["predicted"] - top["actual"]) < max(0.05, 3 * np.sqrt(max(top["predicted"] * (1 - top["predicted"]), 1e-4) / max(len(test) / 5, 1))):
        cal_note = "top quintile calibration is consistent"
    elif top:
        cal_note = (f"top quintile predicted {top['predicted']:.3f} vs "
                    f"actual {top['actual']:.3f}, recalibrate before "
                    "trusting the probabilities")
    else:
        cal_note = "calibration not computable"

    if auc >= 0.75:
        verdict = (
            f"STRONG HOLDOUT: AUC {auc:.3f} on unseen groups "
            f"{holdout_groups} ({len(test):,} rows), {cal_note}"
        )
    elif auc > 0.6:
        verdict = (
            f"MODERATE HOLDOUT: AUC {auc:.3f}, {cal_note}"
        )
    else:
        verdict = (
            f"WEAK HOLDOUT: AUC {auc:.3f} is near chance on unseen "
            f"groups {holdout_groups}; the in-sample story does not "
            "travel"
        )

    return {
        "holdout_groups": [str(g) for g in holdout_groups],
        "n_train": len(train),
        "n_test": len(test),
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "brier_prior_baseline": round(brier_prior, 4),
        "log_loss": round(ll, 4),
        "prior": round(prior, 4),
        "holdout_rate": round(float(y.mean()), 4),
        "calibration": calibration,
        "verdict": verdict,
        "method": "group_level_holdout_logistic",
        "reference": (
            "Saeb et al. (2017) on the perils of random splits; "
            "leave-one-group-out as the honest unit of generalization"
        ),
        "note": (
            "entire group levels are held out, so the model is scored "
            "on units it never saw; the Brier prior baseline is the "
            "constant prediction of the training rate; default model "
            "is a scaled logistic, pass model_factory for any "
            "sklearn pipeline"
        ),
    }
