"""Mediation chains with bootstrap inference and the suppression guard.

The a-path times b-path question, as one call: does the treatment
reach the outcome THROUGH the mediator, and what share of the total
effect travels that road. Paths are standardized, the indirect effect
carries a bootstrap percentile interval over rows, and the share is
only claimed when it is stable.

The suppression guard exists because we shipped the opposite once: a
mediation share of 123 percent with a bootstrap interval from -636 to
+833 percent went into an early draft, and the only honest reading
was that the total effect was too small for a ratio to mean anything.
That lesson is now a verdict: when the total effect is weak relative
to the indirect path, the module reports SUPPRESSION REGIME and
claims no percentage, naming the two links and their strengths
instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mediation_chain(
    df: pd.DataFrame,
    treatment: str,
    mediator: str,
    outcome: str,
    controls: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Standardized three-path mediation with bootstrap intervals.

    Args:
        df: one row per unit.
        treatment: the X of the chain.
        mediator: the M the treatment is claimed to travel through.
        outcome: the Y.
        controls: optional covariates entered in the b and direct
            path regressions.
        n_boot: bootstrap resamples for the indirect interval.
        seed: reproducibility.

    Returns:
        Dict with a, b, direct, total, indirect a*b and its bootstrap
        interval, the mediated share when it is stable, and one of
        three verdicts: MEDIATION ESTABLISHED, NOT ESTABLISHED, or
        SUPPRESSION REGIME.
    """
    cols = [treatment, mediator, outcome] + list(controls or [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"unknown columns: {missing}")
    d = df[cols].dropna().astype(float).reset_index(drop=True)
    n = len(d)
    if n < 30:
        raise ValueError(f"n={n} is too small for a mediation chain")

    Z = d[[treatment, mediator, outcome]].to_numpy()
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
    t, m, y = Z[:, 0], Z[:, 1], Z[:, 2]
    ctrl = (d[controls].to_numpy() if controls else None)
    if ctrl is not None:
        ctrl = (ctrl - ctrl.mean(axis=0)) / (ctrl.std(axis=0) + 1e-12)

    def _paths(t_, m_, y_, c_=None):
        a = float(np.polyfit(t_, m_, 1)[0])
        M1 = np.column_stack([t_, m_, np.ones(len(t_))]
                             + ([] if c_ is None else [c_]))
        coefs = np.linalg.lstsq(M1, y_, rcond=None)[0]
        total = float(np.polyfit(t_, y_, 1)[0])
        return a, float(coefs[1]), float(coefs[0]), total

    a, b, direct, total = _paths(t, m, y, ctrl)
    indirect = a * b

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            if ctrl is None:
                ab = _paths(t[i], m[i], y[i])
            else:
                ab = _paths(t[i], m[i], y[i], ctrl[i])
            boots.append(ab[0] * ab[1])
        except Exception:  # noqa: BLE001 - degenerate resample
            continue
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))

    share = indirect / total if abs(total) > 1e-12 else np.nan
    share_lo = lo / total if abs(total) > 1e-12 else np.nan
    share_hi = hi / total if abs(total) > 1e-12 else np.nan

    suppression = (
        not np.isfinite(share)
        or abs(share) > 1.0
        or (np.isfinite(share_lo) and share_lo < -0.5)
        or (np.isfinite(share_hi) and share_hi > 1.5)
        or total * indirect < 0
    )
    if suppression:
        verdict = (
            "SUPPRESSION REGIME: the total effect is too weak or "
            "opposite-signed relative to the indirect path for a "
            "share to mean anything; report the links (a "
            f"{a:+.2f}, b {b:+.2f}) and claim no percentage"
        )
    elif lo > 0 or hi < 0:
        verdict = (
            f"MEDIATION ESTABLISHED: indirect a*b {indirect:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}] excludes zero, share {share:.0%} "
            f"of the total effect {total:+.3f}"
        )
    else:
        verdict = (
            f"NOT ESTABLISHED: indirect a*b {indirect:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}] crosses zero"
        )

    return {
        "n_rows": n,
        "a_path": round(a, 6),
        "b_path": round(b, 6),
        "indirect_ab": round(indirect, 6),
        "indirect_ci95": [round(lo, 6), round(hi, 6)],
        "direct_path": round(direct, 6),
        "total_path": round(total, 6),
        "mediated_share": round(float(share), 4) if np.isfinite(share) else None,
        "share_ci95": ([round(float(share_lo), 4), round(float(share_hi), 4)]
                       if np.isfinite(share_lo) and np.isfinite(share_hi)
                       else None),
        "n_boot": len(boots),
        "verdict": verdict,
        "method": "standardized_path_bootstrap",
        "reference": (
            "Baron and Kenny (1986) mediation; Preacher and Hayes "
            "(2008) bootstrap indirect intervals; the suppression "
            "guard follows MacKinnon, Fairchild and Fritz (2007)"
        ),
        "note": (
            "all three variables standardized; controls enter the b "
            "and direct paths only; the share is suppressed whenever "
            "the total effect is weak, opposite-signed, or the share "
            "interval leaves sane bounds, because a ratio over a "
            "near-zero denominator is a story, not a statistic"
        ),
    }
