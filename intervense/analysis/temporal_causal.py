"""Temporal causal discovery for time series drivers.

Pairwise Granger causality over the raw channels of a TimeSeriesDriver:
does channel j at lags 1..L improve the prediction of channel i beyond
i's own lags? Each surviving pair becomes a lagged causal edge
j@lag k -> i@lag 0 with an F statistic and a multiple-comparison
corrected p value.

Edges are annotated with the model's own reliance: the share of total
input attribution mass the model places on (source channel, best lag).
Data can say a lagged cause exists while the model ignores it, or the
model can lean on a lag the data cannot justify. Both situations are
findings.

Caveats kept explicit:
  - windows pooled across the background are treated as independent
    realizations, so residual autocorrelation can inflate significance
  - pairwise Granger misses conditional structure; treat edges as
    hypotheses for the intervention tools, not verdicts
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _lag_matrix(series: np.ndarray, max_lag: int, channel: int, row_stride: int = 1):
    """Rows of lagged values for one channel, pooled over windows.

    series: (n_windows, T, C). Returns (rows, max_lag) newest first and
    the matching target vector (rows,) at time t. row_stride > 1 keeps
    every stride-th target time so kept rows share no time points.
    """
    n, T, _ = series.shape
    rows, targets = [], []
    for w in range(n):
        for t in range(max_lag, T, row_stride):
            window = series[w, t - max_lag:t, channel]
            rows.append(window[::-1])  # lag1 first
            targets.append(series[w, t, channel])
    return np.asarray(rows), np.asarray(targets)


def granger_pairwise(series: np.ndarray, max_lag: int = 4,
                     window_stride: int | None = None) -> list[dict]:
    """All ordered channel pairs with Granger F tests.

    Backgrounds usually contain sliding windows that overlap heavily,
    and overlapping rows duplicate time points, which inflates the F
    statistics. Two guards keep the test honest: windows are subsampled
    with window_stride (defaults to non-overlapping) and target rows
    are strided so kept rows share no time points.

    Returns a list of dicts with source, target, best_lag, f_stat,
    p_value, delta_r2, sorted by p value.
    """
    series = np.asarray(series, dtype=float)
    n, T, C = series.shape
    if window_stride is None:
        window_stride = max(1, T)  # keep every T-th window: non-overlapping
    series = series[::window_stride]
    row_stride = max(1, max_lag)

    results = []
    for tgt in range(C):
        own, y = _lag_matrix(series, max_lag, tgt, row_stride=row_stride)
        own = np.column_stack([np.ones(len(own)), own])
        coef_r = np.linalg.lstsq(own, y, rcond=None)[0]
        rss_r = float(np.sum((y - own @ coef_r) ** 2))
        for src in range(C):
            if src == tgt:
                continue
            src_lags, _ = _lag_matrix(series, max_lag, src, row_stride=row_stride)
            full = np.column_stack([own, src_lags])
            coef_f = np.linalg.lstsq(full, y, rcond=None)[0]
            rss_f = float(np.sum((y - full @ coef_f) ** 2))
            n_rows, k_f = full.shape
            if rss_f <= 1e-12 or n_rows <= k_f + max_lag:
                continue
            f_stat = ((rss_r - rss_f) / max_lag) / (rss_f / (n_rows - k_f))
            p = float(stats.f.sf(f_stat, max_lag, n_rows - k_f))
            # which lag carries the improvement: refit with one lag at a time
            best_lag, best_gain = 1, -1.0
            for k in range(max_lag):
                one = np.column_stack([own, src_lags[:, k]])
                coef_k = np.linalg.lstsq(one, y, rcond=None)[0]
                rss_k = float(np.sum((y - one @ coef_k) ** 2))
                gain = rss_r - rss_k
                if gain > best_gain:
                    best_gain, best_lag = gain, k + 1
            tss = float(np.sum((y - y.mean()) ** 2)) or 1.0
            results.append({
                "source": src,
                "target": tgt,
                "best_lag": best_lag,
                "f_stat": round(float(f_stat), 4),
                "p_value": p,
                "delta_r2": round((rss_r - rss_f) / tss, 6),
            })
    results.sort(key=lambda r: r["p_value"])
    return results


def pcmci_edges(series: np.ndarray, names: list[str], max_lag: int = 4,
                alpha: float = 0.05) -> list[dict]:
    """Conditional lagged edges via PCMCI (Tigramite).

    Pairwise Granger asks whether one channel helps predict another
    beyond its own lags. PCMCI conditions each test on the full
    relevant past, so a link survives only when it carries information
    the rest of the system does not already contain.

    PCMCI needs one continuous series; the windowed background is
    reconstructed by concatenating non-overlapping windows. Each join
    can contaminate at most max_lag rows, which is disclosed here
    rather than hidden.
    """
    try:
        from tigramite.data_processing import DataFrame as TigDataFrame
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except Exception as e:  # pragma: no cover - exercised only without the library
        raise ImportError("tigramite is not installed. Install it with: pip install tigramite") from e

    n, T, _ = series.shape
    long_series = np.concatenate(list(series[::T]), axis=0)
    df = TigDataFrame(long_series, var_names=list(names))
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=alpha)
    graph, val, pmat = res["graph"], res["val_matrix"], res["p_matrix"]

    rows = []
    n_vars = len(names)
    for tgt in range(n_vars):
        for src in range(n_vars):
            if src == tgt:
                continue  # autoregressive self links are not cross causes
            for lag in range(1, max_lag + 1):
                if graph[src, tgt, lag] not in ("", "--"):
                    rows.append({
                        "source": src,
                        "target": tgt,
                        "lag": lag,
                        "strength": float(abs(val[src, tgt, lag])),
                        "p_value": float(pmat[src, tgt, lag]),
                    })
    rows.sort(key=lambda r: r["p_value"])
    return rows


def _model_lag_reliance(driver, n_samples: int = 8) -> dict:
    """Mean absolute window attribution mass per (channel, lag).

    Returns {(channel, lag): share of total attribution mass}.
    """
    heat_sum = None
    count = 0
    for i in range(min(n_samples, len(driver.background))):
        heat, names = driver.window_attribution(driver.sample_at(i))
        heat = np.abs(heat)
        heat_sum = heat if heat_sum is None else heat_sum + heat
        count += 1
    if count == 0:
        return {}
    heat_sum = heat_sum / count
    total = heat_sum.sum() or 1.0
    shares = {}
    for f, name in zip(range(heat_sum.shape[1]), driver.base_feature_names):
        for lag in range(heat_sum.shape[0]):
            shares[(name, lag)] = float(heat_sum[lag, f] / total)
    return shares


def temporal_causal_graph(driver, max_lag: int = 4, alpha: float = 0.05,
                          n_windows: int = 256, annotate_model: bool = True,
                          bonferroni: bool = True, method: str = "granger") -> dict:
    """Discover lagged channel causes in the driver background.

    method "granger" runs the pairwise F tests. method "pcmci" runs
    Tigramite PCMCI, which conditions each link test on the rest of
    the system's past, at the cost of the extra dependency and the
    window concatenation caveat documented in pcmci_edges.

    Both apply Bonferroni correction across the tested links and
    annotate each surviving edge with the model's attribution mass on
    that (channel, lag).
    """
    bg = driver.background[:n_windows]
    series = bg.reshape(len(bg), driver.seq_len, driver.n_features_raw)
    names = list(driver.base_feature_names)

    if method == "pcmci":
        raw = pcmci_edges(series, names, max_lag=max_lag, alpha=alpha)
        pairs = [
            {"source": r["source"], "target": r["target"], "best_lag": r["lag"],
             "f_stat": round(r["strength"], 4), "p_value": r["p_value"],
             "delta_r2": None}
            for r in raw
        ]
    elif method == "granger":
        pairs = granger_pairwise(series, max_lag=max_lag)
    else:
        raise ValueError(f"unknown method {method!r}, expected granger or pcmci")
    n_tests = max(len(pairs), 1)
    threshold = alpha / n_tests if bonferroni else alpha

    reliance = {}
    if annotate_model and hasattr(driver, "window_attribution"):
        reliance = _model_lag_reliance(driver)

    edges = []
    for r in pairs:
        significant = r["p_value"] < threshold
        src_name = names[r["source"]]
        edge = {
            "source": src_name,
            "target": names[r["target"]],
            "lag": r["best_lag"],
            "edge": f"{src_name}@lag{r['best_lag']} -> {names[r['target']]}@now",
            "f_stat": r["f_stat"],
            "p_value": round(r["p_value"], 6),
            "p_corrected": round(min(r["p_value"] * n_tests, 1.0), 6),
            "delta_r2": r["delta_r2"],
            "significant": significant,
        }
        key = (src_name, r["best_lag"])
        if key in reliance:
            edge["model_reliance"] = round(reliance[key], 4)
        edges.append(edge)

    survivors = [e for e in edges if e["significant"]]
    if not survivors:
        verdict = (
            f"NO TEMPORAL CAUSES at alpha={alpha} after correction: "
            "no channel improves another's forecast beyond its own lags"
        )
    else:
        top = max(survivors, key=lambda e: e["f_stat"])
        ignored = [e for e in survivors if e.get("model_reliance", 0.0) < 0.02]
        verdict = (
            f"{len(survivors)} significant lagged edges; strongest is "
            f"{top['edge']} (F={top['f_stat']:.1f})"
        )
        if ignored:
            verdict += (
                "; the model places under 2 percent of attribution mass on: "
                + ", ".join(e["edge"] for e in ignored)
                + " (discovered in data but unused by the model)"
            )

    return {
        "max_lag": max_lag,
        "channels": names,
        "n_windows_used": len(series),
        "correction": "bonferroni" if bonferroni else "none",
        "alpha": alpha,
        "n_significant": len(survivors),
        "edges": edges,
        "significant_edges": survivors,
        "verdict": verdict,
        "method": ("pcmci" if method == "pcmci" else "pairwise_granger_f_test"),
        "reference": (
            "Runge et al. (2019) Detecting and quantifying causal associations"
            " in large nonlinear time series, Science Advances"
            if method == "pcmci"
            else "Granger (1969) Investigating Causal Relations by Econometric Models"
        ),
    }
