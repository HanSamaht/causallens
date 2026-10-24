"""Data forensics: the manual audit nobody automates.

Before any causal question, a careful analyst checks the plumbing: how
much is missing and why, which columns drift across cohorts, whether
the join keys actually cover both sides. On the NFL Big Data Bowl 2027
work these checks were done by hand and caught real traps (selection
missingness in traditional drills, protocol drift across 2023 to 2025,
join coverage gaps). This module runs them in one call.

Sections:

  - basics: shape, duplicate rows, constant and duplicate columns
  - missingness: per column inventory plus the dominant missing
    pattern, selection bias shows up as a few columns dominating one
    pattern
  - mechanism hints: for each partly missing column, Mann Whitney
    tests compare every other numeric column (datetimes as epoch
    seconds) between missing and observed rows, chi square tests
    compare categorical witnesses; an association means missing at
    random is unlikely and the column is MAR suspect. A tipping
    sensitivity reports how extreme the missing values would have to
    be (in SD units) to move the column mean by half a standard
    deviation, the honest answer to "but what if it is MNAR"
  - drift: Kruskal Wallis across group labels (for example draft
    year) per numeric column, chi square composition drift per
    categorical column
  - join audit: key uniqueness and two directional coverage between
    two frames

The verdict orders findings by severity. Associations and drift are
statistical flags, not causal claims; the note says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu


def _numeric_view(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Numeric columns plus datetimes as epoch seconds (drift and
    association scans read both; the note discloses the conversion).
    Missing timestamps become NaN: the raw int64 cast would turn NaT
    into the sentinel year 1677 and silently poison every statistic.
    """
    out: dict[str, np.ndarray] = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = df[c].to_numpy(dtype=float)
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            s = df[c]
            # unit-independent epoch seconds: pandas 2 stores ns,
            # pandas 3 defaults to us, so an int64 cast changes scale
            deltas = (s - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
            vals = np.array(deltas, dtype="float64", copy=True)
            vals[s.isna().to_numpy()] = np.nan
            out[c] = vals
    return out


def _categorical_columns(df: pd.DataFrame, max_cardinality: int = 20) -> list[str]:
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or \
                pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        n = df[c].nunique(dropna=True)
        if 2 <= n <= max_cardinality:
            out.append(c)
    return out


def data_forensics(
    df: pd.DataFrame,
    groups=None,
    other_df: pd.DataFrame | None = None,
    key: str | list[str] | None = None,
    alpha: float = 0.05,
    max_assoc_columns: int = 10,
    assertions: list[dict] | None = None,
) -> dict:
    """One call audit of a dataframe before any modeling happens.

    Args:
        df: the frame to audit.
        groups: cohort labels per row. A column name of df or an
            array aligned with df; enables the drift scan.
        other_df: second frame for the join audit.
        key: join key column (or list of columns) shared by df and
            other_df.
        alpha: significance level for the Mann Whitney and Kruskal
            Wallis flags, Bonferroni corrected per section.
        max_assoc_columns: numeric columns tested against each partly
            missing column, runtime guard.
        assertions: declarative plausibility gates, each
            {"name", "column", "statistic" (median/mean/max/min/
            fraction), "op" (< <= > >=), "value", optional
            "where": [column, value] subset}. Violations surface as
            HIGH findings; the pattern caught a features-vs-random-
            players bug that manual review had missed.

    Returns:
        Dict with the five sections plus a severity ordered verdict.
    """
    if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
        raise ValueError("df must be a non empty pandas DataFrame")
    findings: list[dict] = []
    n_rows, n_cols = df.shape

    # ---- basics
    dup_rows = int(df.duplicated().sum())
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    dup_cols: list[list[str]] = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                dup_cols.append([cols[i], cols[j]])
    basics = {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "duplicate_rows": dup_rows,
        "constant_columns": constant,
        "duplicate_columns": dup_cols,
    }
    if dup_rows:
        findings.append({"severity": "WARN", "text": f"{dup_rows} duplicate rows"})
    if constant:
        findings.append({"severity": "WARN",
                         "text": f"constant columns: {', '.join(map(str, constant))}"})
    if dup_cols:
        names = ", ".join(f"{a}=={b}" for a, b in dup_cols)
        findings.append({"severity": "WARN", "text": f"duplicate columns: {names}"})

    # ---- missingness inventory
    na = {c: {"n": int(df[c].isna().sum()),
              "frac": round(float(df[c].isna().mean()), 4)}
          for c in df.columns if df[c].isna().any()}
    pattern_share = 0.0
    if na:
        pat = df.isna().apply(lambda r: tuple(r), axis=1)
        pattern_share = round(float(pat.value_counts(normalize=True).iloc[0]), 4)
    missing = {"columns": na, "dominant_pattern_share": pattern_share}
    heavy = [c for c, v in na.items() if v["frac"] > 0.5]
    if heavy:
        findings.append({"severity": "WARN",
                         "text": f"over 50% missing: {', '.join(map(str, heavy))}"})
    if na and pattern_share > 0.3:
        findings.append({"severity": "WARN",
                         "text": (f"{pattern_share:.0%} of rows share one "
                                  "missingness pattern; check whether one "
                                  "mechanism dominates (selection)")})

    # ---- mechanism hints: MAR suspects
    numeric = _numeric_view(df)
    categorical = _categorical_columns(df)
    assoc: list[dict] = []
    targets = [c for c in df.columns if na.get(c, {"frac": 0})["frac"] >= 0.05
               and na.get(c, {"frac": 0})["frac"] <= 0.95]
    for t in targets:
        miss = df[t].isna()
        if miss.sum() < 10 or (~miss).sum() < 10:
            continue
        num_witnesses = [c for c in numeric if c != t
                         and df.loc[~miss, c].notna().any()]
        num_witnesses = sorted(num_witnesses)[:max_assoc_columns]
        cat_witnesses = [c for c in categorical if c != t]
        n_tests = max(len(num_witnesses) + len(cat_witnesses), 1)
        for c in num_witnesses:
            a = df.loc[miss, c].dropna()
            b = df.loc[~miss, c].dropna()
            if len(a) < 10 or len(b) < 10:
                continue
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            if p < alpha / n_tests:
                assoc.append({
                    "missing_in": str(t),
                    "differs_by": str(c),
                    "witness_kind": "numeric",
                    "p_value": round(float(p), 6),
                    "median_when_missing": round(float(a.median()), 4),
                    "median_when_observed": round(float(b.median()), 4),
                })
        for c in cat_witnesses:
            table = pd.crosstab(miss, df[c].astype(str))
            # trim all-zero rows and columns, then require a 2x2 core;
            # strong associations leave few nonzero cells and must pass
            table = table.loc[(table > 0).any(axis=1), (table > 0).any(axis=0)]
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            stat, p, _dof, _exp = chi2_contingency(table)
            if p < alpha / n_tests:
                share_missing = df.loc[miss, c].value_counts(normalize=True)
                share_observed = df.loc[~miss, c].value_counts(normalize=True)
                top = share_missing.index[0]
                assoc.append({
                    "missing_in": str(t),
                    "differs_by": str(c),
                    "witness_kind": "categorical",
                    "p_value": round(float(p), 6),
                    "top_category_share_when_missing": round(float(share_missing.iloc[0]), 3),
                    "top_category_share_when_observed": round(float(share_observed.get(top, 0.0)), 3),
                    "top_category": str(top),
                })
    mechanism = {"mar_suspects": assoc[:20], "n_targets_tested": len(targets)}
    for a in assoc[:5]:
        findings.append({
            "severity": "HIGH",
            "text": (f"missingness in {a['missing_in']} associates with "
                     f"{a['differs_by']} (p={a['p_value']:.2g}); not MCAR"),
        })

    # ---- MNAR tipping sensitivity: how extreme would the missing values
    # have to be to move the column mean by half an observed SD
    tipping: list[dict] = []
    for c, values in numeric.items():
        n_miss = int(np.isnan(values).sum())
        frac = n_miss / n_rows
        if not (0.05 <= frac <= 0.95):
            continue
        observed = values[~np.isnan(values)]
        sd = float(np.std(observed))
        if sd < 1e-12:
            continue
        # uniform shift d on the missing entries moves the full mean by
        # d * n_miss / n_rows; solve for a half-SD move
        shift_sd = 0.5 * n_rows / n_miss
        tipping.append({
            "column": str(c),
            "frac_missing": round(frac, 3),
            "shift_for_half_sd_in_sds": round(shift_sd, 2),
            "tipping_value": round(float(observed.mean()) + shift_sd * sd, 4),
        })
    mechanism["mnar_tipping"] = tipping[:20]

    # ---- drift across groups
    drift: list[dict] = []
    if groups is not None:
        g = df[groups] if isinstance(groups, str) else pd.Series(np.asarray(groups))
        if len(g) != n_rows:
            raise ValueError(
                f"groups has {len(g)} rows but df has {n_rows}")
        g = g.astype(str)
        levels = [v for v in g.unique() if (g == v).sum() >= 5]
        if len(levels) >= 2:
            group_col = groups if isinstance(groups, str) else None
            g_mask = {v: (g == v).to_numpy() for v in levels}
            scan = [c for c in numeric if c != group_col]
            n_tests = max(len(scan) + len(categorical), 1)
            for c in scan:
                col = numeric[c]
                finite = col[~np.isnan(col)]
                if finite.size == 0 or np.unique(finite).size < 2:
                    continue
                samples = [col[g_mask[v]] for v in levels]
                samples = [s[~np.isnan(s)] for s in samples]
                samples = [s for s in samples if len(s) >= 5]
                if len(samples) < 2:
                    continue
                if len(samples) == 2:
                    stat, p = mannwhitneyu(*samples, alternative="two-sided")
                else:
                    stat, p = kruskal(*samples)
                if p < alpha / n_tests:
                    drift.append({
                        "column": str(c),
                        "p_value": round(float(p), 6),
                        "group_medians": {v: round(float(np.median(col[g_mask[v]][~np.isnan(col[g_mask[v]])])), 4)
                                          for v in levels},
                    })
            # categorical composition drift: chi square on group by
            # category, reported with per group shares of the top level
            for c in categorical:
                if c == group_col:
                    continue
                table = pd.crosstab(g, df[c].astype(str))
                if table.shape[1] < 2 or table.shape[0] < 2:
                    continue
                stat, p, _dof, _exp = chi2_contingency(table)
                if p < alpha / n_tests:
                    top = df[c].value_counts().index[0]
                    drift.append({
                        "column": str(c),
                        "kind": "composition",
                        "p_value": round(float(p), 6),
                        "top_category": str(top),
                        "group_shares": {v: round(float((df.loc[g_mask[v], c].astype(str) == str(top)).mean()), 3)
                                         for v in levels},
                    })
        for d in drift[:5]:
            if d.get("kind") == "composition":
                shares = ", ".join(f"{k}: {v:.0%}"
                                   for k, v in d["group_shares"].items())
                findings.append({
                    "severity": "HIGH",
                    "text": (f"{d['column']} composition drifts across groups "
                             f"(top category {d['top_category']}: {shares}, "
                             f"p={d['p_value']:.2g})"),
                })
            else:
                meds = ", ".join(f"{k}: {v}" for k, v in d["group_medians"].items())
                findings.append({
                    "severity": "HIGH",
                    "text": (f"{d['column']} drifts across groups "
                             f"(p={d['p_value']:.2g}; medians {meds}); "
                             "standardize within group before pooling"),
                })

    # ---- plausibility assertions: domain truths the data must obey
    violations = []
    for spec in (assertions or []):
        try:
            sub = df
            where = spec.get("where")
            if where:
                col_w, val_w = where
                sub = sub[sub[col_w] == val_w]
            if len(sub) == 0 or spec["column"] not in sub.columns:
                violations.append({"name": spec.get("name", "?"),
                                   "error": "empty subset or unknown column"})
                continue
            s = sub[spec["column"]].dropna()
            stat_fn = {"median": np.median, "mean": np.mean,
                       "max": np.max, "min": np.min,
                       "fraction": lambda v: float((v > 0).mean())}
            stat = float(stat_fn[spec["statistic"]](s.astype(float)))
            ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
                   ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}
            ok = ops[spec["op"]](stat, float(spec["value"]))
            if not ok:
                violations.append({
                    "name": spec.get("name", spec["column"]),
                    "statistic": f'{spec["statistic"]}({spec["column"]}'
                                 f'{f" where {where[0]}=={where[1]}" if where else ""})'
                                 f' = {stat:.4f}',
                    "requirement": f'{spec["op"]} {spec["value"]}',
                })
        except Exception as e:  # noqa: BLE001 - malformed assertion
            violations.append({"name": spec.get("name", "?"),
                               "error": f"{type(e).__name__}: {e}"})
    for v in violations:
        findings.append({
            "severity": "HIGH",
            "text": (f"PLAUSIBILITY VIOLATION [{v['name']}]: "
                     f"{v.get('statistic', v.get('error', ''))} "
                     f"required {v.get('requirement', 'valid')}"),
        })

    # ---- join audit
    join_audit = None
    if other_df is not None:
        if key is None:
            raise ValueError("key is required when other_df is given")
        keys = [key] if isinstance(key, str) else list(key)
        left = df[keys].dropna()
        right = other_df[keys].dropna()
        lk, rk = set(map(tuple, left.values)), set(map(tuple, right.values))
        fmt = lambda keys: ["-".join(str(v) for v in t) for t in keys]
        join_audit = {
            "key": keys,
            "left_rows": n_rows,
            "right_rows": int(len(other_df)),
            "left_null_keys": int(len(df) - len(left)),
            "right_null_keys": int(len(other_df) - len(right)),
            "left_duplicate_keys": int(len(left) - len(lk)),
            "right_duplicate_keys": int(len(right) - len(rk)),
            "left_keys_matched": round(len(lk & rk) / max(len(lk), 1), 4),
            "right_keys_matched": round(len(lk & rk) / max(len(rk), 1), 4),
            "unmatched_left_examples": fmt(sorted(lk - rk)[:5]),
            "unmatched_right_examples": fmt(sorted(rk - lk)[:5]),
        }
        null_keys = join_audit["left_null_keys"] + join_audit["right_null_keys"]
        if null_keys:
            findings.append({
                "severity": "WARN",
                "text": (f"{null_keys} rows have null join keys and are "
                         "excluded from the coverage numbers"),
            })
        if join_audit["left_keys_matched"] < 0.9:
            findings.append({
                "severity": "HIGH",
                "text": (f"only {join_audit['left_keys_matched']:.0%} of df keys "
                         f"match other_df on {keys}"),
            })
        if join_audit["right_keys_matched"] < 0.9:
            findings.append({
                "severity": "HIGH",
                "text": (f"only {join_audit['right_keys_matched']:.0%} of "
                         f"other_df keys match df on {keys}"),
            })
        if join_audit["left_duplicate_keys"] or join_audit["right_duplicate_keys"]:
            findings.append({
                "severity": "WARN",
                "text": (f"duplicate join keys: "
                         f"{join_audit['left_duplicate_keys']} left, "
                         f"{join_audit['right_duplicate_keys']} right"),
            })

    if not findings:
        verdict = ("CLEAN PIPE: no constant or duplicate columns, no heavy "
                   "missingness pattern, no MAR suspects"
                   + (", no drift flags" if groups is not None else "")
                   + (", join coverage above 90% both directions"
                      if join_audit is not None else ""))
    else:
        order = {"HIGH": 0, "WARN": 1, "INFO": 2}
        findings.sort(key=lambda f: order[f["severity"]])
        lines = [f"[{f['severity']}] {f['text']}" for f in findings]
        verdict = f"DATA FORENSICS: {len(findings)} findings:\n" + "\n".join(lines)

    return {
        "basics": basics,
        "missingness": missing,
        "mechanism": mechanism,
        "drift": drift[:20],
        "join_audit": join_audit,
        "plausibility": ({"checked": len(assertions or []),
                          "violations": violations} if assertions else None),
        "n_findings": len(findings),
        "verdict": verdict,
        "method": "descriptive_plus_rank_tests",
        "reference": (
            "Little and Rubin (2019) Statistical Analysis with Missing Data "
            "for the MCAR/MAR distinction; Mann Whitney and Kruskal Wallis "
            "rank tests with Bonferroni correction per section"
        ),
        "note": (
            "associations and drift are statistical flags for the analyst, "
            "not causal claims; group medians describe the shift, they do "
            "not explain it; datetimes enter the scans as epoch seconds; "
            "tipping values answer how extreme MNAR values would have to "
            "be, they cannot rule MNAR out"
        ),
    }
