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
    tests compare every other numeric column between missing and
    observed rows; an association means missing at random is unlikely
    and the column is MAR suspect
  - drift: Kruskal Wallis across group labels (for example draft
    year) per numeric column
  - join audit: key uniqueness and two directional coverage between
    two frames

The verdict orders findings by severity. Associations and drift are
statistical flags, not causal claims; the note says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def data_forensics(
    df: pd.DataFrame,
    groups=None,
    other_df: pd.DataFrame | None = None,
    key: str | list[str] | None = None,
    alpha: float = 0.05,
    max_assoc_columns: int = 10,
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
    numeric = _numeric_columns(df)
    assoc: list[dict] = []
    targets = [c for c in df.columns if na.get(c, {"frac": 0})["frac"] >= 0.05
               and na.get(c, {"frac": 0})["frac"] <= 0.95]
    for t in targets:
        miss = df[t].isna()
        others = [c for c in numeric if c != t and df.loc[~miss, c].notna().any()]
        others = sorted(others)[:max_assoc_columns]
        if not others or miss.sum() < 10 or (~miss).sum() < 10:
            continue
        n_tests = len(others)
        for c in others:
            a = df.loc[miss, c].dropna()
            b = df.loc[~miss, c].dropna()
            if len(a) < 10 or len(b) < 10:
                continue
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            if p < alpha / n_tests:
                assoc.append({
                    "missing_in": str(t),
                    "differs_by": str(c),
                    "p_value": round(float(p), 6),
                    "median_when_missing": round(float(a.median()), 4),
                    "median_when_observed": round(float(b.median()), 4),
                })
    mechanism = {"mar_suspects": assoc[:20], "n_targets_tested": len(targets)}
    for a in assoc[:5]:
        findings.append({
            "severity": "HIGH",
            "text": (f"missingness in {a['missing_in']} associates with "
                     f"{a['differs_by']} (p={a['p_value']:.2g}); not MCAR"),
        })

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
            scan = [c for c in numeric if c != group_col]
            n_tests = max(len(scan), 1)
            for c in scan:
                samples = [df.loc[g == v, c].dropna() for v in levels]
                samples = [s for s in samples if len(s) >= 5]
                if len(samples) < 2 or df[c].nunique() < 2:
                    continue
                if len(samples) == 2:
                    stat, p = mannwhitneyu(*samples, alternative="two-sided")
                else:
                    stat, p = kruskal(*samples)
                if p < alpha / n_tests:
                    drift.append({
                        "column": str(c),
                        "p_value": round(float(p), 6),
                        "group_medians": {v: round(float(df.loc[g == v, c].median()), 4)
                                          for v in levels},
                    })
        for d in drift[:5]:
            meds = ", ".join(f"{k}: {v}" for k, v in d["group_medians"].items())
            findings.append({
                "severity": "HIGH",
                "text": (f"{d['column']} drifts across groups "
                         f"(p={d['p_value']:.2g}; medians {meds}); "
                         "standardize within group before pooling"),
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
            "left_duplicate_keys": int(len(left) - len(lk)),
            "right_duplicate_keys": int(len(right) - len(rk)),
            "left_keys_matched": round(len(lk & rk) / max(len(lk), 1), 4),
            "right_keys_matched": round(len(lk & rk) / max(len(rk), 1), 4),
            "unmatched_left_examples": fmt(sorted(lk - rk)[:5]),
            "unmatched_right_examples": fmt(sorted(rk - lk)[:5]),
        }
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
            "not explain it"
        ),
    }
