# Audit hardening: cohort folds, permutation nulls, automated forensics

Benchmarking against the strongest public NFL Big Data Bowl entries
showed the audits three things: a good rival runs leave-one-cohort-out
validation and permutation tests, and every careful team does its data
forensics by hand. Version 0.14.0 folds all three into the framework.

## Cohort fold stability

`specification_audit(treatment, y, groups=...)` now refits the lean
and full specifications inside each cohort (groups with at least 20
rows) and compares per cohort effects against the dominant material
fold sign. The failure it catches is nasty: an effect positive in two
cohorts and negative in a third cancels toward zero in the pool, looks
like noise, and no pooled verdict will ever name the problem. The fold
block returns FOLD FRAGILE with the reversing cohort named. On the
planted scenario the pooled estimate lands inside the chance band
while the fold effects are +2.5, +2.5, -5.1: the audit says which one
is the liar.

The semantics are effect heterogeneity, not out of sample prediction:
each cohort gets its own fit, and the comparison is sign against
materiality. Small cohorts are skipped and disclosed.

## Permutation nulls for reliance shares

`shortcut_diff(y, n_perm=20)` refits the driver's family on shuffled
outcomes and recomputes reliance shares per replicate. Each observed
share gets a p value against that baseline, and the baseline is the
right one: a gradient boosting model fitted on shuffled labels still
distributes reliance across features, so the null median share is
about one third on a three feature model, not zero. On the label echo
scenario the echo tests at 80 percent observed versus 34 percent null
(p = 1/21); the true cause at 18 percent sits below the null median,
honest evidence that its share is compressed by the echo, not absent.

The null is skipped with a disclosure when the family cannot be
refitted, and disabled with n_perm=0.

## Data forensics in one call

`data_forensics(df, groups=..., other_df=..., key=...)` runs the
checks that used to be an afternoon: constant and duplicate columns,
missingness inventory with the dominant pattern, MCAR versus MAR hints
(Mann Whitney between missing and observed rows, Bonferroni per
target), Kruskal Wallis drift across cohort labels, and a two
directional join audit with example unmatched keys.

On the real NFL cohort it reproduces in seconds what the manual
analysis found by hand: draft pick missingness associates with career
snaps (undrafted players get fewer opportunities, medians 14 versus
315), drill features drift across draft years, traditional drills are
over 50 percent missing, join coverage is 97 percent one direction.
The verdict is severity ordered, and the note says what the flags are
not: associations and drift are prompts for the analyst, not causal
claims.

## The pattern

Each upgrade came from watching a strong rival do something the
framework did not, or from noticing that disciplined manual work had
no tool support. That is the honest way an audit framework should
grow: not features for a list, but closes for observed gaps, each one
validated on planted scenarios and, where possible, on the real data
that motivated it.
