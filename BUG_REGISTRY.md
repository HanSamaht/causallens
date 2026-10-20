
## 2026-10: six v0.17-0.18 tools absent from agent schemas

cate_heterogeneity, double_ml_effect, omitted_variable_sensitivity,
panel_fixed_effects, mediation_chain, and holdout_evaluation were
added to the interrogator dispatch table but not to tool_schemas(),
meaning the LLM could execute them by name but could not discover
them from the schema list. Fixed by adding all six schemas. Found in
the v0.18.0 adversarial review.

## 2026-10: cluster sandwich computed White HC0 instead of cluster-robust

panel_fixed_effects computed the meat as the sum of individual outer
products X_i e_i times (X_i e_i)', which is the White/
heteroskedasticity sandwich. The correct cluster sandwich requires
the sum of GROUP-level score outer products: each term is
(X_g' e_g)(X_g' e_g)', capturing the cross-products of residuals
WITHIN each group. The bug discarded all within-group covariance,
making cluster SEs wrong in both directions depending on the sign of
within-group correlation. Verified against statsmodels: the fix
matches to 7 decimal places (difference 1.4e-7). Found in the v0.18.0
adversarial review. Apology issued per project rules.
# Bug Registry

## 2026-10: git add -A swept local-only workspaces into the rename push

The intervense rename commit was staged with git add -A, which
included nfl_market data (300MB of parquet), kernel logs, and the
competition submission workspace, against the standing instruction
to keep competition-specific work local. Fixed by soft reset, restage
of 137 rename-only paths, force push of clean history, and gitignore
entries for every local-only path. Apology issued per project rules.
