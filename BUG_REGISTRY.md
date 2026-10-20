# Bug Registry

## 2026-10: git add -A swept local-only workspaces into the rename push

The intervense rename commit was staged with git add -A, which
included nfl_market data (300MB of parquet), kernel logs, and the
competition submission workspace, against the standing instruction
to keep competition-specific work local. Fixed by soft reset, restage
of 137 rename-only paths, force push of clean history, and gitignore
entries for every local-only path. Apology issued per project rules.
