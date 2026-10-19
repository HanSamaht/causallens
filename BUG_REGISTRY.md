
## 2026-10: NaT datetimes converted to the int64 sentinel, not NaN

_numeric_view cast datetime columns with astype(int64), which turns
missing timestamps into the sentinel value of about minus 9.2 billion
seconds (year 1677) instead of NaN. Drift medians, MAR witness tests,
and MNAR tipping silently absorbed the garbage. Fixed by masking the
NaT positions to NaN after the cast; regression test added. Apology
issued per project rules. Found in the post 0.15.2 adversarial sweep.
# Bug Registry

> Rule: every bug discovered or fixed must be logged here immediately. See AGENTS.md.
> Format: `ID | Date | Area | Symptom | Root Cause | Fix | Status`

## Fixed

- BUG-001 | core | ImportError circular dependency | core/__init__ eagerly imports session, creating a cycle | PEP 562 lazy __getattr__ | Closed
- BUG-002 | server/mcp | MCP client JSON parse failure | Diagnostic output to stdout polluted stdio JSON-RPC channel | All diagnostics to stderr | Closed
- BUG-003 | analysis/attribution | Explainer cache ID collision after GC | Mod
