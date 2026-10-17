# Bug Registry

> Rule: every bug discovered or fixed must be logged here immediately. See AGENTS.md.
> Format: `ID | Date | Area | Symptom | Root Cause | Fix | Status`

## Fixed

- BUG-001 | core | ImportError circular dependency | core/__init__ eagerly imports session, creating a cycle | PEP 562 lazy __getattr__ | Closed
- BUG-002 | server/mcp | MCP client JSON parse failure | Diagnostic output to stdout polluted stdio JSON-RPC channel | All diagnostics to stderr | Closed
- BUG-003 | analysis/attribution | Explainer cache ID collision after GC | Mod
