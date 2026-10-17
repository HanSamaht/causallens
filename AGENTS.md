# CausalLens Project Rules (Mandatory for all AI agents)

## Absolute rule: English only

ALL files must contain ZERO Chinese characters. This includes code, comments, docstrings, variable names, string literals, test assertions, README, documentation, configuration files, concept packs. No exceptions. No Chinese punctuation either. Before any commit, verify zero Chinese. If any Chinese is found, the commit is blocked.

## Writing style

- No exclamation marks
- No em dashes or en dashes. Use colons, commas, or periods.
- No dramatic language
- No rhetorical questions
- No emotional appeals
- No unnecessary bold text
- State facts plainly: subject, verb, object, number
- Let numbers speak. Do not interpret them for the reader.
- Remove filler words
- One sentence says one thing
- Write like a technical report

## Bug logging

1. Every error, bug, test failure must be logged to BUG_REGISTRY.md immediately
2. Log entry before fixing; update after resolution
3. Any finding must be tagged: framework tool output / manual analysis / third-party source
4. Never skip test regression

## Other

- API keys only in .env
- Diagnostic output to stderr
- Large data never enters LLM conversation
- Evidence graded: interventional / observational / prior / manual
- Audit only, do not prevent
- License: AGPL-3.0
