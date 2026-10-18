# CausalLens

Make all black-box models transparent.

CausalLens is a causal reasoning engine that interrogates any model with a
`predict` function. It does not passively explain; it runs intervention
experiments, performs Bayesian hypothesis testing, and enforces evidence
chain discipline.

You give it a model file and background data. It tells you:

- What drives the model's decisions, by causal attribution, not correlation
- What changes would flip a prediction, by counterfactual search
- Which internal units carry the task, by circuit discovery and Crosscoder
  dictionaries
- Whether the explanation itself is trustworthy, by faithfulness verification
- Whether the audit trail is intact, by a chain hashed evidence ledger

Correlation tells you what happens together. Causality tells you what happens
when you intervene. CausalLens does the latter.

## Install

```
pip install causallens
```

Optional extras: `causallens[torch]` for activation patching and circuit
discovery, `causallens[causal-learn]` for structure discovery backends,
`causallens[all]` for everything.

## The 60-second tour

```python
from causallens.core import MicSession
from causallens.demos import train_finance_demo
from causallens.drivers import detect_driver

model, feats, background, auc = train_finance_demo()
session = MicSession(detect_driver(model, feats, background))

# what drives sample 0
print(session.get_attribution(0))

# what would change the decision: a real intervention
print(session.intervene(0, {"momentum": 2.0}))

# the full checkup, no LLM required
print(session.offline_report(sample_index=0))
```

## Where to go next

- [Quickstart](quickstart.md): from zero to your own model in ten minutes
- [Concepts](concepts.md): the evidence model behind every claim
- [API Reference](api.md): the full session surface
- [Tools](tools.md): all interrogation tools grouped by depth level
- [Technical Report](report.md): method, benchmarks, and Kaggle case studies
