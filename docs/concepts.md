# Concepts

The evidence model behind every claim Intervense makes.

## Interrogation, not explanation

An explanation describes. An interrogation intervenes. Intervense treats
the model as a witness that cannot lie under do() experiments: change an
input, hold the rest fixed, measure the response. Every claim in a report
traces to such an experiment or is labeled correlational.

## Evidence tiers

| Tier | Meaning | Example |
|---|---|---|
| interventional | measured under do() | `intervene(0, {"age": 60})` |
| observational | measured without intervention | background correlation |
| prior | domain knowledge, bounded | causal graph prior |
| manual | stated by a human | annotated finding |

Observational evidence may never carry a causal conclusion. The rule is
enforced in the type system (`CausalEdge.kind`) and in the LLM prompts.

## Six depth levels

1. Attribution: which features move the prediction
2. Causal graph: how features relate, with structure discovery backends
3. Mediation: direct effects versus mediated paths
4. Circuits: which units carry the behavior, with bootstrap confidence
5. Parameters: which weights matter, reversible lesion studies
6. Dictionaries: SAE features and Crosscoder concepts across layers and
   models

Each level answers questions the previous one raises.

## The investigator loop

The LLM layer never decides. It proposes hypotheses, and a server-side
HypothesisTracker owns the arithmetic:

- hypotheses are 2 to 4 competing, falsifiable statements
- every evidence update is a bounded likelihood ratio step, cap 5,
  floor 0.2
- conclusions require posterior odds of at least 5 to 1
- below the threshold the verdict is honestly inconclusive

Causal priors (domain knowledge about which edges should exist) are
injected explicitly: bounded to ln(1.5) of log-odds per hypothesis,
logged as evidence, then tested against interventions.

## The evidence ledger

A SQLite ledger chain hashes every record. Hypotheses are bitemporal:
a finding that held in v3 and died in v4 stays queryable. History
becomes prior: confirmed hypotheses from earlier versions seed the next
investigation.

## Attribution is a lower bound

SHAP attribution is treated as a lower bound on causal effect, never as
the effect itself. When the two disagree, that disagreement is a
finding, not noise: see the age case in the
[technical report](report.md).
