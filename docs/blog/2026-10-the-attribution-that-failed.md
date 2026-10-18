# The attribution that failed its own intervention

October 2026

A credit risk model from Kaggle. LightGBM, 22 features, trained on a
standard lending dataset. It approves or denies loans.

Run any explainability tool on it and you get the same headline:

```
person_age attribution: 40.8%
```

Age drives 40.8 percent of this model's lending decisions. That is the
finding. It is also, in most jurisdictions, illegal as a decision
basis, and it is exactly what a fairness review would seize on.

Now run the intervention.

```
do(person_age := 25)  -> prediction unchanged to three decimals
do(person_age := 60)  -> prediction unchanged to three decimals
mean total effect across 40 applicants: 0.0000
```

The model's attribution says age. The model's behavior says age does
not matter. Both statements are about the same model, and they cannot
both be the story.

## What is actually going on

Attribution methods like SHAP decompose a prediction into feature
contributions. They are bookkeeping: who gets credit for this
particular number. They are not experiments. Nothing was changed,
nothing was measured, no intervention happened.

When SHAP mass concentrates on a feature that the model barely responds
to under intervention, you have learned something important: the
feature correlates with the things the model actually uses, so the
attribution rides the correlation. Age is correlated with employment
length, income, and home ownership in this dataset. The model reads the
correlates. SHAP bills the proxy.

A tool that stops at the 40.8 percent number hands you the proxy as the
finding. A tool that runs the do() experiment hands you the finding and
its refutation, and points you toward the real drivers.

For contrast, the Random Forest trained on the identical data:

| | Random Forest | LightGBM |
|---|---|---|
| Top factor | Loan-to-income ratio, 34.2% | Personal age, 40.8% |
| Age intervention effect | 0.0049 | 0.0000 |
| Structure discovery says | loan-to-income causes the output | age carries the attribution |

Same data, same features, one model reads debt pressure and its
structure discovery agrees with its behavior. The other bills age and
its behavior disagrees.

## Why we built CausalLens this way

CausalLens is an interrogation engine. It treats a model as a witness
that cannot lie under experiment:

- every claim traces to an intervention or is labeled correlational,
  enforced in the type system
- an LLM investigator can propose hypotheses, but server-side
  arithmetic owns every probability update, with bounded likelihood
  ratios and a 5:1 odds threshold before any conclusion is allowed
- priors are welcome but bounded: a domain expert's causal graph moves
  hypothesis odds by at most ln(1.5), and every nudge is logged
- everything lands in a chain-hashed SQLite ledger, so the audit trail
  itself is tamper-evident

The age story is our flagship demo because it is the honest version of
what explainability should have been doing all along: run the
experiment, then report.

## Try it

```
pip install causallens
```

```python
from causallens.core import MicSession

# load your model, then:
result = session.fairness_path_decomposition("person_age", 25.0, 60.0)
print(result["verdict"])
```

On the LightGBM above the verdict reads: NEGLIGIBLE, total effect
0.0000, below the materiality bar. On your model it might read DIRECT
DISCRIMINATION RISK with the direct share named. Either way you will
know which one, and why.
