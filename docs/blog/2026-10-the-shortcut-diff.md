# The shortcut diff: right answers riding the wrong features

A model can score beautifully on the validation set and still be a
time bomb, because part of its accuracy rides on correlations that
carry no causal path to the outcome. When the population shifts, that
part breaks and nobody warned you. The shortcut diff, new in
Intervense 0.13.0, is the warning.

## Two graphs, one call

`shortcut_diff(y)` builds two structures from the same artifact.

The world graph comes from structure discovery (PC, GES, or
DirectLiNGAM) over the feature background plus the real outcome you
supply. It says which features have directed paths to the outcome and
which are merely adjacent. Domain knowledge edges can pin the
orientations the CPDAG leaves undirected, and the disclosure says
which edges are assumptions.

The model-implied reliance comes from intervention contrasts on the
frozen model: move each feature from its p10 to its p90 and measure
how much the prediction moves. That is what the model uses, not what
an attribution method says it uses.

The diff classifies every relied upon feature:

- CAUSAL PATH: a directed path to the outcome exists. Structurally
  supported reliance.
- PROXY: the outcome has a directed path into the feature. The model
  reads a consequence of the label. Whether that is leakage or a
  legitimate pre-outcome measurement depends on timing, and the
  verdict says so instead of guessing.
- UNRESOLVED PATH: connectivity only through undirected edges. The
  data cannot license the direction.
- INDEPENDENT: no connection at all. The model leans on something the
  discovered structure says is unrelated to the outcome.

The headline number is the reliance at risk: the share of
interventional reliance sitting on anything that is not a causal path.

## Three planted worlds

On a clean two-cause model the audit returns SUPPORTED, 100 percent of
reliance on causal paths.

Add a spurious sibling (a feature riding the same latent cause as the
real driver) and an overfit noise feature: the sibling lands UNRESOLVED
PATH at 12 percent, the noise feature lands INDEPENDENT at 4 percent,
verdict SHORTCUT RISK. Those are exactly the two flavors that break
differently: the sibling breaks when the population's correlation
structure changes, the noise feature is already broken and only
memorization is holding it.

Add a label echo, a feature that is the outcome plus noise: PROXY at
84 percent reliance while the true cause carries 12 percent. The model
mostly reads its own answer sheet. No accuracy metric on the training
distribution will ever flag this; the graph direction does.

## What this is not

It is not a new definition of shortcut learning. Bellamy et al. (2022)
characterized shortcut features in causal DAG terms, Sahidullah et al.
(2025) built an intervention framework for shortcut discovery in black
box classifiers, Geirhos et al. (2020) named the failure mode. The
contribution is the packaging: both graphs from one call, per feature
classification with explicit epistemic grades, reliance share
accounting, and refusal style disclosure of what CPDAG orientations
cannot decide. An audit that silently upgrades assumptions into facts
would be the thing it exists to catch.

```python
session.register_data("y", y_train.tolist())
out = session.shortcut_diff(y_train, edges=[("x", "outcome")])
print(out["verdict"], out["reliance_at_risk"])
```

The specification audit asked whether your effect survives
respecification. The shortcut diff asks whether your reliance survives
the world. Between the two, the two most common ways a multivariate
model lies to its owner now have names, shares, and verdicts.
