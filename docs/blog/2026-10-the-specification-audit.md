# The specification audit: when the control set decides your sign

In October 2026 we were finishing an analysis for the NFL Big Data Bowl:
does Combine drill speed predict NFL pressure rate? The lean model said
yes, +0.024 per p10 to p90 move. The 16-feature model said no, -0.004.
Same data, same family, same weights. Feature importance, meanwhile,
looked perfectly calm about the whole thing.

The standard workflow has no step that catches this. You read the
importance ranking, the ranking looks sane, you move on. So we built the
step the workflow was missing, and this post is about what it found,
including what it found about our own earlier claims.

## The method

`specification_audit(treatment, y)` answers one question: does the
intervention effect keep its sign when the model is re-specified?

For each control subset S (the specification lattice), it refits the
driver's own model family on treatment plus S against the training
outcome y, computes the mean do-contrast from p10 to p90, and collects
the curve. Three layers of discipline sit on top:

- a chance band, max(2 percent of sd(y), 2.5 times the chance contrast
  sd_y/sd_t times width over sqrt(n)), so noise level sign jitter never
  counts as a flip
- minimal flip sets, the smallest control subsets whose inclusion flips
  the sign, found by breadth first search up to depth 3: the audit names
  which variables kill your finding
- a closed form for linear and ridge families, where the whole curve is
  exact arithmetic instead of refits

The verdicts are STABLE SIGN, FRAGILE SIGN, and NEGLIGIBLE, and each
one says which specifications disagree.

The provenance is honest: specification curve analysis (Simonsohn,
Simmons, Nelson 2020) and multiverse robustness have done this for
regression coefficients in the social sciences for years. What was
missing is the application to do-contrasts of black-box models, with
chance-band discipline and minimal flip sets as an audit product. That
intersection is what CausalLens ships.

## What it did to our own finding

We pointed it back at the NFL analysis that motivated it. The audit
refit the ridge family on 200 sampled control subsets, weighted exactly
like the production fit. Result: the drill-speed effect is positive in
28 of 200 specifications above the chance band, negligible in the rest,
and materially negative in zero of them.

Read that again. Our own writeup said collinearity "flips the sign."
The audit said no: the point estimate touches negative once, inside the
noise band, and the honest description is collapse, not flip. We
corrected the writeup, the notebook, and the figure captions. The tool
that generalizes the catch also polices the story told about the catch.

That sentence is the whole point of the framework. Importance rankings
do not self-correct. Point estimates do not self-correct. A curve over
the specification lattice, with a chance band drawn on it, does.

## Where the synthetic case lands

On a planted suppressor structure (a treatment that helps, a
collinear sibling that hurts, both riding a common cause) the audit
returns FRAGILE SIGN with minimal flip set exactly the sibling, and the
closed-form lean effect matches textbook algebra to sampling noise.
When features are independent, every audit comes back STABLE SIGN, and
a pure noise feature comes back NEGLIGIBLE because the chance band is
scaled to the sample size. The tests pin all three behaviors.

## Using it

```python
session.register_data("y", y_train.tolist())
out = session.specification_audit("drill_speed", y_train)
print(out["verdict"], out["minimal_flip_sets"])
```

The audit needs the training outcome supplied by the caller, because
specification search must fit to the real outcome. Distilling to the
frozen model's predictions would erase the noise structure that
produces collapses in the first place, and then the audit would be
auditing a world with no sampling noise in it, which is not the world
your model ships in.

V1 refits sklearn estimators and pipelines. Linear and ridge families
get the exact closed-form path. The method, references, and every
threshold are disclosed in the returned dict, because an audit whose
defaults are invisible is just another black box.
