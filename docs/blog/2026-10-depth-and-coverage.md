# Depth and coverage: graph uncertainty, MNAR bounds, categorical witnesses, the audit playbook

Version 0.15.0 closes the gaps the adversarial review of 0.14.1
exposed. Four upgrades, each with a test pinning the behavior.

## The CPDAG is estimated, so classifications now say so

The shortcut diff classified features against a graph discovered from
finite data, and until now that graph was treated as the truth. It is
not. Version 0.15.0 bootstraps the world graph: every replicate
resamples the rows, re-runs discovery, and reclassifies. Each
classification carries class_prob, its agreement across replicates;
features below 80 percent are named unstable in the verdict. User
edges are domain knowledge and pin orientations in every replicate.

On the label echo scenario the pinned classifications hold at 100
percent across replicates, which is the right answer. The interesting
cases are the ambiguous ones, where a PROXY at 60 percent agreement is
now visibly a coin flip instead of a confident label.

## The MNAR answer: a bound, not a denial

Every MAR finding invites the same question: what if the missingness
depends on the missing values themselves. Nobody can rule that out
from observed data. What data forensics now reports is the tipping
bound: for each partly missing numeric column, the uniform shift, in
observed SD units, the missing entries would need in order to move the
full column mean by half a standard deviation, plus the tipping value
in column units. On the NFL cohort, bench press at 54 percent missing
needs a +0.46 SD shift to tip the mean half a SD: readers can decide
for themselves whether that is plausible. The note says plainly that
this bounds sensitivity, it does not rule MNAR out.

## Categorical and temporal columns join the scans

MAR witnesses now include categorical columns (chi square on the
missingness by category table, trimmed to its nonzero core: a strong
association leaves few nonzero cells and a naive cell count rejects
exactly the strongest signals, a bug caught during development).
Datetime columns enter the drift scan as epoch seconds. Categorical
columns get composition drift: chi square across groups with per group
top category shares, which is how a recording scheme change across
cohots actually shows up. The constant column guard that a Kruskal
Wallis edge case demanded is back and tested.

## Torch families and the self driving auditor

specification_audit accepts estimator_factory, a callable returning an
unfitted fit and predict estimator, so model families sklearn cannot
clone can still be specification audited; the refusal message without
a factory names the escape hatch.

And the auditor role prompt now encodes the playbook the framework
exists to enforce: data forensics before any modeling claim,
shortcut_diff after the first attribution, specification_audit before
quoting any feature effect, and a forbidden list entry for quoting an
effect with no specification verdict behind it. The AI investigator
now runs the discipline in order without being told, which is the
difference between a toolbox and an auditor.
