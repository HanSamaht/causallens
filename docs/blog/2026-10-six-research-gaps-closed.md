# Six research gaps closed, one real bug found on the way

October 2026

The first technical report listed six things CausalLens could not do.
Versions 0.3.0 through 0.8.0 closed all six, each pushed separately,
each tested against real Kaggle artifacts. This post is what shipped
and what the artifacts said back.

## 1. Real effect identification (0.3.0)

The intervention tools say what the model does. Users kept asking the
other question: what is the true causal effect, given confounded
data. `identify_real_effect` answers with DoWhy's backdoor and
front-door machinery over the framework graph. Two honest translations
matter: reciprocal interaction pairs (the drift sweep is symmetric)
collapse to correlational edges, and correlation becomes a latent
common cause. A latent U pointing into both treatment and outcome is
refused outright, stricter than DoWhy's own default, which accepts an
empty adjustment set.

On the credit Random Forest: the loan-to-income effect identifies
through a backdoor set of loan amount, income, and rent status.
On the LGBM twin, the same age that carries 40.8 percent attribution
identifies only after adjusting for loan intent and grades, and the
estimate lands near zero.

## 2. Conditional temporal discovery (0.4.0)

Pairwise Granger lets a link survive when it only repeats information
another channel already carries. PCMCI, from Tigramite, conditions
each test on the system's full past. It runs behind
`temporal_causal_graph(method="pcmci")`, with the windowed background
concatenated from non-overlapping windows and the caveat disclosed.

On the gesture GRU it confirmed the single true lagged edge at a
corrected p below 0.05, with model reliance still at 0.1 percent. Same
conclusion as Granger, harder evidence.

## 3. Attribution intervals (0.5.0)

`attribution_confidence` bootstraps the background and reports
percentile intervals plus top-k rank stability. On the credit Random
Forest, all top five features held their ranks in 100 percent of 20
replicates with tight intervals: the ranking there is real evidence,
not a draw.

## 4. The transcoder (0.6.0)

The SAE describes activations. The Crosscoder compares them. The
transcoder says what a block does: encoder on the input side, decoder
on the output side, every feature a virtual neuron. All three
dictionaries now ship activation example panels, the Neuronpedia
payload.

## 5. Active experiment design (0.7.0)

`plan_next_intervention` ranks candidate interventions by expected
information gain over the standing hypotheses: real model calls
preview each candidate, the measured delta calibrates the outcome
probability, and the ranking is disclosed as a simulated lookahead.
On the credit model with three hypotheses it ranks loan-to-income
first at 0.30 nats and tells you which hypothesis each experiment
discriminates. The LLM investigator can call it as a tool.

## 6. Phrase counterfactuals (0.8.0)

Typed edits instead of word swaps: negation toggles, intensity
shifts, antonym swaps, deletions, contrastive clauses, under an edit
budget, every flip verified. Local lexicons, no generation model.

## The bug the artifacts found

Running the transcoder on a real protein CNN produced a suspicious
result: reconstruction error near zero and every causal effect exactly
zero. The cause: 1D convolution outputs are (1, C, L), and the shared
activation flattener misread them as RNN (1, T, H), took the last
"timestep", and silently received the last channel instead. The
transcoder then trained happily on garbage columns. Layer semantics
are now explicit per module type, rnn, conv, or dense, and after the
fix the same CNN produced 512 virtual neurons with 5 causally
verified features, top effect 0.075.

The lesson is the one this project keeps relearning: a tool that
reports clean numbers is not the same as a tool that is right, and
the only reliable detector is a real artifact with known ground
truth.
