"""Phrase counterfactual tests with a local lexicon driver.

A tiny sentiment scorer stands in for a text model so the typed edit
search runs offline: positive words push the score up, negative words
down, intensifiers scale their neighbor. Ground truth flips are known.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.analysis.phrase_cf import ANTONYMS, phrase_counterfactual
from causallens.drivers.text import TextDriver


class LexiconScorer(TextDriver):
    """Offline stand-in: no tokenizer, pure lexicon scoring."""

    POS = {"love", "great", "good", "best", "excellent", "amazing",
           "happy", "recommend", "worth", "exciting", "interesting",
           "loved", "beautiful"}
    NEG = {"hate", "terrible", "bad", "worst", "awful", "dreadful",
           "sad", "avoid", "waste", "boring", "dull", "disappointing"}

    def __init__(self):  # skip the HF constructor entirely
        self.n_classes = 2
        self.target_class = 1

    def _forward(self, text):
        words = [w.lower().strip(".,!?;:") for w in text.split()]
        score = 0.0
        for i, w in enumerate(words):
            value = 1.0 if w in self.POS else (-1.0 if w in self.NEG else 0.0)
            if i > 0 and words[i - 1] in {"very", "really", "extremely",
                                          "incredibly", "absolutely"}:
                value *= 2.0
            if i > 0 and words[i - 1] in {"slightly", "somewhat", "barely",
                                          "hardly"}:
                value *= 0.3
            if i > 0 and words[i - 1] in {"not", "never"}:
                value *= -1.0
            score += value
        return 1.0 / (1.0 + pow(2.718281828, -2.0 * score))

    def token_attribution(self, text, method="occlusion"):
        words = text.split()
        base = self._forward(text)
        tokens = []
        for i, w in enumerate(words):
            masked = " ".join(words[:i] + words[i + 1:])
            tokens.append({"token": w, "attribution": abs(self._forward(masked) - base)})
        tokens.sort(key=lambda t: -t["attribution"])
        return {"tokens": tokens}


@pytest.fixture(scope="module")
def scorer():
    return LexiconScorer()


def test_antonym_swap_flips_positive(scorer):
    out = phrase_counterfactual(scorer, "I love this great movie, it was really good")
    assert out["flipped"]
    first = out["counterfactuals"][0]
    assert first["n_edits"] <= 2
    assert first["type"] in ("antonym_swap", "antonym_swap + antonym_swap",
                             "antonym_swap + negation_toggle",
                             "antonym_swap + intensity_shift",
                             "antonym_swap + phrase_delete")
    # the reported flip must actually flip the scorer
    assert (scorer._forward(first["text"]) > 0.5) != (scorer._forward(
        "I love this great movie, it was really good") > 0.5)


def test_negation_toggle_flips(scorer):
    out = phrase_counterfactual(scorer, "This film is interesting and beautiful")
    assert out["flipped"]
    types = [c["type"] for c in out["counterfactuals"]]
    assert any("negation_toggle" in t or "antonym_swap" in t or
               "clause_append" in t for t in types)


def test_no_garbage_candidates(scorer):
    """Every reported text is a clean edit of the original, not a rebuild."""
    original = "I love this great movie"
    out = phrase_counterfactual(scorer, original)
    for c in out["counterfactuals"]:
        assert isinstance(c["text"], str)
        # word overlap must stay high: edits are local
        overlap = len({w.lower().strip(".,!?;:") for w in c["text"].split()}
                      & {w.lower().strip(".,!?;:") for w in original.split()})
        assert overlap >= len(original.split()) - 4


def test_neutral_text_flips_only_via_clause(scorer):
    """A neutral sentence has no sentiment words to edit, so the only
    legitimate flips are contrastive clause appends, never invented
    word swaps."""
    out = phrase_counterfactual(scorer, "The chair is in the room")
    for c in out["counterfactuals"]:
        assert "clause_append" in c["type"]
        assert "but" in c["text"]


def test_antonym_lexicon_pairs():
    for w, a in ANTONYMS.items():
        assert ANTONYMS[a] == w
