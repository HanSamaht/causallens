"""Phrase level counterfactual search with typed edit constraints.

The token swap counterfactual answers "which word flips this". Real
users ask "which phrase change flips this", and a bare word
substitution often produces text no human would write. This module
searches over typed phrase edits under an edit budget:

  negation_toggle   insert or strip a negator at a hotspot word
  intensity_shift   swap an intensifier for a downtoner, or delete it
  antonym_swap      swap a hotspot word with its antonym
  phrase_delete     delete a hotspot word entirely
  clause_append     append a contrastive discourse clause

Every candidate is verified against the model before it is reported.
English only, small shipped lexicons, no generation model, no API.
Lineage: Polyjuice control codes, Wu et al. 2021.
"""
from __future__ import annotations

ANTONYMS = {
    "good": "bad", "bad": "good", "great": "terrible", "terrible": "great",
    "love": "hate", "hate": "love", "like": "dislike", "dislike": "like",
    "best": "worst", "worst": "best", "excellent": "awful", "awful": "excellent",
    "amazing": "dreadful", "dreadful": "amazing", "happy": "sad", "sad": "happy",
    "recommend": "avoid", "avoid": "recommend", "worth": "waste", "waste": "worth",
    "boring": "exciting", "exciting": "boring", "interesting": "dull",
    "dull": "interesting", "beautiful": "ugly", "ugly": "beautiful",
}

INTENSIFIERS = {
    "very": "slightly", "really": "somewhat", "extremely": "a bit",
    "incredibly": "barely", "absolutely": "hardly", "totally": "barely",
}

NEGATORS = {"not", "never", "no", "hardly", "barely"}

STOP = {"the", "a", "an", "is", "was", "are", "were", "it", "this",
        "that", "i", "we", "you", "they", "of", "to", "and", "but",
        "in", "on", "at", "as", "for", "with", "movie", "film"}

CLAUSES_POS = ["but overall I loved it", "though I would recommend it"]
CLAUSES_NEG = ["but it was disappointing", "though I would not recommend it"]


def _clean(word: str) -> str:
    return word.lower().strip(".,!?;:")


def _hotspots(text: str, driver, n: int = 6) -> list[int]:
    """Indices of words the model cares about, by attribution heat."""
    words = text.split()
    try:
        attr = driver.token_attribution(text, method="occlusion")
        hot = [r["token"].lower() for r in attr["tokens"][:10]]
    except Exception:
        hot = []
    scored = []
    for i, w in enumerate(words):
        stem = _clean(w)
        if not stem or stem in STOP:
            continue
        heat = 1 if not hot else max(
            (1 if stem == t or (len(t) > 3 and t in stem) else 0) for t in hot
        )
        scored.append((-heat, len(stem), i))
    scored.sort()
    return [i for _, _, i in scored[:n]]


def _edit(lo: int, consume: int, repl: list[str], etype: str, word: str) -> dict:
    return {"lo": lo, "consume": consume, "repl": repl, "type": etype, "word": word}


def _single_edits(text: str, driver) -> list[dict]:
    words = text.split()
    edits: list[dict] = []
    for i in _hotspots(text, driver):
        raw = words[i]
        stem = _clean(raw)
        tail = raw[len(stem):] if raw.lower().startswith(stem) else ""

        if stem in ANTONYMS:
            edits.append(_edit(i, 1, [ANTONYMS[stem] + tail], "antonym_swap", stem))
        if stem in INTENSIFIERS:
            edits.append(_edit(i, 1, [INTENSIFIERS[stem] + tail], "intensity_shift", stem))
            edits.append(_edit(i, 1, [], "intensity_shift", stem + " (deleted)"))
        if stem in NEGATORS:
            edits.append(_edit(i, 1, [], "negation_toggle", stem + " (removed)"))
        elif i == 0 or _clean(words[i - 1]) not in NEGATORS:
            edits.append(_edit(i, 0, ["not"], "negation_toggle", "not + " + stem))
        edits.append(_edit(i, 1, [], "phrase_delete", stem + " (deleted)"))

    # contrastive clause appends, one per direction
    edits.append(_edit(len(words), 0, ["but", "overall", "I", "loved", "it"],
                       "clause_append", "positive clause"))
    edits.append(_edit(len(words), 0, ["but", "it", "was", "disappointing"],
                       "clause_append", "negative clause"))
    return edits


def _apply(words: list[str], edit: dict) -> list[str]:
    return words[:edit["lo"]] + edit["repl"] + words[edit["lo"] + edit["consume"]:]


def _apply_two(words: list[str], a: dict, b: dict) -> list[str] | None:
    """Apply two edits at disjoint positions, later index shifted."""
    first, second = (a, b) if a["lo"] <= b["lo"] else (b, a)
    out = _apply(words, first)
    shift = len(first["repl"]) - first["consume"]
    moved = {**second, "lo": second["lo"] + shift}
    # overlap after the first edit: refuse
    span_a_end = first["lo"] + len(first["repl"])
    if first["consume"] > 0 or moved["consume"] > 0:
        if moved["lo"] < span_a_end and moved["lo"] + moved["consume"] > first["lo"]:
            return None
    return _apply(out, moved)


def phrase_counterfactual(driver, text: str, target_label: int | None = None,
                          max_edits: int = 2, top_k: int = 3) -> dict:
    """Find phrase level edits that flip the prediction, under a budget.

    Single edits first, then pairs at disjoint positions. Every
    reported flip is verified against the model.
    """
    p_base = float(driver._forward(text))
    base_label = 1 if p_base > 0.5 else 0
    if target_label is None:
        target_label = 1 - base_label

    def verified(cand: str):
        p = float(driver._forward(cand))
        return (1 if p > 0.5 else 0) == target_label, p

    words = text.split()
    singles = _single_edits(text, driver)
    found = []
    n_verified = 0

    for e in singles:
        cand = " ".join(_apply(words, e))
        n_verified += 1
        ok, p = verified(cand)
        if ok:
            found.append({
                "text": cand, "type": e["type"], "word": e["word"],
                "n_edits": 1, "prediction": round(p, 4),
            })

    if not found and max_edits >= 2:
        for ai, a in enumerate(singles):
            for b in singles[ai + 1:]:
                if a["lo"] == b["lo"]:
                    continue
                merged_words = _apply_two(words, a, b)
                if merged_words is None:
                    continue
                cand = " ".join(merged_words)
                n_verified += 1
                ok, p = verified(cand)
                if ok:
                    found.append({
                        "text": cand, "type": f"{a['type']} + {b['type']}",
                        "word": f"{a['word']}, {b['word']}",
                        "n_edits": 2, "prediction": round(p, 4),
                    })
                if len(found) >= top_k * 3:
                    break
            if len(found) >= top_k * 3:
                break

    found.sort(key=lambda r: (r["n_edits"], -abs(r["prediction"] - p_base)))
    return {
        "original_text": text,
        "original_prediction": round(p_base, 4),
        "flipped": bool(found),
        "counterfactuals": found[:top_k],
        "n_candidates_verified": n_verified,
        "method": "phrase_edits_constrained",
        "reference": "Polyjuice control codes, Wu et al. (2021)",
        "note": (
            "typed phrase edits: negation_toggle, intensity_shift, "
            "antonym_swap, phrase_delete, clause_append; English only, "
            "local lexicons, every flip verified against the model"
        ),
    }
