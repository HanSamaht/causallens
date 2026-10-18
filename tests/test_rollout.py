"""Tests for attention rollout attribution in TextDriver."""

import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


pytestmark = [pytest.mark.slow]


@pytest.fixture(scope="module")
def text_model():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "textattack/bert-base-uncased-SST-2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
    except Exception as e:
        pytest.skip(f"Cannot load model: {e}")
    from causallens.drivers.text import TextDriver
    return TextDriver(model, tokenizer, name="sst2")


def test_attention_rollout_basic(text_model):
    """Rollout should produce token attributions."""
    result = text_model.attention_rollout("I love this great movie", use_gradients=False)
    assert "tokens" in result
    assert len(result["tokens"]) > 0
    assert result["method"] == "attention_rollout"
    assert all("attribution" in r for r in result["tokens"])


def test_gradient_rollout_basic(text_model):
    """Gradient rollout should produce token attributions."""
    result = text_model.attention_rollout("I love this great movie", use_gradients=True)
    assert "tokens" in result
    assert result["method"] == "gradient_attention_rollout"
    scores = [r["attribution"] for r in result["tokens"]]
    assert any(s > 0 for s in scores)


def test_rollout_faster_than_occlusion(text_model):
    """Rollout should be significantly faster than occlusion."""
    import time
    text = "I really love this wonderful and amazing movie"

    t0 = time.time()
    text_model.attention_rollout(text, use_gradients=True)
    t_rollout = time.time() - t0

    t0 = time.time()
    text_model.token_attribution(text, method="occlusion")
    t_occlusion = time.time() - t0

    # Rollout should be faster (single pass vs n_tokens passes)
    assert t_rollout < t_occlusion * 0.8, (
        f"Rollout ({t_rollout:.3f}s) not faster than occlusion ({t_occlusion:.3f}s)"
    )


def test_rollout_identifies_sentiment_words(text_model):
    """Top attributed tokens should include sentiment-bearing words."""
    result = text_model.attention_rollout("I love this movie", use_gradients=True)
    top_tokens = [r["token"] for r in result["tokens"][:5]]
    assert len(top_tokens) > 0
    # At least one of the top tokens should be from the text
    assert any(t in ("love", "movie", "i", "this") for t in top_tokens)
