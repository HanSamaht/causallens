"""TextDriver tests: token intervention, attribution, counterfactual search."""
import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def text_model():
    """Load a small sentiment model for testing."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "textattack/bert-base-uncased-SST-2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
    except (OSError, ConnectionError, Exception) as e:
        pytest.skip(f"Cannot reach HuggingFace or model unavailable: {e}")

    from causallens.drivers.text import TextDriver
    return TextDriver(model, tokenizer, name="sst2_bert")


def test_predict_text(text_model):
    """Model should predict positive for positive text."""
    p_pos = text_model.predict_text("I love this movie, it is great")
    p_neg = text_model.predict_text("This movie is terrible and awful")
    assert p_pos != p_neg


def test_token_intervention(text_model):
    """Changing one word should change prediction."""
    original = "I love this movie"
    modified, p_before, p_after = text_model.intervene(original, {"love": "hate"})
    assert modified == "I hate this movie"
    assert p_before != p_after


def test_token_attribution_occlusion(text_model):
    """Occlusion attribution should identify sentiment words."""
    result = text_model.token_attribution("I love this great movie", method="occlusion")
    assert "tokens" in result
    assert len(result["tokens"]) > 0
    top_tokens = [r["token"] for r in result["tokens"][:3]]
    assert len(top_tokens) > 0


def test_token_attribution_gradient(text_model):
    """Gradient attribution should produce scores."""
    result = text_model.token_attribution("I love this great movie", method="gradient")
    assert "tokens" in result
    assert len(result["tokens"]) > 0


def test_counterfactual_search(text_model):
    """Should find a token swap that flips prediction."""
    result = text_model.counterfactual_search("I love this movie, it is wonderful")
    assert "flipped" in result
    assert "counterfactual_text" in result
    # If flipped, the text should be different from original
    if result["flipped"]:
        assert result["counterfactual_text"] != result["original_text"]


def test_describe(text_model):
    """Driver should return metadata."""
    d = text_model.describe()
    assert d["kind"] == "text"
    assert d["n_classes"] == 2
