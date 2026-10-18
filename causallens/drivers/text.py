"""Text model driver: interrogate NLP models with token-level interventions."""
from __future__ import annotations

import numpy as np

from .base import MicDriver


class TextDriver(MicDriver):
    """Driver for HuggingFace text models with token-level interventions."""
    kind = "text"
    prediction_semantics = "probability"

    def __init__(self, model, tokenizer, name="text_model", device=None, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        try:
            import torch
            self.device = device or next(model.parameters()).device
        except (StopIteration, Exception):
            import torch
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.n_classes = getattr(model.config, "num_labels", 2)
        self.target_class = min(1, self.n_classes - 1)
        self._vocab_size = getattr(model.config, "vocab_size", 0)
        super().__init__(model, [f"token_{i}" for i in range(max_length)],
                         np.zeros((50, max_length)), name=name)

    def _forward(self, text):
        """Run model and return target class probability."""
        import torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=self.max_length, padding=True)
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in enc.items()})
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            probs = torch.softmax(logits, dim=-1)
            return float(probs[0, self.target_class].cpu())

    def predict(self, X):
        if isinstance(X, str):
            X = [X]
        return np.array([self._forward(t) for t in X])

    def predict_one(self, x):
        return self._forward(str(x))

    def predict_text(self, text):
        return self._forward(text)

    def intervene(self, text, token_assignments):
        """Replace tokens in text and measure prediction change."""
        p_before = self._forward(text)
        modified = text
        for old, new in token_assignments.items():
            modified = modified.replace(old, new)
        p_after = self._forward(modified)
        return modified, p_before, p_after

    def token_attribution(self, text, method="occlusion"):
        """Attribute prediction to tokens via occlusion or gradients."""
        tokens = self.tokenizer.tokenize(text)
        p_base = self._forward(text)
        results = []

        if method == "gradient":
            return self._gradient_attribution(text, tokens)

        for i, tok in enumerate(tokens):
            masked = self._mask_token(text, i)
            p_masked = self._forward(masked) if masked else p_base
            effect = p_base - p_masked
            word = self.tokenizer.convert_tokens_to_string([tok]).strip()
            results.append({"token": word, "position": i, "attribution": round(float(effect), 5)})

        results.sort(key=lambda r: -abs(r["attribution"]))
        return {"text": text, "prediction": round(p_base, 4), "tokens": results, "method": method}

    def _mask_token(self, text, token_idx):
        import torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        ids = enc["input_ids"][0]
        if token_idx + 1 < len(ids) - 1:
            ids[token_idx + 1] = self.tokenizer.mask_token_id
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        return text

    def _gradient_attribution(self, text, tokens):
        import torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        input_ids = enc["input_ids"].to(self.device)
        embed_layer = self._get_embedding_layer()
        embeddings = embed_layer(input_ids).detach()
        embeddings.requires_grad_(True)
        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probs = torch.softmax(logits, dim=-1)
        probs[0, self.target_class].backward()
        grads = embeddings.grad[0].detach().cpu()
        results = []
        for i, tok in enumerate(tokens):
            if i < len(grads):
                word = self.tokenizer.convert_tokens_to_string([tok]).strip()
                results.append({"token": word, "position": i, "attribution": round(float(grads[i].norm().item()), 5)})
        results.sort(key=lambda r: -abs(r["attribution"]))
        return {"text": text, "prediction": 0, "tokens": results, "method": "gradient"}

    def _get_embedding_layer(self):
        if hasattr(self.model, "bert"):
            return self.model.bert.embeddings.word_embeddings
        if hasattr(self.model, "roberta"):
            return self.model.roberta.embeddings.word_embeddings
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        raise ValueError("Could not find embedding layer")

    def counterfactual_search(self, text, target_label=None):
        """Find minimal token changes that flip the prediction."""
        p_base = self._forward(text)
        base_label = 1 if p_base > 0.5 else 0
        if target_label is None:
            target_label = 1 - base_label

        attr = self.token_attribution(text, method="occlusion")
        important = [r["token"] for r in attr["tokens"] if abs(r["attribution"]) > 0.01]
        candidates = ["good", "bad", "great", "terrible", "love", "hate",
                      "best", "worst", "excellent", "awful", "happy", "sad", "not"]

        best = None
        for old in important[:5]:
            for new in candidates:
                if new == old:
                    continue
                modified, _, p_new = self.intervene(text, {old: new})
                new_label = 1 if p_new > 0.5 else 0
                if new_label == target_label:
                    dist = sum(1 for a, b in zip(text.split(), modified.split()) if a != b)
                    if best is None or dist < best[0]:
                        best = (dist, modified, p_new)

        return {
            "original_text": text,
            "original_prediction": round(p_base, 4),
            "counterfactual_text": best[1] if best else None,
            "counterfactual_prediction": round(best[2], 4) if best else None,
            "flipped": best is not None,
            "method": "token_swap",
        }

    def describe(self):
        d = super().describe()
        d["model_type"] = type(self.model).__name__
        d["n_classes"] = self.n_classes
        d["device"] = str(self.device)
        d["note"] = "Text model driver with token-level intervention and attribution"
        return d
