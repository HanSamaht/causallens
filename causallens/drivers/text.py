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


    def phrase_counterfactual(self, text, target_label=None, max_edits=2, top_k=3):
        """Typed phrase level edits that flip the prediction, verified."""
        from ..analysis.phrase_cf import phrase_counterfactual

        return phrase_counterfactual(self, text, target_label=target_label,
                                     max_edits=max_edits, top_k=top_k)

    def attention_rollout(self, text, use_gradients=True):
        """Attention rollout with optional gradient augmentation.

        Recursively propagates attention weights from the output layer back
        to input tokens. When use_gradients=True, multiplies attention by
        gradient magnitudes for sharper attribution (Chefer et al. 2021).

        Faster than occlusion: one forward pass + one backward pass.
        """
        import torch

        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=self.max_length)
        input_ids = enc["input_ids"].to(self.device)

        # Store attention weights
        attentions = []
        hooks = []
        for name, module in self.model.named_modules():
            if "attention" in name and hasattr(module, "output") and hasattr(module.output, "dropout"):
                def _mk(nm):
                    def hook(_m, _i, out):
                        if isinstance(out, tuple):
                            attentions.append(out[1].detach().cpu())
                    return hook
                hooks.append(module.output.register_forward_hook(_mk(name)))

        embed_layer = self._get_embedding_layer()
        embeddings = embed_layer(input_ids).detach()
        embeddings.requires_grad_(True)

        outputs = self.model(inputs_embeds=embeddings, output_attentions=True)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probs = torch.softmax(logits, dim=-1)
        probs[0, self.target_class].backward()

        grad = embeddings.grad[0].detach().cpu()
        grad_norm = grad.norm(dim=-1, keepdim=True)
        grad_norm = grad_norm / (grad_norm.max() + 1e-8)

        if hasattr(outputs, "attentions") and outputs.attentions:
            att_list = [a.detach().cpu() for a in outputs.attentions]
        elif attentions:
            att_list = attentions
        else:
            for h in hooks:
                h.remove()
            return self._occlusion_fallback(text)

        for h in hooks:
            h.remove()

        seq_len = att_list[0].shape[1]
        result = np.eye(seq_len)
        for att in att_list:
            att_avg = att[0].mean(dim=0).numpy()
            if use_gradients:
                gn = grad_norm[:seq_len, 0].numpy()
                att_avg = att_avg * gn[np.newaxis, :]
                row_sums = att_avg.sum(axis=1, keepdims=True) + 1e-8
                att_avg = att_avg / row_sums
            else:
                att_avg = att_avg + np.eye(seq_len)
                att_avg = att_avg / att_avg.sum(axis=1, keepdims=True)
            result = att_avg @ result

        token_scores = result[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        results = []
        for i, (tok, score) in enumerate(zip(tokens, token_scores)):
            if tok not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                results.append({
                    "token": self.tokenizer.convert_tokens_to_string([tok]).strip(),
                    "position": i,
                    "attribution": round(float(score), 6),
                })

        results.sort(key=lambda r: -r["attribution"])
        p_base = float(probs[0, self.target_class].detach().cpu())
        return {
            "text": text,
            "prediction": round(p_base, 4),
            "tokens": results,
            "method": "gradient_attention_rollout" if use_gradients else "attention_rollout",
        }

    def _occlusion_fallback(self, text):
        """Fallback to occlusion when attention weights are unavailable."""
        return self.token_attribution(text, method="occlusion")

    def describe(self):
        d = super().describe()
        d["model_type"] = type(self.model).__name__
        d["n_classes"] = self.n_classes
        d["device"] = str(self.device)
        d["note"] = "Text model driver with token-level intervention and attribution"
        return d
