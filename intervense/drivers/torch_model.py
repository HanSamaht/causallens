"""PyTorch driver:modelinterrogation.

Interventions at every level:
  - activations: get_activations(x) captures every hooked layer
  - activation patching: the causal inference workhorse behind
    mediation analysis and circuit tracing, replace a layer activation
    with the reference sample and watch the output recover
  - input attribution: gradient times input, plus SHAP support
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np

from .base import MicDriver


class TorchDriver(MicDriver):
    """PyTorch driver with activation patching, steering, and gradient attribution."""
    kind = "torch"
    prediction_semantics = "probability"

    def __init__(self, model, feature_names, background, name="torch_model", positive_class=True):
        """Install hooks and set up activation cache."""
        import torch # : torch model

        self.torch = torch
        self.positive_class = positive_class
        super().__init__(model, feature_names, background, name)
        self._hooks = []
        self._activation_cache: dict[str, np.ndarray] = {}
        self._patch_plan: dict[str, np.ndarray] = {}
        self._steer_plan: dict[str, np.ndarray] = {} # steering vector: out += delta
        self._unit_mask: dict[str, set[int]] = {} # unit(reversible,)
        self._install_hooks()

    # ---- -------------------------------------------------------
    def _to_tensor(self, X: np.ndarray):
        """Convert numpy array to torch tensor."""
        t = self.torch.tensor(np.asarray(X, dtype=np.float32))
        return t

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass with no gradient."""
        model = self.model
        was_training = model.training
        model.eval()
        with self.torch.no_grad():
            out = model(self._to_tensor(X))
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            if out.shape[-1] == 1:
                out = out.squeeze(-1)
            if out.dim() > 1 and out.shape[-1] == 2:
                prob = self.torch.softmax(out, dim=-1)[:, 1 if self.positive_class else 0]
            else:
                prob = self.torch.sigmoid(out)
            arr = prob.cpu().numpy()
        if was_training:
            model.train()
        return arr.astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return model predictions."""
        return self._forward(X)

    # ---- activation --------------------------------------------------------
    def _install_hooks(self):
        """Register forward hooks on all named modules."""
        for layer_name, module in self.model.named_modules():
            if layer_name == "":
                continue
            def _mk(nm):
                """Create hook factory for layer."""
                def hook(_m, _inp, out):
                    """Capture, cache, and optionally patch layer output."""
                    # RNN modules return (output, hidden); only the output
                    # tensor is what interrogation cares about
                    is_seq = isinstance(out, (tuple, list))
                    core = out[0] if is_seq else out
                    if hasattr(core, "batch_sizes"): # PackedSequence:/
                        return None
                    if nm in self._steer_plan:
                        core = core + self.torch.tensor(self._steer_plan[nm], dtype=core.dtype)
                    if nm in self._patch_plan:
                        core = self.torch.tensor(self._patch_plan[nm], dtype=core.dtype)
                    if nm in self._unit_mask and self._unit_mask[nm]:
                        core = core.clone()
                        idx = list(self._unit_mask[nm])
                        if core.dim() == 3:
                            core[..., idx] = 0.0
                        else:
                            core[:, idx] = 0.0
                    self._activation_cache[nm] = core.detach().cpu().numpy()
                    if nm in self._patch_plan or nm in self._steer_plan or (nm in self._unit_mask and self._unit_mask[nm]):
                        if is_seq:
                            return (core, *out[1:]) # ,forward not
                        return core
                    return None
                return hook
            self._hooks.append(module.register_forward_hook(_mk(layer_name)))

    def list_layers(self) -> list[str]:
        """Return list of hookable layer names."""
        return [n for n, _ in self.model.named_modules() if n != ""]

    def get_activations(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """,returneverylayeractivation( hook )."""
        self._activation_cache.clear()
        self._forward(np.asarray(x, dtype=float).reshape(1, -1))
        return dict(self._activation_cache)

    @contextmanager
    def ablate_units(self, layer_name: str, units: set[int]):
        """unit(reversible):with thelayerunitoutput.

        "mode" pins the mask for a whole investigation and restores on exit.
        ACDC-lite circuitand SAE feature.
        """
        if layer_name not in dict(self.model.named_modules()):
            raise KeyError(f"layer {layer_name} notin")
        self._unit_mask.setdefault(layer_name, set()).update(units)
        try:
            yield
        finally:
            self._unit_mask[layer_name] -= set(units)
            if not self._unit_mask[layer_name]:
                self._unit_mask.pop(layer_name, None)

    # ---- activation patching(activation patching / causalinanalysis) --------------------
    @contextmanager
    def steering(self, layer_name: str, delta: np.ndarray):
        """:in with thelayeroutputon delta(notweight).

        delta sampleactivation(RepE/steering vector ).
        """
        if layer_name not in dict(self.model.named_modules()):
            raise KeyError(f"layer {layer_name} notin;layer: {self.list_layers()}")
        self._steer_plan[layer_name] = np.asarray(delta, dtype=np.float32)
        try:
            yield
        finally:
            self._steer_plan.pop(layer_name, None)

    @contextmanager
    def patch_activation(self, layer_name: str, reference_x: np.ndarray):
        """in with , layer_name outputreplaceas reference_x activation.

        typical(" corrupt input + clean activation "):
            p_clean = drv.predict_one(x)
            p_corrupt = drv.predict_one(x_corrupt)
            with drv.patch_activation("fc2", x):
                p_patched = drv.predict_one(x_corrupt) # recovery?
            mediation = p_patched - p_corrupt # thelayercausalineffect
        """
        ref_acts = self.get_activations(reference_x)
        if layer_name not in ref_acts:
            raise KeyError(f"layer {layer_name} notin;layer: {self.list_layers()}")
        self._patch_plan[layer_name] = ref_acts[layer_name]
        try:
            yield
        finally:
            self._patch_plan.pop(layer_name, None)

    # ---- ×inputattribution -----------------------------------------------------
    def input_attribution(self, x: np.ndarray) -> np.ndarray:
        """Gradient times input attribution."""
        t = self._to_tensor(np.asarray(x, dtype=float).reshape(1, -1))
        t.requires_grad_(True)
        out = self.model(t)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        target = out[0]
        target.backward()
        grad = t.grad.detach().cpu().numpy().ravel()
        return grad * np.asarray(x, dtype=float).ravel()

    # ---- ------------------------------------------------------------
    def describe(self) -> dict:
        """Return driver metadata."""
        d = super().describe()
        d["layers"] = self.list_layers()
        return d
