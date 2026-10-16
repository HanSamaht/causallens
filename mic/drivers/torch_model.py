"""PyTorch 驱动器：神经模型的全套审讯能力。

在通用干预之外，额外提供机制可解释性的两件核心武器：
  - 激活读取：  get_activations(x) —— 捕获每一层的输出激活
  - 激活修补：  patch_activation() —— 因果中介分析 / activation patching 的原子操作，
                把某层激活替换为"干净样本"的激活，观察输出恢复情况
  - 梯度×输入： input_attribution() —— 无需 SHAP 依赖的快速输入归因
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np

from .base import MicDriver


class TorchDriver(MicDriver):
    kind = "torch"
    prediction_semantics = "probability"

    def __init__(self, model, feature_names, background, name="torch_model", positive_class=True):
        import torch  # 延迟导入：未装 torch 的环境仍可使用树模型路径

        self.torch = torch
        self.positive_class = positive_class
        super().__init__(model, feature_names, background, name)
        self._hooks = []
        self._activation_cache: dict[str, np.ndarray] = {}
        self._patch_plan: dict[str, np.ndarray] = {}
        self._steer_plan: dict[str, np.ndarray] = {}  # steering vector: out += delta
        self._unit_mask: dict[str, set[int]] = {}     # 持久单元损毁（可逆，光遗传式）
        self._install_hooks()

    # ---- 前向传播 -------------------------------------------------------
    def _to_tensor(self, X: np.ndarray):
        t = self.torch.tensor(np.asarray(X, dtype=np.float32))
        return t

    def _forward(self, X: np.ndarray) -> np.ndarray:
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
        return self._forward(X)

    # ---- 激活读取 --------------------------------------------------------
    def _install_hooks(self):
        for layer_name, module in self.model.named_modules():
            if layer_name == "":
                continue
            def _mk(nm):
                def hook(_m, _inp, out):
                    # RNN 类模块输出为 (output, hidden) 元组——只审讯/修补第一个元素
                    is_seq = isinstance(out, (tuple, list))
                    core = out[0] if is_seq else out
                    if hasattr(core, "batch_sizes"):  # PackedSequence：跳过缓存/修补
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
                            return (core, *out[1:])   # 保持元组结构，forward 解包不炸
                        return core
                    return None
                return hook
            self._hooks.append(module.register_forward_hook(_mk(layer_name)))

    def list_layers(self) -> list[str]:
        return [n for n, _ in self.model.named_modules() if n != ""]

    def get_activations(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """前向一次，返回每层激活（会触发 hook 缓存）。"""
        self._activation_cache.clear()
        self._forward(np.asarray(x, dtype=float).reshape(1, -1))
        return dict(self._activation_cache)

    @contextmanager
    def ablate_units(self, layer_name: str, units: set[int]):
        """持久单元损毁（可逆）：with 块内该层指定单元输出恒置零。

        光遗传学的"抑制模式"——跨样本、跨前向持续生效；退出即恢复。
        ACDC-lite 电路剪枝与 SAE 特征证伪的原子操作。
        """
        if layer_name not in dict(self.model.named_modules()):
            raise KeyError(f"层 {layer_name} 不存在")
        self._unit_mask.setdefault(layer_name, set()).update(units)
        try:
            yield
        finally:
            self._unit_mask[layer_name] -= set(units)
            if not self._unit_mask[layer_name]:
                self._unit_mask.pop(layer_name, None)

    # ---- 激活修补（activation patching / 因果中介分析） --------------------
    @contextmanager
    def steering(self, layer_name: str, delta: np.ndarray):
        """推理时转向：在 with 块内给该层输出加上 delta（不改权重）。

        delta 通常来自两组样本的平均激活差（RepE/steering vector 思路）。
        """
        if layer_name not in dict(self.model.named_modules()):
            raise KeyError(f"层 {layer_name} 不存在；可用层: {self.list_layers()}")
        self._steer_plan[layer_name] = np.asarray(delta, dtype=np.float32)
        try:
            yield
        finally:
            self._steer_plan.pop(layer_name, None)

    @contextmanager
    def patch_activation(self, layer_name: str, reference_x: np.ndarray):
        """在 with 块内，把 layer_name 的输出强制替换为 reference_x 前向时的激活。

        典型用法（" corrupt input + clean activation "范式）：
            p_clean = drv.predict_one(x)
            p_corrupt = drv.predict_one(x_corrupt)
            with drv.patch_activation("fc2", x):
                p_patched = drv.predict_one(x_corrupt)   # 恢复了多少？
            mediation = p_patched - p_corrupt             # 该层的因果中介效应
        """
        ref_acts = self.get_activations(reference_x)
        if layer_name not in ref_acts:
            raise KeyError(f"层 {layer_name} 不存在；可用层: {self.list_layers()}")
        self._patch_plan[layer_name] = ref_acts[layer_name]
        try:
            yield
        finally:
            self._patch_plan.pop(layer_name, None)

    # ---- 梯度×输入归因 -----------------------------------------------------
    def input_attribution(self, x: np.ndarray) -> np.ndarray:
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

    # ---- 元信息 ------------------------------------------------------------
    def describe(self) -> dict:
        d = super().describe()
        d["layers"] = self.list_layers()
        return d
