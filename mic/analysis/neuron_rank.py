"""梯度式神经元排序（可选 Captum 后端，未装自动回退，绝不报错）。

排序目标：某层哪些单元对决策的梯度贡献最大。
  装了 captum → NeuronGradient（Meta 官方，逐单元反传）
  没装       → 免依赖回退：h_T 激活 × ∂输出/∂h_T（神经元版梯度×输入，一次反传）
用途：activation_graph 的单元修补预筛（单元数>32）、独立问答
'这个 GRU 的哪些神经元在管这个决策'。
"""
from __future__ import annotations

import numpy as np


def _forward_input(drv, x):
    """按驱动类型构造带梯度的前向输入张量。"""
    t = drv.torch
    if drv.kind == "timeseries":
        flat = np.asarray(x, dtype=float).ravel()
        return t.tensor(flat.reshape(1, drv.seq_len, drv.n_features_raw), dtype=t.float32, requires_grad=True)
    return t.tensor(np.asarray(x, dtype=float).reshape(1, -1), dtype=t.float32, requires_grad=True)


def neuron_ranking(drv, sample_index: int, layer: str | None = None, top_k: int = 10) -> dict:
    """drv: TorchDriver / TimeSeriesDriver。返回指定层（默认最后隐层）单元排序。"""
    assert drv.kind in ("torch", "timeseries"), "神经元排序需要 torch 系驱动"
    t = drv.torch
    x = drv.sample_at(sample_index)
    layers = drv.list_layers()
    layer = layer or (layers[-2] if len(layers) >= 2 else layers[0])  # 最后隐层
    if layer not in layers:
        raise KeyError(f"层 {layer} 不存在；可用: {layers}")

    xt = _forward_input(drv, x)
    engine = "fallback(grad×activation)"

    # ---- 首选：captum NeuronGradient（装了就用） ----
    last_err = None
    try:
        from captum.attr import NeuronGradient

        module = dict(drv.model.named_modules())[layer]
        width = _layer_width(drv, xt, layer)
        ng = NeuronGradient(drv.model, module)

        def _selector(u):
            # 序列层输出 (output,hidden) 元组：选 h_T 的第 u 单元；普通层直接索引
            return (lambda out: out[0][0, -1, u]) if drv.kind == "timeseries" else (lambda out: out.reshape(-1)[u])

        contribs = [float(ng.attribute(xt, neuron_selector=_selector(u)).abs().sum()) for u in range(width)]
        contrib = np.array(contribs)
        engine = "captum(NeuronGradient)"
        act_h, grad_h = _act_grad(drv, xt, layer)  # 仅用于展示
    except ImportError:
        act_h, grad_h = _act_grad(drv, xt, layer)
        contrib = act_h * grad_h
    except Exception as e:  # 降级必须留下痕迹（BUG-009/025 教训）
        last_err = f"{type(e).__name__}: {e}"
        act_h, grad_h = _act_grad(drv, xt, layer)
        contrib = act_h * grad_h

    order = np.argsort(-np.abs(contrib))[:top_k]
    rows = [{"unit": int(u), "activation": round(float(act_h[u]), 4),
             "contribution": round(float(contrib[u]), 5)} for u in order]
    return {
        "layer": layer,
        "n_units": int(act_h.shape[0]),
        "engine": engine,
        "top_units": rows,
        "note": "排序=梯度贡献；配合 activation_graph 的单元修补做因果确认（排序是相关级证据，修补是干预级）",
        **({"captum_fallback_reason": last_err} if last_err else {}),
    }


def _layer_width(drv, xt, layer) -> int:
    acts = drv.get_activations(drv.background[0])
    a = acts[layer]
    return int(a.shape[-1])


def _act_grad(drv, xt, layer):
    """一次前向+反传，取该层 h_T 的激活与梯度。"""
    t = drv.torch
    captured = {}

    def _hook(_m, _i, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        out.retain_grad()          # 非叶子张量：登记后 backward 才会填充 .grad
        captured["act"] = out

    h = dict(drv.model.named_modules())[layer].register_forward_hook(_hook)
    try:
        out = drv.model(xt)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() > 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        out.squeeze().backward()
        act = captured["act"]
        if isinstance(act, (tuple, list)):
            act = act[0]                      # RNN: (output, hidden) 取 output
        act = act.detach().cpu().numpy()
        src = captured["act"]
        if isinstance(src, (tuple, list)):
            src = src[0]
        grad = src.grad
        grad = grad.detach().cpu().numpy() if grad is not None else np.zeros_like(act)
    finally:
        h.remove()
    if act.ndim == 3:      # 序列层取 h_T
        return act[0, -1, :], grad[0, -1, :]
    return act.reshape(-1), grad.reshape(-1)
