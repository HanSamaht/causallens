"""激活级因果图（circuit tracing lite，对齐 Anthropic 归因图的思路）。

比"特征层因果图"深一层：回答"模型内部哪一层、哪些单元在承载这个决策"。

方法（全部基于干预实验，不是相关）——标准因果中介范式：
  clean   = p(sample)          干净样本的预测
  corrupt = p(reference)       参照样本的预测
  patched = p(reference | 第L层激活 ← sample的激活)   修补后的预测
  层中介效应 = patched - corrupt（向 clean 恢复了多少）

单元级：对中介效应最强的层，逐单元把 reference 的该单元换成 sample 的值，
找出承载决策的关键单元。
"""
from __future__ import annotations

import numpy as np

from ..drivers.torch_model import TorchDriver


def build_activation_graph(
    drv: TorchDriver,
    sample_index: int,
    reference_index: int,
    top_units: int = 6,
) -> dict:
    assert drv.kind in ("torch", "timeseries"), "激活级因果图支持 torch 系驱动（MLP/时序）"
    x = drv.sample_at(sample_index)
    x_ref = drv.sample_at(reference_index)

    p_clean = drv.predict_one(x)
    p_ref = drv.predict_one(x_ref)
    total_gap = p_clean - p_ref

    clean_acts = drv.get_activations(x)       # 干净样本各层激活（移植来源）
    ref_acts = drv.get_activations(x_ref)

    nodes = ["input(sample)", "input(reference)", "prediction"]
    edges = []

    def _patched_pred(layer: str, replacement: np.ndarray) -> float:
        drv._patch_plan[layer] = replacement
        try:
            return drv.predict_one(x_ref)
        finally:
            drv._patch_plan.pop(layer, None)

    layer_effects = {}
    for layer in drv.list_layers():
        if layer not in clean_acts or clean_acts[layer].shape != ref_acts[layer].shape:
            continue
        effect = _patched_pred(layer, clean_acts[layer]) - p_ref
        layer_effects[layer] = float(effect)
        nodes.append(layer)
        edges.append(
            {
                "source": layer,
                "target": "prediction",
                "effect": round(float(effect), 4),
                "recovery": round(float(effect / total_gap), 4) if abs(total_gap) > 1e-6 else None,
                "kind": "layer_mediation",
            }
        )

    if not layer_effects:
        return {"nodes": nodes, "edges": edges, "note": "未捕获到层激活"}

    best_layer = max(layer_effects, key=lambda k: abs(layer_effects[k]))
    unit_edges = []
    base_act, clean_act = ref_acts.get(best_layer), clean_acts.get(best_layer)
    if base_act is not None and base_act.ndim == 2 and clean_act.shape == base_act.shape:
        for u in range(base_act.shape[1]):
            if abs(clean_act[0, u] - base_act[0, u]) < 1e-6:
                continue
            mod = np.copy(base_act)
            mod[0, u] = clean_act[0, u]
            eff = _patched_pred(best_layer, mod) - p_ref
            if abs(eff) > 1e-4:
                node = f"{best_layer}#u{u}"
                nodes.append(node)
                unit_edges.append(
                    {
                        "source": node,
                        "target": "prediction",
                        "effect": round(float(eff), 4),
                        "recovery": round(float(eff / total_gap), 4) if abs(total_gap) > 1e-6 else None,
                        "kind": "unit_mediation",
                    }
                )
        unit_edges.sort(key=lambda e: -abs(e["effect"]))

    # ---- 序列张量 (B, T, H)：GRU/LSTM 的两种良定义单元语义 ----
    elif base_act is not None and base_act.ndim == 3 and clean_act.shape == base_act.shape:
        H = base_act.shape[2]
        units = list(range(H)) if H <= 32 else _pre_rank_units(base_act, clean_act, 32)

        def _seq_eff(u, persistent):
            mod = np.copy(base_act)
            if persistent:
                mod[0, :, u] = clean_act[0, :, u]   # 持久修补：全窗口同单元
            else:
                mod[0, -1, u] = clean_act[0, -1, u]  # h_T 修补：决策头读取的最后隐状态
            return _patched_pred(best_layer, mod) - p_ref

        for u in units:
            eff_ht = _seq_eff(u, persistent=False)
            if abs(eff_ht) > 1e-4:
                node = f"{best_layer}#u{u}@h_T"
                nodes.append(node)
                unit_edges.append({
                    "source": node, "target": "prediction",
                    "effect": round(float(eff_ht), 4),
                    "recovery": round(float(eff_ht / total_gap), 4) if abs(total_gap) > 1e-6 else None,
                    "kind": "unit_ht_mediation",
                    "detail": "决策读取的最后隐状态单元（决策因果）",
                })
            eff_ps = _seq_eff(u, persistent=True)
            if abs(eff_ps) > 1e-4 and abs(eff_ps - eff_ht) > 1e-4:
                node = f"{best_layer}#u{u}@persistent"
                nodes.append(node)
                unit_edges.append({
                    "source": node, "target": "prediction",
                    "effect": round(float(eff_ps), 4),
                    "recovery": round(float(eff_ps / total_gap), 4) if abs(total_gap) > 1e-6 else None,
                    "kind": "unit_persistent_mediation",
                    "detail": "全窗口持久单元（Karpathy式'全程载体'细胞的干预版）",
                })
        unit_edges.sort(key=lambda e: -abs(e["effect"]))

    return {
        "nodes": nodes,
        "edges": edges + unit_edges[:top_units],
        "p_sample": round(p_clean, 4),
        "p_reference": round(p_ref, 4),
        "total_gap": round(float(total_gap), 4),
        "best_layer": best_layer,
        "method": "层级/单元级激活修补（干预实验，非相关）；序列模型含 h_T 单元与持久单元双语义",
        "note": "lite 版 circuit tracing：Anthropic 归因图做特征级分解(SAE)，本图做层/单元级中介；"
                "思路参照 LSTMVis/Karpathy 单元分析，方法为因果修补（非观察）",
    }


def _pre_rank_units(base_act, clean_act, k: int) -> list[int]:
    """单元数多时先粗排：按 |clean-reference| 激活差（信息量代理）取前 k。"""
    diff = np.abs(clean_act - base_act).sum(axis=(0, 1)) if base_act.ndim == 3 else np.abs(clean_act - base_act).sum(axis=0)
    return list(np.argsort(-diff)[:k])
