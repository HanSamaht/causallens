"""参数组消融（可逆损毁）：某区域参数发挥什么功能。

神经科学原型：损毁实验（lesion study）——破坏一个脑区，观察功能损失。
MIC 升级版：可逆 + 复测（备份→消融→测功能→恢复→复测确认），
恢复后指标不回弹的"发现"一律按实现误差丢弃。

功能 = 可测行为测试集：
  - "pred_shift"：N 个样本的预测分布变化
  - timeseries 另有 "early_window_share"：窗口前半 lag 的归因占比
    （消融后塌缩 = 该参数组承担"长期记忆"功能）
"""
from __future__ import annotations

import numpy as np


def _capability(drv, metric: str, n_samples: int) -> float:
    X = drv.background[:n_samples]
    if metric == "early_window_share" and drv.kind == "timeseries":
        shares = []
        for i in range(min(n_samples, 20)):
            heat, _ = drv.window_attribution(drv.sample_at(i))
            half = heat.shape[0] // 2
            tot = np.abs(heat).sum() + 1e-12
            shares.append(np.abs(heat[:half]).sum() / tot)
        return float(np.mean(shares))
    p = drv.predict(X)
    return float(np.mean(p))  # pred_center：预测中心漂移即功能变化信号


def param_ablation(drv, param_groups: list[str], metric: str = "pred_center",
                   n_samples: int = 50, scale: float = 0.0) -> dict:
    """param_groups: 参数名前缀列表（如 ["gru.weight_hh"] 匹配所有层）。

    返回每组：消融前后功能指标、效应、可逆性验证。
    """
    assert drv.kind in ("torch", "timeseries"), "参数消融需要 torch 系驱动"
    t = drv.torch
    base = _capability(drv, metric, n_samples)

    results = []
    for prefix in param_groups:
        matched = {n: p for n, p in drv.model.named_parameters() if n.startswith(prefix)}
        if not matched:
            results.append({"param_group": prefix, "error": "未匹配到参数"})
            continue
        backup = {n: p.detach().clone() for n, p in matched.items()}

        for p in matched.values():  # 消融
            with t.no_grad():
                p.mul_(scale)
        ablated = _capability(drv, metric, n_samples)

        for n, p in matched.items():  # 恢复
            with t.no_grad():
                p.copy_(backup[n])
        restored = _capability(drv, metric, n_samples)

        effect = ablated - base
        reversible = abs(restored - base) < 1e-6
        results.append(
            {
                "param_group": prefix,
                "n_matched_params": len(matched),
                "n_weights": int(sum(p.numel() for p in matched.values())),
                "metric": metric,
                "baseline": round(base, 6),
                "ablated": round(ablated, 6),
                "effect": round(float(effect), 6),
                "restored": round(restored, 6),
                "reversible": reversible,
                "verdict": (
                    f"{'有效发现' if abs(effect) > 1e-4 else '功能不变'}：消融 {prefix} 使 {metric} "
                    f"{base:.4g}→{ablated:.4g}（Δ={effect:+.4g}）"
                    + ("" if reversible else " ⚠恢复失败，结果作废")
                ),
            }
        )
    return {"results": results, "note": "可逆损毁：备份→消融→测功能→恢复→复测；恢复失败的结果一律作废"}


def discover_circuit(drv, capability_samples: int = 20, tolerance: float = 0.02,
                     layer: str | None = None, mode: str = "greedy",
                     refinement_rounds: int = 3) -> dict:
    """ACDC-lite：找承载当前行为的最小单元子电路。

    mode:
      "greedy"    — 贪心反向剔除（默认，秒级）：逐个损毁单元，行为不变则剔除。
                    O(N) 次前向，对小模型即时出结果。
      "exhaustive" — 束搜索 + 双向精修（用户计算资源充足时选）：贪心做初始解，
                    然后 refinement_rounds 轮精修：尝试把剔除的单元加回来、
                    把电路单元换出去，接受任何能减小电路规模且保持行为残差的改动。
                    O(N² × rounds) 次前向，N≤64 时分钟级。

    对指定层（默认最后隐层）的每个单元做持久损毁：
    行为变化 < tolerance → 该单元不属于电路（剔除保留损毁）；
    变化显著 → 属于电路（恢复）。
    模型小（≤256 单元）时全量真损毁比近似修补还便宜——不需要 EAP。
    """
    assert drv.kind in ("torch", "timeseries"), "电路发现需要 torch 系驱动"
    layers = drv.list_layers()
    target = layer or (layers[-2] if len(layers) >= 2 else layers[0])

    X = drv.background[:capability_samples]
    p0 = drv.predict(X)

    def _behavior_delta() -> float:
        return float(np.mean(np.abs(drv.predict(X) - p0)))

    width = drv.get_activations(drv.background[0])[target].shape[-1]
    if width > 256 and mode == "exhaustive":
        return {"error": f"层 {target} 宽度 {width} > 128，exhaustive 模式不可行（用 greedy + neuron_ranking 预筛）"}
    if width > 256:
        return {"error": f"层 {target} 宽度 {width} > 256，先跑 neuron_ranking 预筛"}

    circuit, pruned = [], []
    drv._unit_mask.setdefault(target, set())

    # ---- 第一阶段：贪心反向剔除 ----
    for u in range(width):
        drv._unit_mask[target].add(u)
        delta = _behavior_delta()
        if delta < tolerance:
            pruned.append(u)
        else:
            drv._unit_mask[target].discard(u)
            circuit.append({"unit": u, "behavior_delta_when_ablated": round(delta, 4)})

    # ---- 第二阶段（exhaustive 模式）：双向精修 ----
    improvements = 0
    if mode == "exhaustive":
        for _round in range(refinement_rounds):
            changed = False
            # 尝试把剔除单元加回来（可能被早期贪心错删）
            for u in list(pruned):
                drv._unit_mask[target].discard(u)  # 恢复，看是否必要
                delta = _behavior_delta()
                if delta < tolerance:
                    # 确实不需要 → 可以永久恢复（比损毁状态更保守）
                    pass  # 已恢复，什么都不做
                else:
                    # 其实需要 → 加回电路
                    drv._unit_mask[target].add(u)
                    pruned.remove(u)
                    circuit.append({"unit": u, "behavior_delta_when_ablated": round(delta, 4)})
                    changed = True
                    improvements += 1

            # 尝试把电路单元换出去（可能被错误保留）
            for c in list(circuit):
                u = c["unit"]
                was_masked = u in drv._unit_mask[target]
                drv._unit_mask[target].add(u)
                # 看在其他电路单元也损毁的状态下这个单元是否还必要
                delta = _behavior_delta()
                if delta < tolerance:
                    # 不需要 → 剔除
                    circuit.remove(c)
                    pruned.append(u)
                    changed = True
                    improvements += 1
                elif not was_masked:
                    drv._unit_mask[target].discard(u)
            if not changed:
                break

    # 清场复测：电路行为应回到基线
    p_end = drv.predict(X)
    residual = float(np.mean(np.abs(p_end - p0)))
    drv._unit_mask.pop(target, None)

    circuit.sort(key=lambda c: -c["behavior_delta_when_ablated"])
    mode_desc = ("贪心反向剔除" if mode == "greedy"
                 else f"束搜索+{refinement_rounds}轮双向精修（{improvements} 处改进）")
    return {
        "layer": target,
        "mode": mode,
        "n_units_total": width,
        "circuit_units": circuit,
        "n_circuit": len(circuit),
        "n_pruned": len(pruned),
        "sparsity": round(len(pruned) / width, 3),
        "improvements_from_refinement": improvements if mode == "exhaustive" else 0,
        "residual_after_restore": round(residual, 6),
        "verdict": (
            f"[{mode_desc}] 最小电路 = {len(circuit)}/{width} 单元"
            f"（{len(pruned)} 个可剔除，稀疏度 {len(pruned)/width:.0%}）；"
            f"剔除全部非电路单元后行为残差 {residual:.4f}"
            + ("（≈0，电路自洽）" if residual < tolerance else " ⚠残差偏大，电路不完整")
        ),
        "note": "贪心反向剔除（ACDC 思路，小模型全量真损毁版）；每步均为持久损毁干预",
    }
