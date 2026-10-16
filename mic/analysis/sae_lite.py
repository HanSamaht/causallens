"""SAE-lite：领域小模型的稀疏字典 + 特征自动标注 + 因果证伪。

谱系：Olshausen & Field 1996 稀疏编码(V1) → Anthropic Scaling Monosemanticity(2024)
→ 本实现（面向 H=16~256 隐层的领域模型，CPU 分钟级）。

三步：
  1. 训练过完备稀疏自编码器于某层 h_T 激活（expand H→4H，L1 稀疏）
  2. 自动标注：每个 SAE 特征与输入 base 特征做相关 → "特征k ↔ 动量@lag0 (r=0.82)"
     （领域小模型的独有优势：top-activating 样本就是可读的输入窗口）
  3. 因果证伪（NeurIPS 2025 精神）：消融该特征（SAE 域置零→解码→修补回模型）
     预测变化显著的才是真特征，其余标记为"未通过因果检验"
"""
from __future__ import annotations

import numpy as np


def collect_hidden(drv, layer: str | None = None, n: int = 300) -> np.ndarray:
    """收集该层 h_T 激活矩阵 (n, H)。"""
    target = layer or (drv.list_layers()[-2] if len(drv.list_layers()) >= 2 else drv.list_layers()[0])
    rows = [drv.get_activations(drv.sample_at(i))[target] for i in range(min(n, len(drv.background)))]
    arr = np.stack(rows)
    if arr.ndim == 4:        # RNN: (n, 1, T, H) → h_T
        return arr[:, 0, -1, :]
    if arr.ndim == 3:        # MLP: (n, 1, H)
        return arr[:, 0, :]
    return arr.reshape(arr.shape[0], -1)


def train_sae(hidden: np.ndarray, expand: int = 4, l1: float = 0.01, epochs: int = 300,
              seed: int = 0) -> dict:
    t = __import__("torch")
    t.manual_seed(seed)
    X = t.tensor(hidden, dtype=t.float32)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    H = hidden.shape[1]
    D = H * expand
    enc_w = t.nn.Parameter(t.randn(H, D) * 0.1)
    dec_w = t.nn.Parameter(t.randn(D, H) * 0.1)
    opt = t.optim.Adam([enc_w, dec_w], lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        z = t.relu(Xn @ enc_w)
        recon = z @ dec_w
        loss = ((recon - Xn) ** 2).mean() + l1 * z.abs().mean()
        loss.backward()
        opt.step()
    with t.no_grad():
        Z = t.relu(Xn @ enc_w).numpy()
    return {"enc_w": enc_w.detach(), "dec_w": dec_w.detach(), "mu": mu, "sd": sd,
            "Z": Z, "n_features": D, "recon_mse": float(((recon - Xn) ** 2).mean().item()) if hasattr((recon - Xn) ** 2, "item") else float(((recon - Xn) ** 2).mean()),
            "mean_active_frac": float((Z > 0.01).mean())}


def sae_features(drv, layer: str | None = None, n: int = 300, expand: int = 4,
                 top_k: int = 8, causal_threshold: float = 0.005,
                 l1: float = 0.01, epochs: int = 300) -> dict:
    """expand: 字典倍率（4=默认，8/16=更多更细粒度特征——更大更慢，但分离出更多概念）"""
    """训练 + 自动标注 + 因果证伪，一步到位。"""
    target = layer or (drv.list_layers()[-2] if len(drv.list_layers()) >= 2 else drv.list_layers()[0])
    hidden = collect_hidden(drv, target, n)
    sae = train_sae(hidden, expand=expand, l1=l1, epochs=epochs)
    Z, t = sae["Z"], drv.torch

    # ---- 自动标注：SAE 特征 × base 输入特征 相关 ----
    X_inputs = drv.background[: len(Z)]
    base_names = getattr(drv, "base_feature_names", drv.feature_names)
    input_view = X_inputs
    if drv.kind == "timeseries":  # 取窗口每 base 特征均值做标注参照
        T, F = drv.seq_len, drv.n_features_raw
        input_view = X_inputs.reshape(len(X_inputs), T, F).mean(axis=1)

    rows = []
    for k in range(sae["n_features"]):
        zk = Z[:, k]
        if (zk > 0.01).sum() < 5:
            continue
        cors = {}
        for j, bname in enumerate(base_names):
            if input_view.shape[1] != len(base_names):
                break
            r = float(np.corrcoef(zk, input_view[:, j])[0, 1]) if np.std(zk) > 1e-9 else 0.0
            if abs(r) > 0.3:
                cors[bname] = round(r, 2)
        if not cors:
            continue
        rows.append({"feature": k, "active_frac": round(float((zk > 0.01).mean()), 3),
                     "correlates_with": dict(sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:3])})

    # ---- 因果证伪：消融特征 → 修补回模型 → 预测变化 ----
    verified = []
    for row in rows[: top_k * 2]:
        k = row["feature"]
        top_idx = int(np.argsort(-Z[:, k])[0])
        x = drv.sample_at(top_idx)
        ref_act = drv.get_activations(x)[target]
        flat_act = ref_act[0, -1, :] if ref_act.ndim == 3 else ref_act.reshape(-1)
        zt = t.relu((t.tensor(flat_act, dtype=t.float32)
                     - sae["mu"].squeeze()) / sae["sd"].squeeze() @ sae["enc_w"])
        zt[k] = 0.0
        patched = (zt @ sae["dec_w"]).detach().numpy() * sae["sd"].squeeze().numpy() + sae["mu"].squeeze().numpy()
        shape = ref_act.shape
        patch_tensor = ref_act.copy()
        if ref_act.ndim == 3:
            patch_tensor[0, -1, :] = patched
        else:
            patch_tensor[0, :] = patched
        drv._patch_plan[target] = patch_tensor
        try:
            p_abl = drv.predict_one(x)
        finally:
            drv._patch_plan.pop(target, None)
        p_base = drv.predict_one(x)
        row["causal_effect"] = round(float(abs(p_abl - p_base)), 5)
        row["verdict"] = "通过因果检验" if row["causal_effect"] > causal_threshold else "未通过（相关非因果）"
        verified.append(row)

    verified.sort(key=lambda r: -r.get("causal_effect", 0))
    n_pass = sum(1 for r in verified if r["verdict"].startswith("通过"))
    return {
        "layer": target,
        "n_features": sae["n_features"],
        "recon_mse": round(sae["recon_mse"], 4),
        "active_fraction": round(sae["mean_active_frac"], 3),
        "features": verified[:top_k],
        "verdict": f"{n_pass}/{len(verified)} 个标注特征通过因果消融检验（其余为相关级候选）",
        "note": "标注=SAE特征与输入base特征的相关（自动）；因果检验=特征消融修补后预测变化（干预级）",
    }
