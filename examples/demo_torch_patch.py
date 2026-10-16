"""Torch 模型机制可解释性示例：激活读取 + 激活修补（因果中介分析）。

运行： python examples/demo_torch_patch.py  （需要 pip install torch）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mic.core import ConceptPack, MicSession
from mic.demos import FinanceMLP
from mic.drivers.torch_model import TorchDriver

# 1) 训练一个小 MLP（与金融 demo 同任务）
net, feats, background, auc = FinanceMLP.train()
print(f"MLP auc = {auc:.4f}")

# 2) Torch 驱动器自带激活武器库
drv = TorchDriver(net, feats, background, name="quant_mlp")
mic = MicSession(drv, ConceptPack.load(os.path.join("concepts", "finance_zh.json")))

print("层列表:", drv.list_layers())

# 3) 激活修补：样本#1 的预测是 0.99，把样本#0 的第2层激活植入后会发生什么？
p_clean = drv.predict_one(drv.sample_at(0))
p_other = drv.predict_one(drv.sample_at(1))
print(f"\n样本#0 预测={p_clean:.4f} | 样本#1 预测={p_other:.4f}")

with drv.patch_activation("2", drv.sample_at(0)):   # 植入"干净"激活
    p_patched = drv.predict_one(drv.sample_at(1))
print(f"修补样本#1 的第2层激活后 → 预测={p_patched:.4f}")
print("恢复率 =", (p_patched - p_other) / (p_clean - p_other))
print("解读：该层几乎完全中介了这个决策——模型的关键判断发生在这里之前")

# 4) 梯度×输入归因（TorchDriver 免 SHAP 的快速路径）
attr = drv.input_attribution(drv.sample_at(0))
ranked = sorted(zip(feats, attr), key=lambda t: -abs(t[1]))[:3]
print("\n梯度×输入 Top3:", [(f, round(v, 3)) for f, v in ranked])
