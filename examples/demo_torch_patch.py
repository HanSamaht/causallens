"""Torch mechanistic interpretability demo: activation readout plus
activation patching, the core of causal mediation analysis.

Run: python examples/demo_torch_patch.py (needs pip install torch)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.core import ConceptPack, MicSession
from causallens.demos import FinanceMLP
from causallens.drivers.torch_model import TorchDriver

# 1) train a small MLP on the finance demo task
net, feats, background, auc = FinanceMLP.train()
print(f"MLP auc = {auc:.4f}")

# 2) the torch driver ships the activation toolkit
drv = TorchDriver(net, feats, background, name="quant_mlp")
session = MicSession(drv, ConceptPack.load(os.path.join("concepts", "finance.json")))

print("layers:", drv.list_layers())

# 3) activation patching: sample 1 predicts 0.99; implant the layer 2
#    activations of sample 0 and watch what happens
p_clean = drv.predict_one(drv.sample_at(0))
p_other = drv.predict_one(drv.sample_at(1))
print(f"sample 0 prediction = {p_clean:.4f} | sample 1 prediction = {p_other:.4f}")

with drv.patch_activation("2", drv.sample_at(0)):  # implant the clean activations
    p_patched = drv.predict_one(drv.sample_at(1))
print(f"after patching layer 2 of sample 1 -> prediction = {p_patched:.4f}")
print("recovery =", (p_patched - p_other) / (p_clean - p_other))
print("reading: this layer almost fully mediates the decision, the key"
      " computation happens before it")

# 4) gradient times input attribution, the SHAP-free fast path
attr = drv.input_attribution(drv.sample_at(0))
ranked = sorted(zip(feats, attr), key=lambda t: -abs(t[1]))[:3]
print("gradient times input top 3:", [(f, round(v, 3)) for f, v in ranked])
