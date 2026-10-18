"""Programmatic usage demo: interrogate a finance model in five steps.

Run: python examples/demo_interrogate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.core import ConceptPack, MicSession
from causallens.demos import train_finance_demo
from causallens.drivers import detect_driver

# 1) prepare the model (in a real case, load your own model plus background data)
model, feats, background, auc = train_finance_demo()
print(f"model auc = {auc:.4f}")

# 2) plug in the microphone: auto detect the driver, load the concept pack
driver = detect_driver(model, feats, background, name="my_quant_model")
pack = ConceptPack.load(os.path.join("concepts", "finance.json"))
session = MicSession(driver, pack)

# 3) let the model speak: one call for a full single sample explanation
result = session.get_attribution(sample_index=0)
print(f"sample 0 prediction = {result['prediction']}")
for a in result["attributions"][:3]:
    print(f"  {a['concept']} ({a['feature']}) value={a['value']:+.2f} contribution={a['shap']:+.3f}")

# 4) ask what if: a real intervention
r = session.intervene(0, {"momentum": 2.0})
print(f"do(momentum := 2.0): prediction {r['prediction_before']} -> {r['prediction_after']}")

# 5) full checkup: causal graph, counterfactuals, faithfulness, auditor
#    (deterministic report, no LLM needed)
print()
print(session.offline_report(sample_index=0))
