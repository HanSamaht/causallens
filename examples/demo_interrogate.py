"""MIC 编程式使用示例：五步审讯一个金融量化模型。

运行： python examples/demo_interrogate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.core import ConceptPack, MicSession
from causallens.demos import train_finance_demo
from causallens.drivers import detect_driver

# 1) 准备模型（你的场景：换成自己的模型 + 背景数据即可）
model, feats, background, auc = train_finance_demo()
print(f"模型 auc = {auc:.4f}")

# 2) 插上麦克风：自动识别驱动器 + 加载领域概念包
driver = detect_driver(model, feats, background, name="my_quant_model")
pack = ConceptPack.load(os.path.join("concepts", "finance.json"))
mic = MicSession(driver, pack)

# 3) 让模型开口：一句话拿到单样本完整解释（含中文概念接地）
result = mic.get_attribution(sample_index=0)
print(f"\n样本#0 预测 = {result['prediction']}")
for a in result["attributions"][:3]:
    print(f"  {a['concept']}({a['feature']}) 值={a['value']:+.2f} 贡献={a['shap']:+.3f}")

# 4) 问"如果……会怎样"：干预实验
r = mic.intervene(0, {"momentum": 2.0})
print(f"\n干预 do(动量:=2.0): 预测 {r['prediction_before']} → {r['prediction_after']}")

# 5) 全面体检：因果图 + 反事实 + 忠实性 + 审查员（确定性报告，无需 LLM）
print()
print(mic.offline_report(sample_index=0))
