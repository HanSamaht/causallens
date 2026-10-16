# MIC — 模型的话筒 🎙️

> **让每个黑箱模型开口说话。**
> 它靠什么做决定？改变什么会翻转预测？哪些神经元在承载决策？这个解释本身可不可信？

**相关性告诉你什么一起发生，因果性告诉你改变什么会发生什么。MIC 做的是后者。**

给 MIC 一个模型（模型文件 + 背景数据），它会告诉你：
- 模型**靠什么做决定**（因果归因，不只是相关）
- **改变哪个条件**，预测会翻转（反事实推演）
- 模型内部**哪些神经元在承载决策**（电路发现、SAE 特征字典）
- 这个模型的解释**本身可不可信**（三层忠实性验证）

这不是"解释工具"——解释是被动的。**这是因果推演引擎：主动干预实验 + 贝叶斯假设检验 + 证据链纪律。**

```bash
pip install -e .
python -m mic hero    # 30 秒看懂：一个成交假设虚增回测 8 倍（无需数据/GPU/API）
```

---

## 30 秒理解 MIC 在做什么

你有一个量化交易模型。回测显示 PF=5、胜率 56%——看起来是棵摇钱树。

你问 MIC："这靠谱吗？"

MIC 做了一次干预实验：把回测引擎的成交假设从"按理想价格成交"翻转为"按实际市价成交"：

```
do(成交语义 := 市价)  →  胜率 56% → 27%  |  PF 4.97 → 0.62
⚡ 唯一变量对照 → PF 虚增 8.0 倍
```

**一个隐藏假设，虚增了 8 倍的"盈利能力"。** 这不是解释——这是因果推演。

---

## 为什么不是又一个 XAI 工具

| | SHAP/LIME 类 | MiroFish 类模拟 | **MIC** |
|---|---|---|---|
| 回答的问题 | 哪些特征重要（相关） | 未来可能怎样（模拟） | **改变 X 会怎样（干预+反事实）** |
| 证据纪律 | 无 | 无 | **LLM 只引用实验数字；优势比不足时拒绝下结论** |
| 因果分级 | 无 | 无 | **干预证据 / 观测相关 严格分家，相关边强制标注** |
| 可审讯对象 | 模型 | 文档 | **任何有 predict 函数的系统（含回测引擎、决策日志）** |
| 解释深度 | 特征归因 | 表面模拟 | **六级：特征→因果图→层→单元→参数→SAE 语义** |
| 忠实性验证 | 无 | 无 | **三层：掩码→跨样本档案→explain-simulate-score** |

---

## 已验证：15 个领域、24 个真实模型

| 领域 | 模型 | MIC 发现 |
|---|---|---|
| **量化/加密** | v3d_prod（9 Booster 集成） | 42/50 特征漂移；模型学的是旧世界的市场 |
| **量化/加密** | m4h_long/short（LGBM 集成） | ATR 家族占 32-40%——本质是波动率状态机，不是方向预测器 |
| **信贷** | credit_lgbm（Kaggle 陌生模型） | **年龄占归因 40.8% = 公平性红旗**（多司法辖区禁止放贷年龄歧视） |
| **医疗** | heart_risk_rf（Kaggle） | ST 段压低 19% + 最大心率 11%——**与临床共识一致** |
| **农业** | crop_rec（Kaggle） | 磷 29% + 钾 16%——土壤养分主导，符合农学 |
| **食品** | wine_quality（Kaggle） | 挥发性酸度 26% + 酒精 24%——**与酿酒学文献一致** |
| **动作** | gesture_gru（Kaggle BiGRU） | 逆向重建+全套深水区：电路 9/16 单元，SAE 特征编码 IMU 信号 |
| **生物** | protein_bilstm（Kaggle） | 从 state_dict 逆向重建 2 层 BiLSTM，神经元排序定位决策单元 |
| **反垃圾** | spam_sms（Kaggle） | 移除词频 9% + 美元符号 8%——垃圾短信典型特征 |

---

## 六级解释深度

```
第 1 级  特征归因      SHAP + 概念接地 + 因果角色占比         全部模型
第 2 级  因果图        direct / 干预交互 / 相关 三级边         全部模型
第 3 级  层级中介      激活修补：哪层承载决策                  torch/时序
第 4 级  单元电路      h_T 单元 + 持久单元 + ACDC-lite         torch/时序
第 5 级  参数功能      可逆损毁：某区域参数管什么              torch/时序
第 6 级  语义字典      SAE-lite + 自动标注 + 因果证伪          torch/时序
```

每一级都是**干预实验**（不是相关性推断），结论带忠实性边界。

**实测**：GRU 的 SAE 特征与动量因子相关 r=0.83 且通过因果消融——"这块权重在算动量"。

---

## 五分钟上手

### 1. 问一个问题（因果推演）

```bash
python -m mic ask --demo geo --concepts concepts/geo_zh.json \
    "如果降水量增加50%，最典型的样本滑坡风险怎么变"
```
```
【推演】do(降水 := 1853)   1235 → 1853 (+50%)
【回答】该系统概率: 0.685 → 0.725（+0.040）
【边界】模型内推演，非现实预测
```

### 2. 让 AI 调查员审讯模型

```bash
python -m mic chat --demo finance --concepts concepts/finance_zh.json --role auditor
```
AI 调查员会自主：形成竞争假设 → 设计实验 → 贝叶斯更新 → 收敛或如实报告"证据不足"。

### 3. 一键让陌生模型开口

```bash
python -m mic attach --model your_model.pkl --background your_data.csv --domain "信用风控"
```

### 4. 生成 HTML 审讯报告

```bash
python -m mic visualize --demo timeseries --concepts concepts/timeseries_zh.json
# 自动打开浏览器，暗色主题，含归因+电路+SAE+忠实性面板
```

### 5. 更多命令

| 命令 | 用途 |
|---|---|
| `mic hero` | 30 秒英雄演示（回测虚增 8 倍） |
| `mic report --demo finance` | 完整审讯报告（归因/因果图/反事实/忠实性） |
| `mic audit --demo agent` | 审讯无 predict 函数的系统（决策日志→替代模型） |
| `mic drift` | 部署漂移检测（背景分布过时→SHAP 基准失效） |
| `mic graph-export / graph-apply` | 因果图人工审查修正（新增边自动过干预实验验证） |
| `mic probe` | 框架自检（已知真值的合成探针回归测试） |
| `mic llm-check` | LLM 配置自检（配好 key 后先跑这个） |
| `mic history` | 案件档案库（跨版本审讯记忆） |

---

## 核心能力

### 8 种驱动器——任何有 predict 函数的系统即插即审

| 驱动器 | 适用模型 | 专属能力 |
|---|---|---|
| `TreeDriver` | sklearn/LightGBM/XGBoost | SHAP 快路径 |
| `TorchDriver` | PyTorch（MLP） | 激活修补/steering/梯度归因 |
| `TimeSeriesDriver` | GRU/LSTM（时序） | lag 级干预 + h_T/持久单元双语义 + 四档归因 |
| `EnsembleTreeDriver` | 多模型加权集成 | SHAP 精确分解 |
| `OnnxDriver` | ONNX 格式 | 安全加载（无任意代码执行风险） |
| `PipelineDriver` | sklearn Pipeline | 自动拆步骤 |
| `BacktestDriver` | 回测引擎 | 成交假设缺口/滑点压力/回测-实盘忠实性 |
| `DecisionLogDriver` | 决策日志/智能体轨迹 | 替代模型 + 动作一致率边界 |

### 四档时序归因——精度/速度/可读性用户自选

```python
s.ts_attribution(0, granularity="channel")   # 通道级（最快，秒级）
s.ts_attribution(0, granularity="segment")   # 通道×时段（均衡）
s.ts_attribution(0, granularity="full")      # 全分辨率（最准）
s.ts_attribution(0, granularity="gradient")  # 梯度近似（毫秒级兜底）
s.ts_attribution(0, granularity="auto")      # 自动推荐
```

### FastSHAP 摊销解释器——全分辨率从分钟变毫秒

```python
s.fit_fastshap()           # 训练一次（CPU 秒级）
s.fastshap_attribution(0)  # 之后任意样本毫秒级出全分辨率 SHAP
```

### Investigator 循环——LLM 假设-实验-证据

LLM 调查员不是"调 audit 然后复述"——它自主形成竞争假设、设计区分性实验、服务端贝叶斯更新（LLM 无法篡改），优势比>5 才许结论，否则如实报告 inconclusive。

实测：GLM 最终回答"优势比 1.9<5.0，证据不足以区分假设"——**AI 拒绝过度宣称**。

### 证据账本与案件档案

- **链式哈希防篡改**：篡改任意一条证据，整条链断裂
- **双时态假设**："发现 X 在 v3 成立、于 v4 失效"
- **跨版本先验**：新版本调查自动加载历史假设胜负

### 泄漏与污染检测

| 工具 | 检测什么 |
|---|---|
| `duplicate_overlap` | 训练/测试行级重叠（污染实锤） |
| `single_feature_leak_scan` | 某特征单独就打满 AUC（标签泄漏） |
| `lookahead_scan` | 特征滞后一期后 IC 断崖（偷看未来） |
| `deflated_sharpe` | PF/夏普扣掉挑选运气后还剩多少 |

---

## 深水区三件套（torch 系模型的终极审讯）

| 工具 | 什么能力 | 实测 |
|---|---|---|
| `param_ablation` | 可逆损毁：某区域参数发挥什么功能 | `gru.weight_hh` → 长期记忆功能 |
| `discover_circuit` | ACDC-lite：最小单元子电路（greedy/exhaustive） | GRU 16 单元中 10 个成电路、6 个可剔除 |
| `sae_features` | SAE-lite：隐状态稀疏字典 + 自动标注 + 因果证伪 | 特征与动量相关 r=0.83 且通过消融 |

---

## 接入你自己的模型（三行代码）

```python
from mic.core import ConceptPack, MicSession
from mic.drivers import detect_driver

driver = detect_driver(model, feature_names, background_data)
mic = MicSession(driver, ConceptPack.load("concepts/finance_zh.json"))
print(mic.offline_report(sample_index=0))
```

---

## 架构

```
入口    CLI(15+命令) · MCP Server(stdio/HTTP) · LLM 调查员(翻译官/审查员/修复师)
核心    MicSession 统一审讯 API —— 证据一致的单一来源
分析    SHAP归因(概念接地) · do干预 · 多路径反事实 · 干预式因果图 · 符号蒸馏
        · 四档归因(channel/segment/full/gradient) · FastSHAP
验证    单样本忠实性 · 跨样本档案 · explain→simulate→score · 合成探针
深水    激活修补 · 单元电路(greedy/exhaustive) · SAE-lite · 参数消融 · steering
泄漏    污染扫描 · 标签泄漏 · 偷看检测 · 缩水夏普
驱动    tree · sklearn · torch · timeseries · ensemble · onnx · pipeline · backtest · decision_log
领域    概念包: 金融/医学/时序/地理/信用/回测（LLM 自动起草 + 专家审核闭环）
档案    SQLite 链式哈希 · 双时态假设 · 跨版本先验
```

---

## 支持的 LLM

| 供应商 | 环境变量 | 默认模型 |
|---|---|---|
| 智谱 GLM | `ZHIPUAI_API_KEY` | glm-4-flash（免费） |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
| 通义千问 | `DASHSCOPE_API_KEY` | qwen-plus |
| Kimi | `MOONSHOT_API_KEY` | moonshot-v1-8k |
| 任意自定义 | `MIC_LLM_BASE_URL` + `MIC_LLM_API_KEY` | — |

---

## 忠实性是架构，不是口号

LLM 的一切结论必须来自它调用的干预实验。大数据不进对话（服务端数据注册表）、假设状态在服务端（无法篡改）、每条证据进日志可回溯。框架自身用已知真值的合成探针回归——归因排名 vs 真系数 Spearman=1.000、注入泄漏特征 84.9% 占比被审查员当场抓获。

---

## 方法论参照

MAIA（MIT CSAIL）· Anthropic Circuit Tracing · OpenAI auto-interp · LLM-as-an-Investigator · Causal-INSIGHT · PBO/Deflated Sharpe（Bailey & López de Prado）· SHAP（Lundberg & Lee）· TimeSHAP（Feedzai）· LSTMVis · Karpathy et al. 2015 · FastSHAP（ICLR 2022）· Captum（可选后端）· GS-SHAP（2026）

---

## License

MIT
