# CausalLens: 让所有黑箱模型变透明

CausalLens 是一个审讯黑箱模型的框架。接受任何有 `predict` 函数的模型，执行干预实验、忠实性验证，以及（配置 LLM 后）贝叶斯假设检验。

给定模型文件和背景数据，输出：

- 带概念包角色标注的特征归因
- 反事实分析：翻转预测的最小输入改动
- torch 模型内部分析：激活修补、SAE。电路发现仅适用于非 RNN 架构。
- 对自身解释的忠实性验证
- 泄漏检测：训练/测试污染、标签泄漏、偷看偏差
- 分布漂移监控

## 示例：信贷风险模型

Kaggle 上的 LightGBM 模型，22 个特征，借贷数据集训练。

归因结果：`person_age` 占总归因份额的 40.8%。

SHAP 不执行的额外检查：

| 检查项 | 结果 |
|---|---|
| 忠实性（抹掉 top 特征，与随机基线对照） | 失败。Top-k 掩码效应不强于随机基线。 |
| 干预（设定年龄为不同值，测量预测变化） | 年龄对预测有因果效应。确认该特征被使用，但整体解释质量仍差。 |
| 审查员（概念包角色标注） | 标记年龄为公平性风险 |
| 证据账本 | 所有发现以链式哈希记录 |

来自不同 Kaggle 作者的 Random Forest，同一数据集训练，显示了不同的归因分布：

| | Random Forest | LightGBM |
|---|---|---|
| 第一因子 | 贷款收入比 34% | person_age 40.8% |
| 忠实性 | 75% 通过率 | 0% 通过率 |

## 与 SHAP 的对比

CausalLens 内部使用 SHAP 做特征归因。SHAP 是其中一个组件。以下能力 SHAP 不具备：

- 直接干预：设定特征为具体值并测量预测变化（SHAP 的 intervention mode 是在背景分布上边际化，不同方法）
- 反事实搜索：翻转预测的最小特征改动
- 激活修补、电路发现、SAE（torch 模型）
- 忠实性验证：掩码测试、跨样本档案、explain-simulate-score
- 泄漏检测：污染、标签泄漏、偷看
- 漂移监控：背景分布过时检测
- 回测引擎审计：成交假设分析、滑点压力
- 决策日志审讯：替代模型加保真度边界
- LLM 调查员：假设驱动，贝叶斯更新，优势比不足时不下结论

## 解释深度

```
1  特征归因      SHAP + 概念接地                    全部模型
2  因果图        直接 / 交互 / 相关边               全部模型
3  层级中介      激活修补                           torch
4  单元电路      h_T 单元 + ACDC-lite              torch
5  参数功能      可逆消融                           torch
6  语义字典      SAE-lite + 因果证伪               torch
```

Level 2-6 使用干预实验。Level 1 使用 SHAP。结论附带忠实性边界。

## 测试结果

21 个模型成功审讯，覆盖 15 个领域（24 个尝试，3 个因 pickle 不兼容失败）。

| 领域 | 模型 | 发现 |
|---|---|---|
| 信贷 | credit_lgbm (Kaggle) | 年龄 40.8% (公平性风险) |
| 信贷 | credit_rf (Kaggle) | 贷款收入比 34% |
| 农业 | crop_rec (Kaggle) | 磷 29%，钾 16% |
| 医疗 | heart_risk_rf (Kaggle) | ST 压低 19%，最大心率 11% |
| 食品 | wine_quality (Kaggle) | 挥发性酸度 26%，酒精 24% |
| 生物 | protein_bilstm (Kaggle) | 从 state_dict 推断架构 |
| 反垃圾 | spam_sms (Kaggle) | 移除词频 9%，美元符号 8% |
| 地质 | earthquake_reg (Kaggle) | f1 73% |
| 环境 | water_quality (Kaggle) | pH 23%，氯胺 19% |
| 加密 | v3d_prod（生产模型） | 42/50 特征漂移 |
| 加密 | m4h_long/short（生产模型） | ATR 家族 32-40% |

## 案例

### 信贷风险：年龄偏见

**模型：** credit_lgbm（LightGBM，22 个独热编码特征，Kaggle）

同一数据集上的两个模型归因分布：

| | credit_rf | credit_lgbm |
|---|---|---|
| 第一因子 | 贷款收入比 34% | person_age 40.8% |
| 第二 | 个人收入 16% | 贷款目的-教育 14.3% |
| 忠实性 | 75% 通过 | 失败（faithfulness 和 simulate_score 检查） |

年龄占借贷决策的 40.8% 违反多个司法辖区的公平性法规。审查员通过概念包角色标注标记了此问题。


### 心脏病：与临床共识一致

**模型：** heart_risk_rf（Random Forest，18 个特征，Kaggle）

| 排名 | 因子 | 归因 |
|---|---|---|
| 1 | ST 段压低最大值 | 19% |
| 2 | ST 段斜率向上 | 14% |
| 3 | 最大心率 | 11% |
| 4 | 运动诱发心绞痛 | 9% |

与心脏病学运动测试指标一致。概念包由 GLM 从特征名起草，未提供医学知识。

### 蛋白质 BiLSTM：无源码

**模型：** protein_bilstm（2 层 BiLSTM，嵌入 50001 到 128，Kaggle）

无源码，仅有 `.pt` 权重文件。

1. 使用 CausalLens 的检查工具从 state_dict 键名手动推断架构
2. 手动重建模型类：`Embedding(50001, 128)` + `LSTM(128 到 256, 双向)` + `FC(512 到 10)`
3. 以 `strict=True` 加载权重
4. 神经元排序定位决策单元

## 快速上手

```bash
pip install -e .
python -m causallens hero
```

```bash
python -m causallens ask --demo geo --concepts concepts/geo.json \
    "if precipitation increases by 50%, how does landslide risk change?"
```

```
[Intervention] do(precipitation := 1853)   1235 → 1853 (+50%)
[Answer] System probability: 0.685 → 0.725 (+0.040)
[Boundary] In-model reasoning, not real-world prediction
```

```bash
python -m causallens attach --model your_model.pkl --background data.csv --domain "credit risk"
python -m causallens visualize --demo timeseries
```

## 架构

```
入口     CLI / MCP Server (stdio+HTTP) / LLM 调查员
核心     LensSession 统一审讯 API
分析     SHAP、do 干预、反事实、因果图、
          符号蒸馏、四档归因、FastSHAP
验证     单样本忠实性、跨样本档案、
          explain-simulate-score
深水     激活修补、单元电路、SAE-lite、
          参数消融、steering
泄漏     污染、标签泄漏、偷看、缩水夏普
驱动     tree、sklearn、torch、timeseries、ensemble、
          onnx、pipeline、backtest、decision_log
领域     概念包：金融、医疗、时序、地理、信贷、回测
档案     SQLite 链式哈希、双时态假设、跨版本先验
```

## 调查员

LLM 调查员形成竞争假设、设计实验、在服务端执行贝叶斯更新。结论要求优势比大于 5.0。

测试结果：GLM 输出为“优势比 1.9，低于 5.0，证据不足。”

## 证据账本

- 链式哈希：修改任何证据条目会中断验证
- 双时态假设：“发现 X 在 v3 成立、v4 失效”
- 跨版本先验：新审讯加载历史结果

## 泄漏检测

| 工具 | 检测内容 |
|---|---|
| duplicate_overlap | 训练/测试行级重叠 |
| single_feature_leak_scan | 单特征近乎完美的 AUC |
| lookahead_scan | 特征滞后一期后信息量下降 |
| deflated_sharpe | 扣除选择偏差后的夏普比率 |

## LLM 支持

| 供应商 | 变量 |
|---|---|
| 智谱 GLM | ZHIPUAI_API_KEY |
| OpenAI | OPENAI_API_KEY |
| DeepSeek | DEEPSEEK_API_KEY |
| 通义千问 | DASHSCOPE_API_KEY |
| Moonshot | MOONSHOT_API_KEY |
| 自定义 | MIC_LLM_BASE_URL + MIC_LLM_API_KEY |

## 自验证

已知真值的测试探针：

| 探针 | 测试 | 结果 |
|---|---|---|
| linear | 归因排名 vs 真系数 | Spearman = 1.000 |
| leak | 注入泄漏特征 | 定位到 84.9% 归因 |
| interact | y = x1·x2 | 检测到 3 条干预边 |

## 参照

MAIA（MIT CSAIL）、Anthropic Circuit Tracing、OpenAI auto-interp、LLM-as-an-Investigator、Causal-INSIGHT、PBO and Deflated Sharpe（Bailey, Lopez de Prado）、SHAP（Lundberg, Lee）、TimeSHAP（Feedzai）、LSTMVis、Karpathy et al. 2015、FastSHAP（ICLR 2022）、Captum、GS-SHAP（2026）

## 许可证

AGPL-3.0
