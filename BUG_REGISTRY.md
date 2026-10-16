# BUG 注册表（Bug Registry）

> 章程见 AGENTS.md：发现即登记，修复即补记。任何条目不得删除，只能标 superseded。
> 格式：`ID | 日期 | 区域 | 症状 | 根因 | 修复 | 状态`

## A. 已修复的框架 Bug

- BUG-001 | 2026-10-13 | core | ImportError: partially initialized module | core/__init__ 急切导入 session，session 又导入 analysis，analysis 回头导入 core.concepts 成环 | PEP 562 惰性 `__getattr__` 导出 MicSession | ✅
- BUG-002 | 2026-10-13 | server/mcp | MCP 客户端 JSON 解析失败 | `_load_session` 诊断信息打到 stdout，污染 stdio JSON-RPC 协议通道 | 诊断一律走 stderr | ✅
- BUG-003 | 2026-10-13 | analysis/attribution | 探针测试偶发失败、归因错乱 | SHAP 解释器缓存以 `id(driver)` 为键，对象回收后 id 复用 → 新模型拿到别人的解释器 | 缓存挂到 driver 实例属性 `_mic_explainer` | ✅（严重：静默错结果）
- BUG-004 | 2026-10-13 | analysis/activation_graph | 层级中介效应恒为 0 | 中介方向写反：把参照样本自己的激活替换回它自己的前向（等于没换） | 修正为 clean→corrupt 移植范式（sample 激活植入 reference 前向） | ✅
- BUG-005 | 2026-10-13 | drivers/ensemble | `TypeError: 'Booster' object is not subscriptable` | v3d_prod 成员是 (Booster, weight) 元组，book_models 是裸 Booster，驱动只认前者 | 构造时归一化两种格式 | ✅
- BUG-006 | 2026-10-13 | cli | timeseries demo `UnboundLocalError: auc` | 分支设置 `r2`，共享打印行引用 `auc` | 统一指标变量 | ✅
- BUG-007 | 2026-10-13 | cli | `--demo backtest` 构造的 driver 被分支链覆盖 | 共享尾部分支 `detect_driver(model,...)` 无条件执行 | backtest/agent 走 pass 特判 | ✅
- BUG-008 | 2026-10-13 | drivers/onnx | `TypeError: unsupported operand - dict and float` | skl2onnx 默认 ZipMap 输出 list-of-dict，predict 按数组处理 | OnnxDriver 兼容 dict 输出 | ✅
- BUG-009 | 2026-10-13 | drivers/onnx | ImportError 提示"需要 onnxruntime"但 onnxruntime 已安装 | 多分类重写删了 `_predict_proba_pos`，onnx 的僵尸导入炸出 ImportError，被宽泛 except 包装成误导性消息 | 删除僵尸导入；教训：except 不能吞真实根因再换皮 | ✅
- BUG-010 | 2026-10-13 | analysis/counterfactual | SyntaxError: default 参数在非 default 前 | 给 `_search_large` 插参数时位置放错 | 调整到尾部 | ✅
- BUG-011 | 2026-10-13 | agents | GLM 对 150×50 矩阵的 function call 格式崩溃（工具名和参数混进正文） | 大 payload 内联进对话破坏小模型的函数调用格式 | 服务端数据注册表（register_data/data_ref），大数据永不进对话 | ✅（架构级修复）
- BUG-012 | 2026-10-13 | agents/prompts | 审查员宣称"分布外脆弱性通过"但 audit 根本不做 OOD | 工具描述范围不清，LLM 越界宣称 | 收紧工具描述并明示范围；未做的检查不得宣称通过 | ✅
- BUG-013 | 2026-10-13 | agents/prompts | 审查员"调 audit 然后复述"，不主动实验 | 8 轮工具循环无假设空间/停止阈值；glm-flash 顺从倾向 | Investigator 协议（互斥假设+零假设+服务端贝叶斯+优势比阈值）+ 收尾护栏 | ✅
- BUG-014 | 2026-10-13 | ask | GLM 自作主张把用户没提的坡度也 +21% 干预 | LLM 输出无范围约束 | `_filter_mentioned`：服务端只放行问题中明确提到的特征 | ✅（纪律级修复）
- BUG-015 | 2026-10-13 | ask | 样本选择器收到模板占位符 "index:N" 崩溃 | LLM 原样抄回 prompt 模板 | 服务端 isdigit 校验 + 常用说法（最典型/最危险）服务端裁决 | ✅
- BUG-016 | 2026-10-13 | ledger | v2 调查读不到 v1 先验 | 先验在结论归档之后读取，此时 v1 同陈述假设已被双时态关闭 | 先验改为调查开始时读取 | ✅
- BUG-017 | 2026-10-13 | ledger | `NameError: case_priors`（2 个测试红） | 连锁补丁失误：第一段补丁因后置断言失败整体未写盘，尾段替换却成功 → 定义缺失只有引用 | 补齐定义块；教训：多段脚本补丁必须原子化，断言失败=全部不落盘 | ✅
- BUG-018 | 2026-10-13 | demos/geo | region_id 伪特征占比 0.6%，审查员抓不到 | 伪影信号强度不足以让模型学习 | 伪影加强至 2.2 + 可疑特征告警阈值降为 3%（先验怀疑者门槛应更低） | ✅
- BUG-019 | 2026-10-13 | analysis | LightGBM SHAP 警告刷屏 | shap 0.53 行为变更提示未过滤 | warnings 上下文管理器抑制 | ✅
- BUG-020 | 2026-10-13 | tests | 多处测试自身错误（舍入容差、编辑条件过宽把 momentum 5 条边全确认） | 测试代码不严谨 | 修正；测试也是代码，同样过审 | ✅

## B. 设计缺口（转为 Roadmap，非 bug）

- GAP-001 | 2026-10-13 | audit 缺"预测分布健康检查"——gat_best 输出 std=0.0013（近常数）是**人工手工**发现的，框架无此工具。教训：近常数输出的模型 IC 靠微小排序差，audit 应含输出分布统计 | 待建
- GAP-002 | 2026-10-13 | 表格驱动器背景数据无时间轴 → 时间泄漏检验（滞后敏感度）无法直接做 | 待建（见 leakage.py 首批工具）
- GAP-003 | 2026-10-13 | m4h sanity 标签是手工粗制代理（+24h>2%），与真实训练标签语义不同——AUC 数字不能当结论用 | 已在汇报中声明

## C. 流程教训（人不修，只记）

- L-001 | 多次 README Write 因"file modified"失败 → 重复读文件耗预算；编辑前必须重读或用脚本原子替换
- L-002 | 2026-10-13 汇报中把"框架发现"与"手工观察"混在同一叙事，被用户质询后逐条摊开——证据分级纪律适用于框架，也适用于框架的开发者本人

- BUG-021 | 2026-10-13 | analysis/leakage | 新模块写入 `analysis/leakage.py`（根目录）而非包内 `mic/analysis/`，导入失败 | 文件落盘路径错位 | 移入包内；写盘后立即 `ls` 验证 | ✅

- BUG-022 | 2026-10-14 | core/session | timeseries 模型调 activation_graph 断言失败 | `has_torch_internals` 只认 kind=="torch"，漏了同为 torch 系的 "timeseries" | 条件扩展为两种 kind；同排检查 steering_repair/neuron_ranking 路径（均已走 torch 断言，一并受益 | ✅

- BUG-023 | 2026-10-14 | drivers/timeseries | timeseries 模型接不了激活因果图（无钩子基础设施，两处断言拦截） | TimeSeriesDriver 未继承 TorchDriver，缺激活钩子/修补机制；assert 与工具门控又硬编码 kind=="torch" | 重构为继承 TorchDriver 只覆写 _to_tensor；断言/门控放行 timeseries | ✅

- BUG-024 | 2026-10-14 | drivers/torch_model | RNN 模型接钩子即崩：'tuple' object has no attribute 'detach' | nn.GRU 前向返回 (output, hidden) 元组，钩子只处理张量 | 钩子兼容元组输出：审讯/修补首元素，返回时保持元组结构 | ✅

- BUG-025 | 2026-10-14 | analysis/neuron_rank | captum 已安装但 engine 仍显示 fallback（静默降级掩盖真实错误） | 宽泛 except Exception 吞掉 captum 路径的真实异常——重蹈 BUG-009 模式 | 调试暴露根因；元组输出层需函数选择器 out[0][0,-1,u]；降级路径现在附带 captum_fallback_reason | ✅

- BUG-026 | 2026-10-14 | analysis/sae_lite | corrcoef 维度崩溃（64 vs 300） | RNN 钩子输出堆叠为 4 维 (n,1,T,H)，collect_hidden 未抽 h_T，隐藏矩阵带假维度 | ndim==4 → [:,0,-1,:]；修补路径同步改 [0,-1,:] | ✅

- BUG-027 | 2026-10-15 | drivers/torch_model | 变长序列模型(packed)接入钩子即崩：'PackedSequence' object has no attribute 'detach' | 作者源码用 pack_padded_sequence，钩子只处理 Tensor | 钩子兼容 PackedSequence（跳过缓存/修补，记录 data 形状标记）| ✅

- BUG-028 | 2026-10-15 | core/session | visualize() 返回 _log dict 而非 path 字符串，调用方 .getsize 崩 | 返回值语义混淆 | _log 后 return path | ✅

- BUG-029 | 2026-10-15 | analysis/attribution | max_evals 参数传入 shap.Explainer 导致 ExactExplainer 崩（TypeError） | shap.Explainer() 的 max_evals 仅对 Permutation 子类合法，对 Exact 子类不认 | 条件传参：只在需要 Permutation 时才传 max_evals | ✅

- BUG-030 | 2026-10-15 | 分析/attribution | shap.Explainer 传 max_evals 对 ExactExplainer 崩（TypeError），3 个测试红 | max_evals 仅对 Permutation 子类合法 | try/except 条件传参 | ✅
