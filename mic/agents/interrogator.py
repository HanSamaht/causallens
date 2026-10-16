"""审讯智能体：LLM 通过工具循环对模型做实验并回答/审查/建议。

无 LLM 配置时，ask() 自动退化为"离线报告"模式，框架依然可用。
"""
from __future__ import annotations

import json

from ..core.session import MicSession
from .llm import LLMClient
from .prompts import AUDITOR, FIXER, TRANSLATOR

ROLE_PROMPTS = {"translator": TRANSLATOR, "auditor": AUDITOR, "fixer": FIXER}

MAX_TOOL_ROUNDS = 8


class Interrogator:
    def __init__(self, session: MicSession, llm: LLMClient | None = None, role: str = "translator"):
        self.session = session
        self.llm = llm
        self.role = role if role in ROLE_PROMPTS else "translator"

    # ---- 暴露给 LLM 的工具清单（OpenAI function calling 格式） -------------
    def tool_schemas(self) -> list[dict]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "describe_model",
                    "description": "查看被审讯模型的基本信息（类型、特征列表、预测语义）",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_attribution",
                    "description": "获取某样本的预测与逐特征 SHAP 归因（含概念接地与角色标注）",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "intervene",
                    "description": "干预实验：把指定特征设为指定值，观察预测如何变化（do 操作）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sample_index": {"type": "integer"},
                            "assignments": {
                                "type": "object",
                                "description": "特征名 -> 要设定的值，如 {\"momentum\": 1.5}",
                            },
                        },
                        "required": ["sample_index", "assignments"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_causal_graph",
                    "description": "构建因果图：direct 边(SHAP总效应) + interaction 边(干预实验) + correlational 边(仅相关)",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_counterfactual",
                    "description": "寻找让预测翻转的最小特征改动（反事实）",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_faithfulness",
                    "description": "忠实性验证：完备性/充分性/随机基线对照",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "audit",
                    "description": (
                        "运行确定性审查（注意范围：可疑/代理特征依赖、单因子集中度、单样本与跨样本忠实性、"
                        "explain-simulate-score）。不包含分布外(OOD)测试与数据漂移检测——"
                        "未做过的检查不得宣称通过"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_faithfulness_profile",
                    "description": "跨样本忠实性档案（12-20 个样本聚合）：通过率、完备性分位数、整体判定",
                    "parameters": {
                        "type": "object",
                        "properties": {"n_samples": {"type": "integer", "description": "默认 20"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "simulate_score",
                    "description": "explain→simulate→score：SHAP声称的重要性 vs 干预实测敏感度的Spearman一致性打分",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_samples",
                    "description": "列出代表性样本（最阳性/最阴性/边界），拿到可审讯的样本索引",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": "integer", "description": "返回数量，默认 5"}},
                    },
                },
            },
        ]
        if self.session.has_activation_tools():
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": "get_activations",
                        "description": "读取某样本前向传播中每层的激活统计",
                        "parameters": {
                            "type": "object",
                            "properties": {"sample_index": {"type": "integer"}},
                            "required": ["sample_index"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "patch_activation_test",
                        "description": "激活修补实验（因果中介分析）：把参照样本的某层激活换成本样本的，测输出恢复",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "sample_index": {"type": "integer", "description": "干净样本"},
                                "reference_index": {"type": "integer", "description": "被修补的样本"},
                                "layer": {"type": "string"},
                            },
                            "required": ["sample_index", "reference_index", "layer"],
                        },
                    },
                },
            ]
        tools += [
            {
                "type": "function",
                "function": {
                    "name": "form_hypotheses",
                    "description": "形成2-4条相互竞争、可检验的假设（完整陈述句，含可证伪的量）——调查的第一步",
                    "parameters": {
                        "type": "object",
                        "properties": {"statements": {"type": "array", "items": {"type": "string"}}},
                        "required": ["statements"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_hypotheses",
                    "description": "查看假设当前状态与后验概率（服务端维护，不可篡改）",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_hypotheses",
                    "description": "实验后更新假设：evidence_summary必须引用数字，direction=supports/against/neutral",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "updates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "hypothesis_id": {"type": "string"},
                                        "tool_used": {"type": "string"},
                                        "evidence_summary": {"type": "string"},
                                        "direction": {"type": "string", "enum": ["supports", "against", "neutral"]},
                                        "weight": {"type": "number", "description": "证据强度0.2-5，默认1"},
                                    },
                                    "required": ["hypothesis_id", "tool_used", "evidence_summary", "direction"],
                                },
                            }
                        },
                        "required": ["updates"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "conclude_investigation",
                    "description": "给出调查结论：优势比达标宣布胜出假设，否则如实报告inconclusive",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_contamination",
                    "description": "训练/测试污染检测：行级重叠+标签口径对账（data_ref 引用已注册测试数据；标签可用 register_data 的 y_bg/y_new 说明或省略）",
                    "parameters": {
                        "type": "object",
                        "properties": {"data_ref": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "leak_scan",
                    "description": "单特征泄漏扫描（需真实标签，通常由宿主注册 datasets['y']）——找单独就打满AUC的泄漏嫌疑特征",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "symbolic_distill",
                    "description": "符号蒸馏：把模型行为压缩成一条可审计公式（'模型在算什么'的最压缩表达）",
                    "parameters": {
                        "type": "object",
                        "properties": {"n_samples": {"type": "integer", "description": "默认 400"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_drift",
                    "description": "分布漂移检测（KS+PSI+预测漂移）。优先用 data_ref 引用已注册数据集（先用 list_datasets 查看可用名）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_ref": {"type": "string", "description": "已注册数据集名，如 'val_2026_10'"},
                            "new_data": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "number"}},
                                "description": "小规模内联新数据（大于20行请改用 data_ref）",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "register_data",
                    "description": "（一般由宿主程序调用）注册服务端数据集供工具引用",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "data": {"type": "array"}},
                        "required": ["name", "data"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_datasets",
                    "description": "列出已注册的服务端数据集（漂移检测的可用引用）",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        if self.session.has_backtest_internals():
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": "slippage_stress",
                        "description": "滑点压力测试：PF对成交劣化的生存曲线；PF在几十bps内塌缩到1='edge依赖成交假设'的特征信号",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "assumption_gap",
                        "description": "唯一变量对照：阶梯价记账 vs 市价成交，量化回测器成交假设造成的PF虚增倍数",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "live_faithfulness",
                        "description": "回测vs实盘吻合度（喂入两边实测指标：pf/win_rate/avg_pnl_pct）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "backtest": {"type": "object"},
                                "live": {"type": "object"},
                            },
                            "required": ["backtest", "live"],
                        },
                    },
                },
            ]
        if self.session.has_torch_internals():
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": "param_ablation",
                    "description": "可逆损毁：消融某区域参数测功能变化（如 gru.weight_hh 是否承担长期记忆）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param_groups": {"type": "array", "items": {"type": "string"}},
                            "metric": {"type": "string", "enum": ["pred_center", "early_window_share"]},
                        },
                        "required": ["param_groups"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "discover_circuit",
                    "description": "ACDC-lite：自动找出承载当前行为的最小单元子电路（贪心反向剔除）",
                    "parameters": {
                        "type": "object",
                        "properties": {"tolerance": {"type": "number"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sae_features",
                    "description": "SAE-lite：训练稀疏字典自动标注隐特征并做因果证伪（'这块权重在算什么'）",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "neuron_ranking",
                    "description": "梯度式神经元排序（哪些单元在管这个决策；排序=相关级，用 activation_graph 修补做因果确认）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sample_index": {"type": "integer"},
                            "layer": {"type": "string"},
                        },
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "activation_graph",
                        "description": "激活级因果图（circuit tracing lite）：层级+单元级干预修补，定位决策关键内部组件",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "sample_index": {"type": "integer"},
                                "reference_index": {"type": "integer"},
                            },
                            "required": ["sample_index", "reference_index"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "steering_repair",
                        "description": "修复闭环：提取 steering vector 削弱对可疑特征的依赖（推理时干预），并复诊验证效果",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "sample_index": {"type": "integer"},
                                "feature": {"type": "string", "description": "可疑特征名"},
                                "layer": {"type": "string", "description": "目标层名，可选"},
                            },
                            "required": ["feature"],
                        },
                    },
                },
            ]
        return tools

    def _execute_tool(self, name: str, args: dict) -> dict:
        s = self.session
        dispatch = {
            "describe_model": lambda: s.describe_model(),
            "get_attribution": lambda: s.get_attribution(int(args["sample_index"])),
            "intervene": lambda: s.intervene(int(args["sample_index"]), args["assignments"]),
            "get_causal_graph": lambda: s.get_causal_graph(args.get("sample_index")),
            "run_counterfactual": lambda: s.run_counterfactual(int(args["sample_index"])),
            "run_faithfulness": lambda: s.run_faithfulness(int(args["sample_index"])),
            "audit": lambda: s.audit(args.get("sample_index")),
            "run_faithfulness_profile": lambda: s.run_faithfulness_profile(int(args.get("n_samples", 20))),
            "simulate_score": lambda: s.simulate_score(int(args["sample_index"])),
            "list_samples": lambda: s.list_samples(int(args.get("n", 5))),
            "get_activations": lambda: s.get_activations(int(args["sample_index"])),
            "patch_activation_test": lambda: s.patch_activation_test(
                int(args["sample_index"]), int(args["reference_index"]), args["layer"]
            ),
            "check_contamination": lambda: s.check_contamination(None, args.get("data_ref")),
            "leak_scan": lambda: s.leak_scan(s.datasets.get("y", [1] * len(s.driver.background))),
            "symbolic_distill": lambda: s.symbolic_distill(int(args.get("n_samples", 400))),
            "check_drift": lambda: s.check_drift(args.get("new_data"), args.get("data_ref")),
            "register_data": lambda: s.register_data(args["name"], args["data"]),
            "list_datasets": lambda: s.list_datasets(),
            "form_hypotheses": lambda: s.form_hypotheses(args["statements"]),
            "get_hypotheses": lambda: s.get_hypotheses(),
            "update_hypotheses": lambda: s.update_hypotheses(args["updates"]),
            "conclude_investigation": lambda: s.conclude_investigation(),
            "slippage_stress": lambda: s.slippage_stress(),
            "assumption_gap": lambda: s.assumption_gap(),
            "live_faithfulness": lambda: s.live_faithfulness(args["backtest"], args["live"]),
            "param_ablation": lambda: s.param_ablation(args["param_groups"], args.get("metric", "pred_center")),
            "discover_circuit": lambda: s.discover_circuit(tolerance=float(args.get("tolerance", 0.02))),
            "sae_features": lambda: s.sae_features(),
            "neuron_ranking": lambda: s.neuron_ranking(int(args["sample_index"]), args.get("layer")),
            "activation_graph": lambda: s.activation_graph(
                int(args["sample_index"]), int(args["reference_index"])
            ),
            "steering_repair": lambda: s.steering_repair(
                args.get("sample_index"), args["feature"], args.get("layer")
            ),
        }
        if name not in dispatch:
            return {"error": f"未知工具 {name}"}
        try:
            return dispatch[name]()
        except Exception as e:  # 工具失败要让 LLM 看到错误，而不是中断对话
            return {"error": f"{type(e).__name__}: {e}"}

    # ---- 对话主循环 -------------------------------------------------------
    def ask(self, question: str) -> str:
        if self.llm is None:
            return self._offline_answer(question)

        messages = [
            {"role": "system", "content": ROLE_PROMPTS[self.role]},
            {
                "role": "user",
                "content": f"被审讯模型信息：{json.dumps(self.session.describe_model(), ensure_ascii=False)}\n\n问题：{question}",
            },
        ]
        tools = self.tool_schemas()
        start_log = len(self.session.evidence_log)

        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.llm.chat(messages, tools=tools)
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # 协议执行护栏：调查员形成假设后未下结论就收尾 → 强制补一轮
                if self.role == "auditor" and self.session.investigation.hypotheses:
                    concluded = any(
                        e["tool"] == "conclude_investigation"
                        for e in self.session.evidence_log[start_log:]
                    )
                    if not concluded:
                        messages.append({"role": "assistant", "content": msg.get("content") or ""})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "调查协议未完成：假设已形成但未收敛。请继续设计区分性实验并调用 "
                                    "update_hypotheses；证据足够后必须调用 conclude_investigation "
                                    "（不达标也允许如实报告 inconclusive）。"
                                ),
                            }
                        )
                        continue
                return msg.get("content") or "(空回复)"
            # 回显原始 tool_calls（保持各供应商协议一致），执行用归一化版本
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
            from .llm import LLMClient

            for tc in LLMClient.normalize_tool_calls(msg):
                result = self._execute_tool(tc["name"], tc["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False, default=str)[:6000],
                    }
                )
        return "（达到工具调用轮次上限；以上是已完成的实验证据，请基于现有证据回答）"

    # ---- 无 LLM 退化 ------------------------------------------------------
    def _offline_answer(self, question: str) -> str:
        samples = self.session.list_samples(5)["samples"]
        idx = samples[0]["index"] if samples else 0
        if self.role == "auditor":
            return json.dumps(self.session.audit(idx), ensure_ascii=False, indent=2)
        if self.role == "fixer":
            audit = self.session.audit(idx)
            suggestions = [
                f"- {f['message']}\n  建议：核查该特征的数据来源与稳定性；用 intervene 工具预演'屏蔽该特征'后预测分布的变化；必要时从训练集中移除并重训对比。"
                for f in audit["findings"]
                if f["code"] != "none"
            ]
            return "【离线修复建议（无 LLM，基于确定性审查）】\n" + ("\n".join(suggestions) or "- 未发现问题，无需修复。")
        return self.session.offline_report(idx)
