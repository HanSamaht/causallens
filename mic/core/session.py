"""MicSession：统一审讯 API。

CLI（`mic report / audit / chat`）、LLM 智能体（工具调用）、MCP Server（外部 LLM 客户端）
三个入口共用同一套方法——保证"模型说的话"无论从哪个口出来，证据完全一致。

无 LLM 时也能工作：offline_report / audit 输出确定性中文报告。
"""
from __future__ import annotations

import numpy as np

from ..analysis import (
    aggregate_faithfulness,
    build_causal_graph,
    default_threshold,
    evaluate_faithfulness,
    explain_sample,
    explain_simulate_score,
    search_counterfactuals,
)
from ..analysis.drift import check_drift
from ..analysis.symbolic import symbolic_distill
from ..drivers.base import MicDriver
from .concepts import ConceptPack


class MicSession:
    def __init__(self, driver: MicDriver, pack: ConceptPack | None = None):
        self.driver = driver
        self.pack = pack or ConceptPack()
        self.evidence_log: list[dict] = []
        self.datasets: dict[str, "np.ndarray"] = {}
        from ..agents.investigator import HypothesisTracker

        self.investigation = HypothesisTracker()
        self.ledger = None       # 证据账本（可选，attach_ledger 启用持久化）
        self.case_id = None

    # ================= 证据账本（持久化卷宗） =================

    def attach_ledger(self, ledger, model_version: str = "v1") -> dict:
        """接入账本并打开/创建案件档案：之后所有证据自动落库。"""
        self.ledger = ledger
        self.case_id = ledger.open_case(self.driver.name, model_version)
        return self._log("attach_ledger", {"case_id": self.case_id,
                                           "model": self.driver.name, "model_version": model_version})

    def case_briefing(self) -> dict:
        """历史档案简报：同名模型的历史假设胜负（作为调查先验）。"""
        if self.ledger is None:
            return {"note": "未接入账本（mic --ledger mic_ledger.db 启用持久档案）"}
        return {
            "case_id": self.case_id,
            "chain": self.ledger.verify_chain(self.case_id),
            "priors": self.ledger.hypothesis_priors(self.driver.name),
        }

    # ================= Investigator 循环（假设空间 + 证据阈值） =================

    def form_hypotheses(self, statements: list[str]) -> dict:
        return self._log("form_hypotheses", self.investigation.form(statements))

    def get_hypotheses(self) -> dict:
        return self._log("get_hypotheses", self.investigation.snapshot())

    def update_hypotheses(self, updates: list[dict]) -> dict:
        return self._log("update_hypotheses", self.investigation.update(updates))

    def conclude_investigation(self) -> dict:
        v = self.investigation.verdict()
        if self.ledger is not None and self.case_id is not None:
            snap = self.investigation.snapshot()["hypotheses"]
            winner = v.get("winner") if v["verdict"] == "confirmed" else None
            for hid, h in snap.items():
                verdict = ("confirmed" if hid == winner else
                           "rejected" if v["verdict"] == "confirmed" else "inconclusive")
                self.ledger.record_hypothesis(
                    self.case_id, h["statement"], verdict,
                    h["posterior"] or 0.0,
                )
        return self._log("conclude_investigation", v)

    def investigate(self, n_samples: int = 10) -> dict:
        """确定性调查员（无 LLM）：从审查发现自动形成竞争假设，逐个用实验检验。

        升级点：不再是"audit 复述"——每条发现变成一个可竞争的假设，
        实验证据驱动贝叶斯更新，优势比不足就如实报告 inconclusive。
        """
        audit = self.audit(None)
        codes = {f["code"]: f for f in audit["findings"]}

        # 先验在调查开始时读取（早于本次结论归档——历史用于指导调查，而非事后罗列）
        case_priors = None
        if self.ledger is not None:
            try:
                cur_ver = dict(self.ledger.conn.execute(
                    "SELECT model_version FROM cases WHERE id=?", (self.case_id,)
                ).fetchone())["model_version"]
                case_priors = [p for p in self.ledger.hypothesis_priors(self.driver.name)
                               if p["still_valid"] and p["model_version"] != cur_ver] or None
            except Exception:
                case_priors = None

        # ---- 形成假设（来自确定性发现 + 概念包角色标注） ----
        statements = []
        if "spurious_dependence" in codes or "proxy_dependence" in codes:
            statements.append("模型依赖了概念包标注的可疑/代理特征（数据伪影风险）")
        if "concentration" in codes:
            statements.append("模型单因子集中（决策由单一特征主导）")
        if "faithfulness_profile" in codes or "faithfulness" in codes:
            statements.append("归因解释不忠实（解释与模型真实决策脱节）")
        if "simulate_score" in codes:
            statements.append("SHAP声称的重要性与干预实测敏感度不一致（解释失真）")
        statements.append("模型未见明显异常（上述风险均不成立）")

        if len(statements) < 2:
            return {"verdict": "no_findings", "note": "无可检验的候选假设"}
        self.form_hypotheses(statements[:4])
        shares = audit["global_attribution_share"]

        # ---- 逐假设做实验、按数字更新 ----
        profile = self.run_faithfulness_profile(n_samples=n_samples)
        sim = self.simulate_score(0)

        updates = []
        if "spurious_dependence" in codes:
            f = codes["spurious_dependence"]
            updates.append(
                {
                    "hypothesis_id": self._match_hyp("可疑"),
                    "tool_used": "audit",
                    "evidence_summary": f["message"],
                    "direction": "supports",
                    "weight": 3.0,
                }
            )
        if "concentration" in codes:
            top_share = list(shares.values())[0]
            updates.append(
                {
                    "hypothesis_id": self._match_hyp("单因子"),
                    "tool_used": "audit.global_share",
                    "evidence_summary": f"Top特征归因占比 {top_share:.0%}",
                    "direction": "supports",
                    "weight": 2.0,
                }
            )
        # 忠实性证据：双向检验（支持"不忠实"假设的同时反对"未见异常"）
        hyp_unfaith = self._match_hyp("不忠实")
        hyp_clean = self._match_hyp("未见明显异常")
        if profile["verdict"] == "存疑":
            updates.append(
                {
                    "hypothesis_id": hyp_unfaith,
                    "tool_used": "run_faithfulness_profile",
                    "evidence_summary": f"通过率 {profile['pass_rate']:.0%} < 80%",
                    "direction": "supports",
                    "weight": 2.0,
                }
            )
            updates.append(
                {
                    "hypothesis_id": hyp_clean,
                    "tool_used": "run_faithfulness_profile",
                    "evidence_summary": f"通过率 {profile['pass_rate']:.0%} < 80%",
                    "direction": "against",
                    "weight": 2.0,
                }
            )
        else:
            updates.append(
                {
                    "hypothesis_id": hyp_clean,
                    "tool_used": "run_faithfulness_profile",
                    "evidence_summary": f"通过率 {profile['pass_rate']:.0%} ≥ 80%",
                    "direction": "supports",
                    "weight": 1.5,
                }
            )
        if sim["spearman"] is not None:
            d = "supports" if sim["spearman"] < 0.3 else "against"
            updates.append(
                {
                    "hypothesis_id": self._match_hyp("干预实测敏感度不一致"),
                    "tool_used": "simulate_score",
                    "evidence_summary": f"Spearman={sim['spearman']} verdict={sim['verdict']}",
                    "direction": d,
                    "weight": 2.0,
                }
            )
        # 无对应假设的更新自动丢弃（防御性）
        valid = {h.hid for h in self.investigation.hypotheses.values()}
        updates = [u for u in updates if u["hypothesis_id"] in valid]
        if updates:
            self.update_hypotheses(updates)

        result = {
            "hypotheses": self.investigation.snapshot(),
            "verdict": self.conclude_investigation(),
            "audit_findings": [f["message"] for f in audit["findings"]],
        }
        if case_priors:
            result["case_priors"] = case_priors  # 历史假设先验（跨案件记忆，调查开始时已读）
        return result

    def _match_hyp(self, keyword: str) -> str:
        for h in self.investigation.hypotheses.values():
            if keyword in h.statement:
                return h.hid
        return next(iter(self.investigation.hypotheses), "H1")

    def _log(self, tool: str, payload: dict) -> dict:
        self.evidence_log.append({"tool": tool, "payload": payload})
        if self.ledger is not None and self.case_id is not None:
            try:
                self.ledger.append_evidence(self.case_id, tool, payload)
            except Exception:
                pass  # 账本故障不阻断审讯（内存证据仍在）
        return payload

    # ================= 审讯工具（LLM / MCP 也只允许调用这些） =================

    def set_target_class(self, k: int) -> dict:
        """多分类审讯：切换目标类别（归因/反事实聚焦到第 k 类），清空解释器缓存。"""
        self.driver.target_class = int(k)
        if hasattr(self.driver, "_mic_explainer"):
            self.driver._mic_explainer = None
        return self._log("set_target_class", {"target_class": int(k), "n_classes": getattr(self.driver, "n_classes", 2)})

    def describe_model(self) -> dict:
        d = self.driver.describe()
        d["domain"] = self.pack.domain
        d["concept_pack"] = self.pack.name or "(未加载，特征名即概念)"
        d["n_classes"] = getattr(self.driver, "n_classes", 2)
        d["target_class"] = getattr(self.driver, "target_class", 1)
        return self._log("describe_model", d)

    def list_samples(self, n: int = 5) -> dict:
        """按预测置信度挑展示样本：最看涨 + 最看跌 + 边界样本。"""
        p = self.driver.predict(self.driver.background)
        order_hi = np.argsort(-p)
        order_lo = np.argsort(p)
        idxs, seen = [], set()
        for cand in list(order_hi[: n // 2 + 1]) + list(order_lo[: n // 2 + 1]) + [int(np.argmin(abs(p - 0.5)))]:
            if cand not in seen:
                idxs.append(int(cand))
                seen.add(cand)
            if len(idxs) >= n:
                break
        return self._log(
            "list_samples",
            {"samples": [{"index": int(i), "prediction": round(float(p[i]), 4)} for i in idxs]},
        )

    def get_attribution(self, sample_index: int) -> dict:
        x = self.driver.sample_at(sample_index)
        p = self.driver.predict_one(x)
        attrs = explain_sample(self.driver, x, self.pack)
        total = sum(abs(a.shap_value) for a in attrs) or 1.0
        return self._log(
            "get_attribution",
            {
                "sample_index": int(sample_index),
                "prediction": round(p, 4),
                "semantics": self.driver.prediction_semantics,
                "attributions": [
                    {
                        "feature": a.feature,
                        "concept": a.concept,
                        "value": round(a.feature_value, 4),
                        "shap": round(a.shap_value, 4),
                        "share": round(abs(a.shap_value) / total, 4),
                        "role": a.role,
                    }
                    for a in attrs
                ],
            },
        )

    def intervene(self, sample_index: int, assignments: dict[str, float]) -> dict:
        """do(feature := value) 后的预测。因果图的原子操作，对 LLM 开放。"""
        x = self.driver.sample_at(sample_index)
        x2 = self.driver.intervene(x, assignments)
        return self._log(
            "intervene",
            {
                "sample_index": int(sample_index),
                "assignments": assignments,
                "prediction_before": round(self.driver.predict_one(x), 4),
                "prediction_after": round(self.driver.predict_one(x2), 4),
            },
        )

    def get_causal_graph(self, sample_index: int | None = None) -> dict:
        anchor = (
            self.driver.sample_at(sample_index)
            if sample_index is not None
            else np.median(self.driver.background, axis=0)
        )
        g = build_causal_graph(self.driver, anchor)
        edges = [
            e.to_dict() | {"concept": self.pack.concept_of(e.source) if e.kind != "correlational" else ""}
            for e in g.edges
        ]
        return self._log(
            "get_causal_graph",
            {
                "anchor_sample": sample_index,
                "direct_edges": [e for e in edges if e["kind"] == "direct"],
                "interaction_edges": [e for e in edges if e["kind"] == "interaction"],
                "correlational_edges": [e for e in edges if e["kind"] == "correlational"],
                "method": g.method,
            },
        )

    def run_counterfactual(self, sample_index: int) -> dict:
        x = self.driver.sample_at(sample_index)
        paths = search_counterfactuals(self.driver, x, self.pack)
        threshold = default_threshold(self.driver)
        out_paths = []
        for cf in paths:
            out_paths.append(
                {
                    "changed_features": {
                        f: {
                            "concept": self.pack.concept_of(f.split("@")[0]),
                            "from": round(v[0], 4),
                            "to": round(v[1], 4),
                        }
                        for f, v in cf.changed_features.items()
                    },
                    "prediction_after": round(cf.cf_prediction, 4),
                    "distance": round(cf.distance, 4),
                    "method": cf.method,
                }
            )
        best = paths[0] if paths else None
        return self._log(
            "run_counterfactual",
            {
                "sample_index": int(sample_index),
                "threshold": threshold,
                "base_prediction": round(self.driver.predict_one(x), 4),
                "counterfactual_prediction": round(best.cf_prediction, 4) if best else None,
                "changed_features": out_paths[0]["changed_features"] if out_paths else {},
                "flipped": bool(best),
                "paths": out_paths,
            },
        )

    def ts_attribution(self, sample_index: int, granularity: str = "auto",
                       n_segments: int = 4, top_k: int = 10) -> dict:
        """时序四档归因：channel(最快) / segment(均衡) / full(最准) / gradient(兜底) / auto(推荐)。"""
        from ..analysis.granularity import timeseries_attribution

        assert self.driver.kind in ("timeseries", "torch"), "四档归因需要时序/torch 驱动"
        x = self.driver.sample_at(sample_index)
        result = timeseries_attribution(self.driver, x, granularity=granularity,
                                        n_segments=n_segments, top_k=top_k)
        result["sample_index"] = int(sample_index)
        return self._log("ts_attribution", result)

    def fit_fastshap(self, n_samples: int = 150, epochs: int = 200) -> dict:
        """训练 FastSHAP 摊销解释器——之后 full 档归因从分钟级变毫秒级。"""
        from ..analysis.fastshap import AmortizedExplainer
        from ..analysis.attribution import shap_matrix

        assert self.driver.kind in ("timeseries", "torch"), "FastSHAP 需要 torch 系驱动"
        n_feat = len(self.driver.feature_names)
        self._fastshap = AmortizedExplainer(n_feat)
        info = self._fastshap.fit(
            lambda X: shap_matrix(self.driver, X),
            self.driver.background, n_samples=n_samples, epochs=epochs,
        )
        return self._log("fit_fastshap", info)

    def fastshap_attribution(self, sample_index: int) -> dict:
        """毫秒级全分辨率归因（需先 fit_fastshap）。"""
        assert hasattr(self, "_fastshap") and self._fastshap.fitted, "先调用 fit_fastshap()"
        x = self.driver.sample_at(sample_index)
        sv = self._fastshap.attribute(x)
        T = getattr(self.driver, "seq_len", 1)
        F = getattr(self.driver, "n_features_raw", len(self.driver.feature_names))
        base = getattr(self.driver, "base_feature_names", self.driver.feature_names)
        from ..analysis.granularity import _aggregate_channel
        items = _aggregate_channel(sv, T, F, base)
        items.sort(key=lambda a: -abs(a["shap"]))
        return self._log("fastshap_attribution", {
            "engine": "FastSHAP(摊销,毫秒级)", "sample_index": int(sample_index),
            "attributions": [{**a, "role": "normal"} for a in items[:10]],
        })

    def run_faithfulness(self, sample_index: int) -> dict:
        x = self.driver.sample_at(sample_index)
        rep = evaluate_faithfulness(self.driver, x)
        return self._log("run_faithfulness", rep.to_dict())

    def run_faithfulness_profile(self, n_samples: int = 20) -> dict:
        """跨样本忠实性档案：单样本可能是运气，通过率才说明解释整体可不可信。"""
        return self._log("run_faithfulness_profile", aggregate_faithfulness(self.driver, n_samples=n_samples))

    def simulate_score(self, sample_index: int) -> dict:
        """explain→simulate→score：SHAP 声称的重要性 vs 干预实验实测敏感度，Spearman 打分。"""
        x = self.driver.sample_at(sample_index)
        result = explain_simulate_score(self.driver, x)
        result["sample_index"] = int(sample_index)
        return self._log("simulate_score", result)

    # ================= 部署期工具：漂移 / 背景更新 =================

    def register_data(self, name: str, data) -> dict:
        """注册服务端数据集（背景/新数据）。大矩阵不进对话——LLM 用名字引用。"""
        import numpy as np

        self.datasets[name] = np.asarray(data, dtype=float)
        return self._log("register_data", {"name": name, "shape": list(self.datasets[name].shape)})

    def list_datasets(self) -> dict:
        return self._log(
            "list_datasets",
            {"datasets": {k: list(v.shape) for k, v in getattr(self, "datasets", {}).items()}},
        )

    def check_drift(self, new_data=None, data_ref: str | None = None) -> dict:
        """背景参考分布 vs 新数据。漂移被检出 → SHAP 基准失效风险提示。

        data_ref 优先：引用 register_data 注册的数据集；new_data 为内联小矩阵。
        """
        import numpy as np

        if data_ref is not None:
            new = self.datasets[data_ref]
        elif new_data is not None:
            new = np.asarray(new_data, dtype=float)
        else:
            raise ValueError("需要 data_ref 或 new_data 之一")
        return self._log("check_drift", check_drift(self.driver, new))

    def update_background(self, new_data) -> dict:
        """漂移确认后：用新数据替换/扩充背景参考分布（重启解释基准）。"""
        import numpy as np

        new = np.asarray(new_data, dtype=float)
        self.driver.background = np.vstack([self.driver.background, new])[-2000:]
        return self._log(
            "update_background",
            {"n_background": len(self.driver.background), "note": "背景已更新，后续归因/反事实基于新参考分布"},
        )

    # ================= 高级分析：符号蒸馏 / 激活因果图 / 修复闭环 =================

    def symbolic_distill(self, n_samples: int = 400) -> dict:
        """符号蒸馏：把模型行为压缩成可审计的公式（PySR 可选，Lasso 基展开兜底）。"""
        return self._log("symbolic_distill", symbolic_distill(self.driver, n_samples=n_samples))

    def has_torch_internals(self) -> bool:
        return self.driver.kind in ("torch", "timeseries")

    def activation_graph(self, sample_index: int, reference_index: int) -> dict:
        """激活级因果图（circuit tracing lite）：层级+单元级修补，定位决策的关键内部组件。"""
        from ..analysis.activation_graph import build_activation_graph

        assert self.has_torch_internals(), "激活因果图当前支持 torch 驱动"
        result = build_activation_graph(self.driver, sample_index, reference_index)
        result["sample_index"], result["reference_index"] = int(sample_index), int(reference_index)
        return self._log("activation_graph", result)

    def neuron_ranking(self, sample_index: int, layer: str | None = None, top_k: int = 10) -> dict:
        """梯度式神经元排序（Captum 可选后端，未装自动回退）——哪些单元在管这个决策。"""
        from ..analysis.neuron_rank import neuron_ranking as _rank

        result = _rank(self.driver, sample_index, layer=layer, top_k=top_k)
        result["sample_index"] = int(sample_index)
        return self._log("neuron_ranking", result)

    def param_ablation(self, param_groups: list[str], metric: str = "pred_center",
                       n_samples: int = 50) -> dict:
        """可逆损毁：某区域参数发挥什么功能（备份→消融→测功能→恢复→复测）。"""
        from ..analysis.param_ablation import param_ablation as _pa

        result = _pa(self.driver, param_groups, metric=metric, n_samples=n_samples)
        return self._log("param_ablation", result)

    def discover_circuit(self, capability_samples: int = 20, tolerance: float = 0.02,
                         layer: str | None = None, mode: str = "greedy",
                         refinement_rounds: int = 3) -> dict:
        """ACDC-lite：找最小单元子电路。mode='greedy'(秒级) | 'exhaustive'(精修，分钟级)。"""
        from ..analysis.param_ablation import discover_circuit as _dc

        result = _dc(self.driver, capability_samples=capability_samples,
                     tolerance=tolerance, layer=layer, mode=mode,
                     refinement_rounds=refinement_rounds)
        return self._log("discover_circuit", result)

    def visualize(self, sample_index: int = 0, output_path: str = "mic_report.html",
                  expand: int = 4, circuit_mode: str = "greedy") -> str:
        """Neuronpedia 风格 HTML 审讯报告（自包含，浏览器打开即看）。"""
        from ..analysis.visualize import generate_html

        path = generate_html(self, sample_index=sample_index, output_path=output_path)
        self._log("visualize", {"path": path})
        return path

    def sae_features(self, top_k: int = 8, expand: int = 4, epochs: int = 300) -> dict:
        """SAE-lite：隐状态稀疏字典 + 自动标注 + 因果证伪。expand=4/8/16（更大=更细粒度）。"""
        from ..analysis.sae_lite import sae_features as _sf

        result = _sf(self.driver, top_k=top_k, expand=expand, epochs=epochs)
        return self._log("sae_features", result)

    def steering_repair(self, sample_index: int | None, feature: str, layer: str | None = None) -> dict:
        """修复闭环：steering vector 削弱对可疑特征的依赖，并用审讯工具复诊。"""
        from ..analysis.steering import steering_repair as _repair

        assert self.has_torch_internals(), "修复闭环当前支持 torch 驱动"
        result = _repair(self, feature, layer=layer)
        result["sample_index"] = sample_index
        return self._log("steering_repair", result)

    # ================= 泄漏与污染检测 =================

    def check_contamination(self, new_data=None, data_ref: str | None = None,
                            y_bg=None, y_new=None) -> dict:
        """训练背景 vs 新数据：行级重叠 + 标签口径对账（喂真实训练/测试集查污染）。"""
        import numpy as np

        from ..analysis.leakage import duplicate_overlap, label_distribution_check

        new = self.datasets[data_ref] if data_ref is not None else np.asarray(new_data, dtype=float)
        result = {"overlap": duplicate_overlap(self.driver.background, new)}
        if y_bg is not None and y_new is not None:
            result["label_check"] = label_distribution_check(y_bg, y_new)
        return self._log("check_contamination", result)

    def leak_scan(self, y) -> dict:
        """单特征泄漏扫描：背景数据 + 真实标签 → 找'单独就打满 AUC'的泄漏嫌疑特征。"""
        import numpy as np

        from ..analysis.leakage import single_feature_leak_scan

        y = np.asarray(y)
        if len(y) != len(self.driver.background):
            y = y[: len(self.driver.background)]
        return self._log("leak_scan", single_feature_leak_scan(y, self.driver.background,
                                                               self.driver.feature_names))

    # ================= 评估器审讯（BacktestDriver 专属） =================

    def has_backtest_internals(self) -> bool:
        return self.driver.kind == "backtest"

    def slippage_stress(self) -> dict:
        """滑点压力测试：PF 对成交劣化的生存曲线。PF 在几十 bps 内塌缩 = edge 依赖成交假设。"""
        assert self.has_backtest_internals(), "滑点压力测试需要 BacktestDriver"
        from ..backtest.stress import slippage_stress as _stress

        return self._log("slippage_stress", _stress(self.driver))

    def assumption_gap(self) -> dict:
        """唯一变量对照：阶梯价记账 vs 市价成交——量化回测器的成交假设虚增。"""
        assert self.has_backtest_internals(), "假设缺口需要 BacktestDriver"
        from ..backtest.stress import assumption_gap as _gap

        return self._log("assumption_gap", _gap(self.driver))

    def live_faithfulness(self, backtest: dict, live: dict) -> dict:
        """回测 vs 实盘吻合度——评估器自己的忠实性考卷（喂入两边实测指标）。"""
        from ..backtest.stress import live_faithfulness as _lf

        return self._log("live_faithfulness", _lf(backtest, live))

    # ---- Torch 专属工具（存在时自动可用） --------------------------------
    def has_activation_tools(self) -> bool:
        return self.driver.kind in ("torch", "timeseries")

    def get_activations(self, sample_index: int) -> dict:
        drv = self.driver
        assert drv.kind == "torch", "激活工具仅支持 torch 驱动"
        x = drv.sample_at(sample_index)
        acts = drv.get_activations(x)
        return self._log(
            "get_activations",
            {
                "sample_index": int(sample_index),
                "layers": {k: {"shape": list(v.shape), "mean": round(float(v.mean()), 4)} for k, v in acts.items()},
            },
        )

    def patch_activation_test(self, sample_index: int, reference_index: int, layer: str) -> dict:
        """因果中介分析：corrupt 样本 + 干净参照的层激活 → 输出恢复多少。"""
        drv = self.driver
        assert drv.kind == "torch", "激活工具仅支持 torch 驱动"
        x = drv.sample_at(sample_index)
        x_ref = drv.sample_at(reference_index)
        p_clean = drv.predict_one(x)
        p_corrupt = drv.predict_one(x_ref)
        with drv.patch_activation(layer, x):
            p_patched = drv.predict_one(x_ref)
        return self._log(
            "patch_activation_test",
            {
                "layer": layer,
                "sample_index": int(sample_index),
                "reference_index": int(reference_index),
                "p_sample": round(p_clean, 4),
                "p_reference_baseline": round(p_corrupt, 4),
                "p_reference_patched": round(p_patched, 4),
                "mediation_effect": round(p_patched - p_corrupt, 4),
                "recovery_ratio": round((p_patched - p_corrupt) / (p_clean - p_corrupt), 4)
                if abs(p_clean - p_corrupt) > 1e-6 else None,
            },
        )

    # ================= 离线审讯报告（无 LLM，确定性） ========================

    def offline_report(self, sample_index: int) -> str:
        attr = self.get_attribution(sample_index)
        cf = self.run_counterfactual(sample_index)
        faith = self.run_faithfulness(sample_index)
        graph = self.get_causal_graph(sample_index)

        d = self.driver
        sem = "正类概率" if d.prediction_semantics == "probability" else "分数"
        lines = []
        lines.append("=" * 62)
        lines.append(f"MIC 审讯报告 | 模型 {d.name} ({d.kind}) | 领域 {self.pack.domain}")
        lines.append("=" * 62)
        p = attr["prediction"]
        verdict = "阳性" if p > (0.5 if d.prediction_semantics == "probability" else (1.0 if d.kind == "backtest" else 0)) else "阴性"
        label = "PF(>1为盈利)" if d.kind == "backtest" else ("正类概率" if d.prediction_semantics == "probability" else "分数")
        lines.append(f"样本 #{sample_index}: 预测={verdict} ({label} {p:.3f})")
        lines.append("")
        lines.append("── 归因（按 |SHAP| 降序，Top 8）──")
        for a in attr["attributions"][:8]:
            flag = "  ⚠️可疑特征" if a["role"] == "spurious" else ""
            lines.append(
                f"  {a['concept']:<12} 值={a['value']:+.2f}  贡献={a['shap']:+.3f} (占比 {a['share']:.0%}){flag}"
            )
        lines.append("")
        lines.append("── 干预式因果图 ──")
        direct = sorted(graph["direct_edges"], key=lambda e: -e["effect"])[:5]
        for e in direct:
            lines.append(f"  [直接] {self.pack.concept_of(e['source'])} → 预测   {e['detail']}")
        for e in graph["interaction_edges"][:4]:
            lines.append(
                f"  [干预] {self.pack.concept_of(e['source'])} → 影响 {self.pack.concept_of(e['target'])} 的归因"
                f" (漂移 {e['effect']:.3f})"
            )
        for e in graph["correlational_edges"][:3]:
            lines.append(f"  [相关*] {e['source']} ~ {e['target']} (r={e['effect']:+.2f})  *仅观测相关")
        lines.append("")
        lines.append("── 反事实（最小翻转手术，最多 3 条路径）──")
        if cf["paths"]:
            for i, path in enumerate(cf["paths"], 1):
                changes = "；".join(
                    f"{v['concept']} {v['from']:+.2f}→{v['to']:+.2f}" for v in path["changed_features"].values()
                )
                lines.append(
                    f"  路径{i}: 把 {changes} → 预测 {path['prediction_after']:.3f}（翻转, 距离 {path['distance']:.2f}）"
                )
        else:
            lines.append("  在背景分布支持范围内未找到翻转方案（改动 ≤3 特征）")
        lines.append("")
        lines.append("── 忠实性验证（单样本）──")
        f_ = faith
        status = "通过 ✓" if f_["passed"] else "未通过 ✗"
        lines.append(
            f"  [{status}] 完备性 Δ={f_['comprehensiveness']:.3f} | 随机基线 Δ={f_['random_baseline']:.3f}"
            f" | 充分性 Δ={f_['sufficiency']:.3f} (top-k={f_['top_k']})"
        )
        for n in f_["notes"]:
            lines.append(f"    - {n}")
        lines.append("")
        profile = self.run_faithfulness_profile(n_samples=12)
        sim = self.simulate_score(sample_index)
        lines.append("── 忠实性档案（跨样本）──")
        lines.append(
            f"  通过率 {profile['pass_rate']:.0%} | 完备性均值 {profile['comprehensiveness']['mean']:.3f}"
            f"（p25-p75: {profile['comprehensiveness']['p25']:.3f}-{profile['comprehensiveness']['p75']:.3f}）"
            f" | 判定: {profile['verdict']}"
        )
        lines.append(
            f"  explain→simulate→score: Spearman={sim['spearman']} 判定={sim['verdict']}"
            f"（SHAP声称重要性 vs 干预实测敏感度）"
        )
        lines.append("")
        if self.driver.kind == "torch":
            lines.append("")
            lines.append("── 激活审讯（前向传播内部）──")
            try:
                acts = self.get_activations(sample_index)
                ref = (sample_index + 1) % len(self.driver.background)
                for layer in self.driver.list_layers()[:6]:
                    r = self.patch_activation_test(sample_index, ref, layer)
                    rec = r["recovery_ratio"]
                    lines.append(
                        f"  层[{layer}] 激活均值={acts['layers'][layer]['mean']:+.3f} "
                        f"中介恢复率={rec if rec is None else round(rec, 2)}"
                    )
                lines.append("  （恢复率=把本样本该层激活植入参照样本后输出向本样本回归的比例；1.0=该层完全承载决策）")
            except Exception as e:
                lines.append(f"  激活审讯不可用: {type(e).__name__}")

        findings = self.audit(sample_index)
        lines.append("── 审查员意见 ──")
        for f in findings["findings"]:
            lines.append(f"  [{f['severity']}] {f['message']}")
        lines.append("=" * 62)
        return "\n".join(lines)

    # ================= 审查员（确定性规则 + 概念包角色标注） ==================

    def audit(self, sample_index: int | None = None) -> dict:
        """离线审查：用干预证据扫描模型的常见病。"""
        findings: list[dict] = []

        # 1) 全局归因集中度（背景样本）
        bg = self.driver.background[:200]
        from ..analysis.attribution import shap_matrix

        sm = shap_matrix(self.driver, bg)
        mean_abs = np.abs(sm).mean(axis=0)
        share = mean_abs / (mean_abs.sum() or 1.0)
        order = np.argsort(-share)
        feats = self.driver.feature_names

        if share[order[0]] > 0.5:
            findings.append(
                {
                    "severity": "警告",
                    "code": "concentration",
                    "message": (
                        f"单特征集中风险：'{self.pack.concept_of(feats[order[0]])}' "
                        f"占总归因 {share[order[0]]:.0%}，模型行为高度依赖单一因子"
                    ),
                }
            )

        # 2) 可疑/代理特征依赖（概念包 role 标注 + 归因占比；已标注先验怀疑，门槛 3%）
        for j, f in enumerate(feats):
            entry = self.pack.ground(f)
            if entry and entry.role in {"spurious", "proxy"} and share[j] > 0.03:
                findings.append(
                    {
                        "severity": "高危" if share[j] > 0.15 else "警告",
                        "code": f"{entry.role}_dependence",
                        "message": (
                            f"模型依赖了{ '可疑' if entry.role == 'spurious' else '代理'}特征 "
                            f"'{entry.concept}'({f})：归因占比 {share[j]:.0%}。"
                            f"该特征被概念包标注为 {entry.role}，建议核查其在部署分布中是否稳定"
                        ),
                    }
                )

        # 3) 单样本忠实性
        idx = sample_index if sample_index is not None else int(np.argmin(abs(share)))
        x = self.driver.sample_at(idx)
        rep = evaluate_faithfulness(self.driver, x)
        if not rep.passed:
            for n in rep.notes:
                findings.append({"severity": "警告", "code": "faithfulness", "message": f"样本 #{idx}: {n}"})

        # 4) 跨样本忠实性档案 + explain→simulate→score
        prof = aggregate_faithfulness(self.driver, n_samples=12)
        if prof["verdict"] != "可信":
            findings.append(
                {
                    "severity": "警告",
                    "code": "faithfulness_profile",
                    "message": (
                        f"跨样本忠实性通过率仅 {prof['pass_rate']:.0%}（阈值 80%）——"
                        f"该模型的归因解释整体可信度不足，解释结论请谨慎使用"
                    ),
                }
            )
        sim = explain_simulate_score(self.driver, x)
        if sim["verdict"] in {"失真/无法判定", "部分失真"}:
            findings.append(
                {
                    "severity": "警告",
                    "code": "simulate_score",
                    "message": (
                        f"explain→simulate→score 判定「{sim['verdict']}」(Spearman={sim['spearman']})："
                        f"SHAP 声称的特征重要性与干预实验实测敏感度不一致，解释可能失真"
                    ),
                }
            )

        if not findings:
            findings.append(
                {
                    "severity": "通过",
                    "code": "none",
                    "message": "未发现明显问题：归因分散度、可疑特征依赖、忠实性检查均通过",
                }
            )

        return self._log(
            "audit",
            {
                "global_attribution_share": {
                    self.pack.concept_of(feats[j]): round(float(share[j]), 4) for j in order
                },
                "findings": findings,
            },
        )
