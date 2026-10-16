"""MIC 命令行入口。

    python -m mic demo-finance            # 训练金融 demo 模型并存盘
    python -m mic demo-medicine           # 训练医学 demo 模型并存盘
    python -m mic report   --model ... --sample 3 --concepts concepts/finance_zh.json
    python -m mic audit    --model ...
    python -m mic chat     --model ... --role auditor          # 需要配置 LLM
    python -m mic serve    --model ...                        # MCP stdio server
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd


def _save_artifacts(tag: str, model, features, background: np.ndarray, auc: float, extra=None):
    os.makedirs("examples/artifacts", exist_ok=True)
    path_model = f"examples/artifacts/{tag}_model.pkl"
    path_bg = f"examples/artifacts/{tag}_background.csv"
    with open(path_model, "wb") as f:
        pickle.dump(model, f)
    pd.DataFrame(background, columns=features).to_csv(path_bg, index=False)
    print(f"[mic] {tag} 模型已存盘  auc={auc:.4f}\n  model={path_model}\n  background={path_bg}")
    if extra:
        print("  " + extra)


def _load_session(args):
    from mic.core import ConceptPack, MicSession
    from mic.drivers import detect_driver

    if getattr(args, "demo", None):
        # 直接现场训练（不依赖 pickle 文件）
        if args.demo == "finance":
            from mic.demos import train_finance_demo

            model, feats, bg, auc = train_finance_demo()
        elif args.demo == "medicine":
            from mic.demos import train_medicine_demo

            model, feats, bg, auc = train_medicine_demo()
        elif args.demo == "torch":
            from mic.demos import FinanceMLP

            model, feats, bg, auc = FinanceMLP.train()
        elif args.demo == "agent":
            from mic.demos import AGENT_LOG_FEATURES, make_agent_logs
            from mic.drivers.decision_log import DecisionLogDriver

            X, y = make_agent_logs()
            driver = DecisionLogDriver(X, y, AGENT_LOG_FEATURES, name="审批智能体日志")
            auc = driver.fidelity["action_agreement"]
        elif args.demo == "iris":
            from mic.demos import train_iris_demo

            model, feats, bg, auc = train_iris_demo()
        elif args.demo == "credit":
            from mic.demos import train_credit_demo

            model, feats, bg, auc = train_credit_demo()
        elif args.demo == "geo":
            from mic.demos import train_geo_demo

            model, feats, bg, auc = train_geo_demo()
        elif args.demo == "backtest":
            from mic.backtest import make_synthetic_4h
            from mic.drivers.backtest import BacktestDriver

            driver = BacktestDriver(make_synthetic_4h(8000), name="bt_engine")
            auc = driver.run(fill_market=0.0)["pf"]
        elif args.demo == "timeseries":
            from mic.demos import train_timeseries_demo
            from mic.drivers.timeseries import TimeSeriesDriver

            model, feats, bg, r2 = train_timeseries_demo()
            auc = r2
        else:
            sys.exit(f"未知 demo: {args.demo}")
        print(f"[mic] 现场训练 {args.demo} demo (指标={auc:.3f})", file=sys.stderr)
        if args.demo in ("backtest", "agent"):
            pass  # driver 已在上方分支构造
        elif args.demo == "timeseries":
            from mic.drivers.timeseries import TimeSeriesDriver

            driver = TimeSeriesDriver(model, feats, bg, name="ts_returns_demo")
        else:
            driver = detect_driver(model, feats, bg, name=f"{args.demo}_demo")
    else:
        if str(getattr(args, "model", "")).lower().endswith(".onnx"):
            pass  # ONNX 是纯计算图，安全
        else:
            print("[mic] ⚠️ pickle 模型加载会执行任意代码，仅用于可信来源；第三方模型请转 ONNX", file=sys.stderr)
        if str(getattr(args, "model", "")).lower().endswith(".onnx"):
            from mic.drivers import detect_driver as _dd

            bg_df = pd.read_csv(args.background)
            feats = list(bg_df.columns)
            driver = _dd(None, feats, bg_df.values, name=os.path.basename(args.model), model_path=args.model)
            pack = ConceptPack.load(args.concepts) if getattr(args, "concepts", None) else None
            return MicSession(driver, pack)
        with open(args.model, "rb") as f:
            model = pickle.load(f)
        bg_df = pd.read_csv(args.background)
        feats = list(bg_df.columns)
        driver = detect_driver(model, feats, bg_df.values, name=os.path.basename(args.model))

    pack = ConceptPack.load(args.concepts) if getattr(args, "concepts", None) else None
    session = MicSession(driver, pack)
    if getattr(args, "ledger", None):
        from mic.ledger import Ledger

        session.attach_ledger(Ledger(args.ledger), args.model_version)
    return session


def cmd_demo(args):
    if args.demo == "finance":
        from mic.demos import train_finance_demo

        model, feats, bg, auc = train_finance_demo()
        _save_artifacts("finance", model, feats, bg, auc)
    elif args.demo == "medicine":
        from mic.demos import train_medicine_demo

        model, feats, bg, auc = train_medicine_demo()
        _save_artifacts("medicine", model, feats, bg, auc)
    elif args.demo == "geo":
        from mic.demos import train_geo_demo

        model, feats, bg, auc = train_geo_demo()
        _save_artifacts("geo", model, feats, bg, auc)
    elif args.demo == "torch":
        import torch

        from mic.demos import FinanceMLP

        net, feats, bg, auc = FinanceMLP.train()
        os.makedirs("examples/artifacts", exist_ok=True)
        path = "examples/artifacts/torch_model.pt"
        torch.save(net.state_dict(), path)
        pd.DataFrame(bg, columns=feats).to_csv("examples/artifacts/torch_background.csv", index=False)
        print(f"[mic] torch demo 已存盘 auc={auc:.4f}\n  state_dict={path}")


def cmd_report(args):
    session = _load_session(args)
    print(session.offline_report(args.sample))


def cmd_audit(args):
    session = _load_session(args)
    result = session.audit(args.sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_chat(args):
    from mic.agents import Interrogator, LLMClient

    try:
        llm = LLMClient()
    except Exception as e:
        print(f"[mic] {e}\n[mic] 退化为离线模式（配置 ZHIPUAI_API_KEY / OPENAI_API_KEY 启用 LLM 对话）\n")
        llm = None

    session = _load_session(args)
    interrogator = Interrogator(session, llm=llm, role=args.role)
    print(f"[mic] 对话已就绪  模型={session.driver.name}  角色={args.role}")
    print("[mic] 输入问题开始审讯；exit 退出")
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        print(f"\n{args.role}> {interrogator.ask(q)}")


def cmd_attach(args):
    """一键让模型开口：接入 → LLM起草概念包 → 完整审讯报告。"""
    import json

    from mic.concepts_draft import draft_concept_pack
    from mic.core import ConceptPack, MicSession
    from mic.agents import LLMClient

    session = _load_session_with(args, ConceptPack())
    llm = None
    try:
        llm = LLMClient()
    except Exception:
        pass
    pack_data = draft_concept_pack(session.driver.feature_names, llm=llm, domain_hint=args.domain or "")
    pack = ConceptPack.from_dict(pack_data)
    session.pack = pack
    import os

    os.makedirs("concepts", exist_ok=True)
    out = f"concepts/auto_{session.driver.name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pack_data, f, ensure_ascii=False, indent=2)
    print(f"[mic] 概念包已起草 → {out}（引擎: {'LLM' if llm else '通用回退'}）")
    print(session.offline_report(args.sample))


def _load_session_with(args, pack):
    from mic.core import MicSession

    ns = argparse.Namespace(**vars(args))
    s = _load_session(ns)
    return s


def cmd_visualize(args):
    session = _load_session(args)
    path = session.visualize(args.sample, args.out)
    print(f"[mic] HTML 审讯报告已生成 → {path}")
    import webbrowser
    webbrowser.open(f"file:///{path.replace(chr(92), '/')}")


def cmd_ask(args):
    from mic.ask import AskEngine
    from mic.agents import LLMClient

    session = _load_session(args)
    llm = None
    try:
        llm = LLMClient()
    except Exception:
        pass  # 无 key 走确定性解析
    print(AskEngine(session, llm=llm).ask(args.question))


def cmd_graph_export(args):
    from mic.core.graph_review import export_graph

    session = _load_session(args)
    print(json.dumps(export_graph(session, args.out, args.sample), ensure_ascii=False, indent=2))


def cmd_graph_apply(args):
    from mic.core.graph_review import apply_corrections

    session = _load_session(args)
    result = apply_corrections(session, args.corrections)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def cmd_history(args):
    from mic.ledger import Ledger

    led = Ledger(args.ledger)
    cases = led.history(args.model)
    if not cases:
        print("[mic] 档案库为空")
        return
    for c in cases:
        print(f"案件#{c['id']}  {c['model_name']} @ {c['model_version']}  "
              f"证据 {c['n_evidence']} 条 | 假设 {c['n_hypotheses']} | 修正 {c['n_corrections']}"
              f"  ({c['created_at']})")
        if args.detail:
            s = led.case_summary(c["id"])
            print(f"  链完整性: {'✓ 完好' if s['chain']['intact'] else '✗ 在 #' + str(s['chain'].get('broken_at_id')) + ' 断裂'}")
            for h in s["hypotheses"]:
                mark = "→已失效" if h["valid_to"] else ""
                print(f"  [{h['verdict']:<11}] {h['statement'][:44]} 后验={h['posterior']} {mark}")
            for r in s["corrections"]:
                print(f"  [修正:{r['operation']:<9}] {r['edge']}")


def cmd_llm_check(args):
    """三步自检：配置识别 → 普通对话 → 工具调用回路。GLM 实测前先跑这个。"""
    from mic.agents.llm import LLMClient, LLMConfigError
    from mic.agents.providers import env_status

    print("[mic] LLM 供应商配置状态：")
    for row in env_status():
        mark = "✓" if row["configured"] else " "
        print(f"  [{mark}] {row['provider']:<12} {row['default_model']:<28} key: {row['key_env']}")
    print()

    try:
        llm = LLMClient()
    except LLMConfigError as e:
        print(f"[mic] ✗ {e}")
        sys.exit(1)

    src = llm.provider.name if llm.provider else "custom"
    print(f"[mic] 当前使用: {src} | model={llm.model} | base={llm.base_url}")

    # ---- 步骤 1：普通对话 ----
    try:
        resp = llm.chat([{"role": "user", "content": "只回复两个字：正常"}], temperature=0.0)
        content = resp["choices"][0]["message"].get("content") or "(空)"
        print(f"[mic] ✓ 步骤1 普通对话: {content[:40]}")
    except Exception as e:
        print(f"[mic] ✗ 步骤1 失败: {e}")
        sys.exit(1)

    # ---- 步骤 2：工具调用回路 ----
    echo_tool = [{
        "type": "function",
        "function": {
            "name": "mic_echo",
            "description": "回声测试工具：原样返回 text 参数",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }]
    try:
        messages = [
            {"role": "system", "content": "你必须调用工具 mic_echo 完成任务，不要直接回答。"},
            {"role": "user", "content": "请调用工具，text 参数填「麦克风测试」"},
        ]
        resp = llm.chat(messages, tools=echo_tool, temperature=0.0)
        msg = resp["choices"][0]["message"]
        calls = LLMClient.normalize_tool_calls(msg)
        if not calls:
            print(f"[mic] ✗ 步骤2 失败: 模型未发起工具调用（可能该模型不支持 function calling）。回复: {(msg.get('content') or '')[:60]}")
            sys.exit(1)
        print(f"[mic] ✓ 步骤2 工具调用: {calls[0]['name']}({calls[0]['arguments']})")
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls")})
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "麦克风测试-OK"})
        resp2 = llm.chat(messages, tools=echo_tool, temperature=0.0)
        final = resp2["choices"][0]["message"].get("content") or "(空)"
        print(f"[mic] ✓ 步骤3 结果回传: {final[:60]}")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[mic] ✗ 步骤2/3 失败: {e}")
        sys.exit(1)

    print("\n[mic] 全部通过 —— 可以开始对话：python -m mic chat --demo finance --concepts concepts/finance_zh.json --role auditor")


def cmd_drift(args):
    session = _load_session(args)
    new = pd.read_csv(args.new).values
    print(json.dumps(session.check_drift(new), ensure_ascii=False, indent=2))


def cmd_diff(args):
    """版本对比审讯：v1 vs v2 的行为差异画像。"""
    from mic.analysis.model_diff import diff_sessions

    s_old = _load_session(args)
    # 构造 v2：--demo-b（可带 --seed-b 重训出不同版本）或 --model-b
    ns = argparse.Namespace()
    if getattr(args, "demo_b", None):
        seed_b = getattr(args, "seed_b", None)
        if args.demo_b == "finance":
            import mic.demos as demos

            model, feats, bg, _ = demos.train_finance_demo() if seed_b is None else _finance_with_seed(seed_b)
        elif args.demo_b == "medicine":
            from mic.demos import train_medicine_demo

            model, feats, bg, _ = train_medicine_demo()
        else:
            sys.exit(f"不支持的 --demo-b: {args.demo_b}")
        from mic.core import MicSession
        from mic.drivers import detect_driver

        s_new = MicSession(detect_driver(model, feats, bg, name="v2"), s_old.pack)
    elif getattr(args, "model_b", None):
        ns.demo, ns.model, ns.background = None, args.model_b, args.background_b
        ns.concepts = args.concepts
        s_new = _load_session(ns)
    else:
        sys.exit("需要 --demo-b 或 --model-b 指定第二版本")
    print(json.dumps(diff_sessions(s_old.driver, s_new.driver, s_old.pack), ensure_ascii=False, indent=2))


def _finance_with_seed(seed: int):
    import mic.demos as demos
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = demos.make_finance_data(seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)
    return model, demos.FINANCE_FEATURES, X_te, float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))


def cmd_probe(args):
    from mic.probes import run_probes

    results = run_probes()
    all_ok = True
    for name, r in results.items():
        mark = "✓" if r["passed"] else "✗"
        print(f"[{mark}] {name:<10} {r['detail']}")
        all_ok &= r["passed"]
    sys.exit(0 if all_ok else 1)


def cmd_serve(args):
    session = _load_session(args)
    if getattr(args, "http", False):
        from mic.server.http_mcp import serve_http

        serve_http(session, port=args.port)
        return
    from mic.server import serve_stdio

    print(f"[mic] MCP stdio server 已启动: {session.driver.name}", file=sys.stderr)
    serve_stdio(session)


def main(argv=None):
    p = argparse.ArgumentParser(prog="mic", description="MIC 模型麦克风 —— 让黑箱模型开口说话")
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="训练演示模型并存盘")
    demo.add_argument("demo", choices=["finance", "medicine", "torch", "timeseries", "geo", "backtest", "credit", "iris", "agent"])
    demo.set_defaults(func=cmd_demo)

    def add_model_args(sp, need_model=True):
        g = sp.add_mutually_exclusive_group(required=need_model)
        g.add_argument("--model", help="pickle 模型路径（sklearn 系）")
        g.add_argument("--demo", choices=["finance", "medicine", "torch", "timeseries", "geo", "backtest", "credit", "iris", "agent"], help="用内置 demo 模型")
        sp.add_argument("--background", help="背景数据 CSV（--model 模式必填）")
        sp.add_argument("--concepts", help="概念包 JSON 路径")
        sp.add_argument("--ledger", help="证据账本 SQLite 路径（启用持久卷宗）", default=None)
        sp.add_argument("--model-version", dest="model_version", default="v1", help="案件档案版本号")

    rep = sub.add_parser("report", help="离线审讯报告（无需 LLM）")
    add_model_args(rep)
    rep.add_argument("--sample", type=int, default=0)
    rep.set_defaults(func=cmd_report)

    aud = sub.add_parser("audit", help="确定性模型审查")
    add_model_args(aud)
    aud.add_argument("--sample", type=int, default=None)
    aud.set_defaults(func=cmd_audit)

    chat = sub.add_parser("chat", help="与模型对话（LLM 三角色）")
    add_model_args(chat)
    chat.add_argument("--role", choices=["translator", "auditor", "fixer"], default="translator")
    chat.set_defaults(func=cmd_chat)

    drf = sub.add_parser("drift", help="分布漂移检测（背景参考分布 vs 新数据）")
    add_model_args(drf)
    drf.add_argument("--new", required=True, help="新数据 CSV 路径")
    drf.set_defaults(func=cmd_drift)

    dfp = sub.add_parser("diff", help="版本对比审讯：v1 vs v2 行为差异")
    dfp.add_argument("--model", help="v1 pickle/onnx 模型")
    dfp.add_argument("--demo", choices=["finance", "medicine", "torch", "timeseries", "geo", "backtest", "credit", "iris", "agent"])
    dfp.add_argument("--background", help="v1 背景数据 CSV")
    dfp.add_argument("--concepts", help="概念包 JSON")
    dfp.add_argument("--model-b", help="v2 pickle/onnx 模型")
    dfp.add_argument("--background-b", help="v2 背景数据 CSV")
    dfp.add_argument("--demo-b", choices=["finance", "medicine"])
    dfp.add_argument("--seed-b", type=int, help="--demo-b finance 时用不同种子重训出 v2")
    dfp.set_defaults(func=cmd_diff)

    prb = sub.add_parser("probe", help="运行合成探针回归测试（框架自身正确性）")
    prb.set_defaults(func=cmd_probe)

    hist = sub.add_parser("history", help="案件档案库：查看模型审讯历史与卷宗")
    hist.add_argument("--ledger", default="mic_ledger.db")
    hist.add_argument("--model", default=None, help="按模型名过滤")
    hist.add_argument("--detail", action="store_true", help="展开假设与修正明细")
    hist.set_defaults(func=cmd_history)

    gex = sub.add_parser("graph-export", help="导出因果图为可编辑 JSON（人工审查入口）")
    add_model_args(gex)
    gex.add_argument("--out", default="causal_graph.json")
    gex.add_argument("--sample", type=int, default=None)
    gex.set_defaults(func=cmd_graph_export)

    gap = sub.add_parser("graph-apply", help="回灌人工修正：新增边自动过干预实验验证")
    add_model_args(gap)
    gap.add_argument("--corrections", required=True, help="编辑后的因果图 JSON 路径")
    gap.set_defaults(func=cmd_graph_apply)

    att = sub.add_parser("attach", help="一键开口：模型+背景数据 → 自动概念包 → 审讯报告")
    add_model_args(att)
    att.add_argument("--domain", default="", help="领域提示词（如 '信用风控'），供 LLM 起草概念包")
    att.add_argument("--sample", type=int, default=0)
    att.set_defaults(func=cmd_attach)

    viz = sub.add_parser("visualize", help="生成 Neuronpedia 风格 HTML 审讯报告（浏览器打开）")
    add_model_args(viz)
    viz.add_argument("--sample", type=int, default=0)
    viz.add_argument("--out", default="mic_report.html")
    viz.set_defaults(func=cmd_visualize)

    ask = sub.add_parser("ask", help="因果推演：问一个 what-if 问题（『如果X增加50%，Y会怎样』）")
    add_model_args(ask)
    ask.add_argument("question", help="what-if 问题，如『如果降雨量增加50%，最容易滑坡的样本风险怎么变』")
    ask.set_defaults(func=cmd_ask)

    hero = sub.add_parser("hero", help="30秒英雄演示：审讯一个回测引擎（无需数据/API）")
    hero.set_defaults(func=lambda a: __import__("mic.cli_hero", fromlist=["cmd_hero"]).cmd_hero(a))

    chk = sub.add_parser("llm-check", help="LLM 配置与工具调用自检（配好 key 后、chat 前先跑）")
    chk.set_defaults(func=cmd_llm_check, model=None, background=None, concepts=None)

    serve = sub.add_parser("serve", help="MCP server（默认 stdio；--http 走 Streamable HTTP）")
    add_model_args(serve)
    serve.add_argument("--http", action="store_true", help="启用 HTTP 传输（需 pip install mcp）")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    if getattr(args, "model", None) and not getattr(args, "background", None):
        sys.exit("--model 模式需要同时指定 --background CSV")
    args.func(args)


if __name__ == "__main__":
    main()
