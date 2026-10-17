"""MIC .

    python -m mic demo-finance # training demo model
    python -m mic demo-medicine # training demo model
    python -m mic report --model ... --sample 3 --concepts concepts/finance.json
    python -m mic audit --model ...
    python -m mic chat --model ... --role auditor # requireconfiguration LLM
    python -m mic serve --model ... # MCP stdio server
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
    """Save artifacts."""
    os.makedirs("examples/artifacts", exist_ok=True)
    path_model = f"examples/artifacts/{tag}_model.pkl"
    path_bg = f"examples/artifacts/{tag}_background.csv"
    with open(path_model, "wb") as f:
        pickle.dump(model, f)
    pd.DataFrame(background, columns=features).to_csv(path_bg, index=False)
    print(f"[mic] {tag} model auc={auc:.4f}\n model={path_model}\n background={path_bg}")
    if extra:
        print(" " + extra)


def _load_session(args):
    """Load session."""
    from causallens.core import ConceptPack, MicSession
    from causallens.drivers import detect_driver

    if getattr(args, "demo", None):
        # directtraining(not pickle )
        if args.demo == "finance":
            from causallens.demos import train_finance_demo

            model, feats, bg, auc = train_finance_demo()
        elif args.demo == "medicine":
            from causallens.demos import train_medicine_demo

            model, feats, bg, auc = train_medicine_demo()
        elif args.demo == "torch":
            from causallens.demos import FinanceMLP

            model, feats, bg, auc = FinanceMLP.train()
        elif args.demo == "agent":
            from causallens.demos import AGENT_LOG_FEATURES, make_agent_logs
            from causallens.drivers.decision_log import DecisionLogDriver

            X, y = make_agent_logs()
            driver = DecisionLogDriver(X, y, AGENT_LOG_FEATURES, name="log")
            auc = driver.fidelity["action_agreement"]
        elif args.demo == "iris":
            from causallens.demos import train_iris_demo

            model, feats, bg, auc = train_iris_demo()
        elif args.demo == "credit":
            from causallens.demos import train_credit_demo

            model, feats, bg, auc = train_credit_demo()
        elif args.demo == "geo":
            from causallens.demos import train_geo_demo

            model, feats, bg, auc = train_geo_demo()
        elif args.demo == "backtest":
            from causallens.backtest import make_synthetic_4h
            from causallens.drivers.backtest import BacktestDriver

            driver = BacktestDriver(make_synthetic_4h(8000), name="bt_engine")
            auc = driver.run(fill_market=0.0)["pf"]
        elif args.demo == "timeseries":
            from causallens.demos import train_timeseries_demo
            from causallens.drivers.timeseries import TimeSeriesDriver

            model, feats, bg, r2 = train_timeseries_demo()
            auc = r2
        else:
            sys.exit(f" demo: {args.demo}")
        print(f"[mic] training {args.demo} demo (={auc:.3f})", file=sys.stderr)
        if args.demo in ("backtest", "agent"):
            pass # driver inon
        elif args.demo == "timeseries":
            from causallens.drivers.timeseries import TimeSeriesDriver

            driver = TimeSeriesDriver(model, feats, bg, name="ts_returns_demo")
        else:
            driver = detect_driver(model, feats, bg, name=f"{args.demo}_demo")
    else:
        if str(getattr(args, "model", "")).lower().endswith(".onnx"):
            pass # ONNX iscompute, safe
        else:
            print("[mic] ⚠️ pickle modelloadingexecute, ; model ONNX", file=sys.stderr)
        if str(getattr(args, "model", "")).lower().endswith(".onnx"):
            from causallens.drivers import detect_driver as _dd

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
        from causallens.ledger import Ledger

        session.attach_ledger(Ledger(args.ledger), args.model_version)
    return session


def cmd_demo(args):
    """Cmd demo."""
    if args.demo == "finance":
        from causallens.demos import train_finance_demo

        model, feats, bg, auc = train_finance_demo()
        _save_artifacts("finance", model, feats, bg, auc)
    elif args.demo == "medicine":
        from causallens.demos import train_medicine_demo

        model, feats, bg, auc = train_medicine_demo()
        _save_artifacts("medicine", model, feats, bg, auc)
    elif args.demo == "geo":
        from causallens.demos import train_geo_demo

        model, feats, bg, auc = train_geo_demo()
        _save_artifacts("geo", model, feats, bg, auc)
    elif args.demo == "torch":
        import torch

        from causallens.demos import FinanceMLP

        net, feats, bg, auc = FinanceMLP.train()
        os.makedirs("examples/artifacts", exist_ok=True)
        path = "examples/artifacts/torch_model.pt"
        torch.save(net.state_dict(), path)
        pd.DataFrame(bg, columns=feats).to_csv("examples/artifacts/torch_background.csv", index=False)
        print(f"[mic] torch demo auc={auc:.4f}\n state_dict={path}")


def cmd_report(args):
    """Cmd report."""
    session = _load_session(args)
    print(session.offline_report(args.sample))


def cmd_audit(args):
    """Cmd audit."""
    session = _load_session(args)
    result = session.audit(args.sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_chat(args):
    """Cmd chat."""
    from causallens.agents import Interrogator, LLMClient

    try:
        llm = LLMClient()
    except Exception as e:
        print(f"[mic] {e}\n[mic] asmode(configuration ZHIPUAI_API_KEY / OPENAI_API_KEY LLM )\n")
        llm = None

    session = _load_session(args)
    interrogator = Interrogator(session, llm=llm, role=args.role)
    print(f"[mic] model={session.driver.name} role={args.role}")
    print("[mic] inputprobleminterrogation; exit ")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        print(f"\n{args.role}> {interrogator.ask(q)}")


def cmd_attach(args):
    """model: - LLMconcept pack - interrogationreport. """
    import json

    from causallens.concepts_draft import draft_concept_pack
    from causallens.core import ConceptPack, MicSession
    from causallens.agents import LLMClient

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
    print(f"[mic] concept pack - {out}(engine: {'LLM' if llm else 'fallback'})")
    print(session.offline_report(args.sample))


def _load_session_with(args, pack):
    """Load session with."""
    from causallens.core import MicSession

    ns = argparse.Namespace(**vars(args))
    s = _load_session(ns)
    return s


def cmd_visualize(args):
    """Cmd visualize."""
    session = _load_session(args)
    path = session.visualize(args.sample, args.out)
    print(f"[mic] HTML interrogationreportgenerate - {path}")
    import webbrowser
    webbrowser.open(f"file:///{path.replace(chr(92), '/')}")


def cmd_ask(args):
    """Cmd ask."""
    from causallens.ask import AskEngine
    from causallens.agents import LLMClient

    session = _load_session(args)
    llm = None
    try:
        llm = LLMClient()
    except Exception:
        pass # key
    print(AskEngine(session, llm=llm).ask(args.question))


def cmd_graph_export(args):
    """Cmd graph export."""
    from causallens.core.graph_review import export_graph

    session = _load_session(args)
    print(json.dumps(export_graph(session, args.out, args.sample), ensure_ascii=False, indent=2))


def cmd_graph_apply(args):
    """Cmd graph apply."""
    from causallens.core.graph_review import apply_corrections

    session = _load_session(args)
    result = apply_corrections(session, args.corrections)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def cmd_history(args):
    """Cmd history."""
    from causallens.ledger import Ledger

    led = Ledger(args.ledger)
    cases = led.history(args.model)
    if not cases:
        print("[mic] as")
        return
    for c in cases:
        print(f"#{c['id']} {c['model_name']} @ {c['model_version']} "
              f"evidence {c['n_evidence']} | hypothesis {c['n_hypotheses']} | {c['n_corrections']}"
              f" ({c['created_at']})")
        if args.detail:
            s = led.case_summary(c["id"])
            print(f" : {'✓ ' if s['chain']['intact'] else '✗ in #' + str(s['chain'].get('broken_at_id')) + ' '}")
            for h in s["hypotheses"]:
                mark = "-" if h["valid_to"] else ""
                print(f" [{h['verdict']:<11}] {h['statement'][:44]} posterior={h['posterior']} {mark}")
            for r in s["corrections"]:
                print(f" [:{r['operation']:<9}] {r['edge']}")


def cmd_llm_check(args):
    """: configurationidentify - - tool. GLM this. """
    from causallens.agents.llm import LLMClient, LLMConfigError
    from causallens.agents.providers import env_status

    print("[mic] LLM configuration: ")
    for row in env_status():
        mark = "✓" if row["configured"] else " "
        print(f" [{mark}] {row['provider']:<12} {row['default_model']:<28} key: {row['key_env']}")
    print()

    try:
        llm = LLMClient()
    except LLMConfigError as e:
        print(f"[mic] ✗ {e}")
        sys.exit(1)

    src = llm.provider.name if llm.provider else "custom"
    print(f"[mic] when: {src} | model={llm.model} | base={llm.base_url}")

    # ---- 1: ----
    try:
        resp = llm.chat([{"role": "user", "content": "only: normal"}], temperature=0.0)
        content = resp["choices"][0]["message"].get("content") or "()"
        print(f"[mic] ✓ 1 : {content[:40]}")
    except Exception as e:
        print(f"[mic] ✗ 1 : {e}")
        sys.exit(1)

    # ---- 2: tool ----
    echo_tool = [{
        "type": "function",
        "function": {
            "name": "mic_echo",
            "description": "testtool: return text parameter",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }]
    try:
        messages = [
            {"role": "system", "content": "musttool mic_echo , notdirect. "},
            {"role": "user", "content": "tool, text parameter'test'"},
        ]
        resp = llm.chat(messages, tools=echo_tool, temperature=0.0)
        msg = resp["choices"][0]["message"]
        calls = LLMClient.normalize_tool_calls(msg)
        if not calls:
            print(f"[mic] ✗ 2 : modeltool(themodelnotsupport function calling). : {(msg.get('content') or '')[:60]}")
            sys.exit(1)
        print(f"[mic] ✓ 2 tool: {calls[0]['name']}({calls[0]['arguments']})")
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls")})
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "test-OK"})
        resp2 = llm.chat(messages, tools=echo_tool, temperature=0.0)
        final = resp2["choices"][0]["message"].get("content") or "()"
        print(f"[mic] ✓ 3 result: {final[:60]}")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[mic] ✗ 2/3 : {e}")
        sys.exit(1)

    print("\n[mic] allpassed —— can: python -m mic chat --demo finance --concepts concepts/finance.json --role auditor")


def cmd_drift(args):
    """Cmd drift."""
    session = _load_session(args)
    new = pd.read_csv(args.new).values
    print(json.dumps(session.check_drift(new), ensure_ascii=False, indent=2))


def cmd_diff(args):
    """versioncontrastinterrogation: v1 vs v2 behaviordifference. """
    from causallens.analysis.model_diff import diff_sessions

    s_old = _load_session(args)
    # v2: --demo-b( --seed-b retrainnotversion)or --model-b
    ns = argparse.Namespace()
    if getattr(args, "demo_b", None):
        seed_b = getattr(args, "seed_b", None)
        if args.demo_b == "finance":
            import causallens.demos as demos

            model, feats, bg, _ = demos.train_finance_demo() if seed_b is None else _finance_with_seed(seed_b)
        elif args.demo_b == "medicine":
            from causallens.demos import train_medicine_demo

            model, feats, bg, _ = train_medicine_demo()
        else:
            sys.exit(f"notsupport --demo-b: {args.demo_b}")
        from causallens.core import MicSession
        from causallens.drivers import detect_driver

        s_new = MicSession(detect_driver(model, feats, bg, name="v2"), s_old.pack)
    elif getattr(args, "model_b", None):
        ns.demo, ns.model, ns.background = None, args.model_b, args.background_b
        ns.concepts = args.concepts
        s_new = _load_session(ns)
    else:
        sys.exit("require --demo-b or --model-b version")
    print(json.dumps(diff_sessions(s_old.driver, s_new.driver, s_old.pack), ensure_ascii=False, indent=2))


def _finance_with_seed(seed: int):
    """Finance with seed."""
    import causallens.demos as demos
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = demos.make_finance_data(seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)
    return model, demos.FINANCE_FEATURES, X_te, float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))


def cmd_probe(args):
    """Cmd probe."""
    from causallens.probes import run_probes

    results = run_probes()
    all_ok = True
    for name, r in results.items():
        mark = "✓" if r["passed"] else "✗"
        print(f"[{mark}] {name:<10} {r['detail']}")
        all_ok &= r["passed"]
    sys.exit(0 if all_ok else 1)


def cmd_serve(args):
    """Cmd serve."""
    session = _load_session(args)
    if getattr(args, "http", False):
        from causallens.server.http_mcp import serve_http

        serve_http(session, port=args.port)
        return
    from causallens.server import serve_stdio

    print(f"[mic] MCP stdio server start: {session.driver.name}", file=sys.stderr)
    serve_stdio(session)


def main(argv=None):
    """Main."""
    p = argparse.ArgumentParser(prog="mic", description="MIC model —— model")
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="trainingmodel")
    demo.add_argument("demo", choices=["finance", "medicine", "torch", "timeseries", "geo", "backtest", "credit", "iris", "agent"])
    demo.set_defaults(func=cmd_demo)

    def add_model_args(sp, need_model=True):
        """Add model args."""
        g = sp.add_mutually_exclusive_group(required=need_model)
        g.add_argument("--model", help="pickle model(sklearn )")
        g.add_argument("--demo", choices=["finance", "medicine", "torch", "timeseries", "geo", "backtest", "credit", "iris", "agent"], help=" demo model")
        sp.add_argument("--background", help="background CSV(--model mode)")
        sp.add_argument("--concepts", help="concept pack JSON ")
        sp.add_argument("--ledger", help="evidence ledger SQLite ()", default=None)
        sp.add_argument("--model-version", dest="model_version", default="v1", help="case archiveversion")

    rep = sub.add_parser("report", help="interrogationreport( LLM)")
    add_model_args(rep)
    rep.add_argument("--sample", type=int, default=0)
    rep.set_defaults(func=cmd_report)

    aud = sub.add_parser("audit", help="model")
    add_model_args(aud)
    aud.add_argument("--sample", type=int, default=None)
    aud.set_defaults(func=cmd_audit)

    chat = sub.add_parser("chat", help="andmodel(LLM role)")
    add_model_args(chat)
    chat.add_argument("--role", choices=["translator", "auditor", "fixer"], default="translator")
    chat.set_defaults(func=cmd_chat)

    drf = sub.add_parser("drift", help="distributiondrift(backgrounddistribution vs )")
    add_model_args(drf)
    drf.add_argument("--new", required=True, help=" CSV ")
    drf.set_defaults(func=cmd_drift)

    dfp = sub.add_parser("diff", help="versioncontrastinterrogation: v1 vs v2 behaviordifference")
    dfp.add_argument("--model", help="v1 pickle/onnx model")
    dfp.add_argument("--demo", choices=["finance", "medicine", "torch", "timeseries", "geo", "backtest", "credit", "iris", "agent"])
    dfp.add_argument("--background", help="v1 background CSV")
    dfp.add_argument("--concepts", help="concept pack JSON")
    dfp.add_argument("--model-b", help="v2 pickle/onnx model")
    dfp.add_argument("--background-b", help="v2 background CSV")
    dfp.add_argument("--demo-b", choices=["finance", "medicine"])
    dfp.add_argument("--seed-b", type=int, help="--demo-b finance notretrain v2")
    dfp.set_defaults(func=cmd_diff)

    prb = sub.add_parser("probe", help="runtest(framework)")
    prb.set_defaults(func=cmd_probe)

    hist = sub.add_parser("history", help="case archive: modelinterrogationhistoryand")
    hist.add_argument("--ledger", default="mic_ledger.db")
    hist.add_argument("--model", default=None, help="model")
    hist.add_argument("--detail", action="store_true", help="hypothesisand")
    hist.set_defaults(func=cmd_history)

    gex = sub.add_parser("graph-export", help="causal graphas JSON()")
    add_model_args(gex)
    gex.add_argument("--out", default="causal_graph.json")
    gex.add_argument("--sample", type=int, default=None)
    gex.set_defaults(func=cmd_graph_export)

    gap = sub.add_parser("graph-apply", help=": intervention experimentvalidate")
    add_model_args(gap)
    gap.add_argument("--corrections", required=True, help="causal graph JSON ")
    gap.set_defaults(func=cmd_graph_apply)

    att = sub.add_parser("attach", help=": model+background - concept pack - interrogationreport")
    add_model_args(att)
    att.add_argument("--domain", default="", help="domain( ''), LLM concept pack")
    att.add_argument("--sample", type=int, default=0)
    att.set_defaults(func=cmd_attach)

    viz = sub.add_parser("visualize", help="generate Neuronpedia HTML interrogationreport()")
    add_model_args(viz)
    viz.add_argument("--sample", type=int, default=0)
    viz.add_argument("--out", default="mic_report.html")
    viz.set_defaults(func=cmd_visualize)

    ask = sub.add_parser("ask", help="causal reasoning: what-if problem('ifXincreases50%, Y')")
    add_model_args(ask)
    ask.add_argument("question", help="what-if problem, 'ifincreases50%, mostsample'")
    ask.set_defaults(func=cmd_ask)

    hero = sub.add_parser("hero", help="30: interrogationbacktestengine(/API)")
    hero.set_defaults(func=lambda a: __import__("causallens.cli_hero", fromlist=["cmd_hero"]).cmd_hero(a))

    chk = sub.add_parser("llm-check", help="LLM configurationandtool( key , chat )")
    chk.set_defaults(func=cmd_llm_check, model=None, background=None, concepts=None)

    serve = sub.add_parser("serve", help="MCP server(default stdio; --http Streamable HTTP)")
    add_model_args(serve)
    serve.add_argument("--http", action="store_true", help=" HTTP ( pip install mcp)")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    if getattr(args, "model", None) and not getattr(args, "background", None):
        sys.exit("--model moderequire --background CSV")
    args.func(args)


if __name__ == "__main__":
    main()
