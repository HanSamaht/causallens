"""主线 1/2 测试：mic ask 推演引擎 + 多分类 + 概念包起草。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ================= mic ask：确定性解析 =================

@pytest.fixture(scope="module")
def geo_session():
    from mic.core import ConceptPack, MicSession
    from mic.demos import train_geo_demo
    from mic.drivers import detect_driver

    model, feats, bg, _ = train_geo_demo()
    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "geo_zh.json"))
    return MicSession(detect_driver(model, feats, bg, name="geo"), pack)


def test_ask_deterministic_scale(geo_session):
    from mic.ask import AskEngine

    ans = AskEngine(geo_session, llm=None).ask("如果降水量增加50%，最典型的样本滑坡风险怎么变")
    for kw in ["【推演】", "do(降水", "【回答】", "该系统在此干预下的行为变化", "不是对现实世界的预测"]:
        assert kw in ans, f"回答缺少 {kw}"
    # +50% 干预确实执行
    assert "+50%" in ans


def test_ask_deterministic_quantile_and_sample(geo_session):
    from mic.ask import AskEngine

    ans = AskEngine(geo_session, llm=None).ask("如果坡度升到历史90分位，最安全样本的风险怎么变")
    assert "do(坡度" in ans


def test_ask_deterministic_unparseable(geo_session):
    from mic.ask import AskEngine

    with pytest.raises(ValueError):
        AskEngine(geo_session, llm=None).ask("今天天气怎么样")


def test_ask_only_mentioned_features(geo_session):
    """纪律：解析结果只允许干预问题中明确提到的特征。"""
    from mic.ask import AskEngine

    eng = AskEngine(geo_session, llm=None)
    plan = {"assignments": [
        {"feature": "precipitation", "op": "scale", "value": 1.5},
        {"feature": "slope", "op": "scale", "value": 1.2},  # 未提及，必须被过滤
    ], "sample_selector": "median"}
    kept = eng._filter_mentioned(plan["assignments"], "如果降水量增加50%")
    assert [a["feature"] for a in kept] == ["precipitation"]


# ================= 多分类 =================

def test_multiclass_iris_full_pipeline():
    from mic.core import ConceptPack, MicSession
    from mic.demos import train_iris_demo
    from mic.drivers import detect_driver
    from mic.analysis import search_counterfactuals

    model, feats, bg, acc = train_iris_demo()
    assert acc > 0.9
    drv = detect_driver(model, feats, bg, name="iris")
    assert drv.n_classes == 3

    pack = ConceptPack.from_dict({"domain": "botany", "features": {
        "sepal_length": {"concept": "萼片长"}, "sepal_width": {"concept": "萼片宽"},
        "petal_length": {"concept": "花瓣长"}, "petal_width": {"concept": "花瓣宽"}}})
    session = MicSession(drv, pack)

    # 目标类切换：归因聚焦到该类
    session.set_target_class(2)
    assert drv.target_class == 2
    attr = session.get_attribution(0)
    assert len(attr["attributions"]) == 4
    top = attr["attributions"][0]["feature"]
    assert top in ("petal_width", "petal_length")  # 鸢尾花判别靠花瓣

    # 多分类反事实 = argmax 翻转
    idx = int(np.argmax(drv.predict(bg)))
    base_label = int(drv.predict_label(bg[idx:idx + 1])[0])
    paths = search_counterfactuals(drv, bg[idx])
    assert paths
    x2 = drv.intervene(bg[idx], {f: v[1] for f, v in paths[0].changed_features.items()})
    assert int(drv.predict_label(x2.reshape(1, -1))[0]) != base_label


def test_binary_regression_unchanged(finance_env=None):
    """回归护栏：二分类语义不受多分类改造影响。"""
    from mic.core import ConceptPack, MicSession
    from mic.demos import train_finance_demo
    from mic.drivers import detect_driver
    from mic.analysis import search_counterfactuals

    model, feats, bg, _ = train_finance_demo()
    drv = detect_driver(model, feats, bg, name="fin_guard")
    assert drv.n_classes == 2
    idx = int(np.argmax(drv.predict(bg)))
    paths = search_counterfactuals(drv, bg[idx])
    assert paths and paths[0].flipped


# ================= 概念包起草 =================

def test_draft_fallback_without_llm():
    from mic.concepts_draft import draft_concept_pack

    pack = draft_concept_pack(["f1", "f2"], llm=None, domain_hint="test")
    assert pack["features"]["f1"]["concept"] == "f1"  # 无 LLM 退化为特征名


def test_attach_concepts_written(tmp_path):
    from mic.concepts_draft import draft_concept_pack

    # 用假 LLM（返回固定 JSON）验证补漏逻辑
    class FakeLLM:
        def chat(self, messages, temperature=0.1, tools=None):
            return {"choices": [{"message": {"content":
                '{"domain":"t","name":"n","features":{"a":{"concept":"甲","direction":"?"}}}'}}]}

    pack = draft_concept_pack(["a", "b"], llm=FakeLLM(), domain_hint="t")
    assert pack["features"]["a"]["concept"] == "甲"
    assert pack["features"]["b"]["concept"] == "b"  # LLM 漏掉的自动补齐
