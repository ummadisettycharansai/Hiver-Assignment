import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from hiver_agent.data.cleaner import clean_tweet_text, clean_raw_dataframe
from hiver_agent.data.conversations import reconstruct_conversations
from hiver_agent.intents.taxonomy import IntentTaxonomy
from hiver_agent.intents.discovery import get_default_intent_taxonomy
from hiver_agent.intents.classifier import MajorityBaselineClassifier, TFIDFBaselineClassifier
from hiver_agent.generation.validation import validate_generated_reply
from hiver_agent.decision.policy import DecisionPolicy
from hiver_agent.evaluation.metrics import compute_decision_metrics
from hiver_agent.utils.cache import SimpleCache
from scripts.create_golden_set import assign_rule_guided_gold_intent

def test_assign_rule_guided_gold_intent_compatibility():
    intent, decision, difficulty = assign_rule_guided_gold_intent("My package is delayed again")
    assert intent == "order_delivery_delay"
    assert decision == "AUTO_HANDLE"
    assert difficulty in {"easy", "medium", "hard"}

def test_clean_tweet_text():
    raw = "  Hello   world!  @AmazonHelp   \n  "
    cleaned = clean_tweet_text(raw)
    assert cleaned == "Hello world! @AmazonHelp"

def test_clean_raw_dataframe():
    df_raw = pd.DataFrame([
        {"tweet_id": "1", "text": "   Need help   "},
        {"tweet_id": "2", "text": None},
        {"tweet_id": "1", "text": "Duplicate"}
    ])
    df_clean = clean_raw_dataframe(df_raw)
    assert len(df_clean) == 1
    assert df_clean.iloc[0]["tweet_id"] == "1"
    assert df_clean.iloc[0]["text"] == "Need help"

def test_intent_taxonomy_loader():
    tax = get_default_intent_taxonomy()
    assert isinstance(tax, IntentTaxonomy)
    assert tax.brand == "AmazonHelp"
    assert len(tax.intents) >= 8
    assert "order_delivery_delay" in tax.get_intent_names()

def test_majority_baseline_classifier():
    clf = MajorityBaselineClassifier()
    clf.fit(["a", "b", "c"], ["order_delivery_delay", "order_delivery_delay", "refund_return_status"])
    res = clf.predict_one("I need help")
    assert res["intent"] == "order_delivery_delay"
    assert res["confidence"] == 1.0

def test_tfidf_baseline_classifier():
    clf = TFIDFBaselineClassifier()
    clf.fit(
        ["where is my package", "refund my money please", "broken jar arrived"],
        ["order_delivery_delay", "refund_return_status", "damaged_defective_item"]
    )
    res = clf.predict_one("where is my order shipped")
    assert res["intent"] in ["order_delivery_delay", "refund_return_status", "damaged_defective_item"]
    assert 0.0 <= res["confidence"] <= 1.0

def test_escalation_policy_auto_handle():
    policy = DecisionPolicy(min_intent_confidence=0.6, min_retrieval_similarity=0.4)
    res = policy.evaluate_decision(
        intent="order_delivery_delay",
        intent_confidence=0.85,
        retrieved_evidence=[{"similarity_score": 0.75}],
        generation_result={"validation_passed": True}
    )
    assert res["decision"] == "AUTO_HANDLE"

def test_escalation_policy_escalate_low_confidence():
    policy = DecisionPolicy(min_intent_confidence=0.6, min_retrieval_similarity=0.4)
    res = policy.evaluate_decision(
        intent="order_delivery_delay",
        intent_confidence=0.35,
        retrieved_evidence=[{"similarity_score": 0.75}],
        generation_result={"validation_passed": True}
    )
    assert res["decision"] == "ESCALATE"
    assert "Low intent confidence" in res["reason"]

def test_response_validation_safety():
    val = validate_generated_reply(
        reply_text="We will give you a 100% refund immediately with $50 credit!",
        retrieved_evidence=[],
        customer_message="My box was late"
    )
    assert val["passed"] is False
    assert any("unsupported promise" in r.lower() for r in val["reasons"])

def test_decision_metrics():
    gold = ["AUTO_HANDLE", "AUTO_HANDLE", "ESCALATE", "ESCALATE"]
    pred = ["AUTO_HANDLE", "ESCALATE", "ESCALATE", "AUTO_HANDLE"]
    m = compute_decision_metrics(gold, pred)
    assert m["total_examples"] == 4
    assert m["decision_accuracy"] == 0.5
    assert m["safe_auto_handle_rate"] == 0.5
    assert m["unsafe_auto_handle_rate"] == 0.5

def test_simple_cache(tmp_path):
    cache = SimpleCache(cache_dir=str(tmp_path / "cache"))
    cache.set("prompt_key_1", {"response": "ok"})
    val = cache.get("prompt_key_1")
    assert val == {"response": "ok"}
    assert cache.get("non_existent_key") is None
