"""
annotation_server.py — v2
=========================
Flask backend for the Hiver Human Annotation Tool.

Endpoints
---------
GET  /api/examples           → all golden set examples
GET  /api/intents            → intent taxonomy
POST /api/save               → save ONE intent/decision annotation (human only)
GET  /api/stats              → human-only stats
GET  /api/replies            → reply-quality subset (30–50 examples with generated replies)
POST /api/save-reply         → save one reply-quality annotation

Safety rules
------------
* /api/save will REFUSE to overwrite an existing human annotation unless
  the payload includes "force": true.
* annotated_by_human is set to True only on an explicit /api/save call.
* Model recommendations are served read-only; they are NEVER written as gold.

Usage
-----
    python scripts/annotation_server.py

Then open http://localhost:5050
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    from flask import Flask, jsonify, request, send_from_directory

GOLDEN_JSONL   = ROOT / "evaluation" / "golden_set.jsonl"
GOLDEN_CSV     = ROOT / "evaluation" / "golden_set.csv"
PREDICTIONS    = ROOT / "results" / "predictions" / "agent_predictions.jsonl"
REPLY_QA_FILE  = ROOT / "evaluation" / "reply_quality_annotations.jsonl"
UI_DIR         = ROOT / "scripts" / "annotation_ui"

INTENTS = [
    {"name": "order_delivery_delay",        "description": "Tracking, missing package, shipment delays"},
    {"name": "refund_return_status",        "description": "Returns, refund status, reimbursement"},
    {"name": "damaged_defective_item",      "description": "Broken, damaged, wrong product received"},
    {"name": "account_login_prime",         "description": "Prime membership, account lock, OTP, subscription"},
    {"name": "payment_billing_issue",       "description": "Double charge, gift card, payment failure"},
    {"name": "digital_services",            "description": "Prime Video, Kindle, Fire TV, Alexa streaming"},
    {"name": "customer_service_complaint",  "description": "Agent critique, rude rep, escalation request"},
    {"name": "general_inquiry_availability","description": "Stock availability, pre-order, price match"},
    {"name": "out_of_scope_other",          "description": "Foreign language, greetings, spam, emojis, unclear"},
]

ESCALATION_CATEGORIES = [
    "Low confidence / ambiguity",
    "No reliable historical evidence",
    "Sensitive or high-risk issue",
    "Novel / unseen issue type",
    "Multiple competing intents",
    "Requires account-specific investigation",
    "Historical resolution is unclear",
    "Non-English / language barrier",
    "Customer expressed extreme frustration",
    "Other",
]

app = Flask(__name__, static_folder=str(UI_DIR))


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_golden_csv(examples: list[dict]) -> None:
    fieldnames = [
        "example_id", "conversation_id", "brand", "customer_message",
        "conversation_context", "model_recommended_intent",
        "model_recommended_decision", "gold_intent", "gold_decision",
        "escalation_reason", "escalation_category", "difficulty",
        "annotation_note", "flag_ambiguous", "flag_difficult",
        "annotated_by_human",
    ]
    with open(GOLDEN_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(examples)


def load_reply_qa() -> list[dict]:
    if not REPLY_QA_FILE.exists():
        return []
    return load_jsonl(REPLY_QA_FILE)


def save_reply_qa(rows: list[dict]) -> None:
    REPLY_QA_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, REPLY_QA_FILE)


# ── Compute human stats (ONLY from annotated_by_human=True examples) ──────────

def compute_stats(examples: list[dict]) -> dict:
    human = [e for e in examples if e.get("annotated_by_human") is True]
    intent_dist: dict[str, int] = {}
    decision_dist: dict[str, int] = {}
    for e in human:
        k = e.get("gold_intent", "?")
        intent_dist[k] = intent_dist.get(k, 0) + 1
        d = e.get("gold_decision", "?")
        decision_dist[d] = decision_dist.get(d, 0) + 1
    return {
        "total": len(examples),
        "human_done": len(human),
        "remaining": len(examples) - len(human),
        "pct": round(len(human) / len(examples) * 100, 1) if examples else 0,
        "auto_handle": decision_dist.get("AUTO_HANDLE", 0),
        "escalate": decision_dist.get("ESCALATE", 0),
        "intent_distribution": intent_dist,
        "decision_distribution": decision_dist,
    }


# ── API: Golden set ───────────────────────────────────────────────────────────

@app.route("/api/examples")
def api_examples():
    return jsonify(load_jsonl(GOLDEN_JSONL))


@app.route("/api/intents")
def api_intents():
    return jsonify(INTENTS)


@app.route("/api/escalation-categories")
def api_escalation_categories():
    return jsonify(ESCALATION_CATEGORIES)


@app.route("/api/stats")
def api_stats():
    examples = load_jsonl(GOLDEN_JSONL)
    return jsonify(compute_stats(examples))


@app.route("/api/save", methods=["POST"])
def api_save():
    """
    Save a single human intent/decision annotation.

    Required body fields:
      example_id       : str
      gold_intent      : str  (must be a valid intent name)
      gold_decision    : "AUTO_HANDLE" | "ESCALATE"

    Optional:
      escalation_category : str
      escalation_reason   : str  (required when gold_decision == "ESCALATE")
      annotation_note     : str
      flag_ambiguous      : bool
      flag_difficult      : bool
      force               : bool  (set true to overwrite existing human annotation)
    """
    body = request.get_json(force=True)
    example_id    = body.get("example_id", "").strip()
    gold_intent   = body.get("gold_intent", "").strip()
    gold_decision = body.get("gold_decision", "").strip()
    force         = bool(body.get("force", False))

    # -- Validation --
    if not example_id:
        return jsonify({"error": "example_id is required"}), 400

    valid_intents = {i["name"] for i in INTENTS}
    if gold_intent not in valid_intents:
        return jsonify({"error": f"Invalid intent: '{gold_intent}'"}), 400

    if gold_decision not in ("AUTO_HANDLE", "ESCALATE"):
        return jsonify({"error": "gold_decision must be AUTO_HANDLE or ESCALATE"}), 400

    if gold_decision == "ESCALATE" and not body.get("escalation_reason", "").strip():
        return jsonify({"error": "escalation_reason is required when decision is ESCALATE"}), 400

    examples = load_jsonl(GOLDEN_JSONL)
    target = next((e for e in examples if e["example_id"] == example_id), None)
    if target is None:
        return jsonify({"error": f"example_id '{example_id}' not found"}), 404

    # -- Safety: refuse to silently overwrite existing human annotation --
    if target.get("annotated_by_human") and not force:
        return jsonify({
            "error": "already_annotated",
            "message": f"'{example_id}' was already annotated by a human. Send force=true to overwrite.",
        }), 409

    # -- Write annotation --
    target["gold_intent"]          = gold_intent
    target["gold_decision"]        = gold_decision
    target["escalation_category"]  = body.get("escalation_category") or None
    target["escalation_reason"]    = body.get("escalation_reason", "").strip() or None
    target["annotation_note"]      = body.get("annotation_note", "").strip() or None
    target["flag_ambiguous"]       = bool(body.get("flag_ambiguous", False))
    target["flag_difficult"]       = bool(body.get("flag_difficult", False))
    target["annotated_by_human"]   = True   # ← set ONLY here

    save_jsonl(examples, GOLDEN_JSONL)
    save_golden_csv(examples)

    stats = compute_stats(examples)
    return jsonify({
        "ok": True,
        "example_id": example_id,
        "human_done": stats["human_done"],
        "total": stats["total"],
        "remaining": stats["remaining"],
    })


# ── API: Reply quality ────────────────────────────────────────────────────────

@app.route("/api/replies")
def api_replies():
    """
    Return up to 50 examples that have a generated_reply.
    Merge in any existing human reply-quality annotations.
    """
    if not PREDICTIONS.exists():
        return jsonify([])

    preds = load_jsonl(PREDICTIONS)
    existing_qa = {r["example_id"]: r for r in load_reply_qa()}

    subset = []
    for p in preds:
        if not p.get("generated_reply"):
            continue
        ex_id = p["example_id"]
        item = {
            "example_id":             ex_id,
            "customer_message":       p.get("customer_message", ""),
            "gold_intent":            p.get("gold_intent", ""),
            "generated_reply":        p.get("generated_reply", ""),
            "retrieved_evidence_count": p.get("retrieved_evidence_count", 0),
            "top_retrieval_score":    p.get("top_retrieval_score", 0),
            # LLM judge scores (read-only reference)
            "llm_judge":              p.get("judge_metrics", {}),
            # Existing human annotation (if any)
            "human_qa":               existing_qa.get(ex_id, {}).get("human_qa"),
            "human_qa_annotated":     ex_id in existing_qa,
        }
        subset.append(item)
        if len(subset) >= 50:
            break

    return jsonify(subset)


@app.route("/api/save-reply", methods=["POST"])
def api_save_reply():
    """
    Save a single human reply-quality annotation.

    Required:
      example_id : str
      human_qa   : {
        relevance    : 1–5
        correctness  : 1–5
        grounding    : 1–5
        helpfulness  : 1–5
        tone         : 1–5
        flags        : list[str]   (e.g. ["hallucination", "should_escalate"])
        note         : str
      }
    """
    body = request.get_json(force=True)
    example_id = body.get("example_id", "").strip()
    human_qa   = body.get("human_qa", {})

    if not example_id:
        return jsonify({"error": "example_id is required"}), 400

    scores = ["relevance", "correctness", "grounding", "helpfulness", "tone"]
    for s in scores:
        v = human_qa.get(s)
        if v is None or not (1 <= int(v) <= 5):
            return jsonify({"error": f"'{s}' must be 1–5"}), 400

    rows = load_reply_qa()
    existing = next((r for r in rows if r["example_id"] == example_id), None)
    if existing:
        existing["human_qa"]             = human_qa
        existing["human_qa_annotated"]   = True
    else:
        rows.append({"example_id": example_id, "human_qa": human_qa, "human_qa_annotated": True})

    save_reply_qa(rows)
    done_count = sum(1 for r in rows if r.get("human_qa_annotated"))
    return jsonify({"ok": True, "reply_qa_done": done_count, "total_in_set": len(rows)})


@app.route("/api/reply-stats")
def api_reply_stats():
    rows = load_reply_qa()
    done = [r for r in rows if r.get("human_qa_annotated")]
    return jsonify({"total": len(rows), "human_done": len(done)})


# ── Static UI ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(UI_DIR), "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(str(UI_DIR), path)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  HIVER AI AGENT — HUMAN ANNOTATION TOOL  v2")
    print("=" * 65)
    print(f"  Golden Set     : {GOLDEN_JSONL}")
    print(f"  Predictions    : {PREDICTIONS}")
    print(f"  Reply QA file  : {REPLY_QA_FILE}")
    print(f"  UI             : {UI_DIR}")
    print(f"\n  Open browser:   http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
