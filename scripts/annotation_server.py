"""
annotation_server.py
====================
Lightweight Flask server that serves the browser-based annotation UI
and exposes REST endpoints for reading/writing golden set labels.

Usage:
    python scripts/annotation_server.py

Then open: http://localhost:5050
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make sure project src is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    from flask import Flask, jsonify, request, send_from_directory

GOLDEN_JSONL = ROOT / "evaluation" / "golden_set.jsonl"
GOLDEN_CSV   = ROOT / "evaluation" / "golden_set.csv"
UI_DIR       = ROOT / "scripts" / "annotation_ui"

INTENTS = [
    {"name": "order_delivery_delay",       "description": "Tracking, missing package, shipment delays"},
    {"name": "refund_return_status",       "description": "Returns, refund status, reimbursement"},
    {"name": "damaged_defective_item",     "description": "Broken, damaged, wrong product received"},
    {"name": "account_login_prime",        "description": "Prime membership, account lock, OTP"},
    {"name": "payment_billing_issue",      "description": "Double charge, gift card issue, payment error"},
    {"name": "digital_services",          "description": "Prime Video, Kindle, Fire TV, Alexa streaming"},
    {"name": "customer_service_complaint", "description": "Agent critique, rude rep, escalation request"},
    {"name": "general_inquiry_availability","description": "Stock availability, pre-order, price match"},
    {"name": "out_of_scope_other",        "description": "Foreign language, greetings, spam, emojis"},
]

app = Flask(__name__, static_folder=str(UI_DIR))


def load_examples() -> list[dict]:
    examples = []
    with open(GOLDEN_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_examples(examples: list[dict]) -> None:
    """Save examples to both JSONL and CSV atomically."""
    # JSONL
    with open(GOLDEN_JSONL, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    # CSV (simple)
    import csv
    fieldnames = [
        "example_id", "conversation_id", "brand", "customer_message",
        "conversation_context", "model_recommended_intent",
        "model_recommended_decision", "gold_intent", "gold_decision",
        "escalation_reason", "difficulty", "annotation_note", "annotated_by_human",
    ]
    with open(GOLDEN_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(examples)


# ── API ROUTES ──────────────────────────────────────────────────────────────

@app.route("/api/examples")
def api_examples():
    examples = load_examples()
    return jsonify(examples)


@app.route("/api/intents")
def api_intents():
    return jsonify(INTENTS)


@app.route("/api/save", methods=["POST"])
def api_save():
    """
    Expects JSON body:
      {
        "example_id": "gold_001",
        "gold_intent": "...",
        "gold_decision": "AUTO_HANDLE|ESCALATE",
        "escalation_reason": "...",   # optional
        "annotation_note": "..."      # optional
      }
    """
    data = request.get_json(force=True)
    example_id = data.get("example_id")
    if not example_id:
        return jsonify({"error": "missing example_id"}), 400

    examples = load_examples()
    updated = False
    for ex in examples:
        if ex["example_id"] == example_id:
            ex["gold_intent"]        = data.get("gold_intent", ex.get("gold_intent"))
            ex["gold_decision"]      = data.get("gold_decision", ex.get("gold_decision"))
            ex["escalation_reason"]  = data.get("escalation_reason") or None
            ex["annotation_note"]    = data.get("annotation_note", ex.get("annotation_note", ""))
            ex["annotated_by_human"] = True
            updated = True
            break

    if not updated:
        return jsonify({"error": f"example_id {example_id!r} not found"}), 404

    save_examples(examples)
    human_done = sum(1 for e in examples if e.get("annotated_by_human"))
    return jsonify({"ok": True, "human_done": human_done, "total": len(examples)})


@app.route("/api/stats")
def api_stats():
    examples = load_examples()
    human_done = [e for e in examples if e.get("annotated_by_human")]
    intent_dist: dict[str, int] = {}
    decision_dist: dict[str, int] = {}
    for ex in human_done:
        intent_dist[ex.get("gold_intent", "?")] = intent_dist.get(ex.get("gold_intent", "?"), 0) + 1
        decision_dist[ex.get("gold_decision", "?")] = decision_dist.get(ex.get("gold_decision", "?"), 0) + 1
    return jsonify({
        "total": len(examples),
        "human_done": len(human_done),
        "remaining": len(examples) - len(human_done),
        "intent_distribution": intent_dist,
        "decision_distribution": decision_dist,
    })


# ── SERVE STATIC UI ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(UI_DIR), "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(str(UI_DIR), path)


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  HIVER AI AGENT — HUMAN ANNOTATION TOOL")
    print("="*65)
    print(f"  Golden Set : {GOLDEN_JSONL}")
    print(f"  UI Dir     : {UI_DIR}")
    print(f"\n  Open your browser at:  http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
