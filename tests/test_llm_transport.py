"""Offline tests of the LLM transport: fallbacks, JSON-mode handling, reasoning answers, budgets."""
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import requests

import ai_brain
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


class Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text or json.dumps(body or {})

    def json(self):
        return self._body


def ok(content, reasoning=None, finish="stop"):
    return Resp(200, {"choices": [{"message": {"content": content, "reasoning_content": reasoning},
                                   "finish_reason": finish}], "usage": {"completion_tokens": 10}})


calls = []


def install(script):
    """script: {model: [response-or-exception, ...]} consumed in order per model."""
    queues = {m: list(v) for m, v in script.items()}

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"model": json["model"], "json_mode": "response_format" in json, "body": json})
        item = queues[json["model"]].pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    ai_brain._session.post = post
    calls.clear()


config.DEEPSEEK_API_KEY = "test"
config.DEEPSEEK_MODEL = "primary"
config.LLM_FALLBACK_MODELS = ["backup-a", "backup-b"]
config.DEEPSEEK_MAX_RETRIES = 2
ai_brain.time.sleep = lambda s: None
GOOD = '{"signal": "HOLD", "confidence_score": 20, "logic": "x"}'

install({"primary": [ok(GOOD)]})
r = ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
check("Primary answers first time", r["content"] == GOOD and len(calls) == 1 and calls[0]["json_mode"])

install({"primary": [requests.ReadTimeout("slow"), requests.ReadTimeout("slow")], "backup-a": [ok(GOOD)]})
r = ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
check("Timeouts on primary -> fallback model answers", r["model"] == "backup-a" and [c["model"] for c in calls] == ["primary", "primary", "backup-a"])

install({"primary": [Resp(404, text="Function not found")], "backup-a": [Resp(503, text="overloaded"), ok(GOOD)]})
r = ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
check("404 skips straight to fallback; 503 is retried", r["model"] == "backup-a" and len(calls) == 3)

install({"primary": [ok(None, reasoning="thinking only"), ok(GOOD)]})
r = ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
check("Empty answer in JSON mode -> retried without JSON mode", r["content"] == GOOD and calls[0]["json_mode"] and not calls[1]["json_mode"])

install({"primary": [Resp(400, text="response_format not supported"), ok(GOOD)]})
r = ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
check("HTTP 400 on JSON mode -> retried without it", r["content"] == GOOD and not calls[1]["json_mode"])

install({"primary": [ok(None, reasoning='Let me think... final: {"signal": "BUY", "confidence_score": 70}')]})
r = ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
check("JSON only in reasoning_content is still used", ai_brain.parse_json_payload(r["content"])["signal"] == "BUY")

install({"primary": [ok("thinking... thinking...", finish="length"), ok("still thinking", finish="length"), ok("more", finish="length")],
         "backup-a": [ok(GOOD)]})
r = ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
check("Ran out of tokens -> next model", r["model"] == "backup-a")

install({"primary": [Resp(402, text="Insufficient Balance")], "backup-a": [ok(GOOD)]})
try:
    ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
    check("402 stops immediately (same key for every model)", False)
except ai_brain.DeepSeekError as exc:
    check("402 stops immediately (same key for every model)", "402" in str(exc) and len(calls) == 1)

install({"primary": [Resp(503)] * 2, "backup-a": [Resp(404)], "backup-b": [requests.ConnectionError("down")] * 2})
try:
    ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
    check("All models failing -> one combined error", False)
except ai_brain.DeepSeekError as exc:
    check("All models failing -> one combined error", all(m in str(exc) for m in ("primary", "backup-a", "backup-b")), str(exc)[:120])

config.LLM_REASONING = "off"
install({"primary": [ok(GOOD)]})
ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
check("LLM_REASONING=off sends the thinking switch", calls[0]["body"].get("chat_template_kwargs") == {"enable_thinking": False})
config.LLM_REASONING = "default"
install({"primary": [ok(GOOD)]})
ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
check("Default sends no thinking switch", "chat_template_kwargs" not in calls[0]["body"])
check("Default token budget is large enough for reasoning", calls[0]["body"]["max_tokens"] == config.LLM_MAX_TOKENS >= 4096)

ai_brain.LLM_CALL_BUDGET_SECONDS = 0
install({"primary": [ok(GOOD)]})
try:
    ai_brain.deepseek_chat([{"role": "user", "content": "x"}])
    check("Time budget stops further attempts", False)
except ai_brain.DeepSeekError as exc:
    check("Time budget stops further attempts", "time budget" in str(exc) and not calls)
ai_brain.LLM_CALL_BUDGET_SECONDS = 300

p = ai_brain.parse_json_payload
check("parse: <think> block ignored", p('<think>{"signal":"BUY"} maybe</think>{"signal":"HOLD"}')["signal"] == "HOLD")
check("parse: last object wins after reasoning prose", p('I considered {"signal": "BUY"} but final {"signal": "SELL", "c": 1}')["signal"] == "SELL")
check("parse: nested auditor object kept whole", p('{"rules": [{"a": 1}], "summary": "s"}')["summary"] == "s")
check("parse: fenced JSON", p('```json\n{"signal": "HOLD"}\n```')["signal"] == "HOLD")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
