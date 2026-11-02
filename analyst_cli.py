"""One AI Analyst turn for AI Bubble Tracker, run with Market Analyst's own Python.

AI Bubble Tracker's server has none of the analyst's dependencies, so it starts this
script as a subprocess: the same agent, tools, AI credits and usage log serve both
apps, and a question asked on either spends from the same monthly allowance.

stdin:  {"op": "ask" | "credits", "user": {"customer": "cus_…", "tier": "plus"} | {"tier": "owner"},
         "messages": [...]}                    (messages in Anthropic format, for "ask")
stdout: one JSON line: {"text", "messages", "tools", "charts", "credits_left", "error"}
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

# Market Analyst's settings when started outside its own systemd unit
_env = os.path.join(HERE, ".env")
if os.path.exists(_env):
    for _line in open(_env, encoding="utf-8"):
        _k, _, _v = _line.strip().partition("=")
        if _k and not _k.startswith("#") and _v:
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
if os.path.exists("/opt/analizator-data/analizator.db"):
    os.environ.setdefault("ANALIZATOR_DB_PATH", "/opt/analizator-data/analizator.db")

AI_BUBBLE_NOTE = """
The reader is on AI Bubble Tracker, this site's dashboard on whether the AI boom's
finances add up. Most questions are about it: call get_ai_bubble first (the overview
for anything general, then the section the question is about), get_sec_filing for
anything only a filing holds, and show at most two charts with show_ai_bubble_chart.
The reader is already on the tracker, so refer to its pages by tab name ("the
Hyperscalers tab") rather than with links. Every other tool is still available for
the wider market around it. Most answers need two to five lookups; once the figures
that answer the question are in hand, stop looking and write.
"""


def _user(raw: dict, access) -> dict | None:
    if raw.get("tier") == "owner":
        return access.owner()
    if raw.get("tier") in access.TIERS and str(raw.get("customer", "")).startswith("cus_"):
        return {"customer": raw["customer"], "tier": raw["tier"]}
    return None


def _credits(user: dict, access) -> dict:
    left = access.credits_left(user)
    return {"credits_left": left, "credits_month": access.CREDITS.get(user["tier"]),
            "cost": access.COST["analyst"], "reset": access.next_reset().isoformat()}


def main() -> dict:
    req = json.loads(sys.stdin.read() or "{}")
    import access
    user = _user(req.get("user") or {}, access)
    if not user:
        return {"error": "The AI Analyst is part of the paid plans."}
    if req.get("op") == "credits":
        return _credits(user, access)

    import agent
    if agent.ai_outage():
        return {"error": "AI features are temporarily unavailable on our side. No credits were used; "
                         "try again in a few minutes.", **_credits(user, access)}
    err = access.charge(user, "analyst")
    if err:
        return {"error": err, **_credits(user, access)}
    try:
        # Opus for every question: on the tracker they are analysis questions however short,
        # and the 3-credit price was set on Opus's cost.
        res = agent.run_agent_turn(req.get("messages") or [], force_model=agent.OPUS_MODEL,
                                   extra_system=AI_BUBBLE_NOTE)
    except Exception as e:
        access.refund(user, "analyst")
        return {"error": f"The analyst failed: {type(e).__name__}. No credits were used.", **_credits(user, access)}
    if res["text"].startswith("⚠️"):
        access.refund(user, "analyst")

    import ai_bubble
    charts = []
    for cid in dict.fromkeys(str(tc["args"].get("chart")) for tc in res.get("tool_calls", [])
                             if tc["name"] == "show_ai_bubble_chart"):
        fig = ai_bubble.chart(cid)
        if fig is not None and len(charts) < 3:
            charts.append({"id": cid, "title": ai_bubble.CHARTS[cid][0], "tab": ai_bubble.CHARTS[cid][1],
                           "figure": json.loads(fig.to_json())})
    return {"text": res["text"], "messages": res["messages"],
            "tools": [{"name": tc["name"], "args": tc["args"]} for tc in res.get("tool_calls", [])],
            "charts": charts, **_credits(user, access)}


if __name__ == "__main__":
    try:
        out = main()
    except Exception as e:  # the caller reads one JSON line whatever happens
        out = {"error": f"{type(e).__name__}: {e}"}
    sys.stdout.write("\n" + json.dumps(out, default=str) + "\n")
