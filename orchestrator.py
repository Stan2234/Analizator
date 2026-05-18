"""
Multi-agent orchestrator for Analizator Deep Brief.

Spawns N specialist sub-agents (news, macro, crypto, equity/smart-money,
sentiment) in parallel — each with a curated subset of the existing tools
from agent.py — collects their structured JSON briefings, then feeds them
to a synthesizer agent that produces a single executive brief.

Public entry point: run_deep_brief(user_query, ...)

All agents run on Opus by design (quality > cost for this workflow).
"""
from __future__ import annotations

import re
import json
import logging
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from agent import TOOLS, TOOL_DISPATCH, get_anthropic_client, OPUS_MODEL

log = logging.getLogger("analizator.orchestrator")

SUBAGENT_MAX_TOKENS = 2500
SUBAGENT_MAX_ITER = 6
SYNTH_MAX_TOKENS = 3500


# ---------------- sub-agent specifications ----------------

SUBAGENT_SPECS: List[Dict[str, Any]] = [
    {
        "id": "news",
        "title": "📰 News & Catalysts",
        "tools": ["search_news", "get_company_news"],
        "system": (
            "You are a market news analyst. Your job is to surface the most "
            "important news events of the last 24-48 hours that are moving or "
            "about to move financial markets — across macro (Fed/ECB/BOJ, "
            "geopolitics, regulation), equities (earnings, M&A, downgrades, "
            "guidance), crypto (regulation, ETF flows, hacks), and commodities. "
            "Cite real headlines with sources and timestamps. Skip clickbait. "
            "Group findings by impact, not by chronology.\n\n"
            "CRITICAL RULES:\n"
            "- DO NOT report asset PRICES or LEVELS in your findings. Prices in "
            "news headlines are stale by hours-to-days and you have no live "
            "quote tool. Other agents own the live price data. You report "
            "EVENTS, CATALYSTS, NARRATIVES — not numbers like 'gold at $X'.\n"
            "- Every finding must reference a real article you saw via "
            "search_news/get_company_news, with source name. If you cannot "
            "point to a specific tool result, do not include the claim.\n"
            "- If you are tempted to cite a number, only cite numbers that "
            "appeared verbatim in the news headlines/descriptions you "
            "retrieved, and qualify them as 'as reported by <source>'."
        ),
        "default_task": (
            "Produce a news briefing covering the last 24-48 hours. Call "
            "search_news multiple times across categories (markets, macro, "
            "crypto, geopolitics). Identify the 5-7 most market-relevant stories."
        ),
    },
    {
        "id": "macro",
        "title": "🌐 Macro & Rates",
        "tools": ["get_fred_series", "get_economic_calendar", "get_market_quote"],
        "system": (
            "You are a macro economist. Read the current macro regime from FRED "
            "(inflation: CPIAUCSL, CPILFESL; labor: UNRATE, PAYEMS, ICSA; "
            "growth: GDPC1; rates: FEDFUNDS, DGS10, DGS2, T10Y2Y; liquidity: M2SL; "
            "risk: VIXCLS; commodities: DCOILWTICO) and from the upcoming "
            "economic calendar. Classify the regime: tightening vs easing, "
            "disinflation vs reflation, expansion vs recession-risk. "
            "Identify what scheduled events in the next 2 weeks could shift it."
        ),
        "default_task": (
            "Pull the most recent observations for the key FRED series above, "
            "check the economic calendar 14 days ahead, and assess the current "
            "macro regime plus the next regime-shifting catalyst."
        ),
    },
    {
        "id": "crypto",
        "title": "🪙 Crypto",
        "tools": ["get_crypto_overview", "get_market_quote", "get_live_quote",
                  "get_polymarket_predictions", "get_signal"],
        "system": (
            "You are a digital asset analyst. Read the current state of crypto: "
            "BTC and ETH price/signal, total market cap, BTC dominance, ETH "
            "dominance, fear & greed, what's trending, and what Polymarket is "
            "pricing for crypto-related events. Focus on what matters to a "
            "serious investor — flow, structure, positioning — not retail noise."
        ),
        "default_task": (
            "Pull crypto overview, BTC and ETH quotes plus signals, and "
            "Polymarket predictions filtered to crypto. Produce a structural "
            "read on the current crypto regime."
        ),
    },
    {
        "id": "equity",
        "title": "🏛️ Smart Money & Equity",
        "tools": ["get_institution_filings", "get_institution_top_holdings",
                  "get_analyst_recommendations", "get_market_quote",
                  "get_company_news"],
        "system": (
            "You are a smart-money tracker. Surface what large institutions "
            "(JPMorgan, Goldman, BlackRock, Berkshire, Bridgewater, Renaissance, "
            "Citadel, Two Sigma, Point72, Tiger Global) are doing: recent SEC "
            "filings (13F-HR for positions — REMEMBER these lag 45 days — "
            "Form 4 for insider trades, 8-K for material events), top holdings, "
            "and analyst recommendation trends on the most-mentioned tickers. "
            "Look for signal: large new positions, concentrated exits, insider "
            "clusters, analyst rating shifts. Be honest about 13F staleness."
        ),
        "default_task": (
            "Pull recent filings for 2-3 of the tracked institutions, surface "
            "top holdings, and get analyst recs on any standout tickers. "
            "Identify the most actionable smart-money signal right now."
        ),
    },
    {
        "id": "sentiment",
        "title": "💭 Sentiment & Positioning",
        "tools": ["get_sentiment_indexes", "get_polymarket_predictions",
                  "get_crypto_overview"],
        "system": (
            "You are a positioning and sentiment analyst. Read the crowd across "
            "all four fear & greed indexes (crypto, stocks, commodities, macro) "
            "and across Polymarket prediction prices (rate cuts, recession, "
            "election outcomes, geopolitical events, asset price targets). "
            "When sentiment is extreme on either tail, contrarian reads matter. "
            "Flag where sentiment indexes disagree across asset classes — those "
            "divergences are often the most informative."
        ),
        "default_task": (
            "Pull all sentiment indexes and the most relevant Polymarket "
            "markets. Identify extremes, divergences, and what they imply for "
            "positioning."
        ),
    },
]


OUTPUT_SCHEMA_INSTRUCTION = """
After your tool work, output ONLY a JSON object (no prose before or after, no markdown fences) with EXACTLY this schema:

{
  "headline": "one specific sentence summarizing your most important finding",
  "key_findings": ["finding 1 with numbers/names/dates", "finding 2", "..."],
  "data_points": {"label": "value", "...": "..."},
  "outlook": "your forward-looking read in 1-2 sentences",
  "risks": ["risk 1", "risk 2"],
  "confidence": "high"
}

Rules:
- key_findings: 3-6 items, each with concrete numbers/sources/timestamps where possible
- data_points: 5-10 key numeric values (e.g. "BTC price": "$67,432", "10y yield": "4.28%")
- confidence must be exactly one of: "high", "medium", "low"
- Do NOT wrap the JSON in code fences. Do NOT add commentary before or after.
"""


# ---------------- sub-agent runner ----------------

def _filter_tools(tool_names: List[str]) -> List[Dict[str, Any]]:
    name_set = set(tool_names)
    return [t for t in TOOLS if t["name"] in name_set]


def _parse_subagent_output(text: str, agent_id: str) -> Dict[str, Any]:
    """Best-effort: parse JSON from the sub-agent's final text response."""
    if not text:
        return _failed_parse("(empty response)", agent_id)
    s = text.strip()
    # strip ```json fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # try to extract the first {...} block
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return _failed_parse(text, agent_id)


def _failed_parse(text: str, agent_id: str) -> Dict[str, Any]:
    return {
        "headline": (text or "(no response)")[:240],
        "key_findings": [],
        "data_points": {},
        "outlook": "",
        "risks": [],
        "confidence": "low",
        "_parse_failed": True,
    }


def _run_subagent(spec: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    client = get_anthropic_client()
    tools = _filter_tools(spec["tools"])
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    system = (
        f"{spec['system']}\n\n"
        f"Today (UTC): {today}\n\n"
        f"{OUTPUT_SCHEMA_INSTRUCTION}"
    )
    if user_query and user_query.strip():
        task = f"{spec['default_task']}\n\nUser focus for this brief: {user_query.strip()}"
    else:
        task = spec["default_task"]

    convo: List[Dict[str, Any]] = [{"role": "user", "content": task}]
    tool_log: List[Dict[str, Any]] = []

    for _ in range(SUBAGENT_MAX_ITER):
        try:
            resp = client.messages.create(
                model=OPUS_MODEL,
                max_tokens=SUBAGENT_MAX_TOKENS,
                system=system,
                tools=tools,
                messages=convo,
            )
        except Exception as e:
            log.exception("subagent %s anthropic call failed", spec["id"])
            return {
                "id": spec["id"], "title": spec["title"],
                "result": _failed_parse(f"API error: {e}", spec["id"]),
                "tool_log": tool_log, "raw": "", "error": str(e),
            }

        assistant_blocks: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        tool_uses: List[Dict[str, Any]] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_blocks.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            elif btype == "tool_use":
                tu = {"type": "tool_use", "id": block.id,
                      "name": block.name, "input": dict(block.input or {})}
                assistant_blocks.append(tu)
                tool_uses.append(tu)
        convo.append({"role": "assistant", "content": assistant_blocks})

        if resp.stop_reason != "tool_use" or not tool_uses:
            text = "\n".join(text_parts).strip()
            parsed = _parse_subagent_output(text, spec["id"])
            return {
                "id": spec["id"], "title": spec["title"],
                "result": parsed, "tool_log": tool_log, "raw": text,
            }

        tool_results_content: List[Dict[str, Any]] = []
        for tu in tool_uses:
            fn = TOOL_DISPATCH.get(tu["name"])
            try:
                result = fn(tu["input"]) if fn else {"error": f"unknown tool {tu['name']}"}
            except Exception as e:
                log.exception("subagent %s tool %s failed", spec["id"], tu["name"])
                result = {"error": str(e)}
            tool_log.append({"name": tu["name"], "args": tu["input"],
                             "result_preview": str(result)[:200]})
            tool_results_content.append({
                "type": "tool_result", "tool_use_id": tu["id"],
                "content": json.dumps(result, default=str)[:15000],
            })
        convo.append({"role": "user", "content": tool_results_content})

    return {
        "id": spec["id"], "title": spec["title"],
        "result": _failed_parse("(iteration limit hit)", spec["id"]),
        "tool_log": tool_log, "raw": "",
    }


# ---------------- synthesizer ----------------

SYNTH_SYSTEM = """You are the Chief Strategist at a multi-strategy fund. Five specialist analysts (news, macro, crypto, equity/smart-money, sentiment) have just produced structured briefings. Your job is to weave them into a single executive brief that a sophisticated investor can act on.

Output as clean markdown with these sections in order:

## TL;DR
2-3 sentences. The single most important read on markets right now.

## Cross-Asset Themes
3-5 themes that emerge across multiple agents' findings. Cite which agents support each theme (e.g. "macro + sentiment both flag...").

## Agreements & Contradictions
Where do the agents agree? Where do they conflict? What does the conflict imply?

## Actionable Insights
3-5 specific, concrete things a sophisticated investor should consider doing or watching this week. Cite numbers and assets.

## Key Risks
2-4 risks that could invalidate the current read.

Rules:
- Be specific. Cite numbers, tickers, levels, and dates from the agent outputs.
- Don't be generic ("monitor inflation" is useless; "watch DGS10 above 4.50% as a trigger" is useful).
- If an agent reported _parse_failed=true or low confidence, weight it accordingly.

DATA INTEGRITY RULES (critical):
- If two agents report DIFFERENT NUMERIC VALUES for the same asset/metric
  (e.g. gold at $3,200 vs $4,540, or 10Y yield at 4.28% vs 4.47%), you MUST
  explicitly flag the contradiction in 'Agreements & Contradictions'. Do NOT
  silently pick one. Do NOT average them.
- When flagging such a numeric contradiction, identify which source is more
  likely correct: agents with live-quote tools (crypto, equity, macro using
  get_market_quote/get_live_quote/get_fred_series) have authoritative price
  data. The news agent does NOT have price tools — any price it reports came
  from a possibly-stale article and should be treated as low-confidence.
- For your TL;DR and Actionable Insights, use prices/levels from the
  price-authoritative agents, never from news. If only news provides a
  number, qualify it as "reportedly" or omit it.
- Never invent or interpolate a number that no agent provided.
"""


def _synthesize(subagent_results: Dict[str, Any], user_query: str) -> str:
    client = get_anthropic_client()
    payload = {
        "user_focus": user_query.strip() if user_query else "General market briefing",
        "today_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        "agent_briefings": {
            sid: {
                "title": r.get("title"),
                "result": r.get("result"),
            }
            for sid, r in subagent_results.items()
        },
    }
    user_msg = (
        "Here are the five specialist briefings. Synthesize them into the "
        "executive brief described in the system prompt.\n\n"
        + json.dumps(payload, default=str, indent=2)
    )
    try:
        resp = client.messages.create(
            model=OPUS_MODEL,
            max_tokens=SYNTH_MAX_TOKENS,
            system=SYNTH_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        parts = [getattr(b, "text", "") for b in resp.content
                 if getattr(b, "type", None) == "text"]
        return "\n".join(p for p in parts if p).strip() or "(synthesizer returned no text)"
    except Exception as e:
        log.exception("synthesizer failed")
        return f"⚠️ Synthesizer error: {e}"


# ---------------- public entry point ----------------

def run_deep_brief(
    user_query: str = "",
    subagent_ids: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run the full multi-agent deep brief.

    Args:
        user_query: optional user focus (e.g. "focus on Fed and BTC this week").
                    Empty string = general briefing.
        subagent_ids: optional subset of sub-agent ids to run. Default = all.
        progress_callback: optional callable(agent_id, result_dict) called as
                           each sub-agent completes. Useful for Streamlit UI.

    Returns:
        {
          "subagents": {id: {id, title, result, tool_log, raw, ...}, ...},
          "synthesis": "<markdown brief>",
          "generated_at": "<iso utc>",
          "user_query": str,
        }
    """
    specs = SUBAGENT_SPECS
    if subagent_ids:
        wanted = set(subagent_ids)
        specs = [s for s in specs if s["id"] in wanted]
    if not specs:
        return {
            "subagents": {}, "synthesis": "(no sub-agents selected)",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "user_query": user_query,
        }

    results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(specs))) as ex:
        fut_map = {ex.submit(_run_subagent, s, user_query): s for s in specs}
        for fut in as_completed(fut_map):
            spec = fut_map[fut]
            try:
                results[spec["id"]] = fut.result()
            except Exception as e:
                log.exception("subagent %s crashed", spec["id"])
                results[spec["id"]] = {
                    "id": spec["id"], "title": spec["title"],
                    "result": _failed_parse(f"crashed: {e}", spec["id"]),
                    "tool_log": [], "raw": "", "error": str(e),
                }
            if progress_callback:
                try:
                    progress_callback(spec["id"], results[spec["id"]])
                except Exception:
                    log.exception("progress_callback raised")

    synthesis = _synthesize(results, user_query)

    return {
        "subagents": results,
        "synthesis": synthesis,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "user_query": user_query,
    }


def list_subagents() -> List[Dict[str, str]]:
    """Return a UI-friendly list of available sub-agents."""
    return [{"id": s["id"], "title": s["title"], "tools": ", ".join(s["tools"])}
            for s in SUBAGENT_SPECS]
