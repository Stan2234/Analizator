"""
Conversational Claude agent for the AI Macro Agent.

The agent has tool access to the SQLite data layer and live API sources.
It uses Anthropic's tool-use protocol so the model decides which tools to
call based on the user's question. No fixed answer templates.

Model routing:
  - 'claude-opus-4-6'  for complex / analytical questions
  - 'claude-haiku-4-5' for short factual questions

Set ANTHROPIC_API_KEY in env or Streamlit secrets.
"""
from __future__ import annotations

import os
import json
import logging
import datetime as dt
from typing import Any, Dict, List, Optional

import data_layer as dl
import sources as src

log = logging.getLogger("analizator.agent")

OPUS_MODEL  = "claude-opus-4-5-20250929"  # adjust if Anthropic releases newer
HAIKU_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 2048


# ---------------- secrets ----------------

def _get_secret(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v:
        return v.strip()
    try:
        import streamlit as st  # type: ignore
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default


def get_anthropic_client():
    try:
        from anthropic import Anthropic
    except Exception as e:
        raise RuntimeError("anthropic package not installed") from e
    key = _get_secret("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return Anthropic(api_key=key)


# ---------------- tool definitions ----------------

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "search_news",
        "description": "Search the local news database (NewsAPI + RSS feeds: Reuters, Bloomberg, FT, CNBC, CoinDesk, Federal Reserve, ECB, BBC, AP, etc). Use this to find recent news on any topic, company, asset, or event.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword to match in title/description (optional)"},
                "category": {"type": "string", "enum": ["markets","macro","crypto","geopolitics","tech","general"], "description": "Optional category filter"},
                "since_hours": {"type": "integer", "description": "Only items newer than N hours (default 48)"},
                "limit": {"type": "integer", "description": "Max results (default 30, max 100)"}
            }
        }
    },
    {
        "name": "get_market_quote",
        "description": "Get the latest cached market quote for a symbol. Yahoo symbols (e.g. ^GSPC, NVDA, BTC-USD, EURUSD=X, GC=F) or Binance symbols (e.g. BTCUSDT, ETHUSDT).",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]
        }
    },
    {
        "name": "get_live_quote",
        "description": "Force a live fetch (Yahoo or Binance) for a symbol, bypassing cache. Slower than get_market_quote — use only when freshness matters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "source": {"type": "string", "enum": ["yahoo","binance","auto"]}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_signal",
        "description": "Get the latest technical signal (trend, RSI, MACD, BB, score, BUY/SELL recommendation) for a symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]
        }
    },
    {
        "name": "get_crypto_overview",
        "description": "Total crypto market cap, BTC/ETH dominance, fear & greed index, top trending coins, and the top-50 coins by market cap with 24h change.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "get_economic_calendar",
        "description": "Upcoming economic events (CPI, FOMC, NFP, earnings, etc).",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {"type": "integer", "description": "Default 14"},
                "country": {"type": "string", "description": "Optional ISO country filter"}
            }
        }
    },
    {
        "name": "get_fred_series",
        "description": "Federal Reserve Economic Data (FRED). Common series: CPIAUCSL, CPILFESL, UNRATE, PAYEMS, GDPC1, FEDFUNDS, DGS10, DGS2, T10Y2Y, M2SL, DCOILWTICO, VIXCLS, ICSA. Returns the most recent N observations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string"},
                "n": {"type": "integer", "description": "Default 5"}
            },
            "required": ["series_id"]
        }
    },
    {
        "name": "get_institution_filings",
        "description": "Recent SEC filings for one of the tracked institutions: JPMorgan Chase, Goldman Sachs, BlackRock, Berkshire Hathaway, Bridgewater Associates, Renaissance Technologies, Citadel Advisors, Two Sigma Investments, Point72 Asset Management, Tiger Global Management.",
        "input_schema": {
            "type": "object",
            "properties": {
                "institution": {"type": "string"},
                "form_type": {"type": "string", "description": "e.g. '13F-HR', '4', '8-K' (optional)"},
                "limit": {"type": "integer", "description": "Default 20"}
            },
            "required": ["institution"]
        }
    },
    {
        "name": "get_institution_top_holdings",
        "description": "Top holdings (from latest 13F) for a tracked institution. NOTE: 13F filings are quarterly and lag ~45 days — they are not real-time positions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "institution": {"type": "string"},
                "top_n": {"type": "integer", "description": "Default 25"}
            },
            "required": ["institution"]
        }
    },
    {
        "name": "get_company_news",
        "description": "Recent news headlines for a specific stock ticker via Finnhub.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "days_back": {"type": "integer", "description": "Default 7"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_analyst_recommendations",
        "description": "Analyst buy/hold/sell recommendation trend for a stock symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]
        }
    },
    {
        "name": "list_all_symbols",
        "description": "List every symbol the system currently has cached price data for.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "get_polymarket_predictions",
        "description": "Get live prediction market data from Polymarket. Shows real-money bets on whether stocks/crypto/commodities will go up or down, price targets, economic events, etc. Filter by keyword.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional keyword filter (e.g. 'Bitcoin', 'S&P', 'Gold', 'Tesla')"},
                "limit": {"type": "integer", "description": "Max results (default 20)"}
            }
        }
    },
    {
        "name": "get_sentiment_indexes",
        "description": "Get all 4 sentiment / fear & greed indexes: crypto (alternative.me), US stocks (CNN), commodities (synthetic from gold/silver/oil momentum), and global macro risk (synthetic from VIX, yield curve, HY spread, DXY). Each returns 0-100 (higher = greedier/risk-on).",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "get_db_health",
        "description": "Diagnostics: how many news/market/signal/SEC rows are stored, and the timestamp of the latest news.",
        "input_schema": {"type": "object", "properties": {}}
    },
]


# ---------------- tool implementations ----------------

def _tool_search_news(args: Dict[str, Any]) -> Any:
    rows = dl.query_news(
        since_hours=int(args.get("since_hours") or 48),
        category=args.get("category"),
        keyword=args.get("query"),
        limit=min(int(args.get("limit") or 30), 100),
    )
    return [{"title": r["title"], "source": r["source"], "url": r["url"],
             "published_at": r["published_at"], "category": r.get("category"),
             "description": (r.get("description") or "")[:300]} for r in rows]


def _tool_get_market_quote(args: Dict[str, Any]) -> Any:
    snap = dl.latest_market_snapshot(args["symbol"])
    return snap or {"error": f"no cached data for {args['symbol']}"}


def _tool_get_live_quote(args: Dict[str, Any]) -> Any:
    sym = args["symbol"]
    source = (args.get("source") or "auto").lower()
    if source == "auto":
        source = "binance" if sym.endswith("USDT") else "yahoo"
    if source == "binance":
        q = src.fetch_binance_24h(sym)
    else:
        q = src.fetch_yahoo_quote(sym)
    if q and q.get("price") is not None:
        dl.upsert_market_snapshot(sym, source, float(q["price"]),
                                  q.get("change_pct"), q.get("volume"))
    return q or {"error": "fetch failed"}


def _tool_get_signal(args: Dict[str, Any]) -> Any:
    s = dl.latest_signal(args["symbol"])
    return s or {"error": f"no signal for {args['symbol']}"}


def _tool_get_crypto_overview(_args: Dict[str, Any]) -> Any:
    cm = dl.latest_crypto_market()
    if not cm:
        return {"error": "no crypto market data yet — scheduler may still be warming up"}
    p = cm.get("payload") or {}
    return {
        "total_mcap_usd": cm.get("total_mcap_usd"),
        "total_vol_usd":  cm.get("total_vol_usd"),
        "btc_dominance":  cm.get("btc_dominance"),
        "eth_dominance":  cm.get("eth_dominance"),
        "fear_greed":     cm.get("fear_greed"),
        "fear_greed_label": cm.get("fear_greed_label"),
        "trending":       p.get("trending", []),
        "top_coins":      [{"symbol": c.get("symbol"), "name": c.get("name"),
                            "price": c.get("current_price"),
                            "change_24h_pct": c.get("price_change_percentage_24h"),
                            "mcap": c.get("market_cap")} for c in (p.get("top_coins") or [])][:50],
        "snapshot_at": cm.get("snapshot_at"),
    }


def _tool_get_economic_calendar(args: Dict[str, Any]) -> Any:
    days = int(args.get("days_ahead") or 14)
    country = args.get("country")
    rows = dl.query_econ_events(days_ahead=days, days_back=1)
    if country:
        rows = [r for r in rows if (r.get("country") or "").upper() == country.upper()]
    return rows[:200]


def _tool_get_fred_series(args: Dict[str, Any]) -> Any:
    n = int(args.get("n") or 5)
    rows = dl.latest_fred(args["series_id"], n=n)
    if rows:
        return {"series_id": args["series_id"], "name": src.FRED_KEY_SERIES.get(args["series_id"], ""),
                "observations": rows}
    # Live fallback
    api_key = _get_secret("FRED_API_KEY")
    if api_key:
        obs = src.fetch_fred_observations(api_key, args["series_id"], limit=n)
        dl.upsert_fred_series(args["series_id"], obs)
        return {"series_id": args["series_id"], "observations": obs[:n], "live": True}
    return {"error": "no FRED data"}


def _tool_get_institution_filings(args: Dict[str, Any]) -> Any:
    rows = dl.query_sec_filings(institution=args["institution"],
                                 form_type=args.get("form_type"),
                                 limit=int(args.get("limit") or 20))
    return rows


def _tool_get_institution_top_holdings(args: Dict[str, Any]) -> Any:
    rows = dl.latest_holdings_for(args["institution"], top_n=int(args.get("top_n") or 25))
    if not rows:
        return {"warning": "no parsed holdings stored. 13F filings are listed via get_institution_filings; full position parsing is on the roadmap.",
                "filings": dl.query_sec_filings(institution=args["institution"], form_type="13F-HR", limit=5)}
    return rows


def _tool_get_company_news(args: Dict[str, Any]) -> Any:
    api_key = _get_secret("FINNHUB_API_KEY")
    items = src.fetch_finnhub_company_news(api_key, args["symbol"], days_back=int(args.get("days_back") or 7))
    if items:
        dl.upsert_news_items(items)
    return [{"title": i["title"], "source": i["source"], "url": i["url"],
             "published_at": i["published_at"]} for i in items[:30]]


def _tool_get_analyst_recommendations(args: Dict[str, Any]) -> Any:
    api_key = _get_secret("FINNHUB_API_KEY")
    return src.fetch_finnhub_recommendation(api_key, args["symbol"])


def _tool_list_all_symbols(_args: Dict[str, Any]) -> Any:
    return [s["symbol"] for s in dl.all_latest_snapshots()]


def _tool_get_polymarket_predictions(args: Dict[str, Any]) -> Any:
    markets = src.fetch_polymarket_finance_markets(limit=int(args.get("limit") or 50))
    query = (args.get("query") or "").lower()
    if query:
        markets = [m for m in markets if query in m["question"].lower()]
    return [{"question": m["question"],
             "outcomes": m["outcomes"],
             "volume": m["volume"],
             "end_date": m.get("end_date", "")} for m in markets[:int(args.get("limit") or 20)]]


def _tool_get_db_health(_args: Dict[str, Any]) -> Any:
    return dl.db_health()


def _tool_get_sentiment_indexes(_args: Dict[str, Any]) -> Any:
    fred_key = _get_secret("FRED_API_KEY")
    return {
        "crypto":      src.fetch_crypto_fear_greed(),
        "stocks":      src.fetch_stocks_fear_greed(),
        "commodities": src.fetch_commodities_sentiment(),
        "macro":       src.fetch_macro_sentiment(fred_key),
    }


TOOL_DISPATCH = {
    "search_news":                  _tool_search_news,
    "get_market_quote":             _tool_get_market_quote,
    "get_live_quote":               _tool_get_live_quote,
    "get_signal":                   _tool_get_signal,
    "get_crypto_overview":          _tool_get_crypto_overview,
    "get_economic_calendar":        _tool_get_economic_calendar,
    "get_fred_series":              _tool_get_fred_series,
    "get_institution_filings":      _tool_get_institution_filings,
    "get_institution_top_holdings": _tool_get_institution_top_holdings,
    "get_company_news":             _tool_get_company_news,
    "get_analyst_recommendations":  _tool_get_analyst_recommendations,
    "list_all_symbols":             _tool_list_all_symbols,
    "get_polymarket_predictions":   _tool_get_polymarket_predictions,
    "get_db_health":                _tool_get_db_health,
    "get_sentiment_indexes":        _tool_get_sentiment_indexes,
}


# ---------------- system prompt ----------------

SYSTEM_PROMPT = """You are Analizator, an autonomous markets and macro analyst with live tool access to a database of news, market quotes, technical signals, economic calendar events, FRED macro series, SEC filings (13F/Form 4/8-K) for major institutions (JPMorgan, Goldman Sachs, BlackRock, Berkshire, Bridgewater, Renaissance, Citadel, Two Sigma, Point72, Tiger Global), Finnhub analyst data, and a global crypto market overview (CoinGecko + Fear & Greed).

Your job is to answer the user's questions freely, like an intelligent analyst. There are no scripted answers. Use your tools whenever they would help — usually they will. Chain multiple tools when useful (e.g. pull a quote + signal + news for the same symbol before answering).

Rules:
- Be direct and specific. Cite real numbers and dates pulled from tools.
- If a tool returns no data, say so honestly. Don't invent numbers.
- 13F filings are 45+ days delayed by SEC rule — never describe them as current positions.
- When asked "what's happening today", call search_news with since_hours=24 across multiple categories.
- When asked about a specific company, get the quote, signal, news, and analyst recs in parallel if possible.
- Format prices to a sensible number of decimals. Use thousand separators for big numbers.
- If the user asks something outside markets/macro, answer briefly and offer to bring it back to markets.
- Never refuse a question just because data is incomplete — give your best read with caveats.
- Today's date in UTC: {today}
"""


def _build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(today=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))


# ---------------- chat loop ----------------

def _route_model(user_msg: str) -> str:
    """Cheap heuristic: short factual = haiku, otherwise opus."""
    if len(user_msg) < 80 and not any(k in user_msg.lower() for k in
                                       ("analyz", "explain", "why", "compare", "scenario", "risk", "forecast", "outlook")):
        return HAIKU_MODEL
    return OPUS_MODEL


def run_agent_turn(messages: List[Dict[str, Any]], force_model: Optional[str] = None,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
    """
    Run one user-turn through Claude with tool-use loop.
    `messages` is the running conversation in Anthropic format
    (list of {role: 'user'|'assistant', content: ...}).
    Returns: {"text": str, "messages": updated_messages, "tool_calls": [...]}
    """
    client = get_anthropic_client()
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    user_text = ""
    if isinstance(last_user.get("content"), str):
        user_text = last_user["content"]
    elif isinstance(last_user.get("content"), list):
        for c in last_user["content"]:
            if isinstance(c, dict) and c.get("type") == "text":
                user_text = c.get("text", "")
                break
    model = force_model or _route_model(user_text)

    tool_calls_log: List[Dict[str, Any]] = []
    convo = list(messages)
    safety_iter = 0

    while safety_iter < 8:
        safety_iter += 1
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_build_system_prompt(),
                tools=TOOLS,
                messages=convo,
            )
        except Exception as e:
            log.exception("anthropic call failed")
            return {"text": f"⚠️ Agent error: {e}", "messages": convo, "tool_calls": tool_calls_log}

        # Build assistant message from response content blocks
        assistant_blocks = []
        text_out_parts: List[str] = []
        tool_uses = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_blocks.append({"type": "text", "text": block.text})
                text_out_parts.append(block.text)
            elif btype == "tool_use":
                tu = {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input or {})}
                assistant_blocks.append(tu)
                tool_uses.append(tu)

        convo.append({"role": "assistant", "content": assistant_blocks})

        if resp.stop_reason != "tool_use" or not tool_uses:
            return {"text": "\n".join(text_out_parts).strip() or "(no response)",
                    "messages": convo, "tool_calls": tool_calls_log}

        # Run the tools, return tool_results
        tool_results_content = []
        for tu in tool_uses:
            name = tu["name"]
            args = tu["input"]
            fn = TOOL_DISPATCH.get(name)
            try:
                result = fn(args) if fn else {"error": f"unknown tool {name}"}
            except Exception as e:
                log.exception("tool %s failed", name)
                result = {"error": str(e)}
            tool_calls_log.append({"name": name, "args": args, "result_preview": str(result)[:200]})
            tool_results_content.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": json.dumps(result, default=str)[:20000],
            })
        convo.append({"role": "user", "content": tool_results_content})

    return {"text": "(agent hit tool-use loop limit)", "messages": convo, "tool_calls": tool_calls_log}
