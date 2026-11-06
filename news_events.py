"""The day's news grouped by event rather than by shared words.

Word overlap groups "Trump taps Miller as chief of staff" with "Trump names Miller chief of
staff", but it cannot see that "Nasdaq sinks", "Oracle falls as Microsoft winds down deal"
and "Bitcoin ETFs bleed as tech rout tests buyers" are one market event. On a day like that
the event was split into a dozen one-outlet stories and a Cybertruck factbox led the page.

A small model reads the last day's headlines every 30 minutes and says which ones report
the same event and how much that event matters to markets. Only the grouping comes from
the model: the outlet counts are recounted here from the headlines it pointed at, and the
headline shown is a real one from the group, never text the model wrote. When the stored
grouping is missing or stale the page falls back to the word-overlap clusters.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("analizator.news_events")

KV_KEY = "news_events"
MAX_HEADLINES = 350
FRESH_HOURS = 3
# outlets count for more when the event moves markets: a sell-off told by eight outlets
# outranks a product recall told by five
RELEVANCE = {"high": 3.0, "medium": 1.5, "low": 0.5}

SYSTEM = (
    "You group news headlines into events for a markets website. You get numbered headlines "
    "from the last 24 hours as 'index | outlet | headline'.\n\n"
    "Return the 10 events reported by the most different outlets, each with at least two "
    "headlines. Headlines belong to one event when they report the same happening or its "
    "direct market consequences: a sell-off in tech stocks, the companies falling with it and "
    "the flows it caused are one event; two unrelated earnings reports are not. Do not merge "
    "events only because they share a theme such as 'AI' or 'Trump'. The same day's broad market "
    "moves (indices falling or rising, a sector rout, stocks at the close) are one event, not one "
    "per index or region.\n\n"
    "For each event give:\n"
    "- label: a neutral description of the event in at most 12 words, using only facts in "
    "its headlines\n"
    "- relevance: 'high' if it moves broad markets, rates, currencies or commodities; "
    "'medium' if it moves a sector or a large company; 'low' otherwise (politics with no "
    "market angle, sport, crime, single small companies, product stories)\n"
    "- lead: the index of the headline that states the event most plainly\n"
    "- items: the indices of all its headlines\n\n"
    'Reply with JSON only: {"events": [{"label": "...", "relevance": "high", "lead": 3, '
    '"items": [3, 17, 42]}]}'
)


def group_events() -> Optional[Dict[str, Any]]:
    """Scheduler job: ask the model for the event grouping and store it."""
    import agent as ai_agent
    import data_layer as dl

    if ai_agent.ai_outage():
        return None
    rows = [r for r in (dl.query_news(since_hours=24, limit=1000) or []) if r.get("title") and r.get("url")]
    rows = rows[:MAX_HEADLINES]
    if len(rows) < 10:
        return None
    lines = "\n".join(f"{i} | {r.get('source')} | {str(r['title'])[:200]}" for i, r in enumerate(rows))
    resp = ai_agent.create_message(
        ai_agent.get_anthropic_client(), route="news_events", model=ai_agent.HAIKU_MODEL,
        max_tokens=3000, system=SYSTEM, messages=[{"role": "user", "content": lines}])
    raw = "\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    start, end = raw.find("{"), raw.rfind("}")
    try:
        data = json.loads(raw[start:end + 1])
    except Exception:
        log.warning("news events: unreadable reply")
        return None
    events = []
    for e in data.get("events") or []:
        idx = sorted({int(i) for i in e.get("items") or [] if str(i).lstrip("-").isdigit()
                      and 0 <= int(i) < len(rows)})
        if len(idx) < 2:
            continue
        lead = int(e["lead"]) if str(e.get("lead", "")).isdigit() and int(e["lead"]) in idx else idx[0]
        events.append({"label": str(e.get("label") or "")[:120],
                       "relevance": e.get("relevance") if e.get("relevance") in RELEVANCE else "low",
                       "lead_url": rows[lead]["url"], "urls": [rows[i]["url"] for i in idx]})
    out = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "n_headlines": len(rows), "events": events}
    dl.kv_set(KV_KEY, out)
    return out


def event_clusters(df, stored: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The stored grouping applied to the headlines on the page, in the shape news_clusters()
    returns, ranked by the number of outlets weighted by market relevance. Empty when the
    grouping is missing or older than FRESH_HOURS."""
    import pandas as pd

    if df is None or df.empty or not stored or not stored.get("events"):
        return []
    try:
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(stored["ts"])
    except Exception:
        return []
    if age > dt.timedelta(hours=FRESH_HOURS):
        return []
    by_url = {u: r for u, r in zip(df["url"], df.to_dict("records")) if isinstance(u, str)}
    out = []
    for e in stored["events"]:
        items = [by_url[u] for u in e["urls"] if u in by_url]
        if len(items) < 2:
            continue
        lead = by_url.get(e["lead_url"]) or items[0]
        sources = sorted({str(i["source"]) for i in items})
        desc = max(items, key=lambda i: len(str(i.get("description") or ""))).get("description")
        out.append({"title": lead["title"], "url": lead["url"], "lead": desc, "label": e["label"],
                    "relevance": e["relevance"], "n": len(items), "sources": sources,
                    "n_sources": len(sources), "latest": max(i["published_at"] for i in items),
                    "category": pd.Series([i.get("category") or i.get("keyword") for i in items]).mode().iloc[0]})
    return sorted(out, key=lambda c: (RELEVANCE[c["relevance"]] * c["n_sources"], c["n"]), reverse=True)
