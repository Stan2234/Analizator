"""The day's news grouped by event rather than by shared words.

Word overlap groups "Trump taps Miller as chief of staff" with "Trump names Miller chief of
staff", but it cannot see that "Nasdaq sinks", "Oracle falls as Microsoft winds down deal"
and "Bitcoin ETFs bleed as tech rout tests buyers" are one market event. On a day like that
the event was split into a dozen one-outlet stories and a Cybertruck factbox led the page.

Every hour a small model names the last day's events and sorts each headline into one, so it says which ones report
the same event and how much that event matters to markets. Only the grouping comes from
the model: the outlet counts are recounted here from the headlines it pointed at, and the
headline shown is a real one from the group, never text the model wrote. When the stored
grouping is missing or stale the page falls back to the word-overlap clusters.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger("analizator.news_events")

KV_KEY = "news_events"
MAX_HEADLINES = 350
FRESH_HOURS = 3
ASSIGN_CHUNK = 70
# outlets count for more when the event moves markets: a sell-off told by eight outlets
# outranks a product recall told by five
RELEVANCE = {"high": 3.0, "medium": 1.5, "low": 0.5}

NAME_SYSTEM = (
    "You find the events in a day of news headlines for a markets website. You get numbered "
    "headlines from the last 24 hours as 'index | outlet | headline'.\n\n"
    "List the 12 events told by the most headlines and outlets. An event is one happening "
    "with its direct consequences: a sell-off in tech stocks, the companies falling with it, "
    "the flows it caused and the day's index moves are one event; two unrelated earnings "
    "reports are not. Do not merge events only because they share a theme such as 'AI' or "
    "'Trump'.\n\n"
    "For each event give:\n"
    "- label: a neutral description in at most 12 words, using only facts in the headlines\n"
    "- relevance: 'high' if it moves broad markets, rates, currencies or commodities; "
    "'medium' if it moves a sector or a large company; 'low' otherwise (politics with no "
    "market angle, sport, crime, single small companies, product stories)\n\n"
    'Reply with JSON only: {"events": [{"label": "...", "relevance": "high"}]}'
)

ASSIGN_SYSTEM = (
    "You sort news headlines into a fixed list of events. You get the numbered events, then "
    "numbered headlines as 'index | outlet | headline'. For every headline, in order, write "
    "one line 'index:event', where event is the number of the event it reports, including "
    "headlines about a company or asset caught up in that event, or 0 if it belongs to none. "
    "Write a line for every headline and nothing else."
)


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9$%]+", str(text).lower()) if len(w) > 2}


def group_events() -> Optional[Dict[str, Any]]:
    """Scheduler job: the model names the day's events, then assigns every headline to one of
    them. Two steps because one step left most of a big event's headlines out, and by a
    different amount on each run; sorting a list into fixed events is the easier task."""
    import agent as ai_agent
    import data_layer as dl

    if ai_agent.ai_outage():
        return None
    rows = [r for r in (dl.query_news(since_hours=24, limit=1000) or []) if r.get("title") and r.get("url")]
    rows = rows[:MAX_HEADLINES]
    if len(rows) < 10:
        return None
    lines = "\n".join(f"{i} | {r.get('source')} | {str(r['title'])[:200]}" for i, r in enumerate(rows))
    client = ai_agent.get_anthropic_client()

    def ask(system: str, user: str, max_tokens: int) -> str:
        resp = ai_agent.create_message(client, route="news_events", model=ai_agent.HAIKU_MODEL,
                                       max_tokens=max_tokens, temperature=0, system=system,
                                       messages=[{"role": "user", "content": user}])
        return "\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()

    raw = ask(NAME_SYSTEM, lines, 1500)
    try:
        named = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]).get("events") or []
    except Exception:
        log.warning("news events: unreadable event list")
        return None
    named = [e for e in named if e.get("label")][:12]
    if not named:
        return None
    menu = "\n".join(f"{k} | {e['label']}" for k, e in enumerate(named, 1))
    # in chunks: given all 350 at once it answered for fewer than half of them
    groups: Dict[int, List[int]] = {}
    all_lines = lines.split("\n")
    for c in range(0, len(all_lines), ASSIGN_CHUNK):
        chunk = "\n".join(all_lines[c:c + ASSIGN_CHUNK])
        raw = ask(ASSIGN_SYSTEM, f"Events:\n{menu}\n\nHeadlines:\n{chunk}", 1200)
        for i, k in re.findall(r"^\s*(\d+)\s*:\s*(\d+)\s*$", raw, re.M):
            i, k = int(i), int(k)
            if c <= i < c + ASSIGN_CHUNK and 1 <= k <= len(named):
                groups.setdefault(k, []).append(i)
    events = []
    for k, e in enumerate(named, 1):
        idx = sorted(set(groups.get(k, [])))
        if len(idx) < 2:
            continue
        # the lead is the real headline closest to the label, never text the model wrote
        # the head of the label names the event ("Tech stock selloff ..."); its tail lists
        # side effects ("... and crypto losses") that would pick a side-story headline
        lab = _tokens(" ".join(str(e["label"]).split()[:5]))
        lead = max(idx, key=lambda i: (len(lab & _tokens(rows[i]["title"])), -i))
        events.append({"label": str(e["label"])[:120],
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
