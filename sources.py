"""
External data source clients for the AI Macro Agent.

All functions are defensive: they catch exceptions and return [] / None
on failure so the scheduler keeps running. Each function returns plain
dicts ready to be passed to data_layer upsert helpers.

Sources implemented:
  - NewsAPI broad (top-headlines + everything)
  - RSS aggregator (~25 free finance/macro/crypto/geopolitics feeds)
  - CoinGecko (global crypto market + top coins)
  - Alternative.me (Crypto Fear & Greed)
  - CNN Fear & Greed (best-effort scrape)
  - FRED (key macro series)
  - Finnhub (earnings calendar, economic calendar, IPO calendar, analyst recs)
  - SEC EDGAR (recent filings per institution; 13F-HR + Form 4)
  - Yahoo Finance (live quote)
  - Binance (24h ticker)
"""
from __future__ import annotations

import os
import re
import json
import time
import html as ihtml
import datetime as dt
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests

USER_AGENT = "AnalizatorBot/1.0 (contact: analizator@example.com)"
REQ_TIMEOUT = 20

# ---------------- helpers ----------------

def _get(url: str, params: Optional[Dict[str, Any]] = None,
         headers: Optional[Dict[str, str]] = None, timeout: int = REQ_TIMEOUT) -> Optional[requests.Response]:
    h = {"User-Agent": USER_AGENT, "Accept": "application/json, text/xml, */*"}
    if headers:
        h.update(headers)
    try:
        r = requests.get(url, params=params, headers=h, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception:
        return None
    return None


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_date(s: Optional[str]) -> str:
    if not s:
        return _iso_now()
    try:
        # NewsAPI: 2024-01-15T12:34:56Z, RSS: many formats
        from email.utils import parsedate_to_datetime
        try:
            d = parsedate_to_datetime(s)
        except Exception:
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return _iso_now()


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = ihtml.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _categorize(title: str, source: str) -> str:
    t = (title or "").lower()
    if any(k in t for k in ("bitcoin", "btc", "ethereum", "eth", "crypto", "altcoin", "binance", "coinbase", "stablecoin")):
        return "crypto"
    if any(k in t for k in ("fed", "fomc", "powell", "ecb", "lagarde", "boj", "boe", "cpi", "inflation", "jobs report", "unemployment", "gdp", "recession", "yield")):
        return "macro"
    if any(k in t for k in ("war", "ukraine", "russia", "israel", "gaza", "china", "taiwan", "iran", "nato", "election", "sanction", "tariff")):
        return "geopolitics"
    if any(k in t for k in ("nvidia", "apple", "microsoft", "tesla", "google", "amazon", "ai chip", "openai", "earnings")):
        return "tech"
    if any(k in t for k in ("stock", "market", "s&p", "nasdaq", "dow", "treasury", "oil", "gold", "silver")):
        return "markets"
    return "general"


# ---------------- NewsAPI broad ----------------

NEWSAPI_BASE = "https://newsapi.org/v2"

def fetch_newsapi_top_headlines(api_key: str, max_per_category: int = 30) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    out: List[Dict[str, Any]] = []
    categories = ["business", "technology", "general"]
    for cat in categories:
        r = _get(f"{NEWSAPI_BASE}/top-headlines",
                 params={"category": cat, "language": "en", "pageSize": max_per_category, "apiKey": api_key})
        if not r:
            continue
        try:
            data = r.json()
        except Exception:
            continue
        for a in data.get("articles") or []:
            title = a.get("title") or ""
            url = a.get("url") or ""
            if not title or not url:
                continue
            out.append({
                "source": (a.get("source") or {}).get("name") or "NewsAPI",
                "source_kind": "newsapi",
                "title": title,
                "description": a.get("description") or "",
                "url": url,
                "published_at": _parse_date(a.get("publishedAt")),
                "category": _categorize(title, ""),
                "keywords": [cat],
            })
    return out


def fetch_newsapi_everything(api_key: str, query: str, page_size: int = 30,
                              hours_back: int = 24) -> List[Dict[str, Any]]:
    if not api_key or not query:
        return []
    frm = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = _get(f"{NEWSAPI_BASE}/everything",
             params={"q": query, "from": frm, "sortBy": "publishedAt",
                     "language": "en", "pageSize": page_size, "apiKey": api_key})
    if not r:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    out = []
    for a in data.get("articles") or []:
        title = a.get("title") or ""
        url = a.get("url") or ""
        if not title or not url:
            continue
        out.append({
            "source": (a.get("source") or {}).get("name") or "NewsAPI",
            "source_kind": "newsapi",
            "title": title,
            "description": a.get("description") or "",
            "url": url,
            "published_at": _parse_date(a.get("publishedAt")),
            "category": _categorize(title, ""),
            "keywords": [query],
        })
    return out


# ---------------- RSS aggregator ----------------

RSS_FEEDS: Dict[str, str] = {
    # Finance / markets
    "Reuters Business":      "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Markets":       "https://feeds.reuters.com/reuters/marketsNews",
    "Reuters World":         "https://feeds.reuters.com/Reuters/worldNews",
    "MarketWatch Top":       "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "MarketWatch Realtime":  "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "Yahoo Finance":         "https://finance.yahoo.com/news/rssindex",
    "CNBC Top":              "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Markets":          "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    "CNBC Economy":          "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "FT Markets":            "https://www.ft.com/markets?format=rss",
    "WSJ Markets":           "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "WSJ World":             "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "Bloomberg (via Google)":"https://news.google.com/rss/search?q=site:bloomberg.com&hl=en-US&gl=US&ceid=US:en",
    "Investing.com":         "https://www.investing.com/rss/news.rss",
    "ZeroHedge":             "https://feeds.feedburner.com/zerohedge/feed",
    "SeekingAlpha Market":   "https://seekingalpha.com/market_currents.xml",
    # Crypto
    "CoinDesk":              "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph":         "https://cointelegraph.com/rss",
    "TheBlock":              "https://www.theblock.co/rss.xml",
    "Decrypt":               "https://decrypt.co/feed",
    "Bitcoin Magazine":      "https://bitcoinmagazine.com/.rss/full/",
    # Macro / central banks
    "Federal Reserve":       "https://www.federalreserve.gov/feeds/press_all.xml",
    "ECB Press":             "https://www.ecb.europa.eu/rss/press.html",
    "BLS News":              "https://www.bls.gov/feed/news_release.rss",
    "BIS":                   "https://www.bis.org/list/press_releases/index.rss",
    # Geopolitics
    "AP World":              "https://feeds.apnews.com/rss/apf-worldnews",
    "BBC World":             "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Politico":              "https://rss.politico.com/politics-news.xml",
    "Al Jazeera":            "https://www.aljazeera.com/xml/rss/all.xml",
}


def _parse_rss_xml(xml_text: str) -> List[Dict[str, Any]]:
    """Lightweight RSS/Atom parser without extra deps. Good enough for headlines."""
    items: List[Dict[str, Any]] = []
    if not xml_text:
        return items
    try:
        import xml.etree.ElementTree as ET
        # Strip BOM
        xml_text = xml_text.lstrip("\ufeff")
        root = ET.fromstring(xml_text)
    except Exception:
        return items

    def _txt(el, tag, ns=None):
        if el is None:
            return ""
        for t in (tag, f"{{{ns}}}{tag}" if ns else tag):
            x = el.find(t)
            if x is not None and x.text:
                return x.text
        return ""

    # RSS 2.0
    for item in root.iter("item"):
        title = _txt(item, "title")
        link  = _txt(item, "link")
        desc  = _txt(item, "description")
        date  = _txt(item, "pubDate") or _txt(item, "{http://purl.org/dc/elements/1.1/}date")
        if title and link:
            items.append({"title": _strip_html(title), "url": link.strip(),
                          "description": _strip_html(desc), "published_at": _parse_date(date)})
    if items:
        return items
    # Atom
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title = _txt(entry, "{http://www.w3.org/2005/Atom}title")
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.get("href") if link_el is not None else ""
        desc = _txt(entry, "{http://www.w3.org/2005/Atom}summary") or _txt(entry, "{http://www.w3.org/2005/Atom}content")
        date = _txt(entry, "{http://www.w3.org/2005/Atom}updated") or _txt(entry, "{http://www.w3.org/2005/Atom}published")
        if title and link:
            items.append({"title": _strip_html(title), "url": link.strip(),
                          "description": _strip_html(desc), "published_at": _parse_date(date)})
    return items


def fetch_rss_feed(name: str, url: str, max_items: int = 40) -> List[Dict[str, Any]]:
    r = _get(url, headers={"Accept": "application/rss+xml, application/xml, text/xml, */*"})
    if not r:
        return []
    parsed = _parse_rss_xml(r.text)[:max_items]
    out = []
    for it in parsed:
        title = it.get("title") or ""
        out.append({
            "source": name,
            "source_kind": "rss",
            "title": title,
            "description": it.get("description") or "",
            "url": it.get("url") or "",
            "published_at": it.get("published_at") or _iso_now(),
            "category": _categorize(title, name),
            "keywords": [],
        })
    return out


def fetch_all_rss(max_per_feed: int = 30) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, url in RSS_FEEDS.items():
        try:
            out.extend(fetch_rss_feed(name, url, max_items=max_per_feed))
        except Exception:
            continue
    return out


# ---------------- CoinGecko ----------------

CG_BASE = "https://api.coingecko.com/api/v3"

def fetch_coingecko_global() -> Optional[Dict[str, Any]]:
    r = _get(f"{CG_BASE}/global")
    if not r:
        return None
    try:
        d = r.json().get("data") or {}
    except Exception:
        return None
    return {
        "total_mcap_usd": (d.get("total_market_cap") or {}).get("usd"),
        "total_vol_usd":  (d.get("total_volume") or {}).get("usd"),
        "btc_dominance":  (d.get("market_cap_percentage") or {}).get("btc"),
        "eth_dominance":  (d.get("market_cap_percentage") or {}).get("eth"),
        "active_cryptos": d.get("active_cryptocurrencies"),
        "markets":        d.get("markets"),
    }


def fetch_coingecko_top(n: int = 100) -> List[Dict[str, Any]]:
    r = _get(f"{CG_BASE}/coins/markets",
             params={"vs_currency": "usd", "order": "market_cap_desc",
                     "per_page": n, "page": 1, "price_change_percentage": "1h,24h,7d"})
    if not r:
        return []
    try:
        return r.json() or []
    except Exception:
        return []


def fetch_coingecko_trending() -> List[Dict[str, Any]]:
    r = _get(f"{CG_BASE}/search/trending")
    if not r:
        return []
    try:
        return [c.get("item", {}) for c in (r.json().get("coins") or [])]
    except Exception:
        return []


# ---------------- Fear & Greed ----------------

def fetch_crypto_fear_greed() -> Optional[Dict[str, Any]]:
    r = _get("https://api.alternative.me/fng/?limit=1")
    if not r:
        return None
    try:
        d = (r.json().get("data") or [{}])[0]
        return {"value": int(d.get("value")), "label": d.get("value_classification"),
                "timestamp": d.get("timestamp")}
    except Exception:
        return None


def fetch_stocks_fear_greed() -> Optional[Dict[str, Any]]:
    """
    CNN Fear & Greed first; on failure, build a synthetic stocks F&G from:
      - VIX (inverted): low VIX = greed
      - S&P 500 14d momentum
      - S&P 500 distance from 50-day MA
      - 52-week high proximity
      - Put/Call (skipped if unavailable)
    """
    # Try CNN first (works from some IPs / browsers)
    try:
        r = _get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                 headers={
                     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                     "Accept": "application/json, text/plain, */*",
                     "Origin": "https://www.cnn.com",
                     "Referer": "https://www.cnn.com/",
                 })
        if r:
            d = r.json() or {}
            fg = d.get("fear_and_greed") or {}
            val = fg.get("score")
            if val is not None:
                return {"value": int(round(float(val))),
                        "label": fg.get("rating") or _label_from_score(int(round(float(val)))),
                        "timestamp": fg.get("timestamp"),
                        "source": "CNN"}
    except Exception:
        pass

    # Fallback: synthetic from Yahoo
    try:
        scores = []
        details: Dict[str, Any] = {}

        # VIX
        r = _get(YAHOO_QUOTE.format("^VIX"), params={"interval": "1d", "range": "1mo"})
        if r:
            try:
                result = (r.json().get("chart") or {}).get("result") or []
                closes = ((result[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
                closes = [c for c in closes if c is not None]
                if closes:
                    vix = closes[-1]
                    details["VIX"] = round(vix, 2)
                    z = (20.0 - vix) / 6.0  # 14=greed, 26=fear
                    scores.append(z)
            except Exception:
                pass

        # S&P 500 — momentum + 50d distance + 52w high proximity
        r = _get(YAHOO_QUOTE.format("^GSPC"), params={"interval": "1d", "range": "1y"})
        if r:
            try:
                result = (r.json().get("chart") or {}).get("result") or []
                closes = ((result[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
                closes = [c for c in closes if c is not None]
                if len(closes) >= 60:
                    last = closes[-1]
                    # 14d momentum
                    ret_14d = (last / closes[-14] - 1.0) * 100.0
                    details["SPX_14d_pct"] = round(ret_14d, 2)
                    scores.append(ret_14d / 2.0)  # +2% over 14d ≈ +1z
                    # Distance from 50d MA
                    sma50 = sum(closes[-50:]) / 50.0
                    dist = (last / sma50 - 1.0) * 100.0
                    details["SPX_vs_50dMA_pct"] = round(dist, 2)
                    scores.append(dist / 2.0)
                    # 52w high proximity
                    hi = max(closes[-252:]) if len(closes) >= 252 else max(closes)
                    drawdown = (last / hi - 1.0) * 100.0  # negative
                    details["SPX_drawdown_from_52wk_pct"] = round(drawdown, 2)
                    # 0% = greed, -10% = fear
                    scores.append((drawdown + 5.0) / 3.0)
            except Exception:
                pass

        if not scores:
            return None
        avg_z = sum(scores) / len(scores)
        v = _z_to_score(avg_z)
        return {"value": v, "label": _label_from_score(v),
                "details": details, "source": "synthetic"}
    except Exception:
        return None


def _label_from_score(v: int) -> str:
    if v >= 75: return "Extreme Greed"
    if v >= 55: return "Greed"
    if v >= 45: return "Neutral"
    if v >= 25: return "Fear"
    return "Extreme Fear"


def _z_to_score(z: float) -> int:
    """Map a z-score (-2..+2) to 0..100 (higher = more bullish/greedy)."""
    import math
    s = 50 + 25 * max(min(z, 2.0), -2.0)
    return int(round(max(0, min(100, s))))


def fetch_commodities_sentiment() -> Optional[Dict[str, Any]]:
    """
    Synthetic commodities sentiment from gold/silver/oil 14d momentum.
    Higher = risk-on / commodities rallying.
    """
    try:
        symbols = ["GC=F", "SI=F", "DCOILWTICO"]  # gold, silver, WTI (via Yahoo: CL=F)
        # Use Yahoo for all three for consistency
        ymap = {"GC=F": "GC=F", "SI=F": "SI=F", "WTI": "CL=F"}
        scores = []
        details = {}
        for label, sym in ymap.items():
            r = _get(YAHOO_QUOTE.format(quote_plus(sym)),
                     params={"interval": "1d", "range": "1mo"})
            if not r:
                continue
            try:
                result = (r.json().get("chart") or {}).get("result") or []
                closes = ((result[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
                closes = [c for c in closes if c is not None]
                if len(closes) < 10:
                    continue
                ret_14d = (closes[-1] / closes[-min(14, len(closes))] - 1.0) * 100.0
                # Commodities z: ~5% over 14d ≈ +1z
                z = ret_14d / 5.0
                scores.append(z)
                details[label] = round(ret_14d, 2)
            except Exception:
                continue
        if not scores:
            return None
        avg_z = sum(scores) / len(scores)
        v = _z_to_score(avg_z)
        return {"value": v, "label": _label_from_score(v), "details": details}
    except Exception:
        return None


def fetch_macro_sentiment(fred_api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Synthetic global macro risk sentiment.
    Inputs (higher score = risk-on / greedy):
      - VIX (inverted): low VIX = greed
      - 10Y-2Y yield curve (inverted = fear)
      - HY OAS spread (inverted)
      - DXY momentum (strong USD = risk-off, inverted)
    Falls back to Yahoo if FRED key missing.
    """
    try:
        details: Dict[str, Any] = {}
        scores = []

        # VIX via Yahoo ^VIX
        r = _get(YAHOO_QUOTE.format("^VIX"), params={"interval": "1d", "range": "1mo"})
        try:
            if r:
                result = (r.json().get("chart") or {}).get("result") or []
                closes = ((result[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
                closes = [c for c in closes if c is not None]
                if closes:
                    vix = closes[-1]
                    details["VIX"] = round(vix, 2)
                    # VIX 12 = greed, 30 = fear
                    z = (20.0 - vix) / 6.0
                    scores.append(z)
        except Exception:
            pass

        # Yield curve via FRED T10Y2Y or fallback to Yahoo ^TNX-^IRX
        if fred_api_key:
            obs = fetch_fred_observations(fred_api_key, "T10Y2Y", limit=5)
            for o in obs:
                try:
                    val = float(o.get("value"))
                    details["T10Y2Y"] = round(val, 2)
                    # Inverted (-0.5) = fear, steep (+1) = greed
                    z = val / 0.5
                    scores.append(z)
                    break
                except Exception:
                    continue

        # HY spread via FRED BAMLH0A0HYM2
        if fred_api_key:
            obs = fetch_fred_observations(fred_api_key, "BAMLH0A0HYM2", limit=5)
            for o in obs:
                try:
                    val = float(o.get("value"))
                    details["HY_OAS"] = round(val, 2)
                    # 3% = greed, 6% = fear
                    z = (4.5 - val) / 1.0
                    scores.append(z)
                    break
                except Exception:
                    continue

        # DXY via Yahoo DX-Y.NYB momentum
        r = _get(YAHOO_QUOTE.format("DX-Y.NYB"), params={"interval": "1d", "range": "1mo"})
        try:
            if r:
                result = (r.json().get("chart") or {}).get("result") or []
                closes = ((result[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
                closes = [c for c in closes if c is not None]
                if len(closes) >= 14:
                    ret_14d = (closes[-1] / closes[-14] - 1.0) * 100.0
                    details["DXY_14d_pct"] = round(ret_14d, 2)
                    # Strong USD (>+1.5%) = risk-off
                    z = -ret_14d / 1.5
                    scores.append(z)
        except Exception:
            pass

        if not scores:
            return None
        avg_z = sum(scores) / len(scores)
        v = _z_to_score(avg_z)
        return {"value": v, "label": _label_from_score(v), "details": details}
    except Exception:
        return None


# ---------------- FRED ----------------

FRED_BASE = "https://api.stlouisfed.org/fred"

# Key series we always want fresh
FRED_KEY_SERIES = {
    "CPIAUCSL":     "CPI All Items",
    "CPILFESL":     "CPI Core",
    "UNRATE":       "Unemployment Rate",
    "PAYEMS":       "Nonfarm Payrolls",
    "GDP":          "GDP",
    "GDPC1":        "Real GDP",
    "FEDFUNDS":     "Fed Funds Rate",
    "DGS10":        "10Y Treasury",
    "DGS2":         "2Y Treasury",
    "T10Y2Y":       "10Y-2Y Spread",
    "DFF":          "Effective Fed Funds",
    "M2SL":         "M2 Money Stock",
    "DCOILWTICO":   "WTI Crude",
    "DEXUSEU":      "USD/EUR",
    "VIXCLS":       "VIX Close",
    "BAMLH0A0HYM2": "HY Spread",
    "UMCSENT":      "Consumer Sentiment",
    "ICSA":         "Initial Jobless Claims",
}

def fetch_fred_observations(api_key: str, series_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    r = _get(f"{FRED_BASE}/series/observations",
             params={"series_id": series_id, "api_key": api_key, "file_type": "json",
                     "sort_order": "desc", "limit": limit})
    if not r:
        return []
    try:
        return r.json().get("observations") or []
    except Exception:
        return []


# ---------------- Finnhub ----------------

FH_BASE = "https://finnhub.io/api/v1"

def fetch_finnhub_earnings_calendar(api_key: str, days_ahead: int = 14) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    today = dt.date.today()
    to = today + dt.timedelta(days=days_ahead)
    r = _get(f"{FH_BASE}/calendar/earnings",
             params={"from": today.isoformat(), "to": to.isoformat(), "token": api_key})
    if not r:
        return []
    try:
        return r.json().get("earningsCalendar") or []
    except Exception:
        return []


def fetch_finnhub_economic_calendar(api_key: str) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    r = _get(f"{FH_BASE}/calendar/economic", params={"token": api_key})
    if not r:
        return []
    try:
        return r.json().get("economicCalendar") or []
    except Exception:
        return []


def fetch_finnhub_ipo_calendar(api_key: str, days_ahead: int = 30) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    today = dt.date.today()
    to = today + dt.timedelta(days=days_ahead)
    r = _get(f"{FH_BASE}/calendar/ipo",
             params={"from": today.isoformat(), "to": to.isoformat(), "token": api_key})
    if not r:
        return []
    try:
        return r.json().get("ipoCalendar") or []
    except Exception:
        return []


def fetch_finnhub_recommendation(api_key: str, symbol: str) -> List[Dict[str, Any]]:
    if not api_key or not symbol:
        return []
    r = _get(f"{FH_BASE}/stock/recommendation", params={"symbol": symbol, "token": api_key})
    if not r:
        return []
    try:
        return r.json() or []
    except Exception:
        return []


def fetch_finnhub_company_news(api_key: str, symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
    if not api_key or not symbol:
        return []
    today = dt.date.today()
    frm = today - dt.timedelta(days=days_back)
    r = _get(f"{FH_BASE}/company-news",
             params={"symbol": symbol, "from": frm.isoformat(), "to": today.isoformat(), "token": api_key})
    if not r:
        return []
    try:
        out = []
        for a in r.json() or []:
            title = a.get("headline") or ""
            url = a.get("url") or ""
            if not title or not url:
                continue
            ts = a.get("datetime")
            try:
                pub = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else _iso_now()
            except Exception:
                pub = _iso_now()
            out.append({
                "source": a.get("source") or "Finnhub",
                "source_kind": "finnhub",
                "title": title,
                "description": a.get("summary") or "",
                "url": url,
                "published_at": pub,
                "category": _categorize(title, ""),
                "keywords": [symbol],
            })
        return out
    except Exception:
        return []


# ---------------- SEC EDGAR ----------------

# CIKs (10-digit padded) for default institutions
SEC_INSTITUTIONS: Dict[str, str] = {
    "JPMorgan Chase":           "0000019617",
    "Goldman Sachs":            "0000886982",
    "BlackRock":                "0001364742",
    "Berkshire Hathaway":       "0001067983",
    "Bridgewater Associates":   "0001350694",
    "Renaissance Technologies": "0001037389",
    "Citadel Advisors":         "0001423053",
    "Two Sigma Investments":    "0001179392",
    "Point72 Asset Management": "0001603466",
    "Tiger Global Management":  "0001167483",
}

SEC_HEADERS = {
    "User-Agent": "AnalizatorBot research@example.com",
    "Accept": "application/json",
}

def fetch_sec_recent_filings(cik: str, institution: str, forms: List[str] = ("13F-HR", "4", "8-K")) -> List[Dict[str, Any]]:
    cik10 = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []
    recent = (data.get("filings") or {}).get("recent") or {}
    out: List[Dict[str, Any]] = []
    n = len(recent.get("accessionNumber") or [])
    for i in range(n):
        form = recent["form"][i]
        if forms and form not in forms:
            continue
        acc = recent["accessionNumber"][i]
        filed = recent["filingDate"][i]
        period = recent.get("reportDate", [None]*n)[i]
        primary_doc = recent.get("primaryDocument", [""]*n)[i]
        acc_clean = acc.replace("-", "")
        url_filing = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik10}&type={form}"
        url_doc = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_clean}/{primary_doc}" if primary_doc else url_filing
        out.append({
            "cik": cik10,
            "institution": institution,
            "form_type": form,
            "filed_at": filed,
            "period": period,
            "accession_no": acc,
            "url": url_doc,
            "summary": {"primary_doc": primary_doc},
        })
    return out


def fetch_sec_all_institutions() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, cik in SEC_INSTITUTIONS.items():
        try:
            out.extend(fetch_sec_recent_filings(cik, name))
            time.sleep(0.2)  # SEC fair-use throttle
        except Exception:
            continue
    return out


# ---------------- Yahoo / Binance live quotes ----------------

YAHOO_QUOTE = "https://query2.finance.yahoo.com/v8/finance/chart/{}"

def fetch_yahoo_quote(symbol: str) -> Optional[Dict[str, Any]]:
    r = _get(YAHOO_QUOTE.format(quote_plus(symbol)), params={"interval": "1d", "range": "5d"})
    if not r:
        return None
    try:
        result = (r.json().get("chart") or {}).get("result") or []
        if not result:
            return None
        meta = result[0].get("meta") or {}
        return {
            "symbol": symbol,
            "price": meta.get("regularMarketPrice"),
            "prev_close": meta.get("chartPreviousClose"),
            "currency": meta.get("currency"),
            "exchange": meta.get("exchangeName"),
        }
    except Exception:
        return None


def fetch_binance_24h(symbol: str) -> Optional[Dict[str, Any]]:
    r = _get("https://api.binance.com/api/v3/ticker/24hr", params={"symbol": symbol})
    if not r:
        return None
    try:
        d = r.json()
        return {
            "symbol": symbol,
            "price": float(d.get("lastPrice") or 0),
            "change_pct": float(d.get("priceChangePercent") or 0),
            "volume": float(d.get("quoteVolume") or 0),
        }
    except Exception:
        return None
