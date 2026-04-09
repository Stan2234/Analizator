"""
SQLite data layer for the AI Macro Agent.

Stores news, RSS items, market snapshots, economic events, SEC filings,
FRED series, fear & greed, crypto market overview. Single file DB.

DB path resolution:
  1. ANALIZATOR_DB_PATH env var if set (Render persistent disk -> /data/analizator.db)
  2. ./data/analizator.db (local dev)
"""
from __future__ import annotations

import os
import json
import sqlite3
import threading
import datetime as dt
from typing import Any, Dict, List, Optional, Iterable

_DB_LOCK = threading.RLock()
_CONN: Optional[sqlite3.Connection] = None


def get_db_path() -> str:
    p = os.environ.get("ANALIZATOR_DB_PATH")
    if p:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        return p
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "analizator.db")


def get_conn() -> sqlite3.Connection:
    global _CONN
    with _DB_LOCK:
        if _CONN is None:
            _CONN = sqlite3.connect(get_db_path(), check_same_thread=False, timeout=30.0)
            _CONN.row_factory = sqlite3.Row
            _CONN.execute("PRAGMA journal_mode=WAL;")
            _CONN.execute("PRAGMA synchronous=NORMAL;")
            _CONN.execute("PRAGMA foreign_keys=ON;")
            _init_schema(_CONN)
        return _CONN


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS news (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,
    source_kind  TEXT NOT NULL,         -- 'newsapi' | 'rss' | 'manual'
    title        TEXT NOT NULL,
    description  TEXT,
    url          TEXT NOT NULL,
    published_at TEXT NOT NULL,         -- ISO8601 UTC
    fetched_at   TEXT NOT NULL,         -- ISO8601 UTC
    keywords     TEXT,                  -- JSON list of matched keywords
    category     TEXT,                  -- 'markets'|'macro'|'crypto'|'geopolitics'|'tech'|'general'
    sentiment    REAL,                  -- optional, -1..1
    UNIQUE(url)
);
CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_category   ON news(category, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_source     ON news(source, published_at DESC);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    source       TEXT NOT NULL,         -- 'yahoo'|'binance'
    asset_class  TEXT,
    price        REAL,
    change_pct   REAL,
    volume       REAL,
    extra_json   TEXT,
    snapshot_at  TEXT NOT NULL,
    UNIQUE(symbol, source, snapshot_at)
);
CREATE INDEX IF NOT EXISTS idx_mkt_symbol ON market_snapshots(symbol, snapshot_at DESC);

CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    source       TEXT NOT NULL,
    timeframe    TEXT,
    signal       TEXT,
    score        REAL,
    confidence   REAL,
    payload_json TEXT,                  -- full signal dict
    computed_at  TEXT NOT NULL,
    UNIQUE(symbol, source, timeframe, computed_at)
);
CREATE INDEX IF NOT EXISTS idx_sig_symbol ON signals(symbol, computed_at DESC);

CREATE TABLE IF NOT EXISTS econ_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time   TEXT NOT NULL,         -- ISO8601 UTC
    country      TEXT,
    title        TEXT NOT NULL,
    importance   TEXT,                  -- 'low'|'medium'|'high'
    actual       TEXT,
    forecast     TEXT,
    previous     TEXT,
    source       TEXT,
    extra_json   TEXT,
    UNIQUE(event_time, title, country)
);
CREATE INDEX IF NOT EXISTS idx_econ_time ON econ_events(event_time);

CREATE TABLE IF NOT EXISTS fred_series (
    series_id    TEXT NOT NULL,
    obs_date     TEXT NOT NULL,
    value        REAL,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (series_id, obs_date)
);

CREATE TABLE IF NOT EXISTS sec_filings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    cik           TEXT NOT NULL,
    institution   TEXT,
    form_type     TEXT NOT NULL,        -- '13F-HR'|'4'|'8-K' etc.
    filed_at      TEXT NOT NULL,
    period        TEXT,
    accession_no  TEXT NOT NULL,
    url           TEXT,
    summary_json  TEXT,
    UNIQUE(accession_no)
);
CREATE INDEX IF NOT EXISTS idx_sec_inst ON sec_filings(institution, filed_at DESC);
CREATE INDEX IF NOT EXISTS idx_sec_form ON sec_filings(form_type, filed_at DESC);

CREATE TABLE IF NOT EXISTS sec_holdings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    accession_no  TEXT NOT NULL,
    institution   TEXT,
    period        TEXT,
    ticker        TEXT,
    name          TEXT,
    cusip         TEXT,
    shares        REAL,
    value_usd     REAL,
    pct_portfolio REAL,
    FOREIGN KEY (accession_no) REFERENCES sec_filings(accession_no) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_holdings_inst ON sec_holdings(institution, period DESC);
CREATE INDEX IF NOT EXISTS idx_holdings_tkr  ON sec_holdings(ticker);

CREATE TABLE IF NOT EXISTS crypto_market (
    snapshot_at      TEXT PRIMARY KEY,
    total_mcap_usd   REAL,
    total_vol_usd    REAL,
    btc_dominance    REAL,
    eth_dominance    REAL,
    fear_greed       INTEGER,
    fear_greed_label TEXT,
    payload_json     TEXT
);

CREATE TABLE IF NOT EXISTS kv_store (
    key       TEXT PRIMARY KEY,
    value     TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS chat_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,         -- 'user'|'assistant'|'tool'
    content     TEXT NOT NULL,
    tool_name   TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id, created_at);
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    with _DB_LOCK:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------- NEWS ----------------

def upsert_news_items(items: Iterable[Dict[str, Any]]) -> int:
    """items: dicts with keys: source, source_kind, title, description, url,
    published_at, keywords (list), category"""
    conn = get_conn()
    now = now_utc_iso()
    inserted = 0
    with _DB_LOCK:
        for it in items:
            url = (it.get("url") or "").strip()
            title = (it.get("title") or "").strip()
            if not url or not title:
                continue
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO news
                    (source, source_kind, title, description, url, published_at,
                     fetched_at, keywords, category, sentiment)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        it.get("source") or "",
                        it.get("source_kind") or "rss",
                        title,
                        it.get("description") or "",
                        url,
                        it.get("published_at") or now,
                        now,
                        json.dumps(it.get("keywords") or []),
                        it.get("category") or "general",
                        it.get("sentiment"),
                    ),
                )
                if conn.total_changes:
                    inserted += 1
            except Exception:
                continue
        conn.commit()
    return inserted


def query_news(
    since_hours: Optional[int] = None,
    category: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    conn = get_conn()
    sql = "SELECT * FROM news WHERE 1=1"
    args: List[Any] = []
    if since_hours is not None:
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=since_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        sql += " AND published_at >= ?"
        args.append(cutoff)
    if category:
        sql += " AND category = ?"
        args.append(category)
    if keyword:
        sql += " AND (title LIKE ? OR description LIKE ? OR keywords LIKE ?)"
        like = f"%{keyword}%"
        args.extend([like, like, like])
    sql += " ORDER BY published_at DESC LIMIT ?"
    args.append(int(limit))
    with _DB_LOCK:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def prune_news(retention_days: int = 90) -> int:
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=retention_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = get_conn()
    with _DB_LOCK:
        cur = conn.execute("DELETE FROM news WHERE published_at < ?", (cutoff,))
        conn.commit()
        return cur.rowcount


# ---------------- MARKET ----------------

def upsert_market_snapshot(symbol: str, source: str, price: float,
                            change_pct: Optional[float] = None,
                            volume: Optional[float] = None,
                            asset_class: Optional[str] = None,
                            extra: Optional[Dict[str, Any]] = None) -> None:
    conn = get_conn()
    with _DB_LOCK:
        conn.execute(
            """INSERT OR REPLACE INTO market_snapshots
               (symbol, source, asset_class, price, change_pct, volume, extra_json, snapshot_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (symbol, source, asset_class, price, change_pct, volume,
             json.dumps(extra or {}), now_utc_iso()),
        )
        conn.commit()


def latest_market_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    with _DB_LOCK:
        row = conn.execute(
            "SELECT * FROM market_snapshots WHERE symbol=? ORDER BY snapshot_at DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    return dict(row) if row else None


def all_latest_snapshots() -> List[Dict[str, Any]]:
    conn = get_conn()
    with _DB_LOCK:
        rows = conn.execute(
            """SELECT m.* FROM market_snapshots m
               INNER JOIN (
                  SELECT symbol, MAX(snapshot_at) AS mx
                  FROM market_snapshots GROUP BY symbol
               ) t ON m.symbol = t.symbol AND m.snapshot_at = t.mx"""
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------- SIGNALS ----------------

def upsert_signal(symbol: str, source: str, timeframe: str, payload: Dict[str, Any]) -> None:
    conn = get_conn()
    with _DB_LOCK:
        conn.execute(
            """INSERT OR REPLACE INTO signals
               (symbol, source, timeframe, signal, score, confidence, payload_json, computed_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (symbol, source, timeframe,
             payload.get("signal"), payload.get("score"), payload.get("confidence"),
             json.dumps(payload), now_utc_iso()),
        )
        conn.commit()


def latest_signal(symbol: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    with _DB_LOCK:
        row = conn.execute(
            "SELECT * FROM signals WHERE symbol=? ORDER BY computed_at DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["payload"] = json.loads(d.get("payload_json") or "{}")
    except Exception:
        d["payload"] = {}
    return d


# ---------------- ECON ----------------

def upsert_econ_events(events: Iterable[Dict[str, Any]]) -> int:
    conn = get_conn()
    n = 0
    with _DB_LOCK:
        for e in events:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO econ_events
                       (event_time, country, title, importance, actual, forecast, previous, source, extra_json)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (e.get("event_time"), e.get("country"), e.get("title"),
                     e.get("importance"), e.get("actual"), e.get("forecast"),
                     e.get("previous"), e.get("source"), json.dumps(e.get("extra") or {})),
                )
                n += 1
            except Exception:
                continue
        conn.commit()
    return n


def query_econ_events(days_ahead: int = 14, days_back: int = 1) -> List[Dict[str, Any]]:
    conn = get_conn()
    now = dt.datetime.now(dt.timezone.utc)
    lo = (now - dt.timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hi = (now + dt.timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _DB_LOCK:
        rows = conn.execute(
            "SELECT * FROM econ_events WHERE event_time BETWEEN ? AND ? ORDER BY event_time ASC",
            (lo, hi),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------- FRED ----------------

def upsert_fred_series(series_id: str, observations: List[Dict[str, Any]]) -> int:
    conn = get_conn()
    now = now_utc_iso()
    n = 0
    with _DB_LOCK:
        for obs in observations:
            try:
                val = obs.get("value")
                if val in (None, "", "."):
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO fred_series (series_id, obs_date, value, fetched_at)
                       VALUES (?,?,?,?)""",
                    (series_id, obs.get("date"), float(val), now),
                )
                n += 1
            except Exception:
                continue
        conn.commit()
    return n


def latest_fred(series_id: str, n: int = 1) -> List[Dict[str, Any]]:
    conn = get_conn()
    with _DB_LOCK:
        rows = conn.execute(
            "SELECT * FROM fred_series WHERE series_id=? ORDER BY obs_date DESC LIMIT ?",
            (series_id, n),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------- SEC ----------------

def upsert_sec_filing(filing: Dict[str, Any], holdings: Optional[List[Dict[str, Any]]] = None) -> None:
    conn = get_conn()
    with _DB_LOCK:
        conn.execute(
            """INSERT OR IGNORE INTO sec_filings
               (cik, institution, form_type, filed_at, period, accession_no, url, summary_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (filing.get("cik"), filing.get("institution"), filing.get("form_type"),
             filing.get("filed_at"), filing.get("period"), filing.get("accession_no"),
             filing.get("url"), json.dumps(filing.get("summary") or {})),
        )
        if holdings:
            for h in holdings:
                conn.execute(
                    """INSERT INTO sec_holdings
                       (accession_no, institution, period, ticker, name, cusip,
                        shares, value_usd, pct_portfolio)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (filing.get("accession_no"), filing.get("institution"), filing.get("period"),
                     h.get("ticker"), h.get("name"), h.get("cusip"),
                     h.get("shares"), h.get("value_usd"), h.get("pct_portfolio")),
                )
        conn.commit()


def query_sec_filings(institution: Optional[str] = None, form_type: Optional[str] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    sql = "SELECT * FROM sec_filings WHERE 1=1"
    args: List[Any] = []
    if institution:
        sql += " AND institution = ?"
        args.append(institution)
    if form_type:
        sql += " AND form_type = ?"
        args.append(form_type)
    sql += " ORDER BY filed_at DESC LIMIT ?"
    args.append(int(limit))
    with _DB_LOCK:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def latest_holdings_for(institution: str, top_n: int = 25) -> List[Dict[str, Any]]:
    conn = get_conn()
    with _DB_LOCK:
        period_row = conn.execute(
            "SELECT MAX(period) AS p FROM sec_holdings WHERE institution=?",
            (institution,),
        ).fetchone()
        if not period_row or not period_row["p"]:
            return []
        rows = conn.execute(
            """SELECT * FROM sec_holdings
               WHERE institution=? AND period=?
               ORDER BY value_usd DESC LIMIT ?""",
            (institution, period_row["p"], top_n),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------- CRYPTO MARKET ----------------

def upsert_crypto_market(snapshot: Dict[str, Any]) -> None:
    conn = get_conn()
    with _DB_LOCK:
        conn.execute(
            """INSERT OR REPLACE INTO crypto_market
               (snapshot_at, total_mcap_usd, total_vol_usd, btc_dominance, eth_dominance,
                fear_greed, fear_greed_label, payload_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (now_utc_iso(),
             snapshot.get("total_mcap_usd"), snapshot.get("total_vol_usd"),
             snapshot.get("btc_dominance"), snapshot.get("eth_dominance"),
             snapshot.get("fear_greed"), snapshot.get("fear_greed_label"),
             json.dumps(snapshot)),
        )
        conn.commit()


def latest_crypto_market() -> Optional[Dict[str, Any]]:
    conn = get_conn()
    with _DB_LOCK:
        row = conn.execute("SELECT * FROM crypto_market ORDER BY snapshot_at DESC LIMIT 1").fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["payload"] = json.loads(d.get("payload_json") or "{}")
    except Exception:
        d["payload"] = {}
    return d


# ---------------- KV ----------------

def kv_set(key: str, value: Any) -> None:
    conn = get_conn()
    with _DB_LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?,?,?)",
            (key, json.dumps(value), now_utc_iso()),
        )
        conn.commit()


def kv_get(key: str, default: Any = None) -> Any:
    conn = get_conn()
    with _DB_LOCK:
        row = conn.execute("SELECT value FROM kv_store WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default


# ---------------- CHAT ----------------

def append_chat(session_id: str, role: str, content: str, tool_name: Optional[str] = None) -> None:
    conn = get_conn()
    with _DB_LOCK:
        conn.execute(
            """INSERT INTO chat_history (session_id, role, content, tool_name, created_at)
               VALUES (?,?,?,?,?)""",
            (session_id, role, content, tool_name, now_utc_iso()),
        )
        conn.commit()


def get_chat(session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    conn = get_conn()
    with _DB_LOCK:
        rows = conn.execute(
            "SELECT * FROM chat_history WHERE session_id=? ORDER BY id ASC LIMIT ?",
            (session_id, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_chat(session_id: str) -> int:
    conn = get_conn()
    with _DB_LOCK:
        cur = conn.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
        conn.commit()
        return cur.rowcount


# ---------------- HEALTH ----------------

def db_health() -> Dict[str, Any]:
    conn = get_conn()
    with _DB_LOCK:
        n_news = conn.execute("SELECT COUNT(*) c FROM news").fetchone()["c"]
        n_mkt  = conn.execute("SELECT COUNT(*) c FROM market_snapshots").fetchone()["c"]
        n_sig  = conn.execute("SELECT COUNT(*) c FROM signals").fetchone()["c"]
        n_econ = conn.execute("SELECT COUNT(*) c FROM econ_events").fetchone()["c"]
        n_sec  = conn.execute("SELECT COUNT(*) c FROM sec_filings").fetchone()["c"]
        last_news = conn.execute("SELECT MAX(published_at) m FROM news").fetchone()["m"]
    return {
        "db_path": get_db_path(),
        "news_count": n_news,
        "market_count": n_mkt,
        "signal_count": n_sig,
        "econ_count": n_econ,
        "sec_filings": n_sec,
        "latest_news_at": last_news,
    }
