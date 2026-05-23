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

CREATE TABLE IF NOT EXISTS symbols_universe (
    symbol        TEXT PRIMARY KEY,
    company_name  TEXT NOT NULL,
    sector        TEXT,
    industry      TEXT,
    asset_type    TEXT,    -- 'equity' | 'index' | 'commodity' | 'fx_index' | 'vol_index' | 'rate_index'
    is_core       INTEGER DEFAULT 0,   -- 1 = always refresh
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_universe_sector ON symbols_universe(sector);
CREATE INDEX IF NOT EXISTS idx_universe_core ON symbols_universe(is_core);
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


# ---------------- SYMBOLS UNIVERSE ----------------

def seed_universe(force: bool = False) -> int:
    """Populate symbols_universe table from symbol_universe.py.

    Idempotent — uses INSERT OR REPLACE so re-running refreshes metadata
    without duplicating rows. If `force=False` and the table is already
    non-empty, this is a no-op (skips the work). Returns rows inserted.
    """
    try:
        import symbol_universe as su
    except Exception:
        return 0
    conn = get_conn()
    with _DB_LOCK:
        if not force:
            existing = conn.execute("SELECT COUNT(*) c FROM symbols_universe").fetchone()["c"]
            if existing >= len(su.SP500_COMPANIES) + len(su.MAJOR_INDICES) - 5:
                return 0  # already seeded
        now = now_utc_iso()
        core_set = set(su.CORE_WATCHLIST)
        rows = []
        for r in su.SP500_COMPANIES:
            rows.append((r["symbol"], r["name"], r["sector"], r["industry"],
                         "equity", 1 if r["symbol"] in core_set else 0, now))
        for r in su.MAJOR_INDICES:
            rows.append((r["symbol"], r["name"], "Index/Commodity", r["type"],
                         r["type"], 1, now))  # all indices are core
        # ETFs in core watchlist that aren't S&P members - register them too
        sp500_set = {x["symbol"] for x in su.SP500_COMPANIES}
        idx_set = {x["symbol"] for x in su.MAJOR_INDICES}
        for sym in su.CORE_WATCHLIST:
            if sym in sp500_set or sym in idx_set:
                continue
            rows.append((sym, sym, "ETF/Other", "ETF", "etf", 1, now))
        conn.executemany(
            """INSERT OR REPLACE INTO symbols_universe
               (symbol, company_name, sector, industry, asset_type, is_core, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
        return len(rows)


def universe_all(asset_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    sql = "SELECT * FROM symbols_universe"
    params: List[Any] = []
    if asset_types:
        placeholders = ",".join("?" for _ in asset_types)
        sql += f" WHERE asset_type IN ({placeholders})"
        params.extend(asset_types)
    sql += " ORDER BY sector, symbol"
    with _DB_LOCK:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def universe_by_sector(sector: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    with _DB_LOCK:
        rows = conn.execute(
            "SELECT * FROM symbols_universe WHERE sector=? ORDER BY symbol",
            (sector,),
        ).fetchall()
    return [dict(r) for r in rows]


def universe_core() -> List[Dict[str, Any]]:
    conn = get_conn()
    with _DB_LOCK:
        rows = conn.execute(
            "SELECT * FROM symbols_universe WHERE is_core=1 ORDER BY asset_type, symbol"
        ).fetchall()
    return [dict(r) for r in rows]


def universe_meta(symbol: str) -> Optional[Dict[str, Any]]:
    if not symbol:
        return None
    conn = get_conn()
    with _DB_LOCK:
        row = conn.execute(
            "SELECT * FROM symbols_universe WHERE symbol=?", (symbol.upper(),)
        ).fetchone()
    return dict(row) if row else None


def universe_search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search by symbol prefix or company name substring."""
    q = (query or "").strip()
    if not q:
        return []
    conn = get_conn()
    like = f"%{q.lower()}%"
    sym_like = f"{q.upper()}%"
    with _DB_LOCK:
        rows = conn.execute(
            """SELECT *,
                  CASE
                    WHEN UPPER(symbol) = ? THEN 0
                    WHEN UPPER(symbol) LIKE ? THEN 1
                    WHEN LOWER(company_name) LIKE ? THEN 2
                    ELSE 3
                  END AS rank
               FROM symbols_universe
               WHERE UPPER(symbol) LIKE ? OR LOWER(company_name) LIKE ?
               ORDER BY rank, symbol
               LIMIT ?""",
            (q.upper(), sym_like, like, sym_like, like, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def universe_with_quotes(asset_types: Optional[List[str]] = None,
                          sector: Optional[str] = None,
                          query: Optional[str] = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
    """Join universe metadata with latest market snapshots.

    Returns rows with: symbol, company_name, sector, industry, asset_type,
    is_core, price, change_pct, volume, snapshot_at. price/change may be
    NULL if no snapshot is cached yet.
    """
    conn = get_conn()
    sql = (
        "SELECT u.symbol, u.company_name, u.sector, u.industry, u.asset_type, "
        "u.is_core, m.price, m.change_pct, m.volume, m.snapshot_at "
        "FROM symbols_universe u "
        "LEFT JOIN market_snapshots m ON u.symbol = m.symbol "
        "WHERE 1=1"
    )
    params: List[Any] = []
    if asset_types:
        sql += f" AND u.asset_type IN ({','.join('?' for _ in asset_types)})"
        params.extend(asset_types)
    if sector:
        sql += " AND u.sector = ?"
        params.append(sector)
    if query:
        q = f"%{query.lower()}%"
        sql += " AND (UPPER(u.symbol) LIKE ? OR LOWER(u.company_name) LIKE ?)"
        params.extend([f"{query.upper()}%", q])
    sql += " ORDER BY u.sector, u.symbol LIMIT ?"
    params.append(int(limit))
    with _DB_LOCK:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def top_movers(n: int = 10, direction: str = "up",
                asset_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Top N gainers (direction='up') or losers (direction='down') across
    the universe. Joins universe metadata. Excludes symbols with NULL
    change_pct or |change| > 50% (likely bad data / splits)."""
    conn = get_conn()
    order = "DESC" if direction.lower() == "up" else "ASC"
    sql = (
        "SELECT u.symbol, u.company_name, u.sector, u.asset_type, "
        "m.price, m.change_pct, m.snapshot_at "
        "FROM symbols_universe u "
        "JOIN market_snapshots m ON u.symbol = m.symbol "
        "WHERE m.change_pct IS NOT NULL "
        "AND ABS(m.change_pct) < 50"
    )
    params: List[Any] = []
    if asset_types:
        sql += f" AND u.asset_type IN ({','.join('?' for _ in asset_types)})"
        params.extend(asset_types)
    sql += f" ORDER BY m.change_pct {order} LIMIT ?"
    params.append(int(n))
    with _DB_LOCK:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def sector_performance() -> List[Dict[str, Any]]:
    """Average % change per GICS sector, plus the corresponding sector
    ETF's change for comparison. Returns rows sorted by avg_change DESC."""
    conn = get_conn()
    sql = """
    SELECT u.sector,
           AVG(m.change_pct) AS avg_change,
           COUNT(m.change_pct) AS n_with_data,
           COUNT(u.symbol) AS n_total,
           SUM(CASE WHEN m.change_pct > 0 THEN 1 ELSE 0 END) AS n_up,
           SUM(CASE WHEN m.change_pct < 0 THEN 1 ELSE 0 END) AS n_down
    FROM symbols_universe u
    LEFT JOIN market_snapshots m ON u.symbol = m.symbol
    WHERE u.asset_type = 'equity'
    GROUP BY u.sector
    ORDER BY avg_change DESC NULLS LAST
    """
    with _DB_LOCK:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def universe_sector_counts() -> Dict[str, int]:
    conn = get_conn()
    with _DB_LOCK:
        rows = conn.execute(
            "SELECT sector, COUNT(*) c FROM symbols_universe GROUP BY sector"
        ).fetchall()
    return {r["sector"]: r["c"] for r in rows}


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
