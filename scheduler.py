"""
APScheduler background jobs for the AI Macro Agent.

The scheduler is started once per process via start_scheduler().
It is idempotent — calling start_scheduler() multiple times is safe.
All jobs are wrapped in try/except so a failure in one source doesn't
kill the others.
"""
from __future__ import annotations

import os
import logging
import threading
import datetime as dt
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

import data_layer as dl
import sources as src

log = logging.getLogger("analizator.scheduler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

_SCHED: Optional[BackgroundScheduler] = None
_LOCK = threading.Lock()


def _get_secret(name: str, default: str = "") -> str:
    """Read secret from env first, then Streamlit secrets if available."""
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


# ---------------- jobs ----------------

def job_refresh_news() -> None:
    """Pull NewsAPI top headlines + everything for broad queries."""
    try:
        api_key = _get_secret("NEWSAPI_KEY")
        items = []
        if api_key:
            items += src.fetch_newsapi_top_headlines(api_key)
            for q in ("markets OR economy OR Fed OR inflation",
                      "Bitcoin OR Ethereum OR crypto",
                      "geopolitics OR war OR election",
                      "earnings OR Nvidia OR Apple OR Tesla"):
                items += src.fetch_newsapi_everything(api_key, q, page_size=30, hours_back=12)
        n = dl.upsert_news_items(items)
        log.info("news job: %d new items (NewsAPI)", n)
    except Exception as e:
        log.exception("job_refresh_news failed: %s", e)


def job_refresh_rss() -> None:
    try:
        items = src.fetch_all_rss(max_per_feed=30)
        n = dl.upsert_news_items(items)
        log.info("rss job: %d new items from %d feeds", n, len(src.RSS_FEEDS))
    except Exception as e:
        log.exception("job_refresh_rss failed: %s", e)


def job_refresh_crypto_market() -> None:
    try:
        glob = src.fetch_coingecko_global() or {}
        fng = src.fetch_crypto_fear_greed() or {}
        snapshot = {
            "total_mcap_usd": glob.get("total_mcap_usd"),
            "total_vol_usd":  glob.get("total_vol_usd"),
            "btc_dominance":  glob.get("btc_dominance"),
            "eth_dominance":  glob.get("eth_dominance"),
            "fear_greed":     fng.get("value"),
            "fear_greed_label": fng.get("label"),
            "trending":       src.fetch_coingecko_trending()[:10],
            "top_coins":      src.fetch_coingecko_top(50),
        }
        dl.upsert_crypto_market(snapshot)
        log.info("crypto market job: mcap=%s f&g=%s", snapshot.get("total_mcap_usd"), snapshot.get("fear_greed"))
    except Exception as e:
        log.exception("job_refresh_crypto_market failed: %s", e)


def job_refresh_fred() -> None:
    try:
        api_key = _get_secret("FRED_API_KEY")
        if not api_key:
            return
        total = 0
        for sid in src.FRED_KEY_SERIES.keys():
            obs = src.fetch_fred_observations(api_key, sid, limit=24)
            total += dl.upsert_fred_series(sid, obs)
        log.info("fred job: %d observations stored", total)
    except Exception as e:
        log.exception("job_refresh_fred failed: %s", e)


def job_refresh_finnhub_calendars() -> None:
    try:
        api_key = _get_secret("FINNHUB_API_KEY")
        if not api_key:
            return
        # Earnings
        earnings = src.fetch_finnhub_earnings_calendar(api_key, days_ahead=14)
        events = []
        for e in earnings:
            d = e.get("date")
            if not d:
                continue
            events.append({
                "event_time": f"{d}T13:30:00Z",
                "country": "US",
                "title": f"Earnings: {e.get('symbol','')} ({e.get('hour','') or ''})",
                "importance": "medium",
                "actual": str(e.get("epsActual") or ""),
                "forecast": str(e.get("epsEstimate") or ""),
                "previous": "",
                "source": "Finnhub",
                "extra": e,
            })
        # Economic
        econ = src.fetch_finnhub_economic_calendar(api_key)
        for e in econ:
            t = e.get("time")
            if not t:
                continue
            events.append({
                "event_time": t.replace(" ", "T") + "Z" if "T" not in t else t,
                "country": e.get("country") or "",
                "title": e.get("event") or "",
                "importance": {"low":"low","medium":"medium","high":"high"}.get(str(e.get("impact","")).lower(), "low"),
                "actual": str(e.get("actual") or ""),
                "forecast": str(e.get("estimate") or ""),
                "previous": str(e.get("prev") or ""),
                "source": "Finnhub",
                "extra": e,
            })
        n = dl.upsert_econ_events(events)
        log.info("finnhub calendars job: %d events", n)
    except Exception as e:
        log.exception("job_refresh_finnhub_calendars failed: %s", e)


def job_refresh_sec() -> None:
    try:
        filings = src.fetch_sec_all_institutions()
        for f in filings:
            dl.upsert_sec_filing(f)
        log.info("sec job: %d filings processed", len(filings))
    except Exception as e:
        log.exception("job_refresh_sec failed: %s", e)


def job_refresh_quotes() -> None:
    """Live quotes for the assets shown in the live ticker."""
    try:
        yahoo_syms = ["GC=F", "SI=F", "EURUSD=X", "GBPUSD=X", "JPY=X",
                      "^GSPC", "^NDX", "BTC-USD", "ETH-USD"]
        for s in yahoo_syms:
            q = src.fetch_yahoo_quote(s)
            if q and q.get("price") is not None:
                prev = q.get("prev_close") or 0
                chg = ((q["price"] - prev) / prev * 100.0) if prev else None
                dl.upsert_market_snapshot(s, "yahoo", float(q["price"]), chg)
        binance_syms = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
        for s in binance_syms:
            q = src.fetch_binance_24h(s)
            if q:
                dl.upsert_market_snapshot(s, "binance", q["price"], q.get("change_pct"), q.get("volume"))
        log.info("quotes job: refreshed %d symbols", len(yahoo_syms) + len(binance_syms))
    except Exception as e:
        log.exception("job_refresh_quotes failed: %s", e)


def job_prune() -> None:
    try:
        n = dl.prune_news(retention_days=90)
        log.info("prune job: removed %d old news rows", n)
    except Exception as e:
        log.exception("job_prune failed: %s", e)


# ---------------- entrypoint ----------------

def start_scheduler(run_now: bool = True) -> BackgroundScheduler:
    global _SCHED
    with _LOCK:
        if _SCHED is not None and _SCHED.running:
            return _SCHED
        sched = BackgroundScheduler(timezone="UTC", job_defaults={"coalesce": True, "max_instances": 1})

        sched.add_job(job_refresh_quotes, IntervalTrigger(minutes=5), id="quotes", replace_existing=True)
        sched.add_job(job_refresh_rss, IntervalTrigger(minutes=20), id="rss", replace_existing=True)
        sched.add_job(job_refresh_news, IntervalTrigger(minutes=30), id="newsapi", replace_existing=True)
        sched.add_job(job_refresh_crypto_market, IntervalTrigger(minutes=15), id="crypto_market", replace_existing=True)
        sched.add_job(job_refresh_finnhub_calendars, IntervalTrigger(hours=2), id="finnhub_cal", replace_existing=True)
        sched.add_job(job_refresh_fred, IntervalTrigger(hours=6), id="fred", replace_existing=True)
        sched.add_job(job_refresh_sec, IntervalTrigger(hours=6), id="sec", replace_existing=True)
        sched.add_job(job_prune, CronTrigger(hour=3, minute=0), id="prune", replace_existing=True)

        sched.start()
        _SCHED = sched
        log.info("scheduler started, db=%s", dl.get_db_path())

        if run_now:
            # Kick off the cheap ones immediately so the UI has data on first load
            for fn in (job_refresh_quotes, job_refresh_rss, job_refresh_crypto_market):
                try:
                    threading.Thread(target=fn, daemon=True).start()
                except Exception:
                    pass
        return _SCHED


def scheduler_status() -> dict:
    s = _SCHED
    if s is None:
        return {"running": False, "jobs": []}
    return {
        "running": s.running,
        "jobs": [
            {"id": j.id, "next_run": str(j.next_run_time)}
            for j in s.get_jobs()
        ],
    }
