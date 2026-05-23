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
    """Live quotes for the header tickers + core watchlist.

    Runs frequently (every 5 min). Includes the original header symbols
    (gold, silver, FX, indices, BTC, ETH) plus the universe CORE_WATCHLIST
    (sector ETFs, mega caps, broad ETFs).
    """
    try:
        # Original header symbols (kept for backward compat)
        header_syms = ["GC=F", "SI=F", "EURUSD=X", "GBPUSD=X", "JPY=X",
                       "^GSPC", "^NDX", "BTC-USD", "ETH-USD"]
        # Core watchlist from the universe (ETFs, indices, mega caps)
        core_syms: list = []
        try:
            dl.seed_universe()  # idempotent
            core_syms = [r["symbol"] for r in dl.universe_core()]
        except Exception:
            log.exception("could not read core watchlist")

        all_yahoo = sorted(set(header_syms) | set(core_syms))
        results = src.fetch_yahoo_batch(all_yahoo, max_workers=8)
        ok = 0
        for sym, q in results.items():
            if "error" in q:
                continue
            price = q.get("price")
            if price is None:
                continue
            dl.upsert_market_snapshot(sym, "yahoo", float(price),
                                       q.get("change_pct"), q.get("volume"))
            ok += 1

        binance_syms = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
        for s in binance_syms:
            q = src.fetch_binance_24h(s)
            if q:
                dl.upsert_market_snapshot(s, "binance", q["price"], q.get("change_pct"), q.get("volume"))
        log.info("quotes job: refreshed %d/%d yahoo + %d binance",
                 ok, len(all_yahoo), len(binance_syms))
    except Exception as e:
        log.exception("job_refresh_quotes failed: %s", e)


def job_refresh_universe_full() -> None:
    """Refresh prices for the FULL S&P 500 universe (500+ symbols).

    Runs less frequently (every 30 min) to avoid Yahoo rate-limiting.
    Uses batch fetcher with parallelism. Skips the core symbols since
    they're already refreshed by job_refresh_quotes every 5 min.
    """
    try:
        dl.seed_universe()  # idempotent
        all_rows = dl.universe_all()
        core_set = {r["symbol"] for r in dl.universe_core()}
        non_core = [r["symbol"] for r in all_rows
                    if r["symbol"] not in core_set and r["asset_type"] == "equity"]
        if not non_core:
            log.info("universe refresh: no non-core equities to refresh")
            return
        log.info("universe refresh: starting %d symbols", len(non_core))
        t0 = dt.datetime.utcnow()
        results = src.fetch_yahoo_batch(non_core, max_workers=10)
        ok = 0
        for sym, q in results.items():
            if "error" in q:
                continue
            price = q.get("price")
            if price is None:
                continue
            dl.upsert_market_snapshot(sym, "yahoo", float(price),
                                       q.get("change_pct"), q.get("volume"))
            ok += 1
        elapsed = (dt.datetime.utcnow() - t0).total_seconds()
        log.info("universe refresh: done %d/%d ok in %.0fs",
                 ok, len(non_core), elapsed)
    except Exception as e:
        log.exception("job_refresh_universe_full failed: %s", e)


def job_prune() -> None:
    try:
        n = dl.prune_news(retention_days=90)
        log.info("prune job: removed %d old news rows", n)
    except Exception as e:
        log.exception("job_prune failed: %s", e)


def job_generate_deep_brief() -> None:
    """Run the multi-agent deep brief and persist it to kv_store cache.

    Opt-in: only registered when ENABLE_SCHEDULED_BRIEF=true. Default
    schedule is configurable via DEEP_BRIEF_CRON (cron-style 'h m' or full
    cron expression). The brief is cached so the user sees a ready-made
    briefing when they open the app in the morning.
    """
    try:
        import orchestrator as orch
        log.info("scheduled deep brief: starting")
        t0 = dt.datetime.utcnow()
        brief = orch.run_deep_brief(user_query="")
        elapsed = (dt.datetime.utcnow() - t0).total_seconds()
        n_agents = len(brief.get("subagents", {}))
        log.info("scheduled deep brief: done in %.0fs, %d agents", elapsed, n_agents)
    except Exception as e:
        log.exception("job_generate_deep_brief failed: %s", e)


def _parse_brief_cron() -> Optional[CronTrigger]:
    """Read DEEP_BRIEF_CRON from env. Returns None on parse failure.

    Accepts either a 5-field cron expression ("0 7 * * 1-5") or a simple
    "HH:MM" form (defaults to weekdays).
    """
    raw = _get_secret("DEEP_BRIEF_CRON", "0 7 * * 1-5")
    if not raw:
        return None
    raw = raw.strip()
    # Simple "HH:MM" form -> weekdays at that time UTC
    if ":" in raw and len(raw.split()) == 1:
        try:
            hh, mm = raw.split(":")
            return CronTrigger(hour=int(hh), minute=int(mm), day_of_week="mon-fri")
        except Exception:
            log.warning("DEEP_BRIEF_CRON invalid HH:MM '%s'", raw)
            return None
    # 5-field cron
    try:
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError("expected 5 cron fields")
        return CronTrigger(minute=parts[0], hour=parts[1], day=parts[2],
                           month=parts[3], day_of_week=parts[4])
    except Exception:
        log.warning("DEEP_BRIEF_CRON invalid cron '%s'", raw)
        return None


# ---------------- entrypoint ----------------

def start_scheduler(run_now: bool = True) -> BackgroundScheduler:
    global _SCHED
    with _LOCK:
        if _SCHED is not None and _SCHED.running:
            return _SCHED
        sched = BackgroundScheduler(timezone="UTC", job_defaults={"coalesce": True, "max_instances": 1})

        sched.add_job(job_refresh_quotes, IntervalTrigger(minutes=5), id="quotes", replace_existing=True)
        sched.add_job(job_refresh_universe_full, IntervalTrigger(minutes=30),
                      id="universe_full", replace_existing=True)
        sched.add_job(job_refresh_rss, IntervalTrigger(minutes=20), id="rss", replace_existing=True)
        sched.add_job(job_refresh_news, IntervalTrigger(minutes=30), id="newsapi", replace_existing=True)
        sched.add_job(job_refresh_crypto_market, IntervalTrigger(minutes=15), id="crypto_market", replace_existing=True)
        sched.add_job(job_refresh_finnhub_calendars, IntervalTrigger(hours=2), id="finnhub_cal", replace_existing=True)
        sched.add_job(job_refresh_fred, IntervalTrigger(hours=6), id="fred", replace_existing=True)
        sched.add_job(job_refresh_sec, IntervalTrigger(hours=6), id="sec", replace_existing=True)
        sched.add_job(job_prune, CronTrigger(hour=3, minute=0), id="prune", replace_existing=True)

        # Optional: scheduled deep brief generation (opt-in via env var)
        if _get_secret("ENABLE_SCHEDULED_BRIEF", "").lower() in ("1", "true", "yes", "on"):
            trig = _parse_brief_cron()
            if trig is not None:
                sched.add_job(job_generate_deep_brief, trig,
                              id="deep_brief", replace_existing=True)
                log.info("scheduled deep brief enabled with trigger=%s", trig)
            else:
                log.warning("ENABLE_SCHEDULED_BRIEF=true but cron parse failed; not scheduled")

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
