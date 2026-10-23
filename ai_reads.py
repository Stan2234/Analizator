"""What every AI read on the site shares: the layout of a forecast note, the numbers its
probabilities are anchored on, and the record of every call it makes.

A note that only summarises the page adds nothing a reader could not see. So each read
takes a view for the next one to three months and puts probabilities on three scenarios.
Those probabilities are the model's judgment, and nothing on the site has tested them,
which is why two things sit next to them:

- forecast_inputs(): how often the asset has risen over the same horizon from a similar
  trend in its own history, and how far its volatility says it can travel. The note is
  told to start from these and to say when and why it departs from them.
- record_call(): every view is stored with the price at the time, so the calls can be
  scored once enough of them have matured, instead of being trusted on their tone.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("analizator.ai_reads")

VIEW_WORDS = ["Bullish", "Mildly bullish", "Neutral", "Mildly bearish", "Bearish"]

DISCLAIMER = (
    "The view and the scenario probabilities are the AI analyst's judgment, anchored on the "
    "asset's own history and volatility. They are not the output of a tested forecasting model, "
    "every call is recorded so its accuracy can be checked, and none of this is investment advice."
)


def forecast_style(subject: str, horizon: str = "the next 1-3 months",
                   view_scale: Optional[List[str]] = None, extra: str = "") -> str:
    """The layout every AI read follows: a call, the reasons, three scenarios with
    probabilities, the levels that matter and what would change the call."""
    scale = " | ".join(view_scale or VIEW_WORDS)
    return (
        f"Write a research note on {subject} for investors, in this layout (markdown):\n\n"
        "**<Headline: your call in one sentence, with the number that carries it>**\n\n"
        f"**View for {horizon}: <one of: {scale}>** · Confidence: <low | medium | high>\n\n"
        "- three bullets: the findings that drive the view, each with a number and what it is "
        "compared with (its own history, its peers, or what was expected)\n\n"
        "### <a question heading about the main driver>\n3-5 sentences.\n"
        "### <a question heading about the second driver, or the main risk>\n3-5 sentences.\n"
        "### <a question heading about positioning, flows or the price trend, whichever the data "
        "covers>\n2-4 sentences.\n\n"
        f"### Scenarios for {horizon}\n"
        "| Scenario | Probability | Where it could be | What has to happen | First sign to watch |\n"
        "|---|---|---|---|---|\n"
        "| Bull | xx% | a level or range | ... | ... |\n"
        "| Base | xx% | ... | ... | ... |\n"
        "| Bear | xx% | ... | ... | ... |\n"
        "One line under the table: what the probabilities are anchored on.\n\n"
        "### Levels to watch\nTwo to four levels taken from the data (moving averages, the "
        "52-week high and low, the record, round numbers near the price) and why each matters.\n\n"
        "### What would change the view\nTwo or three concrete triggers, dated where the "
        "calendar allows.\n\n"
        "**What this means:** two sentences on what the view implies for someone who already "
        "holds it and for someone considering it, in general terms. No position sizes, no "
        "personal advice.\n\n"
        + (extra + "\n\n" if extra else "")
        + "Rules:\n"
        "- 600-900 words. Every claim carries a number and what it is compared with; never "
        "'strong' or 'weak' alone. Short sentences. Explain any jargon once in brackets.\n"
        "- The probabilities are your judgment, but anchored: start from the historical base "
        "rates and volatility ranges in forecast_inputs when they are given, move away from "
        "them only for a reason you state, and make the three add up to 100. Scenario levels "
        "must fit the volatility ranges; a move beyond the 2-sigma range needs a stated reason.\n"
        "- Never claim certainty. Use only the figures given; add no prices, events or quotes "
        "from memory. If the view needs something that is missing, say so in one line at the "
        "end under **Not in the data:**.\n"
        "- Finish with this machine-readable line on its own, with nothing after it:\n"
        "VIEW|<view>|<confidence>|bull=<n>|base=<n>|bear=<n>"
    )


# --------------------------------------------------------------------------
# The numbers the probabilities are anchored on
# --------------------------------------------------------------------------

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def forecast_inputs(close: pd.Series, horizons: Tuple[int, ...] = (21, 63),
                    periods_per_year: int = 252, name: str = "",
                    levels: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Base rates and volatility ranges for a daily price series.

    - base_rates: over the asset's own history, from days in the same trend state as today
      (above or below its 50- and 200-day averages), how often the price was higher
      `h` days later, the median move and the 10th-90th percentile spread. Windows overlap,
      so the effective sample is smaller than the count shown, which is reported anyway.
    - ranges: today's price moved by one and two standard deviations of recent volatility
      over each horizon, with no drift assumed.
    - touch: the chance of trading through each level within each horizon under the same
      volatility (reflection principle), a measure of reach, not of direction.
    """
    s = pd.to_numeric(close, errors="coerce").dropna()
    s = s[s > 0]
    out: Dict[str, Any] = {"asset": name, "available": False}
    if len(s) < 260:
        out["note"] = f"only {len(s)} daily prices, too few for base rates"
        return out
    last = float(s.iloc[-1])
    lr = np.log(s).diff().dropna()
    vol_d = float(lr.tail(63).std())
    out.update({
        "available": True, "last_price": last, "as_of": str(s.index[-1])[:10],
        "history_from": str(s.index[0])[:10], "n_days": int(len(s)),
        "volatility_annual_pct": round(vol_d * math.sqrt(periods_per_year) * 100, 1),
    })
    ma50, ma200 = s.rolling(50).mean(), s.rolling(200).mean()
    state = pd.Series(np.where(s > ma200, "above 200d", "below 200d"), index=s.index) + \
        pd.Series(np.where(s > ma50, ", above 50d", ", below 50d"), index=s.index)
    state[ma200.isna()] = None
    now_state = state.iloc[-1]
    out["trend_state_now"] = now_state

    out["base_rates"], out["ranges"] = {}, {}
    for h in horizons:
        fwd = np.log(s.shift(-h) / s)
        df = pd.DataFrame({"fwd": fwd, "state": state}).dropna()
        same = df[df["state"] == now_state]["fwd"]
        allr = df["fwd"]
        label = f"{h} trading days" if periods_per_year == 252 else f"{h} days"

        def summ(x: pd.Series) -> Dict[str, Any]:
            if len(x) < 30:
                return {"n": int(len(x)), "note": "too few cases"}
            return {"n": int(len(x)), "share_higher_pct": round(float((x > 0).mean() * 100), 1),
                    "median_move_pct": round(float(np.expm1(x.median()) * 100), 1),
                    "p10_move_pct": round(float(np.expm1(x.quantile(0.1)) * 100), 1),
                    "p90_move_pct": round(float(np.expm1(x.quantile(0.9)) * 100), 1)}
        out["base_rates"][label] = {"same_trend_state": summ(same), "all_days": summ(allr)}

        sig = vol_d * math.sqrt(h)
        out["ranges"][label] = {
            "one_sigma_68pct": [round(last * math.exp(-sig), 4), round(last * math.exp(sig), 4)],
            "two_sigma_95pct": [round(last * math.exp(-2 * sig), 4), round(last * math.exp(2 * sig), 4)],
        }
        if levels:
            touch = {}
            for lname, lv in levels.items():
                try:
                    lv = float(lv)
                except (TypeError, ValueError):
                    continue
                if lv <= 0 or not math.isfinite(lv):
                    continue
                d = abs(math.log(lv / last))
                touch[f"{lname} ({lv:,.4g})"] = round(min(1.0, 2 * (1 - _phi(d / sig))) * 100, 1)
            out["ranges"][label]["chance_to_trade_through_pct"] = touch

    out["levels"] = {"50-day average": round(float(ma50.iloc[-1]), 4),
                     "200-day average": round(float(ma200.iloc[-1]), 4),
                     "52-week high": round(float(s.tail(periods_per_year).max()), 4),
                     "52-week low": round(float(s.tail(periods_per_year).min()), 4),
                     "high in this history": round(float(s.max()), 4)}
    out["how_to_read"] = ("base_rates: from days in today's trend state, how often the price was "
                          "higher after the horizon, in the asset's own history (overlapping "
                          "windows). ranges: where one and two standard deviations of recent "
                          "volatility reach, no drift assumed. Neither is a forecast on its own.")
    return out


def default_levels(close: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < 200:
        return {}
    return {"50-day average": float(s.tail(50).mean()), "200-day average": float(s.tail(200).mean()),
            "52-week high": float(s.tail(periods_per_year).max()),
            "52-week low": float(s.tail(periods_per_year).min())}


# --------------------------------------------------------------------------
# The record of every call
# --------------------------------------------------------------------------

_VIEW_RE = re.compile(r"^\s*VIEW\|([^|]+)\|([^|]+)\|bull=(\d+)\|base=(\d+)\|bear=(\d+)\s*$", re.M)


def split_view(text: str) -> Tuple[str, Dict[str, Any]]:
    """The note without its machine line, and the call parsed from it."""
    m = _VIEW_RE.search(text or "")
    if not m:
        return (text or "").strip(), {}
    meta = {"view": m.group(1).strip(), "confidence": m.group(2).strip(),
            "bull": int(m.group(3)), "base": int(m.group(4)), "bear": int(m.group(5))}
    return (text[:m.start()] + text[m.end():]).strip(), meta


def record_call(panel: str, focus: str, meta: Dict[str, Any], price: Optional[float]) -> None:
    """Keep every call with the price it was made at, so it can be scored later."""
    if not meta:
        return
    try:
        import data_layer as dl
        calls = dl.kv_get("ai_calls") or []
        calls.append({"ts": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"), "panel": panel,
                      "focus": focus, "price": price, **meta})
        dl.kv_set("ai_calls", calls[-5000:])
    except Exception:
        log.exception("could not record the call")


def save_read(panel: str, focus: str, text: str, meta: Dict[str, Any]) -> None:
    """The latest read for this panel and subject, shown to the next visitor."""
    try:
        import data_layer as dl
        dl.kv_set(f"ai_read:{panel}:{focus}", {"text": text, "meta": meta,
                                               "ts": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")})
    except Exception:
        log.exception("could not store the read")


def load_read(panel: str, focus: str) -> Optional[Dict[str, Any]]:
    try:
        import data_layer as dl
        return dl.kv_get(f"ai_read:{panel}:{focus}")
    except Exception:
        return None


# --------------------------------------------------------------------------
# The free market read, written three times a day by the scheduler
# --------------------------------------------------------------------------

GLOBAL_FOCUS = "Global macro view"
_FRED = [("DFF", "Fed funds effective", "%"), ("DGS2", "2y Treasury", "%"), ("DGS10", "10y Treasury", "%"),
         ("T10Y2Y", "10y-2y curve", "pp"), ("UNRATE", "Unemployment rate", "%"),
         ("ICSA", "Initial jobless claims", ""), ("BAMLH0A0HYM2", "High-yield spread", "pp"),
         ("VIXCLS", "VIX", ""), ("DCOILWTICO", "WTI crude", "$")]


def global_news_read() -> Optional[str]:
    """The market-wide read everyone sees on the News tab, written from what the database
    already holds: the last day's headlines, macro series, sector breadth, the week's
    calendar, and base rates for the S&P 500."""
    import agent as ai_agent
    import data_layer as dl
    import sources as src

    if ai_agent.ai_outage():
        log.info("global read skipped: AI outage flag is set")
        return None
    ctx: Dict[str, Any] = {"as_of_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")}
    try:
        rows = dl.query_news(since_hours=24, limit=400) or []
        ctx["headlines_last_24h"] = [{"source": r.get("source"), "title": r.get("title"),
                                      "topic": r.get("category"), "at": r.get("published_at")}
                                     for r in rows[:150]]
    except Exception:
        ctx["headlines_last_24h"] = []
    macro = {}
    for sid, label, unit in _FRED:
        try:
            obs = dl.latest_fred(sid, n=2)
        except Exception:
            obs = []
        if obs:
            macro[label] = {"latest": f"{obs[0].get('value')}{unit}", "date": obs[0].get("obs_date"),
                            "previous": f"{obs[1].get('value')}{unit}" if len(obs) > 1 else None}
    ctx["macro"] = macro
    try:
        ctx["sector_breadth_today"] = dl.sector_performance()[:11]
    except Exception:
        pass
    try:
        now = dt.datetime.now(dt.timezone.utc)
        ctx["calendar_this_week"] = [
            {"when_utc": e.get("event_time"), "country": e.get("country"), "event": e.get("title"),
             "importance": e.get("importance"), "forecast": e.get("forecast"), "previous": e.get("previous")}
            for e in dl.query_econ_events(days_ahead=7, days_back=1)
            if not str(e.get("title") or "").lower().startswith("earnings")
            and str(e.get("importance") or "").lower() in ("high", "medium")][:40]
    except Exception:
        ctx["calendar_this_week"] = []
    spx = pd.Series(dtype=float)
    try:
        r = src._get("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC",
                     params={"range": "10y", "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"})
        res = r.json()["chart"]["result"][0]
        spx = pd.Series(res["indicators"]["quote"][0]["close"],
                        index=pd.to_datetime(res["timestamp"], unit="s")).dropna()
    except Exception:
        log.warning("global read: no S&P 500 history")
    if not spx.empty:
        ctx["forecast_inputs_sp500"] = forecast_inputs(spx, name="S&P 500", levels=default_levels(spx))

    system = (
        "You are the chief markets strategist at a research house writing the market note "
        "readers see first. You get the last 24 hours of headlines, the latest US macro "
        "series, today's sector breadth, this week's calendar with forecasts, and base rates "
        "and volatility ranges for the S&P 500.\n\n"
        "Pick the two or three stories that carry real consequence for markets and explain the "
        "mechanism, say how stocks, bonds, the dollar, gold and crypto are likely to respond "
        "and why, and put the view and the scenarios on the S&P 500 as the reference market. "
        "Name the headline behind each claim. If the news flow is thin, say so.\n\n"
        + forecast_style("global markets, with the S&P 500 as the reference", "the next 1-3 months"))
    text = ai_agent.complete(system, json.dumps(ctx, default=str), max_tokens=9000, effort="medium")
    clean, meta = split_view(text)
    price = (ctx.get("forecast_inputs_sp500") or {}).get("last_price")
    save_read("news", GLOBAL_FOCUS, clean, meta)
    dl.kv_set(f"news_forecast:{GLOBAL_FOCUS}",
              {"text": clean, "ts": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")})
    record_call("news", GLOBAL_FOCUS, meta, price)
    return clean
