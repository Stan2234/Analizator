"""The stock market as an asset class: indices, breadth, sectors, earnings and valuation.

Two background jobs do the heavy lifting once a day and keep the result in kv_store:

- breadth: daily closes of every S&P 500 member for two years (one batched Yahoo
  download), turned into the share of stocks above their 50- and 200-day averages,
  new 52-week highs and lows, and each stock's returns.
- valuation: Yahoo's quote summary for every member, summed bottom-up into the
  index's trailing and forward P/E, earnings yield, margins and growth, overall and
  by sector. Each run is appended to a history, so the P/E has a past from the day
  the job started recording.

Everything else (index and sector-ETF prices) is fetched on demand by the app.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("analizator.equities")

US_INDICES = [("^GSPC", "S&P 500"), ("^NDX", "Nasdaq-100"), ("^DJI", "Dow Jones"), ("^RUT", "Russell 2000")]
WORLD_INDICES = [
    ("^STOXX50E", "Euro Stoxx 50", "Europe"), ("^GDAXI", "DAX", "Europe"), ("^FTSE", "FTSE 100", "Europe"),
    ("^FCHI", "CAC 40", "Europe"), ("^N225", "Nikkei 225", "Asia"), ("^HSI", "Hang Seng", "Asia"),
    ("000300.SS", "CSI 300", "Asia"), ("^KS11", "KOSPI", "Asia"), ("^BSESN", "Sensex", "Asia"),
    ("^AXJO", "ASX 200", "Asia"), ("^GSPTSE", "TSX", "Americas"), ("^BVSP", "Bovespa", "Americas"),
]
SECTOR_ETFS = {
    "XLK": "Technology", "XLC": "Communication services", "XLY": "Consumer discretionary",
    "XLP": "Consumer staples", "XLE": "Energy", "XLF": "Financials", "XLV": "Health care",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real estate",
}
GICS_ETF = {"Information Technology": "XLK", "Communication Services": "XLC", "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP", "Energy": "XLE", "Financials": "XLF", "Health Care": "XLV",
            "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE"}
DEFENSIVE = {"XLP", "XLU", "XLV", "XLRE"}
CYCLICAL = {"XLY", "XLF", "XLI", "XLB", "XLE"}
INDEX_NAMES = {s: n for s, n in US_INDICES} | {s: n for s, n, _ in WORLD_INDICES}

HORIZONS = {"1 day": 1, "1 week": 7, "1 month": 30, "3 months": 91, "1 year": 365}


# ---------------- returns ----------------

def change_over(s: pd.Series, days: int) -> Optional[float]:
    """Percent change against the last close at least `days` calendar days back; 1 = previous close."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return None
    if days <= 1:
        prev = float(s.iloc[-2])
    else:
        past = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
        if past.empty:
            return None
        prev = float(past.iloc[-1])
    return (float(s.iloc[-1]) / prev - 1) * 100 if prev else None


def ytd(s: pd.Series) -> Optional[float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    prior = s[s.index.year < s.index[-1].year]
    return (float(s.iloc[-1]) / float(prior.iloc[-1]) - 1) * 100 if not prior.empty else None


def row_for(s: pd.Series) -> Dict[str, Any]:
    """Level, returns over the usual horizons, the trend and the distance from the 52-week high."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    out: Dict[str, Any] = {"last": float(s.iloc[-1]) if len(s) else None, "as_of": s.index[-1] if len(s) else None}
    for label, d in HORIZONS.items():
        out[label] = change_over(s, d)
    out["YTD"] = ytd(s)
    year = s[s.index >= s.index[-1] - pd.Timedelta(days=365)] if len(s) else s
    out["from 52w high"] = (float(s.iloc[-1]) / float(year.max()) - 1) * 100 if len(year) else None
    out["vs 200-day"] = (float(s.iloc[-1]) / float(s.tail(200).mean()) - 1) * 100 if len(s) >= 200 else None
    out["vs 50-day"] = (float(s.iloc[-1]) / float(s.tail(50).mean()) - 1) * 100 if len(s) >= 50 else None
    return out


def returns_table(closes: Dict[str, pd.Series], names: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for sym, s in closes.items():
        if s is None or len(s) < 2:
            continue
        rows.append({"Symbol": sym, "Name": names.get(sym, sym), **row_for(s)})
    return pd.DataFrame(rows)


# ---------------- breadth (job) ----------------

def _download(symbols: List[str], period: str = "2y") -> pd.DataFrame:
    """Daily closes, one column per symbol. Yahoo spells class shares with a dash (BRK-B)."""
    import yfinance as yf
    ysyms = [s.replace(".", "-") for s in symbols]
    frames = []
    for i in range(0, len(ysyms), 100):        # batches keep one bad answer from sinking the run
        chunk = ysyms[i:i + 100]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                                 progress=False, threads=True, group_by="column")
                close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
                frames.append(close)
                break
            except Exception as e:
                log.warning("breadth download batch %d failed (%s), retrying", i, e)
                time.sleep(5 * (attempt + 1))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out.columns = [str(c).replace("-", ".") if str(c).replace("-", ".") in symbols else str(c) for c in out.columns]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.dropna(how="all")


def breadth_from_closes(px: pd.DataFrame, sectors: Dict[str, str]) -> Dict[str, Any]:
    """Market breadth from member closes: how many stocks take part in the index's move."""
    px = px.loc[:, px.notna().sum() >= 210]
    if px.empty:
        return {}
    ma50, ma200 = px.rolling(50).mean(), px.rolling(200).mean()
    hi252, lo252 = px.rolling(252, min_periods=200).max(), px.rolling(252, min_periods=200).min()
    valid200 = ma200.notna() & px.notna()
    pct200 = ((px > ma200) & valid200).sum(axis=1) / valid200.sum(axis=1).replace(0, np.nan) * 100
    valid50 = ma50.notna() & px.notna()
    pct50 = ((px > ma50) & valid50).sum(axis=1) / valid50.sum(axis=1).replace(0, np.nan) * 100
    new_hi = ((px >= hi252) & hi252.notna()).sum(axis=1)
    new_lo = ((px <= lo252) & lo252.notna()).sum(axis=1)
    chg = px.pct_change(fill_method=None)
    adv, dec = (chg > 0).sum(axis=1), (chg < 0).sum(axis=1)
    keep = pct200.dropna().index[-260:]
    last = px.ffill().iloc[-1]
    stocks = []
    for sym in px.columns:
        s = px[sym].dropna()
        if len(s) < 30:
            continue
        stocks.append({"symbol": sym, "sector": sectors.get(sym, ""), "last": round(float(s.iloc[-1]), 2),
                       "1 day": change_over(s, 1), "1 week": change_over(s, 7), "1 month": change_over(s, 30),
                       "3 months": change_over(s, 91), "YTD": ytd(s),
                       "above_200": bool(len(s) >= 200 and s.iloc[-1] > s.tail(200).mean()),
                       "above_50": bool(s.iloc[-1] > s.tail(50).mean())})
    by_sector: Dict[str, Dict[str, float]] = {}
    for sec in sorted({x["sector"] for x in stocks if x["sector"]}):
        mem = [x for x in stocks if x["sector"] == sec]
        by_sector[sec] = {"n": len(mem), "pct_above_200": round(100 * sum(x["above_200"] for x in mem) / len(mem), 1),
                          "pct_above_50": round(100 * sum(x["above_50"] for x in mem) / len(mem), 1)}
    r = lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 2)
    return {
        "as_of": str(px.index[-1].date()), "n": int(last.notna().sum()),
        "pct_above_200": r(pct200.iloc[-1]), "pct_above_50": r(pct50.iloc[-1]),
        "new_highs": int(new_hi.iloc[-1]), "new_lows": int(new_lo.iloc[-1]),
        "advancers": int(adv.iloc[-1]), "decliners": int(dec.iloc[-1]),
        "history": [{"date": str(d.date()), "pct_above_200": r(pct200[d]), "pct_above_50": r(pct50[d]),
                     "new_highs": int(new_hi[d]), "new_lows": int(new_lo[d])} for d in keep],
        "by_sector": by_sector,
        "stocks": [{k: (r(v) if isinstance(v, float) else v) for k, v in x.items()} for x in stocks],
    }


def compute_breadth(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    sectors = {m["symbol"]: m.get("sector") or "" for m in members}
    px = _download(list(sectors))
    out = breadth_from_closes(px, sectors)
    if out:
        out["computed_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


# ---------------- valuation (job) ----------------

def _info(sym: str) -> Optional[Dict[str, Any]]:
    import yfinance as yf
    for attempt in range(3):
        try:
            i = yf.Ticker(sym.replace(".", "-")).info or {}
            if i.get("marketCap"):
                return i
            return None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def _earnings(i: Dict[str, Any]) -> Optional[tuple]:
    """(trailing, forward) earnings in dollars, losses included as negatives. Scaled by market
    value over price rather than share count, which for dual-class companies counts one class."""
    px = i.get("currentPrice") or i.get("regularMarketPrice")
    te, fe = i.get("trailingEps"), i.get("forwardEps")
    if not px or te is None or fe is None:
        return None
    shares = i["marketCap"] / px
    return te * shares, fe * shares


def aggregate(infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Index-style aggregates: total market value over total earnings (not an average of P/Es,
    which one loss-maker or one 300x stock would wreck), losses counted. Trailing earnings are
    reported (GAAP); forward ones are analysts' estimates of adjusted earnings for the next
    fiscal year, which usually run above reported, so the gap between the two P/Es is not a
    growth forecast and is not offered as one."""
    both = [(i, e) for i in infos for e in [_earnings(i)] if e]
    cap_b = sum(i["marketCap"] for i, _ in both)
    t_earn = sum(e[0] for _, e in both)
    f_earn = sum(e[1] for _, e in both)
    cap = sum(i["marketCap"] for i in infos)
    rev = sum(i.get("totalRevenue") or 0 for i in infos)
    ni = sum(i.get("netIncomeToCommon") or 0 for i in infos)
    sales = sum(i["marketCap"] / i["priceToSalesTrailing12Months"] for i in infos
                if (i.get("priceToSalesTrailing12Months") or 0) > 0)
    div = sum(i["marketCap"] * (i.get("dividendYield") or 0) / 100 for i in infos)
    eg = [i["earningsGrowth"] for i in infos if i.get("earningsGrowth") is not None]
    rg = [i["revenueGrowth"] for i in infos if i.get("revenueGrowth") is not None]
    rnd = lambda v, n=2: None if v is None else round(v, n)
    return {
        "companies": len(infos), "with_both_eps": len(both), "market_cap_tn": rnd(cap / 1e12, 2),
        "trailing_pe": rnd(cap_b / t_earn if t_earn > 0 else None, 1),
        "forward_pe": rnd(cap_b / f_earn if f_earn > 0 else None, 1),
        "earnings_yield_fwd_pct": rnd(f_earn / cap_b * 100 if cap_b else None, 2),
        "price_to_sales": rnd(cap / sales if sales else None, 2),
        "net_margin_pct": rnd(ni / rev * 100 if rev else None, 1),
        "dividend_yield_pct": rnd(div / cap * 100 if cap else None, 2),
        "median_quarterly_earnings_growth_pct": rnd(float(np.median(eg)) * 100 if eg else None, 1),
        "median_revenue_growth_pct": rnd(float(np.median(rg)) * 100 if rg else None, 1),
        "loss_makers": sum(1 for i in infos if (i.get("trailingEps") or 0) < 0),
    }


def compute_valuation(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    sectors = {m["symbol"]: m.get("sector") or "" for m in members}
    with ThreadPoolExecutor(max_workers=4) as ex:      # Yahoo answers 429 to wider bursts
        fetched = [i for i in ex.map(_info, list(sectors)) if i]
    # Both share classes of Alphabet, Fox and News Corp are index members, and Yahoo gives
    # each the whole company's market value: keep one line per company.
    seen, infos = set(), []
    for i in fetched:
        key = i.get("longName") or i.get("shortName") or i.get("symbol")
        if key not in seen:
            seen.add(key)
            infos.append(i)
    for i in infos:
        i["_sector"] = sectors.get(str(i.get("symbol", "")).replace("-", "."), "")
    if not infos:
        return {}
    top = sorted(infos, key=lambda i: -i["marketCap"])
    cap = sum(i["marketCap"] for i in infos)
    return {
        "as_of": dt.date.today().isoformat(),
        "coverage": f"{len(fetched)} of {len(sectors)} members ({len(infos)} companies)",
        "index": aggregate(infos),
        "top10_share_pct": round(sum(i["marketCap"] for i in top[:10]) / cap * 100, 1),
        "caps_bn": {str(i.get("symbol", "")).replace("-", "."): round(i["marketCap"] / 1e9, 1) for i in fetched},
        "top10": [{"symbol": i.get("symbol"), "name": i.get("shortName"), "weight_pct": round(i["marketCap"] / cap * 100, 2),
                   "forward_pe": i.get("forwardPE") and round(i["forwardPE"], 1)} for i in top[:10]],
        "without_top10": aggregate(top[10:]),
        "by_sector": {sec: {**aggregate([i for i in infos if i["_sector"] == sec]),
                            "weight_pct": round(sum(i["marketCap"] for i in infos if i["_sector"] == sec) / cap * 100, 1)}
                      for sec in sorted({i["_sector"] for i in infos if i["_sector"]})},
    }


# ---------------- earnings ----------------

def earnings_rows(cal: List[Dict[str, Any]], members: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Finnhub's earnings calendar cut to S&P 500 members, with the surprise where reported."""
    rows = []
    for e in cal or []:
        sym = str(e.get("symbol") or "")
        if sym not in members:
            continue
        est, act = e.get("epsEstimate"), e.get("epsActual")
        rest, ract = e.get("revenueEstimate"), e.get("revenueActual")
        rows.append({
            "date": e.get("date"), "symbol": sym, "company": members[sym].get("company_name") or sym,
            "sector": members[sym].get("sector") or "",
            "when": {"bmo": "before open", "amc": "after close", "dmh": "during hours"}.get(e.get("hour") or "", ""),
            "eps_estimate": est, "eps_actual": act,
            "eps_surprise_pct": ((act - est) / abs(est) * 100) if (act is not None and est not in (None, 0)) else None,
            "revenue_estimate_bn": rest / 1e9 if rest else None, "revenue_actual_bn": ract / 1e9 if ract else None,
            "revenue_surprise_pct": ((ract - rest) / rest * 100) if (ract and rest) else None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "symbol"]).drop_duplicates(["symbol", "date"])
    return df
