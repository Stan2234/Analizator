"""Fundamentals for whole markets: currencies, metals, crypto, oil and the stock market.

A company has accounts; a currency, a metal or a barrel of oil has drivers. Each
function here gathers the drivers a desk would read for that market, from free
official sources, and returns the same shape so one panel can show any of them:

    {"headline", "cards": [{topic, state, tone, text}], "rows": [...], "series": {...},
     "context": {...}, "sources"}

Cards and rows describe; none of them adds up to a score. Where a driver has a
textbook direction ("a higher real rate usually supports a currency") the row says
which side it leans to and that it is a tendency, never a total.

Sources: IMF World Economic Outlook (datamapper API), FRED (needs FRED_API_KEY),
BIS through central_banks.py, EIA weekly petroleum files, CoinGecko, blockchain.com,
Yahoo futures, the World Gold Council files and the stock jobs in equities.py.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("analizator.market_fundamentals")
UA = {"User-Agent": "Mozilla/5.0"}
_cache: Dict[Any, Any] = {}


def _cached(key: Any, ttl: float, fn: Callable[[], Any]) -> Any:
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        v = fn()
    except Exception as e:
        log.warning("fetch %s failed: %s", key, e)
        return hit[1] if hit else None
    if v is not None and not (hasattr(v, "__len__") and len(v) == 0):
        _cache[key] = (time.time(), v)
    return v


def _series(v: Any) -> pd.Series:
    return v if v is not None else pd.Series(dtype=float)


def _r(v: Any, nd: int = 2) -> Optional[float]:
    try:
        return None if v is None or pd.isna(v) else round(float(v), nd)
    except Exception:
        return None


def _p(v: Optional[float], nd: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{nd}f}%"


def _n(v: Optional[float], nd: int = 2, unit: str = "") -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:,.{nd}f}{unit}"


def change(s: pd.Series, days: int) -> Optional[float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    past = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    return None if past.empty or not past.iloc[-1] else (float(s.iloc[-1]) / float(past.iloc[-1]) - 1) * 100


def diff(s: pd.Series, days: int) -> Optional[float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    past = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    return None if past.empty else float(s.iloc[-1]) - float(past.iloc[-1])


def pctile(s: pd.Series, v: Optional[float] = None) -> Optional[float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    v = float(s.iloc[-1]) if v is None else v
    return float((s < v).mean() * 100)


# ---------------- sources ----------------

def fred(series_id: str, n: int = 2000) -> pd.Series:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        return pd.Series(dtype=float)

    def get():
        r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=30, params={
            "series_id": series_id, "api_key": key, "file_type": "json", "sort_order": "desc", "limit": n})
        r.raise_for_status()
        obs = r.json().get("observations") or []
        return pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs
                          if o.get("value") not in (".", "", None)}).sort_index()
    return _series(_cached(("fred", series_id, n), 6 * 3600, get))


IMF_INDICATORS = {
    "NGDP_RPCH": "Real GDP growth, %", "PCPIPCH": "Inflation (average), %", "LUR": "Unemployment, %",
    "BCA_NGDPD": "Current account, % of GDP", "GGXCNL_NGDP": "Budget balance, % of GDP",
    "GGXWDG_NGDP": "Government debt, % of GDP",
}


def imf(indicator: str) -> Dict[str, Dict[int, float]]:
    """{IMF country code: {year: value}} from the World Economic Outlook, actuals and projections."""
    def get():
        # the IMF turns away a bare browser user agent; the library default gets through
        r = requests.get(f"https://www.imf.org/external/datamapper/api/v1/{indicator}", timeout=40)
        r.raise_for_status()
        vals = (r.json().get("values") or {}).get(indicator) or {}
        return {c: {int(y): float(v) for y, v in ys.items() if v is not None} for c, ys in vals.items()}
    return _cached(("imf", indicator), 24 * 3600, get) or {}


def eia_weekly(series_id: str) -> pd.Series:
    """An EIA weekly petroleum series from its history file (no key needed)."""
    def get():
        r = requests.get(f"https://www.eia.gov/dnav/pet/hist_xls/{series_id}w.xls", timeout=60, headers=UA)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", header=2)
        return pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                         index=pd.to_datetime(df.iloc[:, 0])).dropna().sort_index()
    return _series(_cached(("eia", series_id), 6 * 3600, get))


def yahoo(symbol: str, years: float = 5) -> pd.Series:
    def get():
        now = int(time.time())
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", timeout=20, headers=UA,
                         params={"period1": now - int(years * 365.25 * 86400), "period2": now, "interval": "1d"})
        res = r.json()["chart"]["result"][0]
        s = pd.Series(res["indicators"]["quote"][0]["close"], index=pd.to_datetime(res["timestamp"], unit="s"))
        s.index = s.index.normalize()
        return s.dropna()[~s.dropna().index.duplicated(keep="last")]
    return _series(_cached(("yahoo", symbol, years), 1800, get))


def _corr_of_changes(a: pd.Series, b: pd.Series, freq: str = "W-FRI", days: int = 365,
                     a_pct: bool = True, b_pct: bool = False) -> Optional[float]:
    """Correlation of weekly changes over the last `days`: how closely two things have moved together."""
    a = pd.to_numeric(a, errors="coerce").dropna().resample(freq).last()
    b = pd.to_numeric(b, errors="coerce").dropna().resample(freq).last()
    da = a.pct_change() if a_pct else a.diff()
    db = b.pct_change() if b_pct else b.diff()
    df = pd.concat([da, db], axis=1).dropna()
    df = df[df.index >= df.index[-1] - pd.Timedelta(days=days)] if len(df) else df
    return _r(df.iloc[:, 0].corr(df.iloc[:, 1]), 2) if len(df) >= 20 else None


# ---------------- currencies ----------------

# currency -> (economy, IMF code, BIS/FRED REER code, OECD code for the 10-year yield)
ECONOMIES = {
    "USD": ("the US", "USA", "US", "US"), "EUR": ("the euro area", "EURO", "XM", "DE"),
    "JPY": ("Japan", "JPN", "JP", "JP"), "GBP": ("the UK", "GBR", "GB", "GB"),
    "CHF": ("Switzerland", "CHE", "CH", "CH"), "AUD": ("Australia", "AUS", "AU", "AU"),
    "CAD": ("Canada", "CAN", "CA", "CA"), "NZD": ("New Zealand", "NZL", "NZ", "NZ"),
    "SEK": ("Sweden", "SWE", "SE", "SE"), "NOK": ("Norway", "NOR", "NO", "NO"),
    "CNY": ("China", "CHN", "CN", None), "INR": ("India", "IND", "IN", "IN"),
    "ZAR": ("South Africa", "ZAF", "ZA", "ZA"), "TRY": ("Türkiye", "TUR", "TR", "TR"),
    "BRL": ("Brazil", "BRA", "BR", None), "MXN": ("Mexico", "MEX", "MX", "MX"),
    "PLN": ("Poland", "POL", "PL", "PL"),
}
TEN_YEAR_NOTE = {"EUR": "German 10-year Bund, the euro area's benchmark"}


def ten_year(ccy: str) -> pd.Series:
    code = ECONOMIES[ccy][3]
    return fred(f"IRLTLT01{code}M156N", 200) if code else pd.Series(dtype=float)


def reer(ccy: str) -> pd.Series:
    return fred(f"RB{ECONOMIES[ccy][2]}BIS", 400)


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _poss(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


def pair_symbol(base: str, quote: str) -> str:
    return f"{base}{quote}=X"


def currency_fundamentals(base: str, quote: str, rates: pd.DataFrame, infl: pd.DataFrame,
                          pair: pd.Series) -> Dict[str, Any]:
    yr = dt.date.today().year
    rows: List[Dict[str, Any]] = []
    ctx: Dict[str, Any] = {"pair": f"{base}/{quote}", "economies": {base: ECONOMIES[base][0], quote: ECONOMIES[quote][0]}}

    def lean(gap: Optional[float], rule: Optional[str], thr: float) -> str:
        if gap is None or rule is None or abs(gap) < thr:
            return "neither"
        return (base if gap > 0 else quote) if rule == "higher" else (quote if gap > 0 else base)

    def add(driver: str, b: Optional[float], q: Optional[float], unit: str, rule: Optional[str], thr: float,
            why: str, nd: int = 2) -> None:
        gap = None if b is None or q is None else b - q
        rows.append({"Driver": driver, base: _n(b, nd, unit), quote: _n(q, nd, unit),
                     "Gap": "n/a" if gap is None else f"{gap:+.{nd}f}{unit}",
                     "Usually favours": lean(gap, rule, thr), "Why it matters": why,
                     "_b": b, "_q": q, "_gap": gap})

    pol = {c: (float(rates[c].dropna().iloc[-1]) if c in rates.columns and rates[c].notna().any() else None)
           for c in (base, quote)}
    pol_1y = {c: (float(rates[c].dropna()[rates[c].dropna().index <= rates.index[-1] - pd.Timedelta(days=365)].iloc[-1])
                  if c in rates.columns and len(rates[c].dropna()) > 300 else None) for c in (base, quote)}
    cpi = {c: (float(infl.loc[c, "cpi_yoy"]) if c in infl.index and pd.notna(infl.loc[c, "cpi_yoy"]) else None)
           for c in (base, quote)}
    tens = {c: ten_year(c) for c in (base, quote)}
    ten = {c: (float(tens[c].iloc[-1]) if len(tens[c]) else None) for c in (base, quote)}
    add("Policy rate", pol[base], pol[quote], "%", "higher", 0.25,
        "The overnight return on holding the currency; money tends to flow to the higher one.")
    real = {c: (pol[c] - cpi[c] if pol[c] is not None and cpi[c] is not None else None) for c in (base, quote)}
    add("Inflation (CPI, year on year)", cpi[base], cpi[quote], "%", None, 0,
        "Erodes what the rate pays; a central bank facing high inflation has to keep rates up.", 1)
    add("Real policy rate (rate minus inflation)", real[base], real[quote], " pp", "higher", 0.5,
        "What the rate pays after inflation, the better measure of how tight policy is.")
    add("10-year government bond yield", ten[base], ten[quote], "%", "higher", 0.25,
        "What the market expects rates to average over a decade; spreads here move currencies more than "
        "policy rates do.")
    ind = {k: imf(k) for k in IMF_INDICATORS}
    g = lambda k, c, y: ind[k].get(ECONOMIES[c][1], {}).get(y)
    add(f"GDP growth {yr} (IMF)", g("NGDP_RPCH", base, yr), g("NGDP_RPCH", quote, yr), "%", "higher", 0.3,
        "Faster growth draws investment and lets the central bank keep rates higher.", 1)
    add(f"GDP growth {yr + 1} (IMF forecast)", g("NGDP_RPCH", base, yr + 1), g("NGDP_RPCH", quote, yr + 1), "%",
        "higher", 0.3, "Markets trade the outlook more than the past.", 1)
    add(f"Unemployment {yr} (IMF)", g("LUR", base, yr), g("LUR", quote, yr), "%", "lower", 0.5,
        "Slack in the labour market: high unemployment pushes a central bank toward cuts.", 1)
    add(f"Current account {yr}, % of GDP (IMF)", g("BCA_NGDPD", base, yr), g("BCA_NGDPD", quote, yr), "%",
        "higher", 1.0, "A surplus means the economy earns more foreign currency than it spends, a steady "
                       "source of demand for its own; a deficit has to be financed by inflows.", 1)
    add(f"Budget balance {yr}, % of GDP (IMF)", g("GGXCNL_NGDP", base, yr), g("GGXCNL_NGDP", quote, yr), "%",
        "higher", 1.0, "Large deficits need foreign buyers for the debt and can weigh on the currency.", 1)
    add(f"Government debt {yr}, % of GDP (IMF)", g("GGXWDG_NGDP", base, yr), g("GGXWDG_NGDP", quote, yr), "%",
        "lower", 10.0, "A slow-moving risk: it matters when markets start to doubt it can be financed.", 0)
    re_dev = {}
    for c in (base, quote):
        s = reer(c)
        tail = s[s.index >= s.index[-1] - pd.Timedelta(days=3650)] if len(s) else s
        re_dev[c] = (float(s.iloc[-1]) / float(tail.mean()) - 1) * 100 if len(tail) > 24 else None
    add("Real effective exchange rate vs its 10-year average", re_dev[base], re_dev[quote], "%", "lower", 3.0,
        "The currency against its trading partners after inflation (BIS). Well above its own average is "
        "expensive and below is cheap; it pulls back slowly, over years, not weeks.", 1)

    # the price against the 10-year yield spread
    spread = (tens[base] - tens[quote]).dropna() if len(tens[base]) and len(tens[quote]) else pd.Series(dtype=float)
    link = None
    if len(spread) > 30 and len(pair) > 300:
        pm = pair.resample("MS").last()
        df = pd.concat([pm.pct_change(), spread.resample("MS").last().diff()], axis=1).dropna().tail(36)
        link = _r(df.iloc[:, 0].corr(df.iloc[:, 1]), 2) if len(df) >= 18 else None

    # cards
    cards: List[Dict[str, str]] = []
    if pol[base] is not None and pol[quote] is not None:
        gap = pol[base] - pol[quote]
        gap1 = (pol_1y[base] - pol_1y[quote]) if None not in (pol_1y[base], pol_1y[quote]) else None
        hi = base if gap > 0 else quote
        text = (f"{_cap(_poss(ECONOMIES[base][0]))} policy rate is {pol[base]:.2f}% and {_poss(ECONOMIES[quote][0])} "
                f"{pol[quote]:.2f}%: holding {hi} pays {abs(gap):.2f} points more a year")
        if gap1 is not None:
            moved = ("widened" if abs(gap) > abs(gap1) + 0.1 else "narrowed" if abs(gap) < abs(gap1) - 0.1 else "held")
            text += (f"; a year ago the gap was {abs(gap1):.2f} points in favour of {base if gap1 > 0 else quote}"
                     if gap1 * gap > 0 or abs(gap1) < 0.05 else f"; a year ago {base if gap1 > 0 else quote} paid {abs(gap1):.2f} more")
            text += f", so it has {moved}" if gap1 * gap > 0 else ""
        if ten[base] is not None and ten[quote] is not None:
            text += f". Ten-year yields: {ten[base]:.2f}% against {ten[quote]:.2f}%"
        cards.append({"topic": "Interest rates", "state": f"{hi} pays more" if abs(gap) >= 0.25 else "About level",
                      "tone": "flat", "text": text + "."})
    if real[base] is not None and real[quote] is not None:
        rg = real[base] - real[quote]
        cards.append({"topic": "After inflation", "state": f"{base if rg > 0 else quote} pays more" if abs(rg) >= 0.5 else "About level",
                      "tone": "flat",
                      "text": f"Inflation is {cpi[base]:.1f}% in {ECONOMIES[base][0]} and {cpi[quote]:.1f}% in "
                              f"{ECONOMIES[quote][0]}, so the real policy rate is {real[base]:+.2f}% against "
                              f"{real[quote]:+.2f}%. A higher real rate usually supports a currency."})
    gb, gq = g("NGDP_RPCH", base, yr + 1), g("NGDP_RPCH", quote, yr + 1)
    if gb is not None and gq is not None:
        cards.append({"topic": "Growth outlook (IMF)", "state": f"{base if gb > gq else quote} grows faster" if abs(gb - gq) >= 0.3 else "Similar",
                      "tone": "flat",
                      "text": f"The IMF expects {ECONOMIES[base][0]} to grow {g('NGDP_RPCH', base, yr) or 0:.1f}% this year "
                              f"and {gb:.1f}% next, against {g('NGDP_RPCH', quote, yr) or 0:.1f}% and {gq:.1f}% for "
                              f"{ECONOMIES[quote][0]}."})
    cb_, cq_ = g("BCA_NGDPD", base, yr), g("BCA_NGDPD", quote, yr)
    if cb_ is not None and cq_ is not None:
        cards.append({"topic": "Trade and capital flows", "state": f"{base if cb_ > cq_ else quote} stronger" if abs(cb_ - cq_) >= 1 else "Similar",
                      "tone": "flat",
                      "text": f"Current account: {cb_:+.1f}% of GDP for {ECONOMIES[base][0]} and {cq_:+.1f}% for "
                              f"{ECONOMIES[quote][0]} (IMF, {yr}). A deficit has to be financed by foreign money, "
                              "which leaves a currency exposed when investors pull back."})
    if re_dev[base] is not None or re_dev[quote] is not None:
        def word(v):
            return "n/a" if v is None else ("expensive" if v > 5 else "cheap" if v < -5 else "near its average")
        cards.append({"topic": "Valuation (real exchange rate)",
                      "state": f"{base} {word(re_dev[base])}",
                      "tone": "flat",
                      "text": f"Against its trading partners and after inflation, {base} is {_p(re_dev[base])} from "
                              f"its 10-year average and {quote} {_p(re_dev[quote])} (BIS). Valuation pulls slowly, "
                              "over years; it says where the risk sits, not when it turns."})
    bb, bq = g("GGXCNL_NGDP", base, yr), g("GGXCNL_NGDP", quote, yr)
    db, dq = g("GGXWDG_NGDP", base, yr), g("GGXWDG_NGDP", quote, yr)
    if None not in (bb, bq, db, dq):
        cards.append({"topic": "Public finances (IMF)", "state": f"{base if bb < bq else quote} borrows more",
                      "tone": "flat",
                      "text": f"Budget balance {bb:+.1f}% of GDP and debt {db:.0f}% for {ECONOMIES[base][0]}, against "
                              f"{bq:+.1f}% and {dq:.0f}% for {ECONOMIES[quote][0]}."})
    if link is not None:
        cards.append({"topic": "What the price has followed", "state": f"Correlation {link:+.2f}",
                      "tone": "flat",
                      "text": f"Over the last three years the monthly moves in {base}/{quote} had a correlation of "
                              f"{link:+.2f} with the change in the 10-year yield gap ({base} minus {quote}). "
                              + ("A strong positive link: the pair has traded on rate expectations."
                                 if link > 0.4 else "A weak link: something other than rates has driven the pair."
                                 if abs(link) < 0.2 else "A moderate link.")})
    leans = [r["Usually favours"] for r in rows if r["_gap"] is not None]
    headline = (f"{base}/{quote}: of the drivers with a textbook direction, {leans.count(base)} usually favour {base}, "
                f"{leans.count(quote)} favour {quote} and {leans.count('neither')} {'is' if leans.count('neither') == 1 else 'are'} too close to call. A count, not a forecast: "
                "the drivers are not equally strong and the market already prices most of them.")
    ctx.update({"drivers": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
                "ten_year_note": TEN_YEAR_NOTE, "pair_vs_10y_spread_correlation_36m": link,
                "pair_last": _r(pair.iloc[-1], 5) if len(pair) else None,
                "pair_change_pct": {"1m": _r(change(pair, 30)), "3m": _r(change(pair, 91)), "1y": _r(change(pair, 365))}})
    return {"headline": headline, "cards": cards, "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
            "series": {"pair": pair, "spread": spread}, "context": ctx,
            "sources": "BIS (policy rates, CPI, real effective exchange rates via FRED), OECD 10-year yields via "
                       "FRED, IMF World Economic Outlook (growth, unemployment, current account, budget, debt), "
                       "Yahoo (the pair)."}


# ---------------- metals ----------------

def metal_fundamentals(metal: str, px: pd.DataFrame, cpi: pd.Series, wgc_ctx: Dict[str, Any],
                       metals_mod: Any) -> Dict[str, Any]:
    """The drivers of one metal: real yields, the dollar, inflation, who buys, its real price and ratios."""
    M = metals_mod.METALS
    s = pd.to_numeric(px.get(M[metal].yahoo), errors="coerce").dropna() if M[metal].yahoo in px else pd.Series(dtype=float)
    ry, usd, be = fred("DFII10", 3000), fred("DTWEXBGS", 3000), fred("T10YIE", 400)
    cards: List[Dict[str, str]] = []
    rows: List[Dict[str, Any]] = []
    ctx: Dict[str, Any] = {"metal": metal, "unit": M[metal].unit, "what_it_is": M[metal].note}

    def row(driver, now, ago, context):
        rows.append({"Driver": driver, "Now": now, "A year ago": ago, "Reading": context})

    # real yields
    if len(ry) and len(s):
        c1, c5 = _corr_of_changes(s, ry, days=365), _corr_of_changes(s, ry, days=5 * 365)
        r_now, r_1y = float(ry.iloc[-1]), diff(ry, 365)
        row("US 10-year real yield (TIPS)", f"{r_now:.2f}%", _n(r_now - r_1y if r_1y is not None else None, 2, "%"),
            f"Weekly correlation with {metal.lower()}: {c1:+.2f} over 1 year, {c5:+.2f} over 5 years."
            if c1 is not None and c5 is not None else "Correlation not available.")
        if metal in ("Gold", "Silver"):
            state = "Headwind" if (diff(ry, 91) or 0) > 0.15 else "Tailwind" if (diff(ry, 91) or 0) < -0.15 else "Steady"
            cards.append({"topic": "Real interest rates", "state": state,
                          "tone": {"Headwind": "warn", "Tailwind": "up"}.get(state, "flat"),
                          "text": f"The US 10-year real yield (what a Treasury pays after expected inflation) is "
                                  f"{r_now:.2f}%, {diff(ry, 91) or 0:+.2f} points over three months. "
                                  f"{metal} pays nothing, so higher real yields raise the cost of holding it. "
                                  + (f"The link has weakened: weekly correlation {c1:+.2f} over the past year against "
                                     f"{c5:+.2f} over five." if c1 is not None and c5 is not None and abs(c1) < abs(c5) - 0.1
                                     else f"Weekly correlation over the past year: {c1:+.2f}." if c1 is not None else "")})
        ctx["real_yield"] = {"now": _r(r_now), "change_3m_pp": _r(diff(ry, 91)), "change_1y_pp": _r(r_1y),
                             "corr_weekly_1y": c1, "corr_weekly_5y": c5}
    if len(usd) and len(s):
        u3 = change(usd, 91)
        cu = _corr_of_changes(s, usd, days=365, b_pct=True)
        row("Broad dollar index (Fed)", _n(float(usd.iloc[-1]), 1), _n(float(usd[usd.index <= usd.index[-1] - pd.Timedelta(days=365)].iloc[-1]) if len(usd) > 300 else None, 1),
            f"3-month change {_p(u3)}; weekly correlation with {metal.lower()} over 1 year {cu:+.2f}." if cu is not None else f"3-month change {_p(u3)}.")
        state = "Headwind" if (u3 or 0) > 2 else "Tailwind" if (u3 or 0) < -2 else "Steady"
        cards.append({"topic": "The dollar", "state": state, "tone": {"Headwind": "warn", "Tailwind": "up"}.get(state, "flat"),
                      "text": f"The Fed's broad dollar index moved {_p(u3)} over three months. Metals are priced in "
                              f"dollars, so a stronger dollar makes them dearer for everyone else"
                              + (f"; over the past year {metal.lower()}'s weekly moves had a {cu:+.2f} correlation with it." if cu is not None else ".")})
        ctx["dollar"] = {"broad_index": _r(usd.iloc[-1], 1), "change_3m_pct": _r(u3), "corr_weekly_1y": cu}
    # inflation
    cpi_s = pd.to_numeric(cpi, errors="coerce").dropna()
    if len(cpi_s) > 13:
        yoy = (cpi_s.iloc[-1] / cpi_s.iloc[-13] - 1) * 100
        row("US inflation (CPI, year on year)", f"{yoy:.1f}%", _n((cpi_s.iloc[-13] / cpi_s.iloc[-25] - 1) * 100 if len(cpi_s) > 25 else None, 1, "%"),
            f"Market-implied 10-year inflation (breakeven): {float(be.iloc[-1]):.2f}%." if len(be) else "")
        ctx["inflation"] = {"cpi_yoy": _r(yoy, 1), "breakeven_10y": _r(be.iloc[-1]) if len(be) else None}
    # real price
    if len(s) and len(cpi_s):
        rp = metals_mod.real_price(s, cpi_s)
        if rp.get("available"):
            nd = 0 if rp["real_last"] >= 100 else 2
            row(f"Price in today's money (CPI-adjusted)", _n(rp["real_last"], nd), "",
                f"{rp['below_real_peak_pct']:+.1f}% from its inflation-adjusted record of {rp['real_peak']:,.{nd}f} "
                f"({rp['real_peak_date']}); data from {rp['from']}.")
            at = rp["at_real_high"] or rp["below_real_peak_pct"] > -2
            cards.append({"topic": "Price after inflation", "state": "At a real record" if at else f"{rp['below_real_peak_pct']:+.0f}% from real record",
                          "tone": "warn" if at else "flat",
                          "text": f"In today's money {metal.lower()}'s record was {rp['real_peak']:,.{nd}f} on {rp['real_peak_date']}; "
                                  f"the price now is {rp['real_last']:,.{nd}f} ({rp['below_real_peak_pct']:+.1f}%). A real "
                                  "record means buyers pay more than at any time in the data, not just more dollars."})
            ctx["real_price"] = {k: rp[k] for k in ("real_last", "real_peak", "real_peak_date", "below_real_peak_pct", "from")}
    # ratios
    for r in metals_mod.RATIOS:
        if metal in (r["num"], r["den"]):
            ra = metals_mod.ratio(px, r["num"], r["den"])
            if ra.get("available"):
                row(r["name"], _n(ra["last"], 4 if not r["comparable"] else 2), "",
                    f"Percentile {ra['percentile']:.0f} since {ra['from']}; {ra['vs_median_pct']:+.0f}% from its median. {r['asks']}")
                ctx.setdefault("ratios", {})[r["name"]] = {"last": _r(ra["last"], 4), "percentile": _r(ra["percentile"], 0),
                                                           "vs_median_pct": _r(ra["vs_median_pct"], 1), "since": ra["from"]}
    # who buys gold
    if metal == "Gold" and wgc_ctx and not wgc_ctx.get("error"):
        cb = wgc_ctx.get("central_banks") or {}
        etf = wgc_ctx.get("gold_etfs") or {}
        nt = (cb.get("net_tonnes") or {})
        if nt:
            last12 = next((v for k, v in nt.items() if k.startswith("last_")), None)
            cards.append({"topic": "Central banks", "state": f"{last12:+,.0f} t in 12 months" if last12 is not None else "n/a",
                          "tone": "up" if (last12 or 0) > 0 else "warn",
                          "text": f"Central banks reported net {'purchases' if (last12 or 0) > 0 else 'sales'} of "
                                  f"{abs(last12 or 0):,.0f} tonnes over 12 months against {nt.get('prior_12m', 0):+,.0f} t "
                                  f"in the 12 before (World Gold Council, as of {cb.get('as_of')}). "
                                  + (f"Largest buyers: {', '.join(x['country'] for x in (cb.get('top_buyers') or [])[:3])}."
                                     if cb.get("top_buyers") else "")})
            row("Central-bank net buying, 12 months", f"{last12:+,.0f} t" if last12 is not None else "n/a",
                f"{nt.get('prior_12m', 0):+,.0f} t", f"World Gold Council, as of {cb.get('as_of')}.")
        dem = (etf.get("demand_t") or {}).get("12m") or {}
        if dem:
            tot = sum(dem.values())
            cards.append({"topic": "Gold ETFs", "state": f"{tot:+,.0f} t in 12 months", "tone": "up" if tot > 0 else "warn",
                          "text": f"Gold ETFs {'added' if tot > 0 else 'shed'} {abs(tot):,.0f} tonnes over 12 months "
                                  f"and hold {etf.get('holdings_tonnes', 0):,.0f} t (record {(etf.get('holdings_peak') or {}).get('tonnes', 0):,.0f} t "
                                  f"in {(etf.get('holdings_peak') or {}).get('month', '')}). ETF buying is investors; "
                                  "it turns faster than central banks."})
            row("Gold ETF flows, 12 months", f"{tot:+,.0f} t", "", f"Holdings {etf.get('holdings_tonnes', 0):,.0f} t; WGC as of {etf.get('as_of')}.")
        ctx["world_gold_council"] = wgc_ctx
    # silver / platinum / palladium / copper: the industrial link
    if metal in ("Silver", "Platinum", "Palladium", "Copper") and "HG=F" in px and "GC=F" in px and len(s):
        cc, cg = _corr_of_changes(s, px["HG=F"], b_pct=True), _corr_of_changes(s, px["GC=F"], b_pct=True)
        if metal != "Copper" and cc is not None and cg is not None:
            cards.append({"topic": "Industrial or monetary", "state": "Trades like gold" if cg > cc + 0.1 else "Trades like copper" if cc > cg + 0.1 else "Both",
                          "tone": "flat",
                          "text": f"Over the past year {metal.lower()}'s weekly moves had a {cg:+.2f} correlation with "
                                  f"gold and {cc:+.2f} with copper: whether the market treats it as money or as an "
                                  "industrial input right now."})
        ctx["correlations_weekly_1y"] = {"gold": cg, "copper": cc}
    if metal == "Copper" and "^TNX" in px and len(s):
        cards.append({"topic": "Growth signal", "state": f"{_p(change(s, 91))} in 3 months", "tone": "flat",
                      "text": f"Copper is read as a gauge of global industry. Over three months it moved {_p(change(s, 91))} "
                              f"while the 10-year Treasury yield changed {diff(px['^TNX'], 91) or 0:+.2f} points; the two "
                              "usually rise together when growth expectations improve."})
    trend = f"{metal} at {_n(float(s.iloc[-1]), 2)} {M[metal].unit}, {_p(change(s, 30))} over a month and {_p(change(s, 365))} over a year." if len(s) else ""
    ctx["price"] = {"last": _r(s.iloc[-1]) if len(s) else None, "change_1m_pct": _r(change(s, 30)), "change_1y_pct": _r(change(s, 365))}
    return {"headline": trend, "cards": cards, "rows": rows,
            "series": {"price": s, "real_yield": ry, "dollar": usd}, "context": ctx,
            "sources": "Yahoo futures, FRED (TIPS real yield DFII10, broad dollar DTWEXBGS, breakeven T10YIE), "
                       "BIS CPI, World Gold Council (central banks, ETFs)."}


# ---------------- oil ----------------

EIA = {"WCESTUS1": "US commercial crude stocks (ex SPR), million barrels",
       "WGTSTUS1": "US gasoline stocks, million barrels", "WDISTUS1": "US distillate stocks, million barrels",
       "WCSSTUS1": "Strategic Petroleum Reserve, million barrels", "WCRFPUS2": "US crude production, million b/d",
       "WPULEUS3": "US refinery utilisation, %"}
MONTH_CODES = "FGHJKMNQUVXZ"


def vs_five_year(s: pd.Series) -> Dict[str, Any]:
    """The latest week against the same week of the five previous years."""
    s = s.dropna()
    if len(s) < 300:
        return {}
    last_d, last = s.index[-1], float(s.iloc[-1])
    same = []
    for k in range(1, 6):
        d = last_d - pd.DateOffset(years=k)
        near = s[(s.index >= d - pd.Timedelta(days=4)) & (s.index <= d + pd.Timedelta(days=4))]
        if len(near):
            same.append(float(near.iloc[0]))
    avg = float(np.mean(same)) if same else None
    yago = same[0] if same else None
    return {"date": str(last_d.date()), "last": last, "five_year_avg": avg,
            "vs_avg_pct": (last / avg - 1) * 100 if avg else None, "year_ago": yago,
            "week_change": last - float(s.iloc[-2]) if len(s) > 1 else None}


def futures_curve(root: str = "CL", months: int = 12) -> List[Dict[str, Any]]:
    """Prices of the next `months` contracts (Yahoo symbols like CLZ26.NYM)."""
    out = []
    d = dt.date.today().replace(day=1)
    for i in range(months + 2):
        m = (d.month - 1 + i) % 12 + 1
        y = d.year + (d.month - 1 + i) // 12
        sym = f"{root}{MONTH_CODES[m - 1]}{str(y)[2:]}.NYM"
        s = yahoo(sym, years=0.05)
        if len(s):
            out.append({"contract": f"{dt.date(y, m, 1):%b %Y}", "symbol": sym, "price": float(s.iloc[-1])})
        if len(out) >= months:
            break
    return out


def oil_fundamentals(benchmark: str = "WTI") -> Dict[str, Any]:
    front = yahoo("CL=F" if benchmark == "WTI" else "BZ=F", 10)
    other = yahoo("BZ=F" if benchmark == "WTI" else "CL=F", 3)
    eia = {k: eia_weekly(k) for k in EIA}
    st = {k: vs_five_year(v / (1000 if k in ("WCESTUS1", "WGTSTUS1", "WDISTUS1", "WCSSTUS1", "WCRFPUS2") else 1))
          for k, v in eia.items() if len(v)}
    curve = futures_curve("CL" if benchmark == "WTI" else "BZ", 12)
    cards, rows = [], []
    for k, v in st.items():
        if v:
            rows.append({"Driver": EIA[k], "Latest week": f"{v['last']:,.1f}", "Same week, 5-year average":
                         _n(v["five_year_avg"], 1), "Vs average": _p(v["vs_avg_pct"]), "A year ago": _n(v["year_ago"], 1),
                         "Week change": _n(v["week_change"], 2), "Week of": v["date"]})
    c = st.get("WCESTUS1") or {}
    if c:
        state = "Tight" if (c["vs_avg_pct"] or 0) < -4 else "Ample" if (c["vs_avg_pct"] or 0) > 4 else "Normal"
        g, d = st.get("WGTSTUS1") or {}, st.get("WDISTUS1") or {}
        cards.append({"topic": "US inventories", "state": state, "tone": {"Tight": "up", "Ample": "warn"}.get(state, "flat"),
                      "text": f"Commercial crude stocks are {c['last']:,.1f} million barrels, {_p(c['vs_avg_pct'])} against "
                              f"the same week's five-year average ({c['week_change']:+.1f} m last week). Gasoline "
                              f"{_p(g.get('vs_avg_pct'))} and distillates {_p(d.get('vs_avg_pct'))} against theirs. Low "
                              "stocks mean less cushion for a supply shock, which supports prices (EIA, week of "
                              f"{c['date']})."})
    if len(curve) >= 7:
        f0, f6 = curve[0]["price"], curve[6]["price"]
        sp = (f0 / f6 - 1) * 100
        state = "Backwardation" if sp > 1 else "Contango" if sp < -1 else "Flat"
        cards.append({"topic": "Futures curve", "state": state, "tone": {"Backwardation": "up", "Contango": "warn"}.get(state, "flat"),
                      "text": f"The {curve[0]['contract']} contract trades at {f0:,.2f} and {curve[6]['contract']} at "
                              f"{f6:,.2f} ({sp:+.1f}%). "
                              + ("Near-term barrels cost more than later ones: the market is paying for oil now, the "
                                 "usual sign of a tight market." if state == "Backwardation" else
                                 "Later barrels cost more: oil is plentiful enough that holders are paid to store it."
                                 if state == "Contango" else "Near and later barrels cost about the same.")})
    p = st.get("WCRFPUS2") or {}
    if p:
        cards.append({"topic": "US supply", "state": f"{p['last']:.2f} m b/d",
                      "tone": "flat",
                      "text": f"US crude output is {p['last']:.2f} million barrels a day against {_n(p['year_ago'], 2)} a year "
                              f"ago. Refineries run at {(st.get('WPULEUS3') or {}).get('last', float('nan')):.1f}% of "
                              f"capacity; the Strategic Petroleum Reserve holds {(st.get('WCSSTUS1') or {}).get('last', float('nan')):,.0f} m barrels."})
    if len(front) and len(other):
        spread = float(yahoo("BZ=F", 3).iloc[-1] - yahoo("CL=F", 3).iloc[-1])
        cards.append({"topic": "Brent against WTI", "state": f"{spread:+.2f} a barrel", "tone": "flat",
                      "text": f"Brent (the seaborne, international benchmark) trades {spread:+.2f} against WTI (US). A wide "
                              "gap pulls US crude into exports; a narrow one keeps it at home."})
    usd = fred("DTWEXBGS", 400)
    if len(usd):
        cards.append({"topic": "The dollar", "state": _p(change(usd, 91)) + " in 3 months", "tone": "flat",
                      "text": f"Oil is priced in dollars; the Fed's broad dollar index moved {_p(change(usd, 91))} over "
                              "three months. A stronger dollar makes a barrel dearer for most buyers."})
    head = (f"{benchmark} at {float(front.iloc[-1]):,.2f}, {_p(change(front, 30))} over a month and {_p(change(front, 365))} "
            "over a year." if len(front) else "")
    ctx = {"benchmark": benchmark, "price": {"last": _r(front.iloc[-1]) if len(front) else None,
                                            "change_1m_pct": _r(change(front, 30)), "change_1y_pct": _r(change(front, 365))},
           "eia_weekly": {EIA[k]: {kk: _r(vv) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in st.items()},
           "futures_curve": curve, "not_in_the_data": "OPEC+ quotas and output, global (non-US) inventories, Chinese "
                                                       "imports, refinery margins"}
    return {"headline": head, "cards": cards, "rows": rows, "series": {"price": front, "crude_stocks": eia.get("WCESTUS1", pd.Series(dtype=float)) / 1000},
            "curve": curve, "context": ctx,
            "sources": "EIA Weekly Petroleum Status Report (history files), Yahoo futures, FRED broad dollar."}


# ---------------- crypto ----------------

def coingecko_coin(coin_id: str) -> Dict[str, Any]:
    def get():
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}", timeout=30, headers=UA, params={
            "localization": "false", "tickers": "false", "community_data": "true", "developer_data": "true"})
        r.raise_for_status()
        return r.json()
    return _cached(("cg", coin_id), 1800, get) or {}


def coingecko_history(coin_id: str, days: int = 365) -> pd.DataFrame:
    def get():
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart", timeout=30, headers=UA,
                         params={"vs_currency": "usd", "days": days, "interval": "daily"})
        r.raise_for_status()
        j = r.json()
        px = pd.Series({pd.Timestamp(t, unit="ms").normalize(): v for t, v in j.get("prices", [])})
        mc = pd.Series({pd.Timestamp(t, unit="ms").normalize(): v for t, v in j.get("market_caps", [])})
        vol = pd.Series({pd.Timestamp(t, unit="ms").normalize(): v for t, v in j.get("total_volumes", [])})
        return pd.DataFrame({"price": px, "mcap": mc, "volume": vol}).dropna()
    return _cached(("cgh", coin_id, days), 3600, get)


COINS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", "BNB": "binancecoin",
         "DOGE": "dogecoin", "ADA": "cardano", "TRX": "tron", "LINK": "chainlink", "AVAX": "avalanche-2",
         "TON": "the-open-network", "DOT": "polkadot", "LTC": "litecoin", "BCH": "bitcoin-cash", "SUI": "sui",
         "XLM": "stellar", "HBAR": "hedera-hashgraph", "NEAR": "near", "UNI": "uniswap", "AAVE": "aave"}


def stablecoin_cap_bn() -> Optional[float]:
    def get():
        r = requests.get("https://api.coingecko.com/api/v3/coins/categories", timeout=30, headers=UA)
        r.raise_for_status()
        return next((c["market_cap"] / 1e9 for c in r.json() if c.get("id") == "stablecoins" and c.get("market_cap")), None)
    return _cached(("cg_stable",), 3600, get)


def blockchain_chart(name: str) -> pd.Series:
    def get():
        r = requests.get(f"https://api.blockchain.info/charts/{name}", timeout=30, headers=UA,
                         params={"timespan": "2years", "format": "json", "sampled": "true"})
        r.raise_for_status()
        return pd.Series({pd.Timestamp(v["x"], unit="s"): v["y"] for v in r.json().get("values", [])}).sort_index()
    return _series(_cached(("bc", name), 6 * 3600, get))


def crypto_fundamentals(coin_id: str, funding: Optional[float] = None,
                        stablecoin_cap_bn: Optional[float] = None) -> Dict[str, Any]:
    c = coingecko_coin(coin_id)
    md = c.get("market_data") or {}
    if not md:
        return {"headline": "", "cards": [], "rows": [], "series": {}, "context": {},
                "sources": "CoinGecko", "error": f"no CoinGecko data for {coin_id}"}
    usd = lambda k: (md.get(k) or {}).get("usd")
    mc, fdv, vol = usd("market_cap"), usd("fully_diluted_valuation"), usd("total_volume")
    circ, tot, mx = md.get("circulating_supply"), md.get("total_supply"), md.get("max_supply")
    h = coingecko_history(coin_id, 365)
    supply_growth = None
    if h is not None and len(h) > 300:
        sup = (h["mcap"] / h["price"]).replace([np.inf, -np.inf], np.nan).dropna()
        supply_growth = (float(sup.iloc[-1]) / float(sup.iloc[0]) - 1) * 100 if len(sup) > 300 else None
    name, sym = c.get("name", coin_id), (c.get("symbol") or "").upper()
    cards, rows = [], []
    rows += [
        {"Measure": "Market value", "Value": f"${mc / 1e9:,.1f}bn" if mc else "n/a", "Reading": f"Rank {c.get('market_cap_rank')}"},
        {"Measure": "Fully diluted value (all coins ever)", "Value": f"${fdv / 1e9:,.1f}bn" if fdv else "n/a",
         "Reading": f"Market value is {mc / fdv * 100:.0f}% of it" if mc and fdv else "No maximum supply"},
        {"Measure": "Coins in circulation", "Value": _n(circ, 0), "Reading": f"of a maximum {mx:,.0f} ({circ / mx * 100:.1f}% issued)" if circ and mx else "No hard cap"},
        {"Measure": "Supply growth, past year", "Value": _p(supply_growth, 2), "Reading": "New coins dilute holders like new shares"},
        {"Measure": "Trading volume / market value (24h)", "Value": f"{vol / mc * 100:.1f}%" if vol and mc else "n/a", "Reading": "Liquidity"},
        {"Measure": "From all-time high", "Value": _p(md.get("ath_change_percentage", {}).get("usd")),
         "Reading": f"Record ${usd('ath'):,.2f} on {str((md.get('ath_date') or {}).get('usd', ''))[:10]}" if usd("ath") else ""},
        {"Measure": "Developer commits, past 4 weeks", "Value": _n((c.get("developer_data") or {}).get("commit_count_4_weeks"), 0),
         "Reading": "GitHub activity on the main repository (CoinGecko)"},
    ]
    if circ and mx:
        cards.append({"topic": "Supply", "state": f"{circ / mx * 100:.1f}% issued", "tone": "flat",
                      "text": f"{circ:,.0f} of a maximum {mx:,.0f} {sym} exist; supply grew {_p(supply_growth, 2)} over the "
                              "past year. New coins are paid to miners or stakers and sold over time, a steady supply the "
                              "price has to absorb."})
    elif supply_growth is not None:
        cards.append({"topic": "Supply", "state": "No hard cap", "tone": "flat",
                      "text": f"{name} has no maximum supply; coins in circulation changed {_p(supply_growth, 2)} over the "
                              "past year. Whether issuance outruns what is burned or locked decides the dilution."})
    if mc and fdv and fdv > mc * 1.05:
        cards.append({"topic": "Dilution ahead", "state": f"{(fdv / mc - 1) * 100:.0f}% more to come", "tone": "warn",
                      "text": f"At today's price all coins would be worth ${fdv / 1e9:,.0f}bn against ${mc / 1e9:,.0f}bn "
                              "in circulation: the unissued supply is an overhang that unlocks over time."})
    ath = (md.get("ath_change_percentage") or {}).get("usd")
    if ath is not None:
        cards.append({"topic": "Against its record", "state": _p(ath, 0), "tone": "flat" if ath > -20 else "warn",
                      "text": f"{name} is {abs(ath):.0f}% below its record of ${usd('ath'):,.2f} "
                              f"({str((md.get('ath_date') or {}).get('usd', ''))[:10]}); over a year it moved "
                              f"{_p(md.get('price_change_percentage_1y'))}."})
    if coin_id == "bitcoin":
        hr, fees, ntx = blockchain_chart("hash-rate"), blockchain_chart("transaction-fees-usd"), blockchain_chart("n-transactions")
        if len(hr):
            h90 = change(hr.rolling(7, min_periods=1).mean(), 90)
            cards.append({"topic": "Network security", "state": f"Hash rate {_p(h90, 0)} in 90 days", "tone": "up" if (h90 or 0) > 0 else "warn",
                          "text": f"Miners' computing power on the network changed {_p(h90, 0)} over 90 days and "
                                  f"{_p(change(hr.rolling(7, min_periods=1).mean(), 365), 0)} over a year. Rising hash rate means "
                                  "miners keep investing at today's price; a fall often follows a price drop that "
                                  "pushes the weakest out."})
            rows.append({"Measure": "Hash rate, change over a year", "Value": _p(change(hr.rolling(7, min_periods=1).mean(), 365), 0),
                         "Reading": "blockchain.com"})
        if len(ntx):
            rows.append({"Measure": "Transactions a day (7-day average)", "Value": _n(float(ntx.tail(7).mean()), 0),
                         "Reading": f"{_p(change(ntx.rolling(7, min_periods=1).mean(), 365), 0)} on a year ago (blockchain.com)"})
        if len(fees):
            rows.append({"Measure": "Fees paid to miners a day (7-day average)", "Value": f"${float(fees.tail(7).mean()):,.0f}",
                         "Reading": "Demand for block space (blockchain.com)"})
    if funding is not None:
        cards.append({"topic": "Leverage", "state": f"Funding {funding * 100:+.3f}% / 8h", "tone": "warn" if abs(funding) > 0.0003 else "flat",
                      "text": f"Perpetual futures funding on Binance is {funding * 100:+.3f}% every eight hours "
                              f"({funding * 3 * 365 * 100:+.0f}% a year). Positive means longs pay shorts to hold their "
                              "bets: high positive funding is a crowded, leveraged long."})
    if stablecoin_cap_bn:
        rows.append({"Measure": "Stablecoins in circulation (dry powder)", "Value": f"${stablecoin_cap_bn:,.0f}bn",
                     "Reading": "Dollars on crypto rails that can buy coins quickly"})
    head = f"{name} at ${usd('current_price'):,.2f}, {_p(md.get('price_change_percentage_30d'))} over 30 days and {_p(md.get('price_change_percentage_1y'))} over a year."
    ctx = {"coin": name, "symbol": sym, "measures": rows, "categories": (c.get("categories") or [])[:6],
           "hashing_algorithm": c.get("hashing_algorithm"), "genesis_date": c.get("genesis_date"),
           "funding_rate_8h": funding,
           "not_in_the_data": "on-chain holder data (MVRV, exchange balances, long-term holders), ETF flows, token unlock schedules"}
    return {"headline": head, "cards": cards, "rows": rows, "series": {"price": h["price"] if h is not None and len(h) else pd.Series(dtype=float)},
            "context": ctx, "sources": "CoinGecko (supply, market data, developer activity), blockchain.com (bitcoin network), Binance funding."}


# ---------------- the stock market ----------------

def stock_market_fundamentals(val: Dict[str, Any], hist: List[Dict[str, Any]], earn_beat: Optional[float],
                              equities_gdp: Optional[List[List[Any]]]) -> Dict[str, Any]:
    """The S&P 500 as one business: what it earns, what it costs against bonds and the economy."""
    ix = (val or {}).get("index") or {}
    if not ix:
        return {"headline": "", "cards": [], "rows": [], "series": {}, "context": {}, "sources": "",
                "error": "the bottom-up valuation has not been computed yet; it runs after each US close"}
    y10, ry = fred("DGS10", 400), fred("DFII10", 400)
    y10v = float(y10.iloc[-1]) if len(y10) else None
    ryv = float(ry.iloc[-1]) if len(ry) else None
    ey = ix["earnings_yield_fwd_pct"]
    cards, rows = [], []
    cards.append({"topic": "What it costs", "state": f"{ix['forward_pe']:.1f}x forward", "tone": "flat",
                  "text": f"Summed over all {ix['companies']} companies, the S&P 500 costs {ix['trailing_pe']:.1f} times "
                          f"the last 12 months' reported earnings and {ix['forward_pe']:.1f} times analysts' estimates for "
                          f"the next fiscal year, with a price-to-sales ratio of {ix['price_to_sales']:.2f}."})
    if y10v is not None:
        gap = ey - y10v
        cards.append({"topic": "Against bonds", "state": f"{gap:+.2f} pp over Treasuries", "tone": "warn" if gap < 1 else "flat",
                      "text": f"The forward earnings yield is {ey:.2f}% against {y10v:.2f}% on the 10-year Treasury"
                              + (f" and {ryv:.2f}% on its inflation-protected version" if ryv is not None else "")
                              + f". The gap ({gap:+.2f} points) is the extra return stocks offer for their risk; "
                                "a thin gap leaves less room if earnings disappoint."})
    top10 = val.get("top10_share_pct")
    ex = (val.get("without_top10") or {}).get("forward_pe")
    if top10:
        cards.append({"topic": "Concentration", "state": f"Top 10 = {top10:.0f}%", "tone": "warn" if top10 > 35 else "flat",
                      "text": f"The ten largest companies are {top10:.0f}% of the index's value"
                              + (f"; without them it trades at {ex:.1f} times forward earnings against "
                                 f"{ix['forward_pe']:.1f} for the whole." if ex else ".")})
    cards.append({"topic": "Profitability and growth", "state": f"{ix['net_margin_pct']:.1f}% net margin", "tone": "flat",
                  "text": f"Together the companies keep {ix['net_margin_pct']:.1f} cents of each dollar of sales as profit. "
                          f"The median company grew revenue {_p(ix['median_revenue_growth_pct'])} and its latest "
                          f"quarter's earnings {_p(ix['median_quarterly_earnings_growth_pct'])} on a year earlier."
                          + (f" {earn_beat:.0f}% of last week's reports beat the estimate." if earn_beat is not None else "")})
    if equities_gdp:
        vals = pd.Series({pd.Timestamp(d): v for d, v in equities_gdp})
        pc = pctile(vals)
        cards.append({"topic": "Against the economy", "state": f"{float(vals.iloc[-1]):.0f}% of GDP", "tone": "warn" if (pc or 0) > 90 else "flat",
                      "text": f"US stocks are worth {float(vals.iloc[-1]):.0f}% of GDP (Fed Z.1, latest quarter), higher than in "
                              f"{pc:.0f}% of quarters since {vals.index[0].year}; the dot-com peak was "
                              f"{float(vals[vals.index < '2003'].max()):.0f}%. A slow-moving gauge that says how much is priced "
                              "in, not when it unwinds."})
    for k, label in (("trailing_pe", "P/E, last 12 months' reported earnings"), ("forward_pe", "P/E, next fiscal year's estimates"),
                     ("earnings_yield_fwd_pct", "Forward earnings yield, %"), ("price_to_sales", "Price to sales"),
                     ("net_margin_pct", "Net margin, %"), ("dividend_yield_pct", "Dividend yield, %"),
                     ("median_revenue_growth_pct", "Median revenue growth, %"),
                     ("median_quarterly_earnings_growth_pct", "Median latest-quarter earnings growth, %")):
        first = next((h for h in hist if h.get(k) is not None), None) if hist else None
        rows.append({"Measure": label, "Now": ix.get(k), "Without the top 10": (val.get("without_top10") or {}).get(k),
                     "Recorded since": f"{first[k]} on {first['date']}" if first else ""})
    rows.append({"Measure": "10-year Treasury yield, %", "Now": y10v, "Without the top 10": None, "Recorded since": ""})
    rows.append({"Measure": "10-year real yield (TIPS), %", "Now": ryv, "Without the top 10": None, "Recorded since": ""})
    head = (f"The S&P 500 trades at {ix['forward_pe']:.1f} times next year's estimated earnings, a {ey:.2f}% earnings yield"
            + (f" against {y10v:.2f}% on the 10-year Treasury." if y10v is not None else "."))
    ctx = {"valuation": {k: v for k, v in val.items() if k != "caps_bn"}, "valuation_history": hist[-90:],
           "ten_year_pct": y10v, "real_ten_year_pct": ryv, "equities_to_gdp_latest": equities_gdp[-1] if equities_gdp else None,
           "not_in_the_data": "Shiller CAPE and long valuation history (recording started recently), earnings revisions, buybacks"}
    return {"headline": head, "cards": cards, "rows": rows, "series": {}, "context": ctx,
            "sources": "Every S&P 500 member's Yahoo quote summary, summed; FRED (10-year and TIPS yields); "
                       "Fed Z.1 via AI Bubble Tracker (stocks/GDP); Finnhub earnings."}
