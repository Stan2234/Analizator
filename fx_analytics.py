"""FX analytics — the calculations behind the Currencies tab.

Deliberately free of Streamlit so it can be tested directly.

The organising idea is that a currency pair is a *ratio*, not an instrument.
"EUR/USD is up" says nothing about whether the euro rose or the dollar fell,
and that distinction decides the trade. So everything here is built on one
normalised quantity — the USD value of one unit of each currency — from which
any cross, any strength ranking, and any correlation follows consistently.

Nothing in this module forecasts. It measures what happened, what a position
currently earns or pays, and which macro driver a pair is actually tracking
right now. Where a number is stale (policy rates change a few times a year)
its as-of date travels with it, because a rate differential is worth nothing
if you don't know how old it is.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("analizator.fx")

FX_BARS_PER_YEAR = 252


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
# `yahoo` is the ticker; `invert` says whether that ticker quotes USD per unit
# of the currency (EURUSD -> False) or units per USD (USDJPY -> True). Getting
# this wrong silently flips the sign of every downstream number, so it is
# declared once here rather than inferred at each call site.

class Ccy:
    __slots__ = ("code", "name", "yahoo", "invert", "group", "flag")

    def __init__(self, code, name, yahoo, invert, group, flag):
        self.code, self.name, self.yahoo = code, name, yahoo
        self.invert, self.group, self.flag = invert, group, flag


CURRENCIES: Dict[str, Ccy] = {c.code: c for c in [
    Ccy("USD", "US Dollar",        None,        False, "G10", "🇺🇸"),
    Ccy("EUR", "Euro",             "EURUSD=X",  False, "G10", "🇪🇺"),
    Ccy("JPY", "Japanese Yen",     "JPY=X",     True,  "G10", "🇯🇵"),
    Ccy("GBP", "British Pound",    "GBPUSD=X",  False, "G10", "🇬🇧"),
    Ccy("CHF", "Swiss Franc",      "CHF=X",     True,  "G10", "🇨🇭"),
    Ccy("AUD", "Australian Dollar", "AUDUSD=X", False, "G10", "🇦🇺"),
    Ccy("CAD", "Canadian Dollar",  "CAD=X",     True,  "G10", "🇨🇦"),
    Ccy("NZD", "New Zealand Dollar", "NZDUSD=X", False, "G10", "🇳🇿"),
    Ccy("SEK", "Swedish Krona",    "SEK=X",     True,  "G10", "🇸🇪"),
    Ccy("NOK", "Norwegian Krone",  "NOK=X",     True,  "G10", "🇳🇴"),
    Ccy("CNY", "Chinese Yuan",     "CNY=X",     True,  "EM",  "🇨🇳"),
    Ccy("MXN", "Mexican Peso",     "MXN=X",     True,  "EM",  "🇲🇽"),
    Ccy("ZAR", "South African Rand", "ZAR=X",   True,  "EM",  "🇿🇦"),
    Ccy("BRL", "Brazilian Real",   "BRL=X",     True,  "EM",  "🇧🇷"),
    Ccy("INR", "Indian Rupee",     "INR=X",     True,  "EM",  "🇮🇳"),
    Ccy("TRY", "Turkish Lira",     "TRY=X",     True,  "EM",  "🇹🇷"),
    Ccy("PLN", "Polish Zloty",     "PLN=X",     True,  "EM",  "🇵🇱"),
]}

G10 = [c for c, v in CURRENCIES.items() if v.group == "G10"]
EM = [c for c, v in CURRENCIES.items() if v.group == "EM"]

# Macro drivers, used to work out empirically what a pair is tracking rather
# than assuming the textbook relationship still holds this quarter.
DRIVERS: Dict[str, Dict[str, str]] = {
    "US 10y yield": {"ticker": "^TNX",     "why": "rate differential / USD carry"},
    "US 5y yield":  {"ticker": "^FVX",     "why": "belly of the curve"},
    "US 3m yield":  {"ticker": "^IRX",     "why": "front-end policy expectations"},
    "S&P 500":      {"ticker": "^GSPC",    "why": "risk appetite"},
    "VIX":          {"ticker": "^VIX",     "why": "volatility / safe-haven demand"},
    "Gold":         {"ticker": "GC=F",     "why": "real rates, debasement hedge"},
    "WTI crude":    {"ticker": "CL=F",     "why": "terms of trade for oil exporters"},
    "Copper":       {"ticker": "HG=F",     "why": "global growth / China cycle"},
    "Dollar index": {"ticker": "DX-Y.NYB", "why": "broad USD"},
}


# --------------------------------------------------------------------------
# Policy rates
# --------------------------------------------------------------------------
# Carry is a rate differential, so it is only as good as the rates behind it.
# FRED carries daily series for a couple of these and nothing usable for the
# rest, so the fallbacks below are maintained by hand — and every rate carries
# the date it was set plus where it came from. A carry number whose vintage you
# can't see is worse than no carry number.

FRED_POLICY_SERIES: Dict[str, str] = {
    "USD": "DFF",      # effective fed funds, daily
    "EUR": "ECBDFR",   # ECB deposit facility rate, daily
}

# FRED carries daily series for those two and nothing clean for the rest — its
# other "policy rate" entries are monthly interbank proxies, not the rate a
# desk funds at. So the remainder is seeded by hand.
#
# A seeded rate is a claim, and an unverified claim on a carry screen is worse
# than a blank: it is checkable, and a bank analyst will check it. Each entry
# therefore records when it was last confirmed against the central bank, and
# anything without a recent confirmation is published as unverified and kept
# out of the rankings rather than quietly averaged into them.
#
# Verified 2026-08-16. Re-confirm before relying on any of these.
SEED_VERIFIED_ON = "2026-08-16"

POLICY_RATE_SEED: Dict[str, Dict[str, Any]] = {
    # code:  rate,  effective from,  confirmed on,      what the number is
    "USD": {"rate": 4.50,  "effective": "2026-03-18", "verified": SEED_VERIFIED_ON,
            "note": "Fed target range upper bound"},
    "EUR": {"rate": 2.25,  "effective": "2026-07-23", "verified": SEED_VERIFIED_ON,
            "note": "ECB deposit facility rate"},
    "JPY": {"rate": 1.00,  "effective": "2026-06-16", "verified": SEED_VERIFIED_ON,
            "note": "BoJ short-term policy rate"},
    "TRY": {"rate": 37.00, "effective": "2026-08-01", "verified": SEED_VERIFIED_ON,
            "note": "CBRT one-week repo"},
    "BRL": {"rate": 14.25, "effective": "2026-06-18", "verified": SEED_VERIFIED_ON,
            "note": "Selic target"},
    "MXN": {"rate": 6.50,  "effective": "2026-06-26", "verified": SEED_VERIFIED_ON,
            "note": "Banxico overnight"},
    # Not confirmed at the last review — shown, but excluded from rankings.
    "GBP": {"rate": 4.50,  "effective": "2026-03-19", "verified": None,
            "note": "BoE Bank Rate"},
    "CHF": {"rate": 0.25,  "effective": "2026-03-20", "verified": None,
            "note": "SNB policy rate"},
    "AUD": {"rate": 4.10,  "effective": "2026-04-01", "verified": None,
            "note": "RBA cash rate"},
    "CAD": {"rate": 2.75,  "effective": "2026-03-12", "verified": None,
            "note": "BoC overnight target"},
    "NZD": {"rate": 3.50,  "effective": "2026-04-09", "verified": None,
            "note": "RBNZ official cash rate"},
    "SEK": {"rate": 2.25,  "effective": "2026-03-20", "verified": None,
            "note": "Riksbank policy rate"},
    "NOK": {"rate": 4.25,  "effective": "2026-03-27", "verified": None,
            "note": "Norges Bank policy rate"},
    "CNY": {"rate": 3.10,  "effective": "2026-03-20", "verified": None,
            "note": "PBoC 1y LPR"},
    "ZAR": {"rate": 7.25,  "effective": "2026-03-20", "verified": None,
            "note": "SARB repo rate"},
    "INR": {"rate": 6.00,  "effective": "2026-04-09", "verified": None,
            "note": "RBI repo rate"},
    "PLN": {"rate": 5.75,  "effective": "2026-03-05", "verified": None,
            "note": "NBP reference rate"},
}

# A confirmation older than this is treated as no confirmation. Most central
# banks meet eight times a year, so a quarter is already a meeting or two.
VERIFICATION_MAX_AGE_DAYS = 90


def policy_rates(fred_key: str = "",
                 overrides: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Current policy rate per currency, with provenance and a trust flag.

    Columns: rate, as_of, source, stale_days, note, status.

    `status` is what callers should gate on:
      live       — pulled from FRED this run
      verified   — hand-seeded and confirmed within VERIFICATION_MAX_AGE_DAYS
      override   — supplied by the caller, e.g. corrected in the UI
      unverified — seeded but not recently confirmed; display, do not rank on
      missing    — no rate at all

    `overrides` maps a currency code to a rate the user has entered, which
    always wins. Rates change a handful of times a year at unpredictable
    moments; letting someone correct one in place beats shipping a stale
    constant and hoping nobody checks.
    """
    import sources as src  # lazy; the module must stay importable offline

    today = dt.date.today()
    overrides = overrides or {}
    rows = []

    for code in CURRENCIES:
        rate = as_of = source = note = None
        status = "missing"

        series = FRED_POLICY_SERIES.get(code)
        if series and fred_key:
            try:
                obs = src.fetch_fred_observations(fred_key, series, limit=10)
                for o in obs:  # newest first; FRED writes "." for missing
                    if o.get("value") not in (None, "", "."):
                        rate, as_of = float(o["value"]), o.get("date")
                        source, status = f"FRED:{series}", "live"
                        break
            except Exception:
                log.exception("FRED policy rate failed for %s", code)

        if rate is None and code in POLICY_RATE_SEED:
            seed = POLICY_RATE_SEED[code]
            rate, as_of, note = seed["rate"], seed["effective"], seed.get("note")
            source = "seeded"
            verified = seed.get("verified")
            status = "unverified"
            if verified:
                try:
                    age = (today - dt.date.fromisoformat(verified)).days
                    if age <= VERIFICATION_MAX_AGE_DAYS:
                        status = "verified"
                        source = f"seeded, confirmed {verified}"
                except Exception:
                    pass

        if code in overrides and overrides[code] is not None:
            rate = float(overrides[code])
            source, status = "entered in app", "override"
            as_of = today.isoformat()

        stale = None
        if as_of:
            try:
                stale = (today - dt.date.fromisoformat(as_of)).days
            except Exception:
                stale = None

        rows.append({"ccy": code, "rate": rate, "as_of": as_of, "source": source,
                     "stale_days": stale, "note": note, "status": status})

    return pd.DataFrame(rows).set_index("ccy")


TRUSTED_RATE_STATUS = ("live", "verified", "override")


# --------------------------------------------------------------------------
# Core: USD value of each currency
# --------------------------------------------------------------------------

def usd_values(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw Yahoo closes to 'USD per 1 unit of currency'.

    `prices` is a DataFrame of closes keyed by Yahoo ticker. Everything else in
    this module derives from the frame returned here, so a pair, a cross, and a
    strength ranking can never disagree with one another.
    """
    out = pd.DataFrame(index=prices.index)
    for code, c in CURRENCIES.items():
        if code == "USD":
            out["USD"] = 1.0
            continue
        if c.yahoo not in prices.columns:
            continue
        s = pd.to_numeric(prices[c.yahoo], errors="coerce")
        s = s.where(s > 0)
        out[code] = (1.0 / s) if c.invert else s
    return out.dropna(axis=1, how="all")


def cross(uv: pd.DataFrame, base: str, quote: str) -> pd.Series:
    """The base/quote rate — how many units of `quote` buy one `base`."""
    if base not in uv.columns or quote not in uv.columns:
        return pd.Series(dtype=float)
    return (uv[base] / uv[quote]).dropna()


# --------------------------------------------------------------------------
# Performance and strength
# --------------------------------------------------------------------------

HORIZON_BARS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def _pct_change_over(s: pd.Series, bars: int) -> Optional[float]:
    s = s.dropna()
    if len(s) <= bars:
        return None
    prev = float(s.iloc[-1 - bars])
    if prev == 0:
        return None
    return (float(s.iloc[-1]) / prev - 1.0) * 100.0


def ytd_change(s: pd.Series) -> Optional[float]:
    s = s.dropna()
    if s.empty:
        return None
    jan1 = pd.Timestamp(dt.date(s.index[-1].year, 1, 1))
    prior = s[s.index <= jan1]
    base = float(prior.iloc[-1]) if len(prior) else float(s.iloc[0])
    if base == 0:
        return None
    return (float(s.iloc[-1]) / base - 1.0) * 100.0


def _aligned(uv: pd.DataFrame, codes: List[str]) -> pd.DataFrame:
    """Restrict to sessions where every requested currency actually printed.

    Feeds are ragged — one ticker goes quiet on a holiday its neighbours trade
    through. Measured on their own indices, "21 bars ago" lands on different
    calendar dates for different currencies, and a matrix built that way is
    comparing slightly different weeks in each cell. Cross-sectional views
    align first; the time-series ones (vol, correlation) keep the raw index so
    no synthetic prints are invented.
    """
    live = [c for c in codes if c in uv.columns]
    if not live:
        return pd.DataFrame()
    return uv[live].dropna(how="any")


def performance_table(uv: pd.DataFrame, codes: List[str]) -> pd.DataFrame:
    """Each currency's move against the USD across standard horizons.

    Positive means the currency appreciated against the dollar, for every row —
    including the ones whose market convention is quoted the other way up. That
    consistency is the whole point of normalising through `usd_values` first.
    """
    rows = []
    for code in codes:
        if code not in uv.columns or code == "USD":
            continue
        s = uv[code]
        row = {"ccy": code, "name": CURRENCIES[code].name, "flag": CURRENCIES[code].flag}
        for label, bars in HORIZON_BARS.items():
            row[label] = _pct_change_over(s, bars)
        row["YTD"] = ytd_change(s)
        rows.append(row)
    return pd.DataFrame(rows).set_index("ccy")


def cross_matrix(uv: pd.DataFrame, codes: List[str], bars: int) -> pd.DataFrame:
    """Row currency vs column currency, % move over `bars` sessions.

    The desk's first screen: read across a row to see what that currency did
    against everything, read down a column to see what everything did to it.
    Anti-symmetric by construction, so the diagonal is zero and a strong row is
    necessarily a weak column.
    """
    aligned = _aligned(uv, codes)
    if aligned.empty:
        return pd.DataFrame()
    perf = {c: _pct_change_over(aligned[c], bars) for c in aligned.columns}
    live = [c for c in codes if perf.get(c) is not None]
    m = pd.DataFrame(index=live, columns=live, dtype=float)
    for b in live:
        for q in live:
            # (1+rb)/(1+rq) - 1 — compounding the two legs, not subtracting them
            m.loc[b, q] = 0.0 if b == q else (
                (1 + perf[b] / 100.0) / (1 + perf[q] / 100.0) - 1.0) * 100.0
    return m


def currency_strength(uv: pd.DataFrame, codes: List[str], bars: int) -> pd.DataFrame:
    """Average move of each currency against all the others in `codes`.

    This is the answer to "is EUR strong or is USD weak" — a question a single
    pair can never settle. A currency that rose against everything is genuinely
    bid; one that rose only against the dollar is really a dollar story.
    """
    m = cross_matrix(uv, codes, bars)
    if m.empty:
        return pd.DataFrame()
    strength = m.mean(axis=1).sort_values(ascending=False)
    return pd.DataFrame({
        "strength": strength,
        "name": [CURRENCIES[c].name for c in strength.index],
        "flag": [CURRENCIES[c].flag for c in strength.index],
    })


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------

def realized_vol(s: pd.Series, window: int, bars_per_year: int = FX_BARS_PER_YEAR) -> Optional[float]:
    r = np.log(s.dropna().astype(float)).diff().dropna()
    if len(r) < window:
        return None
    return float(r.tail(window).std() * np.sqrt(bars_per_year) * 100.0)


def vol_profile(uv: pd.DataFrame, base: str, quote: str) -> Dict[str, Any]:
    """Realized vol at three tenors, plus where current vol sits historically.

    The percentile matters more than the level: 8% vol is calm for GBP/JPY and
    an emergency for EUR/CHF, and only the pair's own history says which.
    """
    s = cross(uv, base, quote)
    if s.empty:
        return {}
    out = {
        "vol_1m": realized_vol(s, 21),
        "vol_3m": realized_vol(s, 63),
        "vol_1y": realized_vol(s, 252),
    }
    r = np.log(s.astype(float)).diff().dropna()
    roll = r.rolling(21).std() * np.sqrt(FX_BARS_PER_YEAR) * 100.0
    roll = roll.dropna()
    if len(roll) > 60:
        out["vol_percentile"] = float((roll < roll.iloc[-1]).mean() * 100.0)
        out["vol_1y_low"] = float(roll.tail(252).min())
        out["vol_1y_high"] = float(roll.tail(252).max())
    # Short vol above long vol means the market is pricing near-term stress.
    if out.get("vol_1m") and out.get("vol_3m"):
        out["term_structure"] = out["vol_1m"] - out["vol_3m"]
    return out


# --------------------------------------------------------------------------
# Carry
# --------------------------------------------------------------------------

def annualized_drift(s: pd.Series, window: int = 252,
                     bars_per_year: int = FX_BARS_PER_YEAR) -> Optional[float]:
    """Annualized rate of the spot trend over `window` sessions, in percent."""
    r = np.log(s.dropna().astype(float)).diff().dropna()
    if len(r) < 60:
        return None
    return float(r.tail(window).mean() * bars_per_year * 100.0)


def carry_table(uv: pd.DataFrame, rates: pd.DataFrame, codes: List[str],
                vs: str = "USD") -> pd.DataFrame:
    """Annual carry of being long each currency against `vs`, and what it cost.

    Raw carry ranks badly on its own: the highest-yielding currencies yield
    what they do precisely because they are risky. The usual correction is
    carry_to_vol — the differential over realized volatility, a Sharpe-like
    instinct — and on floating currencies it works.

    It breaks on managed ones, and it breaks in the most dangerous direction.
    A crawling peg depreciates a little every day and barely wobbles, so its
    volatility is tiny while its losses are relentless. Divide a large carry by
    that small volatility and the screen will rank the currency first, right up
    until the peg goes. That is the shape of nearly every carry blow-up.

    So this table also carries `spot_1y` (what the currency actually did) and
    `total_1y` = carry + spot, which is what the position would really have
    earned. `drift_to_vol` compares the size of the trend to the size of the
    noise; when it is large the currency is being steered rather than traded,
    `regime` says so, and carry_to_vol should not be read as a ranking.
    """
    rows = []
    for code in codes:
        if code == vs or code not in uv.columns:
            continue
        r_base = rates["rate"].get(code)
        r_quote = rates["rate"].get(vs)
        if r_base is None or r_quote is None or pd.isna(r_base) or pd.isna(r_quote):
            continue

        carry = float(r_base) - float(r_quote)
        px = cross(uv, code, vs)
        vol = realized_vol(px, 63)
        spot_1y = annualized_drift(px, 252)

        drift_to_vol = None
        regime = "floating"
        if vol and vol > 0 and spot_1y is not None:
            drift_to_vol = abs(spot_1y) / vol
            # A trend several times the size of the daily noise is not how a
            # freely traded currency behaves.
            if drift_to_vol >= 2.0:
                regime = "managed / crawling peg"
            elif drift_to_vol >= 1.0:
                regime = "heavily trending"

        # Both legs must be trustworthy: the differential is only as sound as
        # the weaker of the two rates behind it.
        st_base = rates["status"].get(code) if "status" in rates.columns else "verified"
        st_quote = rates["status"].get(vs) if "status" in rates.columns else "verified"
        trusted = (st_base in TRUSTED_RATE_STATUS) and (st_quote in TRUSTED_RATE_STATUS)

        rows.append({
            "ccy": code,
            "name": CURRENCIES[code].name,
            "flag": CURRENCIES[code].flag,
            "rate": float(r_base),
            "carry": carry,
            "vol_3m": vol,
            "carry_to_vol": (carry / vol) if (vol and vol > 0) else None,
            "spot_1y": spot_1y,
            "total_1y": (carry + spot_1y) if spot_1y is not None else None,
            "drift_to_vol": drift_to_vol,
            "regime": regime,
            "rate_status": st_base,
            "trusted": trusted,
            "as_of": rates["as_of"].get(code),
            "stale_days": rates["stale_days"].get(code),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.set_index("ccy")
    # Trusted rows rank first, then by what the trade actually paid — not by
    # the ratio a managed currency games. Rows resting on an unconfirmed rate
    # stay visible but sink below the ones that can be defended.
    return df.sort_values(["trusted", "total_1y"], ascending=[False, False],
                          na_position="last")


# --------------------------------------------------------------------------
# Correlation and drivers
# --------------------------------------------------------------------------

def correlation_matrix(uv: pd.DataFrame, pairs: List[Tuple[str, str]],
                       window: int = 63) -> pd.DataFrame:
    """Correlation of daily returns between pairs, over the last `window` days.

    Read it as a concentration check. Three positions with 0.9 correlation are
    one position at triple size, which is how an FX book that looks diversified
    on paper loses three times as much as expected on the same day.
    """
    cols = {}
    for b, q in pairs:
        s = cross(uv, b, q)
        if not s.empty:
            cols[f"{b}/{q}"] = np.log(s.astype(float)).diff()
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols).dropna()
    if len(df) < 20:
        return pd.DataFrame()
    return df.tail(window).corr()


def crowded_clusters(corr: pd.DataFrame, threshold: float = 0.7) -> List[List[str]]:
    """Group pairs whose correlation exceeds `threshold` — effectively one trade."""
    if corr.empty:
        return []
    names = list(corr.columns)
    seen, groups = set(), []
    for n in names:
        if n in seen:
            continue
        grp = [n] + [m for m in names
                     if m != n and m not in seen and abs(corr.loc[n, m]) >= threshold]
        if len(grp) > 1:
            groups.append(grp)
            seen.update(grp)
    return groups


def driver_correlations(uv: pd.DataFrame, base: str, quote: str,
                        drivers: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """What the pair has actually been tracking, ranked by strength.

    Textbook FX drivers stop working for quarters at a time — AUD decouples
    from copper, the yen ignores rate differentials while intervention risk
    dominates. Rather than asserting a relationship, this measures it over the
    recent window and lets the ranking speak.
    """
    s = cross(uv, base, quote)
    if s.empty or drivers is None or drivers.empty:
        return pd.DataFrame()
    r_pair = np.log(s.astype(float)).diff()

    rows = []
    for label, meta in DRIVERS.items():
        tkr = meta["ticker"]
        if tkr not in drivers.columns:
            continue
        d = pd.to_numeric(drivers[tkr], errors="coerce")
        r_drv = np.log(d.where(d > 0)).diff()
        joined = pd.concat([r_pair, r_drv], axis=1).dropna().tail(window)
        if len(joined) < 20:
            continue
        c = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        if np.isnan(c):
            continue
        rows.append({"driver": label, "corr": c, "abs": abs(c),
                     "channel": meta["why"], "n": len(joined)})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values("abs", ascending=False)
            .drop(columns="abs").set_index("driver"))


def dominant_driver(uv: pd.DataFrame, base: str, quote: str,
                    drivers: pd.DataFrame, window: int = 63) -> Optional[Dict[str, Any]]:
    """The single strongest driver correlation, or None when nothing dominates."""
    df = driver_correlations(uv, base, quote, drivers, window)
    if df.empty:
        return None
    top = df.iloc[0]
    return {"driver": df.index[0], "corr": float(top["corr"]),
            "channel": top["channel"]}


# --------------------------------------------------------------------------
# Event risk
# --------------------------------------------------------------------------

# Tier-1 events move FX on release; everything else is noise for this purpose.
TIER1_KEYWORDS = (
    "interest rate", "rate decision", "policy rate", "fomc", "ecb", "boe", "boj",
    "cpi", "inflation", "gdp", "employment", "payroll", "unemployment",
    "pmi", "retail sales", "trade balance",
)

# Finnhub reports country codes; map them to the currency they move.
COUNTRY_TO_CCY = {
    "US": "USD", "EU": "EUR", "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR",
    "JP": "JPY", "GB": "GBP", "UK": "GBP", "CH": "CHF", "AU": "AUD", "CA": "CAD",
    "NZ": "NZD", "SE": "SEK", "NO": "NOK", "CN": "CNY", "MX": "MXN",
    "ZA": "ZAR", "BR": "BRL", "IN": "INR", "TR": "TRY", "PL": "PLN",
}


def fx_event_risk(events: List[Dict[str, Any]], codes: List[str],
                  days_ahead: int = 14) -> pd.DataFrame:
    """Upcoming tier-1 releases for the currencies on screen.

    Carry and momentum both die at scheduled events, so a screen that ranks
    pairs without showing what is on the calendar is only half the picture.
    """
    if not events:
        return pd.DataFrame()
    today = dt.date.today()
    horizon = today + dt.timedelta(days=days_ahead)
    wanted = set(codes)

    rows = []
    for e in events:
        country = str(e.get("country") or "").upper()[:2]
        ccy = COUNTRY_TO_CCY.get(country)
        if ccy not in wanted:
            continue
        # Finnhub calls them time/event/impact; the local cache stores
        # event_time/title/importance. Accept either.
        raw = str(e.get("event_time") or e.get("time") or e.get("date") or "")[:10]
        try:
            when = dt.date.fromisoformat(raw)
        except Exception:
            continue
        if not (today <= when <= horizon):
            continue
        label = str(e.get("title") or e.get("event") or "")
        low = label.lower()
        if not any(k in low for k in TIER1_KEYWORDS):
            continue
        rows.append({
            "date": when, "ccy": ccy, "flag": CURRENCIES[ccy].flag,
            "event": label,
            "impact": e.get("importance") or e.get("impact") or "",
            "is_cb": any(k in low for k in
                         ("rate decision", "interest rate", "policy rate",
                          "fomc", "ecb", "boe", "boj")),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "ccy"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Pair summary
# --------------------------------------------------------------------------

def pair_snapshot(uv: pd.DataFrame, base: str, quote: str,
                  rates: Optional[pd.DataFrame] = None,
                  drivers: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Everything the desk wants on one pair, in one dict."""
    s = cross(uv, base, quote)
    if s.empty:
        return {}

    out: Dict[str, Any] = {
        "pair": f"{base}/{quote}",
        "spot": float(s.iloc[-1]),
        "n_bars": len(s),
    }
    for label, bars in HORIZON_BARS.items():
        out[f"chg_{label}"] = _pct_change_over(s, bars)
    out["chg_YTD"] = ytd_change(s)

    out.update(vol_profile(uv, base, quote))

    # 52-week range position: 0 = at the low, 100 = at the high.
    last_year = s.tail(252)
    if len(last_year) > 60:
        lo, hi = float(last_year.min()), float(last_year.max())
        out["range_low"], out["range_high"] = lo, hi
        out["range_pct"] = ((out["spot"] - lo) / (hi - lo) * 100.0) if hi > lo else None

    if rates is not None and not rates.empty:
        rb, rq = rates["rate"].get(base), rates["rate"].get(quote)
        if rb is not None and rq is not None and not (pd.isna(rb) or pd.isna(rq)):
            out["carry"] = float(rb) - float(rq)
            v = out.get("vol_3m")
            out["carry_to_vol"] = (out["carry"] / v) if (v and v > 0) else None
            out["rate_base"], out["rate_quote"] = float(rb), float(rq)
            out["rate_base_as_of"] = rates["as_of"].get(base)
            out["rate_quote_as_of"] = rates["as_of"].get(quote)
            # What the carry actually paid once the spot move is included.
            drift = annualized_drift(s, 252)
            out["spot_1y"] = drift
            if drift is not None:
                out["total_1y"] = out["carry"] + drift
                if v and v > 0:
                    out["drift_to_vol"] = abs(drift) / v

    if drivers is not None and not drivers.empty:
        dom = dominant_driver(uv, base, quote, drivers)
        if dom:
            out["dominant_driver"] = dom

    return out
