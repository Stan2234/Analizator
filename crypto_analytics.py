"""Crypto: how much of this is one position, and who is already in it.

Two facts shape almost everything a crypto book does, and neither is visible
on a page of prices.

The first is that the asset class is mostly one asset. Coins are marketed on
their differences and traded on their similarities, and a portfolio of twelve
of them is usually bitcoin held twelve times — with the resemblance strongest
exactly when it matters, because the thing that moves them together is the
thing that moves them down. So the beta of each coin to bitcoin, and the
clusters that fall together, come before anything else here.

The second is that positioning is observable in crypto in a way it is not in
FX or metals. A perpetual future has no expiry, so it is pinned to spot by a
funding payment that one side pays the other every eight hours. That payment
is a live, public price for being long — and when it is high and paid by longs
for weeks, the market is telling you who is already in and what a reversal
would have to unwind. The other panels in this app have to say positioning
data is unavailable. Here it is free.

Everything is measured and reported separately. There is no composite crypto
score, for the same reason there is none anywhere else in this project: a beta
of 1.4 and funding at 30% annualised are two different facts about two
different things, and averaging them produces a number that answers no
question at all.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 30
_UA = {"User-Agent": "Analizator/1.0 (macro dashboard)"}

COINGECKO = "https://api.coingecko.com/api/v3"

# Binance geo-blocks its main host from some regions with a 451, and Render
# sits in one of them. The .vision mirror serves the same public data and is
# tried first for that reason.
BINANCE_SPOT_HOSTS = [
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api.binance.com",
]
BINANCE_FUTURES_HOSTS = [
    "https://fapi.binance.com",
]

# Excluded from the board rather than displayed and explained away. A
# stablecoin's return is zero by construction, so it contributes nothing to a
# ranking except a row that looks broken; and CoinGecko's "down 24% from
# all-time high" for a dollar peg is an artefact of one bad print years ago.
STABLECOINS = {
    "usdt", "usdc", "usds", "dai", "fdusd", "tusd", "pyusd", "usde", "busd",
    "usd1", "rlusd", "usdg", "usdy", "usdd", "frax", "lusd", "gusd", "usdp",
    "susds", "usdf", "buidl", "usyc", "syrupusdc", "usdtb", "usdx",
}

# Tokenised metal, not crypto. Their beta to bitcoin is near zero because they
# are gold, and leaving them in a crypto board misreports the asset class.
TOKENISED_COMMODITIES = {"xaut", "paxg", "kau", "xaum"}

# A perpetual settles its funding three times a day.
FUNDINGS_PER_DAY = 3
FUNDINGS_PER_YEAR = FUNDINGS_PER_DAY * 365


# --------------------------------------------------------------------------
# What exists
# --------------------------------------------------------------------------

def _get(hosts: List[str], path: str, params: Optional[Dict[str, Any]] = None,
         timeout: int = HTTP_TIMEOUT) -> Optional[Any]:
    """First host that answers wins; None if none do."""
    last = None
    for host in hosts:
        try:
            r = requests.get(f"{host}{path}", params=params,
                             headers=_UA, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as exc:
            last = str(exc)[:90]
    if last:
        log.warning("all hosts failed for %s: %s", path, last)
    return None


def market_table(top_n: int = 100) -> pd.DataFrame:
    """Market capitalisation, price and drawdown for the largest coins.

    The column that says the most is the distance from the all-time high.
    Equities are usually discussed in terms of their last year; crypto is
    better described by where it sits against its own peak, because the peaks
    are enormous and the recoveries take cycles rather than quarters.

    Columns: symbol, name, rank, price, market_cap, volume_24h, chg_24h,
    chg_7d, chg_30d, chg_1y, from_ath_pct, ath, ath_date.
    """
    rows: List[Dict[str, Any]] = []
    per_page = min(250, max(top_n, 1))

    data = _get([COINGECKO], "/coins/markets", {
        "vs_currency": "usd", "order": "market_cap_desc",
        "per_page": per_page, "page": 1,
        "price_change_percentage": "24h,7d,30d,1y",
    }, timeout=45)
    if not data:
        return pd.DataFrame()

    for c in data:
        sym = str(c.get("symbol", "")).lower()
        if not sym or sym in STABLECOINS:
            continue
        rows.append({
            "symbol": sym.upper(),
            "name": c.get("name"),
            "rank": c.get("market_cap_rank"),
            "price": c.get("current_price"),
            "market_cap": c.get("market_cap"),
            "volume_24h": c.get("total_volume"),
            "chg_24h": c.get("price_change_percentage_24h_in_currency"),
            "chg_7d": c.get("price_change_percentage_7d_in_currency"),
            "chg_30d": c.get("price_change_percentage_30d_in_currency"),
            "chg_1y": c.get("price_change_percentage_1y_in_currency"),
            "from_ath_pct": c.get("ath_change_percentage"),
            "ath": c.get("ath"),
            "ath_date": (str(c.get("ath_date"))[:10] if c.get("ath_date") else None),
            "tokenised_commodity": sym in TOKENISED_COMMODITIES,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("symbol")


def global_stats() -> Dict[str, Any]:
    """Total capitalisation and each large coin's share of it."""
    data = _get([COINGECKO], "/global", timeout=30)
    if not data or "data" not in data:
        return {}
    d = data["data"]
    try:
        return {
            "total_mcap_usd": float(d["total_market_cap"]["usd"]),
            "total_volume_usd": float(d["total_volume"]["usd"]),
            "mcap_chg_24h_pct": float(d.get("market_cap_change_percentage_24h_usd", 0.0)),
            "dominance": {k.upper(): float(v)
                          for k, v in (d.get("market_cap_percentage") or {}).items()},
            "n_coins": int(d.get("active_cryptocurrencies", 0)),
        }
    except (KeyError, TypeError, ValueError):
        log.exception("global stats had an unexpected shape")
        return {}


def binance_usdt_symbols() -> set:
    """Every spot symbol quoted in USDT that Binance is currently trading."""
    data = _get(BINANCE_SPOT_HOSTS, "/api/v3/ticker/24hr", timeout=45)
    if not data:
        return set()
    return {t["symbol"] for t in data
            if isinstance(t, dict) and str(t.get("symbol", "")).endswith("USDT")}


def build_universe(market: pd.DataFrame, tradable: set,
                   limit: int = 40,
                   include_commodities: bool = False) -> pd.DataFrame:
    """The coins that are both large enough to matter and priced on Binance.

    Restricted to what Binance quotes because everything downstream — beta,
    correlation, funding — needs a price history from the same venue. A coin
    ranked on one exchange and measured on another introduces a basis nobody
    asked for, and for the thin names that basis is not small.
    """
    if market.empty:
        return pd.DataFrame()

    df = market.copy()
    if not include_commodities:
        df = df[~df["tokenised_commodity"]]

    df["binance_symbol"] = [f"{s}USDT" for s in df.index]
    df = df[df["binance_symbol"].isin(tradable)]
    return df.head(limit)


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------

def closes_matrix(symbols: List[str], fetch_klines: Callable,
                  interval: str = "1d", limit: int = 400) -> pd.DataFrame:
    """Daily closes for a list of Binance symbols, on one date index.

    `fetch_klines` is injected so the caller's mirror fallback, caching and
    retry behaviour apply rather than being reimplemented here.

    Indexed by calendar date. Crypto trades every day, so unlike the equity
    side there are no weekend gaps to reconcile — but the index still has to
    be normalised, because anything joined against a stock or a metal later
    will have them.

    The timestamp is taken from the `open_time` column rather than from the
    frame's index, because this app's Binance client returns a plain row
    numbering with the time held as a column. Reading the index instead turned
    0, 1, 2 into nanoseconds after 1970, normalised four hundred bars onto a
    single day, and left a one-row matrix that every calculation downstream
    then declined to compute — a failure with no error attached to it.
    """
    frames: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            df = fetch_klines(sym, interval=interval, limit=limit)
        except Exception:
            log.warning("klines unavailable for %s", sym)
            continue
        if df is None or getattr(df, "empty", True):
            continue

        col = next((c for c in df.columns if str(c).lower() == "close"), None)
        if col is None:
            continue

        time_col = next((c for c in df.columns
                         if str(c).lower() in ("open_time", "time", "date",
                                               "timestamp")), None)
        idx = pd.to_datetime(df[time_col] if time_col is not None else df.index,
                             errors="coerce")

        s = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=idx)
        s = s[s.index.notna()].dropna()
        if s.empty:
            continue
        try:
            s.index = s.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="last")]

        # A series that arrives at the epoch, or collapses to a handful of
        # dates from hundreds of bars, has had its timestamps misread. Drop it
        # loudly rather than letting it quantly poison the matrix.
        if len(s) < 20 or s.index.min().year < 2000:
            log.warning("discarding %s: %d usable bars from %s",
                        sym, len(s), s.index.min())
            continue

        frames[sym.replace("USDT", "")] = s

    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


# --------------------------------------------------------------------------
# How much of this is bitcoin
# --------------------------------------------------------------------------

def btc_relationship(closes: pd.DataFrame, window: int = 180,
                     benchmark: str = "BTC") -> pd.DataFrame:
    """Each coin against bitcoin: beta, correlation, and variance explained.

    Beta says how far a coin travels for a given bitcoin move. R² says how
    much of the coin's movement bitcoin accounts for at all, and it is the
    more sobering number — a coin can have a modest beta and still be almost
    entirely a bitcoin position, because the beta describes the slope while
    the R² describes whether anything else is happening.

    `excess_1y` is what the coin returned beyond bitcoin over the window: the
    only part of holding it that the bitcoin position did not already provide.
    """
    if closes.empty or benchmark not in closes.columns:
        return pd.DataFrame()

    rets = np.log(closes / closes.shift(1)).replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all").tail(window)
    if len(rets) < 30:
        return pd.DataFrame()

    bench = rets[benchmark]
    bench_var = float(bench.var())
    rows = []

    for coin in closes.columns:
        if coin == benchmark:
            continue
        joined = pd.concat([rets[coin].rename("c"), bench.rename("b")],
                           axis=1).dropna()
        if len(joined) < 30:
            continue

        corr = float(joined["c"].corr(joined["b"]))
        beta = (float(joined["c"].cov(joined["b"]) / bench_var)
                if bench_var > 0 else None)

        px = closes[coin].dropna()
        bpx = closes[benchmark].dropna()
        common = px.index.intersection(bpx.index)
        excess = None
        if len(common) > 30:
            common = common[-min(window, len(common)):]
            c0, c1 = float(px.loc[common[0]]), float(px.loc[common[-1]])
            b0, b1 = float(bpx.loc[common[0]]), float(bpx.loc[common[-1]])
            if c0 > 0 and b0 > 0:
                excess = ((c1 / c0) - (b1 / b0)) * 100.0

        rows.append({
            "coin": coin, "beta": beta, "corr": corr,
            "r2": (corr ** 2 if corr is not None and np.isfinite(corr) else None),
            "excess_pct": excess, "n_obs": len(joined),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("coin").sort_values("beta", ascending=False)


def clusters(closes: pd.DataFrame, threshold: float = 0.8,
             window: int = 180) -> List[List[str]]:
    """Groups whose members move together above `threshold`.

    A higher bar than the FX and metals panels use, because crypto correlations
    sit high as a matter of course — at 0.7 nearly everything would come back
    as one group and the finding would carry no information.
    """
    if closes.empty or len(closes.columns) < 2:
        return []

    rets = np.log(closes / closes.shift(1)).replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all").tail(window)
    if len(rets) < 30:
        return []

    corr = rets.corr()
    names = list(corr.columns)
    seen: set = set()
    groups: List[List[str]] = []

    for a in names:
        if a in seen:
            continue
        group = [a]
        for b in names:
            if b == a or b in seen:
                continue
            try:
                if abs(float(corr.loc[a, b])) >= threshold:
                    group.append(b)
            except (KeyError, TypeError, ValueError):
                continue
        if len(group) > 1:
            groups.append(group)
            seen.update(group)
    return groups


def alt_index(closes: pd.DataFrame, benchmark: str = "BTC",
              window: int = 365) -> pd.DataFrame:
    """Bitcoin against an equal-weighted basket of everything else, rebased.

    A price-based stand-in for the dominance cycle. Historical dominance is
    not available without paying for it, but the question dominance answers —
    is money moving into bitcoin or out along the risk curve — is visible in
    the relative performance, and this version has the advantage of being
    computed from prices the panel already holds rather than taken on trust.

    Equal weights on purpose: capitalisation weights would make the basket
    ethereum with decoration, and the alt cycle is precisely a story about the
    smaller names.

    The basket is the median of the rebased series, not the mean. One coin
    twenty times higher on the year — there is usually one — moves a mean of
    thirty enough to carry the whole line on its own, and the resulting chart
    reports that single position as though it were the market. The median is
    the typical coin, which is the question being asked.
    """
    if closes.empty or benchmark not in closes.columns:
        return pd.DataFrame()

    df = closes.tail(window).dropna(axis=1, how="any")
    if benchmark not in df.columns or len(df.columns) < 3 or len(df) < 30:
        return pd.DataFrame()

    others = [c for c in df.columns if c != benchmark]
    rebased = df / df.iloc[0] * 100.0
    return pd.DataFrame({
        benchmark: rebased[benchmark],
        f"Median of the rest ({len(others)})": rebased[others].median(axis=1),
    })


# --------------------------------------------------------------------------
# Who is already in
# --------------------------------------------------------------------------

def funding_rates(symbols: List[str]) -> pd.DataFrame:
    """The current perpetual funding rate for each symbol, annualised.

    A perpetual future never expires, so nothing drags it back to spot except
    a payment: when it trades above spot, longs pay shorts, and when below,
    shorts pay longs. The rate is therefore a live public price for holding a
    leveraged long, quoted every eight hours.

    Annualised by multiplying by three payments a day and 365 days, which is
    how a desk compares it against anything else — but it is emphatically not
    a rate anyone earns for a year. It resets every eight hours and spends
    much of its time near zero.

    This is a measure of positioning, not a signal. Funding has stayed
    expensive through entire trends and has flipped negative at the bottom of
    them; it says who is currently paying to be in, not what happens next.
    """
    data = _get(BINANCE_FUTURES_HOSTS, "/fapi/v1/premiumIndex", timeout=40)
    if not data:
        return pd.DataFrame()

    latest = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        sym = row.get("symbol")
        if sym not in symbols:
            continue
        try:
            rate = float(row.get("lastFundingRate"))
            mark = float(row.get("markPrice"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rate):
            continue
        latest[sym.replace("USDT", "")] = {
            "funding_8h_pct": rate * 100.0,
            "funding_annual_pct": rate * FUNDINGS_PER_YEAR * 100.0,
            "mark_price": mark,
        }

    if not latest:
        return pd.DataFrame()
    out = pd.DataFrame(latest).T
    out.index.name = "coin"
    return out.sort_values("funding_annual_pct", ascending=False)


def funding_history(symbol: str, limit: int = 90) -> pd.Series:
    """Recent funding payments for one symbol, annualised.

    The history is what makes the current print readable. A single elevated
    reading is noise; the same reading held for three weeks is a crowd.
    """
    data = _get(BINANCE_FUTURES_HOSTS, "/fapi/v1/fundingRate",
                {"symbol": symbol, "limit": min(int(limit), 1000)}, timeout=30)
    if not data:
        return pd.Series(dtype=float)

    points = {}
    for row in data:
        try:
            ts = pd.to_datetime(int(row["fundingTime"]), unit="ms")
            points[ts] = float(row["fundingRate"]) * FUNDINGS_PER_YEAR * 100.0
        except (KeyError, TypeError, ValueError):
            continue
    if not points:
        return pd.Series(dtype=float)
    return pd.Series(points).sort_index()


def open_interest(symbols: List[str],
                  marks: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Positions outstanding per symbol, in coins and in dollars.

    Open interest counts positions that are still open, so it measures how
    much capital the market has committed rather than how much has changed
    hands. Rising open interest alongside a rising price is new money; rising
    price on falling open interest is a short squeeze closing out, and the two
    have opposite implications for what happens when the move stalls.

    Binance reports it in coins, which cannot be compared across symbols: a
    hundred thousand bitcoin and three hundred million XRP look like a
    difference of three thousand times and are in fact a difference of about
    two. Multiplying by the mark price puts every row in the same unit, and
    the dollar column is the only one worth ranking on.

    `marks` comes from funding_rates(), which already carries the mark price,
    so this needs no extra request per symbol.
    """
    marks = marks or {}
    rows = []
    for sym in symbols:
        data = _get(BINANCE_FUTURES_HOSTS, "/fapi/v1/openInterest",
                    {"symbol": sym}, timeout=20)
        if not data:
            continue
        try:
            coins = float(data["openInterest"])
        except (KeyError, TypeError, ValueError):
            continue
        coin = sym.replace("USDT", "")
        mark = marks.get(coin)
        rows.append({
            "coin": coin,
            "open_interest_coins": coins,
            "open_interest_usd": (coins * float(mark)
                                  if mark and np.isfinite(float(mark)) else None),
        })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).set_index("coin")
            .sort_values("open_interest_usd", ascending=False, na_position="last"))


# --------------------------------------------------------------------------
# Risk asset or digital gold
# --------------------------------------------------------------------------

MACRO_REFERENCES: Dict[str, str] = {
    "NASDAQ 100": "^NDX",
    "Gold": "GC=F",
    "Dollar index": "DX-Y.NYB",
    "S&P 500": "^GSPC",
}


def macro_correlation(btc: pd.Series, macro: pd.DataFrame,
                      window: int = 90) -> Dict[str, Any]:
    """Rolling correlation of bitcoin to the assets it is compared with.

    Bitcoin is argued about as two incompatible things — a long-duration risk
    asset that trades with the Nasdaq, and a monetary hedge that trades with
    gold. The argument is settled differently in different years, and it is
    measurable rather than a matter of opinion: this reports which of the two
    it has actually been tracking lately, and how that has moved.

    Crypto trades every day and the references do not, so the series are
    joined on the days both were open. Filling the weekend forward would
    manufacture two flat days a week in the reference and pull every
    correlation toward zero.
    """
    out: Dict[str, Any] = {"available": False, "window": window}
    btc = pd.to_numeric(btc, errors="coerce").dropna()
    if btc.empty or macro is None or macro.empty:
        return out

    btc_ret = np.log(btc / btc.shift(1)).replace([np.inf, -np.inf], np.nan)
    series: Dict[str, pd.Series] = {}
    latest: Dict[str, float] = {}

    for label, ticker in MACRO_REFERENCES.items():
        if ticker not in macro.columns:
            continue
        ref = pd.to_numeric(macro[ticker], errors="coerce").dropna()
        ref_ret = np.log(ref / ref.shift(1)).replace([np.inf, -np.inf], np.nan)
        joined = pd.concat([btc_ret.rename("btc"), ref_ret.rename("ref")],
                           axis=1).dropna()
        if len(joined) < window + 10:
            continue
        roll = joined["btc"].rolling(window).corr(joined["ref"]).dropna()
        if roll.empty:
            continue
        series[label] = roll
        latest[label] = float(roll.iloc[-1])

    if not series:
        return out

    out.update({"available": True, "series": pd.DataFrame(series),
                "latest": latest, "n_obs": int(len(next(iter(series.values()))))})

    equity = latest.get("NASDAQ 100")
    gold = latest.get("Gold")
    if equity is not None and gold is not None:
        gap = equity - gold
        if gap > 0.15:
            out["reading"] = "trading as a risk asset"
        elif gap < -0.15:
            out["reading"] = "trading as a monetary hedge"
        else:
            out["reading"] = "tracking neither more than the other"
        out["equity_minus_gold"] = float(gap)
    return out
