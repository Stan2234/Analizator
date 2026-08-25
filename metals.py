"""Metals: what moved, and whether it was the metal or the money.

The question a metals desk actually asks is rarely "what is gold doing". Gold
is quoted in dollars, so a dollar that falls ten percent lifts the gold price
ten percent without a single ounce changing hands. Half the commentary written
about gold is really commentary about the dollar, and the two are separable —
priced in every major currency at once, the part of the move that belongs to
the metal and the part that belongs to the money fall apart cleanly.

That decomposition is the centre of this module. Around it sit the ratios a
desk reads for regime (gold to silver, copper to gold), the real price that
says whether a nominal record is a record, and the correlations that say what
the complex is currently trading on.

No composite score, in keeping with the rest of this project. A ratio at its
95th percentile is a fact about where it sits in its own history; what that
implies depends on why it got there, and averaging it against three other
facts would destroy the only thing that made it useful.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252


class Metal:
    __slots__ = ("name", "yahoo", "family", "unit", "note")

    def __init__(self, name: str, yahoo: str, family: str, unit: str, note: str):
        self.name, self.yahoo = name, yahoo
        self.family, self.unit, self.note = family, unit, note


# Futures rather than ETFs: an ETF carries a fee and a tracking difference, and
# for palladium and platinum the fund is small enough that its own flows show
# up in the price. The future is the price the trade actually clears at.
METALS: Dict[str, Metal] = {
    "Gold":      Metal("Gold", "GC=F", "precious", "$/troy oz",
                       "The monetary metal. Trades against real yields and the dollar "
                       "more than against its own supply."),
    "Silver":    Metal("Silver", "SI=F", "precious", "$/troy oz",
                       "Half monetary, half industrial — which is why it moves further "
                       "than gold in both directions."),
    "Platinum":  Metal("Platinum", "PL=F", "precious", "$/troy oz",
                       "Mostly industrial: autocatalysts and refining. Supply is "
                       "concentrated in South Africa."),
    "Palladium": Metal("Palladium", "PA=F", "precious", "$/troy oz",
                       "Almost entirely autocatalyst demand, and substitutable with "
                       "platinum, which caps how far the two can diverge."),
    "Copper":    Metal("Copper", "HG=F", "industrial", "$/lb",
                       "The growth metal. Read as a demand indicator before reading it "
                       "as a position."),
    "Aluminium": Metal("Aluminium", "ALI=F", "industrial", "$/tonne",
                       "Energy in solid form — smelting cost dominates, so it tracks "
                       "power prices as much as demand."),
}

# Mining equity against the metal it digs. The equity is a levered claim on the
# metal price net of a cost base, so it moves further in both directions — and
# when it stops doing so, the market is saying something about the cost base.
MINERS: Dict[str, Tuple[str, str]] = {
    "Gold miners":        ("GDX", "Gold"),
    "Junior gold miners": ("GDXJ", "Gold"),
    "Copper miners":      ("COPX", "Copper"),
}

# Thematic baskets. Held separately from the metals because none of them is a
# price of a metal — they are equity baskets whose relationship to the
# underlying commodity is loose and worth not implying.
THEMES: Dict[str, str] = {
    "Uranium":          "URA",
    "Lithium":          "LIT",
    "Rare earths":      "REMX",
    "Metals & mining":  "XME",
}

DRIVERS: Dict[str, str] = {
    "10y Treasury yield": "^TNX",
    "Dollar index":       "DX-Y.NYB",
    "S&P 500":            "^GSPC",
}

# The ratios a desk reads, and what each one is actually asking.
#
# `comparable` marks whether the two legs are quoted in the same unit. Gold
# against silver is two dollar-per-ounce prices and its level means something —
# sixty-nine is a number a desk recognises. Copper against gold is dollars per
# pound over dollars per ounce, and its level is an artefact of the contract
# specifications: 0.0014 says nothing to anyone. Those are shown as an index
# against their own median instead, so the reader sees the position rather than
# a number that only looks precise.
RATIOS: List[Dict[str, Any]] = [
    {"name": "Gold / Silver", "num": "Gold", "den": "Silver", "comparable": True,
     "asks": "Whether the precious complex is trading on fear or on reflation. "
             "Silver's industrial half sells off in a growth scare while gold's "
             "monetary half is bid, so the ratio widens when the market is "
             "frightened and compresses when it expects activity."},
    {"name": "Copper / Gold", "num": "Copper", "den": "Gold", "comparable": False,
     "asks": "Growth against fear, in one number. It has tracked the long bond "
             "closely enough that a divergence between the two is usually worth "
             "chasing down — one of them is early."},
    {"name": "Platinum / Gold", "num": "Platinum", "den": "Gold", "comparable": True,
     "asks": "Industrial demand against monetary demand. Platinum traded above "
             "gold for most of its history and has not since 2015; the ratio is "
             "a running measure of how much that regime has changed."},
    {"name": "Palladium / Platinum", "num": "Palladium", "den": "Platinum",
     "comparable": True,
     "asks": "Substitution pressure. The two compete in autocatalysts, so a wide "
             "spread creates its own correction as refiners switch."},
]


# --------------------------------------------------------------------------
# The board
# --------------------------------------------------------------------------

def bars_per_year(index: pd.Index) -> float:
    """How many observations a year this series actually carries.

    Never assume 252. Yahoo silently ignores `interval=1d` when asked for its
    maximum range and answers with monthly bars, and nothing in the response
    body distinguishes them from daily ones once the granularity field is
    dropped. Annualising monthly returns with the square root of 252 overstates
    volatility by a factor of four and a half, and the number that comes out
    looks entirely plausible — which is what makes it dangerous.

    Counted as observations over elapsed time, not from the spacing between
    them. The median gap between consecutive trading days is one day — Friday
    to Monday is the minority — so a spacing-based estimate answers 365 for a
    series that carries 252 bars a year, and annualised volatility comes out
    twenty percent too high. Counting over the span absorbs the weekends and
    holidays automatically, because they are exactly the days with no bar.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    if len(idx) < 3:
        return float(TRADING_DAYS)
    span_days = (idx[-1] - idx[0]).days
    if span_days <= 0:
        return float(TRADING_DAYS)
    return float(len(idx) / (span_days / 365.25))


def _ret(series: pd.Series, calendar_days: int) -> Optional[float]:
    """Percentage change over a calendar span, not a bar count.

    Counting back a fixed number of rows means "one week" is five bars, which
    is a week in daily data and five months in monthly data. Counting back a
    calendar span and taking the last bar on or before it means the label is
    true at any frequency — which matters, because the frequency is not always
    what was asked for.
    """
    s = series.dropna()
    if len(s) < 3:
        return None
    end = s.index[-1]
    target = end - pd.Timedelta(days=calendar_days)
    prior_idx = s.index[s.index <= target]
    if len(prior_idx) == 0:
        return None

    # And if the nearest bar sits far from the date asked for, decline. A
    # weekly return computed off monthly bars is a monthly return wearing the
    # wrong label, and half a horizon of slippage is enough to make the number
    # answer a different question than the column heading asks.
    actual = (end - prior_idx[-1]).days
    if abs(actual - calendar_days) > max(calendar_days * 0.5, 4):
        return None

    prior = float(s.loc[prior_idx[-1]])
    if prior == 0:
        return None
    return (float(s.iloc[-1]) / prior - 1.0) * 100.0


def board(prices: pd.DataFrame,
          universe: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Level, returns over several horizons, volatility and 52-week position.

    `prices` is keyed by Yahoo ticker. Position in the 52-week range is
    included because a return says how far something travelled and the range
    says where it arrived — an asset up 40% sitting mid-range has already given
    a good deal of it back, and the return alone hides that.
    """
    universe = universe or {m.name: m.yahoo for m in METALS.values()}
    rows = []

    for name, ticker in universe.items():
        if ticker not in prices.columns:
            continue
        s = pd.to_numeric(prices[ticker], errors="coerce").dropna()
        if len(s) < 30:
            continue

        bpy = bars_per_year(s.index)
        year = s[s.index >= s.index[-1] - pd.Timedelta(days=365)]
        hi, lo = float(year.max()), float(year.min())
        last = float(s.iloc[-1])
        rng = (last - lo) / (hi - lo) * 100.0 if hi > lo else None

        ytd = None
        this_year = s[s.index.year == s.index[-1].year]
        if len(this_year) > 1 and float(this_year.iloc[0]) != 0:
            ytd = (last / float(this_year.iloc[0]) - 1.0) * 100.0

        rets = np.log(s / s.shift(1)).dropna()
        recent = rets[rets.index >= rets.index[-1] - pd.Timedelta(days=365)]
        # Annualised with the frequency this series actually has, not the one
        # it was requested at.
        vol = (float(recent.std()) * np.sqrt(bpy) * 100.0
               if len(recent) >= 20 else None)

        meta = next((m for m in METALS.values() if m.yahoo == ticker), None)
        rows.append({
            "name": name, "ticker": ticker,
            "family": meta.family if meta else "",
            "unit": meta.unit if meta else "",
            "note": meta.note if meta else "",
            "last": last,
            "ret_1w": _ret(s, 7), "ret_1m": _ret(s, 30),
            "ret_3m": _ret(s, 91), "ret_ytd": ytd, "ret_1y": _ret(s, 365),
            "vol_1y": vol, "bars_per_year": round(bpy),
            "high_52w": hi, "low_52w": lo, "pct_of_range": rng,
            "as_of": s.index[-1].date().isoformat(),
        })

    return pd.DataFrame(rows).set_index("name") if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# Metal, or money?
# --------------------------------------------------------------------------

def in_currencies(metal: pd.Series, usd_values: pd.DataFrame,
                  days: int = 365) -> Dict[str, Any]:
    """Split a dollar-quoted metal's move into the metal's part and the dollar's.

    A metal quoted in dollars rises when the metal is bid and equally when the
    dollar is sold, and the price alone cannot tell those apart. Repricing it
    into every other currency can: the move that survives translation into
    seventeen currencies is the metal's, and the residual is the dollar's.

    Formally, since price_in_X = price_in_USD / (USD per X), the return in X is
    the dollar return less X's own return against the dollar. So the spread
    between the dollar return and the typical local-currency return *is* the
    dollar's broad move over the window.

    The median rather than the mean across currencies, because one managed or
    collapsing currency — a lira at thirty percent inflation — would otherwise
    drag the estimate on its own.

    Returns per-currency returns, the median local return, and the residual
    attributed to the dollar.
    """
    out: Dict[str, Any] = {"available": False}
    metal = pd.to_numeric(metal, errors="coerce").dropna()
    if metal.empty or usd_values is None or usd_values.empty:
        return out

    joined = pd.concat([metal.rename("_metal"), usd_values], axis=1).dropna(how="all")
    joined["_metal"] = joined["_metal"].ffill()
    joined = joined.dropna(subset=["_metal"])
    if len(joined) < 10:
        return out

    # Bounded by a calendar span rather than a row count, so the label on the
    # panel means the same thing whatever frequency the feed returned.
    cutoff = joined.index[-1] - pd.Timedelta(days=days)
    window = joined[joined.index >= cutoff]
    if len(window) < 5:
        return out
    start, end = window.iloc[0], window.iloc[-1]

    if not np.isfinite(start["_metal"]) or start["_metal"] == 0:
        return out
    usd_return = (end["_metal"] / start["_metal"] - 1.0) * 100.0

    per_ccy: Dict[str, float] = {}
    for ccy in usd_values.columns:
        if ccy == "USD":
            continue
        s0, s1 = start.get(ccy), end.get(ccy)
        if s0 is None or s1 is None or not np.isfinite(s0) or not np.isfinite(s1) or s0 == 0:
            continue
        # price in ccy = price in USD / (USD per ccy)
        p0, p1 = start["_metal"] / s0, end["_metal"] / s1
        if p0 == 0:
            continue
        per_ccy[ccy] = (p1 / p0 - 1.0) * 100.0

    if len(per_ccy) < 3:
        return out

    median_local = float(np.median(list(per_ccy.values())))
    out.update({
        "available": True,
        "days": days,
        "start": window.index[0].date().isoformat(),
        "end": window.index[-1].date().isoformat(),
        "usd_return_pct": float(usd_return),
        "per_currency": dict(sorted(per_ccy.items(), key=lambda kv: -kv[1])),
        "median_local_pct": median_local,
        "dollar_contribution_pct": float(usd_return - median_local),
        "n_currencies": len(per_ccy),
    })

    # Which of the two did the work. Stated as a share of the dollar move, and
    # only when that move is large enough for a share to mean anything.
    if abs(usd_return) > 1.0:
        share = (usd_return - median_local) / usd_return * 100.0
        out["dollar_share_pct"] = float(share)
        if share >= 60:
            out["reading"] = "mostly the dollar"
        elif share <= 20:
            out["reading"] = "mostly the metal"
        else:
            out["reading"] = "both, in similar measure"
    return out


def in_currencies_series(metal: pd.Series, usd_values: pd.DataFrame,
                         currencies: List[str]) -> pd.DataFrame:
    """The metal repriced into each named currency, rebased to 100.

    Rebasing is what makes the chart readable: an ounce costs a few thousand
    dollars and a few hundred thousand lira, and on one axis the comparison is
    invisible.
    """
    metal = pd.to_numeric(metal, errors="coerce").dropna()
    if metal.empty or usd_values is None or usd_values.empty:
        return pd.DataFrame()

    joined = pd.concat([metal.rename("_metal"), usd_values], axis=1)
    joined["_metal"] = joined["_metal"].ffill()
    joined = joined.dropna(subset=["_metal"])

    out: Dict[str, pd.Series] = {}
    for ccy in currencies:
        if ccy == "USD":
            s = joined["_metal"]
        elif ccy in joined.columns:
            s = joined["_metal"] / pd.to_numeric(joined[ccy], errors="coerce")
        else:
            continue
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty or float(s.iloc[0]) == 0:
            continue
        out[ccy] = s / float(s.iloc[0]) * 100.0

    return pd.DataFrame(out).dropna(how="all")


# --------------------------------------------------------------------------
# Ratios
# --------------------------------------------------------------------------

def ratio(prices: pd.DataFrame, numerator: str, denominator: str) -> Dict[str, Any]:
    """One ratio, with where it sits in its own history.

    The percentile is over whatever history the feed provides, which is stated
    rather than assumed — a ratio in its 95th percentile since 2000 is a
    different claim from one in its 95th percentile since 1970, and only the
    first is available here.
    """
    out: Dict[str, Any] = {"available": False,
                           "numerator": numerator, "denominator": denominator}

    num_m, den_m = METALS.get(numerator), METALS.get(denominator)
    if not num_m or not den_m:
        return out
    if num_m.yahoo not in prices.columns or den_m.yahoo not in prices.columns:
        return out

    a = pd.to_numeric(prices[num_m.yahoo], errors="coerce")
    b = pd.to_numeric(prices[den_m.yahoo], errors="coerce")
    s = (a / b.where(b > 0)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 60:
        return out

    last = float(s.iloc[-1])
    out.update({
        "available": True,
        "series": s,
        "last": last,
        "percentile": float((s < last).mean() * 100.0),
        "median": float(s.median()),
        "min": float(s.min()), "max": float(s.max()),
        "from": s.index[0].date().isoformat(),
        "to": s.index[-1].date().isoformat(),
        "ret_1y": _ret(s, 365),
        "vs_median_pct": (last / float(s.median()) - 1.0) * 100.0 if s.median() else None,
    })
    return out


def ratio_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Every ratio in RATIOS, with its level, percentile and distance from median.

    `display` carries the level a reader should be shown: the ratio itself
    where the two legs share a unit, and an index against its own median where
    they do not. `comparable` says which, so the panel can label it.
    """
    rows = []
    for spec in RATIOS:
        r = ratio(prices, spec["num"], spec["den"])
        if not r.get("available"):
            continue
        comparable = bool(spec.get("comparable", True))
        median = r["median"]
        display = r["last"] if comparable else (
            (r["last"] / median * 100.0) if median else None)
        rows.append({
            "ratio": spec["name"], "asks": spec["asks"],
            "comparable": comparable,
            "last": r["last"], "display": display,
            "percentile": r["percentile"],
            "median": median, "vs_median_pct": r["vs_median_pct"],
            "ret_1y": r["ret_1y"], "from": r["from"],
        })
    return pd.DataFrame(rows).set_index("ratio") if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# Real prices
# --------------------------------------------------------------------------

def real_price(metal: pd.Series, cpi: pd.Series) -> Dict[str, Any]:
    """A metal deflated by consumer prices, in today's money.

    Whether a nominal record is a real record is a different question with a
    different answer, and it is the second one that says something about the
    metal. Deflating to the latest month rather than to some base year means
    the real series is quoted in money the reader currently holds.
    """
    out: Dict[str, Any] = {"available": False}
    metal = pd.to_numeric(metal, errors="coerce").dropna()
    cpi = pd.to_numeric(cpi, errors="coerce").dropna()
    if metal.empty or cpi.empty:
        return out

    # CPI is monthly and published in arrears; the metal is daily. Reindexing
    # the index forward onto trading days holds each month's level until the
    # next prints, which is what it means, and stops short of extrapolating it.
    monthly = cpi.resample("MS").last()
    aligned = monthly.reindex(
        monthly.index.union(metal.index)).ffill().reindex(metal.index)
    aligned = aligned.dropna()
    if aligned.empty:
        return out

    common = metal.loc[aligned.index]
    latest_cpi = float(aligned.iloc[-1])
    real = common * (latest_cpi / aligned)

    nominal_peak_at = common.idxmax()
    real_peak_at = real.idxmax()
    last_real = float(real.iloc[-1])

    out.update({
        "available": True,
        "nominal": common, "real": real,
        # The month the deflator actually last printed, not the last day the
        # forward-fill reached. Reporting the filled date would date today's
        # price level to a CPI reading that does not exist yet, and the gap is
        # exactly the window in which a real series is least reliable.
        "cpi_last": monthly.index[-1].date().isoformat(),
        "cpi_lag_days": int((common.index[-1] - monthly.index[-1]).days),
        "nominal_last": float(common.iloc[-1]),
        "real_last": last_real,
        "nominal_peak": float(common.max()),
        "nominal_peak_date": nominal_peak_at.date().isoformat(),
        "real_peak": float(real.max()),
        "real_peak_date": real_peak_at.date().isoformat(),
        "below_real_peak_pct": (last_real / float(real.max()) - 1.0) * 100.0,
        "from": common.index[0].date().isoformat(),
        "at_nominal_high": bool(nominal_peak_at == common.index[-1]),
        "at_real_high": bool(real_peak_at == real.index[-1]),
    })
    return out


# --------------------------------------------------------------------------
# What the complex is trading on
# --------------------------------------------------------------------------

def correlations(prices: pd.DataFrame, tickers: Dict[str, str],
                 window: int = 182) -> pd.DataFrame:
    """Correlation of daily returns between everything named in `tickers`.

    Returns rather than levels. Two rising series correlate near one whatever
    they are, which measures the fact that both went up and nothing else.
    """
    cols = {name: t for name, t in tickers.items() if t in prices.columns}
    if len(cols) < 2:
        return pd.DataFrame()

    frame = pd.DataFrame({
        name: pd.to_numeric(prices[t], errors="coerce")
        for name, t in cols.items()})
    rets = np.log(frame / frame.shift(1)).replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all")
    if rets.empty:
        return pd.DataFrame()
    rets = rets[rets.index >= rets.index[-1] - pd.Timedelta(days=window)]
    if len(rets) < 20:
        return pd.DataFrame()
    return rets.corr()


def driver_betas(prices: pd.DataFrame, window: int = 365) -> pd.DataFrame:
    """Each metal against each macro driver: correlation over the window.

    Reported as correlation rather than beta because the drivers are not in
    comparable units — a yield moves in percentage points and an index in
    percent, so a beta between them is a number without a meaning.
    """
    rows = []
    for name, metal in METALS.items():
        if metal.yahoo not in prices.columns:
            continue
        m = np.log(pd.to_numeric(prices[metal.yahoo], errors="coerce")).diff()
        row: Dict[str, Any] = {"metal": name}
        for dname, dticker in DRIVERS.items():
            if dticker not in prices.columns:
                row[dname] = None
                continue
            d = np.log(pd.to_numeric(prices[dticker], errors="coerce")).diff()
            joined = pd.concat([m, d], axis=1).replace(
                [np.inf, -np.inf], np.nan).dropna()
            if not joined.empty:
                joined = joined[joined.index >= joined.index[-1]
                                - pd.Timedelta(days=window)]
            row[dname] = (float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
                          if len(joined) >= 20 else None)
        rows.append(row)
    return pd.DataFrame(rows).set_index("metal") if rows else pd.DataFrame()


def miner_spread(prices: pd.DataFrame, window: int = 365) -> pd.DataFrame:
    """Mining equity against its metal: relative performance and sensitivity.

    A miner is a levered claim on the metal net of a cost base, so it should
    move further than the metal in both directions. `beta` measures how much
    further. When it falls toward one while the metal rallies, the market is
    pricing the cost base eating the upside — which is information about the
    industry that the metal price cannot carry.
    """
    rows = []
    for label, (ticker, metal_name) in MINERS.items():
        metal = METALS.get(metal_name)
        if not metal or ticker not in prices.columns or metal.yahoo not in prices.columns:
            continue

        eq = np.log(pd.to_numeric(prices[ticker], errors="coerce")).diff()
        mt = np.log(pd.to_numeric(prices[metal.yahoo], errors="coerce")).diff()
        joined = pd.concat([eq.rename("eq"), mt.rename("mt")], axis=1)
        joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
        if joined.empty:
            continue
        joined = joined[joined.index >= joined.index[-1] - pd.Timedelta(days=window)]
        if len(joined) < 20:
            continue

        var = float(joined["mt"].var())
        beta = float(joined["eq"].cov(joined["mt"]) / var) if var > 0 else None

        eq_px = pd.to_numeric(prices[ticker], errors="coerce").dropna()
        mt_px = pd.to_numeric(prices[metal.yahoo], errors="coerce").dropna()
        rows.append({
            "pair": label, "metal": metal_name, "ticker": ticker,
            "beta": beta,
            "corr": float(joined["eq"].corr(joined["mt"])),
            "equity_1y": _ret(eq_px, window),
            "metal_1y": _ret(mt_px, window),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("pair")
    out["excess_1y"] = out["equity_1y"] - out["metal_1y"]
    return out


def clustered(corr: pd.DataFrame, threshold: float = 0.7) -> List[List[str]]:
    """Groups whose members correlate above `threshold` with each other.

    The same warning the FX panel carries applies to metals: three positions in
    one cluster are one position sized three times, and they will not diversify
    when it matters because the thing that moves them is the same thing.
    """
    if corr is None or corr.empty:
        return []

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
