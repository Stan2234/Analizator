"""Fundamental analysis — valuation, profitability, growth, balance sheet.

Free of Streamlit so it can be tested directly.

The app measures price six ways and had nothing to say about the business
underneath it. That is the gap this fills, and the guiding idea is that a
fundamental number is almost meaningless on its own. A P/E of 35 is expensive
for a utility, ordinary for a software company and cheap for one compounding at
40%. So every metric here is reported next to its sector's distribution, and
the percentile is usually the more informative half.

Nothing here scores or ranks a company into a recommendation. A composite
"quality score" is exactly the kind of unvalidated construct this project has
been removing: it would look authoritative, and nothing measured whether it
predicts anything. The job here is to put the number in context and let the
reader judge.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("analizator.fundamentals")


# --------------------------------------------------------------------------
# What we pull, and what each field means
# --------------------------------------------------------------------------
# `higher_is_better` drives percentile direction only. None means the metric is
# genuinely ambiguous — leverage is not "bad", it is a choice whose merit
# depends on the business — so it is shown without a judgement attached.

class Metric:
    __slots__ = ("key", "label", "group", "fmt", "higher_is_better", "note")

    def __init__(self, key, label, group, fmt, higher_is_better, note):
        self.key, self.label, self.group = key, label, group
        self.fmt, self.higher_is_better, self.note = fmt, higher_is_better, note


METRICS: List[Metric] = [
    # -- valuation: what you pay ------------------------------------------
    Metric("trailingPE", "P/E (trailing)", "Valuation", "x", False,
           "Price over the last twelve months of earnings. Meaningless if earnings are negative or depressed."),
    Metric("forwardPE", "P/E (forward)", "Valuation", "x", False,
           "Price over expected earnings. Depends on analyst estimates, which are systematically optimistic."),
    Metric("priceToBook", "P/B", "Valuation", "x", False,
           "Price over book value. Informative for banks and asset-heavy firms, nearly useless for software."),
    Metric("priceToSalesTrailing12Months", "P/S", "Valuation", "x", False,
           "Price over revenue. The fallback when earnings are negative; ignores whether revenue converts to profit."),
    Metric("enterpriseToEbitda", "EV/EBITDA", "Valuation", "x", False,
           "Enterprise value over EBITDA. Capital-structure neutral, so it compares across different leverage."),
    # -- profitability: what the business earns ---------------------------
    Metric("grossMargins", "Gross margin", "Profitability", "%", True,
           "Revenue left after the direct cost of delivery. The ceiling on every margin below it."),
    Metric("operatingMargins", "Operating margin", "Profitability", "%", True,
           "Margin after running the business but before financing and tax."),
    Metric("profitMargins", "Net margin", "Profitability", "%", True,
           "What reaches the bottom line."),
    Metric("returnOnEquity", "Return on equity", "Profitability", "%", True,
           "Profit per unit of shareholder capital. Leverage inflates it, which is why the DuPont split matters."),
    # -- growth: which direction --------------------------------------------
    Metric("revenueGrowth", "Revenue growth", "Growth", "%", True,
           "Year-on-year revenue change, most recent quarter."),
    Metric("earningsGrowth", "Earnings growth", "Growth", "%", True,
           "Year-on-year earnings change. Volatile — a weak base year flatters it."),
    Metric("earningsQuarterlyGrowth", "Earnings growth (qtr)", "Growth", "%", True,
           "Quarterly year-on-year earnings change."),
    # -- balance sheet: what could break ------------------------------------
    Metric("debtToEquity", "Debt / equity", "Balance sheet", "raw", None,
           "Leverage. Not good or bad on its own — it raises returns and raises fragility."),
    Metric("currentRatio", "Current ratio", "Balance sheet", "x", True,
           "Current assets over current liabilities. Below 1 means short-term obligations exceed liquid assets."),
    Metric("totalCash", "Cash", "Balance sheet", "cur", None, "Cash and equivalents."),
    Metric("totalDebt", "Total debt", "Balance sheet", "cur", None, "All interest-bearing debt."),
    Metric("freeCashflow", "Free cash flow", "Balance sheet", "cur", True,
           "Cash left after capex. Harder to manage than reported earnings, which is why it is worth checking against them."),
    # -- shareholder returns / market view ----------------------------------
    Metric("dividendYield", "Dividend yield", "Shareholder returns", "%", None,
           "Annual dividend over price."),
    Metric("payoutRatio", "Payout ratio", "Shareholder returns", "%", None,
           "Share of earnings paid out. Above 1 means the dividend exceeds earnings."),
    Metric("beta", "Beta", "Market", "raw", None,
           "Historical sensitivity to the market. Backward-looking."),
]

METRIC_BY_KEY = {m.key: m for m in METRICS}

# Fields carried through without percentile treatment.
PASSTHROUGH = ["marketCap", "enterpriseValue", "trailingEps", "forwardEps",
               "targetMeanPrice", "recommendationKey", "numberOfAnalystOpinions",
               "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "sector", "industry",
               "shortName", "longName", "currentPrice", "regularMarketPrice"]

# yfinance reports ratios as fractions but a couple as percentages already.
# dividendYield comes back as 0.35 meaning 0.35%, not 35%.
_ALREADY_PERCENT = {"dividendYield"}


def normalise(info: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the fields we use out of a raw quote payload, in consistent units.

    Margins and growth arrive as fractions (0.276 = 27.6%) and are scaled to
    percent; dividend yield already arrives in percent and is left alone.
    """
    out: Dict[str, Any] = {}
    for m in METRICS:
        v = info.get(m.key)
        if v is None:
            out[m.key] = None
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            out[m.key] = None
            continue
        if m.fmt == "%" and m.key not in _ALREADY_PERCENT:
            v *= 100.0
        out[m.key] = v
    for k in PASSTHROUGH:
        out[k] = info.get(k)
    return out


# --------------------------------------------------------------------------
# Peer context
# --------------------------------------------------------------------------

# Yahoo classifies sectors with its own scheme; the app's universe is GICS.
# Left unmapped, a lookup for Apple's "Technology" finds nothing in a table
# that calls it "Information Technology", the peer set comes back empty, and
# every percentile silently disappears — which is how a comparison panel ends
# up showing a company against itself.
YF_SECTOR_TO_GICS: Dict[str, str] = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}

# Below this, the median is noise and a percentile is meaningless.
MIN_PEERS = 5


def gics_sector(yf_sector: Optional[str]) -> Optional[str]:
    """Translate a Yahoo sector label to the GICS name the universe uses."""
    if not yf_sector:
        return None
    return YF_SECTOR_TO_GICS.get(yf_sector, yf_sector)


def percentile_rank(value: Optional[float], peers: pd.Series,
                    higher_is_better: Optional[bool]) -> Optional[float]:
    """Where `value` sits in the peer distribution, 0-100.

    Oriented so 100 always reads as the favourable end when the metric has a
    favourable end: a low P/E and a high margin both score high. Metrics whose
    direction is genuinely ambiguous return the raw percentile, and the caller
    should present it without a verdict.
    """
    s = pd.to_numeric(peers, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if value is None or pd.isna(value) or len(s) < MIN_PEERS:
        return None
    raw = float((s < float(value)).mean() * 100.0)
    if higher_is_better is False:
        return 100.0 - raw
    return raw


def peer_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Normalised metrics for a list of raw quote payloads, indexed by symbol."""
    recs = []
    for r in rows:
        sym = r.get("symbol") or r.get("_symbol")
        if not sym:
            continue
        rec = normalise(r)
        rec["symbol"] = sym
        recs.append(rec)
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs).set_index("symbol")


def compare_to_peers(target: str, peers: pd.DataFrame) -> pd.DataFrame:
    """Every metric for `target` beside its peer median and percentile.

    The percentile is the point of the table. A P/E of 35 says nothing; a P/E
    in the 20th percentile of its sector says the market is paying up for this
    company relative to its peers, and that is a claim you can act on.
    """
    if peers.empty or target not in peers.index:
        return pd.DataFrame()

    rows = []
    for m in METRICS:
        if m.key not in peers.columns:
            continue
        col = peers[m.key]
        val = col.get(target)
        med = pd.to_numeric(col, errors="coerce").replace(
            [np.inf, -np.inf], np.nan).dropna().median()
        rows.append({
            "group": m.group,
            "metric": m.label,
            "value": None if val is None or pd.isna(val) else float(val),
            "peer_median": None if pd.isna(med) else float(med),
            "percentile": percentile_rank(val, col, m.higher_is_better),
            "directional": m.higher_is_better is not None,
            "fmt": m.fmt,
            "note": m.note,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# DuPont
# --------------------------------------------------------------------------

def dupont(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Split return on equity into margin, turnover and leverage.

    Two companies can post the same ROE for opposite reasons — one earns it on
    every sale, the other borrows to magnify a thin one. The split says which,
    and the second is far more fragile when conditions turn.

    Asset turnover is not published directly, so it is backed out of the
    identity ROE = margin x turnover x leverage. That makes it a residual: it
    absorbs any inconsistency between the three reported inputs, and is
    reported as derived rather than measured.
    """
    net_margin = rec.get("profitMargins")
    roe = rec.get("returnOnEquity")
    d_e = rec.get("debtToEquity")

    if net_margin is None or roe is None:
        return {}

    out: Dict[str, Any] = {"roe": roe, "net_margin": net_margin}
    # yfinance reports debt/equity as a percentage (78.4 means 0.784x).
    if d_e is not None:
        equity_multiplier = 1.0 + (float(d_e) / 100.0)
        out["equity_multiplier"] = equity_multiplier
        denom = net_margin * equity_multiplier
        if abs(denom) > 1e-9:
            # roe and net_margin are both in percent, so the ratio is unitless.
            out["asset_turnover"] = (roe / denom)
            out["derived"] = "asset_turnover"
    return out


# --------------------------------------------------------------------------
# Cross-checks a reader would otherwise have to do by hand
# --------------------------------------------------------------------------

def quality_flags(rec: Dict[str, Any]) -> List[Dict[str, str]]:
    """Observations worth a second look, each with the numbers behind it.

    These are not a score and not a verdict. Each is a condition that a person
    reading a filing would notice and want explained — the point is to surface
    it, not to conclude from it.
    """
    flags: List[Dict[str, str]] = []

    fcf, ni_margin = rec.get("freeCashflow"), rec.get("profitMargins")
    mcap = rec.get("marketCap")
    if fcf is not None and mcap:
        try:
            fcf_yield = float(fcf) / float(mcap) * 100.0
            flags.append({
                "level": "info",
                "text": f"Free cash flow yield {fcf_yield:.1f}% "
                        f"(FCF {fcf/1e9:.1f}bn on {float(mcap)/1e9:.0f}bn market cap). "
                        "Compare against the earnings yield — a large gap between the two "
                        "is where accounting and cash diverge."})
        except Exception:
            pass

    cur = rec.get("currentRatio")
    if cur is not None and cur < 1.0:
        flags.append({
            "level": "warn",
            "text": f"Current ratio {cur:.2f} — short-term liabilities exceed liquid "
                    "assets. Normal for businesses that collect cash before they pay "
                    "suppliers; a strain for those that do not."})

    payout = rec.get("payoutRatio")
    if payout is not None and payout > 100:
        flags.append({
            "level": "warn",
            "text": f"Payout ratio {payout:.0f}% — the dividend exceeds earnings, so it "
                    "is being funded from cash or debt rather than profit."})

    rg, eg = rec.get("revenueGrowth"), rec.get("earningsGrowth")
    if rg is not None and eg is not None:
        if eg > rg + 10:
            flags.append({
                "level": "info",
                "text": f"Earnings growing {eg:.0f}% against revenue {rg:.0f}% — margin "
                        "expansion or buybacks, not volume. Ask which, and whether it repeats."})
        elif rg > eg + 10:
            flags.append({
                "level": "info",
                "text": f"Revenue growing {rg:.0f}% against earnings {eg:.0f}% — growth is "
                        "not reaching the bottom line."})

    tpe, fpe = rec.get("trailingPE"), rec.get("forwardPE")
    if tpe and fpe and tpe > 0 and fpe > 0:
        if fpe < tpe * 0.8:
            flags.append({
                "level": "info",
                "text": f"Forward P/E {fpe:.1f} well below trailing {tpe:.1f} — the price "
                        "embeds a large expected earnings increase. That estimate is the "
                        "thing to test."})
        elif fpe > tpe * 1.2:
            flags.append({
                "level": "warn",
                "text": f"Forward P/E {fpe:.1f} above trailing {tpe:.1f} — analysts expect "
                        "earnings to fall."})

    roe, d_e = rec.get("returnOnEquity"), rec.get("debtToEquity")
    if roe is not None and d_e is not None and roe > 30 and d_e > 150:
        flags.append({
            "level": "warn",
            "text": f"Return on equity {roe:.0f}% on debt/equity of {d_e:.0f}% — much of "
                    "the return is leverage rather than operating performance. See the "
                    "DuPont split."})

    price = rec.get("currentPrice") or rec.get("regularMarketPrice")
    tgt = rec.get("targetMeanPrice")
    n_an = rec.get("numberOfAnalystOpinions")
    if price and tgt:
        try:
            gap = (float(tgt) / float(price) - 1) * 100
            flags.append({
                "level": "info",
                "text": f"Mean analyst target {float(tgt):.0f} is {gap:+.0f}% from the "
                        f"current price"
                        + (f", across {int(n_an)} estimates" if n_an else "")
                        + ". Sell-side targets cluster above spot as a rule, so read the "
                          "dispersion rather than the level."})
        except Exception:
            pass

    return flags


def range_position(rec: Dict[str, Any]) -> Optional[float]:
    """Where the price sits in its 52-week range, 0 = low, 100 = high."""
    price = rec.get("currentPrice") or rec.get("regularMarketPrice")
    lo, hi = rec.get("fiftyTwoWeekLow"), rec.get("fiftyTwoWeekHigh")
    try:
        price, lo, hi = float(price), float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if hi <= lo:
        return None
    return (price - lo) / (hi - lo) * 100.0


def sector_summary(peers: pd.DataFrame, keys: Optional[List[str]] = None) -> pd.DataFrame:
    """Median of each metric across the peer set, for a sector-level read."""
    if peers.empty:
        return pd.DataFrame()
    keys = keys or [m.key for m in METRICS]
    rows = []
    for k in keys:
        if k not in peers.columns:
            continue
        s = pd.to_numeric(peers[k], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        m = METRIC_BY_KEY[k]
        rows.append({"metric": m.label, "group": m.group, "fmt": m.fmt,
                     "median": float(s.median()), "p25": float(s.quantile(0.25)),
                     "p75": float(s.quantile(0.75)), "n": int(len(s))})
    return pd.DataFrame(rows)


def fmt_value(v: Optional[float], fmt: str) -> str:
    """Render a metric for display, in the units the metric is quoted in."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if fmt == "%":
        return f"{v:.1f}%"
    if fmt == "x":
        return f"{v:.1f}x"
    if fmt == "cur":
        av = abs(v)
        for div, suf in ((1e12, "T"), (1e9, "bn"), (1e6, "m")):
            if av >= div:
                return f"{v/div:.1f}{suf}"
        return f"{v:,.0f}"
    return f"{v:.2f}"
