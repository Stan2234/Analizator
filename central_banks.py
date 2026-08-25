"""Central bank policy: the rate, how it got there, and what it costs in real terms.

Everything here comes from the institutions themselves rather than from a
hand-maintained table. That distinction is not pedantry. The seeded rates this
module replaces were wrong for twelve of seventeen currencies at the moment
they were checked — the dollar by 87bp, the zloty by 200bp — and every one of
them fed the carry arithmetic downstream. A policy rate is checkable, a bank
analyst will check it, and a number that is merely plausible is worse than no
number because nothing about it looks wrong.

Sources, all free and none requiring a key:

  BIS WS_CBPOL      policy rates for 38 central banks, daily, full history.
                    The BIS is where central banks themselves publish, so it
                    is the reference the desks use.
  BIS WS_LONG_CPI   consumer prices, monthly, as both an index and a published
                    year-on-year rate — which lets the two be checked against
                    each other rather than trusted.
  NY Fed / ECB / BoE  the three largest banks, read directly, so the BIS
                    figure is confirmed by a second independent source rather
                    than assumed correct.
  World Gold Council  official gold holdings. The only file here, because the
                    download sits behind a login; see gold_holdings().
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 45
_UA = {"User-Agent": "Analizator/1.0 (macro dashboard)"}

BIS_BASE = "https://stats.bis.org/api/v1/data"


# --------------------------------------------------------------------------
# Who is who
# --------------------------------------------------------------------------
# The instrument matters as much as the number. "The policy rate" is a
# different animal in each jurisdiction — a target range midpoint in the US, a
# deposit facility in the euro area, a one-week repo in Turkey — and comparing
# them without saying which is which invites a false precision. The BIS
# normalises to the operative rate, and the note records what that is.

class CB:
    __slots__ = ("ccy", "area", "bank", "instrument", "region")

    def __init__(self, ccy: str, area: str, bank: str, instrument: str, region: str):
        self.ccy, self.area = ccy, area
        self.bank, self.instrument, self.region = bank, instrument, region


CENTRAL_BANKS: Dict[str, CB] = {
    "USD": CB("USD", "US", "Federal Reserve",        "Fed funds target midpoint", "G10"),
    "EUR": CB("EUR", "XM", "European Central Bank",  "Deposit facility rate",     "G10"),
    "JPY": CB("JPY", "JP", "Bank of Japan",          "Short-term policy rate",    "G10"),
    "GBP": CB("GBP", "GB", "Bank of England",        "Bank Rate",                 "G10"),
    "CHF": CB("CHF", "CH", "Swiss National Bank",    "SNB policy rate",           "G10"),
    "AUD": CB("AUD", "AU", "Reserve Bank of Australia", "Cash rate target",       "G10"),
    "CAD": CB("CAD", "CA", "Bank of Canada",         "Overnight rate target",     "G10"),
    "NZD": CB("NZD", "NZ", "Reserve Bank of New Zealand", "Official cash rate",   "G10"),
    "SEK": CB("SEK", "SE", "Sveriges Riksbank",      "Policy rate",               "G10"),
    "NOK": CB("NOK", "NO", "Norges Bank",            "Policy rate",               "G10"),
    "CNY": CB("CNY", "CN", "People's Bank of China", "1-year loan prime rate",    "EM"),
    "INR": CB("INR", "IN", "Reserve Bank of India",  "Repo rate",                 "EM"),
    "ZAR": CB("ZAR", "ZA", "South African Reserve Bank", "Repo rate",             "EM"),
    "TRY": CB("TRY", "TR", "Central Bank of Türkiye", "One-week repo rate",       "EM"),
    "BRL": CB("BRL", "BR", "Banco Central do Brasil", "Selic target",             "EM"),
    "MXN": CB("MXN", "MX", "Banco de México",        "Overnight target rate",     "EM"),
    "PLN": CB("PLN", "PL", "Narodowy Bank Polski",   "Reference rate",            "EM"),
}

AREA_TO_CCY: Dict[str, str] = {cb.area: c for c, cb in CENTRAL_BANKS.items()}

# A rate that has not printed in this long is almost certainly a feed problem
# rather than a very quiet central bank.
RATE_STALE_DAYS = 21


# --------------------------------------------------------------------------
# BIS
# --------------------------------------------------------------------------

def _bis_get(flow: str, key: str, **params) -> Optional[str]:
    """One BIS query. Returns the raw SDMX body, or None if it failed."""
    try:
        r = requests.get(f"{BIS_BASE}/{flow}/{key}/all",
                         params=params, headers=_UA, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            log.warning("BIS %s/%s -> HTTP %s", flow, key, r.status_code)
            return None
        return r.text
    except Exception:
        log.exception("BIS request failed for %s/%s", flow, key)
        return None


def _parse_sdmx(xml_text: str) -> Dict[str, List[Tuple[str, float]]]:
    """SDMX structure-specific XML -> {ref_area: [(period, value), ...]}.

    Values that come back non-numeric are dropped rather than coerced. BIS
    pads series forward with empty observations, so the last row of a series
    is frequently not the last real reading — taking it on faith would date a
    live rate to a period that has not happened yet.
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.exception("BIS returned unparseable XML")
        return out

    for series in root.iter():
        if series.tag.split("}")[-1] != "Series":
            continue
        area = series.attrib.get("REF_AREA")
        if not area:
            continue
        pairs: List[Tuple[str, float]] = []
        for obs in series:
            if obs.tag.split("}")[-1] != "Obs":
                continue
            period, raw = obs.attrib.get("TIME_PERIOD"), obs.attrib.get("OBS_VALUE")
            if not period or raw in (None, "", "NaN"):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                pairs.append((period, value))
        if pairs:
            out.setdefault(area, []).extend(pairs)

    for area in out:
        out[area].sort(key=lambda p: p[0])
    return out


def policy_rate_history(years: int = 6,
                        currencies: Optional[List[str]] = None
                        ) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Daily policy rate per currency, plus each one's true last observation.

    Forward-filling is right here and wrong almost everywhere else: a policy
    rate genuinely is a step function that holds its level between decisions,
    so the filled value is the rate that was actually in force, not an
    interpolation standing in for a missing one.

    But the fill must not be allowed to launder the vintage. Carrying the
    series to today makes every rate look like it was confirmed today, when
    the BIS may last have published several days ago and a decision could have
    landed in between. The real last observation per currency is therefore
    returned alongside the frame — as a second value rather than in `.attrs`,
    which Streamlit's cache discards.
    """
    codes = list(currencies or CENTRAL_BANKS.keys())
    areas = "+".join(sorted({CENTRAL_BANKS[c].area for c in codes if c in CENTRAL_BANKS}))
    start = (dt.date.today() - dt.timedelta(days=int(years * 365.25))).isoformat()

    body = _bis_get("WS_CBPOL", f"D.{areas}", startPeriod=start)
    if not body:
        return pd.DataFrame(), {}

    parsed = _parse_sdmx(body)
    series: Dict[str, pd.Series] = {}
    observed: Dict[str, str] = {}
    for area, pairs in parsed.items():
        ccy = AREA_TO_CCY.get(area)
        if not ccy:
            continue
        s = pd.Series({pd.Timestamp(p): v for p, v in pairs}).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        series[ccy] = s
        observed[ccy] = s.index.max().date().isoformat()

    if not series:
        return pd.DataFrame(), {}

    frame = pd.DataFrame(series).sort_index()
    full = pd.date_range(frame.index.min(), dt.date.today(), freq="D")
    return frame.reindex(full).ffill(), observed


def rate_changes(hist: pd.DataFrame, ccy: str) -> List[Dict[str, Any]]:
    """Every move in the rate, as {date, from, to, bp}.

    A change is a change in the level, found by differencing, so this reports
    what the bank did rather than when it met. Meetings that held are absent
    by construction — which is the point, since a hold leaves no trace in the
    rate and cannot be recovered from it.
    """
    if hist.empty or ccy not in hist.columns:
        return []
    s = hist[ccy].dropna()
    if s.empty:
        return []

    moves = []
    diff = s.diff()
    for ts, d in diff.items():
        if pd.isna(d) or abs(d) < 1e-9:
            continue
        prev = s.loc[:ts].iloc[-2] if len(s.loc[:ts]) > 1 else np.nan
        moves.append({"date": ts, "from": float(prev), "to": float(s.loc[ts]),
                      "bp": round(float(d) * 100, 1)})
    return moves


def cycle_summary(hist: pd.DataFrame, ccy: str, lookback_years: int = 6,
                  observed: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Where the rate is relative to its own cycle.

    The level alone says little — 4% is restrictive coming from 1% and easy
    coming from 8%. What positions it is the distance from the cycle peak and
    the direction of the last move, so those are what this returns.

    `observed` carries the true last publication date from
    policy_rate_history(); without it the forward-filled index would date
    every rate to today.
    """
    out: Dict[str, Any] = {"ccy": ccy}
    if hist.empty or ccy not in hist.columns:
        return out
    s = hist[ccy].dropna()
    if s.empty:
        return out

    cutoff = s.index.max() - pd.Timedelta(days=int(lookback_years * 365.25))
    window = s[s.index >= cutoff]
    if window.empty:
        window = s

    today = s.index.max()
    rate = float(s.iloc[-1])
    moves = rate_changes(hist, ccy)

    as_of = (observed or {}).get(ccy) or today.date().isoformat()
    try:
        stale = (dt.date.today() - dt.date.fromisoformat(as_of)).days
    except ValueError:
        stale = None

    out.update({
        "rate": rate,
        "as_of": as_of,
        "stale_days": stale,
        "peak": float(window.max()),
        "peak_date": window.idxmax().date().isoformat(),
        "trough": float(window.min()),
        "trough_date": window.idxmin().date().isoformat(),
        "from_peak_bp": round((rate - float(window.max())) * 100, 1),
        "from_trough_bp": round((rate - float(window.min())) * 100, 1),
        "n_moves": len(moves),
    })

    if moves:
        last = moves[-1]
        out.update({
            "last_change": last["date"].date().isoformat(),
            "last_change_bp": last["bp"],
            "days_since_change": int((today - last["date"]).days),
            "direction": "hiking" if last["bp"] > 0 else "cutting",
        })
        year_ago = today - pd.Timedelta(days=365)
        recent = [m for m in moves if m["date"] >= year_ago]
        out["bp_12m"] = round(sum(m["bp"] for m in recent), 1)
        out["moves_12m"] = len(recent)
    else:
        out.update({"last_change": None, "direction": "unchanged on record",
                    "bp_12m": 0.0, "moves_12m": 0})

    return out


# --------------------------------------------------------------------------
# Inflation, and the real rate
# --------------------------------------------------------------------------

def inflation(currencies: Optional[List[str]] = None) -> pd.DataFrame:
    """Year-on-year CPI per currency, with the published rate checked.

    BIS carries consumer prices twice: as an index (unit 628) and as an
    already-computed year-on-year rate (unit 771). Deriving the change from
    the index and comparing it against the published rate costs one extra
    subtraction and turns an assumption into a test — if a country changes
    base year or revises, the two part company and the row says so instead of
    quietly reporting a number nobody checked.

    Columns: cpi_yoy, as_of, derived_yoy, check (ok | mismatch | underived).
    """
    codes = list(currencies or CENTRAL_BANKS.keys())
    areas = "+".join(sorted({CENTRAL_BANKS[c].area for c in codes if c in CENTRAL_BANKS}))
    start = (dt.date.today() - dt.timedelta(days=900)).isoformat()

    body = _bis_get("WS_LONG_CPI", f"M.{areas}", startPeriod=start)
    if not body:
        return pd.DataFrame()

    # The two units share a REF_AREA, so they must be separated by unit before
    # being parsed together.
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        log.exception("BIS CPI returned unparseable XML")
        return pd.DataFrame()

    published: Dict[str, List[Tuple[str, float]]] = {}
    index: Dict[str, List[Tuple[str, float]]] = {}

    for series in root.iter():
        if series.tag.split("}")[-1] != "Series":
            continue
        area = series.attrib.get("REF_AREA")
        unit = series.attrib.get("UNIT_MEASURE")
        if not area or unit not in ("628", "771"):
            continue
        target = index if unit == "628" else published
        pairs: List[Tuple[str, float]] = []
        for obs in series:
            if obs.tag.split("}")[-1] != "Obs":
                continue
            period, raw = obs.attrib.get("TIME_PERIOD"), obs.attrib.get("OBS_VALUE")
            if not period or raw in (None, "", "NaN"):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                pairs.append((period, value))
        if pairs:
            target.setdefault(area, []).extend(sorted(pairs))

    rows = []
    for ccy in codes:
        cb = CENTRAL_BANKS.get(ccy)
        if not cb:
            continue
        pub = published.get(cb.area) or []
        idx = dict(index.get(cb.area) or [])
        if not pub:
            continue

        period, yoy = pub[-1]
        derived = None
        year, month = period.split("-")[0], period.split("-")[1]
        prior = f"{int(year) - 1}-{month}"
        if period in idx and prior in idx and idx[prior]:
            derived = (idx[period] / idx[prior] - 1.0) * 100.0

        if derived is None:
            check = "underived"
        elif abs(derived - yoy) <= 0.15:
            check = "ok"
        else:
            check = "mismatch"

        rows.append({"ccy": ccy, "cpi_yoy": yoy, "as_of": period,
                     "derived_yoy": derived, "check": check})

    return pd.DataFrame(rows).set_index("ccy") if rows else pd.DataFrame()


def policy_board(hist: pd.DataFrame, infl: pd.DataFrame,
                 currencies: Optional[List[str]] = None,
                 observed: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """One row per central bank: the rate, its cycle, and the real rate.

    The real rate is the nominal policy rate less realised year-on-year
    inflation — ex-post, not ex-ante. The forward-looking version would need
    inflation expectations, which exist as a traded instrument for a handful
    of countries and as a survey for the rest, and mixing the two would make
    the column incomparable across the very rows it exists to compare. Stated
    plainly on the panel rather than hidden in a definition.
    """
    codes = list(currencies or CENTRAL_BANKS.keys())
    rows = []

    for ccy in codes:
        cb = CENTRAL_BANKS.get(ccy)
        if not cb:
            continue
        cyc = cycle_summary(hist, ccy, observed=observed)
        row: Dict[str, Any] = {
            "ccy": ccy, "bank": cb.bank, "instrument": cb.instrument,
            "region": cb.region,
            "rate": cyc.get("rate"), "as_of": cyc.get("as_of"),
            "stale_days": cyc.get("stale_days"),
            "direction": cyc.get("direction"),
            "last_change": cyc.get("last_change"),
            "last_change_bp": cyc.get("last_change_bp"),
            "days_since_change": cyc.get("days_since_change"),
            "bp_12m": cyc.get("bp_12m"), "moves_12m": cyc.get("moves_12m"),
            "peak": cyc.get("peak"), "peak_date": cyc.get("peak_date"),
            "from_peak_bp": cyc.get("from_peak_bp"),
        }

        if not infl.empty and ccy in infl.index:
            i = infl.loc[ccy]
            row["cpi_yoy"] = float(i["cpi_yoy"])
            row["cpi_as_of"] = i["as_of"]
            row["cpi_check"] = i["check"]
            if row["rate"] is not None:
                row["real_rate"] = float(row["rate"]) - float(i["cpi_yoy"])
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    board = pd.DataFrame(rows).set_index("ccy")
    # A board where nothing has a rate is not a board with empty cells, it is
    # an absent board. Returning the shell would let a caller render seventeen
    # rows of dashes as though the banks had been polled and said nothing.
    if board["rate"].isna().all():
        return pd.DataFrame()
    return board


# --------------------------------------------------------------------------
# Independent confirmation of the three largest
# --------------------------------------------------------------------------
# The BIS is a compiler, not the issuing institution, and a compiler can lag a
# decision by a day or carry a convention the reader does not expect. Reading
# the three biggest banks directly turns "the BIS says" into "two independent
# sources agree", which is the difference between a number an analyst can use
# in a meeting and one they have to go and check first.

def crosscheck_fed() -> Dict[str, Any]:
    """Fed target range and effective rate, from the New York Fed."""
    out: Dict[str, Any] = {"ccy": "USD", "source": "Federal Reserve Bank of New York"}
    try:
        r = requests.get("https://markets.newyorkfed.org/api/rates/unsecured/effr/last/1.json",
                         headers=_UA, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code}"
            return out
        recs = (r.json() or {}).get("refRates") or []
        if not recs:
            out["error"] = "no observations"
            return out
        rec = recs[0]
        lo, hi = rec.get("targetRateFrom"), rec.get("targetRateTo")
        out.update({
            "as_of": rec.get("effectiveDate"),
            "effective": rec.get("percentRate"),
            "target_low": lo, "target_high": hi,
            "rate": (float(lo) + float(hi)) / 2 if lo is not None and hi is not None else None,
            "note": f"target range {lo}–{hi}%, effective {rec.get('percentRate')}%",
        })
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


def crosscheck_ecb() -> Dict[str, Any]:
    """ECB deposit facility rate, from the ECB Data Portal."""
    out: Dict[str, Any] = {"ccy": "EUR", "source": "European Central Bank"}
    try:
        r = requests.get("https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV",
                         params={"format": "csvdata", "lastNObservations": "1"},
                         headers=_UA, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code}"
            return out
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            out["error"] = "no observations"
            return out
        row = df.iloc[-1]
        out.update({"as_of": str(row["TIME_PERIOD"]), "rate": float(row["OBS_VALUE"]),
                    "note": "deposit facility rate"})
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


def crosscheck_boe() -> Dict[str, Any]:
    """Bank Rate, from the Bank of England's statistical database."""
    out: Dict[str, Any] = {"ccy": "GBP", "source": "Bank of England"}
    try:
        start = (dt.date.today() - dt.timedelta(days=60)).strftime("%d/%b/%Y")
        r = requests.get("https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp",
                         params={"csv.x": "yes", "Datefrom": start, "Dateto": "now",
                                 "SeriesCodes": "IUDBEDR", "CSVF": "TN",
                                 "UsingCodes": "Y", "VPD": "Y", "VFD": "N"},
                         headers=_UA, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code}"
            return out
        lines = [ln for ln in r.text.strip().splitlines() if "," in ln]
        if len(lines) < 2:
            out["error"] = "no observations"
            return out
        date_str, value = lines[-1].rsplit(",", 1)
        out.update({"as_of": date_str.strip(), "rate": float(value),
                    "note": "Bank Rate"})
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


CROSSCHECK_TOLERANCE = 0.13   # covers a midpoint-vs-bound convention gap


def crosschecks(board: pd.DataFrame) -> pd.DataFrame:
    """Compare the BIS figure against each bank's own publication.

    A disagreement inside the tolerance is a convention difference — the BIS
    reports the Fed at the midpoint of its range, so a source quoting the
    upper bound sits 12.5bp away and is not in conflict. Anything wider is a
    real disagreement and the panel should refuse to pick a winner.
    """
    rows = []
    for fetch in (crosscheck_fed, crosscheck_ecb, crosscheck_boe):
        res = fetch()
        ccy = res.get("ccy")
        bis = float(board.loc[ccy, "rate"]) if (not board.empty
                                                and ccy in board.index
                                                and pd.notna(board.loc[ccy, "rate"])) else None
        own = res.get("rate")
        if res.get("error"):
            verdict = "unavailable"
        elif bis is None or own is None:
            verdict = "no comparison"
        elif abs(bis - own) <= CROSSCHECK_TOLERANCE:
            verdict = "agrees"
        else:
            verdict = "DISAGREES"
        rows.append({"ccy": ccy, "bank_source": res.get("source"),
                     "own_rate": own, "bis_rate": bis,
                     "diff_bp": (round((own - bis) * 100, 1)
                                 if (own is not None and bis is not None) else None),
                     "as_of": res.get("as_of"), "note": res.get("note"),
                     "verdict": verdict, "error": res.get("error")})
    return pd.DataFrame(rows).set_index("ccy")


# --------------------------------------------------------------------------
# What the market expects the Fed to do
# --------------------------------------------------------------------------
# Fed funds futures settle to the average effective rate over their contract
# month, so 100 minus the price is the rate the market is paying to lock in.
# This is an expectation with a risk premium inside it, not a forecast, and it
# is the only central bank in this module for which a traded path is available
# without paying for it — the euro and sterling equivalents are not carried by
# any free feed.

_MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
                7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


def fed_futures_curve(fetch_history, months: int = 15,
                      period: str = "3mo") -> pd.DataFrame:
    """Implied average fed funds rate by contract month.

    `fetch_history` is injected rather than imported so this stays testable
    and so the caller's caching and retry behaviour apply.

    Only the latest price is used, but a quarter is requested rather than a
    month: this app's Yahoo client refuses any series shorter than fifty rows,
    and a month of trading days is twenty-two.

    Columns: contract, label, price, implied, months_out.
    """
    today = dt.date.today()
    rows = []

    for i in range(1, months + 1):
        m = today.month + i
        year, month = today.year + (m - 1) // 12, (m - 1) % 12 + 1
        ticker = f"ZQ{_MONTH_CODES[month]}{str(year)[-2:]}.CBT"
        try:
            hist = fetch_history(ticker, period=period)
        except Exception as exc:
            # Expected rather than exceptional: the far contracts are thin and
            # a month with no trades simply has no price. Logged as a line, not
            # a traceback, so a Yahoo outage does not bury the log in fifteen.
            log.warning("fed futures unavailable for %s: %s", ticker, exc)
            continue
        if hist is None or getattr(hist, "empty", True):
            continue

        # Name the column rather than positioning it. Yahoo's own client
        # returns "Close" and this app's returns "close" with volume last, so
        # taking the final column silently reads volume — a five-digit number
        # that the sanity check below then throws away, leaving an empty curve
        # and no clue why.
        col = next((c for c in hist.columns if str(c).lower() == "close"), None)
        if col is None:
            log.warning("no close column for %s: %s", ticker, list(hist.columns))
            continue

        series = pd.to_numeric(hist[col], errors="coerce").dropna()
        if series.empty:
            continue
        price = float(series.iloc[-1])
        if not (80.0 < price < 101.0):     # a fed funds future cannot sit here
            log.warning("implausible price for %s: %s", ticker, price)
            continue
        rows.append({"contract": ticker,
                     "label": dt.date(year, month, 1).strftime("%b %Y"),
                     "price": price, "implied": 100.0 - price, "months_out": i})

    return pd.DataFrame(rows)


def implied_path(curve: pd.DataFrame, current: Optional[float]) -> Dict[str, Any]:
    """What the curve says about the next year, in basis points.

    Reported as a distance from today's rate rather than as a probability of a
    move. Converting a curve into "72% chance of a cut" requires assuming the
    size of the move and that the meeting is the only thing that matters in
    the month, and the assumption does most of the work while the percentage
    takes all the credit.
    """
    out: Dict[str, Any] = {"available": False}
    if curve is None or curve.empty or current is None or not np.isfinite(current):
        return out

    curve = curve.sort_values("months_out")
    out["available"] = True
    out["current"] = float(current)
    out["front"] = float(curve.iloc[0]["implied"])
    out["front_label"] = curve.iloc[0]["label"]

    for horizon, name in ((6, "6m"), (12, "12m")):
        near = curve[curve["months_out"] <= horizon]
        if near.empty:
            continue
        point = near.iloc[-1]
        out[f"implied_{name}"] = float(point["implied"])
        out[f"bp_{name}"] = round((float(point["implied"]) - float(current)) * 100, 1)
        out[f"label_{name}"] = point["label"]

    far = curve.iloc[-1]
    out["terminal"] = float(far["implied"])
    out["terminal_label"] = far["label"]
    out["terminal_bp"] = round((float(far["implied"]) - float(current)) * 100, 1)

    bp12 = out.get("bp_12m", out["terminal_bp"])
    if bp12 >= 20:
        out["reading"] = "priced to tighten"
    elif bp12 <= -20:
        out["reading"] = "priced to ease"
    else:
        out["reading"] = "priced roughly on hold"
    return out


# --------------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------------
# The World Gold Council compiles the IMF's reserve statistics into the table
# every desk quotes. It is free but sits behind a login, so it ships as a file
# rather than a fetch. That makes its vintage part of the data: the loader
# reads the date out of the sheet instead of trusting the filename, and
# latest_published() checks the public page for a newer one so a stale file
# announces itself rather than passing as current.

GOLD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "refdata", "wgc_official_gold_holdings.xlsx")

GOLD_SOURCE_PAGE = "https://www.gold.org/goldhub/data/gold-reserves-by-country"

# Entities in the table that are not sovereign holders. They belong in the
# ranking — the IMF really is the third-largest holder — but not in a count of
# what countries own, and not in the euro-area aggregate twice over.
GOLD_NON_SOVEREIGN = {"IMF", "BIS", "ECB", "WAEMU", "BEAC"}

# The sheet names countries the IMF's way. Only the ones this app maps to a
# currency need translating.
GOLD_ENTITY_TO_CCY = {
    "United States": "USD", "Japan": "JPY", "United Kingdom": "GBP",
    "Switzerland": "CHF", "China, P.R.: Mainland": "CNY", "India": "INR",
    "Turkey": "TRY", "Türkiye": "TRY", "Brazil": "BRL", "Mexico": "MXN",
    "Poland, Rep. of": "PLN", "South Africa": "ZAR", "Australia": "AUD",
    "Canada": "CAD", "New Zealand": "NZD", "Sweden": "SEK", "Norway": "NOK",
}

_FOOTNOTE = re.compile(r"\d+\)\s*$")


def _clean_entity(name: str) -> str:
    """Strip the trailing footnote marker the sheet appends to some names."""
    return _FOOTNOTE.sub("", str(name).strip()).strip()


def gold_holdings(path: str = GOLD_FILE) -> Dict[str, Any]:
    """Official gold holdings by country.

    The sheet lays 100 rows out as two side-by-side blocks of fifty, so it is
    read as two column ranges and stacked. Each row carries its own reporting
    date — countries report to the IMF on their own schedule, and a table that
    showed one date for all of them would be claiming a simultaneity that does
    not exist.

    Returns {table, world_tonnes, euro_area_tonnes, vintage, as_of_range}.
    """
    out: Dict[str, Any] = {"path": path}
    if not os.path.exists(path):
        out["error"] = "file not present"
        return out

    try:
        raw = pd.read_excel(path, sheet_name=0, header=None)
    except Exception as exc:
        out["error"] = f"unreadable: {str(exc)[:90]}"
        return out

    rows, world, euro = [], None, None

    for _, r in raw.iterrows():
        for offset in (0, 5):
            if offset + 4 >= len(r):
                continue
            rank, name, tonnes, pct, as_of = (r[offset], r[offset + 1], r[offset + 2],
                                              r[offset + 3], r[offset + 4])
            if not isinstance(name, str) or not name.strip():
                continue
            entity = _clean_entity(name)

            try:
                tonnes = float(tonnes)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(tonnes):
                continue

            if entity.lower().startswith("world"):
                world = tonnes
                continue
            if entity.lower().startswith("euro area"):
                euro = {"tonnes": tonnes,
                        "pct_reserves": (float(pct) * 100.0
                                         if isinstance(pct, (int, float)) else None),
                        "as_of": (pd.Timestamp(as_of).date().isoformat()
                                  if isinstance(as_of, (pd.Timestamp, dt.datetime, dt.date))
                                  else None)}
                continue

            try:
                rank = int(float(rank))
            except (TypeError, ValueError):
                continue

            try:
                share = float(pct) * 100.0 if isinstance(pct, (int, float)) else None
                if share is not None and not np.isfinite(share):
                    share = None
            except (TypeError, ValueError):
                share = None

            stamp = None
            if isinstance(as_of, (pd.Timestamp, dt.datetime, dt.date)):
                stamp = pd.Timestamp(as_of).date().isoformat()

            rows.append({"rank": rank, "entity": entity, "tonnes": tonnes,
                         "pct_reserves": share, "as_of": stamp,
                         "sovereign": entity not in GOLD_NON_SOVEREIGN,
                         "ccy": GOLD_ENTITY_TO_CCY.get(entity)})

    if not rows:
        out["error"] = "no rows parsed"
        return out

    table = pd.DataFrame(rows).drop_duplicates(subset="rank").sort_values("rank")
    out["table"] = table
    out["world_tonnes"] = world
    out["euro_area"] = euro
    # The smallest holder on the list bounds what absence from it means: a
    # country missing from the table holds less than this, which is a fact,
    # where "holds none" would be an inference.
    out["cutoff_tonnes"] = float(table["tonnes"].min())

    stamps = table["as_of"].dropna()
    if not stamps.empty:
        out["as_of_range"] = (stamps.min(), stamps.max())

    # The vintage is printed in the header, e.g. "International Financial
    # Statistics, August 2026*".
    for value in raw.iloc[:6, 0].tolist():
        if isinstance(value, str) and "Financial Statistics" in value:
            out["vintage"] = value.replace("*", "").strip()
            break

    out["file_mtime"] = dt.date.fromtimestamp(os.path.getmtime(path)).isoformat()
    return out


def latest_published(timeout: int = 20) -> Dict[str, Any]:
    """Ask the public Goldhub page which file is current.

    The page is readable without a login even though the download is not, so
    the app can tell when the shipped file has been superseded without being
    able to fetch the replacement itself.
    """
    out: Dict[str, Any] = {}
    try:
        r = requests.get(GOLD_SOURCE_PAGE, headers=_UA, timeout=timeout)
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code}"
            return out
        links = re.findall(r"/download/file/\d+/(World_official_gold_holdings[^\"']*?\.xlsx)",
                           r.text)
        if not links:
            out["error"] = "no holdings file linked on the page"
            return out
        out["filename"] = links[0]
        month = re.search(r"as_of_([A-Za-z]{3,9})(\d{4})", links[0])
        if month:
            out["published"] = f"{month.group(1)} {month.group(2)}"
        out["url"] = GOLD_SOURCE_PAGE
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


def gold_vs_reserves(gold: Dict[str, Any], board: pd.DataFrame) -> pd.DataFrame:
    """Gold holdings joined to the policy board, for the currencies we cover.

    Gold as a share of reserves is the informative column, not tonnage. A
    large economy holding a large absolute amount may still hold very little
    of it relative to what it holds in dollars, and it is that ratio, not the
    tonnage, that says something about how the bank thinks about the dollar.

    Two cases the raw table does not answer on its own, both handled here
    rather than left to drop out of the join silently:

    - **The euro area** has no single sovereign row. Its members appear
      individually and the ECB holds its own reserves besides, so the euro is
      matched to the published area aggregate.
    - **A currency absent from the table** is not a missing value. The list
      runs to a hundred holders and bottoms out near two tonnes, so absence
      bounds the holding rather than leaving it unknown — Canada, Norway and
      New Zealand each sold down and fall below that line. The row is kept
      with `status` saying so, because a central bank holding no gold is a
      deliberate policy and worth seeing.

    Adds `status`: reported | area aggregate | below listing threshold.
    """
    if not gold or "table" not in gold or board.empty:
        return pd.DataFrame()

    table = gold["table"]
    mapped = table[table["ccy"].notna()].set_index("ccy")
    cutoff = gold.get("cutoff_tonnes")
    euro = gold.get("euro_area") or {}

    rows = []
    for ccy in board.index:
        base = {"rank": None, "tonnes": None, "pct_reserves": None,
                "gold_as_of": None, "status": None}
        if ccy in mapped.index:
            g = mapped.loc[ccy]
            base.update({"rank": g["rank"], "tonnes": g["tonnes"],
                         "pct_reserves": g["pct_reserves"], "gold_as_of": g["as_of"],
                         "status": "reported"})
        elif ccy == "EUR" and euro.get("tonnes") is not None:
            base.update({"tonnes": euro["tonnes"], "pct_reserves": euro.get("pct_reserves"),
                         "gold_as_of": euro.get("as_of"), "status": "area aggregate"})
        elif cutoff is not None:
            base["status"] = "below listing threshold"
        else:
            continue
        rows.append({"ccy": ccy, **base})

    if not rows:
        return pd.DataFrame()

    joined = board.join(pd.DataFrame(rows).set_index("ccy"), how="inner")
    return joined.sort_values("tonnes", ascending=False, na_position="last")
