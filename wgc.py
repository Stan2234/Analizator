"""World Gold Council data: gold ETF flows and central-bank purchases.

Goldhub's downloads sit behind a (free) login, so the app cannot fetch them.
The files ship in refdata/ and the owner can upload newer ones from the Metals
tab; uploads land in a directory next to the database, which survives deploys,
and win over the shipped copy. Each file's own as-of date is read from inside
it, so a stale file shows its age rather than passing as current.

  wgc_etf_flows.xlsx              "ETF Flows" workbook (gold ETFs, by fund, monthly since 2003)
  wgc_cb_changes.xlsx             "Changes_latest ... IFS" (monthly central-bank changes, tonnes)
  wgc_official_gold_holdings.xlsx "World official gold holdings ... IFS" (read by central_banks.py)

Everything here is descriptive: what was bought and sold, by whom, against the
price. None of it is turned into a signal.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Any, Dict, List, Optional

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED_DIR = os.path.join(HERE, "refdata")
FILES = {
    "etf": "wgc_etf_flows.xlsx",
    "cb_changes": "wgc_cb_changes.xlsx",
    "holdings": "wgc_official_gold_holdings.xlsx",
}
REGIONS = ["North America", "Europe", "Asia", "Other"]
SOURCE = "World Gold Council (Goldhub)"


def upload_dir() -> str:
    """Where uploaded files go: next to the database, so a redeploy keeps them."""
    db = os.environ.get("ANALIZATOR_DB_PATH", "")
    base = os.path.dirname(db) if db else os.path.join(HERE, "data")
    return os.path.join(base, "refdata")


def path_for(kind: str) -> str:
    """The newest copy of a file: an upload if there is one, else the shipped file."""
    name = FILES[kind]
    up = os.path.join(upload_dir(), name)
    shipped = os.path.join(SHIPPED_DIR, name)
    if os.path.exists(up) and (not os.path.exists(shipped)
                               or os.path.getmtime(up) >= os.path.getmtime(shipped)):
        return up
    return shipped


def save_upload(kind: str, data: bytes) -> str:
    os.makedirs(upload_dir(), exist_ok=True)
    path = os.path.join(upload_dir(), FILES[kind])
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def detect_kind(data: bytes) -> Optional[str]:
    """Which WGC file an upload is, from its sheets rather than its name."""
    import io
    try:
        sheets = pd.ExcelFile(io.BytesIO(data)).sheet_names
    except Exception:
        return None
    if "Fund flows by month" in sheets and "Holdings by month" in sheets:
        return "etf"
    if "Monthly" in sheets and "Annual" in sheets:
        return "cb_changes"
    try:
        head = pd.read_excel(io.BytesIO(data), sheet_name=0, header=None, nrows=6)
        if head.astype(str).apply(lambda c: c.str.contains("WORLD OFFICIAL GOLD HOLDINGS",
                                                            case=False)).any().any():
            return "holdings"
    except Exception:
        pass
    return None


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------- ETF flows

def _by_month_sheet(path: str, sheet: str) -> Dict[str, Any]:
    """One of the per-fund monthly sheets, as {price, total, by_fund, meta}.

    Rows 0-5 are a header block (ticker, active, fund type, region, country,
    name); data starts on row 6 with date, gold price, three total columns,
    then one column per fund.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    meta = raw.iloc[:6, 5:].T
    meta.columns = ["ticker", "active", "fund_type", "region", "country", "name"]
    body = raw.iloc[6:].copy()
    dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
    keep = dates.notna()
    body, dates = body[keep], dates[keep]
    price = pd.Series(pd.to_numeric(body.iloc[:, 1], errors="coerce").values, index=dates)
    total_t = pd.Series(pd.to_numeric(body.iloc[:, 3], errors="coerce").values, index=dates)
    funds = body.iloc[:, 5:].apply(pd.to_numeric, errors="coerce")
    funds.index = dates
    funds.columns = range(funds.shape[1])
    return {"price": price, "total_tonnes": total_t, "funds": funds.fillna(0.0),
            "meta": meta.reset_index(drop=True)}


def _regional(sheet: Dict[str, Any]) -> pd.DataFrame:
    regions = sheet["meta"]["region"].fillna("Other").astype(str)
    df = sheet["funds"].T.groupby(regions.values).sum().T
    for r in REGIONS:
        if r not in df.columns:
            df[r] = 0.0
    return df[REGIONS]


def etf_flows(path: Optional[str] = None) -> Dict[str, Any]:
    """Gold ETF holdings, demand (tonnes) and flows (US$) by region, monthly.

    Returns {holdings, demand_t, flows_usd (DataFrames by region), price,
    funds (latest table), as_of, periods, path} or {error}.
    """
    path = path or path_for("etf")
    out: Dict[str, Any] = {"path": path, "source": SOURCE}
    if not os.path.exists(path):
        out["error"] = "no ETF flows file"
        return out
    try:
        hold = _by_month_sheet(path, "Holdings by month")
        dem = _by_month_sheet(path, "Demand by month")
        flo = _by_month_sheet(path, "Fund flows by month")
        out["holdings"] = _regional(hold)
        out["demand_t"] = _regional(dem)
        out["flows_usd"] = _regional(flo)                     # US$ mn
        out["price"] = hold["price"]

        legend = pd.read_excel(path, sheet_name="Periods Legend", header=None)
        out["periods"] = {str(r[0]): str(r[1]) for r in legend.itertuples(index=False)
                          if isinstance(r[0], str) and r[0].isupper()}
        fund = pd.read_excel(path, sheet_name="All flows by fund", header=None)
        fund = fund.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
        asof = str(fund.iloc[0, 0]).replace("As Of Date", "").strip()
        out["as_of"] = (pd.to_datetime(asof, dayfirst=True).date().isoformat()
                        if asof else out["holdings"].index.max().date().isoformat())
        hdr = [str(c).replace("\n", " ").strip() for c in fund.iloc[1]]
        tbl = fund.iloc[2:].copy()
        tbl.columns = hdr
        tbl = tbl[tbl["Name"].notna()]
        tbl["Region"] = tbl["Region"].ffill()
        tbl = tbl[~tbl["Region"].astype(str).str.contains("Total", case=False)]
        for c in tbl.columns[4:]:
            tbl[c] = pd.to_numeric(tbl[c], errors="coerce")
        out["funds"] = tbl.reset_index(drop=True)
    except Exception as exc:
        out["error"] = f"could not read the ETF file: {str(exc)[:160]}"
    return out


def etf_summary(e: Dict[str, Any]) -> Dict[str, Any]:
    """The headline numbers: latest month, last 3 and 12 months, year to date."""
    if e.get("error"):
        return {}
    dem, flo, hold, px = e["demand_t"], e["flows_usd"], e["holdings"], e["price"]
    last = dem.index.max()
    ytd = dem.index[dem.index.year == last.year]
    total_hold = hold.sum(axis=1)
    peak_at = total_hold.idxmax()

    def window(df: pd.DataFrame, n: int) -> Dict[str, float]:
        return {k: round(float(v), 1) for k, v in df.tail(n).sum().items()}

    return {
        "as_of": e.get("as_of"),
        "month": last.strftime("%b %Y"),
        "holdings_tonnes": round(float(total_hold.iloc[-1]), 0),
        "holdings_by_region": {k: round(float(v), 0) for k, v in hold.iloc[-1].items()},
        "holdings_peak": {"tonnes": round(float(total_hold.max()), 0),
                          "month": peak_at.strftime("%b %Y")},
        "aum_usd_bn": (round(float(e["funds"]["AUM (US$mn)"].sum()) / 1000, 1)
                       if "AUM (US$mn)" in e.get("funds", pd.DataFrame()).columns else None),
        "gold_price_avg_month": round(float(px.iloc[-1]), 1),
        "demand_t": {"month": window(dem, 1), "3m": window(dem, 3), "12m": window(dem, 12),
                     "ytd": {k: round(float(v), 1) for k, v in dem.loc[ytd].sum().items()}},
        "flows_usd_mn": {"month": window(flo, 1), "3m": window(flo, 3), "12m": window(flo, 12),
                         "ytd": {k: round(float(v), 0) for k, v in flo.loc[ytd].sum().items()}},
        "months_of_inflow_last_12": int((dem.tail(12).sum(axis=1) > 0).sum()),
    }


def demand_vs_price(e: Dict[str, Any], months: int = 120) -> Dict[str, Any]:
    """How global ETF demand has lined up with the same month's price move.

    A contemporaneous relationship, measured, not a forecast: ETF buying
    and a rising price happen together, and which drives which is not
    something a monthly correlation can settle.
    """
    if e.get("error"):
        return {}
    dem = e["demand_t"].sum(axis=1)
    ret = e["price"].pct_change() * 100
    df = pd.concat({"demand": dem, "ret": ret}, axis=1).dropna().tail(months)
    if len(df) < 24:
        return {}
    up, down = df[df.ret > 0], df[df.ret <= 0]
    return {
        "months": len(df),
        "correlation": round(float(df.demand.corr(df.ret)), 2),
        "avg_demand_when_price_up_t": round(float(up.demand.mean()), 1),
        "avg_demand_when_price_down_t": round(float(down.demand.mean()), 1),
        "months_price_up_while_outflows": int(((df.ret > 0) & (df.demand < 0)).sum()),
    }


# ------------------------------------------------------- central-bank changes

def cb_changes(path: Optional[str] = None) -> Dict[str, Any]:
    """Monthly change in official gold holdings, by country, in tonnes.

    Returns {monthly (DataFrame: month x country), comments, as_of, path} or {error}.
    Countries report to the IMF on their own schedule, so the latest months
    are incomplete: as_of is the last month with any reported change.
    """
    path = path or path_for("cb_changes")
    out: Dict[str, Any] = {"path": path, "source": SOURCE}
    if not os.path.exists(path):
        out["error"] = "no central-bank changes file"
        return out
    try:
        raw = pd.read_excel(path, sheet_name="Monthly", header=None)
        hdr_row = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Country Lookup Column"][0]
        hdr = raw.iloc[hdr_row]
        date_cols = [i for i, v in enumerate(hdr) if isinstance(v, (dt.datetime, pd.Timestamp))]
        body = raw.iloc[hdr_row + 1:]
        body = body[body.iloc[:, 1].notna() & (body.iloc[:, 1].astype(str).str.strip() != "")]
        names = body.iloc[:, 1].astype(str).str.strip()
        vals = body.iloc[:, date_cols].apply(pd.to_numeric, errors="coerce")
        vals.index = names.values
        vals.columns = pd.to_datetime([hdr[i] for i in date_cols]).to_period("M").to_timestamp("M")
        monthly = vals.T.fillna(0.0)
        monthly = monthly.loc[:, ~monthly.columns.duplicated()]
        comments = {n: str(c).strip() for n, c in zip(names, body.iloc[:, 2])
                    if isinstance(c, str) and c.strip()}
        active = monthly.abs().sum(axis=1) > 0.01
        monthly = monthly.loc[: active[active].index.max()] if active.any() else monthly
        out["monthly"] = monthly
        out["comments"] = comments
        out["as_of"] = monthly.index.max().strftime("%b %Y")
    except Exception as exc:
        out["error"] = f"could not read the central-bank file: {str(exc)[:160]}"
    return out


_NAMES = {
    "China, P.R.: Mainland": "China", "Russian Federation": "Russia", "Czech Rep.": "Czechia",
    "Kyrgyz Rep.": "Kyrgyzstan", "Korea, Rep. of": "South Korea", "Turkey*": "Turkey (other holdings)",
    "State Oil Fund of the Republic of Azerbaijan (SOFAZ)": "Azerbaijan (SOFAZ oil fund)",
    "Iran, Islamic Rep. of": "Iran", "Egypt, Arab Rep. of": "Egypt", "Venezuela, Rep. Bolivariana de": "Venezuela",
    "Taiwan Province of China": "Taiwan", "Hong Kong SAR, China": "Hong Kong",
}


def display_name(name: str) -> str:
    """IMF naming ("Poland, Rep. of") to the name a reader expects."""
    if name in _NAMES:
        return _NAMES[name]
    for suffix in (", Rep. of", ", Republic of", ", Kingdom of", ", The", ", Islamic Rep. of"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# The rows that are not countries, kept out of "who bought" so the IMF's own
# moves or a regional aggregate are not counted as a country's purchase.
_NOT_COUNTRIES = ("Euro Area", "World", "IMF", "BIS", "ECB", "Total",
                  "Turkey*")     # Turkey on a gross basis (incl. banks' gold): Turkey counted twice


def cb_summary(c: Dict[str, Any], months: int = 12, top: int = 10) -> Dict[str, Any]:
    if c.get("error"):
        return {}
    m = c["monthly"]
    m = m[[col for col in m.columns if not any(col.startswith(x) for x in _NOT_COUNTRIES)]]
    recent = m.tail(months).sum()
    net_monthly = m.sum(axis=1)
    last = m.index.max()
    ytd = m[m.index.year == last.year].sum()
    buyers = recent[recent > 0.5].sort_values(ascending=False)
    sellers = recent[recent < -0.5].sort_values()
    return {
        "as_of": c.get("as_of"),
        "net_tonnes": {"last_month": round(float(net_monthly.iloc[-1]), 1),
                       f"last_{months}m": round(float(net_monthly.tail(months).sum()), 1),
                       "ytd": round(float(ytd.sum()), 1),
                       "prior_12m": round(float(net_monthly.iloc[-2 * months:-months].sum()), 1)},
        "top_buyers": [{"country": display_name(k), "tonnes": round(float(v), 1)}
                       for k, v in buyers.head(top).items()],
        "top_sellers": [{"country": display_name(k), "tonnes": round(float(v), 1)}
                        for k, v in sellers.head(top).items()],
        "n_buyers": int((recent > 0.5).sum()),
        "n_sellers": int((recent < -0.5).sum()),
        "months": months,
    }


def cb_net_series(c: Dict[str, Any]) -> pd.Series:
    m = c["monthly"]
    m = m[[col for col in m.columns if not any(col.startswith(x) for x in _NOT_COUNTRIES)]]
    return m.sum(axis=1)


def file_status() -> List[Dict[str, Any]]:
    """For the owner: which copy of each file is in use, and how old it is."""
    rows = []
    for kind, name in FILES.items():
        p = path_for(kind)
        rows.append({"file": name, "in use": "uploaded" if p.startswith(upload_dir()) else "shipped",
                     "saved": (dt.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d")
                               if os.path.exists(p) else "missing")})
    return rows
