"""AI Bubble Tracker's data, conclusions and charts, for the AI Analyst.

AI Bubble Tracker runs on the same server and writes its data to disk: snapshot.json
(SEC XBRL quarters, filings, market and macro series, the checks it computes),
manual.json (the private AI labs and other hand-kept figures, each with its source)
and the weekly briefs. This module reads those files directly, so the analyst sees
everything a paying AI Bubble visitor sees, and turns each tab into a compact answer
in billions with links back to the page. `chart()` draws the same charts in Plotly.

The directory is AI_BUBBLE_DATA, else /opt/ai-bubble/data (the VPS), else the
sibling checkout ../ai-bubble/data (local development).
"""
from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_URL = os.environ.get("AI_BUBBLE_URL", "").rstrip("/") or "https://globalmarketanalyst.com/bubble"

SECTIONS = {   # section -> (AI Bubble tab, what it holds)
    "overview": ("overview", "the 18 checks with their zone (calm / watch / diverging), the reason and the "
                             "thresholds, how the counts moved week by week, and the latest weekly brief's headline"),
    "brief": ("brief", "the latest weekly brief in full (what changed in the checks that week) and earlier headlines"),
    "hyperscalers": ("hyper", "Microsoft, Alphabet, Amazon and Meta together by quarter: revenue, operating cash flow, "
                              "capex, free cash flow, paper gains, debt raised, buybacks; capex guidance; cloud revenue"),
    "company": ("hyper", "one tracked company by quarter from its SEC filings (pass ticker): MSFT GOOGL AMZN META ORCL "
                         "CRWV NVDA AVGO AMD MU DELL SMCI VRT"),
    "nvidia": ("nvidia", "Nvidia: receivables and days to collect, inventory, supply and capacity commitments and when "
                         "they fall due, customer concentration, private investments, bonds"),
    "labs": ("labs", "OpenAI and Anthropic: revenue run-rate over time, funding rounds and valuations, reported losses, "
                     "cash burn and projections, compute commitments; every figure with its source and whether it is "
                     "a leak, press report, projection or company statement"),
    "circular": ("circular", "circular money: what each company invested in the AI labs versus what the labs promised "
                             "to spend with it"),
    "hidden_liabilities": ("hidden", "signed data-center leases not yet started and purchase commitments, by company "
                                     "and quarter (off the balance sheet)"),
    "depreciation": ("accounting", "the server useful lives each company uses and how much profit depends on them"),
    "valuation": ("markets", "price/sales, price/earnings, free-cash-flow yield of the AI names; market value / GDP; "
                             "Nasdaq now against the dot-com run; equal-weight vs regular S&P 500"),
    "credit": ("credit", "investment-grade, BBB, high-yield and CCC spreads now against 1 and 3 years; 10-year yield"),
    "sentiment": ("sentiment", "FINRA margin debt, insider selling by company (Form 4), the VIX"),
    "real_economy": ("physical", "TSMC monthly revenue, US data-center construction, electricity prices, GPU rental "
                                 "prices, PJM capacity prices, the share of US businesses using AI"),
    "stress_test": ("domino", "the interactive what-if on the Stress test tab (link only)"),
}

ZONE = {"ok": "calm", "warn": "watch", "bad": "diverging"}

_cache: Dict[str, Any] = {}


def data_dir() -> Optional[str]:
    for d in (os.environ.get("AI_BUBBLE_DATA"), "/opt/ai-bubble/data",
              os.path.join(HERE, "..", "ai-bubble", "data")):
        if d and os.path.exists(os.path.join(d, "snapshot.json")):
            return d
    return None


def _read(name: str) -> Any:
    d = data_dir()
    path = os.path.join(d, name) if d else ""
    if not path or not os.path.exists(path):
        return None
    key = (path, os.path.getmtime(path))
    if _cache.get(name, (None,))[0] != key:
        with open(path, encoding="utf-8") as f:
            _cache[name] = (key, json.load(f))
    return _cache[name][1]


def available() -> bool:
    return _read("snapshot.json") is not None


def link(tab: str) -> str:
    return f"{PUBLIC_URL}/#{tab}"


def bn(v: Any, nd: int = 1) -> Optional[float]:
    return None if v is None else round(v / 1e9, nd)


def _pct(a: Any, b: Any) -> Optional[float]:
    return None if (a is None or not b) else round((a / b - 1) * 100, 1)


def _snap() -> Dict[str, Any]:
    return _read("snapshot.json") or {}


def _manual() -> Dict[str, Any]:
    return _read("manual.json") or {}


def _history() -> Dict[str, Any]:
    return _snap().get("history") or _read("history.json") or {}


def _digests() -> List[Dict[str, Any]]:
    return _snap().get("digests") or _read("digests.json") or []


def _quarters(ticker: str) -> List[Dict[str, Any]]:
    return ((_snap().get("companies") or {}).get(ticker) or {}).get("quarters") or []


def _ttm(ticker: str, key: str, back: int = 0) -> Optional[float]:
    q = _quarters(ticker)
    sl = q[len(q) - 4 - back: len(q) - back] if len(q) >= 4 + back else []
    vals = [r.get(key) for r in sl]
    return sum(vals) if sl and all(v is not None for v in vals) else None


def _series_at(series: List[List[Any]], back: int) -> Optional[List[Any]]:
    return series[-1 - back] if len(series) > back else None


def _thin(rows: List[Any], keep: int) -> List[Any]:
    """At most `keep` rows, evenly spaced, always with the first and the latest."""
    if len(rows) <= keep:
        return list(rows)
    step = (len(rows) - 1) / (keep - 1)
    return [rows[round(i * step)] for i in range(keep)]


# ---------------- sections ----------------

def _overview() -> Dict[str, Any]:
    s = _snap()
    checks = (s.get("derived") or {}).get("checks") or []
    hist = (_history().get("checks") or [])
    counts = {z: sum(c["status"] == k for c in checks) for k, z in ZONE.items()}
    rows = []
    for c in checks:
        past = [(h["date"], (h.get("metrics") or {}).get(c["id"])) for h in hist]
        past = [(d, float(f"{v:.4g}")) for d, v in past if isinstance(v, (int, float))]
        rows.append({"check": c["title"], "now": c["big"], "zone": ZONE.get(c["status"], c["status"]),
                     "why": c["why"], "thresholds": c.get("thr"), "explained": c.get("info"),
                     "recorded_values": _thin(past, 5), "tab": link(c.get("tab") or "overview")})
    d = _digests()
    return {
        "counts": counts,
        "how_to_read": ("Each check is a threshold on today's accounts, not a forecast: calm = normal, watch = "
                        "stretched, diverging = money in and money out have pulled far apart. recorded_values are "
                        "the raw metric on dates since the tracker started recording (ratios as fractions, money "
                        "in USD), oldest first."),
        "checks": rows,
        "counts_history": [{"date": h["date"], **{ZONE[k]: v for k, v in (h.get("counts") or {}).items()}}
                           for h in _thin(hist, 10)],
        "latest_brief_headline": (d[-1]["text"].split("\n", 1)[0].strip("* ") if d else None),
        "latest_brief_date": d[-1]["date"] if d else None,
    }


def _brief() -> Dict[str, Any]:
    d = _digests()
    if not d:
        return {"error": "no weekly brief has been written yet"}
    return {"date": d[-1]["date"], "text": d[-1]["text"],
            "earlier": [{"date": x["date"], "headline": x["text"].split("\n", 1)[0].strip("* ")} for x in d[-5:-1]]}


HYP_KEYS = [("revenue", "revenue"), ("ocf", "operating_cash_flow"), ("capex", "capex"),
            ("fcf", "free_cash_flow"), ("net_income", "net_income"), ("op_income", "operating_income"),
            ("nonop", "non_operating_gains"), ("da", "depreciation"), ("debt_issued", "debt_raised"),
            ("buybacks", "buybacks"), ("dividends", "dividends")]
CLOUD = {"aws": "AWS", "gcloud": "Google Cloud", "ms_cloud": "Microsoft Cloud", "oci": "Oracle Cloud Infrastructure"}


def _hyperscalers() -> Dict[str, Any]:
    s = _snap()
    rows = (s.get("derived") or {}).get("hyperscalers") or []
    quarters = []
    for r in rows[-12:]:
        q = {"quarter": r["cq"], **{name: bn(r.get(k)) for k, name in HYP_KEYS}}
        if r.get("ocf"):
            q["capex_share_of_cash_flow_pct"] = round(r["capex"] / r["ocf"] * 100, 1)
        quarters.append(q)
    segs = (s.get("filings") or {}).get("segments") or {}
    cloud = {}
    for k, label in CLOUD.items():
        ser = (segs.get(k) or {}).get("series") or []
        if ser:
            last, yago = ser[-1], (ser[-5] if len(ser) >= 5 else None)
            cloud[label] = {"latest_quarter": last["cq"], "revenue_bn": bn(last["value"]),
                            "change_vs_year_ago_pct": _pct(last["value"], yago and yago["value"]),
                            "last_5_quarters_bn": [[x["cq"], bn(x["value"])] for x in ser[-5:]]}
    per = {}
    for t in ("MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRWV"):
        q = _quarters(t)
        if q:
            per[t] = {"latest_quarter": q[-1]["cq"], "capex_bn": bn(q[-1].get("capex")),
                      "operating_cash_flow_bn": bn(q[-1].get("ocf")), "free_cash_flow_bn": bn(q[-1].get("fcf")),
                      "capex_12m_bn": bn(_ttm(t, "capex")), "free_cash_flow_12m_bn": bn(_ttm(t, "fcf")),
                      "free_cash_flow_12m_year_before_bn": bn(_ttm(t, "fcf", 4))}
    return {"big4_by_quarter_bn": quarters,
            "note": "Big 4 = Microsoft, Alphabet, Amazon, Meta, summed by calendar quarter from SEC XBRL.",
            "capex_guidance": (_manual().get("capex_guidance") or {}).get("rows"),
            "cloud_revenue": cloud, "by_company": per}


CO_KEYS = ["revenue", "gross_profit", "op_income", "net_income", "nonop", "ocf", "capex", "fcf", "da", "sbc",
           "buybacks", "dividends", "debt_issued", "cash", "st_invest", "debt", "inventory", "receivables",
           "rpo", "op_lease", "fin_lease"]


def _company(ticker: str) -> Dict[str, Any]:
    t = (ticker or "").upper().strip()
    s = _snap()
    c = (s.get("companies") or {}).get(t)
    if not c:
        return {"error": f"{t or 'ticker'} is not tracked by AI Bubble Tracker",
                "tracked": sorted((s.get("companies") or {}).keys())}
    q = c.get("quarters") or []
    val = next((v for v in (s.get("derived") or {}).get("valuation") or [] if v.get("ticker") == t), None)
    lives = ((_manual().get("depreciation") or {}).get("lives") or {}).get(t)
    commit = ((s.get("filings") or {}).get("commitments") or {}).get(t) or []
    return {
        "name": c.get("name"), "group": c.get("group"),
        "quarters_bn": [{"quarter": r["cq"], "period_end": r.get("end"),
                         **{k: bn(r.get(k), 2) for k in CO_KEYS if r.get(k) is not None}} for r in q[-8:]],
        "twelve_months_bn": {k: bn(_ttm(t, k)) for k in ("revenue", "net_income", "ocf", "capex", "fcf")},
        "valuation": val, "quote": ((s.get("market") or {}).get("quotes") or {}).get(t),
        "latest_filing": ((s.get("filings") or {}).get("latest") or {}).get(t),
        "server_useful_life": lives,
        "leases_not_started_bn": [[x["cq"], bn(x.get("lnc"))] for x in commit[-5:]],
        "note": "nonop = non-operating gains (mostly mark-to-market on stakes); rpo = remaining performance "
                "obligations (contracted revenue not yet recognised).",
    }


def _nvidia() -> Dict[str, Any]:
    s = _snap()
    n = (s.get("filings") or {}).get("nvda") or {}
    q = _quarters("NVDA")
    rows = []
    for r in q[-8:]:
        dso = r["receivables"] / r["revenue"] * 91 if r.get("receivables") and r.get("revenue") else None
        rows.append({"quarter": r["cq"], "period_end": r.get("end"), "revenue_bn": bn(r.get("revenue")),
                     "receivables_bn": bn(r.get("receivables")), "days_to_collect": dso and round(dso, 1),
                     "inventory_bn": bn(r.get("inventory")), "operating_cash_flow_bn": bn(r.get("ocf")),
                     "net_income_bn": bn(r.get("net_income")), "cash_and_short_term_bn":
                         bn((r.get("cash") or 0) + (r.get("st_invest") or 0))})
    segs = (s.get("filings") or {}).get("segments") or {}
    dc = {k: [[x["cq"], bn(x["value"])] for x in (segs.get(k) or {}).get("series", [])[-5:]]
          for k in ("nvda_dc", "nvda_hyperscale", "nvda_other_dc")}
    checks = [c for c in (s.get("derived") or {}).get("checks") or [] if c["id"].startswith("nvda")]
    return {
        "quarters": rows, "data_center_revenue_bn": dc,
        "commitments": [{"what": c.get("label"), "total_bn": bn(c.get("total")),
                         "schedule_bn": dict(zip(n.get("schedule_labels") or [], [bn(v) for v in c.get("schedule") or []]))}
                        for c in n.get("commit_types") or []],
        "supply_commitments_history_bn": [[h["end"], bn(h.get("supply"))] for h in (n.get("history") or [])[-8:]],
        "private_investments_bn": [[h["end"], bn(h.get("private"))] for h in (n.get("investments") or [])[-6:]],
        "customer_concentration": (n.get("concentration") or [])[-4:],
        "bonds": n.get("bonds"),
        "checks": [{"check": c["title"], "now": c["big"], "zone": ZONE[c["status"]], "why": c["why"]} for c in checks],
        "source_filing": {"form": n.get("form"), "as_of": n.get("as_of"), "url": n.get("url")},
        "note": "customer_concentration lists the shares of revenue and receivables of each customer above 10%, "
                "unnamed in the filing.",
    }


def _labs() -> Dict[str, Any]:
    m = _manual()
    checks = [c for c in (_snap().get("derived") or {}).get("checks") or [] if c["id"] in ("oa_commit", "lab_cover")]
    return {
        "labs": m.get("labs"),
        "checks": [{"check": c["title"], "now": c["big"], "zone": ZONE[c["status"]], "why": c["why"]} for c in checks],
        "watchlist": m.get("watchlist"),
        "verified_at": m.get("verified_at"),
        "how_to_read": ("run_rate = annualised revenue in USD billions at that date; rounds = amount raised and "
                        "post-money valuation in USD billions; facts.kind: leak = from documents reported by the "
                        "press, press = reported, projection = the company's own forecast as reported, company = "
                        "stated by the company. These labs were private, so none of this is audited; say so when "
                        "you use it, and cite the source."),
    }


def _circular() -> Dict[str, Any]:
    return _manual().get("flows") or {"error": "no circular-money table"}


def _hidden() -> Dict[str, Any]:
    f = _snap().get("filings") or {}
    com = f.get("commitments") or {}
    latest, by_q = {}, {}
    for t, rows in com.items():
        if not rows:
            continue
        last = rows[-1]
        yago = next((r for r in rows if r["cq"] == f"{int(last['cq'][:4]) - 1}{last['cq'][4:]}"), None)
        latest[t] = {"quarter": last["cq"], "leases_not_started_bn": bn(last.get("lnc")),
                     "same_quarter_year_before_bn": bn(yago and yago.get("lnc")),
                     "purchase_commitments_bn": bn(last.get("purchase"))}
        for r in rows:
            if r.get("lnc") is not None:
                by_q.setdefault(r["cq"], {})[t] = r["lnc"]
    totals = [[cq, bn(sum(v.values())), len(v)] for cq, v in sorted(by_q.items())][-8:]
    return {"latest_by_company": latest, "what_purchase_commitments_are": f.get("purchase_note"),
            "total_by_quarter_bn": totals,
            "note": "Leases signed but not yet started are not on the balance sheet. total_by_quarter_bn = "
                    "[quarter, total, companies reporting]; early quarters have fewer companies."}


def _depreciation() -> Dict[str, Any]:
    d = _manual().get("depreciation") or {}
    chk = next((c for c in (_snap().get("derived") or {}).get("checks") or [] if c["id"] == "depr"), None)
    per = {t: {"depreciation_12m_bn": bn(_ttm(t, "da")), "capex_12m_bn": bn(_ttm(t, "capex"))}
           for t in ("MSFT", "GOOGL", "AMZN", "META", "ORCL")}
    return {"useful_lives": d.get("lives"), "note": d.get("_note"),
            "check": chk and {"check": chk["title"], "now": chk["big"], "zone": ZONE[chk["status"]], "why": chk["why"]},
            "depreciation_vs_capex": per}


def _valuation() -> Dict[str, Any]:
    s = _snap()
    vals = []
    for v in (s.get("derived") or {}).get("valuation") or []:
        vals.append({"ticker": v["ticker"], "name": v.get("name"), "price": v.get("price"),
                     "market_cap_bn": bn(v.get("market_cap"), 0),
                     "price_to_sales": v.get("ps") and round(v["ps"], 1),
                     "price_to_earnings": v.get("pe") and round(v["pe"], 1),
                     "free_cash_flow_yield_pct": v.get("fcf_yield") and round(v["fcf_yield"] * 100, 2),
                     "high_52w": v.get("high52"), "low_52w": v.get("low52")})
    mac = s.get("macro") or {}
    now, then = mac.get("nasdaq_now") or [], mac.get("nasdaq_dotcom") or []
    peak = max(then, key=lambda x: x[1]) if then else None
    after = [x for x in then if peak and x[0] > peak[0]]
    trough = min(after, key=lambda x: x[1]) if after else None
    checks = [c for c in (s.get("derived") or {}).get("checks") or [] if c["id"] in ("buffett", "concentration", "it_gdp")]
    eq = mac.get("equities_gdp") or []
    return {
        "ai_names": vals,
        "checks": [{"check": c["title"], "now": c["big"], "zone": ZONE[c["status"]], "why": c["why"]} for c in checks],
        "equities_to_gdp_pct": {"latest": eq[-1] if eq else None,
                                "dotcom_peak": max((x for x in eq if x[0] < "2003"), key=lambda x: x[1], default=None)},
        "nasdaq": {"latest": now[-1] if now else None,
                   "since_chatgpt_pct": _pct(now[-1][1], next((x[1] for x in now if x[0] >= "2022-11-28"), None))
                   if now else None,
                   "dotcom_peak": peak, "dotcom_trough_after_peak": trough,
                   "dotcom_fall_pct": _pct(trough[1], peak[1]) if trough else None},
    }


def _credit() -> Dict[str, Any]:
    s = _snap()
    mac = s.get("macro") or {}
    out = {}
    for k, label in (("ig_oas", "investment grade"), ("bbb_oas", "BBB"), ("hy_oas", "high yield"), ("ccc_oas", "CCC")):
        ser = mac.get(k) or []
        if ser:
            vals = [v for _, v in ser]
            out[label] = {"latest": ser[-1], "year_ago": _series_at(ser, 52),
                          "median_3y": round(statistics.median(vals), 2), "min_3y": min(vals), "max_3y": max(vals)}
    prices = (s.get("market") or {}).get("prices") or {}
    etf = {}
    for t, label in (("HYG", "high-yield bond ETF"), ("BIZD", "BDC (private credit) ETF"), ("IEF", "7-10y Treasury ETF")):
        ser = prices.get(t) or []
        if ser:
            etf[t] = {"what": label, "latest": ser[-1], "change_3m_pct": _pct(ser[-1][1], (_series_at(ser, 13) or [0, None])[1]),
                      "change_1y_pct": _pct(ser[-1][1], (_series_at(ser, 52) or [0, None])[1])}
    chk = next((c for c in (s.get("derived") or {}).get("checks") or [] if c["id"] == "hy"), None)
    return {"spreads_pp": out, "ten_year_treasury": (mac.get("ust10y") or [None])[-1], "credit_etfs": etf,
            "check": chk and {"check": chk["title"], "now": chk["big"], "zone": ZONE[chk["status"]], "why": chk["why"]},
            "note": "Spreads are option-adjusted, in percentage points over Treasuries (ICE BofA via FRED), weekly."}


def _sentiment() -> Dict[str, Any]:
    s = _snap()
    sen = s.get("sentiment") or {}
    m = sen.get("margin") or []
    ins = {}
    for t, v in (sen.get("insiders") or {}).items():
        if isinstance(v, dict):
            top = (v.get("top_sellers") or [{}])[0]
            ins[t] = {"sold_12m_usd_m": v.get("sold") and round(v["sold"] / 1e6), "bought_12m_usd_m":
                      v.get("bought") and round(v["bought"] / 1e6), "share_under_preplanned_10b5_1_plans":
                      v.get("planned_share"), "largest_seller": top.get("owner") and
                      f"{top['owner']} ({top.get('title')}), ${top.get('value', 0) / 1e6:,.0f}M"}
    chk = [c for c in (s.get("derived") or {}).get("checks") or [] if c["id"] in ("margin", "insiders")]
    return {"margin_debt_usd_bn": {"latest": m[-1] and [m[-1][0], round(m[-1][1] / 1000)] if m else None,
                                   "year_ago": [m[-13][0], round(m[-13][1] / 1000)] if len(m) > 13 else None,
                                   "dotcom_peak": max(([x[0], round(x[1] / 1000)] for x in m if x[0] < "2002"),
                                                      key=lambda x: x[1], default=None)},
            "insiders_by_company": ins,
            "vix": ((s.get("market") or {}).get("quotes") or {}).get("^VIX"),
            "checks": [{"check": c["title"], "now": c["big"], "zone": ZONE[c["status"]], "why": c["why"]} for c in chk]}


def _real_economy() -> Dict[str, Any]:
    s = _snap()
    ph = s.get("physical") or {}
    tsmc = sorted((ph.get("tsmc") or {}).items())
    tsmc_rows = [{"month": k, "revenue_ntd_bn": round(v / 1000, 1),
                  "vs_year_ago_pct": _pct(v, dict(tsmc).get(f"{int(k[:4]) - 1}{k[4:]}"))} for k, v in tsmc[-6:]]
    dc = ph.get("census_dc") or []
    dc_rows = [{"month": r["month"], "data_center_usd_bn_annual_rate": round(r["data_center"] / 1000, 1),
                "vs_year_ago_pct": _pct(r["data_center"], next((x["data_center"] for x in dc
                                                                if x["month"] == f"{int(r['month'][:4]) - 1}{r['month'][4:]}"), None))}
               for r in dc[-6:]]
    el = (s.get("power") or {}).get("electricity") or []
    btos = (s.get("demand") or {}).get("btos") or []
    return {"tsmc_monthly_revenue": tsmc_rows,
            "us_data_center_construction": dc_rows,
            "us_electricity_price_usd_kwh": {"latest": el[-1] if el else None,
                                             "year_ago": _series_at(el, 12)},
            "gpu_rental_usd_per_hour": (_history().get("gpu") or [])[-8:] or (s.get("demand") or {}).get("gpu_today"),
            "us_businesses_using_ai_pct": btos[-3:],
            "pjm_capacity_price": _manual().get("pjm"),
            "note": "TSMC in New Taiwan dollars; Census construction spending at a seasonally adjusted annual rate; "
                    "GPU prices are Vast.ai marketplace medians; PJM is the largest US grid's capacity auction ($/MW-day)."}


def section(name: str, ticker: Optional[str] = None) -> Dict[str, Any]:
    """One AI Bubble Tracker tab as data for the analyst, with its date and a link."""
    if not available():
        return {"error": "AI Bubble Tracker's data is not reachable from this server right now."}
    name = (name or "overview").strip().lower()
    if name not in SECTIONS:
        return {"error": f"unknown section {name}", "sections": {k: v[1] for k, v in SECTIONS.items()}}
    fn = {"overview": _overview, "brief": _brief, "hyperscalers": _hyperscalers, "nvidia": _nvidia,
          "labs": _labs, "circular": _circular, "hidden_liabilities": _hidden, "depreciation": _depreciation,
          "valuation": _valuation, "credit": _credit, "sentiment": _sentiment, "real_economy": _real_economy,
          "stress_test": lambda: {"what": "An interactive what-if: cut hyperscaler capex, defer orders and add bad "
                                          "debt, and watch Nvidia's cash cover its commitments. Point the reader to it."}}
    try:
        body = _company(ticker or "") if name == "company" else fn[name]()
    except Exception as e:   # a field missing from an older snapshot should not sink the answer
        body = {"error": f"could not read the {name} section: {type(e).__name__}: {e}"}
    s = _snap()
    return {"source": "AI Bubble Tracker (Global Market Analyst), built from SEC filings, FRED, Yahoo, TSMC, "
                      "Census, FINRA and sourced press figures",
            "data_as_of": s.get("generated_at"), "stale_sections": s.get("stale") or [],
            "page": link(SECTIONS[name][0]), "charts_you_can_show": CHART_IDS_BY_SECTION.get(name, []),
            **body}


# ---------------- charts ----------------

CHARTS = {
    "big4_cash_flow": ("Big 4: operating cash flow vs capex", "hyper"),
    "cloud_revenue": ("Cloud revenue by provider", "hyper"),
    "checks_over_time": ("AI Bubble checks by zone, week by week", "overview"),
    "lab_revenue": ("OpenAI and Anthropic: annualised revenue", "labs"),
    "leases_not_started": ("Signed data-center leases not yet started", "hidden"),
    "nvidia_receivables": ("Nvidia: receivables and days to collect", "nvidia"),
    "valuation": ("AI names: price to earnings and price to sales", "markets"),
    "nasdaq_vs_dotcom": ("Nasdaq since ChatGPT vs 1995-2003", "markets"),
    "credit_spreads": ("Credit spreads: high yield and investment grade", "credit"),
    "margin_debt": ("FINRA margin debt", "sentiment"),
    "tsmc_revenue": ("TSMC monthly revenue", "physical"),
    "gpu_rental_prices": ("GPU rental prices", "physical"),
}
CHART_IDS_BY_SECTION = {
    "overview": ["checks_over_time"], "hyperscalers": ["big4_cash_flow", "cloud_revenue"],
    "company": ["big4_cash_flow"], "labs": ["lab_revenue"], "hidden_liabilities": ["leases_not_started"],
    "nvidia": ["nvidia_receivables"], "valuation": ["valuation", "nasdaq_vs_dotcom"],
    "credit": ["credit_spreads"], "sentiment": ["margin_debt"],
    "real_economy": ["tsmc_revenue", "gpu_rental_prices"],
}


def chart_summary(fig) -> List[Dict[str, Any]]:
    """What each line or bar series in a chart shows: first, latest, low and high, so the text
    under the chart can quote the chart's own numbers."""
    out = []
    for tr in fig.data:
        pts = [(x, y) for x, y in zip(tr.x or [], tr.y or []) if y is not None]
        if not pts:
            continue
        lo, hi = min(pts, key=lambda p: p[1]), max(pts, key=lambda p: p[1])
        out.append({"series": tr.name, "first": pts[0], "latest": pts[-1], "low": lo, "high": hi,
                    "points": len(pts)})
    return out


def chart(chart_id: str):
    """The chart as a Plotly figure, or None if it cannot be drawn from the data on disk."""
    import plotly.graph_objects as go
    if chart_id not in CHARTS or not available():
        return None
    s, m = _snap(), _manual()
    title, _tab = CHARTS[chart_id]
    fig = go.Figure()
    if chart_id == "big4_cash_flow":
        rows = ((s.get("derived") or {}).get("hyperscalers") or [])[-12:]
        x = [r["cq"] for r in rows]
        fig.add_bar(x=x, y=[bn(r.get("ocf")) for r in rows], name="Operating cash flow")
        fig.add_bar(x=x, y=[bn(r.get("capex")) for r in rows], name="Capex")
        fig.add_scatter(x=x, y=[bn(r.get("fcf")) for r in rows], name="Free cash flow", mode="lines+markers")
        fig.update_layout(barmode="group", yaxis_title="USD billions")
    elif chart_id == "cloud_revenue":
        segs = (s.get("filings") or {}).get("segments") or {}
        for k, label in CLOUD.items():
            ser = ((segs.get(k) or {}).get("series") or [])[-12:]
            fig.add_scatter(x=[r["cq"] for r in ser], y=[bn(r["value"]) for r in ser], name=label, mode="lines+markers")
        fig.update_layout(yaxis_title="USD billions a quarter")
    elif chart_id == "checks_over_time":
        hist = _history().get("checks") or []
        for k, color in (("bad", "#d64545"), ("warn", "#e0a030"), ("ok", "#3a9a5b")):
            fig.add_bar(x=[h["date"] for h in hist], y=[(h.get("counts") or {}).get(k, 0) for h in hist],
                        name=ZONE[k], marker_color=color)
        fig.update_layout(barmode="stack", yaxis_title="checks")
    elif chart_id == "lab_revenue":
        for key in ("openai", "anthropic"):
            lab = (m.get("labs") or {}).get(key) or {}
            rr = lab.get("run_rate") or []
            fig.add_scatter(x=[r["date"] for r in rr], y=[r["value"] for r in rr], name=lab.get("name", key),
                            mode="lines+markers")
        fig.update_layout(yaxis_title="USD billions a year (run-rate)")
    elif chart_id == "leases_not_started":
        com = (s.get("filings") or {}).get("commitments") or {}
        for t, rows in com.items():
            rows = [r for r in rows if r.get("lnc") is not None][-8:]
            if rows:
                fig.add_bar(x=[r["cq"] for r in rows], y=[bn(r["lnc"]) for r in rows], name=t)
        fig.update_layout(barmode="stack", yaxis_title="USD billions")
    elif chart_id == "nvidia_receivables":
        q = _quarters("NVDA")[-10:]
        fig.add_bar(x=[r["cq"] for r in q], y=[bn(r.get("receivables")) for r in q], name="Receivables, $bn")
        fig.add_scatter(x=[r["cq"] for r in q], y=[round(r["receivables"] / r["revenue"] * 91, 1)
                                                    if r.get("receivables") and r.get("revenue") else None for r in q],
                        name="Days to collect", yaxis="y2", mode="lines+markers")
        fig.update_layout(yaxis_title="USD billions", yaxis2=dict(title="days", overlaying="y", side="right"))
    elif chart_id == "valuation":
        v = [r for r in (s.get("derived") or {}).get("valuation") or [] if r.get("pe") or r.get("ps")]
        fig.add_bar(x=[r["ticker"] for r in v], y=[r.get("pe") and round(r["pe"], 1) for r in v], name="Price / earnings")
        fig.add_bar(x=[r["ticker"] for r in v], y=[r.get("ps") and round(r["ps"], 1) for r in v], name="Price / sales")
        fig.update_layout(barmode="group", yaxis_title="multiple")
    elif chart_id == "nasdaq_vs_dotcom":
        mac = s.get("macro") or {}

        def indexed(ser, start):
            ser = [x for x in ser if x[0] >= start]
            return ([i for i in range(len(ser))], [round(v / ser[0][1] * 100, 1) for _, v in ser]) if ser else ([], [])
        x1, y1 = indexed(mac.get("nasdaq_now") or [], "2022-11-28")
        x2, y2 = indexed(mac.get("nasdaq_dotcom") or [], "1995-08-07")
        fig.add_scatter(x=x1, y=y1, name="Nasdaq since ChatGPT (Nov 2022)")
        fig.add_scatter(x=x2, y=y2, name="Nasdaq from Netscape IPO (Aug 1995)", line=dict(dash="dash"))
        fig.update_layout(xaxis_title="weeks", yaxis_title="index, 100 at week 0")
    elif chart_id == "credit_spreads":
        mac = s.get("macro") or {}
        for k, label in (("hy_oas", "High yield"), ("bbb_oas", "BBB"), ("ig_oas", "Investment grade")):
            ser = mac.get(k) or []
            fig.add_scatter(x=[d for d, _ in ser], y=[v for _, v in ser], name=label)
        fig.update_layout(yaxis_title="percentage points over Treasuries")
    elif chart_id == "margin_debt":
        ser = (s.get("sentiment") or {}).get("margin") or []
        fig.add_scatter(x=[d for d, _ in ser], y=[round(v / 1000) for _, v in ser], name="Margin debt")
        fig.update_layout(yaxis_title="USD billions")
    elif chart_id == "tsmc_revenue":
        ser = sorted(((s.get("physical") or {}).get("tsmc") or {}).items())[-36:]
        fig.add_bar(x=[k for k, _ in ser], y=[round(v / 1000, 1) for _, v in ser], name="TSMC revenue")
        fig.update_layout(yaxis_title="NT$ billions a month")
    elif chart_id == "gpu_rental_prices":
        hist = _history().get("gpu") or []
        for g in ("B200", "H200", "H100", "A100"):
            fig.add_scatter(x=[h["date"] for h in hist], y=[h.get(g) for h in hist], name=g, mode="lines+markers")
        fig.update_layout(yaxis_title="USD per GPU-hour (Vast.ai median)")
    fig.update_layout(title=title, height=360, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig
