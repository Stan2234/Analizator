"""Read a company's SEC filing itself and return the passages that answer a question.

The other tools carry XBRL figures, which a new issuer does not have yet: an S-1
prospectus is not tagged, so a fresh IPO's losses and cash burn exist only in the
document. This module finds the company on EDGAR (ticker, name, or full-text search
for issuers without a ticker), picks the newest filing of the asked form, turns the
HTML into plain text with table rows kept on one line, and returns the chunks that
match the search terms. A prospectus runs to 15 MB of HTML, so the parser streams it
and the text is kept in memory for six hours.
"""
from __future__ import annotations

import os
import re
import threading
import time
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import requests

UA = os.environ.get("SEC_USER_AGENT", "GlobalMarketAnalyst hello@globalmarketanalyst.com")
FORM_GROUPS = {   # what the analyst asks for -> forms that count, newest first wins
    "S-1": ("S-1", "S-1/A", "F-1", "F-1/A", "424B4", "424B1"),
    "10-K": ("10-K", "10-K/A", "20-F", "40-F"),
    "10-Q": ("10-Q", "10-Q/A"),
    "8-K": ("8-K", "6-K"),
    "DEF 14A": ("DEF 14A",),
}
LATEST_ORDER = ("10-Q", "10-K", "S-1", "20-F", "F-1", "S-1/A", "F-1/A", "424B4")
MAX_RETURN = 14000
TEXT_TTL = 6 * 3600

_tickers: Dict[str, Any] = {"at": 0.0, "map": {}}
_texts: Dict[str, Tuple[float, List[str]]] = {}
_lock = threading.Lock()


def _get(url: str, **params: Any) -> requests.Response:
    for attempt in range(3):
        r = requests.get(url, params=params or None, timeout=40,
                         headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
        if r.status_code in (429, 503) and attempt < 2:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def _ticker_map() -> Dict[str, Tuple[int, str]]:
    if time.time() - _tickers["at"] > 24 * 3600 or not _tickers["map"]:
        rows = _get("https://www.sec.gov/files/company_tickers.json").json().values()
        _tickers["map"] = {r["ticker"].upper(): (int(r["cik_str"]), r["title"]) for r in rows}
        _tickers["at"] = time.time()
    return _tickers["map"]


def resolve(company: str) -> Optional[Tuple[int, str]]:
    """(CIK, name) for a ticker or a company name, including issuers that have no ticker yet."""
    q = (company or "").strip()
    if not q:
        return None
    if q.isdigit():
        return int(q), q
    tm = _ticker_map()
    if q.upper() in tm:
        return tm[q.upper()]
    ql = q.lower()
    named = [v for v in tm.values() if v[1].lower().startswith(ql)] or [v for v in tm.values() if ql in v[1].lower()]
    if named:
        return min(named, key=lambda v: len(v[1]))
    # no ticker (a company going public): EDGAR full-text search knows every filer
    hits = _get("https://efts.sec.gov/LATEST/search-index", q=f'"{q}"').json().get("hits", {}).get("hits", [])
    for h in hits:
        for name in h.get("_source", {}).get("display_names", []):
            m = re.search(r"^(.*?)\s*(?:\(([^)]*)\))?\s*\(CIK (\d+)\)", name)
            if m and ql.split()[0] in m.group(1).lower():
                return int(m.group(3)), m.group(1).strip()
    return None


def filings(cik: int) -> List[Dict[str, Any]]:
    r = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    rec = (r.get("filings") or {}).get("recent") or {}
    out = []
    for i, form in enumerate(rec.get("form") or []):
        acc = rec["accessionNumber"][i]
        doc = rec["primaryDocument"][i]
        out.append({"form": form, "filed": rec["filingDate"][i], "period": rec.get("reportDate", [""] * (i + 1))[i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}",
                    "description": (rec.get("primaryDocDescription") or [""] * (i + 1))[i]})
    return out


def pick(rows: List[Dict[str, Any]], form: str) -> Optional[Dict[str, Any]]:
    f = (form or "latest").strip().upper()
    if f in ("", "LATEST", "ANY"):
        return next((r for r in rows if r["form"] in LATEST_ORDER), None)
    wanted = FORM_GROUPS.get(f, (f,))
    return next((r for r in rows if r["form"] in wanted), None)


class _Text(HTMLParser):
    """HTML to lines: block tags break lines, table cells join with " | ", hidden parts drop."""
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "hr"}
    VOID = {"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr"}
    SKIP = {"script", "style", "ix:header", "head", "title"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.cur: List[str] = []
        self.stack: List[Tuple[str, bool]] = []
        self.hidden = 0
        self.cells = 0      # inside a table cell, paragraphs must not break the row

    def _flush(self) -> None:
        line = re.sub(r"\s+", " ", "".join(self.cur)).strip(" |")
        line = re.sub(r"(\|\s*){2,}", "| ", line)
        line = re.sub(r"\$\s*\|\s*", "$", line)            # "$ | (1,531) | )" -> "$(1,531)"
        line = re.sub(r"\s*\|\s*([)%])", r"\1", line)
        if line:
            self.out.append(line)
        self.cur = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in ("tr", "table"):
            self.cells = 0      # a row never starts inside a cell; recovers from an unclosed <td>
        if tag in self.BLOCK:
            if self.cells and tag not in ("tr", "table"):
                self.cur.append(" ")
            else:
                self._flush()
        if tag in ("td", "th"):
            self.cur.append(" | ")
            self.cells += 1
        if tag in self.VOID:
            return
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        hide = tag in self.SKIP or "display:none" in style
        self.stack.append((tag, hide))
        self.hidden += hide

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self.cells = max(0, self.cells - 1)
        if tag in self.VOID:
            return
        if any(t == tag for t, _ in self.stack):
            while self.stack:
                t, hide = self.stack.pop()
                self.hidden -= hide
                if t == tag:
                    break
        if tag in self.BLOCK and not (self.cells and tag not in ("tr", "table")):
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.cur.append(data)


def text_lines(url: str) -> List[str]:
    with _lock:
        hit = _texts.get(url)
        if hit and time.time() - hit[0] < TEXT_TTL:
            return hit[1]
    p = _Text()
    p.feed(_get(url).text)
    p.close()
    p._flush()
    with _lock:
        for k in [k for k, v in _texts.items() if time.time() - v[0] > TEXT_TTL] + list(_texts)[:-3]:
            _texts.pop(k, None)
        _texts[url] = (time.time(), p.out)
    return p.out


def _chunks(lines: List[str], size: int = 1400) -> List[Tuple[int, str]]:
    out, cur, start = [], [], 0
    for i, ln in enumerate(lines):
        if not cur:
            start = i
        cur.append(ln)
        if sum(len(x) for x in cur) >= size:
            out.append((start, "\n".join(cur)))
            cur = []
    if cur:
        out.append((start, "\n".join(cur)))
    return out


def search(lines: List[str], query: str, limit: int = MAX_RETURN) -> List[Dict[str, Any]]:
    """The chunks that best match the query's phrases (separate them with ; or ,).

    Each phrase ranks the chunks on its own, and the phrases take turns picking, so a
    common word like "revenue" cannot crowd out "net cash used in operating activities".
    A table row that holds the phrase and then figures ranks first: that is where the
    numbers are."""
    phrases = [p.strip().lower() for p in re.split(r"[;,\n]", query or "") if p.strip()]
    if not phrases:
        return []
    chunks = _chunks(lines)
    lows = [ch.lower() for _, ch in chunks]
    ranked = []
    for p in phrases:
        row = re.compile(re.escape(p) + r"[^\n]*\|\s*\$?\(?\d")
        cand = [(5 * bool(row.search(low)) + min(low.count(p), 5) + sum(q in low for q in phrases), i)
                for i, low in enumerate(lows) if p in low]
        ranked.append([i for _, i in sorted(cand, key=lambda x: -x[0])])
    picked, used = set(), 0
    while any(ranked) and used < limit * 0.9:
        for r in ranked:
            while r and r[0] in picked:
                r.pop(0)
            if r:
                i = r.pop(0)
                if used + len(chunks[i][1]) <= limit:
                    picked.add(i)
                    used += len(chunks[i][1])
    return [{"position_pct": round(chunks[i][0] / max(1, len(lines)) * 100, 1), "text": chunks[i][1]}
            for i in sorted(picked)]


def read_filing(company: str, form: str = "latest", query: str = "") -> Dict[str, Any]:
    who = resolve(company)
    if not who:
        return {"error": f"no SEC filer found for '{company}'"}
    cik, name = who
    rows = filings(cik)
    f = pick(rows, form)
    if not f:
        return {"error": f"{name} has no {form} filing on EDGAR", "recent_forms": [r["form"] for r in rows[:15]]}
    lines = text_lines(f["url"])
    out: Dict[str, Any] = {"company": name, "cik": cik, "form": f["form"], "filed": f["filed"],
                           "period": f["period"], "url": f["url"], "document_lines": len(lines)}
    if query:
        hits = search(lines, query)
        out["matches"] = hits
        if not hits:
            out["note"] = "none of the phrases appear in the document; try other wording (e.g. 'net loss' and 'loss from operations')"
    else:
        out["opening"] = "\n".join(lines[:60])[:4000]
        out["note"] = "pass query with the phrases to look for to get the matching passages and tables"
    out["other_recent_filings"] = [{"form": r["form"], "filed": r["filed"]} for r in rows[:8]]
    return out
