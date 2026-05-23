"""
Symbol universe for Analizator.

Hardcoded snapshot of S&P 500 constituents + major global indices,
commodities, and FX indices. Last manually refreshed: 2026-05.

The S&P 500 changes ~quarterly (additions/deletions). To refresh this
file later, the simplest options are:
  1. Pull the current list from Wikipedia
     (https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
  2. Use yfinance.Ticker("^GSPC").components if available
  3. Use a paid data API (Polygon, Finnhub) for an authoritative list

All symbols are Yahoo-Finance tickers.

Public exports:
  GICS_SECTORS         — canonical 11-sector list
  MAJOR_INDICES        — 11 index/commodity/FX tickers for the ticker tape
  SP500_COMPANIES      — list of dicts: {symbol, name, sector, industry}
  CORE_WATCHLIST       — ~40 high-priority tickers auto-refreshed often

Public helpers:
  all_symbols()        — every symbol we know about (indices + equities)
  all_equity_symbols() — just SP500 tickers
  by_sector(sector)    — SP500 entries in a given GICS sector
  get_meta(symbol)     — full record for a symbol (or None)
  sector_counts()      — {sector: n} histogram
  search(query)        — case-insensitive name/symbol fuzzy search
"""
from __future__ import annotations

from typing import Dict, List, Optional


# ---------------- GICS sector taxonomy ----------------

GICS_SECTORS: List[str] = [
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]


# ---------------- major indices / FX / commodities ----------------

MAJOR_INDICES: List[Dict[str, str]] = [
    {"symbol": "^GSPC",    "name": "S&P 500",              "type": "index"},
    {"symbol": "^NDX",     "name": "NASDAQ 100",           "type": "index"},
    {"symbol": "^DJI",     "name": "Dow Jones Industrial", "type": "index"},
    {"symbol": "^RUT",     "name": "Russell 2000",         "type": "index"},
    {"symbol": "^VIX",     "name": "VIX",                  "type": "vol_index"},
    {"symbol": "^TNX",     "name": "US 10Y Yield",         "type": "rate_index"},
    {"symbol": "DX-Y.NYB", "name": "US Dollar Index (DXY)","type": "fx_index"},
    {"symbol": "CL=F",     "name": "WTI Crude Oil",        "type": "commodity"},
    {"symbol": "^FTSE",    "name": "FTSE 100",             "type": "index"},
    {"symbol": "^GDAXI",   "name": "DAX",                  "type": "index"},
    {"symbol": "^N225",    "name": "Nikkei 225",           "type": "index"},
]


# ---------------- S&P 500 constituents (snapshot 2026-05) ----------------
# Format: {symbol, name, sector, industry}
# Sectors follow GICS taxonomy (matches GICS_SECTORS).

SP500_COMPANIES: List[Dict[str, str]] = [
    # ---------------- Information Technology ----------------
    {"symbol": "AAPL",  "name": "Apple",                   "sector": "Information Technology", "industry": "Technology Hardware"},
    {"symbol": "MSFT",  "name": "Microsoft",               "sector": "Information Technology", "industry": "Software"},
    {"symbol": "NVDA",  "name": "NVIDIA",                  "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "AVGO",  "name": "Broadcom",                "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "ORCL",  "name": "Oracle",                  "sector": "Information Technology", "industry": "Software"},
    {"symbol": "CRM",   "name": "Salesforce",              "sector": "Information Technology", "industry": "Software"},
    {"symbol": "ADBE",  "name": "Adobe",                   "sector": "Information Technology", "industry": "Software"},
    {"symbol": "AMD",   "name": "Advanced Micro Devices",  "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "ACN",   "name": "Accenture",               "sector": "Information Technology", "industry": "IT Services"},
    {"symbol": "CSCO",  "name": "Cisco Systems",           "sector": "Information Technology", "industry": "Communications Equipment"},
    {"symbol": "INTC",  "name": "Intel",                   "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "IBM",   "name": "IBM",                     "sector": "Information Technology", "industry": "IT Services"},
    {"symbol": "QCOM",  "name": "Qualcomm",                "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "TXN",   "name": "Texas Instruments",       "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "INTU",  "name": "Intuit",                  "sector": "Information Technology", "industry": "Software"},
    {"symbol": "NOW",   "name": "ServiceNow",              "sector": "Information Technology", "industry": "Software"},
    {"symbol": "AMAT",  "name": "Applied Materials",       "sector": "Information Technology", "industry": "Semi Equipment"},
    {"symbol": "MU",    "name": "Micron Technology",       "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "LRCX",  "name": "Lam Research",            "sector": "Information Technology", "industry": "Semi Equipment"},
    {"symbol": "KLAC",  "name": "KLA Corporation",         "sector": "Information Technology", "industry": "Semi Equipment"},
    {"symbol": "PANW",  "name": "Palo Alto Networks",      "sector": "Information Technology", "industry": "Software"},
    {"symbol": "ADI",   "name": "Analog Devices",          "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "MRVL",  "name": "Marvell Technology",      "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "SNPS",  "name": "Synopsys",                "sector": "Information Technology", "industry": "Software"},
    {"symbol": "CDNS",  "name": "Cadence Design Systems",  "sector": "Information Technology", "industry": "Software"},
    {"symbol": "FTNT",  "name": "Fortinet",                "sector": "Information Technology", "industry": "Software"},
    {"symbol": "CRWD",  "name": "CrowdStrike",             "sector": "Information Technology", "industry": "Software"},
    {"symbol": "ANET",  "name": "Arista Networks",         "sector": "Information Technology", "industry": "Communications Equipment"},
    {"symbol": "ROP",   "name": "Roper Technologies",      "sector": "Information Technology", "industry": "Software"},
    {"symbol": "ADSK",  "name": "Autodesk",                "sector": "Information Technology", "industry": "Software"},
    {"symbol": "WDAY",  "name": "Workday",                 "sector": "Information Technology", "industry": "Software"},
    {"symbol": "MCHP",  "name": "Microchip Technology",    "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "NXPI",  "name": "NXP Semiconductors",      "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "ON",    "name": "ON Semiconductor",        "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "FICO",  "name": "Fair Isaac",              "sector": "Information Technology", "industry": "Software"},
    {"symbol": "MSI",   "name": "Motorola Solutions",      "sector": "Information Technology", "industry": "Communications Equipment"},
    {"symbol": "GLW",   "name": "Corning",                 "sector": "Information Technology", "industry": "Electronic Components"},
    {"symbol": "DELL",  "name": "Dell Technologies",       "sector": "Information Technology", "industry": "Technology Hardware"},
    {"symbol": "HPQ",   "name": "HP Inc",                  "sector": "Information Technology", "industry": "Technology Hardware"},
    {"symbol": "HPE",   "name": "Hewlett Packard Enterprise","sector": "Information Technology","industry": "Technology Hardware"},
    {"symbol": "TER",   "name": "Teradyne",                "sector": "Information Technology", "industry": "Semi Equipment"},
    {"symbol": "TYL",   "name": "Tyler Technologies",      "sector": "Information Technology", "industry": "Software"},
    {"symbol": "ANSS",  "name": "ANSYS",                   "sector": "Information Technology", "industry": "Software"},
    {"symbol": "KEYS",  "name": "Keysight Technologies",   "sector": "Information Technology", "industry": "Electronic Equipment"},
    {"symbol": "STX",   "name": "Seagate Technology",      "sector": "Information Technology", "industry": "Technology Hardware"},
    {"symbol": "WDC",   "name": "Western Digital",         "sector": "Information Technology", "industry": "Technology Hardware"},
    {"symbol": "NTAP",  "name": "NetApp",                  "sector": "Information Technology", "industry": "Technology Hardware"},
    {"symbol": "FFIV",  "name": "F5",                      "sector": "Information Technology", "industry": "Communications Equipment"},
    {"symbol": "JBL",   "name": "Jabil",                   "sector": "Information Technology", "industry": "Electronic Equipment"},
    {"symbol": "PTC",   "name": "PTC",                     "sector": "Information Technology", "industry": "Software"},
    {"symbol": "ZBRA",  "name": "Zebra Technologies",      "sector": "Information Technology", "industry": "Electronic Equipment"},
    {"symbol": "AKAM",  "name": "Akamai Technologies",     "sector": "Information Technology", "industry": "IT Services"},
    {"symbol": "GEN",   "name": "Gen Digital",             "sector": "Information Technology", "industry": "Software"},
    {"symbol": "ENPH",  "name": "Enphase Energy",          "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "GDDY",  "name": "GoDaddy",                 "sector": "Information Technology", "industry": "IT Services"},
    {"symbol": "EPAM",  "name": "EPAM Systems",            "sector": "Information Technology", "industry": "IT Services"},
    {"symbol": "CDW",   "name": "CDW Corporation",         "sector": "Information Technology", "industry": "IT Services"},
    {"symbol": "VRSN",  "name": "VeriSign",                "sector": "Information Technology", "industry": "IT Services"},
    {"symbol": "FSLR",  "name": "First Solar",             "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "TRMB",  "name": "Trimble",                 "sector": "Information Technology", "industry": "Electronic Equipment"},
    {"symbol": "JNPR",  "name": "Juniper Networks",        "sector": "Information Technology", "industry": "Communications Equipment"},
    {"symbol": "SWKS",  "name": "Skyworks Solutions",      "sector": "Information Technology", "industry": "Semiconductors"},
    {"symbol": "QRVO",  "name": "Qorvo",                   "sector": "Information Technology", "industry": "Semiconductors"},

    # ---------------- Health Care ----------------
    {"symbol": "LLY",   "name": "Eli Lilly",               "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "UNH",   "name": "UnitedHealth Group",      "sector": "Health Care", "industry": "Managed Health Care"},
    {"symbol": "JNJ",   "name": "Johnson & Johnson",       "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "ABBV",  "name": "AbbVie",                  "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "MRK",   "name": "Merck",                   "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "TMO",   "name": "Thermo Fisher Scientific","sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "ABT",   "name": "Abbott Laboratories",     "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "ISRG",  "name": "Intuitive Surgical",      "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "DHR",   "name": "Danaher",                 "sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "PFE",   "name": "Pfizer",                  "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "AMGN",  "name": "Amgen",                   "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "SYK",   "name": "Stryker",                 "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "BSX",   "name": "Boston Scientific",       "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "VRTX",  "name": "Vertex Pharmaceuticals",  "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "GILD",  "name": "Gilead Sciences",         "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "MDT",   "name": "Medtronic",               "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "BMY",   "name": "Bristol Myers Squibb",    "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "ELV",   "name": "Elevance Health",         "sector": "Health Care", "industry": "Managed Health Care"},
    {"symbol": "REGN",  "name": "Regeneron Pharmaceuticals","sector":"Health Care", "industry": "Biotechnology"},
    {"symbol": "CI",    "name": "Cigna",                   "sector": "Health Care", "industry": "Managed Health Care"},
    {"symbol": "CVS",   "name": "CVS Health",              "sector": "Health Care", "industry": "Health Care Services"},
    {"symbol": "MCK",   "name": "McKesson",                "sector": "Health Care", "industry": "Health Care Distribution"},
    {"symbol": "HCA",   "name": "HCA Healthcare",          "sector": "Health Care", "industry": "Health Care Facilities"},
    {"symbol": "ZTS",   "name": "Zoetis",                  "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "EW",    "name": "Edwards Lifesciences",    "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "BDX",   "name": "Becton Dickinson",        "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "IDXX",  "name": "IDEXX Laboratories",      "sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "HUM",   "name": "Humana",                  "sector": "Health Care", "industry": "Managed Health Care"},
    {"symbol": "CNC",   "name": "Centene",                 "sector": "Health Care", "industry": "Managed Health Care"},
    {"symbol": "A",     "name": "Agilent Technologies",    "sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "DXCM",  "name": "Dexcom",                  "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "MTD",   "name": "Mettler-Toledo",          "sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "RMD",   "name": "ResMed",                  "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "IQV",   "name": "IQVIA Holdings",          "sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "BIIB",  "name": "Biogen",                  "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "MRNA",  "name": "Moderna",                 "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "WAT",   "name": "Waters",                  "sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "MOH",   "name": "Molina Healthcare",       "sector": "Health Care", "industry": "Managed Health Care"},
    {"symbol": "ALGN",  "name": "Align Technology",        "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "STE",   "name": "STERIS",                  "sector": "Health Care", "industry": "Health Care Equipment"},
    {"symbol": "HOLX",  "name": "Hologic",                 "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "ZBH",   "name": "Zimmer Biomet",           "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "BAX",   "name": "Baxter International",    "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "WST",   "name": "West Pharmaceutical",     "sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "VTRS",  "name": "Viatris",                 "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "COR",   "name": "Cencora",                 "sector": "Health Care", "industry": "Health Care Distribution"},
    {"symbol": "CAH",   "name": "Cardinal Health",         "sector": "Health Care", "industry": "Health Care Distribution"},
    {"symbol": "HSIC",  "name": "Henry Schein",            "sector": "Health Care", "industry": "Health Care Distribution"},
    {"symbol": "TFX",   "name": "Teleflex",                "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "INCY",  "name": "Incyte",                  "sector": "Health Care", "industry": "Biotechnology"},
    {"symbol": "PODD",  "name": "Insulet",                 "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "GEHC",  "name": "GE HealthCare",           "sector": "Health Care", "industry": "Medical Devices"},
    {"symbol": "DGX",   "name": "Quest Diagnostics",       "sector": "Health Care", "industry": "Health Care Services"},
    {"symbol": "LH",    "name": "Labcorp",                 "sector": "Health Care", "industry": "Health Care Services"},
    {"symbol": "VEEV",  "name": "Veeva Systems",           "sector": "Health Care", "industry": "Health Care Technology"},
    {"symbol": "RVTY",  "name": "Revvity",                 "sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "DVA",   "name": "DaVita",                  "sector": "Health Care", "industry": "Health Care Facilities"},
    {"symbol": "TECH",  "name": "Bio-Techne",              "sector": "Health Care", "industry": "Life Sciences Tools"},
    {"symbol": "CTLT",  "name": "Catalent",                "sector": "Health Care", "industry": "Pharmaceuticals"},
    {"symbol": "UHS",   "name": "Universal Health Services","sector":"Health Care", "industry": "Health Care Facilities"},
    {"symbol": "CRL",   "name": "Charles River Laboratories","sector":"Health Care","industry": "Life Sciences Tools"},
    {"symbol": "PFG",   "name": "Principal Financial",     "sector": "Financials",  "industry": "Insurance"},

    # ---------------- Financials ----------------
    {"symbol": "BRK-B", "name": "Berkshire Hathaway",      "sector": "Financials", "industry": "Multi-line Insurance"},
    {"symbol": "JPM",   "name": "JPMorgan Chase",          "sector": "Financials", "industry": "Banks"},
    {"symbol": "V",     "name": "Visa",                    "sector": "Financials", "industry": "Transaction Processing"},
    {"symbol": "MA",    "name": "Mastercard",              "sector": "Financials", "industry": "Transaction Processing"},
    {"symbol": "BAC",   "name": "Bank of America",         "sector": "Financials", "industry": "Banks"},
    {"symbol": "WFC",   "name": "Wells Fargo",             "sector": "Financials", "industry": "Banks"},
    {"symbol": "GS",    "name": "Goldman Sachs",           "sector": "Financials", "industry": "Investment Banking"},
    {"symbol": "MS",    "name": "Morgan Stanley",          "sector": "Financials", "industry": "Investment Banking"},
    {"symbol": "AXP",   "name": "American Express",        "sector": "Financials", "industry": "Consumer Finance"},
    {"symbol": "C",     "name": "Citigroup",               "sector": "Financials", "industry": "Banks"},
    {"symbol": "BLK",   "name": "BlackRock",               "sector": "Financials", "industry": "Asset Management"},
    {"symbol": "SCHW",  "name": "Charles Schwab",          "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "SPGI",  "name": "S&P Global",              "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "PGR",   "name": "Progressive",             "sector": "Financials", "industry": "Insurance"},
    {"symbol": "CB",    "name": "Chubb",                   "sector": "Financials", "industry": "Insurance"},
    {"symbol": "MMC",   "name": "Marsh McLennan",          "sector": "Financials", "industry": "Insurance Brokers"},
    {"symbol": "PNC",   "name": "PNC Financial Services",  "sector": "Financials", "industry": "Banks"},
    {"symbol": "USB",   "name": "U.S. Bancorp",            "sector": "Financials", "industry": "Banks"},
    {"symbol": "CME",   "name": "CME Group",               "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "ICE",   "name": "Intercontinental Exchange","sector": "Financials","industry": "Capital Markets"},
    {"symbol": "AON",   "name": "Aon",                     "sector": "Financials", "industry": "Insurance Brokers"},
    {"symbol": "AJG",   "name": "Arthur J. Gallagher",     "sector": "Financials", "industry": "Insurance Brokers"},
    {"symbol": "TFC",   "name": "Truist Financial",        "sector": "Financials", "industry": "Banks"},
    {"symbol": "COF",   "name": "Capital One Financial",   "sector": "Financials", "industry": "Consumer Finance"},
    {"symbol": "MCO",   "name": "Moody's",                 "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "BX",    "name": "Blackstone",              "sector": "Financials", "industry": "Asset Management"},
    {"symbol": "MET",   "name": "MetLife",                 "sector": "Financials", "industry": "Insurance"},
    {"symbol": "PRU",   "name": "Prudential Financial",    "sector": "Financials", "industry": "Insurance"},
    {"symbol": "AIG",   "name": "American International Group","sector":"Financials","industry": "Insurance"},
    {"symbol": "ALL",   "name": "Allstate",                "sector": "Financials", "industry": "Insurance"},
    {"symbol": "TRV",   "name": "Travelers",               "sector": "Financials", "industry": "Insurance"},
    {"symbol": "AFL",   "name": "Aflac",                   "sector": "Financials", "industry": "Insurance"},
    {"symbol": "FIS",   "name": "Fidelity National Information","sector":"Financials","industry": "Transaction Processing"},
    {"symbol": "FI",    "name": "Fiserv",                  "sector": "Financials", "industry": "Transaction Processing"},
    {"symbol": "PYPL",  "name": "PayPal",                  "sector": "Financials", "industry": "Transaction Processing"},
    {"symbol": "BK",    "name": "Bank of New York Mellon", "sector": "Financials", "industry": "Banks"},
    {"symbol": "STT",   "name": "State Street",            "sector": "Financials", "industry": "Banks"},
    {"symbol": "NTRS",  "name": "Northern Trust",          "sector": "Financials", "industry": "Banks"},
    {"symbol": "FITB",  "name": "Fifth Third Bancorp",     "sector": "Financials", "industry": "Banks"},
    {"symbol": "KEY",   "name": "KeyCorp",                 "sector": "Financials", "industry": "Banks"},
    {"symbol": "HBAN",  "name": "Huntington Bancshares",   "sector": "Financials", "industry": "Banks"},
    {"symbol": "RF",    "name": "Regions Financial",       "sector": "Financials", "industry": "Banks"},
    {"symbol": "CFG",   "name": "Citizens Financial",      "sector": "Financials", "industry": "Banks"},
    {"symbol": "MTB",   "name": "M&T Bank",                "sector": "Financials", "industry": "Banks"},
    {"symbol": "ZION",  "name": "Zions Bancorporation",    "sector": "Financials", "industry": "Banks"},
    {"symbol": "CINF",  "name": "Cincinnati Financial",    "sector": "Financials", "industry": "Insurance"},
    {"symbol": "WTW",   "name": "Willis Towers Watson",    "sector": "Financials", "industry": "Insurance Brokers"},
    {"symbol": "BRO",   "name": "Brown & Brown",           "sector": "Financials", "industry": "Insurance Brokers"},
    {"symbol": "RJF",   "name": "Raymond James",           "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "DFS",   "name": "Discover Financial",      "sector": "Financials", "industry": "Consumer Finance"},
    {"symbol": "SYF",   "name": "Synchrony Financial",     "sector": "Financials", "industry": "Consumer Finance"},
    {"symbol": "GL",    "name": "Globe Life",              "sector": "Financials", "industry": "Insurance"},
    {"symbol": "HIG",   "name": "Hartford Financial",      "sector": "Financials", "industry": "Insurance"},
    {"symbol": "WRB",   "name": "W. R. Berkley",           "sector": "Financials", "industry": "Insurance"},
    {"symbol": "L",     "name": "Loews",                   "sector": "Financials", "industry": "Insurance"},
    {"symbol": "ERIE",  "name": "Erie Indemnity",          "sector": "Financials", "industry": "Insurance"},
    {"symbol": "ACGL",  "name": "Arch Capital Group",      "sector": "Financials", "industry": "Insurance"},
    {"symbol": "TROW",  "name": "T. Rowe Price",           "sector": "Financials", "industry": "Asset Management"},
    {"symbol": "NDAQ",  "name": "Nasdaq",                  "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "MKTX",  "name": "MarketAxess",             "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "MSCI",  "name": "MSCI",                    "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "AMP",   "name": "Ameriprise Financial",    "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "FDS",   "name": "FactSet Research",        "sector": "Financials", "industry": "Capital Markets"},
    {"symbol": "JKHY",  "name": "Jack Henry & Associates", "sector": "Financials", "industry": "Transaction Processing"},
    {"symbol": "EG",    "name": "Everest Group",           "sector": "Financials", "industry": "Insurance"},
    {"symbol": "IVZ",   "name": "Invesco",                 "sector": "Financials", "industry": "Asset Management"},
    {"symbol": "BEN",   "name": "Franklin Resources",      "sector": "Financials", "industry": "Asset Management"},

    # ---------------- Consumer Discretionary ----------------
    {"symbol": "AMZN",  "name": "Amazon",                  "sector": "Consumer Discretionary", "industry": "Internet Retail"},
    {"symbol": "TSLA",  "name": "Tesla",                   "sector": "Consumer Discretionary", "industry": "Automobiles"},
    {"symbol": "HD",    "name": "Home Depot",              "sector": "Consumer Discretionary", "industry": "Home Improvement"},
    {"symbol": "MCD",   "name": "McDonald's",              "sector": "Consumer Discretionary", "industry": "Restaurants"},
    {"symbol": "BKNG",  "name": "Booking Holdings",        "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "LOW",   "name": "Lowe's",                  "sector": "Consumer Discretionary", "industry": "Home Improvement"},
    {"symbol": "NKE",   "name": "Nike",                    "sector": "Consumer Discretionary", "industry": "Apparel"},
    {"symbol": "TJX",   "name": "TJX Companies",           "sector": "Consumer Discretionary", "industry": "Apparel Retail"},
    {"symbol": "SBUX",  "name": "Starbucks",               "sector": "Consumer Discretionary", "industry": "Restaurants"},
    {"symbol": "ABNB",  "name": "Airbnb",                  "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "MAR",   "name": "Marriott International",  "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "ORLY",  "name": "O'Reilly Automotive",     "sector": "Consumer Discretionary", "industry": "Specialty Retail"},
    {"symbol": "GM",    "name": "General Motors",          "sector": "Consumer Discretionary", "industry": "Automobiles"},
    {"symbol": "F",     "name": "Ford Motor",              "sector": "Consumer Discretionary", "industry": "Automobiles"},
    {"symbol": "CMG",   "name": "Chipotle Mexican Grill",  "sector": "Consumer Discretionary", "industry": "Restaurants"},
    {"symbol": "AZO",   "name": "AutoZone",                "sector": "Consumer Discretionary", "industry": "Specialty Retail"},
    {"symbol": "HLT",   "name": "Hilton Worldwide",        "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "ROST",  "name": "Ross Stores",             "sector": "Consumer Discretionary", "industry": "Apparel Retail"},
    {"symbol": "YUM",   "name": "Yum! Brands",             "sector": "Consumer Discretionary", "industry": "Restaurants"},
    {"symbol": "RCL",   "name": "Royal Caribbean",         "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "CCL",   "name": "Carnival",                "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "NCLH",  "name": "Norwegian Cruise Line",   "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "LULU",  "name": "Lululemon Athletica",     "sector": "Consumer Discretionary", "industry": "Apparel"},
    {"symbol": "DRI",   "name": "Darden Restaurants",      "sector": "Consumer Discretionary", "industry": "Restaurants"},
    {"symbol": "EBAY",  "name": "eBay",                    "sector": "Consumer Discretionary", "industry": "Internet Retail"},
    {"symbol": "ETSY",  "name": "Etsy",                    "sector": "Consumer Discretionary", "industry": "Internet Retail"},
    {"symbol": "EXPE",  "name": "Expedia Group",           "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "TSCO",  "name": "Tractor Supply",          "sector": "Consumer Discretionary", "industry": "Specialty Retail"},
    {"symbol": "ULTA",  "name": "Ulta Beauty",             "sector": "Consumer Discretionary", "industry": "Specialty Retail"},
    {"symbol": "DPZ",   "name": "Domino's Pizza",          "sector": "Consumer Discretionary", "industry": "Restaurants"},
    {"symbol": "DECK",  "name": "Deckers Outdoor",         "sector": "Consumer Discretionary", "industry": "Footwear"},
    {"symbol": "BBY",   "name": "Best Buy",                "sector": "Consumer Discretionary", "industry": "Specialty Retail"},
    {"symbol": "POOL",  "name": "Pool Corporation",        "sector": "Consumer Discretionary", "industry": "Specialty Retail"},
    {"symbol": "TPR",   "name": "Tapestry",                "sector": "Consumer Discretionary", "industry": "Apparel"},
    {"symbol": "RL",    "name": "Ralph Lauren",            "sector": "Consumer Discretionary", "industry": "Apparel"},
    {"symbol": "HAS",   "name": "Hasbro",                  "sector": "Consumer Discretionary", "industry": "Leisure Products"},
    {"symbol": "MAT",   "name": "Mattel",                  "sector": "Consumer Discretionary", "industry": "Leisure Products"},
    {"symbol": "WHR",   "name": "Whirlpool",               "sector": "Consumer Discretionary", "industry": "Household Durables"},
    {"symbol": "LEN",   "name": "Lennar",                  "sector": "Consumer Discretionary", "industry": "Homebuilding"},
    {"symbol": "DHI",   "name": "D.R. Horton",             "sector": "Consumer Discretionary", "industry": "Homebuilding"},
    {"symbol": "NVR",   "name": "NVR",                     "sector": "Consumer Discretionary", "industry": "Homebuilding"},
    {"symbol": "PHM",   "name": "PulteGroup",              "sector": "Consumer Discretionary", "industry": "Homebuilding"},
    {"symbol": "GRMN",  "name": "Garmin",                  "sector": "Consumer Discretionary", "industry": "Consumer Electronics"},
    {"symbol": "APTV",  "name": "Aptiv",                   "sector": "Consumer Discretionary", "industry": "Auto Parts"},
    {"symbol": "BWA",   "name": "BorgWarner",              "sector": "Consumer Discretionary", "industry": "Auto Parts"},
    {"symbol": "LKQ",   "name": "LKQ",                     "sector": "Consumer Discretionary", "industry": "Auto Parts"},
    {"symbol": "MGM",   "name": "MGM Resorts",             "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "WYNN",  "name": "Wynn Resorts",            "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "LVS",   "name": "Las Vegas Sands",         "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "CZR",   "name": "Caesars Entertainment",   "sector": "Consumer Discretionary", "industry": "Hotels & Leisure"},
    {"symbol": "DLTR",  "name": "Dollar Tree",             "sector": "Consumer Discretionary", "industry": "Discount Retail"},
    {"symbol": "DG",    "name": "Dollar General",          "sector": "Consumer Staples",       "industry": "Discount Retail"},
    {"symbol": "GPC",   "name": "Genuine Parts",           "sector": "Consumer Discretionary", "industry": "Specialty Retail"},
    {"symbol": "DECK",  "name": "Deckers Outdoor",         "sector": "Consumer Discretionary", "industry": "Footwear"},
    {"symbol": "VFC",   "name": "VF Corp",                 "sector": "Consumer Discretionary", "industry": "Apparel"},
    {"symbol": "PVH",   "name": "PVH",                     "sector": "Consumer Discretionary", "industry": "Apparel"},
    {"symbol": "HRB",   "name": "H&R Block",               "sector": "Consumer Discretionary", "industry": "Consumer Services"},
    {"symbol": "TXRH",  "name": "Texas Roadhouse",         "sector": "Consumer Discretionary", "industry": "Restaurants"},
    {"symbol": "GPS",   "name": "Gap",                     "sector": "Consumer Discretionary", "industry": "Apparel Retail"},

    # ---------------- Communication Services ----------------
    {"symbol": "GOOGL", "name": "Alphabet (Class A)",      "sector": "Communication Services", "industry": "Interactive Media"},
    {"symbol": "GOOG",  "name": "Alphabet (Class C)",      "sector": "Communication Services", "industry": "Interactive Media"},
    {"symbol": "META",  "name": "Meta Platforms",          "sector": "Communication Services", "industry": "Interactive Media"},
    {"symbol": "NFLX",  "name": "Netflix",                 "sector": "Communication Services", "industry": "Entertainment"},
    {"symbol": "DIS",   "name": "Walt Disney",             "sector": "Communication Services", "industry": "Entertainment"},
    {"symbol": "T",     "name": "AT&T",                    "sector": "Communication Services", "industry": "Telecom Services"},
    {"symbol": "VZ",    "name": "Verizon Communications",  "sector": "Communication Services", "industry": "Telecom Services"},
    {"symbol": "TMUS",  "name": "T-Mobile US",             "sector": "Communication Services", "industry": "Wireless Telecom"},
    {"symbol": "CMCSA", "name": "Comcast",                 "sector": "Communication Services", "industry": "Media"},
    {"symbol": "CHTR",  "name": "Charter Communications",  "sector": "Communication Services", "industry": "Media"},
    {"symbol": "WBD",   "name": "Warner Bros. Discovery",  "sector": "Communication Services", "industry": "Entertainment"},
    {"symbol": "PARA",  "name": "Paramount Global",        "sector": "Communication Services", "industry": "Entertainment"},
    {"symbol": "EA",    "name": "Electronic Arts",         "sector": "Communication Services", "industry": "Interactive Entertainment"},
    {"symbol": "TTWO",  "name": "Take-Two Interactive",    "sector": "Communication Services", "industry": "Interactive Entertainment"},
    {"symbol": "OMC",   "name": "Omnicom Group",           "sector": "Communication Services", "industry": "Advertising"},
    {"symbol": "IPG",   "name": "Interpublic Group",       "sector": "Communication Services", "industry": "Advertising"},
    {"symbol": "MTCH",  "name": "Match Group",             "sector": "Communication Services", "industry": "Interactive Media"},
    {"symbol": "NWSA",  "name": "News Corp (Class A)",     "sector": "Communication Services", "industry": "Publishing"},
    {"symbol": "NWS",   "name": "News Corp (Class B)",     "sector": "Communication Services", "industry": "Publishing"},
    {"symbol": "FOXA",  "name": "Fox Corporation (A)",     "sector": "Communication Services", "industry": "Media"},
    {"symbol": "FOX",   "name": "Fox Corporation (B)",     "sector": "Communication Services", "industry": "Media"},
    {"symbol": "DASH",  "name": "DoorDash",                "sector": "Communication Services", "industry": "Interactive Media"},
    {"symbol": "LYV",   "name": "Live Nation Entertainment","sector":"Communication Services", "industry": "Entertainment"},

    # ---------------- Industrials ----------------
    {"symbol": "GE",    "name": "GE Aerospace",            "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "RTX",   "name": "RTX",                     "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "CAT",   "name": "Caterpillar",             "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "BA",    "name": "Boeing",                  "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "UNP",   "name": "Union Pacific",           "sector": "Industrials", "industry": "Railroads"},
    {"symbol": "HON",   "name": "Honeywell International", "sector": "Industrials", "industry": "Industrial Conglomerates"},
    {"symbol": "DE",    "name": "Deere & Company",         "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "UPS",   "name": "United Parcel Service",   "sector": "Industrials", "industry": "Air Freight & Logistics"},
    {"symbol": "LMT",   "name": "Lockheed Martin",         "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "ETN",   "name": "Eaton",                   "sector": "Industrials", "industry": "Electrical Equipment"},
    {"symbol": "ADP",   "name": "Automatic Data Processing","sector": "Industrials", "industry": "HR Services"},
    {"symbol": "TT",    "name": "Trane Technologies",      "sector": "Industrials", "industry": "Building Products"},
    {"symbol": "PH",    "name": "Parker Hannifin",         "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "GD",    "name": "General Dynamics",        "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "NOC",   "name": "Northrop Grumman",        "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "EMR",   "name": "Emerson Electric",        "sector": "Industrials", "industry": "Electrical Equipment"},
    {"symbol": "ITW",   "name": "Illinois Tool Works",     "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "CSX",   "name": "CSX",                     "sector": "Industrials", "industry": "Railroads"},
    {"symbol": "NSC",   "name": "Norfolk Southern",        "sector": "Industrials", "industry": "Railroads"},
    {"symbol": "FDX",   "name": "FedEx",                   "sector": "Industrials", "industry": "Air Freight & Logistics"},
    {"symbol": "WM",    "name": "Waste Management",        "sector": "Industrials", "industry": "Environmental Services"},
    {"symbol": "RSG",   "name": "Republic Services",       "sector": "Industrials", "industry": "Environmental Services"},
    {"symbol": "PCAR",  "name": "PACCAR",                  "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "GWW",   "name": "W.W. Grainger",           "sector": "Industrials", "industry": "Trading Companies"},
    {"symbol": "FAST",  "name": "Fastenal",                "sector": "Industrials", "industry": "Trading Companies"},
    {"symbol": "URI",   "name": "United Rentals",          "sector": "Industrials", "industry": "Trading Companies"},
    {"symbol": "ROK",   "name": "Rockwell Automation",     "sector": "Industrials", "industry": "Electrical Equipment"},
    {"symbol": "CMI",   "name": "Cummins",                 "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "XYL",   "name": "Xylem",                   "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "DOV",   "name": "Dover",                   "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "OTIS",  "name": "Otis Worldwide",          "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "CARR",  "name": "Carrier Global",          "sector": "Industrials", "industry": "Building Products"},
    {"symbol": "PWR",   "name": "Quanta Services",         "sector": "Industrials", "industry": "Construction"},
    {"symbol": "PNR",   "name": "Pentair",                 "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "AME",   "name": "AMETEK",                  "sector": "Industrials", "industry": "Electrical Equipment"},
    {"symbol": "TDG",   "name": "TransDigm Group",         "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "HWM",   "name": "Howmet Aerospace",        "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "AXON",  "name": "Axon Enterprise",         "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "LHX",   "name": "L3Harris Technologies",   "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "TXT",   "name": "Textron",                 "sector": "Industrials", "industry": "Aerospace & Defense"},
    {"symbol": "SWK",   "name": "Stanley Black & Decker",  "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "VRSK",  "name": "Verisk Analytics",        "sector": "Industrials", "industry": "Research Services"},
    {"symbol": "GPN",   "name": "Global Payments",         "sector": "Financials",  "industry": "Transaction Processing"},
    {"symbol": "PAYX",  "name": "Paychex",                 "sector": "Industrials", "industry": "HR Services"},
    {"symbol": "BR",    "name": "Broadridge Financial",    "sector": "Industrials", "industry": "Capital Markets Services"},
    {"symbol": "RHI",   "name": "Robert Half",             "sector": "Industrials", "industry": "HR Services"},
    {"symbol": "EFX",   "name": "Equifax",                 "sector": "Industrials", "industry": "Research Services"},
    {"symbol": "JCI",   "name": "Johnson Controls",        "sector": "Industrials", "industry": "Building Products"},
    {"symbol": "DAL",   "name": "Delta Air Lines",         "sector": "Industrials", "industry": "Airlines"},
    {"symbol": "UAL",   "name": "United Airlines Holdings","sector": "Industrials", "industry": "Airlines"},
    {"symbol": "AAL",   "name": "American Airlines Group", "sector": "Industrials", "industry": "Airlines"},
    {"symbol": "LUV",   "name": "Southwest Airlines",      "sector": "Industrials", "industry": "Airlines"},
    {"symbol": "ALK",   "name": "Alaska Air Group",        "sector": "Industrials", "industry": "Airlines"},
    {"symbol": "EXPD",  "name": "Expeditors International","sector": "Industrials", "industry": "Air Freight & Logistics"},
    {"symbol": "CHRW",  "name": "C.H. Robinson Worldwide", "sector": "Industrials", "industry": "Air Freight & Logistics"},
    {"symbol": "ODFL",  "name": "Old Dominion Freight Line","sector": "Industrials","industry": "Trucking"},
    {"symbol": "JBHT",  "name": "J.B. Hunt Transport",     "sector": "Industrials", "industry": "Trucking"},
    {"symbol": "MAS",   "name": "Masco",                   "sector": "Industrials", "industry": "Building Products"},
    {"symbol": "ALLE",  "name": "Allegion",                "sector": "Industrials", "industry": "Building Products"},
    {"symbol": "BLDR",  "name": "Builders FirstSource",    "sector": "Industrials", "industry": "Building Products"},
    {"symbol": "GNRC",  "name": "Generac Holdings",        "sector": "Industrials", "industry": "Electrical Equipment"},
    {"symbol": "VLTO",  "name": "Veralto",                 "sector": "Industrials", "industry": "Environmental Services"},
    {"symbol": "IR",    "name": "Ingersoll Rand",          "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "FTV",   "name": "Fortive",                 "sector": "Industrials", "industry": "Electrical Equipment"},
    {"symbol": "CTAS",  "name": "Cintas",                  "sector": "Industrials", "industry": "Commercial Services"},
    {"symbol": "ROL",   "name": "Rollins",                 "sector": "Industrials", "industry": "Commercial Services"},
    {"symbol": "UBER",  "name": "Uber Technologies",       "sector": "Industrials", "industry": "Ground Transportation"},
    {"symbol": "WAB",   "name": "Westinghouse Air Brake",  "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "SNA",   "name": "Snap-on",                 "sector": "Industrials", "industry": "Machinery"},
    {"symbol": "NDSN",  "name": "Nordson",                 "sector": "Industrials", "industry": "Machinery"},

    # ---------------- Consumer Staples ----------------
    {"symbol": "WMT",   "name": "Walmart",                 "sector": "Consumer Staples", "industry": "Hypermarkets"},
    {"symbol": "PG",    "name": "Procter & Gamble",        "sector": "Consumer Staples", "industry": "Household Products"},
    {"symbol": "COST",  "name": "Costco Wholesale",        "sector": "Consumer Staples", "industry": "Hypermarkets"},
    {"symbol": "KO",    "name": "Coca-Cola",               "sector": "Consumer Staples", "industry": "Beverages"},
    {"symbol": "PEP",   "name": "PepsiCo",                 "sector": "Consumer Staples", "industry": "Beverages"},
    {"symbol": "PM",    "name": "Philip Morris International","sector":"Consumer Staples","industry": "Tobacco"},
    {"symbol": "MO",    "name": "Altria Group",            "sector": "Consumer Staples", "industry": "Tobacco"},
    {"symbol": "MDLZ",  "name": "Mondelez International",  "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "CL",    "name": "Colgate-Palmolive",       "sector": "Consumer Staples", "industry": "Household Products"},
    {"symbol": "TGT",   "name": "Target",                  "sector": "Consumer Staples", "industry": "General Merchandise"},
    {"symbol": "KMB",   "name": "Kimberly-Clark",          "sector": "Consumer Staples", "industry": "Household Products"},
    {"symbol": "STZ",   "name": "Constellation Brands",    "sector": "Consumer Staples", "industry": "Beverages"},
    {"symbol": "MNST",  "name": "Monster Beverage",        "sector": "Consumer Staples", "industry": "Beverages"},
    {"symbol": "KDP",   "name": "Keurig Dr Pepper",        "sector": "Consumer Staples", "industry": "Beverages"},
    {"symbol": "GIS",   "name": "General Mills",           "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "KHC",   "name": "Kraft Heinz",             "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "SYY",   "name": "Sysco",                   "sector": "Consumer Staples", "industry": "Food Distribution"},
    {"symbol": "ADM",   "name": "Archer Daniels Midland",  "sector": "Consumer Staples", "industry": "Agricultural Products"},
    {"symbol": "HSY",   "name": "Hershey",                 "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "K",     "name": "Kellanova",               "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "MKC",   "name": "McCormick",               "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "CPB",   "name": "Campbell Soup",           "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "CHD",   "name": "Church & Dwight",         "sector": "Consumer Staples", "industry": "Household Products"},
    {"symbol": "CLX",   "name": "Clorox",                  "sector": "Consumer Staples", "industry": "Household Products"},
    {"symbol": "TAP",   "name": "Molson Coors Beverage",   "sector": "Consumer Staples", "industry": "Beverages"},
    {"symbol": "TSN",   "name": "Tyson Foods",             "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "EL",    "name": "Estee Lauder",            "sector": "Consumer Staples", "industry": "Personal Products"},
    {"symbol": "BG",    "name": "Bunge Global",            "sector": "Consumer Staples", "industry": "Agricultural Products"},
    {"symbol": "LW",    "name": "Lamb Weston Holdings",    "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "KR",    "name": "Kroger",                  "sector": "Consumer Staples", "industry": "Food Retail"},
    {"symbol": "WBA",   "name": "Walgreens Boots Alliance","sector":"Consumer Staples","industry": "Drug Retail"},
    {"symbol": "BF-B",  "name": "Brown-Forman (Class B)",  "sector": "Consumer Staples", "industry": "Beverages"},
    {"symbol": "CAG",   "name": "Conagra Brands",          "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "HRL",   "name": "Hormel Foods",            "sector": "Consumer Staples", "industry": "Packaged Foods"},
    {"symbol": "SJM",   "name": "J.M. Smucker",            "sector": "Consumer Staples", "industry": "Packaged Foods"},

    # ---------------- Energy ----------------
    {"symbol": "XOM",   "name": "Exxon Mobil",             "sector": "Energy", "industry": "Integrated Oil & Gas"},
    {"symbol": "CVX",   "name": "Chevron",                 "sector": "Energy", "industry": "Integrated Oil & Gas"},
    {"symbol": "COP",   "name": "ConocoPhillips",          "sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "EOG",   "name": "EOG Resources",           "sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "SLB",   "name": "Schlumberger",            "sector": "Energy", "industry": "Oil & Gas Services"},
    {"symbol": "MPC",   "name": "Marathon Petroleum",      "sector": "Energy", "industry": "Oil & Gas Refining"},
    {"symbol": "PSX",   "name": "Phillips 66",             "sector": "Energy", "industry": "Oil & Gas Refining"},
    {"symbol": "VLO",   "name": "Valero Energy",           "sector": "Energy", "industry": "Oil & Gas Refining"},
    {"symbol": "OXY",   "name": "Occidental Petroleum",    "sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "PXD",   "name": "Pioneer Natural Resources","sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "WMB",   "name": "Williams Companies",      "sector": "Energy", "industry": "Oil & Gas Storage"},
    {"symbol": "KMI",   "name": "Kinder Morgan",           "sector": "Energy", "industry": "Oil & Gas Storage"},
    {"symbol": "OKE",   "name": "ONEOK",                   "sector": "Energy", "industry": "Oil & Gas Storage"},
    {"symbol": "HES",   "name": "Hess",                    "sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "FANG",  "name": "Diamondback Energy",      "sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "DVN",   "name": "Devon Energy",            "sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "BKR",   "name": "Baker Hughes",            "sector": "Energy", "industry": "Oil & Gas Services"},
    {"symbol": "HAL",   "name": "Halliburton",             "sector": "Energy", "industry": "Oil & Gas Services"},
    {"symbol": "TRGP",  "name": "Targa Resources",         "sector": "Energy", "industry": "Oil & Gas Storage"},
    {"symbol": "EQT",   "name": "EQT",                     "sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "APA",   "name": "APA",                     "sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "MRO",   "name": "Marathon Oil",            "sector": "Energy", "industry": "Oil & Gas E&P"},
    {"symbol": "CTRA",  "name": "Coterra Energy",          "sector": "Energy", "industry": "Oil & Gas E&P"},

    # ---------------- Utilities ----------------
    {"symbol": "NEE",   "name": "NextEra Energy",          "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "SO",    "name": "Southern Company",        "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "DUK",   "name": "Duke Energy",             "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "CEG",   "name": "Constellation Energy",    "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "AEP",   "name": "American Electric Power", "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "SRE",   "name": "Sempra",                  "sector": "Utilities", "industry": "Multi-Utilities"},
    {"symbol": "D",     "name": "Dominion Energy",         "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "PCG",   "name": "PG&E",                    "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "EXC",   "name": "Exelon",                  "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "XEL",   "name": "Xcel Energy",             "sector": "Utilities", "industry": "Multi-Utilities"},
    {"symbol": "ED",    "name": "Consolidated Edison",     "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "WEC",   "name": "WEC Energy Group",        "sector": "Utilities", "industry": "Multi-Utilities"},
    {"symbol": "DTE",   "name": "DTE Energy",              "sector": "Utilities", "industry": "Multi-Utilities"},
    {"symbol": "PEG",   "name": "Public Service Enterprise","sector":"Utilities", "industry": "Electric Utilities"},
    {"symbol": "ES",    "name": "Eversource Energy",       "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "AWK",   "name": "American Water Works",    "sector": "Utilities", "industry": "Water Utilities"},
    {"symbol": "ETR",   "name": "Entergy",                 "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "FE",    "name": "FirstEnergy",             "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "PPL",   "name": "PPL",                     "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "AEE",   "name": "Ameren",                  "sector": "Utilities", "industry": "Multi-Utilities"},
    {"symbol": "CMS",   "name": "CMS Energy",              "sector": "Utilities", "industry": "Multi-Utilities"},
    {"symbol": "CNP",   "name": "CenterPoint Energy",      "sector": "Utilities", "industry": "Multi-Utilities"},
    {"symbol": "ATO",   "name": "Atmos Energy",            "sector": "Utilities", "industry": "Gas Utilities"},
    {"symbol": "NI",    "name": "NiSource",                "sector": "Utilities", "industry": "Multi-Utilities"},
    {"symbol": "EIX",   "name": "Edison International",    "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "LNT",   "name": "Alliant Energy",          "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "EVRG",  "name": "Evergy",                  "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "PNW",   "name": "Pinnacle West Capital",   "sector": "Utilities", "industry": "Electric Utilities"},
    {"symbol": "AES",   "name": "AES",                     "sector": "Utilities", "industry": "Independent Power"},
    {"symbol": "NRG",   "name": "NRG Energy",              "sector": "Utilities", "industry": "Independent Power"},
    {"symbol": "VST",   "name": "Vistra",                  "sector": "Utilities", "industry": "Independent Power"},

    # ---------------- Real Estate ----------------
    {"symbol": "PLD",   "name": "Prologis",                "sector": "Real Estate", "industry": "Industrial REIT"},
    {"symbol": "AMT",   "name": "American Tower",          "sector": "Real Estate", "industry": "Specialized REIT"},
    {"symbol": "EQIX",  "name": "Equinix",                 "sector": "Real Estate", "industry": "Specialized REIT"},
    {"symbol": "WELL",  "name": "Welltower",               "sector": "Real Estate", "industry": "Health Care REIT"},
    {"symbol": "CCI",   "name": "Crown Castle",            "sector": "Real Estate", "industry": "Specialized REIT"},
    {"symbol": "PSA",   "name": "Public Storage",          "sector": "Real Estate", "industry": "Specialized REIT"},
    {"symbol": "DLR",   "name": "Digital Realty Trust",    "sector": "Real Estate", "industry": "Specialized REIT"},
    {"symbol": "O",     "name": "Realty Income",           "sector": "Real Estate", "industry": "Retail REIT"},
    {"symbol": "SPG",   "name": "Simon Property Group",    "sector": "Real Estate", "industry": "Retail REIT"},
    {"symbol": "EXR",   "name": "Extra Space Storage",     "sector": "Real Estate", "industry": "Specialized REIT"},
    {"symbol": "AVB",   "name": "AvalonBay Communities",   "sector": "Real Estate", "industry": "Residential REIT"},
    {"symbol": "EQR",   "name": "Equity Residential",      "sector": "Real Estate", "industry": "Residential REIT"},
    {"symbol": "VTR",   "name": "Ventas",                  "sector": "Real Estate", "industry": "Health Care REIT"},
    {"symbol": "INVH",  "name": "Invitation Homes",        "sector": "Real Estate", "industry": "Residential REIT"},
    {"symbol": "ARE",   "name": "Alexandria Real Estate",  "sector": "Real Estate", "industry": "Office REIT"},
    {"symbol": "MAA",   "name": "Mid-America Apartment",   "sector": "Real Estate", "industry": "Residential REIT"},
    {"symbol": "ESS",   "name": "Essex Property Trust",    "sector": "Real Estate", "industry": "Residential REIT"},
    {"symbol": "UDR",   "name": "UDR",                     "sector": "Real Estate", "industry": "Residential REIT"},
    {"symbol": "KIM",   "name": "Kimco Realty",            "sector": "Real Estate", "industry": "Retail REIT"},
    {"symbol": "REG",   "name": "Regency Centers",         "sector": "Real Estate", "industry": "Retail REIT"},
    {"symbol": "FRT",   "name": "Federal Realty",          "sector": "Real Estate", "industry": "Retail REIT"},
    {"symbol": "BXP",   "name": "BXP",                     "sector": "Real Estate", "industry": "Office REIT"},
    {"symbol": "HST",   "name": "Host Hotels & Resorts",   "sector": "Real Estate", "industry": "Hotel REIT"},
    {"symbol": "WY",    "name": "Weyerhaeuser",            "sector": "Real Estate", "industry": "Specialized REIT"},
    {"symbol": "DOC",   "name": "Healthpeak Properties",   "sector": "Real Estate", "industry": "Health Care REIT"},
    {"symbol": "CPT",   "name": "Camden Property Trust",   "sector": "Real Estate", "industry": "Residential REIT"},
    {"symbol": "CBRE",  "name": "CBRE Group",              "sector": "Real Estate", "industry": "Real Estate Services"},
    {"symbol": "VICI",  "name": "VICI Properties",         "sector": "Real Estate", "industry": "Specialized REIT"},
    {"symbol": "IRM",   "name": "Iron Mountain",           "sector": "Real Estate", "industry": "Specialized REIT"},
    {"symbol": "AMP",   "name": "Ameriprise Financial",    "sector": "Financials",  "industry": "Capital Markets"},

    # ---------------- Materials ----------------
    {"symbol": "LIN",   "name": "Linde",                   "sector": "Materials", "industry": "Industrial Gases"},
    {"symbol": "SHW",   "name": "Sherwin-Williams",        "sector": "Materials", "industry": "Specialty Chemicals"},
    {"symbol": "APD",   "name": "Air Products & Chemicals","sector": "Materials", "industry": "Industrial Gases"},
    {"symbol": "ECL",   "name": "Ecolab",                  "sector": "Materials", "industry": "Specialty Chemicals"},
    {"symbol": "FCX",   "name": "Freeport-McMoRan",        "sector": "Materials", "industry": "Copper & Mining"},
    {"symbol": "NEM",   "name": "Newmont",                 "sector": "Materials", "industry": "Gold Mining"},
    {"symbol": "DD",    "name": "DuPont de Nemours",       "sector": "Materials", "industry": "Specialty Chemicals"},
    {"symbol": "DOW",   "name": "Dow",                     "sector": "Materials", "industry": "Commodity Chemicals"},
    {"symbol": "CTVA",  "name": "Corteva",                 "sector": "Materials", "industry": "Fertilizers"},
    {"symbol": "PPG",   "name": "PPG Industries",          "sector": "Materials", "industry": "Specialty Chemicals"},
    {"symbol": "NUE",   "name": "Nucor",                   "sector": "Materials", "industry": "Steel"},
    {"symbol": "STLD",  "name": "Steel Dynamics",          "sector": "Materials", "industry": "Steel"},
    {"symbol": "VMC",   "name": "Vulcan Materials",        "sector": "Materials", "industry": "Construction Materials"},
    {"symbol": "MLM",   "name": "Martin Marietta Materials","sector": "Materials","industry": "Construction Materials"},
    {"symbol": "IFF",   "name": "International Flavors",   "sector": "Materials", "industry": "Specialty Chemicals"},
    {"symbol": "LYB",   "name": "LyondellBasell",          "sector": "Materials", "industry": "Commodity Chemicals"},
    {"symbol": "ALB",   "name": "Albemarle",               "sector": "Materials", "industry": "Specialty Chemicals"},
    {"symbol": "MOS",   "name": "Mosaic",                  "sector": "Materials", "industry": "Fertilizers"},
    {"symbol": "CF",    "name": "CF Industries Holdings",  "sector": "Materials", "industry": "Fertilizers"},
    {"symbol": "FMC",   "name": "FMC",                     "sector": "Materials", "industry": "Fertilizers"},
    {"symbol": "EMN",   "name": "Eastman Chemical",        "sector": "Materials", "industry": "Specialty Chemicals"},
    {"symbol": "CE",    "name": "Celanese",                "sector": "Materials", "industry": "Specialty Chemicals"},
    {"symbol": "PKG",   "name": "Packaging Corp of America","sector": "Materials", "industry": "Paper & Packaging"},
    {"symbol": "IP",    "name": "International Paper",     "sector": "Materials", "industry": "Paper & Packaging"},
    {"symbol": "AVY",   "name": "Avery Dennison",          "sector": "Materials", "industry": "Paper & Packaging"},
    {"symbol": "BALL",  "name": "Ball",                    "sector": "Materials", "industry": "Containers & Packaging"},
    {"symbol": "AMCR",  "name": "Amcor",                   "sector": "Materials", "industry": "Containers & Packaging"},
    {"symbol": "CCK",   "name": "Crown Holdings",          "sector": "Materials", "industry": "Containers & Packaging"},
]


# Dedupe by symbol (defensive — in case of accidental duplicates above)
_seen = set()
_unique: List[Dict[str, str]] = []
for _r in SP500_COMPANIES:
    if _r["symbol"] in _seen:
        continue
    _seen.add(_r["symbol"])
    _unique.append(_r)
SP500_COMPANIES = _unique


# ---------------- core watchlist (always auto-refreshed) ----------------

CORE_WATCHLIST: List[str] = [
    # All major indices/commodities/FX
    "^GSPC", "^NDX", "^DJI", "^RUT", "^VIX", "^TNX",
    "DX-Y.NYB", "CL=F", "^FTSE", "^GDAXI", "^N225",
    # Broad ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Sector SPDR ETFs (full set, 11 sectors)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    # Mega-cap tech ("Magnificent 7")
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Other key bellwethers
    "BRK-B", "JPM", "UNH", "XOM",
    # Crypto majors (already prominent in header but useful for cross-checks)
    "BTC-USD", "ETH-USD",
]


# ---------------- public helpers ----------------

def all_equity_symbols() -> List[str]:
    return [r["symbol"] for r in SP500_COMPANIES]


def all_index_symbols() -> List[str]:
    return [r["symbol"] for r in MAJOR_INDICES]


def all_symbols() -> List[str]:
    return all_index_symbols() + all_equity_symbols()


def by_sector(sector: str) -> List[Dict[str, str]]:
    """All S&P 500 entries in the given GICS sector (case-insensitive)."""
    s = (sector or "").strip().lower()
    return [r for r in SP500_COMPANIES if r["sector"].lower() == s]


def get_meta(symbol: str) -> Optional[Dict[str, str]]:
    """Look up a symbol's full record (index or equity). None if unknown."""
    if not symbol:
        return None
    sym = symbol.strip().upper()
    for r in SP500_COMPANIES:
        if r["symbol"].upper() == sym:
            return r
    for r in MAJOR_INDICES:
        if r["symbol"].upper() == sym:
            # Normalize to same shape (no GICS for indices)
            return {"symbol": r["symbol"], "name": r["name"],
                    "sector": "Index/Commodity", "industry": r["type"]}
    return None


def sector_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {s: 0 for s in GICS_SECTORS}
    for r in SP500_COMPANIES:
        counts[r["sector"]] = counts.get(r["sector"], 0) + 1
    return counts


def search(query: str, limit: int = 20) -> List[Dict[str, str]]:
    """Case-insensitive name/symbol fuzzy search across the universe."""
    q = (query or "").strip().lower()
    if not q:
        return []
    out: List[Dict[str, str]] = []
    # Exact symbol match first
    for r in SP500_COMPANIES + [
        {"symbol": idx["symbol"], "name": idx["name"],
         "sector": "Index/Commodity", "industry": idx["type"]}
        for idx in MAJOR_INDICES
    ]:
        if r["symbol"].lower() == q:
            out.append(r)
    # Then symbol startswith
    for r in SP500_COMPANIES:
        if r["symbol"].lower().startswith(q) and r not in out:
            out.append(r)
    # Then name contains
    for r in SP500_COMPANIES:
        if q in r["name"].lower() and r not in out:
            out.append(r)
    return out[:limit]


def stats() -> Dict[str, int]:
    return {
        "indices": len(MAJOR_INDICES),
        "sp500": len(SP500_COMPANIES),
        "core_watchlist": len(CORE_WATCHLIST),
        "sectors": len(GICS_SECTORS),
    }
