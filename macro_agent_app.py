import os
import datetime as dt
from typing import Dict, Any, List, Optional
import textwrap  # за да махнем водещите интервали от HTML
import json
import re
import html as ihtml
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from binance.client import Client
from openai import OpenAI
from bs4 import BeautifulSoup
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Macro Agent", layout="wide")
if "yahoo_live_errors" not in st.session_state:
    st.session_state["yahoo_live_errors"] = {}
def password_gate():
    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        st.title("Login")
        pwd = st.text_input("Password", type="password")

        if pwd and pwd == st.secrets["APP_PASSWORD"]:
            st.session_state.auth = True
            st.rerun()
        elif pwd:
            st.error("Wrong password")

        st.stop()

password_gate()


def inject_secrets_to_env():
    for key in [
        "NEWSAPI_KEY",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ]:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])

inject_secrets_to_env()

# ------------------------------------
# LOAD ENV
# ------------------------------------
load_dotenv()

def get_secret(name: str, default: str = "") -> str:
    # 1) Streamlit Cloud secrets
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    # 2) Local env/.env
    return os.getenv(name, default).strip()

NEWSAPI_KEY = get_secret("NEWSAPI_KEY")
BINANCE_API_KEY = get_secret("BINANCE_API_KEY")
BINANCE_API_SECRET = get_secret("BINANCE_API_SECRET")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4.1-mini")

# ------------------------------------
# CONFIG
# ------------------------------------
DAYS_BACK = 365
RSI_PERIOD = 14

NEWS_HISTORY_FILE = "news_history.csv"
NEWS_RETENTION_DAYS = 90  # колко дни назад пазим новини в памет

# Retro CSS: фон черен, текст бял; таблиците ги оцветяваме отделно
retro_css = """
<style>
body, .stApp {
    background-color: #000000 !important;
    color: #ffffff !important;
}
</style>
"""
st.markdown(retro_css, unsafe_allow_html=True)

# Yahoo Finance assets (by class)
ASSETS_BY_CLASS: Dict[str, Dict[str, str]] = {
    "commodity": {
        "Gold (futures)": "GC=F",
        "Silver (futures)": "SI=F"

    },
    "index": {
        "S&P 500 index": "^GSPC",
        "Nasdaq 100 index": "^NDX",
    },
    "stock": {
        "NVIDIA": "NVDA",
        "Apple": "AAPL",
        "BlackRock": "BLK",
        "JPMorgan": "JPM",
        "Netflix": "NFLX",
        "Microsoft": "MSFT",
        "Tesla": "TSLA",
        "Alphabet": "GOOGL",
        "Amazon": "AMZN",
        "ASML (NASDAQ ADR)": "ASML",
        "L3Harris Technologies": "LHX",
        "AeroVironment": "AVAV",
        "Kratos Defense & Security": "KTOS",
        "Allianz (XETRA)": "ALV.DE",
        "Rheinmetall (XETRA)": "RHM.DE",
        
    },
    "crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
    },
}

# Binance spot symbols (за таба Crypto)
BINANCE_SYMBOLS: Dict[str, Dict[str, str]] = {
    "BTCUSDT": {"display": "BTC", "class": "crypto_spot"},
    "ETHUSDT": {"display": "ETH", "class": "crypto_spot"},
    "BNBUSDT": {"display": "BNB", "class": "crypto_spot"},
    "SOLUSDT": {"display": "SOL", "class": "crypto_spot"},
    "ADAUSDT": {"display": "ADA", "class": "crypto_spot"},
    "XRPUSDT": {"display": "XRP", "class": "crypto_spot"},
}

BINANCE_TIMEFRAMES = {
    "1d": "1d",
    "4h": "4h",
    "1h": "1h",
    "15m": "15m",
}

# Live ticker – кои символи да показваме хоризонтално (Binance crypto)
LIVE_TICKER_SYMBOLS = [
    ("GC=F", "GOLD"),
    ("SI=F", "SILVER"),

    ("BTCUSDT", "BTC"),
    ("ETHUSDT", "ETH"),
    ("BNBUSDT", "BNB"),
    ("SOLUSDT", "SOL"),
    ("ADAUSDT", "ADA"),
    ("XRPUSDT", "XRP"),
]


NEWS_KEYWORDS: List[str] = [
    "Bitcoin",
    "Ethereum",
    "Gold",
    "Silver",
    "S&P 500",
    "Nasdaq 100",
    "Nvidia",
    "Apple",
    "BlackRock",
    "JPMorgan",
    "Netflix",
    "Microsoft",
    "Tesla",
    "Alphabet",
    "Google",
    "Amazon",
    "Federal Reserve",
    "ECB",
    "Bank of Japan",
    "Bank of England",
    "IMF",
    "World Bank",
    "United States economy",
    "China economy",
    "EU economy",
    "Elon Musk",
    "Bill Gates",
    "Jerome Powell",
    "Christine Lagarde",
    "ASML",
    "L3Harris",
    "AeroVironment",
    "Kratos Defense",
    "Allianz",
    "Rheinmetall",
    "European defense spending",
    "NATO defense"
]

YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{}"
FED_BASE = "https://www.federalreserve.gov"

# ------------------------------------
# TA HELPERS
# ------------------------------------


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_jump_diffusion_metrics(
    close: pd.Series,
    bars_per_year: int = 252,
    jump_z: float = 3.0,
) -> Dict[str, Any]:
    """
    Practical jump-diffusion diagnostics (Merton-style) from log returns.

    - Identifies "jumps" as returns with |z| >= jump_z (z-score on log returns).
    - Estimates:
      lambda_year: expected number of jumps per year
      avg_jump_pct: average jump size in %
      jump_vol_pct: variability of jump sizes in %
      sigma_diffusion_pct: diffusion (non-jump) annualized-ish vol in %
      jump_risk_score: lambda_year * abs(avg_jump_pct)  (simple risk score)
    """
    s = close.dropna().astype(float)
    if len(s) < 120:
        return {
            "lambda_year": None,
            "avg_jump_pct": None,
            "jump_vol_pct": None,
            "sigma_diffusion_pct": None,
            "jump_risk_score": None,
            "jumps_count": 0,
        }

    r = np.log(s).diff().dropna()  # log returns
    if len(r) < 60 or float(r.std()) == 0.0:
        return {
            "lambda_year": None,
            "avg_jump_pct": None,
            "jump_vol_pct": None,
            "sigma_diffusion_pct": None,
            "jump_risk_score": None,
            "jumps_count": 0,
        }

    z = (r - r.mean()) / (r.std() + 1e-12)
    jump_mask = z.abs() >= float(jump_z)

    r_jump = r[jump_mask]
    r_norm = r[~jump_mask]

    jumps_count = int(r_jump.shape[0])
    lambda_year = (jumps_count / max(len(r), 1)) * float(bars_per_year)

    jump_moves_pct = (np.exp(r_jump) - 1.0) * 100.0 if jumps_count > 0 else np.array([])
    avg_jump_pct = float(np.mean(jump_moves_pct)) if jumps_count > 0 else 0.0
    jump_vol_pct = float(np.std(jump_moves_pct)) if jumps_count > 1 else 0.0

    sigma_diffusion = float(r_norm.std()) * np.sqrt(float(bars_per_year)) if len(r_norm) > 5 else None
    sigma_diffusion_pct = float((np.exp(sigma_diffusion) - 1.0) * 100.0) if sigma_diffusion is not None else None

    jump_risk_score = float(lambda_year * abs(avg_jump_pct))

    return {
        "lambda_year": round(lambda_year, 2),
        "avg_jump_pct": round(avg_jump_pct, 2),
        "jump_vol_pct": round(jump_vol_pct, 2),
        "sigma_diffusion_pct": round(sigma_diffusion_pct, 2) if sigma_diffusion_pct is not None else None,
        "jump_risk_score": round(jump_risk_score, 2),
        "jumps_count": jumps_count,
    }

def basic_signal_from_series(close: pd.Series) -> Dict[str, Any]:
    df = pd.DataFrame({"Close": close})
    df["sma50"] = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(200).mean()
    df["rsi14"] = compute_rsi(df["Close"])

    df = df.dropna()
    if df.empty:
        raise ValueError("Not enough data for indicators")

    last = df.iloc[-1]

    close_v = float(last["Close"])
    sma50 = float(last["sma50"])
    sma200 = float(last["sma200"])
    rsi14 = float(last["rsi14"])

    if close_v > sma200 and sma50 > sma200:
        trend = "up"
    elif close_v < sma200 and sma50 < sma200:
        trend = "down"
    else:
        trend = "sideways"

    if rsi14 >= 60:
        momentum = "bullish"
    elif rsi14 <= 40:
        momentum = "bearish"
    else:
        momentum = "neutral"

    if trend == "up" and momentum == "bullish":
        signal = "STRONG BUY"
        confidence = 0.9
    elif trend == "up" and momentum == "neutral":
        signal = "BUY"
        confidence = 0.7
    elif trend == "down" and momentum == "bearish":
        signal = "STRONG SELL"
        confidence = 0.9
    elif trend == "down" and momentum == "neutral":
        signal = "SELL"
        confidence = 0.7
    else:
        signal = "HOLD"
        confidence = 0.5

    return {
        "close": round(close_v, 4),
        "sma50": round(sma50, 4),
        "sma200": round(sma200, 4),
        "rsi14": round(rsi14, 2),
        "trend": trend,
        "momentum": momentum,
        "signal": signal,
        "confidence": round(confidence, 2),
    }


# ------------------------------------
# QUANT HELPERS (Quant Lab)
# ------------------------------------

def parse_selected_asset_to_symbol(asset_label: str) -> str:
    # asset_label пример: "NVIDIA (NVDA)" или "BTC (BTCUSDT)" или "(choose)"
    if not asset_label or asset_label == "(choose)":
        return ""
    m = re.search(r"\(([^)]+)\)\s*$", str(asset_label).strip())
    if m:
        return m.group(1).strip()
    return asset_label.strip()

def detect_source_for_symbol(symbol: str, preferred: str = "Auto") -> str:
    # preferred: Auto/Yahoo/Binance
    if preferred and preferred != "Auto":
        return preferred
    # Auto logic:
    if symbol.endswith("USDT"):
        return "Binance"
    return "Yahoo"

def bars_per_year_for_timeframe(source: str, timeframe: str) -> int:
    # rough trading bars/year for annualization
    if source == "Yahoo":
        return 252
    # Binance
    bars_map = {"1d": 252, "4h": 252 * 6, "1h": 252 * 24, "15m": 252 * 24 * 4}
    return int(bars_map.get(timeframe, 252))

def bars_per_day_for_tf(source: str, timeframe: str) -> int:
    if source == "Yahoo":
        return 1
    mpd = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}
    return int(mpd.get(timeframe, 1))

def fetch_close_series_for_quant(symbol: str, source: str, timeframe: str, lookback_days: int) -> pd.Series:
    if source == "Yahoo":
        # range_str подбираме грубо
        if lookback_days <= 365:
            range_str = "1y"
        elif lookback_days <= 730:
            range_str = "2y"
        else:
            range_str = "5y"
        df = fetch_yahoo_history(symbol, range_str=range_str, interval="1d", max_points=lookback_days)
        s = df["close"].dropna().astype(float)
        return s

    # Binance
    mpd = bars_per_day_for_tf("Binance", timeframe)
    bars_needed = int(lookback_days * mpd)
    # public endpoint limit practical: 1000
    limit = int(min(1000, max(200, bars_needed)))
    df = fetch_binance_klines(symbol, interval=timeframe, limit=limit)
    s = df["close"].dropna().astype(float)
    # ако bars_needed > limit, просто ще работим с това което имаме
    return s

def max_drawdown_from_close(close: pd.Series) -> float:
    s = close.dropna().astype(float)
    if len(s) < 5:
        return 0.0
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min())

def hurst_exponent(close: pd.Series, max_lag: int = 20) -> Optional[float]:
    s = close.dropna().astype(float)
    if len(s) < 120:
        return None
    lags = range(2, max_lag + 1)
    tau = []
    for lag in lags:
        diff = s.diff(lag).dropna()
        tau.append(np.sqrt(np.std(diff)))
    if not tau or any(t <= 0 for t in tau):
        return None
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)

def monte_carlo_forward_distribution(
    close: pd.Series,
    horizon_bars: int,
    sims: int = 5000,
) -> Dict[str, Any]:
    s = close.dropna().astype(float)
    if len(s) < 60 or horizon_bars < 1:
        return {}
    r = np.log(s).diff().dropna()
    mu = float(r.mean())
    sigma = float(r.std())
    s0 = float(s.iloc[-1])

    # 1-step aggregated horizon using normal approximation (OK for quant dashboard)
    z = np.random.normal(loc=0.0, scale=1.0, size=sims)
    rh = (mu * horizon_bars) + (sigma * np.sqrt(horizon_bars) * z)
    st = s0 * np.exp(rh)

    p10 = float(np.percentile(st, 10))
    p50 = float(np.percentile(st, 50))
    p90 = float(np.percentile(st, 90))
    expv = float(np.mean(st))

    return {"mc_p10": p10, "mc_p50": p50, "mc_p90": p90, "mc_mean": expv}

def compute_quant_metrics(
    close: pd.Series,
    bars_per_year: int,
    jump_z: float,
    horizon_bars: int,
) -> Dict[str, Any]:
    s = close.dropna().astype(float)
    if len(s) < 80:
        return {"error": "Not enough data for quant metrics."}

    r = np.log(s).diff().dropna()

    # realized vol annualized
    realized_vol = float(r.std() * np.sqrt(float(bars_per_year))) if len(r) > 5 else None

    # skew/kurtosis (pandas)
    skew = float(r.skew()) if len(r) > 10 else None
    kurt = float(r.kurt()) if len(r) > 10 else None

    # VaR/CVaR 95% (on returns)
    var95 = float(np.percentile(r, 5)) if len(r) > 20 else None
    cvar95 = float(r[r <= var95].mean()) if (var95 is not None and (r <= var95).any()) else None

    # drawdown
    mdd = max_drawdown_from_close(s)

    # vol regime (simple)
    regime = None
    if realized_vol is not None:
        if realized_vol < 0.15:
            regime = "LOW_VOL"
        elif realized_vol < 0.30:
            regime = "NORMAL_VOL"
        else:
            regime = "HIGH_VOL"

    # hurst
    h = hurst_exponent(s, max_lag=20)

    # jump-diffusion pack (uses your existing function)
    jm = compute_jump_diffusion_metrics(s, bars_per_year=bars_per_year, jump_z=jump_z)

    # Monte Carlo horizon distribution
    mc = monte_carlo_forward_distribution(s, horizon_bars=horizon_bars, sims=5000)

    out = {
        "last_price": float(s.iloc[-1]),
        "n_bars": int(len(s)),
        "realized_vol_annual": round(realized_vol, 4) if realized_vol is not None else None,
        "skew": round(skew, 4) if skew is not None else None,
        "kurtosis": round(kurt, 4) if kurt is not None else None,
        "VaR_95_logret": round(var95, 6) if var95 is not None else None,
        "CVaR_95_logret": round(cvar95, 6) if cvar95 is not None else None,
        "max_drawdown": round(mdd, 4),
        "hurst": round(h, 4) if h is not None else None,
        "vol_regime": regime,
        **jm,
        **mc,
    }
    return out

def quant_metrics_to_brief(symbol: str, source: str, timeframe: str, lookback_days: int, horizon_label: str, qm: Dict[str, Any]) -> str:
    # brief text for GPT (numbers only)
    keys = [
        "last_price","n_bars",
        "realized_vol_annual","vol_regime",
        "skew","kurtosis","max_drawdown","hurst",
        "lambda_year","avg_jump_pct","jump_vol_pct","sigma_diffusion_pct","jump_risk_score","jumps_count",
        "mc_p10","mc_p50","mc_p90","mc_mean",
        "VaR_95_logret","CVaR_95_logret",
    ]
    lines = [
        f"SYMBOL: {symbol}",
        f"SOURCE: {source}",
        f"TIMEFRAME: {timeframe}",
        f"LOOKBACK_DAYS: {lookback_days}",
        f"HORIZON: {horizon_label}",
        "METRICS:"
    ]
    for k in keys:
        if k in qm:
            lines.append(f"- {k}: {qm.get(k)}")
    return "\n".join(lines)

def run_quant_gpt_analysis(brief: str) -> str:
    client = get_openai_client()
    if client is None:
        return "Quant GPT analysis error: OpenAI client not configured."

    system_prompt = """
You are a quantitative trading analyst.

Rules:
- Use ONLY the numbers provided in the brief. Do not invent prices, indicators, or news.
- Explain what the metrics imply about: volatility regime, jump risk, tail risk, trend vs mean-reversion tendency.
- Provide 3 strategy-style playbooks (rule-based), NOT investment advice and NOT position sizing.
- Be explicit about limitations and what would invalidate the read.
Output structure:
1) Regime Summary
2) Risk Profile (jumps/tails/drawdown)
3) Scenario ranges (use Monte Carlo percentiles)
4) Strategy Playbooks (3)
5) Red flags / invalidation
"""

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": brief},
        ],
        max_completion_tokens=2000,
        temperature=0.25,
    )
    return completion.choices[0].message.content.strip()

# ------------------------------------
# YAHOO PRICE DATA
# ------------------------------------


def fetch_yahoo_history(
    ticker: str, range_str: str = "1y", interval: str = "1d", max_points: int = DAYS_BACK
) -> pd.DataFrame:
    url = YAHOO_CHART_URL.format(ticker)
    params = {"range": range_str, "interval": interval}
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()

    chart = data.get("chart", {})
    results = chart.get("result")
    if not results:
        raise ValueError(f"No chart result for {ticker}: {data}")

    result = results[0]
    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {}).get("quote", [{}])[0]

    closes = indicators.get("close", [])
    opens = indicators.get("open", [])
    highs = indicators.get("high", [])
    lows = indicators.get("low", [])
    volumes = indicators.get("volume", [])

    if not closes or len(closes) < 50:
        raise ValueError(f"Not enough data for {ticker} (len={len(closes)})")

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(timestamps, unit="s"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    df = df.dropna(subset=["close"])
    df = df.tail(max_points)
    df.set_index("time", inplace=True)
    return df


def analyze_yahoo_asset(name: str, ticker: str, asset_class: str) -> Optional[Dict[str, Any]]:
    try:
        df = fetch_yahoo_history(ticker, range_str="1y", interval="1d", max_points=DAYS_BACK)
        sig = basic_signal_from_series(df["close"])

        jm = compute_jump_diffusion_metrics(
            df["close"],
            bars_per_year=252,   # Yahoo 1D
            jump_z=3.0
        )

        return {
            "name": name,
            "ticker": ticker,
            "asset_class": asset_class,
            **sig,
            **jm,
        }
    except Exception:
        return None



def run_analysis_global(selected_classes: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset_class, mapping in ASSETS_BY_CLASS.items():
        if selected_classes and asset_class not in selected_classes:
            continue
        for name, ticker in mapping.items():
            r = analyze_yahoo_asset(name, ticker, asset_class)
            if r:
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[
        [
            "name",
            "ticker",
            "asset_class",
            "signal",
            "confidence",
            "trend",
            "momentum",
            "rsi14",
            "close",
            "sma50",
            "sma200",
            "lambda_year",
            "avg_jump_pct",
            "jump_vol_pct",
            "sigma_diffusion_pct",
            "jump_risk_score",
            "jumps_count",
        ]
    ]
    return df


# ------------------------------------
# BINANCE LAYER
# ------------------------------------

@st.cache_resource(show_spinner=False)
def get_binance_client(api_key: str, api_secret: str):
    try:
        api_key = (api_key or "").strip()
        api_secret = (api_secret or "").strip()

        # Дори без ключове, python-binance пак прави ping() и може да гръмне,
        # затова го пазим в try/except
        if not api_key or not api_secret:
            return None

        return Client(api_key=api_key, api_secret=api_secret)

    except Exception as e:
        st.session_state["binance_client_error"] = str(e)
        return None




@st.cache_data(ttl=30, show_spinner=False)
def fetch_binance_klines(symbol: str, interval: str = "1d", limit: int = 500) -> pd.DataFrame:
    base_urls = [
        "https://data-api.binance.vision",  # <-- най-често работи при 451
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
        "https://api.binance.com",
    ]

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0"}

    last_err = None
    for base in base_urls:
        try:
            url = f"{base}/api/v3/klines"
            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            klines = r.json()
            if not klines:
                raise ValueError(f"No klines for {symbol} from {base}")

            df = pd.DataFrame(
                klines,
                columns=[
                    "open_time","open","high","low","close","volume","close_time",
                    "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
                ],
            )
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"])
            return df

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All Binance endpoints failed for {symbol}: {last_err}")



def run_analysis_binance(timeframe: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    bars_map = {"1d": 252, "4h": 252 * 6, "1h": 252 * 24, "15m": 252 * 24 * 4}

    for symbol, meta in BINANCE_SYMBOLS.items():
        try:
            df = fetch_binance_klines(symbol, interval=timeframe, limit=500)
            sig = basic_signal_from_series(df["close"])

            bpy = bars_map.get(timeframe, 252)

            jm = compute_jump_diffusion_metrics(
                df["close"],
                bars_per_year=bpy,
                jump_z=3.0
            )

            row = {
                "symbol": symbol,
                "name": meta["display"],
                "asset_class": meta["class"],
                "timeframe": timeframe,
                **sig,
                **jm,
            }
            rows.append(row)
        except Exception as e:
            errors.append(f"{symbol} ({timeframe}): {type(e).__name__}: {e}")

    if errors:
        st.warning("Some Binance symbols failed:\n" + "\n".join(errors))

    return pd.DataFrame(rows)



# ------------------------------------
# NEWS (NewsAPI) + HISTORY
# ------------------------------------


def fetch_news_for_keyword(keyword: str, page_size: int = 5) -> List[Dict[str, Any]]:
    if not NEWSAPI_KEY:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    articles = data.get("articles", [])
    cleaned: List[Dict[str, Any]] = []
    for a in articles:
        cleaned.append(
            {
                "keyword": keyword,
                "source": (a.get("source") or {}).get("name", ""),
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            }
        )
    return cleaned


def load_news_history_df() -> pd.DataFrame:
    if not os.path.exists(NEWS_HISTORY_FILE):
        return pd.DataFrame(
            columns=["keyword", "source", "title", "description", "url", "published_at"]
        )
    df = pd.read_csv(NEWS_HISTORY_FILE)
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df


def save_news_history_df(df: pd.DataFrame) -> None:
    df.to_csv(NEWS_HISTORY_FILE, index=False)


def update_news_history(new_items: List[Dict[str, Any]]) -> pd.DataFrame:
    history = load_news_history_df()
    df_new = pd.DataFrame(new_items)

    if not df_new.empty and "published_at" in df_new.columns:
        df_new["published_at"] = pd.to_datetime(df_new["published_at"], errors="coerce")

    df_all = pd.concat([history, df_new], ignore_index=True)

    if not df_all.empty:
        if "published_at" in df_all.columns:
            df_all["published_at"] = pd.to_datetime(df_all["published_at"], errors="coerce")
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=NEWS_RETENTION_DAYS)

            df_all = df_all[
                df_all["published_at"].isna() | (df_all["published_at"] >= cutoff)
            ]

        df_all = df_all.drop_duplicates(subset=["url"], keep="last")

    save_news_history_df(df_all)
    return df_all


def aggregate_news(keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Дърпа новини от NewsAPI. Ако ударим лимит (429) или има друга грешка,
    връщаме fallback от локалната история (news_history.csv).
    """
    all_news: List[Dict[str, Any]] = []

    try:
        for kw in keywords:
            items = fetch_news_for_keyword(kw, page_size=3)
            all_news.extend(items)

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            st.error(
                "Error fetching news: 429 Too Many Requests (NewsAPI rate limit). "
                "Using saved news history instead."
            )
        else:
            st.error(f"Error fetching news: {e}")

        hist = load_news_history_df()
        if hist.empty:
            return []
        hist_sorted = hist.sort_values("published_at", ascending=False)
        return hist_sorted.to_dict("records")

    except Exception as e:
        st.error(f"Error fetching news: {e}")
        hist = load_news_history_df()
        if hist.empty:
            return []
        hist_sorted = hist.sort_values("published_at", ascending=False)
        return hist_sorted.to_dict("records")

    if not all_news:
        hist = load_news_history_df()
        if hist.empty:
            return []
        hist_sorted = hist.sort_values("published_at", ascending=False)
        return hist_sorted.to_dict("records")

    all_news_sorted = sorted(all_news, key=lambda x: x.get("published_at", ""), reverse=True)
    update_news_history(all_news_sorted)
    return all_news_sorted


def get_relevant_news_for_asset(focus_asset: str, max_items: int = 40) -> List[Dict[str, Any]]:
    hist = load_news_history_df()
    if hist.empty:
        return []

    if not focus_asset or focus_asset == "Global macro view":
        df_rel = hist.sort_values("published_at", ascending=False).head(max_items)
        return df_rel.to_dict("records")

    core_name = focus_asset.split("(")[0].strip()

    mask = (
        hist["title"].fillna("").str.contains(core_name, case=False)
        | hist["description"].fillna("").str.contains(core_name, case=False)
        | hist["keyword"].fillna("").str.contains(core_name, case=False)
    )
    df_rel = hist[mask].sort_values("published_at", ascending=False).head(max_items)

    if df_rel.empty:
        df_rel = hist.sort_values("published_at", ascending=False).head(max_items)

    return df_rel.to_dict("records")

# ------------------------------------
# OPENAI CLIENT + AI ANALYST
# ------------------------------------


@st.cache_resource(show_spinner=False)
def get_openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


def df_to_brief(df: pd.DataFrame, label: str) -> str:
    """
    Тук вече подаваме и реалните close / SMA50 / SMA200,
    за да може анализаторът да работи с истински цени.
    """
    if df is None or df.empty:
        return f"No {label} signals available."
    df_local = df.copy()
    df_local = df_local.sort_values("confidence", ascending=False).head(10)
    cols = [
        c
        for c in df_local.columns
        if c
        in [
    "name",
    "ticker",
    "symbol",
    "asset_class",
    "timeframe",
    "signal",
    "confidence",
    "trend",
    "momentum",
    "rsi14",
    "close",
    "sma50",
    "sma200",
    "lambda_year",
    "avg_jump_pct",
    "jump_vol_pct",
    "sigma_diffusion_pct",
    "jump_risk_score",
    "jumps_count",
]

    ]
    return df_local[cols].to_string(index=False)


def build_ai_context(
    df_global: pd.DataFrame,
    df_crypto: pd.DataFrame,
    news_items: List[Dict[str, Any]],
) -> str:
    global_text = df_to_brief(df_global, "global")
    crypto_text = df_to_brief(df_crypto, "crypto")

    if news_items:
        top_news = news_items[:10]
        news_lines = [
            f"- [{n.get('keyword','')}] {n.get('title','')} (source: {n.get('source','')})"
            for n in top_news
        ]
        news_text = "\n".join(news_lines)
    else:
        news_text = "No news loaded."

    ctx = f"""
GLOBAL SIGNALS (top 10):
{global_text}

CRYPTO SIGNALS (top 10):
{crypto_text}

LATEST NEWS (top headlines):
{news_text}
"""
    return ctx.strip()


def run_ai_analyst(df_global, df_crypto, news_items, target_asset, horizon, user_question):
    try:
        client = get_openai_client()
        if client is None:
            return "AI analysis error: OpenAI client is not configured (missing OPENAI_API_KEY)."

        base_ctx = build_ai_context(
            df_global if df_global is not None else pd.DataFrame(),
            df_crypto if df_crypto is not None else pd.DataFrame(),
            news_items or [],
        )

        focus_block = f"""
FOCUS:
- TARGET ASSET: {target_asset or "none (give a global perspective)"}
- TIME HORIZON: {horizon}
- USER QUESTION: {user_question}
"""

        system_prompt = """
You are an institutional-grade macro–financial and technical analyst.
You analyse cross-asset signals (equities, crypto, indices, commodities, FX)
plus news flow and generate structured, actionable insights.

Constraints:
- Do NOT give investment advice or position sizing.
- Speak in probabilities and scenarios, never certainties.
- Explicitly separate short-term (days/weeks) and medium-term (months) views.
- Use the supplied data, do not hallucinate specific prices or indicators not present in the context.
- You can infer relations (e.g. strong USD usually pressures gold and crypto), but mark these as "typical behaviour".
"""

        context = base_ctx + "\n\n" + focus_block

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            max_completion_tokens=5500,
            temperature=0.4,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"AI analysis error: {e}"


def run_news_forecast(
    df_global, df_crypto, latest_news_items: List[Dict[str, Any]], focus_asset: str
):
    try:
        client = get_openai_client()
        if client is None:
            return "AI news-forecast error: OpenAI client is not configured (missing OPENAI_API_KEY)."

        history_items = get_relevant_news_for_asset(focus_asset)
        effective_news = history_items or latest_news_items or []

        base_ctx = build_ai_context(
            df_global if df_global is not None else pd.DataFrame(),
            df_crypto if df_crypto is not None else pd.DataFrame(),
            effective_news,
        )

        focus_name = focus_asset or "Global macro view"

        user_block = f"""
You are analysing the asset: {focus_name}.

Use the technical signals and especially the NEWS CONTEXT above (both recent and historical headlines)
to produce a DETAILED, NEWS-DRIVEN FORECAST:

For {focus_name}, provide:

1. CURRENT NEWS IMPACT
   - Is the overall newsflow BULLISH / BEARISH / MIXED / UNCLEAR? Why?

2. SHORT-TERM VIEW (next days / 1-2 weeks)
   - Directional bias (up / down / range).
   - Key triggers that could move the price (events, data, company catalysts).

3. MEDIUM-TERM VIEW (1-3 months)
   - Main scenarios (bull / base / bear) with probabilities.
   - What kind of news would confirm each scenario?

4. STRUCTURAL / LONGER-TERM POINTS (if any)
   - Important themes from older headlines that are still relevant.

5. RISKS & WATCHLIST
   - 3-5 concrete risks or "if X happens, reassess" bulletpoints.

6. TRADING / INVESTMENT TAKEAWAYS
   - How a swing trader or position trader could use this information in practice
     (without giving specific financial advice or position size).

Write in rich detail, using bulletpoints and short paragraphs.
"""

        system_prompt = """
You are a macro/news-driven trading analyst.
Your goal is to translate headline flows into directional views and scenarios
for one specific asset at a time.
"""

        context = base_ctx + "\n\n" + user_block

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            max_completion_tokens=2200,
            temperature=0.4,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"AI news-forecast error: {e}"

# ------------------------------------
# FOMC FETCH HELPERS (автоматично дърпане от fed.gov)
# ------------------------------------


def strip_html_tags(html_text: str) -> str:
    """Махаме HTML тагове и оставяме чист текст."""
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return ihtml.unescape(text).strip()


def fetch_fomc_statement_text(url: str) -> str:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    html_text = resp.text
    return strip_html_tags(html_text)


def get_fomc_pressconf_text_from_page(statement_url: str):
    """
    Опитва да намери линк към 'Press Conference' на страницата на FOMC
    изявлението. Ако намери HTML страница (не PDF), връща изчистен текст.
    """
    meta: Dict[str, Any] = {}
    try:
        resp = requests.get(statement_url, timeout=20)
        resp.raise_for_status()
        html_text = resp.text
    except Exception as e:
        meta["pressconf_error"] = f"Error fetching statement page for pressconf scan: {e}"
        meta["pressconf_source"] = statement_url
        return "", meta

    links = re.findall(
        r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html_text,
        flags=re.I | re.S,
    )

    press_url = ""
    for href, label in links:
        label_clean = strip_html_tags(label).lower()
        if "press conference" in label_clean:
            press_url = href
            break

    if not press_url:
        meta["pressconf_error"] = "No press conference link found on statement page."
        meta["pressconf_source"] = statement_url
        return "", meta

    if press_url.startswith("http"):
        full_url = press_url
    elif press_url.startswith("/"):
        full_url = FED_BASE + press_url
    else:
        full_url = FED_BASE + "/" + press_url.lstrip("/")

    meta["pressconf_url"] = full_url

    # ако е PDF – не можем лесно да извадим текста
    if full_url.lower().endswith(".pdf"):
        meta["pressconf_error"] = "Press conference link is PDF – text extraction not supported."
        return "", meta

    try:
        resp2 = requests.get(full_url, timeout=20)
        resp2.raise_for_status()
        press_text = strip_html_tags(resp2.text)
        return press_text, meta
    except Exception as e:
        meta["pressconf_error"] = f"Error fetching press conference page: {e}"
        return "", meta

def extract_pressconf_excerpts(press_text: str, max_items: int = 8) -> List[str]:
    """
    Extracts clean, trusted excerpts from FOMC press conference text.
    Logic:
    - Only real text from fed.gov
    - Filters very short / navigation junk
    - Keeps meaningful Q&A-style sentences
    """

    if not press_text or len(press_text) < 500:
        return []

    # Split by sentence-like chunks
    raw_parts = re.split(r'(?<=[\.\?\!])\s+', press_text)

    excerpts = []
    for part in raw_parts:
        p = part.strip()

        # basic filters
        if len(p) < 120:
            continue
        if any(x in p.lower() for x in [
            "federal reserve",
            "board of governors",
            "subscribe",
            "copyright",
            "home page",
            "press release",
        ]):
            continue

        excerpts.append(p)

        if len(excerpts) >= max_items:
            break

    return excerpts



def get_latest_fomc_statements(year: Optional[int] = None):
    """
    Връща (current_text, previous_text, press_text, meta_dict) за последните две
    FOMC statements. Използва URL pattern за monetaryYYYYMMDDx.htm.
    """
    base_year = year or dt.datetime.utcnow().year
    last_error = None
    html_index = None
    used_index_url = None
    used_year = None

    # 1) Опитваме текущата година, после предишната
    for y in [base_year, base_year - 1]:
        candidate_paths = [
            f"/newsevents/pressreleases/{y}-press-fomc.htm",
            f"/newsevents/pressreleases/{y}-press.htm",
        ]
        for path in candidate_paths:
            index_url = FED_BASE + path
            try:
                resp = requests.get(index_url, timeout=20)
                if resp.status_code == 200:
                    html_index = resp.text
                    used_index_url = index_url
                    used_year = y
                    break
            except Exception as e:
                last_error = f"Error fetching {index_url}: {e}"
        if html_index:
            break

    if not html_index:
        return "", "", "", {
            "error": last_error or "Could not fetch FOMC index page.",
            "index_url": used_index_url or "",
        }

    # 2) Търсим всички monetary линкове
    pattern = r'href="(/newsevents/pressreleases/monetary(\d{8})[a-z]\.htm)"'
    matches = re.findall(pattern, html_index)

    if not matches:
        return "", "", "", {
            "error": "No FOMC statement links found on index page.",
            "index_url": used_index_url,
        }

    # 3) Сортираме по дата (YYYYMMDD) и взимаме последните две
    matches_sorted = sorted(matches, key=lambda x: x[1])
    paths = [m[0] for m in matches_sorted]
    dates = [m[1] for m in matches_sorted]

    current_path, current_date = paths[-1], dates[-1]
    prev_path, prev_date = (paths[-2], dates[-2]) if len(paths) > 1 else ("", "")

    current_url = FED_BASE + current_path
    prev_url = FED_BASE + prev_path if prev_path else ""

    try:
        current_text = fetch_fomc_statement_text(current_url)
    except Exception as e:
        return "", "", "", {
            "error": f"Error fetching current statement: {e}",
            "index_url": used_index_url,
            "current_url": current_url,
        }

    previous_text = ""
    if prev_url:
        try:
            previous_text = fetch_fomc_statement_text(prev_url)
        except Exception as e:
            previous_text = ""
            last_error = f"Error fetching previous statement: {e}"

    # 4) Опитваме да извадим пресконференцията от страницата на текущото изявление
    press_text, press_meta = get_fomc_pressconf_text_from_page(current_url)

    meta: Dict[str, Any] = {
        "index_url": used_index_url,
        "index_year": used_year,
        "current_url": current_url,
        "current_date": current_date,
        "previous_url": prev_url,
        "previous_date": prev_date,
    }
    if last_error:
        meta["warning"] = last_error
    meta.update(press_meta)

    return current_text, previous_text, press_text, meta

# ------------------------------------
# FOMC ANALYZER (GPT-5.1)
# ------------------------------------

def analyze_fomc_with_gpt(
    current_text: str,
    previous_text: str = "",
    pressconf_text: str = "",
) -> Dict[str, Any]:
    """
    Анализира FOMC statement + пресконференция.
    Връща САМО валиден JSON dict със score, тон, ключови промени и трейдинг implications.
    """
    client = get_openai_client()
    if client is None:
        return {"error": "OpenAI client is not configured (missing OPENAI_API_KEY)."}

    system_msg = """
You are an expert macro and FOMC analyst.

Goal:
- Read the CURRENT FOMC statement and (optionally) the PREVIOUS statement and PRESS CONFERENCE excerpts.
- Produce a trading-oriented macro interpretation: hawkish/dovish evaluation, key changes, and market implications.
- Opinions and probabilistic interpretations ARE allowed, but do NOT invent facts or quotes.

Rules:
- Return ONLY valid JSON. No markdown, no comments, no extra text.
- Base factual claims ONLY on the provided text.
- Use probability-based language, not certainty.

Allowed values:
- tone_change: more_hawkish, more_dovish, similar
- trade_bias: risk_on, risk_off, mixed

Numeric constraints:
- hawk_dove_score: -5 to +5 (can be decimal)
- focus fields: integers 0–10

Output MUST strictly follow this JSON structure:
{
  "hawk_dove_score": 0,
  "tone_change": "similar",
  "key_changes": [],
  "inflation_focus": 0,
  "labor_market_focus": 0,
  "growth_risk_focus": 0,
  "financial_stability_focus": 0,
  "summary": "",
  "trade_bias": "mixed",
  "playbook": {
    "before_event": "",
    "first_15min": "",
    "next_24h": ""
  }
}
"""

    user_msg = f"""
CURRENT FOMC STATEMENT:
{current_text}

PREVIOUS FOMC STATEMENT (may be empty):
{previous_text}

PRESS CONFERENCE EXCERPTS (may be empty):
{pressconf_text}

Instructions:
- key_changes: max 8 bullet items, focus on wording changes/emphasis.
- summary: 3-6 sentences, what changed + what it implies.
- playbook: mention volatility + rates/USD/risk-assets bias + tactical behavior.
"""

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,  # ако искаш gpt-5.1, направи OPENAI_MODEL="gpt-5.1"
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_completion_tokens=1200,
    )

    msg = completion.choices[0].message
    raw = msg.content
    refusal = getattr(msg, "refusal", None)

    if refusal:
        return {"error": "Model refusal / blocked output", "refusal": refusal}

    if raw is None or not str(raw).strip():
        try:
            debug = msg.model_dump()
        except Exception:
            debug = str(msg)
        return {"error": "Empty response from OpenAI model", "debug_message": debug}

    raw = str(raw)

    # Ако по някаква причина raw вече е dict (рядко), върни го директно
    if isinstance(raw, dict):
        return raw

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"error": "JSON parsing failed", "raw_response": raw}

    return data


# ------------------------------------
# FOMC PRESS CONFERENCE — LEVEL 2 (Topics + Market Read)
# ------------------------------------

def extract_fomc_pressconf_topics(press_text: str) -> Dict[str, Any]:
    """
    LEVEL 2:
    Extracts WHAT was discussed in the FOMC press conference + interpretive market read.
    (Позволяваме мнение и вероятностни реакции, без да халюцинира факти.)
    """
    client = get_openai_client()
    if client is None or not press_text.strip():
        return {"error": "No press conference text available or OpenAI not configured."}

    system_prompt = """
You are a Federal Reserve press conference macro analyst.

Goal:
- Extract the main topics discussed in the press conference.
- Provide a market-oriented interpretation of the tone and implications.

Hard rules:
- Return ONLY valid JSON (no markdown, no extra text).
- Do NOT invent facts, questions, or quotes not present in the provided text.
- You MAY interpret tone and likely market reaction in probabilistic language.

Allowed values:
- stance: hawkish, dovish, neutral
- overall_tone: hawkish, dovish, neutral, mixed
- trade_bias: risk_on, risk_off, mixed

Output MUST strictly follow this JSON structure:
{
  "event": "FOMC Press Conference",
  "topics": [
    {
      "topic": "",
      "summary": "",
      "stance": "neutral",
      "market_take": ""
    }
  ],
  "overall_tone": "mixed",
  "implied_change_vs_previous": "",
  "trade_bias": "mixed",
  "scenarios": [
    { "name": "Base case", "probability": 60, "description": "" },
    { "name": "Alt case", "probability": 25, "description": "" },
    { "name": "Risk case", "probability": 15, "description": "" }
  ]
}
"""

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": press_text},
        ],
        temperature=0.2,
        max_completion_tokens=1200,
    )

    msg = completion.choices[0].message
    raw = msg.content
    refusal = getattr(msg, "refusal", None)

    if refusal:
        return {"error": "Model refusal / blocked output", "refusal": refusal}

    if raw is None or not str(raw).strip():
        try:
            debug = msg.model_dump()
        except Exception:
            debug = str(msg)
        return {"error": "Empty response from OpenAI model", "debug_message": debug}

    raw = str(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"error": "JSON parsing failed", "raw_response": raw}

    return data




# ------------------------------------
# FOMC LAB UI
# ------------------------------------


def init_fomc_state():
    if "fomc_current" not in st.session_state:
        st.session_state["fomc_current"] = ""
    if "fomc_previous" not in st.session_state:
        st.session_state["fomc_previous"] = ""
    if "fomc_press" not in st.session_state:
        st.session_state["fomc_press"] = ""


def show_fomc_lab():
    """
    Streamlit UI за FOMC анализ – стабилна версия
    """

    # ---------------- STATE INIT ----------------
    if "fomc_current" not in st.session_state:
        st.session_state["fomc_current"] = ""
    if "fomc_previous" not in st.session_state:
        st.session_state["fomc_previous"] = ""
    if "fomc_press" not in st.session_state:
        st.session_state["fomc_press"] = ""
    if "fomc_meta" not in st.session_state:
        st.session_state["fomc_meta"] = {}

    st.title("🏛 FOMC Lab — Speech & Macro Analyzer")

    st.markdown(
        "Автоматично зареждане на FOMC statement, предишно изявление и пресконференция директно от Fed.gov."
    )

    # ---------------- BUTTONS ----------------
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        load_clicked = st.button("📥 Load latest FOMC statements from Fed.gov")

    with col_btn2:
        analyze_clicked = st.button("🔍 Analyze FOMC", type="primary")

    # ---------------- LOAD FROM FED ----------------
    if load_clicked:
        with st.spinner("Зареждам последните FOMC данни от Fed.gov..."):
            cur_text, prev_text, press_text, meta = get_latest_fomc_statements()

        if cur_text:
            st.session_state["fomc_current"] = cur_text
        if prev_text:
            st.session_state["fomc_previous"] = prev_text
        if press_text:
            st.session_state["fomc_press"] = press_text

        st.session_state["fomc_meta"] = meta or {}

        if meta.get("error"):
            st.error(meta["error"])
        else:
            st.success(
                f"Loaded FOMC: {meta.get('current_date','?')} "
                f"| Previous: {meta.get('previous_date','?')}"
            )

            st.caption(f"Statement URL: {meta.get('current_url','')}")
            if meta.get("pressconf_url"):
                st.caption(f"Press Conference URL: {meta.get('pressconf_url')}")
            if meta.get("pressconf_error"):
                st.warning(meta.get("pressconf_error"))

    # ---------------- AUTO PRESS CONF (BEFORE WIDGETS) ----------------
    meta = st.session_state.get("fomc_meta", {})

    if not st.session_state["fomc_press"].strip():
        auto_pressconf = ""

        # 1️⃣ FED.GOV (ако вече е извлечено)
        if meta.get("pressconf_url"):
            auto_pressconf = st.session_state.get("fomc_press", "")

        if auto_pressconf:
            st.session_state["fomc_press"] = auto_pressconf

    # ---------------- TEXT AREAS ----------------
    col1, col2 = st.columns(2)

    with col1:
        current_text = st.text_area(
            "Текущ FOMC Statement (задължително)",
            height=260,
            key="fomc_current",
        )

    with col2:
        previous_text = st.text_area(
            "Предишно FOMC Statement (по желание)",
            height=260,
            key="fomc_previous",
        )

    pressconf_text = st.text_area(
        "Извадки от пресконференцията (по желание)",
        height=180,
        key="fomc_press",
    )

    # ---------------- ANALYZE ----------------
    if analyze_clicked:
        if not current_text.strip():
            st.error("Текущият FOMC statement е задължителен.")
            return

        with st.spinner("Анализирам FOMC текста с GPT..."):
            result = analyze_fomc_with_gpt(
                current_text=current_text,
                previous_text=previous_text,
                pressconf_text=pressconf_text,
            )

        if "error" in result:
            st.error(result.get("error"))
            if "raw_response" in result:
                with st.expander("Raw response"):
                    st.text(result["raw_response"])
            return

        # ---------- RESULTS ----------
        st.subheader("Macro Scoreboard")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Hawk / Dove Score", result.get("hawk_dove_score"))
        with c2:
            st.metric("Tone Change", result.get("tone_change"))
        with c3:
            st.metric("Trade Bias", result.get("trade_bias"))

        st.subheader("Focus by Topic (0–10)")
        c4, c5, c6, c7 = st.columns(4)
        c4.metric("Inflation", result.get("inflation_focus"))
        c5.metric("Labor", result.get("labor_market_focus"))
        c6.metric("Growth", result.get("growth_risk_focus"))
        c7.metric("Stability", result.get("financial_stability_focus"))

        st.subheader("Key Changes")
        for k in result.get("key_changes", []):
            st.markdown(f"- {k}")

        st.subheader("Summary")
        st.write(result.get("summary", ""))

        st.subheader("Trading Playbook")
        pb = result.get("playbook", {})
        st.markdown(f"**Before:** {pb.get('before_event','')}")
        st.markdown(f"**First 15m:** {pb.get('first_15min','')}")
        st.markdown(f"**Next 24h:** {pb.get('next_24h','')}")

        # ---------- LEVEL 2 ----------
        st.markdown("---")
        st.subheader("🧠 FOMC Press Conference — Key Topics (Level 2)")

        if pressconf_text.strip():
            with st.spinner("Extracting key topics..."):
                lvl2 = extract_fomc_pressconf_topics(pressconf_text)

            if "error" in lvl2:
                st.warning(lvl2.get("error"))
            else:
                for t in lvl2.get("topics", []):
                    st.markdown(
                        f"- **{t['topic']}** → {t['summary']} (_{t['stance']}_)"
                    )

                st.markdown(f"**Overall tone:** `{lvl2.get('overall_tone')}`")
                st.markdown(
                    f"**Change vs previous:** {lvl2.get('implied_change_vs_previous')}"
                )

                with st.expander("Raw Level 2 JSON"):
                    st.json(lvl2)
        else:
            st.info("Няма наличен текст от пресконференция.")

        with st.expander("Raw JSON result"):
            st.json(result)


# ------------------------------------
# STREAMLIT UI
# ------------------------------------

st.title("AI Macro Agent — Multi-Asset Dashboard + AI Analyst")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_yahoo_live_quote(symbol: str) -> Dict[str, float]:
    url = YAHOO_CHART_URL.format(symbol)
    params = {"range": "1d", "interval": "1m"}
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()

    result = (data.get("chart", {}) or {}).get("result") or []
    if not result:
        raise ValueError(f"No Yahoo chart result for {symbol}. chart.error={data.get('chart', {}).get('error')}")

    res0 = result[0]
    meta = res0.get("meta", {}) or {}

    # 1) опит: meta regularMarketPrice / previousClose
    last = meta.get("regularMarketPrice", None)
    prev_close = meta.get("previousClose", None)

    # 2) fallback: вземи последния не-None close от indicators
    if last is None or prev_close is None:
        quotes = (res0.get("indicators", {}) or {}).get("quote", []) or []
        closes = (quotes[0] or {}).get("close", []) if quotes else []
        closes_clean = [c for c in closes if c is not None]

        if last is None and closes_clean:
            last = closes_clean[-1]
        if prev_close is None and len(closes_clean) >= 2:
            prev_close = closes_clean[-2]

    if last is None or prev_close is None:
        raise ValueError(f"Yahoo meta missing prices for {symbol}. meta keys={list(meta.keys())[:20]}")

    last = float(last)
    prev_close = float(prev_close)

    pct = ((last - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
    return {"last": last, "pct": pct}



@st.cache_data(ttl=3, show_spinner=False)
def fetch_binance_24h_quote(symbol: str) -> Dict[str, float]:
    base_urls = [
        "https://data-api.binance.vision",
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
    ]
    params = {"symbol": symbol}
    headers = {"User-Agent": "Mozilla/5.0"}

    last_err = None
    for base in base_urls:
        try:
            url = f"{base}/api/v3/ticker/24hr"
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            j = r.json()
            return {
                "last": float(j.get("lastPrice", 0.0)),
                "pct": float(j.get("priceChangePercent", 0.0)),
            }
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Binance 24h quote failed for {symbol}: {last_err}")

# ===== LIVE TICKER HORIZONTAL (CRYPTO, YAHOO STYLE) =====

live_ticker_css = """
<style>
.live-ticker-container {
    position: relative;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
    padding: 0.25rem 0;
    overflow: hidden;
    background-color: #000000;
    color: #ffffff;
}
.live-ticker-row {
    display: flex;
    gap: 0.75rem;
    align-items: stretch;
    overflow-x: auto;
    scroll-behavior: smooth;
    scrollbar-width: none;
    padding: 0 42px; /* място за стрелките */
}
.live-ticker-row::-webkit-scrollbar { display: none; }

.ticker-item {
    min-width: 170px;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    border: 1px solid #00ff00;
    background-color: #000000;
    display: flex;
    flex-direction: column;
    justify-content: center;
    font-size: 0.85rem;
    color: #ffffff;
}

.ticker-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.15rem;
}
.ticker-symbol { font-weight: 700; }
.ticker-source { opacity: 0.7; font-size: 0.7rem; }

.ticker-price-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}
.ticker-price {
    font-family: monospace;
    font-size: 0.95rem;
}

.ticker-change { font-size: 0.8rem; }
.ticker-change.up { color: #00ff00; }
.ticker-change.down { color: #ff4d4d; }

.ticker-arrow {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 34px;
    height: 62px;
    border-radius: 6px;
    border: 1px solid #555555;
    background-color: #111111;
    color: #cccccc;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 10;
    user-select: none;
}
.ticker-arrow.left { left: 6px; }
.ticker-arrow.right { right: 6px; }
.ticker-arrow:hover { background-color: #222222; }
</style>
"""

# 1) Дърпаме Yahoo live само за металите
# --- LIVE PRICE MAPS (INITIAL VALUES) ---

yahoo_live_map: Dict[str, Dict[str, float]] = {}

for sym, _ in LIVE_TICKER_SYMBOLS:
    if sym.endswith("USDT"):
        continue  # това е Binance
    try:
        yahoo_live_map[sym] = fetch_yahoo_live_quote(sym)
    except Exception as e:
        yahoo_live_map[sym] = {"last": float("nan"), "pct": float("nan")}
        st.session_state["yahoo_live_errors"][sym] = str(e)



binance_live_map = {}
for sym, _ in LIVE_TICKER_SYMBOLS:
    if not sym.endswith("=X"):
        try:
            binance_live_map[sym] = fetch_binance_24h_quote(sym)
        except Exception:
            binance_live_map[sym] = {"last": float("nan"), "pct": float("nan")}


ticker_items_html = []
for sym, short in LIVE_TICKER_SYMBOLS:
    source = "Binance" if sym.endswith("USDT") else "Yahoo"


    # initial values (само за Yahoo; Binance ще се обновява от JS)
    initial_last = "..."
    initial_pct = "..."
    initial_class = ""

    if source == "Yahoo":
        q = yahoo_live_map.get(sym, {})
    else:
        q = binance_live_map.get(sym, {})

    last = q.get("last")
    pct = q.get("pct")

    if isinstance(last, (int, float)) and last == last:
        initial_last = f"{last:.4f}"
    if isinstance(pct, (int, float)) and pct == pct:
        initial_pct = f"{pct:.2f}%"
        initial_class = "up" if pct >= 0 else "down"


    item_html = f"""
<div class="ticker-item" data-symbol="{sym}" data-source="{source}">
  <div class="ticker-header">
    <div class="ticker-symbol">{short}</div>
    <div class="ticker-source">{source}</div>
  </div>
  <div class="ticker-price-row">
    <div class="ticker-price" data-symbol="{sym}" data-field="last">{initial_last}</div>
    <div class="ticker-change {initial_class}" data-symbol="{sym}" data-field="chgClass">
      <span data-symbol="{sym}" data-field="changePct">{initial_pct}</span>
    </div>
  </div>
</div>
"""
    ticker_items_html.append(item_html)


symbols_js = [sym for sym, _ in LIVE_TICKER_SYMBOLS]

live_ticker_html = live_ticker_css + textwrap.dedent(f"""
<div class="live-ticker-container">
  <button type="button" class="ticker-arrow left" onclick="scrollTicker(-1)">&#9664;</button>
  <div class="live-ticker-row" id="live-ticker-row">
    {''.join(ticker_items_html)}
  </div>
  <button type="button" class="ticker-arrow right" onclick="scrollTicker(1)">&#9654;</button>
</div>

<script>
(function () {{
  const SYMBOLS = {json.dumps(symbols_js)};
  const ROW_ID = "live-ticker-row";
  const STORAGE_KEY = "ticker_scroll_left_v1";

  function fmtPrice(x) {{
    if (!isFinite(x)) return "...";
    if (x >= 1000) return x.toLocaleString(undefined, {{ maximumFractionDigits: 2 }});
    if (x >= 1) return x.toLocaleString(undefined, {{ maximumFractionDigits: 4 }});
    return x.toLocaleString(undefined, {{ maximumFractionDigits: 8 }});
  }}

  window.scrollTicker = function (direction) {{
    const row = document.getElementById(ROW_ID);
    if (!row) return;
    const item = row.querySelector(".ticker-item");
    const step = item ? (item.offsetWidth + 12) : 180;
    row.scrollBy({{ left: direction * step, behavior: "smooth" }});
  }};

  function saveScroll() {{
    const row = document.getElementById(ROW_ID);
    if (!row) return;
    try {{ localStorage.setItem(STORAGE_KEY, String(row.scrollLeft)); }} catch (e) {{}}
  }}

  function restoreScroll() {{
    const row = document.getElementById(ROW_ID);
    if (!row) return;
    try {{
      const v = localStorage.getItem(STORAGE_KEY);
      if (v !== null) row.scrollLeft = Number(v) || 0;
    }} catch (e) {{}}
  }}

  async function fetchAll24h() {{
    const urls = [
      "https://api.binance.com/api/v3/ticker/24hr",
      "https://data-api.binance.vision/api/v3/ticker/24hr"
    ];
    let lastErr = null;

    for (const url of urls) {{
      try {{
        const r = await fetch(url, {{ cache: "no-store" }});
        if (!r.ok) throw new Error("HTTP " + r.status);
        return await r.json();
      }} catch (e) {{
        lastErr = e;
      }}
    }}
    throw lastErr;
  }}

  function updateDom(map) {{
    for (const sym of SYMBOLS) {{
      if (!sym.endsWith("USDT")) continue; // само Binance се обновява от JS

      const data = map.get(sym);
      if (!data) continue;

      const last = Number(data.lastPrice);
      const pct = Number(data.priceChangePercent);

      const lastEl = document.querySelector(`[data-symbol="${sym}"][data-field="last"]`);
      const pctEl  = document.querySelector(`[data-symbol="${sym}"][data-field="changePct"]`);
      const chgEl  = document.querySelector(`[data-symbol="${sym}"][data-field="chgClass"]`);

      if (lastEl) lastEl.textContent = fmtPrice(last);
      if (pctEl) pctEl.textContent = (isFinite(pct) ? pct.toFixed(2) : "...") + "%";

      if (chgEl) {{
        chgEl.classList.remove("up", "down");
        if (isFinite(pct)) chgEl.classList.add(pct >= 0 ? "up" : "down");
      }}
    }}
  }}

  async function tick() {{
    try {{
      const all = await fetchAll24h();
      const map = new Map();
      for (const item of all) {{
        if (item && item.symbol) map.set(item.symbol, item);
      }}
      updateDom(map);
    }} catch (e) {{
      // ignore
    }}
  }}

  function init() {{
    const row = document.getElementById(ROW_ID);
    if (row) {{
      row.addEventListener("scroll", saveScroll, {{ passive: true }});
      restoreScroll();
    }}
    tick();
    setInterval(tick, 3000);
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", init);
  }} else {{
    init();
  }}
}})();
</script>
""")

components.html(live_ticker_html, height=120, scrolling=False)
with st.expander("Yahoo Live Debug"):
    st.write(st.session_state.get("yahoo_live_errors", {}))



# ===== REST OF HEADER =====

now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
st.caption(f"Last update time (UTC): {now}")
st.markdown("---")

tab_global, tab_crypto, tab_news, tab_quant, tab_ai, tab_fomc = st.tabs(
    ["🌍 Global Signals", "🪙 Crypto (Binance)", "📰 News & Macro", "🧮 Quant Lab", "🤖 AI Analyst", "🏛 FOMC Lab"]

)

# -------- GLOBAL TAB --------
with tab_global:
    st.subheader("Global Signals — Yahoo Finance (1D, ~1y history)")

    all_classes = list(ASSETS_BY_CLASS.keys())
    selected_classes = st.multiselect(
        "Asset classes to show:",
        options=all_classes,
        default=all_classes,
    )

    st.write("Data source: Yahoo Finance chart API.")
    st.write("Logic: SMA50 / SMA200 trend + RSI14 momentum.")

    refresh = st.button("🔄 Refresh global signals")

    if "df_signals_global" not in st.session_state or refresh:
        df_global = run_analysis_global(selected_classes)
        st.session_state["df_signals_global"] = df_global
    else:
        df_global = st.session_state["df_signals_global"]

    if df_global.empty:
        st.error("No global results. Possibly no data or connection issue.")
    else:
        def color_terminal(row):
            return ["color: #00ff00; background-color: #000000;" for _ in row]

        styled_df = df_global.style.apply(color_terminal, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Summary")
        for _, row in df_global.iterrows():
            st.markdown(
                f"**{row['name']}** (`{row['ticker']}`) — "
                f"Signal: **{row['signal']}** (conf: {int(row['confidence']*100)}%), "
                f"trend: `{row['trend']}`, momentum: `{row['momentum']}`, "
                f"RSI: `{row['rsi14']}`, Close: `{row['close']}`"
            )

# -------- CRYPTO TAB --------
with tab_crypto:
    st.subheader("Binance Crypto Signals")

    client_binance = get_binance_client(BINANCE_API_KEY, BINANCE_API_SECRET)
    if client_binance is None:
        err = st.session_state.get("binance_client_error", "")
        st.warning("Binance private client not available (keys/permissions/endpoint). Using public endpoints for klines.")
        if err:
            st.caption(f"Client init error: {err}")

    col_left, col_right = st.columns([1, 3])

    with col_left:
        timeframe_label = st.selectbox(
            "Timeframe",
            options=list(BINANCE_TIMEFRAMES.keys()),
            index=0,
            key="crypto_timeframe_label",
        )
        refresh_crypto = st.button("🔄 Refresh crypto signals", key="refresh_crypto_btn")
        st.write("Data: Binance klines, limit 500.")
        st.write("Logic: SMA50 / SMA200 + RSI14.")

    with col_right:
        # Run / load cached session results
        if "df_signals_binance" not in st.session_state or refresh_crypto:
            tf = BINANCE_TIMEFRAMES[timeframe_label]
            df_crypto = run_analysis_binance(tf)
            st.session_state["df_signals_binance"] = df_crypto
            st.session_state["df_signals_binance_tf"] = tf
        else:
            df_crypto = st.session_state["df_signals_binance"]

        if df_crypto is None or df_crypto.empty:
            st.error("No Binance crypto results.")
        else:
            def color_terminal_c(row):
                return ["color: #00ff00; background-color: #000000;" for _ in row]

            styled_df_crypto = df_crypto.style.apply(color_terminal_c, axis=1)
            st.dataframe(styled_df_crypto, use_container_width=True)

            st.markdown("---")
            st.subheader("Summary")
            for _, row in df_crypto.iterrows():
                st.markdown(
                    f"**{row['name']}** (`{row['symbol']}`, {row['timeframe']}) — "
                    f"Signal: **{row['signal']}** (conf: {int(row['confidence']*100)}%), "
                    f"trend: `{row['trend']}`, momentum: `{row['momentum']}`, "
                    f"RSI: `{row['rsi14']}`, Close: `{row['close']}`"
                )


# -------- NEWS TAB --------
with tab_news:
    st.subheader("Global News & Macro Context")

    if not NEWSAPI_KEY:
        st.info(
            "NEWSAPI_KEY is missing in .env. Add NEWSAPI_KEY=... to load news."
        )
    else:
        st.write(
            "Tracking news for major assets (BTC, ETH, Gold, Silver, Nvidia, Apple, Microsoft, Tesla, Amazon, etc.), "
            "major indices, central banks and key figures."
        )

        df_global_for_ai = st.session_state.get("df_signals_global", pd.DataFrame())
        df_crypto_for_ai = st.session_state.get("df_signals_binance", pd.DataFrame())

        asset_options: List[str] = ["Global macro view"]

        if not df_global_for_ai.empty:
            asset_options.extend(
                df_global_for_ai["name"].astype(str)
                + " ("
                + df_global_for_ai["ticker"].astype(str)
                + ")"
            )
        if not df_crypto_for_ai.empty:
            asset_options.extend(
                df_crypto_for_ai["name"].astype(str)
                + " ("
                + df_crypto_for_ai["symbol"].astype(str)
                + ")"
            )

        asset_options = sorted(set(asset_options))
        current_focus = st.session_state.get("news_focus_asset", "Global macro view")

        focus_asset = st.selectbox(
            "Asset to focus (news-based forecast):",
            options=asset_options,
            index=asset_options.index(current_focus)
            if current_focus in asset_options
            else 0,
        )
        st.session_state["news_focus_asset"] = focus_asset

        # авто-фетч при първо зареждане
        if "news_items" not in st.session_state:
            news_items_initial = aggregate_news(NEWS_KEYWORDS)
            st.session_state["news_items"] = news_items_initial
            if news_items_initial:
                news_forecast_init = run_news_forecast(
                    df_global=df_global_for_ai,
                    df_crypto=df_crypto_for_ai,
                    latest_news_items=news_items_initial,
                    focus_asset=focus_asset,
                )
                st.session_state["news_forecast"] = news_forecast_init
                st.session_state["news_forecast_asset"] = focus_asset

        # бутони
        if st.button("🔄 Refresh news"):
            news_items_prev = st.session_state.get("news_items", [])
            news_items = aggregate_news(NEWS_KEYWORDS)

            if news_items:
                st.session_state["news_items"] = news_items
            else:
                news_items = news_items_prev
                if not news_items:
                    st.warning("No news available yet (even from history).")

            if news_items:
                news_forecast = run_news_forecast(
                    df_global=df_global_for_ai,
                    df_crypto=df_crypto_for_ai,
                    latest_news_items=news_items,
                    focus_asset=focus_asset,
                )
                st.session_state["news_forecast"] = news_forecast
                st.session_state["news_forecast_asset"] = focus_asset

        if st.button("♻️ Re-run forecast for selected asset (no refresh)"):
            news_items_for_run = st.session_state.get("news_items", [])
            if news_items_for_run:
                news_forecast = run_news_forecast(
                    df_global=df_global_for_ai,
                    df_crypto=df_crypto_for_ai,
                    latest_news_items=news_items_for_run,
                    focus_asset=focus_asset,
                )
                st.session_state["news_forecast"] = news_forecast
                st.session_state["news_forecast_asset"] = focus_asset

        news_items = st.session_state.get("news_items", [])
        news_forecast = st.session_state.get("news_forecast")
        news_forecast_asset = st.session_state.get("news_forecast_asset")

        if news_items and (not news_forecast or news_forecast_asset != focus_asset):
            news_forecast = run_news_forecast(
                df_global=df_global_for_ai,
                df_crypto=df_crypto_for_ai,
                latest_news_items=news_items,
                focus_asset=focus_asset,
            )
            st.session_state["news_forecast"] = news_forecast
            st.session_state["news_forecast_asset"] = focus_asset

        st.markdown("### 🤖 AI News-driven forecast")
        st.write(
            "Detailed forecast based on the news flow and signals for the selected asset."
        )

        if news_forecast:
            st.markdown("---")
            st.markdown(news_forecast)
        else:
            st.info(
                "No forecast yet. Press 'Refresh news' or 'Re-run forecast' for the selected asset."
            )

        st.markdown("---")
        st.markdown("### Latest raw headlines")

        if not news_items:
            st.warning("No news loaded yet. Press 'Refresh news'.")
        else:
            for item in news_items[:20]:
                with st.container():
                    st.markdown(f"**[{item['title']}]({item['url']})**")
                    meta_line = (
                        f"{item['source']} • {item['published_at']} • "
                        f"keyword: _{item['keyword']}_"
                    )
                    st.caption(meta_line)
                    if item["description"]:
                        st.write(item["description"])
                    st.markdown("---")

                    
# -------- QUANT TAB --------
with tab_quant:
    st.subheader("🧮 Quant Lab — Quantitative Market Analysis (Yahoo + Binance)")

    df_global_for_q = st.session_state.get("df_signals_global", pd.DataFrame())
    df_crypto_for_q = st.session_state.get("df_signals_binance", pd.DataFrame())

    asset_options: List[str] = ["(choose)"]

    if not df_global_for_q.empty:
        asset_options.extend(
            df_global_for_q["name"].astype(str) + " (" + df_global_for_q["ticker"].astype(str) + ")"
        )
    if not df_crypto_for_q.empty:
        asset_options.extend(
            df_crypto_for_q["name"].astype(str) + " (" + df_crypto_for_q["symbol"].astype(str) + ")"
        )

    asset_options = sorted(set(asset_options))

    colA, colB, colC = st.columns(3)
    with colA:
        focus_asset = st.selectbox("Asset:", options=asset_options, index=0)
        source_pref = st.selectbox("Source:", options=["Auto", "Yahoo", "Binance"], index=0)

    with colB:
        lookback_days = st.slider("Lookback (days):", 90, 730, 365, 30)
        jump_z = st.slider("Jump threshold (z):", 2.0, 5.0, 3.0, 0.25)

    with colC:
        tf_binance = st.selectbox("Binance timeframe:", options=["1d", "4h", "1h", "15m"], index=0)
        horizon_label = st.selectbox("Horizon:", options=["1 day", "1 week", "1 month"], index=1)

    run_quant = st.button("🧮 Run Quant Analysis", type="primary")

    if run_quant:
        sym = parse_selected_asset_to_symbol(focus_asset)
        if not sym:
            st.warning("Select an asset.")
        else:
            source = detect_source_for_symbol(sym, preferred=source_pref)

            # horizon bars
            mpd = bars_per_day_for_tf(source, tf_binance)
            if horizon_label == "1 day":
                horizon_bars = 1 * mpd
            elif horizon_label == "1 week":
                horizon_bars = 7 * mpd
            else:
                horizon_bars = 30 * mpd

            # timeframe used
            tf_used = "1d" if source == "Yahoo" else tf_binance
            bpy = bars_per_year_for_timeframe(source, tf_used)

            try:
                close_series = fetch_close_series_for_quant(sym, source, tf_used, lookback_days)
                qm = compute_quant_metrics(close_series, bars_per_year=bpy, jump_z=float(jump_z), horizon_bars=int(horizon_bars))

                if "error" in qm:
                    st.error(qm["error"])
                else:
                    st.markdown("### Quant Metrics")
                    st.dataframe(pd.DataFrame([qm]), use_container_width=True)

                    st.markdown("---")
                    st.markdown("### GPT Quant Analysis (based ONLY on the numbers above)")
                    brief = quant_metrics_to_brief(sym, source, tf_used, lookback_days, horizon_label, qm)
                    gpt_text = run_quant_gpt_analysis(brief)
                    st.markdown(gpt_text)

            except Exception as e:
                st.error(f"Quant Lab error: {type(e).__name__}: {e}")

# -------- AI TAB (можеш да го ползваш по-късно за други модули) --------
with tab_ai:
    st.subheader("AI Market Analyst (bottom of page)")
    st.info("Основният AI Market Analyst панел е по-долу на страницата. Тук може да добавиш допълнителни AI модули в бъдеще.")

# -------- FOMC TAB --------
with tab_fomc:
    show_fomc_lab()

# ================= AI MARKET ANALYST (GPT) =================

st.markdown("---")
st.subheader("🤖 AI Market Analyst")

col_left, col_right = st.columns([2, 3])

with col_left:
    all_assets: List[str] = []

    df_global_for_ai = st.session_state.get("df_signals_global", pd.DataFrame())
    df_crypto_for_ai = st.session_state.get("df_signals_binance", pd.DataFrame())

    if not df_global_for_ai.empty:
        all_assets.extend(
            df_global_for_ai["name"].astype(str)
            + " ("
            + df_global_for_ai["ticker"].astype(str)
            + ")"
        )
    if not df_crypto_for_ai.empty:
        all_assets.extend(
            df_crypto_for_ai["name"].astype(str)
            + " ("
            + df_crypto_for_ai["symbol"].astype(str)
            + ")"
        )

    all_assets = sorted(set(all_assets))

    target_asset = st.selectbox(
        "Asset to focus on (optional):",
        options=["(none)"] + all_assets,
        index=0,
    )

    horizon = st.selectbox(
        "Time horizon:",
        options=[
            "1 day",
            "1 week",
            "1 month",
            "3 months",
            "6 months",
            "1 year",
        ],
        index=2,
    )

with col_right:
    user_question = st.text_area(
        "Question to the AI analyst (can be general or asset-specific):",
        value="Provide a detailed analysis and trading scenarios for the current markets.",
        height=160,
    )

    if st.button("🚀 Run AI analysis"):
        asset_for_ai = "" if target_asset == "(none)" else target_asset

    # ✅ ВИНАГИ взимай новини при AI анализа (с fallback към history при 429/грешка)
        news_items_for_ai = aggregate_news(NEWS_KEYWORDS)
        st.session_state["news_items"] = news_items_for_ai

        answer = run_ai_analyst(
            df_global=df_global_for_ai,
            df_crypto=df_crypto_for_ai,
            news_items=news_items_for_ai,
            target_asset=asset_for_ai,
            horizon=horizon,
            user_question=user_question,
    )

        st.markdown("---")
        st.markdown(answer)


# ================= END OF APP LAYOUT =================

st.markdown("---")
st.write("Application loaded successfully.")
st.write(
    "Use the tabs above to view Global Signals, Crypto Signals, News & Macro, the FOMC Lab, "
    "or run the AI Market Analyst."
)












































