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
import pdfplumber
import io
import uuid

# New modules: SQLite data layer, source clients, background scheduler, Claude agent
try:
    import data_layer as dl
    import scheduler as bg_scheduler
    import agent as ai_agent
    _HAS_AGENT = True
except Exception as _e:
    _HAS_AGENT = False
    _AGENT_IMPORT_ERROR = str(_e)

st.set_page_config(page_title="AI Macro Agent", layout="wide")

# Start background scheduler exactly once per process
if _HAS_AGENT and "scheduler_started" not in st.session_state:
    try:
        bg_scheduler.start_scheduler(run_now=True)
        st.session_state["scheduler_started"] = True
    except Exception as _e:
        st.session_state["scheduler_started"] = False
        st.session_state["scheduler_error"] = str(_e)
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
NEWS_RETENTION_DAYS = 90
NEWS_LAST_FETCH_FILE = "news_last_fetch.json"

# Schedule: 3 fetches per day at these UTC hours
# 08:00 UTC = before EU open, 14:00 UTC = US open, 21:00 UTC = after US close
NEWS_FETCH_HOURS_UTC = [8, 14, 21]

# Split keywords into 3 rotation groups to stay under NewsAPI rate limits
# (~12 keywords per group x 3 articles = ~36 calls per fetch, well under 100/day limit)
NEWS_KEYWORD_GROUPS = {
    0: [  # Group A: Crypto + Commodities + Major macro
        "Bitcoin", "Ethereum", "Gold", "Silver",
        "S&P 500", "Nasdaq 100", "Federal Reserve",
        "Jerome Powell", "United States economy",
        "ECB", "Christine Lagarde", "China economy",
    ],
    1: [  # Group B: Big tech + Key figures
        "Nvidia", "Apple", "Microsoft", "Tesla",
        "Alphabet", "Google", "Amazon", "Netflix",
        "BlackRock", "JPMorgan", "Elon Musk", "Bill Gates",
    ],
    2: [  # Group C: Defense + Macro + Institutions
        "ASML", "L3Harris", "AeroVironment", "Kratos Defense",
        "Allianz", "Rheinmetall", "European defense spending",
        "NATO defense", "Bank of Japan", "Bank of England",
        "IMF", "World Bank", "EU economy",
    ],
}

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
    "currency": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "NZD/USD": "NZDUSD=X",
        "USD/CNY": "CNY=X",
        "USD/TRY": "TRY=X",
        "USD/RUB": "RUB=X",
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
    ("EURUSD=X", "EUR/USD"),
    ("GBPUSD=X", "GBP/USD"),
    ("JPY=X", "USD/JPY"),

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


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def compute_bollinger_bands(close: pd.Series, window: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bb_pct = (close - lower) / (upper - lower + 1e-12)
    return {"upper": upper, "middle": sma, "lower": lower, "bb_pct": bb_pct}


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d = k.rolling(d_period).mean()
    return {"k": k, "d": d}


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up_move = high.diff()
    down_move = -low.diff()
    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    pos_di = 100.0 * pos_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    neg_di = 100.0 * neg_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    dx = 100.0 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-12)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


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

def basic_signal_from_series(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    df = pd.DataFrame({"Close": close})
    if high is not None:
        df["High"] = high.values if hasattr(high, "values") else high
    if low is not None:
        df["Low"] = low.values if hasattr(low, "values") else low

    df["sma20"]  = df["Close"].rolling(20).mean()
    df["sma50"]  = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(200).mean()
    df["rsi14"]  = compute_rsi(df["Close"])

    macd_out = compute_macd(df["Close"])
    df["macd"]      = macd_out["macd"]
    df["macd_sig"]  = macd_out["signal"]
    df["macd_hist"] = macd_out["histogram"]

    bb_out = compute_bollinger_bands(df["Close"])
    df["bb_upper"] = bb_out["upper"]
    df["bb_lower"] = bb_out["lower"]
    df["bb_pct"]   = bb_out["bb_pct"]

    has_hl = "High" in df.columns and "Low" in df.columns
    if has_hl:
        stoch_out     = compute_stochastic(df["High"], df["Low"], df["Close"])
        df["stoch_k"] = stoch_out["k"]
        df["stoch_d"] = stoch_out["d"]
        df["adx"]     = compute_adx(df["High"], df["Low"], df["Close"])

    df = df.dropna()
    if df.empty:
        raise ValueError("Not enough data for indicators")

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    close_v   = float(last["Close"])
    sma20     = float(last["sma20"])
    sma50     = float(last["sma50"])
    sma200    = float(last["sma200"])
    rsi14     = float(last["rsi14"])
    macd_h    = float(last["macd_hist"])
    macd_h_p  = float(prev["macd_hist"])
    bb_pct    = float(last["bb_pct"])
    macd_v    = float(last["macd"])
    macd_s    = float(last["macd_sig"])

    score = 0.0

    # 1. SMA Trend (weight ~3.5)
    if close_v > sma200 and sma50 > sma200:
        score += 3.0
    elif close_v > sma200:
        score += 1.5
    elif close_v < sma200 and sma50 < sma200:
        score -= 3.0
    elif close_v < sma200:
        score -= 1.5
    score += 0.5 if close_v > sma50 else -0.5

    # 2. RSI (weight ~2.0)
    if rsi14 >= 70:
        score += 1.5
    elif rsi14 >= 60:
        score += 2.0
    elif rsi14 >= 55:
        score += 1.0
    elif rsi14 >= 45:
        score += 0.0
    elif rsi14 >= 40:
        score -= 1.0
    elif rsi14 >= 30:
        score -= 2.0
    else:
        score -= 1.5

    # 3. MACD (weight ~2.0)
    if macd_v > macd_s:
        score += 1.0
        if macd_h > macd_h_p:
            score += 1.0
    elif macd_v < macd_s:
        score -= 1.0
        if macd_h < macd_h_p:
            score -= 1.0

    # 4. Bollinger %B (weight ~1.5)
    if bb_pct > 0.8:
        score += 1.5
    elif bb_pct > 0.6:
        score += 0.75
    elif bb_pct < 0.2:
        score -= 1.5
    elif bb_pct < 0.4:
        score -= 0.75

    # 5. Stochastic + ADX (weight ~1.5, only if H/L available)
    stoch_k = stoch_d = adx_v = None
    if has_hl and "stoch_k" in df.columns:
        stoch_k = float(last["stoch_k"])
        stoch_d = float(last["stoch_d"])
        adx_v   = float(last["adx"])

        if stoch_k > 80 and stoch_k > stoch_d:
            score += 1.5
        elif stoch_k > 60:
            score += 0.75
        elif stoch_k < 20 and stoch_k < stoch_d:
            score -= 1.5
        elif stoch_k < 40:
            score -= 0.75

        if adx_v > 25:
            score *= 1.15
        elif adx_v < 20:
            score *= 0.85

    score = round(float(score), 2)

    if score >= 7:
        signal, confidence = "STRONG BUY",  0.92
    elif score >= 4:
        signal, confidence = "BUY",          0.75
    elif score >= 1.5:
        signal, confidence = "WEAK BUY",     0.60
    elif score >= -1.5:
        signal, confidence = "HOLD",         0.50
    elif score >= -4:
        signal, confidence = "WEAK SELL",    0.60
    elif score >= -7:
        signal, confidence = "SELL",         0.75
    else:
        signal, confidence = "STRONG SELL",  0.92

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

    out: Dict[str, Any] = {
        "close":      round(close_v, 4),
        "sma20":      round(sma20, 4),
        "sma50":      round(sma50, 4),
        "sma200":     round(sma200, 4),
        "rsi14":      round(rsi14, 2),
        "macd_hist":  round(macd_h, 6),
        "bb_pct":     round(bb_pct, 4),
        "score":      score,
        "trend":      trend,
        "momentum":   momentum,
        "signal":     signal,
        "confidence": round(confidence, 2),
    }
    if stoch_k is not None:
        out["stoch_k"] = round(stoch_k, 2)
        out["stoch_d"] = round(stoch_d, 2)
    if adx_v is not None:
        out["adx"] = round(adx_v, 2)

    return out


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


def compute_autocorrelation(returns: pd.Series, max_lag: int = 10) -> Dict[str, float]:
    """Serial correlation of returns — Renaissance-style mean-reversion detection."""
    acf = {}
    for lag in range(1, max_lag + 1):
        corr = returns.autocorr(lag=lag)
        acf[f"lag_{lag}"] = round(float(corr), 4) if not np.isnan(corr) else 0.0
    return acf


def compute_regime_hmm_simple(returns: pd.Series, window: int = 63) -> Dict[str, Any]:
    """
    Simple volatility regime detection using rolling vol percentile.
    Approximates Hidden Markov Model regime states without external libraries.
    """
    roll_vol = returns.rolling(window).std()
    if roll_vol.dropna().empty:
        return {"current_regime": "UNKNOWN", "regime_percentile": None, "regime_duration_bars": 0}

    current_vol = float(roll_vol.iloc[-1])
    pct = float((roll_vol.dropna() < current_vol).mean() * 100.0)

    if pct >= 80:
        regime = "HIGH_VOL_CRISIS"
    elif pct >= 60:
        regime = "ELEVATED_VOL"
    elif pct >= 40:
        regime = "NORMAL_VOL"
    elif pct >= 20:
        regime = "LOW_VOL_COMPLACENT"
    else:
        regime = "ULTRA_LOW_VOL"

    # How long have we been in this regime?
    vol_series = roll_vol.dropna()
    duration = 0
    for v in reversed(vol_series.values):
        v_pct = float((vol_series < v).mean() * 100.0)
        if abs(v_pct - pct) < 20:
            duration += 1
        else:
            break

    return {
        "current_regime": regime,
        "regime_percentile": round(pct, 1),
        "regime_duration_bars": duration,
        "current_vol_annualized": round(float(current_vol * np.sqrt(252)), 4),
    }


def compute_mean_reversion_signals(close: pd.Series) -> Dict[str, Any]:
    """
    Renaissance-style mean reversion metrics:
    - Z-score from rolling mean
    - Ornstein-Uhlenbeck half-life estimate
    - Bollinger band position
    """
    s = close.dropna().astype(float)
    if len(s) < 100:
        return {"z_score_20": None, "z_score_50": None, "ou_half_life": None}

    # Z-scores
    m20 = s.rolling(20).mean()
    std20 = s.rolling(20).std()
    z20 = ((s - m20) / (std20 + 1e-12)).iloc[-1]

    m50 = s.rolling(50).mean()
    std50 = s.rolling(50).std()
    z50 = ((s - m50) / (std50 + 1e-12)).iloc[-1]

    # Ornstein-Uhlenbeck half-life (simplified ADF-style regression)
    spread = s - m50
    spread_lag = spread.shift(1)
    delta_spread = spread.diff()
    valid = pd.concat([delta_spread, spread_lag], axis=1).dropna()
    valid.columns = ["delta", "lag"]

    ou_half_life = None
    if len(valid) > 30 and float(valid["lag"].std()) > 1e-12:
        beta = float(np.cov(valid["delta"], valid["lag"])[0, 1] / (np.var(valid["lag"]) + 1e-12))
        if beta < -0.001:
            ou_half_life = round(-np.log(2) / beta, 1)

    return {
        "z_score_20": round(float(z20), 3),
        "z_score_50": round(float(z50), 3),
        "ou_half_life": ou_half_life,
    }


def compute_momentum_features(close: pd.Series) -> Dict[str, Any]:
    """
    Cross-timeframe momentum — core of trend-following systematic strategies.
    """
    s = close.dropna().astype(float)
    if len(s) < 252:
        return {}

    ret_5d = float((s.iloc[-1] / s.iloc[-5] - 1.0) * 100) if len(s) >= 5 else None
    ret_21d = float((s.iloc[-1] / s.iloc[-21] - 1.0) * 100) if len(s) >= 21 else None
    ret_63d = float((s.iloc[-1] / s.iloc[-63] - 1.0) * 100) if len(s) >= 63 else None
    ret_126d = float((s.iloc[-1] / s.iloc[-126] - 1.0) * 100) if len(s) >= 126 else None
    ret_252d = float((s.iloc[-1] / s.iloc[-252] - 1.0) * 100) if len(s) >= 252 else None

    # Momentum score: weighted average of multi-period returns
    weights = [0.05, 0.15, 0.30, 0.25, 0.25]
    rets = [ret_5d, ret_21d, ret_63d, ret_126d, ret_252d]
    valid = [(w, r) for w, r in zip(weights, rets) if r is not None]
    mom_score = sum(w * r for w, r in valid) / (sum(w for w, _ in valid) + 1e-12) if valid else None

    return {
        "return_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
        "return_21d_pct": round(ret_21d, 2) if ret_21d is not None else None,
        "return_63d_pct": round(ret_63d, 2) if ret_63d is not None else None,
        "return_126d_pct": round(ret_126d, 2) if ret_126d is not None else None,
        "return_252d_pct": round(ret_252d, 2) if ret_252d is not None else None,
        "momentum_composite_score": round(mom_score, 3) if mom_score is not None else None,
    }


def compute_tail_risk_metrics(returns: pd.Series) -> Dict[str, Any]:
    """
    Advanced tail risk analysis beyond simple VaR/CVaR.
    """
    r = returns.dropna()
    if len(r) < 60:
        return {}

    # Sortino ratio (downside deviation)
    downside = r[r < 0]
    downside_std = float(downside.std()) if len(downside) > 5 else None
    sortino = float(r.mean() / (downside_std + 1e-12) * np.sqrt(252)) if downside_std else None

    # Gain/pain ratio
    total_gain = float(r[r > 0].sum())
    total_pain = float(abs(r[r < 0].sum()))
    gain_pain_ratio = round(total_gain / (total_pain + 1e-12), 3)

    # Calmar ratio (annualized return / max drawdown)
    ann_ret = float(r.mean() * 252)
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1.0)
    max_dd = float(dd.min())
    calmar = round(ann_ret / (abs(max_dd) + 1e-12), 3) if max_dd != 0 else None

    # Tail ratio (95th percentile gain / 5th percentile loss)
    p95 = float(np.percentile(r, 95))
    p05 = float(np.percentile(r, 5))
    tail_ratio = round(abs(p95 / (p05 + 1e-12)), 3)

    # Win rate
    win_rate = round(float((r > 0).mean() * 100), 1)

    return {
        "sortino_ratio": round(sortino, 3) if sortino else None,
        "gain_pain_ratio": gain_pain_ratio,
        "calmar_ratio": calmar,
        "tail_ratio": tail_ratio,
        "win_rate_pct": win_rate,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
    }


def compute_correlation_to_benchmarks(close: pd.Series, lookback: int = 63) -> Dict[str, float]:
    """
    Rolling correlation to key benchmarks (SPY, Gold, DXY proxy).
    Uses Yahoo Finance for benchmark data.
    """
    benchmarks = {"SPY": "SPY", "Gold": "GC=F", "DXY_proxy": "UUP"}
    correlations = {}
    s = close.dropna().astype(float)
    r_asset = np.log(s).diff().dropna()

    for label, ticker in benchmarks.items():
        try:
            df_b = fetch_yahoo_history(ticker, range_str="1y", interval="1d", max_points=365)
            r_bench = np.log(df_b["close"].dropna().astype(float)).diff().dropna()

            # Align dates
            common = r_asset.index.intersection(r_bench.index)
            if len(common) < lookback:
                correlations[f"corr_{label}"] = None
                continue

            corr = float(r_asset.loc[common].tail(lookback).corr(r_bench.loc[common].tail(lookback)))
            correlations[f"corr_{label}"] = round(corr, 3) if not np.isnan(corr) else None
        except Exception:
            correlations[f"corr_{label}"] = None

    return correlations


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

@st.cache_data(ttl=60, show_spinner=False)
def run_quant_gpt_analysis(brief: str) -> str:
    client = get_openai_client()
    if client is None:
        return "Quant GPT analysis error: OpenAI client not configured."

    system_prompt = """
You are a quantitative portfolio manager at a systematic hedge fund.
Translate raw quant metrics into clear, actionable strategy insights.
Write like a Two Sigma or Bridgewater internal memo — precise, quantitative, no filler.

Use ONLY the numbers in the brief. Never invent prices or indicators.

Output format (markdown):

## REGIME SUMMARY
What do vol regime, Hurst exponent, skew and kurtosis together reveal about market microstructure?
Is this trending or mean-reverting? Fat-tailed or normal? State regime clearly.

## RISK PROFILE
- **Vol Regime:** interpret LOW/NORMAL/HIGH in context
- **Jump Risk:** frequency (lambda), typical size, danger level
- **Tail Risk:** what the VaR/CVaR numbers mean in plain terms
- **Max Drawdown:** historical pain level and recovery context

## PRICE SCENARIO RANGES (Monte Carlo)
| Percentile | Price | Interpretation |
|------------|-------|----------------|
| P10 (bear) | ...   | worst 10% outcome |
| P50 (base) | ...   | median expected   |
| P90 (bull) | ...   | best 10% outcome  |

## STRATEGY PLAYBOOKS (3 rule-based approaches)
For each: Entry trigger | Exit condition | Risk management rule

1. **[Strategy Name]:** ...
2. **[Strategy Name]:** ...
3. **[Strategy Name]:** ...

## RED FLAGS & INVALIDATION
What conditions would break each playbook? What data would change this analysis?
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
        h = df["high"] if "high" in df.columns else None
        l = df["low"] if "low" in df.columns else None
        sig = basic_signal_from_series(df["close"], h, l)

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
    base_cols = [
        "name", "ticker", "asset_class",
        "signal", "confidence", "score",
        "trend", "momentum",
        "rsi14", "macd_hist", "bb_pct",
        "close", "sma20", "sma50", "sma200",
        "lambda_year", "avg_jump_pct", "jump_vol_pct",
        "sigma_diffusion_pct", "jump_risk_score", "jumps_count",
    ]
    optional_cols = ["adx", "stoch_k", "stoch_d"]
    cols = [c for c in base_cols if c in df.columns] + [c for c in optional_cols if c in df.columns]
    df = df[cols]
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
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
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
            sig = basic_signal_from_series(df["close"], df["high"], df["low"])

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


# ------------------------------------
# SCHEDULED NEWS AUTO-FETCH
# ------------------------------------

def _load_fetch_state() -> Dict[str, Any]:
    """Load the last fetch timestamp and group index."""
    if os.path.exists(NEWS_LAST_FETCH_FILE):
        try:
            with open(NEWS_LAST_FETCH_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_fetch_utc": None, "last_group": -1, "fetch_count_today": 0, "last_date": None}


def _save_fetch_state(state: Dict[str, Any]) -> None:
    try:
        with open(NEWS_LAST_FETCH_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


def _get_current_fetch_slot() -> Optional[int]:
    """
    Determine which scheduled slot we're in based on current UTC hour.
    Returns the slot index (0, 1, 2) if it's time to fetch, or None if not.
    """
    now_utc = dt.datetime.utcnow()
    current_hour = now_utc.hour

    # Find which slot matches (allow 1-hour window after each scheduled hour)
    for i, sched_hour in enumerate(NEWS_FETCH_HOURS_UTC):
        if sched_hour <= current_hour < sched_hour + 1:
            return i
    return None


def _get_nearest_slot_group() -> int:
    """Get the keyword group index for the nearest past fetch slot."""
    now_utc = dt.datetime.utcnow()
    current_hour = now_utc.hour

    # Find the most recent scheduled hour
    best_slot = 0
    for i, sched_hour in enumerate(NEWS_FETCH_HOURS_UTC):
        if current_hour >= sched_hour:
            best_slot = i
    return best_slot


def auto_fetch_news_if_needed() -> Optional[List[Dict[str, Any]]]:
    """
    Smart scheduled news fetching — runs automatically on page load.

    Strategy:
    - 3 fetch windows per day at 08:00, 14:00, 21:00 UTC
    - Each window fetches a different keyword group (~12 keywords)
    - Total: ~36 calls x 3 = ~108 calls/day (fits NewsAPI free tier with margin)
    - If already fetched this window, uses cached history instead
    - Returns new items if fetched, None if skipped (use history)
    """
    if not NEWSAPI_KEY:
        return None

    state = _load_fetch_state()
    now_utc = dt.datetime.utcnow()
    today_str = now_utc.strftime("%Y-%m-%d")
    current_hour = now_utc.hour

    # Reset daily counter if new day
    if state.get("last_date") != today_str:
        state["fetch_count_today"] = 0
        state["last_date"] = today_str

    # Safety: max 3 fetches per day
    if state.get("fetch_count_today", 0) >= 3:
        return None

    # Check if we're in a fetch window
    slot = _get_current_fetch_slot()

    # If not in an exact window, check if we missed any fetches today
    if slot is None:
        # Check if any scheduled slot has been missed today
        last_fetch_str = state.get("last_fetch_utc")
        if last_fetch_str:
            try:
                last_fetch = dt.datetime.fromisoformat(last_fetch_str)
                hours_since = (now_utc - last_fetch).total_seconds() / 3600
                if hours_since < 4:  # fetched recently enough
                    return None
            except Exception:
                pass

        # Find the group for the nearest past slot
        slot = _get_nearest_slot_group()

        # Check if we already fetched for this slot today
        last_group = state.get("last_group", -1)
        if last_group == slot and state.get("last_date") == today_str:
            return None
    else:
        # We're in an exact window — check if already fetched this slot
        last_group = state.get("last_group", -1)
        if last_group == slot and state.get("last_date") == today_str:
            return None

    # Determine which keyword group to fetch
    group_idx = slot % len(NEWS_KEYWORD_GROUPS)
    keywords = NEWS_KEYWORD_GROUPS[group_idx]

    # Fetch!
    all_news: List[Dict[str, Any]] = []
    try:
        for kw in keywords:
            items = fetch_news_for_keyword(kw, page_size=3)
            all_news.extend(items)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            # Rate limited — don't count as a fetch, try again later
            return None
        return None
    except Exception:
        return None

    if all_news:
        all_news_sorted = sorted(all_news, key=lambda x: x.get("published_at", ""), reverse=True)
        update_news_history(all_news_sorted)

        # Update state
        state["last_fetch_utc"] = now_utc.isoformat()
        state["last_group"] = group_idx
        state["fetch_count_today"] = state.get("fetch_count_today", 0) + 1
        state["last_date"] = today_str
        _save_fetch_state(state)

        return all_news_sorted

    return None


def get_news_with_auto_fetch() -> List[Dict[str, Any]]:
    """
    Main entry point for getting news. Tries auto-fetch first,
    falls back to cached history.
    """
    # Try scheduled auto-fetch
    new_items = auto_fetch_news_if_needed()

    if new_items:
        return new_items

    # Fall back to history
    hist = load_news_history_df()
    if hist.empty:
        return []
    return hist.sort_values("published_at", ascending=False).to_dict("records")


def get_news_fetch_status() -> Dict[str, Any]:
    """Return current fetch schedule status for display."""
    state = _load_fetch_state()
    now_utc = dt.datetime.utcnow()
    today_str = now_utc.strftime("%Y-%m-%d")

    fetches_today = state.get("fetch_count_today", 0) if state.get("last_date") == today_str else 0
    last_fetch = state.get("last_fetch_utc", "Never")
    last_group = state.get("last_group", -1)

    # Next fetch time
    current_hour = now_utc.hour
    next_fetch_hour = None
    for h in NEWS_FETCH_HOURS_UTC:
        if current_hour < h:
            next_fetch_hour = h
            break
    if next_fetch_hour is None:
        next_fetch_hour = NEWS_FETCH_HOURS_UTC[0]  # tomorrow

    group_names = {0: "Crypto + Macro", 1: "Tech + Key Figures", 2: "Defense + Institutions"}
    last_group_name = group_names.get(last_group, "None")

    return {
        "fetches_today": fetches_today,
        "max_fetches": 3,
        "last_fetch": last_fetch,
        "last_group": last_group_name,
        "next_fetch_hour_utc": f"{next_fetch_hour:02d}:00 UTC",
        "schedule": [f"{h:02d}:00 UTC" for h in NEWS_FETCH_HOURS_UTC],
    }


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
        c for c in df_local.columns
        if c in [
            "name", "ticker", "symbol", "asset_class", "timeframe",
            "signal", "confidence", "score",
            "trend", "momentum",
            "rsi14", "macd_hist", "bb_pct", "stoch_k", "adx",
            "close", "sma20", "sma50", "sma200",
            "lambda_year", "avg_jump_pct", "jump_vol_pct",
            "sigma_diffusion_pct", "jump_risk_score", "jumps_count",
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
You are a senior institutional macro analyst at a bulge-bracket investment bank.
Your analysis reads like a Goldman Sachs or JPMorgan cross-asset morning note — direct, specific, actionable.
Your audience: professional portfolio managers and sophisticated investors.

Output format (use markdown headers):

## EXECUTIVE SUMMARY
2-3 sentences: the single most important insight right now.

## MACRO REGIME
Identify current regime: growth trend, inflation, central bank posture, risk appetite (risk-on/risk-off).

## CROSS-ASSET VIEW
Brief directional take: Equities | Fixed Income | FX | Crypto | Commodities

## ASSET ANALYSIS
(Deep dive on the target asset, or global view if no target specified)
- Technical positioning: trend, momentum, key levels, indicator confluence
- Macro/fundamental drivers
- Upcoming catalysts

## SCENARIOS & PROBABILITIES
| Scenario | Probability | Trigger | Implication |
|----------|-------------|---------|-------------|
| Bull     | X%          | ...     | ...         |
| Base     | X%          | ...     | ...         |
| Bear     | X%          | ...     | ...         |

## ACTIONABLE PLAYBOOK
- **Day Trader:** ...
- **Swing Trader (1-4 weeks):** ...
- **Position Trader (1-3 months):** ...
- **Long-term Investor:** ...

## TOP RISKS
3-5 concrete risks to the base case with likely market impact.

## BOTTOM LINE
1-2 sentences: what matters most and what to watch.

Rules:
- Use probabilities and scenarios, never certainties.
- Do NOT give investment advice or specific position sizing.
- Mark typical historical behavior as "typical behavior."
- Be direct — no filler, no hedging every sentence.
- Use ONLY the supplied data. Do not hallucinate prices or indicators.
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


@st.cache_data(ttl=120, show_spinner=False)
def run_asset_deep_analysis(
    asset_name: str,
    asset_type: str,
    signal_data: Dict[str, Any],
    quant_data: Dict[str, Any],
    momentum_data: Dict[str, Any],
    regime_data: Dict[str, Any],
    tail_data: Dict[str, Any],
    mean_rev_data: Dict[str, Any],
) -> str:
    client = get_openai_client()
    if client is None:
        return "AI analysis error: OpenAI client not configured."

    brief_lines = [f"ASSET: {asset_name}", f"TYPE: {asset_type}", "", "SIGNAL DATA:"]
    for k, v in signal_data.items():
        brief_lines.append(f"  {k}: {v}")
    brief_lines.append("")
    brief_lines.append("QUANT METRICS:")
    for k, v in quant_data.items():
        brief_lines.append(f"  {k}: {v}")
    brief_lines.append("")
    brief_lines.append("MOMENTUM:")
    for k, v in momentum_data.items():
        brief_lines.append(f"  {k}: {v}")
    brief_lines.append("")
    brief_lines.append("REGIME:")
    for k, v in regime_data.items():
        brief_lines.append(f"  {k}: {v}")
    brief_lines.append("")
    brief_lines.append("TAIL RISK:")
    for k, v in tail_data.items():
        brief_lines.append(f"  {k}: {v}")
    brief_lines.append("")
    brief_lines.append("MEAN REVERSION:")
    for k, v in mean_rev_data.items():
        brief_lines.append(f"  {k}: {v}")

    brief = "\n".join(brief_lines)

    asset_context = ""
    if asset_type == "currency":
        asset_context = """
Context: This is a FOREX currency pair. Consider:
- Central bank policy differentials (Fed, ECB, BoJ, BoE, etc.)
- Interest rate differentials and carry trade dynamics
- Trade balance and current account flows
- Risk sentiment (risk-on vs risk-off flows)
- Commodity linkages (AUD, CAD, NOK = commodity currencies)
- Safe haven dynamics (JPY, CHF, USD in crisis)
- Purchasing power parity and fair value models
"""
    elif asset_type == "crypto":
        asset_context = """
Context: This is a CRYPTOCURRENCY. Consider:
- On-chain metrics implications (hashrate, active addresses, whale flows)
- DeFi TVL and ecosystem growth for relevant chains
- Regulatory environment and institutional adoption
- Bitcoin dominance and altcoin rotation cycles
- Liquidity conditions (Fed policy, stablecoin supply)
- Halving cycles and supply dynamics (for BTC)
- Network upgrades and protocol changes
- Correlation to risk assets (Nasdaq, SPX) and inverse USD
"""

    system_prompt = f"""
You are a senior portfolio manager at a top systematic hedge fund.
Produce a comprehensive, institutional-grade analysis of the given asset.
{asset_context}
Use ONLY the data provided. Do NOT invent prices, levels, or indicators.

Output format (markdown):

## VERDICT
One-line: BULLISH / BEARISH / NEUTRAL with conviction (High/Medium/Low) and 1-sentence reason.

## TECHNICAL STRUCTURE
- Trend analysis (SMA alignment, ADX strength)
- Momentum (RSI, MACD, Stochastic interpretation)
- Mean reversion status (Z-scores, O-U half-life)
- Volatility regime and what it means for positioning

## RISK ASSESSMENT
- Tail risk profile (Sortino, Calmar, max drawdown context)
- Jump risk (frequency, expected size)
- Current regime risk level

## SCENARIOS (next 1-4 weeks)
| Scenario | Probability | Trigger | Expected Move |
|----------|-------------|---------|---------------|
| Bull     | %           | ...     | ...           |
| Base     | %           | ...     | ...           |
| Bear     | %           | ...     | ...           |

## STRATEGY PLAYBOOK
- **Scalper / Day Trader:** ...
- **Swing Trader (1-4 weeks):** ...
- **Position Trader (1-3 months):** ...

## KEY LEVELS & WATCHPOINTS
What specific conditions would change this view?
"""

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": brief},
        ],
        max_completion_tokens=2500,
        temperature=0.3,
    )
    return completion.choices[0].message.content.strip()


def run_deep_analysis_for_asset(asset_name: str, asset_type: str, close_series: pd.Series, signal_data: Dict[str, Any]) -> str:
    """Helper that computes all quant metrics and runs AI analysis for a single asset."""
    returns = np.log(close_series).diff().dropna()
    bpy = 252

    qm = compute_quant_metrics(close_series, bars_per_year=bpy, jump_z=3.0, horizon_bars=7)
    regime = compute_regime_hmm_simple(returns, window=min(63, max(20, len(returns) // 3)))
    mean_rev = compute_mean_reversion_signals(close_series)
    mom = compute_momentum_features(close_series)
    tail = compute_tail_risk_metrics(returns)

    return run_asset_deep_analysis(
        asset_name=asset_name,
        asset_type=asset_type,
        signal_data=signal_data,
        quant_data=qm,
        momentum_data=mom,
        regime_data=regime,
        tail_data=tail,
        mean_rev_data=mean_rev,
    )


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
ASSET UNDER ANALYSIS: {focus_name}

Produce a professional, news-driven market forecast:

## NEWS SENTIMENT ASSESSMENT
- Overall newsflow: BULLISH / BEARISH / MIXED / UNCLEAR (state confidence %)
- 2-3 key stories currently driving the narrative

## SHORT-TERM VIEW (next 1-14 days)
- Directional bias with conviction level (High/Medium/Low)
- Key price catalysts and event risks
- Technical setup context from the signals data

## MEDIUM-TERM SCENARIOS (1-3 months)
| Scenario | Probability | Required Conditions | Expected Move |
|----------|-------------|---------------------|---------------|
| Bull     | %           | ...                 | ...           |
| Base     | %           | ...                 | ...           |
| Bear     | %           | ...                 | ...           |

## STRUCTURAL THEMES (3-12 months)
Important recurring themes from the news that could drive longer-term moves.

## RISK WATCHLIST
Top 3-5 concrete risks: "If X happens → expect Y reaction"

## ACTIONABLE TAKEAWAYS BY PLAYER TYPE
- **Momentum Trader:** ...
- **Swing Trader:** ...
- **Position/Long-term Investor:** ...

Be specific. Reference the actual news where relevant. Use directional language.
"""

        system_prompt = """
You are a macro/news-driven trading analyst at a top hedge fund.
You translate news flow into precise directional views and actionable scenarios.
Write like a Bloomberg Intelligence or Morgan Stanley research note — direct, specific, no filler.
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
    client = get_openai_client()
    if client is None:
        return {"error": "OpenAI client is not configured (missing OPENAI_API_KEY)."}

    system_msg = """
You are a senior macro strategist at a top-tier investment bank with 20+ years of Fed-watching experience.
Analyze the FOMC statement and press conference, then deliver a complete cross-market impact assessment.
Write as if briefing the trading desk and portfolio managers on a Fed decision day.

Hard rules:
- Return ONLY valid JSON. No markdown, no extra text.
- Base ALL factual claims strictly on the provided text.
- You MAY give probabilistic market interpretations using typical historical Fed transmission mechanisms.
- Be specific, direct, and actionable — like a Goldman Sachs macro flash note.

Direction values: "bullish" | "bearish" | "neutral"
Magnitude values: "high" | "medium" | "low"
Allowed tone_change: "more_hawkish" | "more_dovish" | "similar"
Allowed trade_bias: "risk_on" | "risk_off" | "mixed"
hawk_dove_score: -5 (extremely dovish) to +5 (extremely hawkish), decimals allowed.

Output this exact JSON structure (fill every field):
{
  "hawk_dove_score": 0,
  "tone_change": "similar",
  "key_changes": [],
  "inflation_focus": 5,
  "labor_market_focus": 5,
  "growth_risk_focus": 5,
  "financial_stability_focus": 5,
  "summary": "",
  "trade_bias": "mixed",
  "rate_path": {
    "next_meeting_hike_pct": 5,
    "next_meeting_hold_pct": 75,
    "next_meeting_cut_pct": 20,
    "year_end_trajectory": "",
    "key_data_dependency": ""
  },
  "market_impact": {
    "equities": {
      "overall_direction": "neutral",
      "overall_magnitude": "low",
      "sp500": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "nasdaq": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "sectors": {
        "financials": {"direction": "neutral", "magnitude": "low", "rationale": ""},
        "real_estate": {"direction": "neutral", "magnitude": "low", "rationale": ""},
        "technology": {"direction": "neutral", "magnitude": "low", "rationale": ""},
        "utilities": {"direction": "neutral", "magnitude": "low", "rationale": ""},
        "energy": {"direction": "neutral", "magnitude": "low", "rationale": ""},
        "consumer_staples": {"direction": "neutral", "magnitude": "low", "rationale": ""},
        "healthcare": {"direction": "neutral", "magnitude": "low", "rationale": ""}
      }
    },
    "currencies": {
      "usd_index": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "eurusd": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "usdjpy": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "gbpusd": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "audusd": {"direction": "neutral", "magnitude": "low", "rationale": ""}
    },
    "crypto": {
      "bitcoin": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "ethereum": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "overall": {"direction": "neutral", "magnitude": "low", "rationale": ""}
    },
    "commodities": {
      "gold": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "silver": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "oil_wti": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "copper": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "natural_gas": {"direction": "neutral", "magnitude": "low", "rationale": ""}
    },
    "bonds": {
      "us_2y": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "us_10y": {"direction": "neutral", "magnitude": "low", "rationale": ""},
      "yield_curve_shape": ""
    }
  },
  "playbook": {
    "before_next_meeting": "",
    "first_15min": "",
    "next_24h": "",
    "next_week": ""
  },
  "investor_guide": {
    "day_trader": "",
    "swing_trader": "",
    "position_trader": "",
    "long_term_investor": "",
    "risk_manager": ""
  },
  "wall_street_take": ""
}
"""

    # Truncate inputs to avoid exceeding context limits
    max_input = 25000
    cur_t = current_text[:max_input] if len(current_text) > max_input else current_text
    prev_t = previous_text[:max_input] if len(previous_text) > max_input else previous_text
    press_t = pressconf_text[:max_input] if len(pressconf_text) > max_input else pressconf_text

    user_msg = f"""CURRENT FOMC STATEMENT:
{cur_t}

PREVIOUS FOMC STATEMENT (may be empty):
{prev_t}

PRESS CONFERENCE EXCERPTS (may be empty):
{press_t}

Instructions:
- key_changes: up to 8 concise bullets on exact wording/emphasis shifts vs. previous.
- summary: 4-6 sentences — what changed, what it implies for the policy path, what markets must price in.
- market_impact: explain the Fed transmission mechanism for EACH asset. Be specific about WHY each market moves.
  Use typical historical Fed transmission: hawkish = USD up, gold down, bonds down, growth stocks down, financials up, etc.
- investor_guide: 2-4 sentences per player type. What should they watch? How does this change positioning?
- wall_street_take: 1-2 punchy sentences. The "so what" headline a trader sends to their book right now.
"""

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.15,
        max_completion_tokens=4500,
    )

    msg = completion.choices[0].message
    raw = msg.content
    refusal = getattr(msg, "refusal", None)
    finish = completion.choices[0].finish_reason

    if refusal:
        return {"error": "Model refusal / blocked output", "refusal": refusal}

    if raw is None or not str(raw).strip():
        # Possibly hit token limit or content filter
        try:
            debug = msg.model_dump()
        except Exception:
            debug = str(msg)
        hint = ""
        if finish == "length":
            hint = " (output was cut off — token limit reached)"
        return {"error": f"Empty response from OpenAI model{hint}", "finish_reason": finish, "debug_message": debug}

    raw = str(raw)

    if isinstance(raw, dict):
        return raw

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # If truncated due to length, try to note that
        hint = " (possibly truncated — increase max_completion_tokens)" if finish == "length" else ""
        data = {"error": f"JSON parsing failed{hint}", "raw_response": raw[:2000]}

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
        max_completion_tokens=2000,
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
        data = {"error": "JSON parsing failed", "raw_response": raw[:2000]}

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


def extract_text_from_pdf(uploaded_file, max_chars: int = 60000) -> str:
    """Extract text from an uploaded PDF file, capped to max_chars."""
    text_parts = []
    total = 0
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
                total += len(page_text)
                if total >= max_chars:
                    break
    full = "\n\n".join(text_parts)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n\n[... truncated for analysis ...]"
    return full


def show_fomc_lab():
    if "fomc_current" not in st.session_state:
        st.session_state["fomc_current"] = ""
    if "fomc_previous" not in st.session_state:
        st.session_state["fomc_previous"] = ""
    if "fomc_press" not in st.session_state:
        st.session_state["fomc_press"] = ""
    if "fomc_meta" not in st.session_state:
        st.session_state["fomc_meta"] = {}

    st.title("🏛 FOMC Lab — Fed Policy & Cross-Market Impact Analyzer")
    st.markdown(
        "Institutional-grade FOMC analysis: policy tone, rate path, and cross-market impact "
        "across equities, currencies, crypto, commodities and bonds."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        load_clicked = st.button("📥 Load latest FOMC statements from Fed.gov")
    with col_btn2:
        analyze_clicked = st.button("🔍 Analyze FOMC", type="primary")

    if load_clicked:
        with st.spinner("Loading latest FOMC data from Fed.gov..."):
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
                f"Loaded FOMC: {meta.get('current_date','?')} | Previous: {meta.get('previous_date','?')}"
            )
            st.caption(f"Statement URL: {meta.get('current_url','')}")
            if meta.get("pressconf_url"):
                st.caption(f"Press Conference URL: {meta.get('pressconf_url')}")
            if meta.get("pressconf_error"):
                st.warning(meta.get("pressconf_error"))

    meta = st.session_state.get("fomc_meta", {})
    if not st.session_state["fomc_press"].strip() and meta.get("pressconf_url"):
        auto = st.session_state.get("fomc_press", "")
        if auto:
            st.session_state["fomc_press"] = auto

    # ── PDF UPLOAD SECTION ──
    st.markdown("#### 📄 Upload PDF files (or paste text below)")
    pdf_col1, pdf_col2, pdf_col3 = st.columns(3)
    with pdf_col1:
        pdf_current = st.file_uploader(
            "Current FOMC Statement (PDF)",
            type=["pdf"],
            key="pdf_current_upload",
        )
        if pdf_current is not None:
            with st.spinner("Extracting text from current statement PDF..."):
                extracted = extract_text_from_pdf(pdf_current)
            if extracted.strip():
                st.session_state["fomc_current"] = extracted
                st.success(f"Extracted {len(extracted):,} characters from PDF")
            else:
                st.warning("Could not extract text from this PDF (may be scanned/image-based).")

    with pdf_col2:
        pdf_previous = st.file_uploader(
            "Previous FOMC Statement (PDF)",
            type=["pdf"],
            key="pdf_previous_upload",
        )
        if pdf_previous is not None:
            with st.spinner("Extracting text from previous statement PDF..."):
                extracted = extract_text_from_pdf(pdf_previous)
            if extracted.strip():
                st.session_state["fomc_previous"] = extracted
                st.success(f"Extracted {len(extracted):,} characters from PDF")
            else:
                st.warning("Could not extract text from this PDF.")

    with pdf_col3:
        pdf_press = st.file_uploader(
            "Press Conference Transcript (PDF)",
            type=["pdf"],
            key="pdf_press_upload",
        )
        if pdf_press is not None:
            with st.spinner("Extracting text from press conference PDF..."):
                extracted = extract_text_from_pdf(pdf_press)
            if extracted.strip():
                st.session_state["fomc_press"] = extracted
                st.success(f"Extracted {len(extracted):,} characters from PDF")
            else:
                st.warning("Could not extract text from this PDF.")

    st.markdown("---")

    # ── TEXT AREAS (auto-filled from PDF or manual paste) ──
    col1, col2 = st.columns(2)
    with col1:
        current_text = st.text_area(
            "Current FOMC Statement (required)",
            height=260,
            key="fomc_current",
        )
    with col2:
        previous_text = st.text_area(
            "Previous FOMC Statement (optional)",
            height=260,
            key="fomc_previous",
        )
    pressconf_text = st.text_area(
        "Press Conference Excerpts (optional)",
        height=180,
        key="fomc_press",
    )

    if analyze_clicked:
        if not current_text.strip():
            st.error("Current FOMC statement is required.")
            return

        with st.spinner("Analyzing FOMC with GPT — cross-market impact assessment..."):
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

        def dir_emoji(d: str) -> str:
            return {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(d, "⚪")

        def mag_label(m: str) -> str:
            return {"high": "⚡⚡⚡", "medium": "⚡⚡", "low": "⚡"}.get(m, "—")

        # ── WALL STREET TAKE ──
        wst = result.get("wall_street_take", "")
        if wst:
            st.info(f"💬 **Wall Street Take:** {wst}")

        # ── MACRO SCOREBOARD ──
        st.subheader("📊 Macro Scoreboard")
        score = result.get("hawk_dove_score", 0)
        if score > 1.5:
            score_label = "🦅 Hawkish"
        elif score < -1.5:
            score_label = "🕊️ Dovish"
        else:
            score_label = "⚖️ Neutral"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hawk/Dove Score", f"{score:+.1f}", help="-5 = extremely dovish, +5 = extremely hawkish")
        c2.metric("Tone Change", result.get("tone_change", "").replace("_", " ").title())
        c3.metric("Trade Bias", result.get("trade_bias", "").replace("_", " ").upper())
        c4.metric("Policy Signal", score_label)

        st.markdown("#### Policy Focus (0–10)")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("🔥 Inflation", result.get("inflation_focus"))
        c6.metric("💼 Labor Market", result.get("labor_market_focus"))
        c7.metric("📈 Growth Risk", result.get("growth_risk_focus"))
        c8.metric("🏦 Fin. Stability", result.get("financial_stability_focus"))

        # ── RATE PATH ──
        st.markdown("---")
        st.subheader("🗺️ Rate Path Outlook")
        rp = result.get("rate_path", {})
        rp_c1, rp_c2, rp_c3 = st.columns(3)
        rp_c1.metric("Next Meeting: Hike", f"{rp.get('next_meeting_hike_pct', 0)}%")
        rp_c2.metric("Next Meeting: Hold", f"{rp.get('next_meeting_hold_pct', 0)}%")
        rp_c3.metric("Next Meeting: Cut", f"{rp.get('next_meeting_cut_pct', 0)}%")
        st.markdown(f"**Year-end trajectory:** {rp.get('year_end_trajectory', '')}")
        st.markdown(f"**Key data to watch:** {rp.get('key_data_dependency', '')}")

        # ── SUMMARY + KEY CHANGES ──
        st.markdown("---")
        col_s, col_k = st.columns([3, 2])
        with col_s:
            st.subheader("📝 Analysis Summary")
            st.write(result.get("summary", ""))
        with col_k:
            st.subheader("🔄 Key Changes vs. Previous")
            for kc in result.get("key_changes", []):
                st.markdown(f"• {kc}")

        # ── CROSS-MARKET IMPACT ──
        st.markdown("---")
        st.subheader("🌍 Cross-Market Impact")
        mi = result.get("market_impact", {})

        tab_eq, tab_fx, tab_cr, tab_cm, tab_bn = st.tabs(
            ["📈 Equities", "💱 Currencies", "🪙 Crypto", "🛢️ Commodities", "🏛️ Bonds"]
        )

        with tab_eq:
            eq = mi.get("equities", {})
            overall_d = eq.get("overall_direction", "neutral")
            overall_m = eq.get("overall_magnitude", "low")
            st.markdown(
                f"**Overall Equity Outlook:** {dir_emoji(overall_d)} **{overall_d.title()}** "
                f"| Impact Strength: {mag_label(overall_m)}"
            )
            col_idx, col_sec = st.columns(2)
            with col_idx:
                st.markdown("**Indices**")
                for idx_key, idx_label in [("sp500", "S&P 500"), ("nasdaq", "Nasdaq 100")]:
                    d = eq.get(idx_key, {})
                    st.markdown(
                        f"**{idx_label}:** {dir_emoji(d.get('direction',''))} "
                        f"{d.get('direction','').title()} {mag_label(d.get('magnitude',''))}"
                    )
                    st.caption(d.get("rationale", ""))
            with col_sec:
                st.markdown("**Sectors**")
                sector_labels = {
                    "financials": "Financials", "real_estate": "Real Estate",
                    "technology": "Technology", "utilities": "Utilities",
                    "energy": "Energy", "consumer_staples": "Consumer Staples",
                    "healthcare": "Healthcare",
                }
                for key, label in sector_labels.items():
                    d = eq.get("sectors", {}).get(key, {})
                    st.markdown(
                        f"{dir_emoji(d.get('direction',''))} **{label}:** "
                        f"{d.get('direction','').title()} — _{d.get('rationale','')}_"
                    )

        with tab_fx:
            fx = mi.get("currencies", {})
            fx_labels = {
                "usd_index": "USD Index (DXY)", "eurusd": "EUR/USD",
                "usdjpy": "USD/JPY", "gbpusd": "GBP/USD", "audusd": "AUD/USD",
            }
            for key, label in fx_labels.items():
                d = fx.get(key, {})
                st.markdown(
                    f"**{label}:** {dir_emoji(d.get('direction',''))} "
                    f"{d.get('direction','').title()} {mag_label(d.get('magnitude',''))}"
                )
                st.caption(d.get("rationale", ""))
                st.markdown("---")

        with tab_cr:
            cr = mi.get("crypto", {})
            cr_labels = {"bitcoin": "Bitcoin (BTC)", "ethereum": "Ethereum (ETH)", "overall": "Overall Crypto Market"}
            for key, label in cr_labels.items():
                d = cr.get(key, {})
                st.markdown(
                    f"**{label}:** {dir_emoji(d.get('direction',''))} "
                    f"{d.get('direction','').title()} {mag_label(d.get('magnitude',''))}"
                )
                st.caption(d.get("rationale", ""))
                st.markdown("---")
            st.info(
                "💡 **Why crypto reacts to the Fed:** Crypto is highly sensitive to USD liquidity conditions. "
                "Dovish = more liquidity → risk-on → crypto up. Hawkish = tighter liquidity → risk-off → crypto down. "
                "Bitcoin also acts as a partial inflation hedge and digital gold."
            )

        with tab_cm:
            cm = mi.get("commodities", {})
            cm_labels = {
                "gold": "Gold", "silver": "Silver",
                "oil_wti": "Oil (WTI)", "copper": "Copper", "natural_gas": "Natural Gas",
            }
            for key, label in cm_labels.items():
                d = cm.get(key, {})
                st.markdown(
                    f"**{label}:** {dir_emoji(d.get('direction',''))} "
                    f"{d.get('direction','').title()} {mag_label(d.get('magnitude',''))}"
                )
                st.caption(d.get("rationale", ""))
                st.markdown("---")
            st.info(
                "💡 **Key commodity mechanics:** Gold moves inversely to real rates and USD. "
                "Silver follows gold but with more industrial demand exposure. "
                "Oil is a growth proxy — Fed easing boosts demand outlook. "
                "Copper is the global growth barometer."
            )

        with tab_bn:
            bn = mi.get("bonds", {})
            for key, label in [("us_2y", "US 2Y Treasury"), ("us_10y", "US 10Y Treasury")]:
                d = bn.get(key, {})
                st.markdown(
                    f"**{label}:** {dir_emoji(d.get('direction',''))} "
                    f"{d.get('direction','').title()} {mag_label(d.get('magnitude',''))}"
                )
                st.caption(d.get("rationale", ""))
                st.markdown("---")
            yc = bn.get("yield_curve_shape", "")
            if yc:
                st.markdown(f"**Yield Curve Shape:** {yc}")
            st.info(
                "💡 **Bond mechanics:** 2Y yields are most sensitive to Fed policy expectations. "
                "10Y yields reflect both policy and long-term growth/inflation. "
                "Bond prices move OPPOSITE to yields — 'bullish bonds' means yields fall, prices rise."
            )

        # ── TRADING PLAYBOOK ──
        st.markdown("---")
        st.subheader("⚡ Trading Playbook")
        pb = result.get("playbook", {})
        pb_tabs = st.tabs(["Before Next Meeting", "First 15 Minutes", "Next 24 Hours", "Next Week"])
        for tab_obj, key, default in zip(
            pb_tabs,
            ["before_next_meeting", "first_15min", "next_24h", "next_week"],
            ["", "", "", ""],
        ):
            with tab_obj:
                st.write(pb.get(key, default) or "No specific guidance.")

        # ── INVESTOR GUIDE ──
        st.markdown("---")
        st.subheader("👤 Investor Guide — What This Means For You")
        ig = result.get("investor_guide", {})
        ig_items = [
            ("day_trader", "🏃 Day Trader", "Short-term volatility plays, intraday positioning"),
            ("swing_trader", "📊 Swing Trader", "1-4 week directional trades"),
            ("position_trader", "📅 Position Trader", "1-3 month thesis-driven positions"),
            ("long_term_investor", "🏛️ Long-term Investor", "Portfolio allocation changes, multi-month view"),
            ("risk_manager", "🛡️ Risk Manager", "Hedging, correlation changes, tail risk"),
        ]
        for key, label, subtitle in ig_items:
            text = ig.get(key, "")
            if text:
                with st.expander(f"{label} — {subtitle}"):
                    st.write(text)

        # ── PRESS CONF LEVEL 2 ──
        st.markdown("---")
        st.subheader("🧠 FOMC Press Conference — Key Topics")
        if pressconf_text.strip():
            with st.spinner("Extracting key topics..."):
                lvl2 = extract_fomc_pressconf_topics(pressconf_text)
            if "error" in lvl2:
                st.warning(lvl2.get("error"))
            else:
                c_tone, c_bias = st.columns(2)
                c_tone.metric("Overall Tone", lvl2.get("overall_tone", "").title())
                c_bias.metric("Trade Bias", lvl2.get("trade_bias", "").replace("_", " ").upper())
                st.markdown(f"**Change vs. previous:** {lvl2.get('implied_change_vs_previous', '')}")
                st.markdown("**Topics discussed:**")
                for t in lvl2.get("topics", []):
                    st.markdown(
                        f"- **{t.get('topic','')}** ({t.get('stance','')}) — "
                        f"{t.get('summary','')} | _Market take: {t.get('market_take','')}_"
                    )
                st.markdown("**Scenarios:**")
                for sc in lvl2.get("scenarios", []):
                    st.markdown(
                        f"- **{sc.get('name','')}** ({sc.get('probability',0)}%): {sc.get('description','')}"
                    )
                with st.expander("Raw Level 2 JSON"):
                    st.json(lvl2)
        else:
            st.info("No press conference text available. Load from Fed.gov or paste manually.")

        with st.expander("🔍 Raw JSON (full result)"):
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

tab_global, tab_fx, tab_crypto, tab_news, tab_quant, tab_ai, tab_fomc = st.tabs(
    ["🌍 Global Signals", "💱 Currencies", "🪙 Crypto (Binance)", "📰 News & Macro", "🧮 Quant Lab", "🤖 AI Analyst", "🏛 FOMC Lab"]

)

# -------- GLOBAL TAB --------
with tab_global:
    st.subheader("Global Signals — Yahoo Finance (1D, ~1y history)")

    all_classes = [c for c in ASSETS_BY_CLASS.keys() if c != "currency"]
    selected_classes = st.multiselect(
        "Asset classes to show:",
        options=all_classes,
        default=all_classes,
    )

    st.write("Data source: Yahoo Finance chart API.")
    st.write("Logic: Multi-indicator scoring (SMA + RSI + MACD + Bollinger + Stochastic + ADX).")

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
                f"Score: `{row.get('score','N/A')}`, "
                f"trend: `{row['trend']}`, momentum: `{row['momentum']}`, "
                f"RSI: `{row['rsi14']}`, Close: `{row['close']}`"
            )

# -------- FX / CURRENCIES TAB --------
with tab_fx:
    st.subheader("💱 Currencies — Institutional FX Dashboard")

    fx_tab_overview, fx_tab_analysis = st.tabs(["📊 FX Overview", "🔍 Deep Analysis"])

    with fx_tab_overview:
        st.markdown("Major and emerging market currency pairs. Multi-indicator scoring system.")
        refresh_fx = st.button("🔄 Refresh FX signals", key="refresh_fx_btn")

        if "df_signals_fx" not in st.session_state or refresh_fx:
            fx_rows: List[Dict[str, Any]] = []
            for name, ticker in ASSETS_BY_CLASS.get("currency", {}).items():
                r = analyze_yahoo_asset(name, ticker, "currency")
                if r:
                    fx_rows.append(r)
            df_fx = pd.DataFrame(fx_rows)
            if not df_fx.empty:
                base_cols = [
                    "name", "ticker", "asset_class",
                    "signal", "confidence", "score",
                    "trend", "momentum",
                    "rsi14", "macd_hist", "bb_pct",
                    "close", "sma20", "sma50", "sma200",
                ]
                optional_cols = ["adx", "stoch_k", "stoch_d"]
                cols = [c for c in base_cols if c in df_fx.columns] + [c for c in optional_cols if c in df_fx.columns]
                df_fx = df_fx[cols]
            st.session_state["df_signals_fx"] = df_fx
        else:
            df_fx = st.session_state["df_signals_fx"]

        if df_fx is None or df_fx.empty:
            st.error("No FX data available. Check connection.")
        else:
            def color_fx_row(row):
                return ["color: #00ff00; background-color: #000000;" for _ in row]
            styled_fx = df_fx.style.apply(color_fx_row, axis=1)
            st.dataframe(styled_fx, use_container_width=True)

            # Signal summary cards
            st.markdown("---")
            st.subheader("FX Signal Board")
            fx_cols_per_row = 5
            fx_names = df_fx["name"].tolist()
            for i in range(0, len(fx_names), fx_cols_per_row):
                cols_row = st.columns(min(fx_cols_per_row, len(fx_names) - i))
                for j, col in enumerate(cols_row):
                    idx = i + j
                    if idx < len(fx_names):
                        row = df_fx.iloc[idx]
                        sig = row["signal"]
                        if "BUY" in sig:
                            sig_icon = "🟢"
                        elif "SELL" in sig:
                            sig_icon = "🔴"
                        else:
                            sig_icon = "⚪"
                        score_val = row.get("score", "N/A")
                        with col:
                            st.markdown(
                                f"**{row['name']}**\n\n"
                                f"{sig_icon} **{sig}**\n\n"
                                f"Score: `{score_val}` | RSI: `{row['rsi14']}`\n\n"
                                f"Price: `{row['close']}`"
                            )

    with fx_tab_analysis:
        st.markdown("### 🔍 Deep FX Analysis — Select a Currency Pair")

        fx_pair_options = list(ASSETS_BY_CLASS.get("currency", {}).keys())
        if not fx_pair_options:
            st.warning("No FX pairs configured.")
        else:
            selected_fx = st.selectbox("Select currency pair:", fx_pair_options, key="fx_deep_select")
            fx_ticker = ASSETS_BY_CLASS["currency"][selected_fx]

            if st.button("🚀 Run Deep FX Analysis", key="fx_deep_run", type="primary"):
                with st.spinner(f"Analyzing {selected_fx}..."):
                    try:
                        df_hist = fetch_yahoo_history(fx_ticker, range_str="1y", interval="1d", max_points=365)
                        h = df_hist["high"] if "high" in df_hist.columns else None
                        l = df_hist["low"] if "low" in df_hist.columns else None
                        sig = basic_signal_from_series(df_hist["close"], h, l)
                        close_s = df_hist["close"].dropna().astype(float)
                        returns = np.log(close_s).diff().dropna()

                        # Compute all metrics
                        qm = compute_quant_metrics(close_s, bars_per_year=252, jump_z=3.0, horizon_bars=7)
                        regime = compute_regime_hmm_simple(returns, window=min(63, max(20, len(returns)//3)))
                        mean_rev = compute_mean_reversion_signals(close_s)
                        mom = compute_momentum_features(close_s)
                        tail = compute_tail_risk_metrics(returns)

                        # Display signal header
                        sig_val = sig.get("signal", "HOLD")
                        if "BUY" in sig_val:
                            sig_color = "🟢"
                        elif "SELL" in sig_val:
                            sig_color = "🔴"
                        else:
                            sig_color = "⚪"

                        st.markdown(f"## {sig_color} {selected_fx} — {sig_val}")

                        # Key metrics row
                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("Price", f"{sig['close']:.5f}")
                        m2.metric("Score", sig.get("score", "N/A"))
                        m3.metric("RSI", sig["rsi14"])
                        m4.metric("Trend", sig["trend"].title())
                        m5.metric("Vol Regime", regime.get("current_regime", "?"))
                        m6.metric("Hurst", f"{qm.get('hurst', 'N/A')}")

                        # Momentum
                        st.markdown("#### Momentum")
                        if mom:
                            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                            mc1.metric("5D", f"{mom.get('return_5d_pct', 'N/A')}%")
                            mc2.metric("1M", f"{mom.get('return_21d_pct', 'N/A')}%")
                            mc3.metric("3M", f"{mom.get('return_63d_pct', 'N/A')}%")
                            mc4.metric("6M", f"{mom.get('return_126d_pct', 'N/A')}%")
                            mc5.metric("12M", f"{mom.get('return_252d_pct', 'N/A')}%")

                        # Mean reversion
                        st.markdown("#### Mean Reversion")
                        mr1, mr2, mr3 = st.columns(3)
                        mr1.metric("Z-Score (20d)", mean_rev.get("z_score_20", "N/A"))
                        mr2.metric("Z-Score (50d)", mean_rev.get("z_score_50", "N/A"))
                        hl = mean_rev.get("ou_half_life")
                        mr3.metric("O-U Half-Life", f"{hl:.0f} days" if hl else "N/A (trending)")

                        # Risk
                        st.markdown("#### Risk Profile")
                        if tail:
                            r1, r2, r3, r4 = st.columns(4)
                            r1.metric("Sortino", tail.get("sortino_ratio", "N/A"))
                            r2.metric("Max DD", f"{tail.get('max_drawdown_pct', 'N/A')}%")
                            r3.metric("Win Rate", f"{tail.get('win_rate_pct', 'N/A')}%")
                            r4.metric("Tail Ratio", tail.get("tail_ratio", "N/A"))

                        # Monte Carlo
                        st.markdown("#### Monte Carlo (1 week ahead)")
                        mc_p10 = qm.get("mc_p10")
                        mc_p50 = qm.get("mc_p50")
                        mc_p90 = qm.get("mc_p90")
                        last_p = qm.get("last_price", 0)
                        if mc_p10 and last_p:
                            s1, s2, s3 = st.columns(3)
                            s1.metric("🔴 Bear (P10)", f"{mc_p10:.5f}", f"{((mc_p10/last_p)-1)*100:+.2f}%")
                            s2.metric("⚪ Base (P50)", f"{mc_p50:.5f}", f"{((mc_p50/last_p)-1)*100:+.2f}%")
                            s3.metric("🟢 Bull (P90)", f"{mc_p90:.5f}", f"{((mc_p90/last_p)-1)*100:+.2f}%")

                        # AI Analysis
                        st.markdown("---")
                        st.subheader("🧠 AI Deep Analysis")
                        with st.spinner("Generating institutional-grade FX analysis..."):
                            ai_text = run_asset_deep_analysis(
                                asset_name=selected_fx,
                                asset_type="currency",
                                signal_data=sig,
                                quant_data=qm,
                                momentum_data=mom,
                                regime_data=regime,
                                tail_data=tail,
                                mean_rev_data=mean_rev,
                            )
                        st.markdown(ai_text)

                    except Exception as e:
                        st.error(f"FX analysis error: {type(e).__name__}: {e}")

# -------- CRYPTO TAB --------
with tab_crypto:
    st.subheader("🪙 Crypto — Institutional Digital Asset Dashboard")

    crypto_tab_overview, crypto_tab_analysis = st.tabs(["📊 Crypto Overview", "🔍 Deep Analysis"])

    with crypto_tab_overview:
        client_binance = get_binance_client(BINANCE_API_KEY, BINANCE_API_SECRET)
        if client_binance is None:
            err = st.session_state.get("binance_client_error", "")
            st.warning("Binance private client not available. Using public endpoints.")
            if err:
                st.caption(f"Client init error: {err}")

        col_ctrl, col_data = st.columns([1, 4])

        with col_ctrl:
            timeframe_label = st.selectbox(
                "Timeframe",
                options=list(BINANCE_TIMEFRAMES.keys()),
                index=0,
                key="crypto_timeframe_label",
            )
            refresh_crypto = st.button("🔄 Refresh", key="refresh_crypto_btn")

        with col_data:
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
                def color_crypto_row(row):
                    return ["color: #00ff00; background-color: #000000;" for _ in row]

                styled_crypto = df_crypto.style.apply(color_crypto_row, axis=1)
                st.dataframe(styled_crypto, use_container_width=True)

        # Signal cards
        if df_crypto is not None and not df_crypto.empty:
            st.markdown("---")
            st.subheader("Crypto Signal Board")
            crypto_cols = st.columns(min(6, len(df_crypto)))
            for i, col in enumerate(crypto_cols):
                if i < len(df_crypto):
                    row = df_crypto.iloc[i]
                    sig = row["signal"]
                    if "BUY" in sig:
                        sig_icon = "🟢"
                    elif "SELL" in sig:
                        sig_icon = "🔴"
                    else:
                        sig_icon = "⚪"
                    score_val = row.get("score", "N/A")
                    with col:
                        st.markdown(
                            f"**{row['name']}**\n\n"
                            f"{sig_icon} **{sig}**\n\n"
                            f"Score: `{score_val}` | RSI: `{row['rsi14']}`\n\n"
                            f"Price: `${float(row['close']):,.2f}`"
                        )

    with crypto_tab_analysis:
        st.markdown("### 🔍 Deep Crypto Analysis — Select a Coin")

        crypto_options = {v["display"]: k for k, v in BINANCE_SYMBOLS.items()}
        crypto_names = list(crypto_options.keys())

        if not crypto_names:
            st.warning("No crypto symbols configured.")
        else:
            col_sel, col_tf = st.columns([2, 1])
            with col_sel:
                selected_crypto = st.selectbox("Select cryptocurrency:", crypto_names, key="crypto_deep_select")
            with col_tf:
                crypto_analysis_tf = st.selectbox(
                    "Analysis timeframe:",
                    options=["1d", "4h", "1h", "15m"],
                    index=0,
                    key="crypto_analysis_tf",
                )

            crypto_symbol = crypto_options[selected_crypto]

            if st.button("🚀 Run Deep Crypto Analysis", key="crypto_deep_run", type="primary"):
                with st.spinner(f"Analyzing {selected_crypto} ({crypto_symbol})..."):
                    try:
                        df_klines = fetch_binance_klines(crypto_symbol, interval=crypto_analysis_tf, limit=500)
                        close_s = df_klines["close"].dropna().astype(float)
                        high_s = df_klines["high"].dropna().astype(float) if "high" in df_klines.columns else None
                        low_s = df_klines["low"].dropna().astype(float) if "low" in df_klines.columns else None

                        sig = basic_signal_from_series(close_s, high_s, low_s)
                        returns = np.log(close_s).diff().dropna()

                        bars_map = {"1d": 252, "4h": 252*6, "1h": 252*24, "15m": 252*24*4}
                        bpy = bars_map.get(crypto_analysis_tf, 252)
                        mpd_map = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}
                        horizon_bars = 7 * mpd_map.get(crypto_analysis_tf, 1)

                        qm = compute_quant_metrics(close_s, bars_per_year=bpy, jump_z=3.0, horizon_bars=horizon_bars)
                        regime = compute_regime_hmm_simple(returns, window=min(63, max(20, len(returns)//3)))
                        mean_rev = compute_mean_reversion_signals(close_s)
                        mom = compute_momentum_features(close_s)
                        tail = compute_tail_risk_metrics(returns)

                        # Signal header
                        sig_val = sig.get("signal", "HOLD")
                        if "BUY" in sig_val:
                            sig_color = "🟢"
                        elif "SELL" in sig_val:
                            sig_color = "🔴"
                        else:
                            sig_color = "⚪"

                        st.markdown(f"## {sig_color} {selected_crypto} ({crypto_symbol}) — {sig_val}")
                        st.caption(f"Timeframe: {crypto_analysis_tf} | Data: Binance")

                        # Key metrics
                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("Price", f"${float(sig['close']):,.2f}")
                        m2.metric("Score", sig.get("score", "N/A"))
                        m3.metric("RSI", sig["rsi14"])
                        m4.metric("Trend", sig["trend"].title())
                        m5.metric("Vol Regime", regime.get("current_regime", "?"))
                        m6.metric("Hurst", f"{qm.get('hurst', 'N/A')}")

                        # Momentum
                        st.markdown("#### Momentum")
                        if mom:
                            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                            mc1.metric("5D", f"{mom.get('return_5d_pct', 'N/A')}%")
                            mc2.metric("1M", f"{mom.get('return_21d_pct', 'N/A')}%")
                            mc3.metric("3M", f"{mom.get('return_63d_pct', 'N/A')}%")
                            mc4.metric("6M", f"{mom.get('return_126d_pct', 'N/A')}%")
                            mc5.metric("12M", f"{mom.get('return_252d_pct', 'N/A')}%")
                        else:
                            st.info("Not enough data for full momentum (need 252+ bars). Try 1d timeframe.")

                        # Mean Reversion
                        st.markdown("#### Mean Reversion Signals")
                        mr1, mr2, mr3 = st.columns(3)
                        mr1.metric("Z-Score (20d)", mean_rev.get("z_score_20", "N/A"))
                        mr2.metric("Z-Score (50d)", mean_rev.get("z_score_50", "N/A"))
                        hl = mean_rev.get("ou_half_life")
                        mr3.metric("O-U Half-Life", f"{hl:.0f} bars" if hl else "N/A (trending)")

                        z20 = mean_rev.get("z_score_20")
                        if z20 is not None:
                            if z20 > 2.0:
                                st.warning("⚠️ Z-Score > 2.0 — price significantly above mean. Potential reversion risk.")
                            elif z20 < -2.0:
                                st.info("💡 Z-Score < -2.0 — price significantly below mean. Potential bounce zone.")

                        # Risk Profile
                        st.markdown("#### Risk Profile")
                        if tail:
                            r1, r2, r3, r4 = st.columns(4)
                            r1.metric("Sortino", tail.get("sortino_ratio", "N/A"))
                            r2.metric("Max DD", f"{tail.get('max_drawdown_pct', 'N/A')}%")
                            r3.metric("Win Rate", f"{tail.get('win_rate_pct', 'N/A')}%")
                            r4.metric("Tail Ratio", tail.get("tail_ratio", "N/A"))

                        # Jump Diffusion
                        st.markdown("#### Jump Risk")
                        j1, j2, j3, j4 = st.columns(4)
                        j1.metric("Jumps/Year", qm.get("lambda_year", "N/A"))
                        j2.metric("Avg Jump", f"{qm.get('avg_jump_pct', 'N/A')}%")
                        j3.metric("Jump Vol", f"{qm.get('jump_vol_pct', 'N/A')}%")
                        j4.metric("Jump Risk Score", qm.get("jump_risk_score", "N/A"))

                        # Monte Carlo
                        st.markdown("#### Monte Carlo Scenarios (1 week ahead)")
                        mc_p10 = qm.get("mc_p10")
                        mc_p50 = qm.get("mc_p50")
                        mc_p90 = qm.get("mc_p90")
                        last_p = qm.get("last_price", 0)
                        if mc_p10 and last_p:
                            s1, s2, s3 = st.columns(3)
                            s1.metric("🔴 Bear (P10)", f"${mc_p10:,.2f}", f"{((mc_p10/last_p)-1)*100:+.1f}%")
                            s2.metric("⚪ Base (P50)", f"${mc_p50:,.2f}", f"{((mc_p50/last_p)-1)*100:+.1f}%")
                            s3.metric("🟢 Bull (P90)", f"${mc_p90:,.2f}", f"{((mc_p90/last_p)-1)*100:+.1f}%")

                        # AI Analysis
                        st.markdown("---")
                        st.subheader("🧠 AI Deep Analysis")
                        with st.spinner(f"Generating institutional crypto analysis for {selected_crypto}..."):
                            ai_text = run_asset_deep_analysis(
                                asset_name=f"{selected_crypto} ({crypto_symbol})",
                                asset_type="crypto",
                                signal_data=sig,
                                quant_data=qm,
                                momentum_data=mom,
                                regime_data=regime,
                                tail_data=tail,
                                mean_rev_data=mean_rev,
                            )
                        st.markdown(ai_text)

                    except Exception as e:
                        st.error(f"Crypto analysis error: {type(e).__name__}: {e}")


# -------- NEWS TAB --------
with tab_news:
    st.subheader("Global News & Macro Context")

    if not NEWSAPI_KEY:
        st.info(
            "NEWSAPI_KEY is missing in .env. Add NEWSAPI_KEY=... to load news."
        )
    else:
        # ── AUTO-FETCH on page load ──
        if "news_auto_fetched" not in st.session_state:
            auto_items = auto_fetch_news_if_needed()
            if auto_items:
                st.session_state["news_items"] = auto_items
                st.session_state["news_auto_fetch_msg"] = "Auto-fetched new articles on schedule."
            st.session_state["news_auto_fetched"] = True

        # If no news at all, load from history
        if "news_items" not in st.session_state or not st.session_state.get("news_items"):
            hist = load_news_history_df()
            if not hist.empty:
                st.session_state["news_items"] = hist.sort_values("published_at", ascending=False).to_dict("records")

        # ── SCHEDULE STATUS ──
        fetch_status = get_news_fetch_status()
        with st.expander("📅 News Fetch Schedule"):
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Fetches Today", f"{fetch_status['fetches_today']}/{fetch_status['max_fetches']}")
            sc2.metric("Last Group", fetch_status["last_group"])
            sc3.metric("Next Fetch", fetch_status["next_fetch_hour_utc"])

            st.markdown(
                f"**Schedule (UTC):** {', '.join(fetch_status['schedule'])}\n\n"
                f"**Last fetch:** {fetch_status['last_fetch']}\n\n"
                "**How it works:** News keywords are split into 3 rotation groups. "
                "Each scheduled window fetches one group (~12 keywords x 3 articles = ~36 API calls). "
                "Total: ~108 calls/day — fits within NewsAPI free tier limits."
            )

            st.markdown(
                "**Group A (08:00 UTC):** Crypto + Commodities + Major Macro\n\n"
                "**Group B (14:00 UTC):** Big Tech + Key Figures\n\n"
                "**Group C (21:00 UTC):** Defense + Institutions + Central Banks"
            )

        # Show auto-fetch notification
        auto_msg = st.session_state.pop("news_auto_fetch_msg", None)
        if auto_msg:
            st.success(auto_msg)

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

        # ── BUTTONS ──
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            manual_refresh = st.button("🔄 Manual Refresh (uses API quota)")
        with btn_col2:
            rerun_forecast = st.button("♻️ Re-run forecast (no API calls)")
        with btn_col3:
            force_fetch = st.button("⚡ Force fetch all groups (uses 3x quota)")

        if manual_refresh:
            # Fetch only the current rotation group (saves quota)
            slot = _get_nearest_slot_group()
            keywords = NEWS_KEYWORD_GROUPS.get(slot, NEWS_KEYWORD_GROUPS[0])
            news_items = aggregate_news(keywords)

            if news_items:
                st.session_state["news_items"] = news_items
                state = _load_fetch_state()
                now_utc = dt.datetime.utcnow()
                state["last_fetch_utc"] = now_utc.isoformat()
                state["last_group"] = slot
                state["fetch_count_today"] = state.get("fetch_count_today", 0) + 1
                state["last_date"] = now_utc.strftime("%Y-%m-%d")
                _save_fetch_state(state)
                st.success(f"Fetched group {slot} ({len(keywords)} keywords). {len(news_items)} articles.")

        if force_fetch:
            all_kw = []
            for g in NEWS_KEYWORD_GROUPS.values():
                all_kw.extend(g)
            news_items = aggregate_news(all_kw)
            if news_items:
                st.session_state["news_items"] = news_items
                st.success(f"Force-fetched all groups. {len(news_items)} articles total.")

        if rerun_forecast or manual_refresh or force_fetch:
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

        # Auto-run forecast if asset changed
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

        if news_forecast:
            st.markdown("---")
            st.markdown(news_forecast)
        else:
            st.info(
                "No forecast yet. News will auto-fetch at the next scheduled window, "
                "or press 'Manual Refresh' to fetch now."
            )

        st.markdown("---")
        st.markdown("### Latest raw headlines")

        if not news_items:
            st.warning("No news in history. Press 'Manual Refresh' or wait for the next scheduled fetch.")
        else:
            st.caption(f"Showing {min(20, len(news_items))} of {len(news_items)} articles in history.")
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
    st.subheader("🧮 Quant Lab — Renaissance-Grade Quantitative Analysis")
    st.markdown(
        "Institutional systematic analysis inspired by Renaissance Technologies / Two Sigma. "
        "Multi-factor scoring: regime detection, mean-reversion, momentum, tail risk, and Monte Carlo."
    )

    df_global_for_q = st.session_state.get("df_signals_global", pd.DataFrame())
    df_crypto_for_q = st.session_state.get("df_signals_binance", pd.DataFrame())

    asset_options_q: List[str] = ["(choose)"]

    if not df_global_for_q.empty:
        asset_options_q.extend(
            df_global_for_q["name"].astype(str) + " (" + df_global_for_q["ticker"].astype(str) + ")"
        )
    if not df_crypto_for_q.empty:
        asset_options_q.extend(
            df_crypto_for_q["name"].astype(str) + " (" + df_crypto_for_q["symbol"].astype(str) + ")"
        )

    asset_options_q = sorted(set(asset_options_q))

    colA, colB, colC = st.columns(3)
    with colA:
        focus_asset_q = st.selectbox("Asset:", options=asset_options_q, index=0, key="quant_asset")
        source_pref_q = st.selectbox("Source:", options=["Auto", "Yahoo", "Binance"], index=0, key="quant_source")

    with colB:
        lookback_days_q = st.slider("Lookback (days):", 90, 730, 365, 30, key="quant_lookback")
        jump_z_q = st.slider("Jump threshold (z):", 2.0, 5.0, 3.0, 0.25, key="quant_jumpz")

    with colC:
        tf_binance_q = st.selectbox("Binance timeframe:", options=["1d", "4h", "1h", "15m"], index=0, key="quant_tf")
        horizon_label_q = st.selectbox("Horizon:", options=["1 day", "1 week", "1 month", "3 months"], index=1, key="quant_horizon")

    run_quant_btn = st.button("🧮 Run Quant Analysis", type="primary", key="run_quant_btn")

    if run_quant_btn:
        sym = parse_selected_asset_to_symbol(focus_asset_q)
        if not sym:
            st.warning("Select an asset.")
        else:
            source = detect_source_for_symbol(sym, preferred=source_pref_q)
            mpd = bars_per_day_for_tf(source, tf_binance_q)
            horizon_map = {"1 day": 1, "1 week": 7, "1 month": 30, "3 months": 90}
            horizon_bars = horizon_map.get(horizon_label_q, 7) * mpd
            tf_used = "1d" if source == "Yahoo" else tf_binance_q
            bpy = bars_per_year_for_timeframe(source, tf_used)

            try:
                close_series = fetch_close_series_for_quant(sym, source, tf_used, lookback_days_q)
                returns = np.log(close_series).diff().dropna()

                # Core quant metrics
                qm = compute_quant_metrics(close_series, bars_per_year=bpy, jump_z=float(jump_z_q), horizon_bars=int(horizon_bars))

                if "error" in qm:
                    st.error(qm["error"])
                else:
                    # Advanced metrics
                    regime = compute_regime_hmm_simple(returns, window=min(63, len(returns) // 3))
                    mean_rev = compute_mean_reversion_signals(close_series)
                    mom_feat = compute_momentum_features(close_series)
                    tail_risk = compute_tail_risk_metrics(returns)
                    acf = compute_autocorrelation(returns, max_lag=5)

                    # ── REGIME & SIGNAL OVERVIEW ──
                    st.markdown("---")
                    st.subheader("📊 Regime & Signal Overview")
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Vol Regime", regime.get("current_regime", "?"))
                    rc2.metric("Vol Percentile", f"{regime.get('regime_percentile', 0):.0f}%")
                    rc3.metric("Regime Duration", f"{regime.get('regime_duration_bars', 0)} bars")
                    rc4.metric("Ann. Volatility", f"{regime.get('current_vol_annualized', 0):.1%}")

                    # Hurst interpretation
                    hurst = qm.get("hurst")
                    if hurst is not None:
                        if hurst > 0.55:
                            hurst_label = "🟢 Trending (H > 0.55)"
                        elif hurst < 0.45:
                            hurst_label = "🔵 Mean-Reverting (H < 0.45)"
                        else:
                            hurst_label = "⚪ Random Walk (H ≈ 0.5)"
                    else:
                        hurst_label = "N/A"

                    rc5, rc6, rc7, rc8 = st.columns(4)
                    rc5.metric("Hurst Exponent", f"{hurst:.3f}" if hurst else "N/A", help=hurst_label)
                    rc6.metric("Z-Score (20d)", f"{mean_rev.get('z_score_20', 'N/A')}")
                    rc7.metric("Z-Score (50d)", f"{mean_rev.get('z_score_50', 'N/A')}")
                    half_life = mean_rev.get("ou_half_life")
                    rc8.metric("O-U Half-Life", f"{half_life:.0f} bars" if half_life else "N/A (trending)")

                    # ── MOMENTUM DASHBOARD ──
                    st.markdown("---")
                    st.subheader("🚀 Multi-Timeframe Momentum")
                    if mom_feat:
                        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                        mc1.metric("5-Day", f"{mom_feat.get('return_5d_pct', 'N/A')}%")
                        mc2.metric("1-Month", f"{mom_feat.get('return_21d_pct', 'N/A')}%")
                        mc3.metric("3-Month", f"{mom_feat.get('return_63d_pct', 'N/A')}%")
                        mc4.metric("6-Month", f"{mom_feat.get('return_126d_pct', 'N/A')}%")
                        mc5.metric("12-Month", f"{mom_feat.get('return_252d_pct', 'N/A')}%")

                        mom_score = mom_feat.get("momentum_composite_score")
                        if mom_score is not None:
                            if mom_score > 5:
                                mom_label = "🟢 Strong Bullish Momentum"
                            elif mom_score > 1:
                                mom_label = "🟢 Bullish Momentum"
                            elif mom_score > -1:
                                mom_label = "⚪ Neutral / Flat"
                            elif mom_score > -5:
                                mom_label = "🔴 Bearish Momentum"
                            else:
                                mom_label = "🔴 Strong Bearish Momentum"
                            st.markdown(f"**Composite Momentum Score:** `{mom_score:.2f}` → {mom_label}")
                    else:
                        st.info("Not enough data for momentum analysis (need 252+ bars).")

                    # ── TAIL RISK & PERFORMANCE ──
                    st.markdown("---")
                    st.subheader("🛡️ Tail Risk & Performance")
                    if tail_risk:
                        tr1, tr2, tr3, tr4 = st.columns(4)
                        tr1.metric("Sortino Ratio", tail_risk.get("sortino_ratio", "N/A"))
                        tr2.metric("Calmar Ratio", tail_risk.get("calmar_ratio", "N/A"))
                        tr3.metric("Tail Ratio", tail_risk.get("tail_ratio", "N/A"), help="P95 gain / P5 loss — higher is better")
                        tr4.metric("Gain/Pain", tail_risk.get("gain_pain_ratio", "N/A"))

                        tr5, tr6, tr7, tr8 = st.columns(4)
                        tr5.metric("Win Rate", f"{tail_risk.get('win_rate_pct', 'N/A')}%")
                        tr6.metric("Max Drawdown", f"{tail_risk.get('max_drawdown_pct', 'N/A')}%")
                        tr7.metric("Ann. Return", f"{tail_risk.get('annualized_return_pct', 'N/A')}%")
                        var95 = qm.get("VaR_95_logret")
                        tr8.metric("VaR 95%", f"{var95:.4f}" if var95 else "N/A", help="Worst daily loss at 95% confidence")

                    # ── JUMP DIFFUSION ──
                    st.markdown("---")
                    st.subheader("⚡ Jump-Diffusion Analysis (Merton Model)")
                    jc1, jc2, jc3, jc4 = st.columns(4)
                    jc1.metric("Jumps/Year (λ)", qm.get("lambda_year", "N/A"))
                    jc2.metric("Avg Jump Size", f"{qm.get('avg_jump_pct', 'N/A')}%")
                    jc3.metric("Jump Volatility", f"{qm.get('jump_vol_pct', 'N/A')}%")
                    jc4.metric("Jump Risk Score", qm.get("jump_risk_score", "N/A"))

                    # ── AUTOCORRELATION ──
                    st.markdown("---")
                    st.subheader("🔄 Serial Correlation (Autocorrelation)")
                    if acf:
                        acf_data = pd.DataFrame([acf])
                        acf_data.columns = [f"Lag {c.split('_')[1]}" for c in acf_data.columns]
                        st.dataframe(acf_data, use_container_width=True)
                        lag1 = acf.get("lag_1", 0)
                        if lag1 > 0.05:
                            st.markdown("📈 **Positive autocorrelation** — momentum/trend continuation likely. Trend-following strategies favored.")
                        elif lag1 < -0.05:
                            st.markdown("🔄 **Negative autocorrelation** — mean-reversion signal. Contrarian/reversion strategies favored.")
                        else:
                            st.markdown("⚪ **Near-zero autocorrelation** — no strong serial pattern. Market is efficient at this timeframe.")

                    # ── MONTE CARLO ──
                    st.markdown("---")
                    st.subheader(f"🎲 Monte Carlo Scenarios ({horizon_label_q} ahead)")
                    mc_p10 = qm.get("mc_p10")
                    mc_p50 = qm.get("mc_p50")
                    mc_p90 = qm.get("mc_p90")
                    mc_mean = qm.get("mc_mean")
                    last_p = qm.get("last_price", 0)

                    if mc_p10 is not None and last_p:
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("🔴 Bear (P10)", f"${mc_p10:,.2f}", f"{((mc_p10/last_p)-1)*100:+.1f}%")
                        sc2.metric("⚪ Base (P50)", f"${mc_p50:,.2f}", f"{((mc_p50/last_p)-1)*100:+.1f}%")
                        sc3.metric("🟢 Bull (P90)", f"${mc_p90:,.2f}", f"{((mc_p90/last_p)-1)*100:+.1f}%")
                        sc4.metric("📊 Expected", f"${mc_mean:,.2f}", f"{((mc_mean/last_p)-1)*100:+.1f}%")

                    # ── FULL METRICS TABLE ──
                    st.markdown("---")
                    with st.expander("📋 Full Quant Metrics Table"):
                        all_metrics = {**qm, **regime, **mean_rev, **mom_feat, **tail_risk, **acf}
                        st.dataframe(pd.DataFrame([all_metrics]), use_container_width=True)

                    # ── GPT ANALYSIS ──
                    st.markdown("---")
                    st.subheader("🧠 AI Quant Analysis (Renaissance-Grade)")

                    # Build enhanced brief with all metrics
                    all_metrics_for_brief = {**qm, **regime, **mean_rev, **mom_feat, **tail_risk}
                    enhanced_brief_lines = [
                        f"SYMBOL: {sym}",
                        f"SOURCE: {source}",
                        f"TIMEFRAME: {tf_used}",
                        f"LOOKBACK_DAYS: {lookback_days_q}",
                        f"HORIZON: {horizon_label_q}",
                        "",
                        "CORE METRICS:",
                    ]
                    for k, v in all_metrics_for_brief.items():
                        enhanced_brief_lines.append(f"- {k}: {v}")
                    enhanced_brief_lines.append("")
                    enhanced_brief_lines.append("AUTOCORRELATION:")
                    for k, v in acf.items():
                        enhanced_brief_lines.append(f"- {k}: {v}")

                    enhanced_brief = "\n".join(enhanced_brief_lines)
                    gpt_text = run_quant_gpt_analysis(enhanced_brief)
                    st.markdown(gpt_text)

            except Exception as e:
                st.error(f"Quant Lab error: {type(e).__name__}: {e}")

# -------- AI AGENT TAB (Claude with tool access to all data) --------
with tab_ai:
    st.subheader("🤖 Analizator — Conversational Agent")
    st.caption("Powered by Claude. Has tool access to news (NewsAPI + 25 RSS feeds), market quotes (Yahoo + Binance), technical signals, FRED macro data, economic calendar, SEC filings (JPM/GS/BlackRock/Berkshire/Bridgewater/Renaissance/Citadel/Two Sigma/Point72/Tiger Global), Finnhub analyst recs, and the global crypto market overview. Ask anything.")

    if not _HAS_AGENT:
        st.error(f"Agent module failed to load: {_AGENT_IMPORT_ERROR}")
    else:
        # Status row
        col_s1, col_s2, col_s3 = st.columns(3)
        try:
            health = dl.db_health()
            sched_st = bg_scheduler.scheduler_status()
            col_s1.metric("News stored", health.get("news_count", 0))
            col_s2.metric("Market snapshots", health.get("market_count", 0))
            col_s3.metric("SEC filings", health.get("sec_filings", 0))
            with st.expander("Background scheduler"):
                st.json(sched_st)
                st.json(health)
                col_a, col_b, col_c, col_d = st.columns(4)
                if col_a.button("🔄 Refresh news now"):
                    bg_scheduler.job_refresh_news()
                    bg_scheduler.job_refresh_rss()
                    st.success("Done")
                if col_b.button("💹 Refresh quotes now"):
                    bg_scheduler.job_refresh_quotes()
                    st.success("Done")
                if col_c.button("🪙 Refresh crypto now"):
                    bg_scheduler.job_refresh_crypto_market()
                    st.success("Done")
                if col_d.button("🏛 Refresh SEC now"):
                    bg_scheduler.job_refresh_sec()
                    st.success("Done")
        except Exception as e:
            st.warning(f"Status unavailable: {e}")

        # Chat session
        if "agent_session_id" not in st.session_state:
            st.session_state["agent_session_id"] = uuid.uuid4().hex
        if "agent_messages" not in st.session_state:
            st.session_state["agent_messages"] = []

        col_clear, _ = st.columns([1, 5])
        if col_clear.button("🧹 Clear chat"):
            st.session_state["agent_messages"] = []
            st.session_state["agent_session_id"] = uuid.uuid4().hex
            st.rerun()

        # Render history
        for m in st.session_state["agent_messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Pull text blocks for display
                txt_parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        txt_parts.append(c.get("text", ""))
                display_text = "\n".join(txt_parts).strip()
            else:
                display_text = str(content)
            if not display_text:
                continue
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(display_text)

        # Input
        user_input = st.chat_input("Ask the agent anything about markets, macro, news, or institutions...")
        if user_input:
            st.session_state["agent_messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = ai_agent.run_agent_turn(st.session_state["agent_messages"])
                        st.session_state["agent_messages"] = result["messages"]
                        st.markdown(result["text"])
                        if result.get("tool_calls"):
                            with st.expander(f"🔧 Tools used ({len(result['tool_calls'])})"):
                                for tc in result["tool_calls"]:
                                    st.code(f"{tc['name']}({tc['args']})\n→ {tc['result_preview']}")
                    except Exception as e:
                        st.error(f"Agent error: {e}")

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













































