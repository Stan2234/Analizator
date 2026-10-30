"""Plans, the sign-in pass from AI Bubble Tracker, and AI credits.

Subscribers pay on AI Bubble Tracker (Stripe). Its server checks the
subscription and sends them here with a short signed pass:

    base64url("cus_…|plus|1760000000") + "." + hmac_sha256(secret, that)[:32]

The secret (ANALIZATOR_SSO_SECRET) is shared by the two apps. The owner signs in
with a verified account email in OWNER_EMAILS and gets everything without a credit
limit (a local run without Auth0 falls back to APP_PASSWORD).

Credits pay for AI calls at roughly $0.10 of API cost each (measured on Opus
5.5): a panel read 1, the news forecast 1, an analyst question 3, an FOMC read
3, a Deep Brief 10. They reset on the 1st of each month (UTC) and don't roll over.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

import data_layer as dl

SSO_SECRET = os.environ.get("ANALIZATOR_SSO_SECRET", "").strip()
OWNER_EMAILS = {e.strip().lower() for e in
                os.environ.get("OWNER_EMAILS", "02.25.stan@gmail.com").split(",") if e.strip()}
AI_BUBBLE_URL = os.environ.get("AI_BUBBLE_URL", "https://ai-bubble-tracker.onrender.com").rstrip("/")

TIERS = ["basic", "plus", "pro"]
TIER_NAMES = {"free": "Free", "basic": "Basic", "plus": "Plus", "pro": "Pro", "owner": "Owner"}
RANK = {"free": 0, "basic": 1, "plus": 2, "pro": 3}
CREDITS = {"basic": 10, "plus": 60, "pro": 180}
COST = {"read": 1, "news": 1, "analyst": 3, "fomc": 3, "brief": 10}
ACTION_NAMES = {"read": "an AI read", "news": "the AI news forecast", "analyst": "an analyst question",
                "fomc": "an FOMC read", "brief": "a Deep Brief"}

PLANS = {
    "basic": {"month": 9, "year": 90, "points": [
        "Adds Stocks, Currencies, Central Banks, Metals", "10 AI credits a month", "All of AI Bubble Tracker"]},
    "plus": {"month": 24, "year": 240, "points": [
        "Everything in Basic", "Adds Crypto, Fundamentals, News & Macro, Quant Lab, AI Analyst", "60 AI credits a month"]},
    "pro": {"month": 49, "year": 490, "points": [
        "Everything in Plus", "Adds Deep Brief, Carry Trade, FOMC Lab", "180 AI credits a month"]},
}

# The lowest plan that opens each tab.
TAB_TIER = {
    "global": "free", "stocks": "basic", "fx": "basic", "cb": "basic", "metals": "basic",
    "crypto": "plus", "fund": "plus", "news": "plus", "quant": "plus", "ai": "plus",
    "carry": "pro", "fomc": "pro", "brief": "pro",
}


def _sign(payload: str) -> str:
    return hmac.new(SSO_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]


def verify_pass(token: str) -> Optional[Dict[str, Any]]:
    """The subscriber behind a pass, or None if it is forged, malformed or expired.

    With ANALIZATOR_SSO_SECRET set the signature is checked here; without it (the
    normal production setup) AI Bubble Tracker, which signed the pass, is asked.
    """
    if not token or "." not in token:
        return None
    if not SSO_SECRET:
        return _verify_remote(token)
    payload, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(_sign(payload), sig):
        return None
    try:
        customer, tier, exp = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode().split("|")
        exp = int(exp)
    except Exception:
        return None
    if tier not in TIERS or not customer.startswith("cus_") or exp < time.time():
        return None
    return {"customer": customer, "tier": tier, "expires": exp}


def _verify_remote(token: str) -> Optional[Dict[str, Any]]:
    url = f"{AI_BUBBLE_URL}/api/analizator/verify?" + urllib.parse.urlencode({"t": token})
    try:   # AI Bubble's free Render plan can be waking up, hence the long timeout
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "analizator"}), timeout=60) as r:
            d = json.loads(r.read())
    except Exception:
        return None
    if d.get("plan") not in TIERS or not str(d.get("customer", "")).startswith("cus_"):
        return None
    if int(d.get("expires", 0)) < time.time():
        return None
    return {"customer": d["customer"], "tier": d["plan"], "expires": int(d["expires"])}


def owner() -> Dict[str, Any]:
    return {"customer": None, "tier": "owner"}


def is_owner_email(email: Optional[str], verified: bool) -> bool:
    """The owner's account: every tab and unlimited credits. Only a verified email counts,
    or anyone could sign up with the address and skip the inbox check."""
    return bool(verified and email and email.strip().lower() in OWNER_EMAILS)


def free(email: Optional[str] = None) -> Dict[str, Any]:
    """A visitor without a plan: anonymous, or signed in without a subscription."""
    return {"customer": None, "tier": "free", "email": email}


def paid(user: Optional[Dict[str, Any]]) -> bool:
    return bool(user) and user["tier"] in TIERS


def tab_open(user: Optional[Dict[str, Any]], tab: str) -> bool:
    tier = (user or {}).get("tier", "free")
    if tier == "owner":
        return True
    return RANK.get(tier, 0) >= RANK[TAB_TIER[tab]]


def period(now: Optional[dt.datetime] = None) -> str:
    return (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m")


def next_reset(now: Optional[dt.datetime] = None) -> dt.date:
    d = (now or dt.datetime.now(dt.timezone.utc)).date().replace(day=1)
    return (d + dt.timedelta(days=32)).replace(day=1)


def credits_left(user: Optional[Dict[str, Any]]) -> Optional[int]:
    """None means unlimited (the owner)."""
    if user and user["tier"] == "owner":
        return None
    if not paid(user):
        return 0
    return max(0, CREDITS[user["tier"]] - dl.credits_used(user["customer"], period()))


def brief_cost(n_specialists: int) -> int:
    """Deep Brief scales with the specialists picked: all six cost 10, one costs 3."""
    return int(-(-3 * n_specialists // 2)) + 1


def charge(user: Optional[Dict[str, Any]], action: str, cost: Optional[int] = None) -> Optional[str]:
    """Spend the credits for one AI action. Returns None if it may run, else why not."""
    if user and user["tier"] == "owner":
        return None
    if not paid(user):
        return "AI features are part of the paid plans. Log in and choose a plan to use them."
    cost = cost or COST[action]
    if dl.spend_credits(user["customer"], period(), cost, CREDITS[user["tier"]]):
        return None
    left = credits_left(user)
    name = ACTION_NAMES[action]
    return (f"{name[0].upper() + name[1:]} costs {cost} credit{'s' if cost > 1 else ''} and you have "
            f"{left} left this month. Credits reset on {next_reset():%d %B}; a higher plan has more.")


def refund(user: Optional[Dict[str, Any]], action: str, cost: Optional[int] = None) -> None:
    """Give the credits back when the AI call failed before producing anything."""
    if user and user["tier"] != "owner":
        dl.refund_credits(user["customer"], period(), cost or COST[action])


def upgrade_url() -> str:
    """Plan changes go through Stripe's portal (a second checkout would mean two subscriptions)."""
    return AI_BUBBLE_URL + "/api/portal/start?next=analizator"


def checkout_url(plan: str, interval: str) -> str:
    return f"{AI_BUBBLE_URL}/api/checkout/start?plan={plan}&interval={interval}&next=analizator"


def signin_url() -> str:
    """AI Bubble Tracker recognises the subscriber by its cookie and sends a fresh pass back."""
    return AI_BUBBLE_URL + "/api/analizator"
