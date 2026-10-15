"""Stripe for Macro Analyzer: which plan an account's email has, checkout, the billing portal.

A subscription bought here carries the buyer's verified login email in its
metadata (account_email), so signing in with that email from any device finds
the plan again. Prices are the same Stripe prices AI Bubble Tracker sells.

Needs STRIPE_SECRET_KEY (a restricted key: Checkout Sessions write, Customer
portal write, Subscriptions read) and the six STRIPE_PRICE_* ids. Without them
nothing is sold and every signed-in account counts as Free.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

STRIPE_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
# The live price ids, used with a live key when the environment doesn't name them
# (a Render service not created from the blueprint skips render.yaml's envVars).
_LIVE = STRIPE_KEY.startswith(("sk_live_", "rk_live_"))
_env = lambda k, live: (os.environ.get(k) or (live if _LIVE else "")).strip()
PRICES = {   # plan -> interval -> price id (Basic kept its original variable names)
    "basic": {"month": _env("STRIPE_PRICE_MONTH", "price_1UPn4fFvzDXAEXwv6G4iPrE0"),
              "year": _env("STRIPE_PRICE_YEAR", "price_1UPn4dFvzDXAEXwv7ltRExsE")},
    "plus": {"month": _env("STRIPE_PRICE_PLUS_MONTH", "price_1UQMn1FvzDXAEXwv8O30RCW0"),
             "year": _env("STRIPE_PRICE_PLUS_YEAR", "price_1UQMn1FvzDXAEXwvaVxJRx5j")},
    "pro": {"month": _env("STRIPE_PRICE_PRO_MONTH", "price_1UQMn2FvzDXAEXwva0gtwbzX"),
            "year": _env("STRIPE_PRICE_PRO_YEAR", "price_1UQMn3FvzDXAEXwvUcoLj1vm")},
}
PRICE_PLAN = {pid: plan for plan, ps in PRICES.items() for pid in ps.values() if pid}
ENABLED = bool(STRIPE_KEY and all(all(ps.values()) for ps in PRICES.values()))
LIVE_STATUSES = ("active", "trialing", "past_due")
_cache: Dict[str, Any] = {}          # email -> (result, checked_at)
TTL_HIT, TTL_MISS = 600, 120


class StripeError(Exception):
    pass


def _stripe(method: str, path: str, **fields: Any) -> Dict[str, Any]:
    data = urllib.parse.urlencode(fields).encode() if method == "POST" else None
    url = "https://api.stripe.com/v1/" + path + ("?" + urllib.parse.urlencode(fields) if method == "GET" and fields else "")
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + STRIPE_KEY, "Stripe-Version": "2025-03-31.basil"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read())["error"]["message"]
        except Exception:
            msg = f"Stripe error {e.code}"
        raise StripeError(msg) from None


def plan_for_email(email: str, fresh: bool = False) -> Optional[Dict[str, Any]]:
    """{"plan", "customer", "subscription"} for the newest live subscription bought with
    this account email, or None."""
    if not (ENABLED and email):
        return None
    email = email.strip().lower()
    hit = _cache.get(email)
    if hit and not fresh and time.time() - hit[1] < (TTL_HIT if hit[0] else TTL_MISS):
        return hit[0]
    q = f"metadata['account_email']:'{email}'"
    try:
        subs = _stripe("GET", "subscriptions/search", query=q, limit=20)["data"]
    except Exception:
        return hit[0] if hit else None          # Stripe unreachable: keep the last answer
    live = [s for s in subs if s["status"] in LIVE_STATUSES]
    res = None
    if live:
        s = max(live, key=lambda x: x["created"])
        res = {"plan": PRICE_PLAN.get(s["items"]["data"][0]["price"]["id"], "basic"),
               "customer": s["customer"], "subscription": s["id"]}
    _cache[email] = (res, time.time())
    return res


def checkout_url(email: str, plan: str, interval: str, return_url: str) -> str:
    email = email.strip().lower()
    s = _stripe("POST", "checkout/sessions", **{
        "mode": "subscription",
        "line_items[0][price]": PRICES[plan][interval], "line_items[0][quantity]": 1,
        "managed_payments[enabled]": "true",
        "customer_email": email,
        "subscription_data[metadata][account_email]": email,
        "metadata[account_email]": email,
        "success_url": return_url + "?checkout=done",
        "cancel_url": return_url})
    return s["url"]


def portal_url(customer: str, return_url: str) -> str:
    return _stripe("POST", "billing_portal/sessions", customer=customer, return_url=return_url)["url"]


def forget(email: str) -> None:
    _cache.pop((email or "").strip().lower(), None)
