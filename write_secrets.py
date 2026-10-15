"""Write .streamlit/secrets.toml with the [auth] section Streamlit's st.login() needs,
from environment variables, before `streamlit run` starts (see render.yaml).

  AUTH0_DOMAIN, AUTH0_CLIENT_ID   public, in render.yaml
  AUTH0_CLIENT_SECRET             secret, set in the Render dashboard
  PUBLIC_URL                      this app's address, for the OAuth redirect

Without AUTH0_CLIENT_SECRET nothing is written and log in stays off.
The cookie secret is derived from the client secret so no extra secret is needed.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
domain = os.environ.get("AUTH0_DOMAIN", "").strip().removeprefix("https://").rstrip("/")
client_id = os.environ.get("AUTH0_CLIENT_ID", "").strip()
secret = os.environ.get("AUTH0_CLIENT_SECRET", "").strip()
base = os.environ.get("PUBLIC_URL", "http://localhost:8502").strip().rstrip("/")

path = os.path.join(HERE, ".streamlit", "secrets.toml")
if domain and client_id and secret:
    q = json.dumps   # TOML basic strings share JSON's escaping for these values
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "[auth]\n"
            f"redirect_uri = {q(base + '/oauth2callback')}\n"
            f"cookie_secret = {q(hashlib.sha256(('anz-cookie::' + secret).encode()).hexdigest())}\n\n"
            "[auth.auth0]\n"
            f"client_id = {q(client_id)}\n"
            f"client_secret = {q(secret)}\n"
            f"server_metadata_url = {q('https://' + domain + '/.well-known/openid-configuration')}\n"
        )
    print("auth configured for", base)
else:
    print("auth not configured (AUTH0_DOMAIN / AUTH0_CLIENT_ID / AUTH0_CLIENT_SECRET missing)")
