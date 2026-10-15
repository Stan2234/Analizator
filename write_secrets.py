"""Write .streamlit/secrets.toml with the [auth] section Streamlit's st.login() needs,
from environment variables. Runs before `streamlit run` (render.yaml) and again
from the app itself, since a Render service not created from the blueprint
ignores render.yaml's start command.

  AUTH0_CLIENT_SECRET             secret, set in the Render dashboard
  AUTH0_DOMAIN, AUTH0_CLIENT_ID   public; default to this app's Auth0 tenant
  PUBLIC_URL                      this app's address, for the OAuth redirect
                                  (falls back to Render's RENDER_EXTERNAL_URL)

Without AUTH0_CLIENT_SECRET nothing is written and log in stays off.
The cookie secret is derived from the client secret so no extra secret is needed.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, ".streamlit", "secrets.toml")


def write() -> bool:
    env = lambda k, d="": (os.environ.get(k) or d).strip()
    domain = env("AUTH0_DOMAIN", "dev-g5tgi4qivhjrt7f2.eu.auth0.com").removeprefix("https://").rstrip("/")
    client_id = env("AUTH0_CLIENT_ID", "fH0L8UQPkUNxeQemUnL39MjH8wJU4v59")
    secret = env("AUTH0_CLIENT_SECRET")
    base = env("PUBLIC_URL", env("RENDER_EXTERNAL_URL", "http://localhost:8502")).rstrip("/")
    if not secret:
        return False
    q = json.dumps   # TOML basic strings share JSON's escaping for these values
    text = ("[auth]\n"
            f"redirect_uri = {q(base + '/oauth2callback')}\n"
            f"cookie_secret = {q(hashlib.sha256(('anz-cookie::' + secret).encode()).hexdigest())}\n\n"
            "[auth.auth0]\n"
            f"client_id = {q(client_id)}\n"
            f"client_secret = {q(secret)}\n"
            f"server_metadata_url = {q('https://' + domain + '/.well-known/openid-configuration')}\n")
    try:
        with open(PATH, encoding="utf-8") as f:
            if f.read() == text:
                return True
    except OSError:
        pass
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return True


if __name__ == "__main__":
    if write():
        print("auth configured")
    else:
        print("auth not configured (AUTH0_CLIENT_SECRET missing)")
