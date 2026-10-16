#!/usr/bin/env bash
# Deploy the committed HEAD to the VPS: /opt/analizator, systemd unit analizator,
# Streamlit on 127.0.0.1:8502 under /market (Caddy serves globalmarketanalyst.com/market).
# Secrets live in /opt/analizator/.env on the server (chmod 600), never in the repo.
# Usage: ./deploy_vps.sh            (from the Analizator folder, after committing)
set -euo pipefail
HOST=${HOST:-root@77.42.46.129}
KEY=${KEY:-$HOME/.ssh/ict_vps_ed25519}
APP=/opt/analizator
DATA=/opt/analizator-data
SSH="ssh -i $KEY -o BatchMode=yes $HOST"

cd "$(dirname "$0")"
TMP=$(mktemp -d)
git archive --format=tar.gz -o "$TMP/app.tgz" HEAD
scp -q -i "$KEY" "$TMP/app.tgz" "$HOST:/tmp/analizator.tgz"
rm -rf "$TMP"

$SSH bash -s <<EOF
set -euo pipefail
mkdir -p $APP $DATA
tar -xzf /tmp/analizator.tgz -C $APP
rm -f /tmp/analizator.tgz
cd $APP
# Python 3.11 like Render, through uv (the system Python is newer than some wheels)
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null; }
[ -x venv/bin/python ] || uv venv -q -p 3.11 venv
VIRTUAL_ENV=$APP/venv uv pip install -q -r requirements.txt
cat > /etc/systemd/system/analizator.service <<UNIT
[Unit]
Description=Market Analyst (Analizator, Streamlit)
After=network-online.target

[Service]
WorkingDirectory=$APP
EnvironmentFile=-$APP/.env
Environment=ANALIZATOR_DB_PATH=$DATA/analizator.db
Environment=PUBLIC_URL=https://globalmarketanalyst.com/market
Environment=AI_BUBBLE_URL=https://globalmarketanalyst.com/bubble
Environment=SCHEDULER_EXTERNAL=1
ExecStartPre=$APP/venv/bin/python write_secrets.py
ExecStart=$APP/venv/bin/streamlit run macro_agent_app.py --server.port=8502 --server.address=127.0.0.1 --server.headless=true --server.baseUrlPath=market
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
# the data jobs run on their own, so they keep going when nobody has the app open
cat > /etc/systemd/system/analizator-jobs.service <<UNIT
[Unit]
Description=Market Analyst background data jobs
After=network-online.target

[Service]
WorkingDirectory=$APP
EnvironmentFile=-$APP/.env
Environment=ANALIZATOR_DB_PATH=$DATA/analizator.db
ExecStart=$APP/venv/bin/python scheduler.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable -q analizator analizator-jobs
systemctl restart analizator-jobs analizator
for i in \$(seq 1 30); do curl -sf -o /dev/null http://127.0.0.1:8502/market/_stcore/health && break; sleep 2; done
systemctl is-active analizator
curl -s -o /dev/null -w "health %{http_code}\n" http://127.0.0.1:8502/market/_stcore/health
EOF
echo "Live: https://globalmarketanalyst.com/market"
