#!/usr/bin/env bash
# One-time setup of a fresh Oracle Linux 9 server for hosting several projects:
# Docker, certbot, firewall, and the shared nginx gateway in ~/gateway.
# Then deploy with GATEWAY_CONTAINER=gateway GATEWAY_CONF=/home/opc/gateway/conf.d PROXY_NETWORK=proxy.
# Safe to re-run (it also updates the gateway's own files, never projects' confs).
#   PREDICTABLE_HOST=opc@<ip> PREDICTABLE_SSH_KEY=~/key ./deploy/bootstrap-server.sh
set -euo pipefail

HOST="${PREDICTABLE_HOST:?Set PREDICTABLE_HOST, e.g. opc@1.2.3.4}"
KEY="${PREDICTABLE_SSH_KEY:?Set PREDICTABLE_SSH_KEY to the instance private key}"
cd "$(dirname "$0")"

remote() { ssh -i "$KEY" "$HOST" "$@"; }

echo "→ Packages: Docker, Compose, certbot, rsync"
remote 'bash -s' <<'EOF'
set -euo pipefail
sudo dnf install -y -q dnf-plugins-core oracle-epel-release-el9 rsync
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y -q docker-ce docker-ce-cli containerd.io docker-compose-plugin certbot
sudo systemctl enable --now docker

echo "→ Firewall: http + https"
sudo firewall-cmd --permanent --add-service=http --add-service=https >/dev/null
sudo firewall-cmd --reload >/dev/null

# Certificates are requested and renewed per project (deploy.sh --gateway).
sudo mkdir -p /etc/letsencrypt/webroot
mkdir -p ~/gateway
EOF

echo "→ Gateway files"
rsync -az -e "ssh -i $KEY" gateway/ "$HOST:gateway/"

echo "→ Starting the gateway"
remote 'cd ~/gateway && sudo docker compose up -d --wait && sudo docker exec gateway nginx -t'

echo "✓ Server ready. Next: ./deploy/deploy.sh, then ./deploy/deploy.sh --gateway"
