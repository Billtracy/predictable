#!/usr/bin/env bash
# Deploy Predictable from this machine.
#   ./deploy/deploy.sh            sync code + .env + predictable.db, rebuild, restart
#   PREDICTABLE_DOMAIN=x ./deploy/deploy.sh --gateway
#                                 once DNS points at the server: certificate + gateway route
# Defaults target the shared server whose gateway is the WhatsApp analyzer's nginx
# (also hosting Crest and Readi). For a server made by deploy/bootstrap-server.sh, set
#   GATEWAY_CONTAINER=gateway GATEWAY_CONF=/home/opc/gateway/conf.d PROXY_NETWORK=proxy
set -euo pipefail

HOST="${PREDICTABLE_HOST:-opc@84.8.151.225}"
KEY="${PREDICTABLE_SSH_KEY:-$HOME/ssh-key-2026-09-23.key}"
GATEWAY_CONTAINER="${GATEWAY_CONTAINER:-whatsapp_webserver}"
GATEWAY_CONF="${GATEWAY_CONF:-/root/whatsapp-analyzer/docker/nginx/conf.d}"
export PROXY_NETWORK="${PROXY_NETWORK:-whatsapp-analyzer_whatsapp_net}"
cd "$(dirname "$0")/.."

remote() { ssh -i "$KEY" "$HOST" "$@"; }

if [[ "${1:-}" == "--gateway" ]]; then
    DOMAIN="${PREDICTABLE_DOMAIN:?Set PREDICTABLE_DOMAIN, e.g. predictable.mooo.com}"
    echo "→ Checking DNS for $DOMAIN"
    ip=$(remote "getent hosts $DOMAIN | cut -d' ' -f1")
    server_ip=$(remote "curl -s https://ifconfig.me")
    [[ "$ip" == "$server_ip" ]] || { echo "✗ $DOMAIN points at '${ip:-nothing}', not $server_ip"; exit 1; }
    names="$DOMAIN"
    www_ip=$(remote "getent hosts www.$DOMAIN | cut -d' ' -f1")
    [[ "$www_ip" == "$server_ip" ]] && names="$DOMAIN www.$DOMAIN"
    echo "→ Serving: $names"

    sed -e "s/__NAMES__/$names/g" -e "s/__DOMAIN__/$DOMAIN/g" deploy/nginx-predictable.conf | remote "cat > ~/predictable/predictable.conf"
    remote DOMAIN="$DOMAIN" NAMES="'$names'" GATEWAY_CONF="$GATEWAY_CONF" GW="$GATEWAY_CONTAINER" 'bash -s' <<'EOF'
set -euo pipefail
reload() { sudo docker exec $GW nginx -t && sudo docker exec $GW nginx -s reload; }
sudo mkdir -p /etc/letsencrypt/webroot
if ! sudo test -f /etc/letsencrypt/live/$DOMAIN/fullchain.pem; then
    echo "→ Port 80 only, to answer the certificate challenge"
    awk '/^server/{n++} n==1' ~/predictable/predictable.conf | sudo tee $GATEWAY_CONF/predictable.conf >/dev/null
    reload || { sudo rm -f $GATEWAY_CONF/predictable.conf; reload; exit 1; }
fi
# Issues on first run, adds www later (--expand), otherwise a no-op until renewal is due.
domains=(); for n in $NAMES; do domains+=(-d "$n"); done
sudo certbot certonly --webroot -w /etc/letsencrypt/webroot --cert-name $DOMAIN "${domains[@]}" --expand --keep-until-expiring --non-interactive --agree-tos --register-unsafely-without-email --key-type ecdsa
echo "→ Full gateway config"
sudo cp ~/predictable/predictable.conf $GATEWAY_CONF/predictable.conf
reload || { echo "✗ nginx -t failed; removing predictable.conf"; sudo rm -f $GATEWAY_CONF/predictable.conf; reload; exit 1; }

echo "→ Renewal timer"
sudo tee /etc/systemd/system/predictable-cert-renew.service >/dev/null <<UNIT
[Unit]
Description=Renew $DOMAIN certificate and reload the nginx gateway

[Service]
Type=oneshot
ExecStart=/usr/bin/certbot renew --cert-name $DOMAIN --quiet --deploy-hook "docker exec $GW nginx -s reload"
UNIT
sudo tee /etc/systemd/system/predictable-cert-renew.timer >/dev/null <<UNIT
[Unit]
Description=Twice-daily check to renew $DOMAIN certificate

[Timer]
OnCalendar=*-*-* 04,16:41:00
RandomizedDelaySec=1h
Persistent=true

[Install]
WantedBy=timers.target
UNIT
sudo systemctl daemon-reload && sudo systemctl enable --now predictable-cert-renew.timer
EOF
    echo "✓ https://$DOMAIN is routed to Predictable"
    exit 0
fi

echo "→ Syncing code"
rsync -az --delete -e "ssh -i $KEY" --exclude-from=.dockerignore --exclude predictable.conf ./ "$HOST:predictable/"
rsync -az -e "ssh -i $KEY" .env "$HOST:predictable/.env"

echo "→ Building and restarting"
remote "cd ~/predictable && sudo PROXY_NETWORK=$PROXY_NETWORK docker compose up -d --build --wait && sudo docker image prune -f >/dev/null && sudo docker builder prune -f >/dev/null"

echo "✓ Deployed"
