#!/bin/bash
# Server setup script for strategy-with-bt
# Run on a fresh Ubuntu 22.04+ VPS (Oracle Cloud, Hetzner, DigitalOcean, etc.)
#
# Usage: ssh into your server, then:
#   curl -sSL https://raw.githubusercontent.com/jaho5/strategy-with-bt/main/setup-server.sh | bash
#   -- or --
#   scp setup-server.sh user@server:~ && ssh user@server 'bash setup-server.sh'

set -euo pipefail

echo "=========================================="
echo "  strategy-with-bt server setup"
echo "=========================================="

# 1. System packages
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl unzip ufw

# 2. Firewall — SSH only
echo "[2/6] Configuring firewall..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
echo "y" | sudo ufw enable

# 3. Install uv
echo "[3/6] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 4. Clone repo
echo "[4/6] Cloning repository..."
cd ~
if [ -d "strategy-with-bt" ]; then
    echo "  Repo already exists, pulling latest..."
    cd strategy-with-bt && git pull
else
    git clone https://github.com/jaho5/strategy-with-bt.git
    cd strategy-with-bt
fi

# 5. Install Python dependencies
echo "[5/6] Installing dependencies..."
uv sync

# 6. Set up cron
echo "[6/6] Setting up cron job..."
CRON_CMD="0 14 * * 1-5 cd $HOME/strategy-with-bt && $HOME/.local/bin/uv run python -m src.run_daily >> reports/daily_run.log 2>&1"
(crontab -l 2>/dev/null | grep -v "src.run_daily"; echo "$CRON_CMD") | crontab -

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Remaining manual steps:"
echo ""
echo "  1. Copy credentials from your local machine:"
echo "     scp ~/.env schwab_token.json user@$(hostname -I | awk '{print $1}'):~/strategy-with-bt/"
echo ""
echo "  2. Lock down the files:"
echo "     chmod 600 ~/strategy-with-bt/.env ~/strategy-with-bt/schwab_token.json"
echo ""
echo "  3. Test with dry run:"
echo "     cd ~/strategy-with-bt && uv run python -m src.run_daily --dry-run"
echo ""
echo "  4. Cron is set to run at 9 AM ET (14:00 UTC) weekdays."
echo "     Verify with: crontab -l"
echo ""
