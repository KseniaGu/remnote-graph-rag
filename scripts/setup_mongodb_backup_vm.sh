#!/bin/bash
# Run this ONCE on the GCP VM to set up the daily MongoDB backup cron job.
#
# Prerequisites on the VM:
#   - mongodump installed (sudo apt-get install -y mongodb-database-tools)
#   - gsutil available (pre-installed on GCP VMs via Cloud SDK)
#   - VM service account has roles/storage.objectAdmin on the backup bucket
#
# Usage:
#   bash setup_mongodb_backup_vm.sh \
#     --mongodb-uri "mongodb://user:pass@localhost:27017/db?authSource=db" \
#     --db-name "langgraph_checkpoints" \
#     --bucket "remnote-graph-backup"

set -euo pipefail

MONGODB_URI=""
MONGODB_DB_NAME=""
GCS_BUCKET=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --mongodb-uri) MONGODB_URI="$2"; shift 2 ;;
    --db-name)     MONGODB_DB_NAME="$2"; shift 2 ;;
    --bucket)      GCS_BUCKET="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$MONGODB_URI" || -z "$MONGODB_DB_NAME" || -z "$GCS_BUCKET" ]]; then
  echo "Usage: $0 --mongodb-uri <uri> --db-name <name> --bucket <bucket>"
  exit 1
fi

BACKUP_SCRIPT="/usr/local/bin/backup_mongodb.sh"
ENV_FILE="/etc/backup.env"

# Write env file (readable only by root)
echo "Writing ${ENV_FILE}..."
sudo tee "$ENV_FILE" > /dev/null <<EOF
MONGODB_URI=${MONGODB_URI}
MONGODB_DB_NAME=${MONGODB_DB_NAME}
GCS_BUCKET=${GCS_BUCKET}
EOF
sudo chmod 600 "$ENV_FILE"

# Copy backup script to system path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing backup script to ${BACKUP_SCRIPT}..."
sudo cp "${SCRIPT_DIR}/backup_mongodb.sh" "$BACKUP_SCRIPT"
sudo chmod +x "$BACKUP_SCRIPT"

# Install daily cron job (runs at 03:00 UTC, one hour after LangSmith export)
CRON_LINE="0 3 * * * root ${BACKUP_SCRIPT} >> /var/log/mongodb_backup.log 2>&1"
CRON_FILE="/etc/cron.d/mongodb-backup"
echo "Installing cron job at ${CRON_FILE}..."
echo "$CRON_LINE" | sudo tee "$CRON_FILE" > /dev/null
sudo chmod 644 "$CRON_FILE"

echo ""
echo "Setup complete. The backup will run daily at 03:00 UTC."
echo "To run a test backup now:"
echo "  sudo bash ${BACKUP_SCRIPT}"
echo "To view logs:"
echo "  tail -f /var/log/mongodb_backup.log"
