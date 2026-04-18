#!/bin/bash
# Dumps the MongoDB database, compresses it to a single archive, and uploads to GCS,
# overwriting the previous backup file.
#
# Installed to /usr/local/bin/backup_mongodb.sh on the VM by setup_mongodb_backup_vm.sh.
# Reads credentials from /etc/backup.env (chmod 600, root only).

set -euo pipefail

ENV_FILE="/etc/backup.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

: "${MONGODB_URI:?MONGODB_URI is required}"
: "${MONGODB_DB_NAME:?MONGODB_DB_NAME is required}"
: "${GCS_BUCKET:?GCS_BUCKET is required}"

ARCHIVE_NAME="mongodb_backup.archive.gz"
TMP_PATH="/tmp/${ARCHIVE_NAME}"
GCS_DEST="gs://${GCS_BUCKET}/${ARCHIVE_NAME}"

trap 'rm -f "$TMP_PATH"' EXIT

echo "[$(date -u +%FT%TZ)] Starting MongoDB backup: db=${MONGODB_DB_NAME}"

mongodump \
  --uri="${MONGODB_URI}" \
  --db="${MONGODB_DB_NAME}" \
  --archive="${TMP_PATH}" \
  --gzip \
  --quiet

echo "[$(date -u +%FT%TZ)] Backup created: $(du -sh "$TMP_PATH" | cut -f1)"

gsutil -q cp "$TMP_PATH" "$GCS_DEST"

echo "[$(date -u +%FT%TZ)] Uploaded to ${GCS_DEST}. Done."
