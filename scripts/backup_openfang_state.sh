#!/usr/bin/env bash
set -euo pipefail

SRC_HOME="${1:-/var/projects/agilize-content/extra/openfang}"
BACKUP_ROOT="${2:-${SRC_HOME}/backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
DEST="${BACKUP_ROOT}/openfang-${STAMP}"

mkdir -p "${DEST}"

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -e "${src}" ]; then
    mkdir -p "$(dirname "${dest}")"
    cp -a "${src}" "${dest}"
  fi
}

copy_if_exists "${SRC_HOME}/config.toml" "${DEST}/config.toml"
copy_if_exists "${SRC_HOME}/secrets.env" "${DEST}/secrets.env"
copy_if_exists "${SRC_HOME}/daemon.json" "${DEST}/daemon.json"
copy_if_exists "${SRC_HOME}/cron_jobs.json" "${DEST}/cron_jobs.json"
copy_if_exists "${SRC_HOME}/agents" "${DEST}/agents"
copy_if_exists "${SRC_HOME}/workspaces" "${DEST}/workspaces"
copy_if_exists "${SRC_HOME}/workflows" "${DEST}/workflows"
copy_if_exists "${SRC_HOME}/audit" "${DEST}/audit"
copy_if_exists "${SRC_HOME}/memory" "${DEST}/memory"
copy_if_exists "${SRC_HOME}/db" "${DEST}/db"

tar -C "${BACKUP_ROOT}" -czf "${DEST}.tar.gz" "$(basename "${DEST}")"

cat <<EOF
Backup created:
  directory: ${DEST}
  archive:   ${DEST}.tar.gz
EOF
