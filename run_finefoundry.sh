#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${REPO_DIR}/src/main.py" ]]; then
  echo "Could not find src/main.py under: ${REPO_DIR}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  echo "Install uv: https://docs.astral.sh/uv/" >&2
  exit 1
fi

cd "${REPO_DIR}"
exec uv run src/main.py "$@"
