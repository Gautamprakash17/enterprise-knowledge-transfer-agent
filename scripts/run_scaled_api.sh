#!/usr/bin/env bash
# Run API behind nginx with N replicas (default 2).
# Requires Docker Compose v2 with merge support (!reset in docker-compose.scale.yml).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REPLICAS="${1:-2}"
exec docker compose -f docker-compose.yml -f docker-compose.scale.yml up --build --scale "api=${REPLICAS}"
