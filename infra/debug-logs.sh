#!/bin/bash
# Works from any directory: paths are resolved relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE=(docker compose --env-file "$ROOT_DIR/.env" -f "$SCRIPT_DIR/docker-compose.yml")

echo "=== Backend Logs (last 50 lines) ==="
"${COMPOSE[@]}" logs --tail=50 backend

echo ""
echo "=== Qdrant Logs (last 20 lines) ==="
"${COMPOSE[@]}" logs --tail=20 qdrant

echo ""
echo "=== Postgres Logs (last 20 lines) ==="
"${COMPOSE[@]}" logs --tail=20 postgres

echo ""
echo "To see live logs:"
echo "  docker compose --env-file .env -f infra/docker-compose.yml logs -f backend"
