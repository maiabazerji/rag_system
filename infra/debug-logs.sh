#!/bin/bash

echo "=== Backend Logs (last 50 lines) ==="
docker compose -f docker-compose.yml logs --tail=50 backend

echo ""
echo "=== Qdrant Logs (last 20 lines) ==="
docker compose -f docker-compose.yml logs --tail=20 qdrant

echo ""
echo "=== Postgres Logs (last 20 lines) ==="
docker compose -f docker-compose.yml logs --tail=20 postgres

echo ""
echo "To see live logs:"
echo "  docker compose -f infra/docker-compose.yml logs -f backend"
