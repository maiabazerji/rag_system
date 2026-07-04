#!/bin/bash

echo "Stopping containers..."
docker compose -f docker-compose.yml down

echo "Rebuilding images..."
docker compose -f docker-compose.yml build --no-cache

echo "Starting services..."
docker compose -f docker-compose.yml up -d

echo ""
echo "Waiting for services to start (30s)..."
sleep 30

echo ""
echo "Checking health..."
bash check-health.sh

echo ""
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8011/docs"
