#!/bin/bash
# Probe every EvalRAG service on the ports Compose publishes (127.0.0.1 only).

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Checking EvalRAG services..."
echo ""

# -f makes curl fail on an HTTP error status, so a 500 is not reported as OK.
probe() {
    curl -fsS --max-time 5 "$1" > /dev/null 2>&1
}

# Check backend health
echo -n "Backend... "
if probe http://127.0.0.1:8011/health; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ DOWN${NC}"
    echo "  Try: docker logs evalrag-backend-1"
fi

# Check Qdrant. /healthz needs no API key, so this works with QDRANT_API_KEY set.
echo -n "Qdrant (Vector DB)... "
if probe http://127.0.0.1:6333/healthz; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ DOWN${NC}"
    echo "  Try: docker logs evalrag-qdrant-1"
fi

# Check Postgres
echo -n "Postgres (Database)... "
if docker exec evalrag-postgres-1 pg_isready -U evalrag -d evalrag > /dev/null 2>&1 \
    || nc -z 127.0.0.1 5434 2>/dev/null; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ DOWN${NC}"
    echo "  Try: docker logs evalrag-postgres-1"
fi

# Check Frontend
echo -n "Frontend... "
if probe http://127.0.0.1:5173; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ DOWN${NC}"
    echo "  Try: docker logs evalrag-frontend-1"
fi

# Check Langfuse (only runs with --profile tracing)
echo -n "Langfuse (Tracing)... "
if probe http://127.0.0.1:3100/api/public/health; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${YELLOW}~ OPTIONAL (not running; start with --profile tracing)${NC}"
fi

echo ""
echo "URLs:"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8011/docs"
echo "  Langfuse: http://localhost:3100"
