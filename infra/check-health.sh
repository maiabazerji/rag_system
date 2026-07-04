#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Checking EvalRAG services..."
echo ""

# Check backend health
echo -n "Backend... "
if curl -s http://localhost:8011/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ DOWN${NC}"
    echo "  Try: docker logs evalrag-backend-1"
fi

# Check Qdrant
echo -n "Qdrant (Vector DB)... "
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ DOWN${NC}"
fi

# Check Postgres
echo -n "Postgres (Database)... "
if nc -z localhost 5434 2>/dev/null; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ DOWN${NC}"
fi

# Check Redis
echo -n "Redis (Cache)... "
if redis-cli -p 6381 ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ DOWN${NC}"
fi

# Check Frontend
echo -n "Frontend... "
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ DOWN${NC}"
fi

# Check Langfuse
echo -n "Langfuse (Tracing)... "
if curl -s http://localhost:3100 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${YELLOW}~ OPTIONAL${NC}"
fi

echo ""
echo "URLs:"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8011/docs"
echo "  Langfuse: http://localhost:3100"
