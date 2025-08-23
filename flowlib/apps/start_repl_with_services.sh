#!/bin/bash

# Script to start Docker services and run the Flowlib REPL
set -e

echo "🚀 Starting Flowlib REPL with Docker services..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ docker-compose is not available. Please install Docker Compose.${NC}"
    exit 1
fi

# Function to use docker compose or docker-compose
docker_compose_cmd() {
    if docker compose version &> /dev/null; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

echo -e "${BLUE}📦 Starting Docker services...${NC}"

# Check if docker-compose.yml exists in the project root
COMPOSE_FILE="$(dirname "$(dirname "$0")")/docker-compose.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${YELLOW}⚠️  docker-compose.yml not found at $COMPOSE_FILE${NC}"
    echo -e "${BLUE}💡 Running REPL without external services.${NC}"
    echo -e "${BLUE}💡 You can use flowlib with file-based providers (SQLite, local storage, etc.)${NC}"
    echo ""
    echo -e "${GREEN}🤖 Starting Flowlib REPL (standalone mode)...${NC}"
    echo "=================================================="
    echo ""
    # Skip service startup and go directly to REPL
    python "$(dirname "$0")/run_repl.py" "$@"
    exit $?
fi

# Start services in the background
docker_compose_cmd -f "$COMPOSE_FILE" up -d

echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"

# Function to wait for a service to be healthy
wait_for_service() {
    local service_name=$1
    local max_attempts=30
    local attempt=1
    
    echo -e "${BLUE}   Waiting for $service_name...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if docker_compose_cmd -f "$COMPOSE_FILE" ps --services --filter "status=running" | grep -q "^$service_name$"; then
            # Check if the service has a health check
            health_status=$(docker inspect --format='{{.State.Health.Status}}' "${service_name}_flowlib" 2>/dev/null || echo "no-healthcheck")
            health_status=$(echo "$health_status" | tr -d '\n\r')
            echo -e "${BLUE}   Debug: $service_name health status: '$health_status'${NC}"
            
            if [ "$health_status" = "healthy" ] || [ "$health_status" = "no-healthcheck" ]; then
                echo -e "${GREEN}   ✅ $service_name is ready${NC}"
                return 0
            fi
        else
            echo -e "${BLUE}   Debug: $service_name not found in running services${NC}"
        fi
        
        echo -e "${YELLOW}   ⏳ Attempt $attempt/$max_attempts - $service_name not ready yet...${NC}"
        sleep 2
        ((attempt++))
    done
    
    echo -e "${RED}   ❌ $service_name failed to become ready after $max_attempts attempts${NC}"
    return 1
}

# Wait for each service
echo -e "${BLUE}🔍 Checking service health...${NC}"

wait_for_service "neo4j" || {
    echo -e "${RED}❌ Neo4j failed to start properly${NC}"
    echo -e "${YELLOW}💡 You can check the logs with: docker-compose -f $COMPOSE_FILE logs neo4j${NC}"
    exit 1
}

wait_for_service "chroma" || {
    echo -e "${YELLOW}⚠️  Chroma didn't start properly, but continuing...${NC}"
}

wait_for_service "redis" || {
    echo -e "${YELLOW}⚠️  Redis didn't start properly, but continuing...${NC}"
}

wait_for_service "rabbitmq" || {
    echo -e "${YELLOW}⚠️  RabbitMQ didn't start properly, but continuing...${NC}"
}

wait_for_service "qdrant" || {
    echo -e "${YELLOW}⚠️  Qdrant didn't start properly, but continuing...${NC}"
}

wait_for_service "pinecone-local" || {
    echo -e "${YELLOW}⚠️  Pinecone Local didn't start properly, but continuing...${NC}"
}

echo ""
echo -e "${GREEN}✅ Services are ready!${NC}"
echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
echo -e "${BLUE}  • Neo4j Browser: ${NC}http://localhost:7474 (neo4j/pleaseChangeThisPassword)"
echo -e "${BLUE}  • Neo4j Bolt: ${NC}bolt://localhost:7687"
echo -e "${BLUE}  • Chroma API: ${NC}http://localhost:8000"
echo -e "${BLUE}  • Redis: ${NC}localhost:6379"
echo -e "${BLUE}  • RabbitMQ Management: ${NC}http://localhost:15672 (guest/guest)"
echo -e "${BLUE}  • Qdrant Dashboard: ${NC}http://localhost:6333/dashboard"
echo -e "${BLUE}  • Pinecone Local API: ${NC}http://localhost:5080"
echo ""

# Check Python dependencies
echo -e "${BLUE}🐍 Checking Python dependencies...${NC}"

# Check for required packages
missing_packages=()

if ! python -c "import neo4j" 2>/dev/null; then
    missing_packages+=("neo4j")
fi

if ! python -c "import chromadb" 2>/dev/null; then
    missing_packages+=("chromadb")
fi

if ! python -c "import llama_cpp" 2>/dev/null; then
    missing_packages+=("llama-cpp-python")
fi

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing Python packages: ${missing_packages[*]}${NC}"
    echo -e "${BLUE}💡 Install with: pip install ${missing_packages[*]}${NC}"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}⏹️  Stopping services...${NC}"
        docker_compose_cmd down
        exit 1
    fi
fi

echo -e "${GREEN}🤖 Starting Flowlib REPL...${NC}"
echo "=================================================="
echo ""

# Add a trap to stop services when the script exits
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down...${NC}"
    read -p "Stop Docker services? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo -e "${BLUE}⏹️  Stopping Docker services...${NC}"
        docker_compose_cmd -f "$COMPOSE_FILE" down
        echo -e "${GREEN}✅ Services stopped${NC}"
    else
        echo -e "${BLUE}📦 Services left running. Stop with: docker-compose down${NC}"
    fi
}

trap cleanup EXIT

# Run the REPL (from the apps directory)
python "$(dirname "$0")/run_repl.py" "$@"