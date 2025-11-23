#!/bin/bash

# Flowlib Server Run Script
# Starts the FastAPI server with hot-reload for development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-false}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo -e "${GREEN}Starting Flowlib Server...${NC}"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Hot-reload: $RELOAD"
echo "Log level: $LOG_LEVEL"
echo ""

# Determine which Python to use
PYTHON_CMD="python"

# Check if we need to use the flowlib conda environment
if [[ "${CONDA_DEFAULT_ENV}" != "flowlib" ]] && [[ -z "${VIRTUAL_ENV}" ]]; then
    # Try to find the flowlib conda environment
    FLOWLIB_CONDA_PATH="${HOME}/miniconda3/envs/flowlib/bin/python"
    if [[ -f "$FLOWLIB_CONDA_PATH" ]]; then
        echo -e "${YELLOW}Flowlib conda environment not activated, using it directly${NC}"
        PYTHON_CMD="$FLOWLIB_CONDA_PATH"
    else
        echo -e "${YELLOW}Warning: Flowlib conda environment not found${NC}"
        echo "Consider activating the flowlib environment: conda activate flowlib"
        echo ""
    fi
elif [[ "${CONDA_DEFAULT_ENV}" == "flowlib" ]]; then
    echo -e "${GREEN}Using conda environment: flowlib${NC}"
    echo ""
elif [[ -n "${VIRTUAL_ENV}" ]]; then
    echo -e "${GREEN}Using virtual environment: ${VIRTUAL_ENV}${NC}"
    echo ""
fi

# Check if flowlib-server is installed
if ! "$PYTHON_CMD" -c "import server" 2>/dev/null; then
    echo -e "${RED}Error: server package not found${NC}"
    echo "Install it with: pip install -e ."
    exit 1
fi

# Check if flowlib is accessible
if ! "$PYTHON_CMD" -c "import flowlib" 2>/dev/null; then
    echo -e "${YELLOW}Warning: flowlib package not found in Python path${NC}"
    echo "Make sure flowlib is installed: cd ../../flowlib && pip install -e ."
    echo ""
fi

# Run the server using python -m to ensure correct Python version
if [ "$RELOAD" = "true" ]; then
    echo -e "${GREEN}Starting server with hot-reload enabled...${NC}"
    "$PYTHON_CMD" -m uvicorn server.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL"
else
    echo -e "${GREEN}Starting server in production mode...${NC}"
    "$PYTHON_CMD" -m uvicorn server.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL"
fi
