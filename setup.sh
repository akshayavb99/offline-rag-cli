#!/usr/bin/env bash
set -e

echo "Setting up RAG Assistant environment..."

# -------------------------
# Check Python
# -------------------------
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python 3 not found. Install Python 3.10+"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# -------------------------
# Check uv package manager
# -------------------------
if ! command -v uv &> /dev/null; then
    echo "❌ uv package manager not found. Please install uv before running setup."
    echo "Install with: pip install uv"
    exit 1
else
    echo "uv found: using uv for venv and dependency management"
fi

# -------------------------
# Create uv-managed venv if missing
# -------------------------
if [ ! -d ".venv" ]; then
    echo "Creating uv-managed virtual environment..."
    uv venv 
else
    echo "Virtual environment already exists, skipping creation"
fi

# -------------------------
# Activate venv (cross-platform)
# -------------------------
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source ".venv/Scripts/activate"
else
    source ".venv/bin/activate"
fi

# -------------------------
# Sync dependencies with uv
# -------------------------
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies with uv"
    uv pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping dependency sync"
fi

# -------------------------
# Check Docker
# -------------------------
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Install Docker Desktop"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "Docker is installed but not running. Start Docker Desktop."
    exit 1
fi

# -------------------------
# Load .env
# -------------------------
if [ ! -f ".env" ]; then
    echo ".env not found. Define .env based on .env.example"
    exit 1
fi
export $(grep -v '^#' .env | xargs)

# -------------------------
# Pull Ollama Docker image
# -------------------------
echo "Pulling Ollama Docker image..."
docker pull ollama/ollama:latest

echo "Setup complete!"
echo "Run the app with: python -m main"
