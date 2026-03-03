#!/bin/bash
cd "$(dirname "$0")"

# Install dependencies
pip install -q torch fastapi uvicorn 2>/dev/null

# Train model if weights don't exist
if [ ! -f weights/rope_transformer.pt ]; then
    echo "Training model..."
    python model.py
fi

# Start server
echo "Starting RoPE Transformer Explorer on port 8000..."
exec uvicorn server:app --host 0.0.0.0 --port 8000
