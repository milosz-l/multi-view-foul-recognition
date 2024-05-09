#!/bin/sh

# Use the USER environment variable to construct the path
VENV_DIR="/net/tscratch/people/$USER/.venv"
ACTIVATE_PATH="$VENV_DIR/bin/activate"

# Check if the virtual environment exists
if [ -f "$ACTIVATE_PATH" ]; then
    # Activate the virtual environment
    source "$ACTIVATE_PATH"
    echo "Virtual environment activated."
else
    echo "Virtual environment not found. Please make sure it exists in the expected location: $VENV_DIR"
fi

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

python3 download_data.py