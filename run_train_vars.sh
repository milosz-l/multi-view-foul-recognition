#!/bin/sh

VENV_DIR="/net/tscratch/people/plgmiloszl/.venv"
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

srun python3 train_vars.py