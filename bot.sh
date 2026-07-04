#!/bin/bash

INPUT_FILE="$1"

if [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

while true; do
    python bin/bot.py "$INPUT_FILE" -F --max-steps 3
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Completed successfully."
        exit 0
    fi
    
    if [ $EXIT_CODE -eq 1 ]; then
        echo "OOM detected. Restarting in 5 seconds..."
        sleep 5
        continue
    fi
    
    echo "Failed with error code $EXIT_CODE. Not retrying."
    exit $EXIT_CODE
done