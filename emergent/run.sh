#!/bin/bash

echo "🎬 Starting creative loop. Press Ctrl+C to stop."

while true; do
    # Run creative step
    python cli.py "$@"
    CLI_EXIT=$?
    
    if [ $CLI_EXIT -lt 0 ]; then
        echo "❌ Creative step failed with error code $CLI_EXIT"
        break
    fi
    
    # Run video renderer
    python video_runner.py -O feedback_output
    VIDEO_EXIT=$?
    
    if [ $VIDEO_EXIT -lt 0 ]; then
        echo "❌ Video rendering failed with error code $VIDEO_EXIT"
        break
    fi
    
    echo "🔁 Beat $CLI_EXIT complete. Continuing..."
    sleep 1
done