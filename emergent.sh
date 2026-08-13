#!/bin/bash

echo "🎬 Starting creative loop. Press Ctrl+C to stop."

export BRIEF="True"
BEAT=1

uv run config -R

source .env

while true; do
    # Run creative step
    uv run video_creator "$@"
    CLI_EXIT=$?
    
    if [ $CLI_EXIT -eq 255 ]; then
        echo "❌ Creative step failed with error code $CLI_EXIT"
        break
    fi
    
    # Run video renderer (pass same args so it uses the right -O directory)
   uv run video_runner "$@"
    VIDEO_EXIT=$?
    
    if [ $VIDEO_EXIT -eq 255 ]; then
        echo "❌ Video rendering failed with error code $VIDEO_EXIT"
        break
    fi
    
    echo "🔁 Beat $BEAT complete. Continuing..."
    BEAT=$((BEAT + 1))
    sleep 1

done