#!/bin/bash
INPUT_FILE=$1
OUTPUT_DIR=$2
DURATION=${3:-10}        # Default to 10 if not provided
FAST_FLAG=$4             # Optional: pass "--fast" as 4th arg

mkdir -p $OUTPUT_DIR

uv run llm -P "Use the following to generate a detailed text to image for qwen image a prompt that accurately shows the initial situation. Only output the prompt, explanation is not required or desired.  $(cat $INPUT_FILE)" > $OUTPUT_DIR/prompt.txt
uv run llm -P "Examine the prompt and generate only the most likely action of what is about to happen can include dialog if appropriate include all characters and use character description instead of name to identify the character. Return the actions only without any explanation.  $(tail -n +2 $OUTPUT_DIR/prompt.txt)" > $OUTPUT_DIR/actions.txt

echo "🎬 Starting creative loop. Press Ctrl+C to stop."
echo "⏱️  Duration: ${DURATION}s | Fast mode: ${FAST_FLAG:-off}"

export BRIEF="True"
BEAT=1

uv run config -R

source .env

while true; do
    # Run creative step (ignores --fast safely)
    uv run video_creator -P "$(tail -n +2 $OUTPUT_DIR/prompt.txt)" -C "$(tail -n +2 $OUTPUT_DIR/actions.txt)" -G "$(cat $INPUT_FILE)" -O $OUTPUT_DIR -D $DURATION
    CLI_EXIT=$?
    
    if [ $CLI_EXIT -eq 255 ]; then
        echo "❌ Creative step failed with error code $CLI_EXIT"
        break
    fi
    
    # Run video renderer - pass --fast if provided
    uv run video_runner -O $OUTPUT_DIR $FAST_FLAG
    VIDEO_EXIT=$?
    
    if [ $VIDEO_EXIT -eq 255 ]; then
        echo "❌ Video rendering failed with error code $VIDEO_EXIT"
        break
    fi
    
    echo "🔁 Beat $BEAT complete. Continuing..."
    BEAT=$((BEAT + 1))
    sleep 1
done