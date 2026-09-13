#!/bin/bash

uv run config -R
source .env

SEED_FILE="$1"                 # adventure.txt
BEAT="$2"                      # 001
OUTDIR="${SEED_FILE%.*}"       # adventure

mkdir -p "$OUTDIR"

PREVIEWS="$OUTDIR/PREVIEWS"


WIDTH=${#BEAT}
LAST_BEAT=$(printf "%0${WIDTH}d" $((10#$BEAT - 1)))
FINAL="$OUTDIR/beat_${LAST_BEAT}_script.mp4"
STANDARD="$OUTDIR/beat_${LAST_BEAT}.mp4"

# If both exist, archive preview and promote final
if [[ -f "$FINAL" && -f "$STANDARD" ]]; then
    mkdir -p $PREVIEWS 
    mv "$STANDARD" "$PREVIEWS/"
    mv "$FINAL" "$STANDARD"
fi

# If beat already exists, exit
if [[ -f "$OUTDIR/beat_${BEAT}.mp4" ]]; then
    exit
fi

# Generate prompt.txt if missing
if [[ ! -f "$OUTDIR/prompt.txt" ]]; then 
    backup=$THINKING
    THINKING="False"
    uv run llm -P "Use the following to generate a detailed text to image for qwen image a prompt that accurately shows the initial situation. Only output the prompt, explanation is not required or desired.  $(cat "$SEED_FILE")" > "$OUTDIR/prompt.txt"
    THINKING=$backup
fi

if [[ 1 == 1 && ! -f "$OUTDIR/actions.txt" ]]; then
    backup=$THINKING
    THINKING="False"
    uv run llm -P "$(cat <<'EOF'
Examine the text to image prompt below. Generate only the most likely action of what is about to happen. Can include dialog if appropriate.

CRITICAL: Refer to characters as char1 (first character mentioned), char2 (second character), etc. DO NOT use descriptions or names.

Return the actions only without any explanation. No more than 3 total short actions. One action is allowed to be short 5-7 words of dialog.

IMAGE PROMPT:
$(tail -n +2 "$OUTDIR/prompt.txt")
EOF
)" > "$OUTDIR/actions.txt"
    THINKING=$backup
fi

if [[ ! -f  "$OUTDIR/beat_${beat}_script.txt" ]]; then
    uv run video_creator -P "$(tail -n +2 $OUTDIR/prompt.txt)" -G "$(cat $SEED_FILE)"  -O $OUTDIR -D 8
    uv run video_runner -O $OUTDIR --debug
fi

# Preview
uv run video_previewer \
    -B "$OUTDIR/beat_${BEAT}_script.txt" \
    -S "Sci-Fi" \
    -U

# Final render
uv run image_to_video \
    -P "$(tail -n +2 "$OUTDIR/beat_${BEAT}_script_ltx.txt")" \
    -D "$(head -n 1 "$OUTDIR/beat_${BEAT}_script_ltx.txt" | cut -d':' -f2 | xargs)" \
    -O "$OUTDIR/beat_${BEAT}.mp4"

