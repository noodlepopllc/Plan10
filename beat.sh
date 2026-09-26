#!/bin/bash

uv run config -R
source .env

SEED_FILE="$1"                 # adventure.txt
OUTDIR="${SEED_FILE%.*}"       # adventure

mkdir -p "$OUTDIR"

# Generate prompt.txt if missing
if [[ ! -f "$OUTDIR/prompt.txt" ]]; then
    uv run llm -P "Use the following to generate a detailed text to image for qwen image a prompt that accurately shows the initial situation. Only output the prompt, explanation is not required or desired.  $(cat "$SEED_FILE")" > "$OUTDIR/prompt.txt"
fi

uv run video_creator -P "$(tail -n +2 "$OUTDIR/prompt.txt")" -G "$(cat "$SEED_FILE")" -O "$OUTDIR" -D 12 
uv run video_runner -O "$OUTDIR"
EXIT_CODE=$?

# Fatal error case
if [[ $EXIT_CODE -eq 255 ]]; then
    echo "Fatal error in video_runner"
    exit 1
fi

BEAT=$(printf "%03d" "$EXIT_CODE")

VISION_BACKEND="qwen" 
if [[ $LTX != "False" ]]; then
    uv run image_to_video \
        -I "tmp.png" \
        -P "$(tail -n +2 "$OUTDIR/beat_${BEAT}_script_ltx.txt")" \
        -D "$(head -n 1 "$OUTDIR/beat_${BEAT}_script_ltx.txt" | cut -d':' -f2 | xargs)" \
        -O "$OUTDIR/beat_${BEAT}.mp4"
fi
if [[ $MMH3 != "False" ]]; then
    uv run director -I "$OUTDIR/beat_${BEAT}_script.txt" \
        -O "" -W $WIDTH -H $HEIGHT -S 8 --wangp
fi

if [[ -f "$OUTDIR/beat_${BEAT}_script.mp4" ]]; then
    mv "$OUTDIR/beat_${BEAT}_script.mp4" "$OUTDIR/beat_${BEAT}.mp4"
fi
