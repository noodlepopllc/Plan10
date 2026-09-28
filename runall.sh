#!/bin/bash

uv run config -R
source .env

IN="$1"

for f in $IN/beat*.txt; do
if [[ $LTX != "False" ]]; then
    uv run previewer -B "$f"
    ARG="${f/.txt/_ltx.txt}"
    OUT="${f/.txt/_ltx.mp4}"
    uv run image_to_video \
        -P "$(tail -n +2 "$ARG")" \
        -D "$(head -n 1 "$ARG" | cut -d':' -f2 | xargs)" \
        -O "$OUT"
fi
if [[ $MMH3 != "False" ]]; then
    uv run director -I "$f" \
        -O "" -W $WIDTH -H $HEIGHT -S 6 --wangp --low-vram
fi
done


