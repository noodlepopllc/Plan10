#!/bin/bash

uv run config -R
source .env

IN="$1"

for f in $IN/beat*.txt; do
    [[ $f == *prompt* ]] && continue
    if [[ $LTX != "False" ]]; then
        OUT="${f/.txt/_ltx.mp4}"
        if [[ ! -f $OUT ]]; then 
            uv run previewer -B "$f"
            ARG="${f/.txt/.ltx}"
            uv run image_to_video \
                -P "$(tail -n +2 "$ARG")" \
                -D "$(head -n 1 "$ARG" | cut -d':' -f2 | xargs)" \
                -O "$OUT"
            fi
        fi
        if [[ $MMH3 != "False" ]]; then
            OUT="${f/.txt/.mp4}"
            if [[ ! -f $OUT ]]; then
                uv run director -I "$f" \
                    -O "" -W $WIDTH -H $HEIGHT -S 6 --wangp --low-vram
        fi
    fi
done


