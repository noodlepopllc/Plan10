#!/bin/bash
set -euo pipefail

config -R
source .env

mkdir -p $2/output
output="$2/output"

basepath="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "$basepath"

TMP_THINKING="$THINKING"
THINKING="False"

if [[ ! -d "$2/story" ]]; then
    mkdir -p $2/story
    story --seed $1 --output $2/story
fi

if [[ ! -f "$output/script.txt" || ! -f "$output/world.txt" ]]; then
    python $basepath/builders/script.py $1 $output --story $2/story/scene$3.story
fi

render_identity $output/registry.json $2/scene.txt


LLM_BACKUP=$LLM_BACKEND
LLM_BACKEND="transformers"
bot "$2/scene.txt" -F --max-steps 3
LLM_BACKEND=$LLM_BACKUP

THINKING=$TMP_THINKING

create_metadata --context $2/scene/context.json --registry $output/registry.json --script $output/script.txt --output $output/final_script.txt --format header
create_shots $output/final_script.txt $2

echo "✅ Pipeline complete: scene $2"

