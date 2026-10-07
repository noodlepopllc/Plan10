#!/bin/bash
set -euo pipefail

uv run config -R
source .env

mkdir -p $2/output
output="$2/output"

basepath="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "$basepath"

TMP_THINKING="$THINKING"
THINKING="False"

if [[ ! -f "$output/story.txt" ]]; then
    python $basepath/builders/storywriter.py -S $1 -O $output/story.txt
fi

if [[ ! -f "$output/script.txt" || ! -f "$output/world.txt" ]]; then
    python $basepath/builders/script.py $1 $output
fi

if [[ ! -f "$output/narrative.json" ]]; then
    python $basepath/builders/scriptwriter.py $1 $output
fi

python $basepath/renderer/renderer.py $2 minimum > $2/scene.txt

bot "$2/scene.txt" -F --max-steps 3

THINKING=$TMP_THINKING

uv run $basepath/renderer/generate_header.py --context $2/scene/context.json --registry $output/registry.json --script $output/script.txt --output $output/final_script.txt

python $basepath/renderer/scene_converter.py $output/final_script.txt $2

echo "✅ Pipeline complete: scene $2"

