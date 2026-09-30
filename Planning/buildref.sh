#!/bin/bash
set -euo pipefail

uv run config -R
source .env

mkdir -p $2/output
output="$2/output"

basepath="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "$basepath"

if [[ ! -f "$output/story.txt" ]]; then
    python $basepath/builders/storywriter.py -S $1 -O $output/story.txt
fi

if [[ ! -f "$output/script.txt" || ! -f "$output/world.txt" ]]; then
    python $basepath/builders/script.py $1 $output
fi

if [[ ! -f "$output/complete.json" ]]; then
    python $basepath/builders/scriptwriter.py $1 $output
fi

python $basepath/renderer/renderer.py $2 minimum > $2/scene.txt

./bot.sh $2/scene.txt

python $basepath/renderer/scene_converterV2.py $2

echo "✅ Pipeline complete: scene $2"

