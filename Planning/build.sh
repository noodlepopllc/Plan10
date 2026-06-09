#!/bin/bash
set -euo pipefail

mkdir -p $2/output
output="$2/output"

basepath="PlanningV2"

if [[ ! -f "$output/story.txt" ]]; then
    python $basepath/builders/storywriter.py -S $1 -O $output/story.txt
fi

if [[ ! -f "$output/complete.json" ]]; then
    python $basepath/builders/scriptwriter.py $1 $output
fi

if [[ ! -f "$output/registry.json" ]]; then
    # 1. Asset Registry (unchanged)
    echo "creating $output/registry.json"
    python lib/qwen_llm.py -P "$(cat "$output/biography.txt")" -S $basepath/prompts/assets.txt | tail -n +2 > $output/registry.json
fi

python $basepath/renderer/renderer.py $2 $3 > $2/scene.txt


echo "✅ Pipeline complete: scene $2"
