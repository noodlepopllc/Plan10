#!/bin/bash
set -euo pipefail

mkdir -p $2/output
output="$2/output"

if [[ ! -f "$output/complete.json" ]]; then
    python PlanningV3/builders/scriptwriterV3.py $1 $output
fi

if [[ ! -f "$output/registry.json" ]]; then
    # 1. Asset Registry (unchanged)
    echo "creating $output/registry.json"
    python lib/qwen_llm.py -P "$(cat "$output/biography.txt")" -S PlanningV3/prompts/assets.txt | tail -n +2 > $output/registry.json
fi

if [[ ! -f "$output/shots.json" ]]; then
    python PlanningV3/builders/group_shotV2.py $output $output/shots.json
fi

if [[ ! -f "$output/assets$3.json" ]]; then
    python PlanningV3/expanders/assets.py $output $3
fi

if [[ ! -f "$2/scene$3.txt" ]]; then
    python PlanningV3/expanders/renderer.py $output $3 > $2/scene$3.txt
fi

echo "✅ Pipeline complete: scene $3"
