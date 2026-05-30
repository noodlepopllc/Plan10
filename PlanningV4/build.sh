#!/bin/bash
set -euo pipefail

mkdir -p $2/output
output="$2/output"

if [[ ! -f "$output/complete.json" ]]; then
    python PlanningV4/builders/scriptwriter.py $1 $output
fi

if [[ ! -f "$output/registry.json" ]]; then
    # 1. Asset Registry (unchanged)
    echo "creating $output/registry.json"
    python lib/qwen_llm.py -P "$(cat "$output/biography.txt")" -S PlanningV4/prompts/assets.txt | tail -n +2 > $output/registry.json
fi

python PlanningV4/expanders/rendererV3.py $2 | tail -n +2 > $2/scene.txt


echo "✅ Pipeline complete: scene $2"
