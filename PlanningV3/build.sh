#!/bin/bash
set -euo pipefail

mkdir -p $3/output
output="$3/output"

source .env

if [[ -z $SEED ]]; then
    export SEED=$RANDOM
fi

touch $output/empty.txt

if [[ ! -f "$output/screenplay.txt" ]]; then
    python lib/qwen_llm.py -P "$(cat "$1")" -S PlanningV3/prompts/screenwriter.txt | tail -n +2 > $output/screenplay.txt
fi

if [[ ! -f "$output/registry.json" ]]; then
    # 1. Asset Registry (unchanged)
    echo "creating $output/registry.json"
    python lib/qwen_llm.py -P "$(cat "$output/screenplay.txt")" -S PlanningV3/prompts/assets.txt | tail -n +2 > $output/registry.json
fi

if [[ ! -f "$output/beats.json" ]]; then
    echo "creating $output/beats.json"
    python PlanningV3/builders/beats.py $output/screenplay.txt $output/beats.json
fi

if [[ ! -f "$output/scene$2.txt" ]]; then
    echo "creating $output/identity$2.txt"
    python PlanningV3/expanders/identity.py $output/registry.json $2 > "$3/scene$2.txt"
    python PlanningV3/expanders/scene.py $output/registry.json $output/beats.json $2 >> "$3/scene$2.txt"
fi

echo "✅ Pipeline complete: $3"
