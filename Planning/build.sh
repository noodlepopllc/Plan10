#!/bin/bash
set -euo pipefail

mkdir -p Planning/outputs

python lib/qwen_llm.py -P "$(cat "$1")" -S Planning/assets.txt | tail -n +2  > Planning/outputs/registry.json
python Planning/build_sequence_prompt.py Planning/outputs/registry.json $1 > Planning/outputs/combined.txt
python lib/qwen_llm.py -P "$(cat "Planning/outputs/combined.txt")" -S Planning/sequence.txt | tail -n +2 > Planning/outputs/sequence.json
python Planning/build.py Planning/outputs/registry.json Planning/outputs/sequence.json > "$2"

