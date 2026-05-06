#!/bin/bash
set -euo pipefail

python lib/qwen_llm.py -P "$(cat "$1")" -S assets.txt | tail -n +2  > outputs/registry.json
cat outputs/registry.json $1 > outputs/combined.txt
python lib/qwen_llm.py -P "$(cat "outputs/combined.txt")" -S sequence.txt | tail -n +2 > outputs/sequence.json
python Planning/build.py outputs/registry.json outputs/sequence.json > "$2"

